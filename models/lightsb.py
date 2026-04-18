"""
Light Schrödinger Bridge (LightSB) Implementation

Implements an official-style LightSB solver based on the public ICLR 2024
Light Schrödinger Bridge code: sum-exp quadratic Schrödinger potentials with
a Gaussian-mixture-like parameterization and an analytic convolution objective.

Key Features:
    - Sum-exp quadratic Schrödinger potential
    - Gaussian-mixture-like endpoint sampler
    - Analytic log C(x0) - log v(x1) training objective

Reference:
    Korotin et al., "Light Schrödinger Bridge", ICLR 2024

Author: Manus AI
"""

import numpy as np
from typing import Dict, Any, Optional, List
from numba import njit, prange

from models.base import TimeSeriesGenerator

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class QuadraticPotentialBridge(nn.Module):
        """Official-style LightSB potential parameterization.

        This follows the public LightSB implementation pattern: a sum-exp
        quadratic/log-potential parameterized as a Gaussian-mixture-like family.
        The implementation uses diagonal quadratic matrices to avoid adding the
        official repo's optional orthogonal-matrix dependency.
        """

        def __init__(
            self,
            dim: int,
            n_potentials: int = 20,
            epsilon: float = 1.0,
            sampling_batch_size: int = 512,
            s_diagonal_init: float = 0.1,
        ):
            super().__init__()
            self.dim = int(dim)
            self.n_potentials = int(n_potentials)
            self.sampling_batch_size = int(sampling_batch_size)

            self.register_buffer("epsilon", torch.tensor(float(epsilon)))
            init_weights = torch.ones(self.n_potentials) / self.n_potentials
            self.log_alpha_raw = nn.Parameter(self.epsilon * torch.log(init_weights))
            self.r = nn.Parameter(torch.randn(self.n_potentials, self.dim))
            self.s_log_diagonal = nn.Parameter(
                torch.log(float(s_diagonal_init) * torch.ones(self.n_potentials, self.dim))
            )

        def init_r_by_samples(self, samples: torch.Tensor) -> None:
            """Initialize potential centers from target samples when available."""
            if len(samples) == 0:
                return
            if len(samples) >= self.n_potentials:
                idx = torch.randperm(len(samples), device=samples.device)[: self.n_potentials]
            else:
                idx = torch.randint(len(samples), (self.n_potentials,), device=samples.device)
            with torch.no_grad():
                self.r.copy_(samples[idx].to(self.r.device))

        def get_s(self) -> torch.Tensor:
            return torch.exp(self.s_log_diagonal)

        def get_log_alpha(self) -> torch.Tensor:
            return self.log_alpha_raw / self.epsilon

        def _component_logits(self, x: torch.Tensor) -> torch.Tensor:
            s = self.get_s()
            r = self.r
            eps = self.epsilon
            log_alpha = self.get_log_alpha()
            x_s_x = (x[:, None, :] * s[None, :, :] * x[:, None, :]).sum(dim=-1)
            x_r = (x[:, None, :] * r[None, :, :]).sum(dim=-1)
            return (x_s_x + 2.0 * x_r) / (2.0 * eps) + log_alpha[None, :]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Sample target endpoint from the learned conditional map."""
            s = self.get_s()
            eps_s = self.epsilon * s
            samples = []
            batch_size = x.shape[0]
            step = max(1, self.sampling_batch_size)

            for start in range(0, batch_size, step):
                sub_x = x[start : start + step]
                logits = self._component_logits(sub_x)
                loc = self.r[None, :, :] + s[None, :, :] * sub_x[:, None, :]
                scale = torch.sqrt(eps_s)[None, :, :]
                mix = torch.distributions.Categorical(logits=logits)
                comp = torch.distributions.Independent(
                    torch.distributions.Normal(loc=loc, scale=scale),
                    1,
                )
                samples.append(torch.distributions.MixtureSameFamily(mix, comp).sample())
            return torch.cat(samples, dim=0)

        def get_log_potential(self, x: torch.Tensor) -> torch.Tensor:
            """Evaluate the unnormalized log-Schrodinger potential at target samples."""
            log_alpha = self.get_log_alpha()
            mix = torch.distributions.Categorical(logits=log_alpha)
            comp = torch.distributions.Independent(
                torch.distributions.Normal(
                    loc=self.r,
                    scale=torch.sqrt(self.epsilon * self.get_s()),
                ),
                1,
            )
            gmm = torch.distributions.MixtureSameFamily(mix, comp)
            return gmm.log_prob(x) + torch.logsumexp(log_alpha, dim=-1)

        def get_log_c(self, x: torch.Tensor) -> torch.Tensor:
            """Evaluate the analytic convolution term at source samples."""
            return torch.logsumexp(self._component_logits(x), dim=-1)

        def get_drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            """Evaluate the bridge drift used by the official Euler sampler."""
            if t.dim() == 0:
                t = t.expand(x.shape[0])
            x_var = x.detach().clone().requires_grad_(True)
            eps = self.epsilon
            s = self.get_s()
            r = self.r
            log_alpha = self.get_log_alpha()

            one_minus_t = torch.clamp(1.0 - t, min=1e-4)
            a = t[:, None, None] / (eps * one_minus_t[:, None, None]) + 1.0 / (
                eps * s[None, :, :]
            )
            s_log_det = self.s_log_diagonal.sum(dim=-1)
            a_log_det = torch.log(a).sum(dim=-1)
            s_inv = 1.0 / s
            a_inv = 1.0 / a
            c = (
                (x_var / (eps * one_minus_t[:, None]))[:, None, :]
                + (r * s_inv / eps)[None, :, :]
            )
            exp_arg = (
                log_alpha[None, :]
                - 0.5 * s_log_det[None, :]
                - 0.5 * a_log_det
                - 0.5 * ((r * s_inv * r) / eps).sum(dim=-1)[None, :]
                + 0.5 * (c * a_inv * c).sum(dim=-1)
            )
            lse = torch.logsumexp(exp_arg, dim=-1)
            grad = torch.autograd.grad(lse.sum(), x_var, create_graph=False)[0]
            return (-x_var / one_minus_t[:, None] + eps * grad).detach()

        def sample_euler_maruyama(
            self,
            x: torch.Tensor,
            n_steps: int,
        ) -> torch.Tensor:
            """Sample a bridge trajectory with the official Euler-Maruyama recipe."""
            t = torch.zeros(x.shape[0], device=x.device)
            dt = 1.0 / int(n_steps)
            trajectory = [x]
            for _ in range(int(n_steps)):
                drift = self.get_drift(x, t)
                noise = torch.randn_like(x) * np.sqrt(dt) * torch.sqrt(self.epsilon)
                x = x + drift * dt + noise
                t = torch.clamp(t + dt, max=1.0 - 1e-4)
                trajectory.append(x)
            return torch.stack(trajectory, dim=1)


# ============================================
# LightSB Model
# ============================================

class LightSB(TimeSeriesGenerator):
    """
    Official-style Light Schrödinger Bridge for time-series windows.

    The model follows the public LightSB solver design: it learns a
    Gaussian-mixture-like sum-exp quadratic Schrödinger potential and uses the
    analytic LightSB objective log C(x0) - log v(x1). In this repository each
    time-series window is treated as one high-dimensional point.
    
    Usage:
        model = LightSB(config)
        model.fit(data, time_grid)
        generated = model.generate(n_samples)
    """
    
    MODEL_TYPE = "lightsb"
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LightSB model.
        
        Args:
            config: Configuration with LightSB hyperparameters.
        """
        super().__init__(config)
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for LightSB")
        
        self.n_potentials = config.get('lightsb_n_potentials', 20)
        self.epsilon = config.get('lightsb_epsilon', config.get('lightsb_ot_epsilon', 1.0))
        self.s_diagonal_init = config.get('lightsb_s_diagonal_init', 0.1)
        self.sampling_batch_size = config.get('lightsb_sampling_batch_size', 512)
        self.init_centers_from_data = config.get('lightsb_init_centers_from_data', True)
        self.epochs = config.get('lightsb_epochs', 100)
        self.lr = config.get('lightsb_lr', 0.001)
        self.batch_size = config.get('lightsb_batch_size', 256)
        self.weight_decay = config.get('lightsb_weight_decay', 0.0)
        self.grad_clip = config.get('lightsb_grad_clip', 1.0)
        self.source_std = config.get('lightsb_source_std', 1.0)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model components
        self.bridge = None
        self.n_features = None
        self.seq_len = None
        self.flat_dim = None
        self.data_mean = None
        self.data_std = None
        
        # Training history
        self.train_losses = []
    
    def fit(
        self,
        data: np.ndarray,
        time_grid: np.ndarray,
        verbose: bool = True
    ) -> 'LightSB':
        """
        Fit LightSB model.
        
        Args:
            data: Time series data (n_samples, seq_len, n_features)
            time_grid: Time points
            verbose: Whether to print progress
        
        Returns:
            self for method chaining
        """
        # Ensure 3D
        if data.ndim == 2:
            data = data[:, :, np.newaxis]
        
        n_samples, seq_len, n_features = data.shape
        self.n_features = n_features
        self.seq_len = seq_len
        self.flat_dim = seq_len * n_features
        
        # Normalize data
        self.data_mean = np.mean(data)
        self.data_std = np.std(data) + 1e-8
        data_norm = (data - self.data_mean) / self.data_std
        
        # Treat each full window as one high-dimensional target point.
        data_flat = data_norm.reshape(n_samples, self.flat_dim)
        x_target = torch.tensor(data_flat, dtype=torch.float32, device=self.device)

        self.bridge = QuadraticPotentialBridge(
            dim=self.flat_dim,
            n_potentials=self.n_potentials,
            epsilon=self.epsilon,
            sampling_batch_size=self.sampling_batch_size,
            s_diagonal_init=self.s_diagonal_init,
        ).to(self.device)

        if self.init_centers_from_data:
            self.bridge.init_r_by_samples(x_target)
        
        optimizer = torch.optim.Adam(
            self.bridge.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        
        if verbose:
            print("=" * 60)
            print("LightSB Training (official-style potential)")
            print("=" * 60)
        
        self.train_losses = []
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            # Shuffle data
            perm = torch.randperm(n_samples)
            x_target_epoch = x_target[perm]
            
            for i in range(0, n_samples, self.batch_size):
                batch_target = x_target_epoch[i:i+self.batch_size]
                batch_size = len(batch_target)
                
                batch_source = self.source_std * torch.randn_like(batch_target)
                optimizer.zero_grad()
                loss = (
                    self.bridge.get_log_c(batch_source).mean()
                    - self.bridge.get_log_potential(batch_target).mean()
                )
                loss.backward()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.bridge.parameters(), self.grad_clip)
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            self.train_losses.append(avg_loss)
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"  [LightSB] Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")
        
        self.bridge.eval()
        self.is_fitted = True
        
        if verbose:
            print("=" * 60)
            print("LightSB Training Complete!")
            print("=" * 60)
        
        return self
    
    def generate(
        self,
        n_samples: int,
        n_steps: Optional[int] = None,
        x0: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate synthetic time series.
        
        Samples target endpoints from the learned LightSB conditional map.
        
        Args:
            n_samples: Number of samples to generate
            n_steps: Ignored for output length; kept for API compatibility
            x0: Optional flattened source noise of shape (n_samples, seq_len*n_features)
        
        Returns:
            Generated time series (n_samples, seq_len, n_features)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before generation")
        
        del n_steps

        if x0 is None:
            source = self.source_std * torch.randn(n_samples, self.flat_dim, device=self.device)
        else:
            source_np = np.asarray(x0, dtype=np.float32)
            if source_np.ndim == 3:
                source_np = source_np.reshape(source_np.shape[0], -1)
            if source_np.ndim != 2 or source_np.shape != (n_samples, self.flat_dim):
                raise ValueError(
                    "LightSB x0 is source noise and must have shape "
                    f"({n_samples}, {self.flat_dim}) or "
                    f"({n_samples}, {self.seq_len}, {self.n_features})"
                )
            source = torch.tensor(source_np, dtype=torch.float32, device=self.device)
        
        self.bridge.eval()
        with torch.no_grad():
            target = self.bridge(source)
        
        generated = target.cpu().numpy().reshape(n_samples, self.seq_len, self.n_features)
        return generated * self.data_std + self.data_mean
    
    def get_training_history(self) -> List[float]:
        """Get training loss history."""
        return self.train_losses


# ============================================
# Numba-Accelerated Markovian SB
# ============================================

@njit(cache=True, parallel=True)
def _markovian_sb_forward_numba(
    x0: np.ndarray,
    x1: np.ndarray,
    time_grid: np.ndarray,
    sigma: float
) -> np.ndarray:
    """
    Simulate Markovian Schrödinger Bridge forward process.
    
    Uses the closed-form solution for Brownian bridge.
    
    Args:
        x0: Initial conditions (n_samples, n_features)
        x1: Terminal conditions (n_samples, n_features)
        time_grid: Time points (n_steps,)
        sigma: Diffusion coefficient
    
    Returns:
        Paths (n_samples, n_steps, n_features)
    """
    n_samples, n_features = x0.shape
    n_steps = len(time_grid)
    T = time_grid[-1] - time_grid[0]
    
    paths = np.zeros((n_samples, n_steps, n_features))
    
    for i in prange(n_samples):
        paths[i, 0, :] = x0[i, :]
        paths[i, -1, :] = x1[i, :]
        
        for t_idx in range(1, n_steps - 1):
            t = time_grid[t_idx] - time_grid[0]
            
            for k in range(n_features):
                # Brownian bridge mean
                mean = x0[i, k] + (x1[i, k] - x0[i, k]) * t / T
                
                # Brownian bridge variance
                var = sigma ** 2 * t * (T - t) / T
                
                # Sample
                paths[i, t_idx, k] = mean + np.sqrt(var) * np.random.randn()
    
    return paths


class NumbaSB(TimeSeriesGenerator):
    """
    Numba-Accelerated Markovian Schrödinger Bridge.
    
    Simple but fast implementation using closed-form Brownian bridge.
    Useful as a baseline and for quick generation.
    """
    
    MODEL_TYPE = "numba_sb"
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Numba SB.
        
        Args:
            config: Configuration with keys:
                - numba_sb_sigma: Diffusion coefficient (default: 0.1)
        """
        super().__init__(config)
        
        self.sigma = config.get('numba_sb_sigma', 0.1)
        
        self.x0_samples = None
        self.x1_samples = None
        self.time_grid = None
        self.n_features = None
    
    def fit(
        self,
        data: np.ndarray,
        time_grid: np.ndarray,
        verbose: bool = True
    ) -> 'NumbaSB':
        """
        Fit Numba SB (just stores data for sampling).
        
        Args:
            data: Time series data (n_samples, seq_len, n_features)
            time_grid: Time points
            verbose: Whether to print progress
        
        Returns:
            self for method chaining
        """
        if data.ndim == 2:
            data = data[:, :, np.newaxis]
        
        n_samples, seq_len, n_features = data.shape
        self.n_features = n_features
        self.time_grid = time_grid
        
        # Store initial and terminal conditions
        self.x0_samples = data[:, 0, :].copy()
        self.x1_samples = data[:, -1, :].copy()
        
        # Estimate sigma from data
        returns = np.diff(data, axis=1)
        dt = time_grid[1] - time_grid[0] if len(time_grid) > 1 else 1.0
        self.sigma = np.std(returns) / np.sqrt(dt)
        
        self.is_fitted = True
        
        if verbose:
            print(f"[Numba SB] Fitted with σ = {self.sigma:.4f}")
        
        return self
    
    def generate(
        self,
        n_samples: int,
        n_steps: Optional[int] = None,
        x0: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate synthetic time series using Brownian bridge.
        
        Args:
            n_samples: Number of samples to generate
            n_steps: Number of time steps
            x0: Initial conditions (optional)
        
        Returns:
            Generated paths (n_samples, n_steps, n_features)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before generation")
        
        if n_steps is None:
            n_steps = len(self.time_grid)
        
        # Create time grid
        time_grid = np.linspace(
            self.time_grid[0],
            self.time_grid[-1],
            n_steps
        ).astype(np.float64)
        
        # Sample initial and terminal conditions
        if x0 is None:
            idx = np.random.choice(len(self.x0_samples), n_samples, replace=True)
            x0 = self.x0_samples[idx]
        
        idx = np.random.choice(len(self.x1_samples), n_samples, replace=True)
        x1 = self.x1_samples[idx]
        
        # Generate using Numba
        paths = _markovian_sb_forward_numba(
            x0.astype(np.float64),
            x1.astype(np.float64),
            time_grid,
            self.sigma
        )
        
        return paths
