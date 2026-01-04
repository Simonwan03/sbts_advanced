"""
Light Schrödinger Bridge (LightSB) Implementation

Implements the LightSB algorithm from "Light Schrödinger Bridge" paper,
which uses variance annealing for efficient training.

Key Features:
    - Variance Annealing: Gradually reduce noise during training
    - Mini-batch OT: Efficient optimal transport computation
    - Score Matching: Train score network to estimate drift

Reference:
    Korotin et al., "Light Schrödinger Bridge", ICLR 2023

Author: Manus AI
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, Union, List
import warnings
from numba import njit, prange

from models.base import TimeSeriesGenerator

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============================================
# Numba-Accelerated Optimal Transport
# ============================================

@njit(cache=True, parallel=True)
def _compute_cost_matrix_numba(
    X: np.ndarray,
    Y: np.ndarray
) -> np.ndarray:
    """
    Compute squared Euclidean cost matrix.
    
    Args:
        X: Source points (n, d)
        Y: Target points (m, d)
    
    Returns:
        Cost matrix (n, m)
    """
    n = X.shape[0]
    m = Y.shape[0]
    d = X.shape[1]
    
    C = np.zeros((n, m))
    
    for i in prange(n):
        for j in range(m):
            dist = 0.0
            for k in range(d):
                diff = X[i, k] - Y[j, k]
                dist += diff * diff
            C[i, j] = dist
    
    return C


@njit(cache=True)
def _sinkhorn_numba(
    C: np.ndarray,
    epsilon: float,
    n_iter: int = 100,
    tol: float = 1e-6
) -> np.ndarray:
    """
    Sinkhorn algorithm for entropic optimal transport.
    
    Args:
        C: Cost matrix (n, m)
        epsilon: Regularization parameter
        n_iter: Maximum iterations
        tol: Convergence tolerance
    
    Returns:
        Transport plan (n, m)
    """
    n, m = C.shape
    
    # Initialize
    K = np.exp(-C / epsilon)
    u = np.ones(n)
    v = np.ones(m)
    
    a = np.ones(n) / n  # Source marginal
    b = np.ones(m) / m  # Target marginal
    
    for _ in range(n_iter):
        u_prev = u.copy()
        
        # Update u
        Kv = np.dot(K, v)
        u = a / (Kv + 1e-10)
        
        # Update v
        Ku = np.dot(K.T, u)
        v = b / (Ku + 1e-10)
        
        # Check convergence
        if np.max(np.abs(u - u_prev)) < tol:
            break
    
    # Compute transport plan
    P = np.outer(u, v) * K
    
    return P


@njit(cache=True)
def _sample_coupling_numba(
    P: np.ndarray,
    n_samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample from transport plan.
    
    Args:
        P: Transport plan (n, m)
        n_samples: Number of samples
    
    Returns:
        Tuple of (source_indices, target_indices)
    """
    n, m = P.shape
    
    # Flatten and normalize
    P_flat = P.flatten()
    P_flat = P_flat / P_flat.sum()
    
    # Sample indices
    cumsum = np.cumsum(P_flat)
    source_idx = np.zeros(n_samples, dtype=np.int64)
    target_idx = np.zeros(n_samples, dtype=np.int64)
    
    for i in range(n_samples):
        r = np.random.random()
        idx = np.searchsorted(cumsum, r)
        idx = min(idx, len(P_flat) - 1)
        
        source_idx[i] = idx // m
        target_idx[i] = idx % m
    
    return source_idx, target_idx


# ============================================
# Score Network
# ============================================

if TORCH_AVAILABLE:
    
    class ScoreNetwork(nn.Module):
        """
        Score network for estimating ∇ log p_t(x).
        
        Uses a time-conditioned MLP architecture.
        """
        
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 256,
            n_layers: int = 3,
            time_embed_dim: int = 64
        ):
            super().__init__()
            
            self.input_dim = input_dim
            self.time_embed_dim = time_embed_dim
            
            # Time embedding
            self.time_embed = nn.Sequential(
                nn.Linear(1, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim)
            )
            
            # Main network
            layers = []
            in_dim = input_dim + time_embed_dim
            
            for i in range(n_layers):
                out_dim = hidden_dim if i < n_layers - 1 else input_dim
                layers.extend([
                    nn.Linear(in_dim, out_dim),
                    nn.SiLU() if i < n_layers - 1 else nn.Identity()
                ])
                in_dim = hidden_dim
            
            self.net = nn.Sequential(*layers)
        
        def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor
        ) -> torch.Tensor:
            """
            Forward pass.
            
            Args:
                x: Input (batch, input_dim)
                t: Time (batch, 1) or (batch,)
            
            Returns:
                Score estimate (batch, input_dim)
            """
            if t.dim() == 1:
                t = t.unsqueeze(-1)
            
            # Time embedding
            t_embed = self.time_embed(t)
            
            # Concatenate and forward
            h = torch.cat([x, t_embed], dim=-1)
            score = self.net(h)
            
            return score


# ============================================
# LightSB Model
# ============================================

class LightSB(TimeSeriesGenerator):
    """
    Light Schrödinger Bridge for Time Series Generation.
    
    Uses variance annealing and mini-batch optimal transport for
    efficient training of Schrödinger Bridge.
    
    Key Components:
        1. Variance Schedule: σ(t) = σ_min + (σ_max - σ_min) * t
        2. Mini-batch OT: Sinkhorn algorithm for coupling
        3. Score Matching: Train score network on interpolated samples
    
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
            config: Configuration with keys:
                - lightsb_sigma_min: Minimum noise level (default: 0.01)
                - lightsb_sigma_max: Maximum noise level (default: 1.0)
                - lightsb_hidden_dim: Score network hidden dim (default: 256)
                - lightsb_n_layers: Score network layers (default: 3)
                - lightsb_epochs: Training epochs (default: 100)
                - lightsb_lr: Learning rate (default: 0.001)
                - lightsb_batch_size: Batch size (default: 256)
                - lightsb_ot_epsilon: OT regularization (default: 0.1)
                - lightsb_n_steps: Number of diffusion steps (default: 50)
        """
        super().__init__(config)
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for LightSB")
        
        self.sigma_min = config.get('lightsb_sigma_min', 0.01)
        self.sigma_max = config.get('lightsb_sigma_max', 1.0)
        self.hidden_dim = config.get('lightsb_hidden_dim', 256)
        self.n_layers = config.get('lightsb_n_layers', 3)
        self.epochs = config.get('lightsb_epochs', 100)
        self.lr = config.get('lightsb_lr', 0.001)
        self.batch_size = config.get('lightsb_batch_size', 256)
        self.ot_epsilon = config.get('lightsb_ot_epsilon', 0.1)
        self.n_steps = config.get('lightsb_n_steps', 50)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model components
        self.score_net = None
        self.n_features = None
        self.seq_len = None
        self.data_mean = None
        self.data_std = None
        
        # Training history
        self.train_losses = []
    
    def _get_sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Get noise level at time t."""
        return self.sigma_min + (self.sigma_max - self.sigma_min) * t
    
    def _interpolate(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Interpolate between x0 and x1 at time t.
        
        Uses the formula: x_t = (1-t)*x0 + t*x1 + σ(t)*ε
        """
        sigma = self._get_sigma(t)
        noise = torch.randn_like(x0)
        
        # Reshape t for broadcasting
        while t.dim() < x0.dim():
            t = t.unsqueeze(-1)
        while sigma.dim() < x0.dim():
            sigma = sigma.unsqueeze(-1)
        
        x_t = (1 - t) * x0 + t * x1 + sigma * noise
        
        return x_t, noise
    
    def _compute_target_score(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute target score for training.
        
        Target: ∇ log p_t(x_t | x0, x1) = -noise / σ(t)
        """
        sigma = self._get_sigma(t)
        while sigma.dim() < noise.dim():
            sigma = sigma.unsqueeze(-1)
        
        return -noise / (sigma + 1e-8)
    
    def fit(
        self,
        data: np.ndarray,
        time_grid: np.ndarray,
        verbose: bool = True
    ) -> 'LightSB':
        """
        Fit LightSB model.
        
        Training Process:
            1. Compute mini-batch OT coupling between consecutive time slices
            2. Sample interpolated points along the bridge
            3. Train score network to match target score
        
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
        
        # Normalize data
        self.data_mean = np.mean(data)
        self.data_std = np.std(data) + 1e-8
        data_norm = (data - self.data_mean) / self.data_std
        
        # Flatten time series for training
        # Treat each time series as a single high-dimensional point
        flat_dim = seq_len * n_features
        data_flat = data_norm.reshape(n_samples, flat_dim)
        
        # Create source (noise) and target (data) distributions
        X_target = torch.tensor(data_flat, dtype=torch.float32).to(self.device)
        
        # Initialize score network
        self.score_net = ScoreNetwork(
            input_dim=flat_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.score_net.parameters(), lr=self.lr)
        
        if verbose:
            print("=" * 60)
            print("LightSB Training")
            print("=" * 60)
        
        self.train_losses = []
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            # Shuffle data
            perm = torch.randperm(n_samples)
            X_target = X_target[perm]
            
            for i in range(0, n_samples, self.batch_size):
                batch_target = X_target[i:i+self.batch_size]
                batch_size = len(batch_target)
                
                # Sample source (noise)
                batch_source = torch.randn_like(batch_target)
                
                # Compute OT coupling using Numba
                C = _compute_cost_matrix_numba(
                    batch_source.cpu().numpy(),
                    batch_target.cpu().numpy()
                )
                P = _sinkhorn_numba(C, self.ot_epsilon)
                
                # Sample from coupling
                src_idx, tgt_idx = _sample_coupling_numba(P, batch_size)
                
                x0 = batch_source[src_idx]
                x1 = batch_target[tgt_idx]
                
                # Sample random time
                t = torch.rand(batch_size, device=self.device)
                
                # Interpolate
                x_t, noise = self._interpolate(x0, x1, t)
                
                # Compute target score
                target_score = self._compute_target_score(x0, x1, x_t, t, noise)
                
                # Forward pass
                optimizer.zero_grad()
                pred_score = self.score_net(x_t, t)
                
                # Score matching loss
                loss = F.mse_loss(pred_score, target_score)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.score_net.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            self.train_losses.append(avg_loss)
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"  [LightSB] Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")
        
        self.score_net.eval()
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
        
        Uses the trained score network to simulate the reverse SDE.
        
        Args:
            n_samples: Number of samples to generate
            n_steps: Number of diffusion steps (default: self.n_steps)
            x0: Initial noise (default: sample from N(0, I))
        
        Returns:
            Generated time series (n_samples, seq_len, n_features)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before generation")
        
        if n_steps is None:
            n_steps = self.n_steps
        
        flat_dim = self.seq_len * self.n_features
        
        # Initialize from noise
        if x0 is None:
            x = torch.randn(n_samples, flat_dim, device=self.device)
        else:
            x = torch.tensor(x0, dtype=torch.float32, device=self.device)
        
        # Time steps
        dt = 1.0 / n_steps
        
        self.score_net.eval()
        
        with torch.no_grad():
            for i in range(n_steps):
                t = torch.full((n_samples,), i * dt, device=self.device)
                
                # Get score
                score = self.score_net(x, t)
                
                # Get sigma
                sigma = self._get_sigma(t).unsqueeze(-1)
                
                # Euler-Maruyama step (reverse SDE)
                # dx = [drift + σ² * score] dt + σ dW
                drift = score * sigma ** 2
                noise = torch.randn_like(x) * np.sqrt(dt)
                
                x = x + drift * dt + sigma * noise
        
        # Reshape and denormalize
        x = x.cpu().numpy()
        x = x.reshape(n_samples, self.seq_len, self.n_features)
        x = x * self.data_std + self.data_mean
        
        return x
    
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
