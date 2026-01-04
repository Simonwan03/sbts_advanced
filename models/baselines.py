"""
Baseline Models for Comparison

Implements state-of-the-art time series generation baselines:
    - TimeGAN: Yoon et al., NeurIPS 2019
    - Diffusion-TS: Simplified diffusion model for time series

These are simplified implementations for benchmarking purposes.

Author: Manus AI
"""

import numpy as np
from typing import Dict, Any, Optional, List
import warnings

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
# TimeGAN Implementation
# ============================================

if TORCH_AVAILABLE:
    
    class EmbeddingNetwork(nn.Module):
        """Embedding network for TimeGAN."""
        
        def __init__(self, input_dim: int, hidden_dim: int, n_layers: int = 2):
            super().__init__()
            
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True
            )
            self.fc = nn.Linear(hidden_dim, hidden_dim)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h, _ = self.rnn(x)
            return torch.sigmoid(self.fc(h))
    
    
    class RecoveryNetwork(nn.Module):
        """Recovery network for TimeGAN."""
        
        def __init__(self, hidden_dim: int, output_dim: int, n_layers: int = 2):
            super().__init__()
            
            self.rnn = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True
            )
            self.fc = nn.Linear(hidden_dim, output_dim)
        
        def forward(self, h: torch.Tensor) -> torch.Tensor:
            out, _ = self.rnn(h)
            return self.fc(out)
    
    
    class GeneratorNetwork(nn.Module):
        """Generator network for TimeGAN."""
        
        def __init__(self, z_dim: int, hidden_dim: int, n_layers: int = 2):
            super().__init__()
            
            self.rnn = nn.GRU(
                input_size=z_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True
            )
            self.fc = nn.Linear(hidden_dim, hidden_dim)
        
        def forward(self, z: torch.Tensor) -> torch.Tensor:
            h, _ = self.rnn(z)
            return torch.sigmoid(self.fc(h))
    
    
    class SupervisorNetwork(nn.Module):
        """Supervisor network for TimeGAN."""
        
        def __init__(self, hidden_dim: int, n_layers: int = 2):
            super().__init__()
            
            self.rnn = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers - 1,
                batch_first=True
            )
            self.fc = nn.Linear(hidden_dim, hidden_dim)
        
        def forward(self, h: torch.Tensor) -> torch.Tensor:
            out, _ = self.rnn(h)
            return torch.sigmoid(self.fc(out))
    
    
    class DiscriminatorNetwork(nn.Module):
        """Discriminator network for TimeGAN."""
        
        def __init__(self, hidden_dim: int, n_layers: int = 2):
            super().__init__()
            
            self.rnn = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True
            )
            self.fc = nn.Linear(hidden_dim, 1)
        
        def forward(self, h: torch.Tensor) -> torch.Tensor:
            out, _ = self.rnn(h)
            return self.fc(out)


class TimeGAN(TimeSeriesGenerator):
    """
    TimeGAN: Time-series Generative Adversarial Network.
    
    Reference: Yoon et al., "Time-series Generative Adversarial Networks", NeurIPS 2019
    
    Architecture:
        - Embedding Network: Maps data to latent space
        - Recovery Network: Maps latent space back to data
        - Generator: Generates latent sequences from noise
        - Supervisor: Guides generator with temporal dynamics
        - Discriminator: Distinguishes real from fake
    
    Training Phases:
        1. Autoencoder training (embedding + recovery)
        2. Supervised training (supervisor)
        3. Joint training (all networks)
    """
    
    MODEL_TYPE = "timegan"
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize TimeGAN.
        
        Args:
            config: Configuration with keys:
                - timegan_hidden_dim: Hidden dimension (default: 64)
                - timegan_z_dim: Noise dimension (default: 32)
                - timegan_n_layers: Number of RNN layers (default: 2)
                - timegan_epochs: Training epochs per phase (default: 50)
                - timegan_lr: Learning rate (default: 0.001)
                - timegan_batch_size: Batch size (default: 128)
        """
        super().__init__(config)
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for TimeGAN")
        
        self.hidden_dim = config.get('timegan_hidden_dim', 64)
        self.z_dim = config.get('timegan_z_dim', 32)
        self.n_layers = config.get('timegan_n_layers', 2)
        self.epochs = config.get('timegan_epochs', 50)
        self.lr = config.get('timegan_lr', 0.001)
        self.batch_size = config.get('timegan_batch_size', 128)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Networks
        self.embedder = None
        self.recovery = None
        self.generator = None
        self.supervisor = None
        self.discriminator = None
        
        # Data info
        self.n_features = None
        self.seq_len = None
        self.data_mean = None
        self.data_std = None
    
    def fit(
        self,
        data: np.ndarray,
        time_grid: np.ndarray,
        verbose: bool = True
    ) -> 'TimeGAN':
        """
        Fit TimeGAN model.
        
        Args:
            data: Time series data (n_samples, seq_len, n_features)
            time_grid: Time points (unused, for API compatibility)
            verbose: Whether to print progress
        
        Returns:
            self for method chaining
        """
        if data.ndim == 2:
            data = data[:, :, np.newaxis]
        
        n_samples, seq_len, n_features = data.shape
        self.n_features = n_features
        self.seq_len = seq_len
        
        # Normalize
        self.data_mean = np.mean(data)
        self.data_std = np.std(data) + 1e-8
        data_norm = (data - self.data_mean) / self.data_std
        
        X = torch.tensor(data_norm, dtype=torch.float32).to(self.device)
        
        # Initialize networks
        self.embedder = EmbeddingNetwork(n_features, self.hidden_dim, self.n_layers).to(self.device)
        self.recovery = RecoveryNetwork(self.hidden_dim, n_features, self.n_layers).to(self.device)
        self.generator = GeneratorNetwork(self.z_dim, self.hidden_dim, self.n_layers).to(self.device)
        self.supervisor = SupervisorNetwork(self.hidden_dim, self.n_layers).to(self.device)
        self.discriminator = DiscriminatorNetwork(self.hidden_dim, self.n_layers).to(self.device)
        
        if verbose:
            print("=" * 60)
            print("TimeGAN Training")
            print("=" * 60)
        
        # Phase 1: Autoencoder training
        if verbose:
            print("\n[Phase 1] Autoencoder Training...")
        
        ae_optimizer = torch.optim.Adam(
            list(self.embedder.parameters()) + list(self.recovery.parameters()),
            lr=self.lr
        )
        
        for epoch in range(self.epochs):
            perm = torch.randperm(n_samples)
            X = X[perm]
            
            total_loss = 0.0
            n_batches = 0
            
            for i in range(0, n_samples, self.batch_size):
                batch = X[i:i+self.batch_size]
                
                ae_optimizer.zero_grad()
                
                h = self.embedder(batch)
                x_tilde = self.recovery(h)
                
                loss = F.mse_loss(x_tilde, batch)
                loss.backward()
                ae_optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}/{self.epochs}, AE Loss: {total_loss/n_batches:.6f}")
        
        # Phase 2: Supervised training
        if verbose:
            print("\n[Phase 2] Supervised Training...")
        
        sup_optimizer = torch.optim.Adam(self.supervisor.parameters(), lr=self.lr)
        
        for epoch in range(self.epochs):
            perm = torch.randperm(n_samples)
            X = X[perm]
            
            total_loss = 0.0
            n_batches = 0
            
            for i in range(0, n_samples, self.batch_size):
                batch = X[i:i+self.batch_size]
                
                sup_optimizer.zero_grad()
                
                with torch.no_grad():
                    h = self.embedder(batch)
                
                h_hat = self.supervisor(h)
                
                # Supervised loss: predict next step
                loss = F.mse_loss(h_hat[:, :-1, :], h[:, 1:, :])
                loss.backward()
                sup_optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}/{self.epochs}, Sup Loss: {total_loss/n_batches:.6f}")
        
        # Phase 3: Joint training
        if verbose:
            print("\n[Phase 3] Joint Training...")
        
        g_optimizer = torch.optim.Adam(
            list(self.generator.parameters()) + list(self.supervisor.parameters()),
            lr=self.lr
        )
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)
        
        for epoch in range(self.epochs):
            perm = torch.randperm(n_samples)
            X = X[perm]
            
            g_loss_total = 0.0
            d_loss_total = 0.0
            n_batches = 0
            
            for i in range(0, n_samples, self.batch_size):
                batch = X[i:i+self.batch_size]
                batch_size = len(batch)
                
                # Generate fake data
                z = torch.randn(batch_size, seq_len, self.z_dim, device=self.device)
                h_fake = self.generator(z)
                h_fake_sup = self.supervisor(h_fake)
                
                with torch.no_grad():
                    h_real = self.embedder(batch)
                
                # Train discriminator
                d_optimizer.zero_grad()
                
                d_real = self.discriminator(h_real)
                d_fake = self.discriminator(h_fake_sup.detach())
                
                d_loss = F.binary_cross_entropy_with_logits(
                    d_real, torch.ones_like(d_real)
                ) + F.binary_cross_entropy_with_logits(
                    d_fake, torch.zeros_like(d_fake)
                )
                d_loss.backward()
                d_optimizer.step()
                
                # Train generator
                g_optimizer.zero_grad()
                
                z = torch.randn(batch_size, seq_len, self.z_dim, device=self.device)
                h_fake = self.generator(z)
                h_fake_sup = self.supervisor(h_fake)
                
                d_fake = self.discriminator(h_fake_sup)
                
                g_loss = F.binary_cross_entropy_with_logits(
                    d_fake, torch.ones_like(d_fake)
                )
                g_loss.backward()
                g_optimizer.step()
                
                g_loss_total += g_loss.item()
                d_loss_total += d_loss.item()
                n_batches += 1
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}/{self.epochs}, G Loss: {g_loss_total/n_batches:.4f}, D Loss: {d_loss_total/n_batches:.4f}")
        
        self.is_fitted = True
        
        if verbose:
            print("=" * 60)
            print("TimeGAN Training Complete!")
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
        
        Args:
            n_samples: Number of samples
            n_steps: Sequence length (default: training seq_len)
            x0: Unused (for API compatibility)
        
        Returns:
            Generated data (n_samples, seq_len, n_features)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before generation")
        
        if n_steps is None:
            n_steps = self.seq_len
        
        self.generator.eval()
        self.supervisor.eval()
        self.recovery.eval()
        
        with torch.no_grad():
            z = torch.randn(n_samples, n_steps, self.z_dim, device=self.device)
            h_fake = self.generator(z)
            h_fake_sup = self.supervisor(h_fake)
            x_fake = self.recovery(h_fake_sup)
        
        x_fake = x_fake.cpu().numpy()
        x_fake = x_fake * self.data_std + self.data_mean
        
        return x_fake


# ============================================
# Diffusion-TS Implementation
# ============================================

if TORCH_AVAILABLE:
    
    class DiffusionUNet(nn.Module):
        """
        Simple U-Net style architecture for diffusion denoising.
        """
        
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 128,
            time_embed_dim: int = 64
        ):
            super().__init__()
            
            # Time embedding
            self.time_embed = nn.Sequential(
                nn.Linear(1, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim)
            )
            
            # Encoder
            self.enc1 = nn.Linear(input_dim + time_embed_dim, hidden_dim)
            self.enc2 = nn.Linear(hidden_dim, hidden_dim * 2)
            
            # Decoder
            self.dec1 = nn.Linear(hidden_dim * 2, hidden_dim)
            self.dec2 = nn.Linear(hidden_dim * 2, input_dim)  # Skip connection
            
            self.act = nn.SiLU()
        
        def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor
        ) -> torch.Tensor:
            """
            Forward pass.
            
            Args:
                x: Noisy input (batch, seq_len, input_dim)
                t: Time step (batch,) or (batch, 1)
            
            Returns:
                Predicted noise (batch, seq_len, input_dim)
            """
            batch, seq_len, _ = x.shape
            
            if t.dim() == 1:
                t = t.unsqueeze(-1)
            
            # Time embedding
            t_embed = self.time_embed(t)  # (batch, time_embed_dim)
            t_embed = t_embed.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, time_embed_dim)
            
            # Encoder
            h = torch.cat([x, t_embed], dim=-1)
            h1 = self.act(self.enc1(h))
            h2 = self.act(self.enc2(h1))
            
            # Decoder with skip connection
            h3 = self.act(self.dec1(h2))
            h4 = torch.cat([h3, h1], dim=-1)
            out = self.dec2(h4)
            
            return out


class DiffusionTS(TimeSeriesGenerator):
    """
    Diffusion Model for Time Series Generation.
    
    Simplified DDPM-style diffusion model adapted for time series.
    
    Process:
        Forward: x_t = √(α_t) * x_0 + √(1 - α_t) * ε
        Reverse: Predict ε and denoise iteratively
    """
    
    MODEL_TYPE = "diffusion_ts"
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Diffusion-TS.
        
        Args:
            config: Configuration with keys:
                - diffusion_hidden_dim: Hidden dimension (default: 128)
                - diffusion_n_steps: Diffusion steps (default: 100)
                - diffusion_epochs: Training epochs (default: 100)
                - diffusion_lr: Learning rate (default: 0.001)
                - diffusion_batch_size: Batch size (default: 64)
                - diffusion_beta_start: Start beta (default: 0.0001)
                - diffusion_beta_end: End beta (default: 0.02)
        """
        super().__init__(config)
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for Diffusion-TS")
        
        self.hidden_dim = config.get('diffusion_hidden_dim', 128)
        self.n_diffusion_steps = config.get('diffusion_n_steps', 100)
        self.epochs = config.get('diffusion_epochs', 100)
        self.lr = config.get('diffusion_lr', 0.001)
        self.batch_size = config.get('diffusion_batch_size', 64)
        self.beta_start = config.get('diffusion_beta_start', 0.0001)
        self.beta_end = config.get('diffusion_beta_end', 0.02)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Noise schedule
        self.betas = None
        self.alphas = None
        self.alpha_bars = None
        
        # Model
        self.model = None
        self.n_features = None
        self.seq_len = None
        self.data_mean = None
        self.data_std = None
    
    def _setup_noise_schedule(self):
        """Setup linear noise schedule."""
        self.betas = torch.linspace(
            self.beta_start, self.beta_end, self.n_diffusion_steps,
            device=self.device
        )
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
    
    def _q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward diffusion: sample x_t from x_0.
        
        x_t = √(α̅_t) * x_0 + √(1 - α̅_t) * ε
        """
        if noise is None:
            noise = torch.randn_like(x0)
        
        alpha_bar = self.alpha_bars[t]
        while alpha_bar.dim() < x0.dim():
            alpha_bar = alpha_bar.unsqueeze(-1)
        
        return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise
    
    def fit(
        self,
        data: np.ndarray,
        time_grid: np.ndarray,
        verbose: bool = True
    ) -> 'DiffusionTS':
        """
        Fit Diffusion-TS model.
        
        Args:
            data: Time series data
            time_grid: Time points (unused)
            verbose: Whether to print progress
        
        Returns:
            self for method chaining
        """
        if data.ndim == 2:
            data = data[:, :, np.newaxis]
        
        n_samples, seq_len, n_features = data.shape
        self.n_features = n_features
        self.seq_len = seq_len
        
        # Normalize
        self.data_mean = np.mean(data)
        self.data_std = np.std(data) + 1e-8
        data_norm = (data - self.data_mean) / self.data_std
        
        X = torch.tensor(data_norm, dtype=torch.float32).to(self.device)
        
        # Setup noise schedule
        self._setup_noise_schedule()
        
        # Initialize model
        self.model = DiffusionUNet(
            input_dim=n_features,
            hidden_dim=self.hidden_dim
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        if verbose:
            print("=" * 60)
            print("Diffusion-TS Training")
            print("=" * 60)
        
        for epoch in range(self.epochs):
            perm = torch.randperm(n_samples)
            X = X[perm]
            
            total_loss = 0.0
            n_batches = 0
            
            for i in range(0, n_samples, self.batch_size):
                batch = X[i:i+self.batch_size]
                batch_size = len(batch)
                
                optimizer.zero_grad()
                
                # Sample random time steps
                t = torch.randint(0, self.n_diffusion_steps, (batch_size,), device=self.device)
                
                # Sample noise
                noise = torch.randn_like(batch)
                
                # Forward diffusion
                x_t = self._q_sample(batch, t, noise)
                
                # Predict noise
                t_normalized = t.float() / self.n_diffusion_steps
                pred_noise = self.model(x_t, t_normalized)
                
                # Loss
                loss = F.mse_loss(pred_noise, noise)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"  [Diffusion-TS] Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/n_batches:.6f}")
        
        self.model.eval()
        self.is_fitted = True
        
        if verbose:
            print("=" * 60)
            print("Diffusion-TS Training Complete!")
            print("=" * 60)
        
        return self
    
    def generate(
        self,
        n_samples: int,
        n_steps: Optional[int] = None,
        x0: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate synthetic time series via reverse diffusion.
        
        Args:
            n_samples: Number of samples
            n_steps: Sequence length (default: training seq_len)
            x0: Initial noise (default: sample from N(0, I))
        
        Returns:
            Generated data (n_samples, seq_len, n_features)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before generation")
        
        if n_steps is None:
            n_steps = self.seq_len
        
        self.model.eval()
        
        # Start from noise
        x = torch.randn(n_samples, n_steps, self.n_features, device=self.device)
        
        with torch.no_grad():
            # Reverse diffusion
            for t in reversed(range(self.n_diffusion_steps)):
                t_batch = torch.full((n_samples,), t, device=self.device)
                t_normalized = t_batch.float() / self.n_diffusion_steps
                
                # Predict noise
                pred_noise = self.model(x, t_normalized)
                
                # Denoise step
                alpha = self.alphas[t]
                alpha_bar = self.alpha_bars[t]
                
                if t > 0:
                    noise = torch.randn_like(x)
                    sigma = torch.sqrt(self.betas[t])
                else:
                    noise = 0
                    sigma = 0
                
                x = (1 / torch.sqrt(alpha)) * (
                    x - (self.betas[t] / torch.sqrt(1 - alpha_bar)) * pred_noise
                ) + sigma * noise
        
        x = x.cpu().numpy()
        x = x * self.data_std + self.data_mean
        
        return x
