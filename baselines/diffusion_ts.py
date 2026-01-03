"""
Diffusion-TS: Diffusion Model for Time Series Generation

Based on diffusion models for time series synthesis.
References:
- "Diffusion Models for Time Series Applications" (various 2023-2024 papers)
- "TransFusion: Generating Long, High Fidelity Time Series using Diffusion Models"

This is a simplified DDPM-style implementation for benchmarking.

Key Components:
1. Noise Schedule: Linear beta schedule
2. Denoising Network: Transformer-based architecture
3. Sampling: DDPM reverse process
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal embeddings for diffusion timestep."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TransformerBlock(nn.Module):
    """Transformer block for denoising network."""
    def __init__(self, dim, heads=4, dim_head=32, mlp_dim=128, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class DenoisingNetwork(nn.Module):
    """
    Transformer-based denoising network for diffusion.
    
    Takes noisy time series and diffusion timestep,
    predicts the noise to be removed.
    """
    def __init__(self, seq_len, n_features, hidden_dim=64, n_layers=4, n_heads=4):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(n_features, hidden_dim)
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Positional encoding for sequence
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, hidden_dim // n_heads, hidden_dim * 2)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_features)
        )
        
    def forward(self, x, t):
        """
        Forward pass.
        
        Args:
            x: (batch, seq_len, n_features) noisy input
            t: (batch,) diffusion timestep
            
        Returns:
            noise_pred: (batch, seq_len, n_features) predicted noise
        """
        batch_size = x.size(0)
        
        # Project input
        h = self.input_proj(x)  # (batch, seq_len, hidden_dim)
        
        # Add positional encoding
        h = h + self.pos_embed
        
        # Add time embedding (broadcast to all positions)
        t_emb = self.time_mlp(t)  # (batch, hidden_dim)
        h = h + t_emb.unsqueeze(1)
        
        # Transformer blocks
        for block in self.blocks:
            h = block(h)
        
        # Output projection
        noise_pred = self.output_proj(h)
        
        return noise_pred


class DiffusionTS:
    """
    Diffusion Model for Time Series Generation.
    
    Implements DDPM-style diffusion for generating time series data.
    
    Usage:
        diffusion = DiffusionTS(seq_len=60, n_features=5)
        diffusion.fit(training_data, epochs=100)
        generated = diffusion.generate(n_samples=100)
    """
    
    def __init__(self, seq_len, n_features, n_steps=100, hidden_dim=64, 
                 n_layers=4, device=None):
        """
        Initialize Diffusion-TS.
        
        Args:
            seq_len: Length of time series
            n_features: Number of features
            n_steps: Number of diffusion steps
            hidden_dim: Hidden dimension
            n_layers: Number of transformer layers
            device: Torch device
        """
        self.seq_len = seq_len
        self.n_features = n_features
        self.n_steps = n_steps
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Noise schedule (linear)
        self.betas = torch.linspace(1e-4, 0.02, n_steps).to(self.device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(self.device), 
                                               self.alphas_cumprod[:-1]])
        
        # Precompute values for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1 / self.alphas)
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        
        # Denoising network
        self.model = DenoisingNetwork(
            seq_len, n_features, hidden_dim, n_layers
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.mse_loss = nn.MSELoss()
        
    def _q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0).
        
        Args:
            x_0: Clean data
            t: Timestep
            noise: Optional noise (will be sampled if None)
            
        Returns:
            x_t: Noisy data at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        
        return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise
    
    def fit(self, data, epochs=100, batch_size=64, verbose=True):
        """
        Train diffusion model.
        
        Args:
            data: (N, T, D) numpy array of training data
            epochs: Number of epochs
            batch_size: Batch size
            verbose: Whether to print progress
        """
        # Normalize data
        self.data_mean = data.mean(axis=(0, 1), keepdims=True)
        self.data_std = data.std(axis=(0, 1), keepdims=True) + 1e-8
        data_norm = (data - self.data_mean) / self.data_std
        
        # Create dataloader
        dataset = TensorDataset(torch.FloatTensor(data_norm))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        
        if verbose:
            print("   [Diffusion-TS] Training denoising network...")
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                x_0 = batch[0].to(self.device)
                batch_size_curr = x_0.size(0)
                
                # Sample random timesteps
                t = torch.randint(0, self.n_steps, (batch_size_curr,), device=self.device)
                
                # Sample noise
                noise = torch.randn_like(x_0)
                
                # Get noisy data
                x_t = self._q_sample(x_0, t, noise)
                
                # Predict noise
                noise_pred = self.model(x_t, t.float())
                
                # Loss
                loss = self.mse_loss(noise_pred, noise)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
        
        if verbose:
            print(f"   [Diffusion-TS] Training complete. Final loss: {total_loss/len(dataloader):.6f}")
    
    @torch.no_grad()
    def generate(self, n_samples):
        """
        Generate samples using DDPM sampling.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            generated: (n_samples, seq_len, n_features) numpy array
        """
        self.model.eval()
        
        # Start from pure noise
        x = torch.randn(n_samples, self.seq_len, self.n_features).to(self.device)
        
        # Reverse diffusion process
        for t in reversed(range(self.n_steps)):
            t_batch = torch.full((n_samples,), t, device=self.device, dtype=torch.float)
            
            # Predict noise
            noise_pred = self.model(x, t_batch)
            
            # Compute mean
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]
            
            mean = (1 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * noise_pred
            )
            
            # Add noise (except for t=0)
            if t > 0:
                noise = torch.randn_like(x)
                variance = self.posterior_variance[t]
                x = mean + torch.sqrt(variance) * noise
            else:
                x = mean
        
        # Denormalize
        x = x.cpu().numpy()
        x = x * self.data_std + self.data_mean
        
        return x.squeeze()
