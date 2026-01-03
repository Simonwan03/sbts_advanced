"""
TimeGAN: Time-series Generative Adversarial Network

Based on: "Time-series Generative Adversarial Networks" (Yoon et al., NeurIPS 2019)
Paper: https://papers.nips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html

This is a simplified implementation for benchmarking purposes.
For production use, consider the official implementation.

Key Components:
1. Embedding Network: Maps real data to latent space
2. Recovery Network: Maps latent space back to data space
3. Generator: Generates latent sequences from noise
4. Discriminator: Distinguishes real from generated latent sequences
5. Supervisor: Captures temporal dynamics in latent space
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class EmbeddingNetwork(nn.Module):
    """Maps data space to latent space."""
    def __init__(self, input_dim, hidden_dim, num_layers=3):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        h, _ = self.rnn(x)
        return self.activation(self.fc(h))


class RecoveryNetwork(nn.Module):
    """Maps latent space back to data space."""
    def __init__(self, hidden_dim, output_dim, num_layers=3):
        super().__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, h):
        h_out, _ = self.rnn(h)
        return self.fc(h_out)


class GeneratorNetwork(nn.Module):
    """Generates latent sequences from noise."""
    def __init__(self, noise_dim, hidden_dim, num_layers=3):
        super().__init__()
        self.rnn = nn.GRU(noise_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Sigmoid()
        
    def forward(self, z):
        h, _ = self.rnn(z)
        return self.activation(self.fc(h))


class SupervisorNetwork(nn.Module):
    """Captures temporal dynamics in latent space."""
    def __init__(self, hidden_dim, num_layers=2):
        super().__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Sigmoid()
        
    def forward(self, h):
        h_out, _ = self.rnn(h)
        return self.activation(self.fc(h_out))


class DiscriminatorNetwork(nn.Module):
    """Discriminates real from generated latent sequences."""
    def __init__(self, hidden_dim, num_layers=3):
        super().__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, h):
        h_out, _ = self.rnn(h)
        return self.fc(h_out)


class TimeGAN:
    """
    TimeGAN: Time-series Generative Adversarial Network.
    
    A GAN-based approach for generating realistic time series data
    that captures both static and temporal features.
    
    Usage:
        timegan = TimeGAN(seq_len=60, n_features=5)
        timegan.fit(training_data, epochs=100)
        generated = timegan.generate(n_samples=100)
    """
    
    def __init__(self, seq_len, n_features, hidden_dim=64, noise_dim=32, 
                 num_layers=3, device=None):
        """
        Initialize TimeGAN.
        
        Args:
            seq_len: Length of time series sequences
            n_features: Number of features/dimensions
            hidden_dim: Hidden dimension for all networks
            noise_dim: Dimension of noise input to generator
            num_layers: Number of RNN layers
            device: Torch device
        """
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.embedder = EmbeddingNetwork(n_features, hidden_dim, num_layers).to(self.device)
        self.recovery = RecoveryNetwork(hidden_dim, n_features, num_layers).to(self.device)
        self.generator = GeneratorNetwork(noise_dim, hidden_dim, num_layers).to(self.device)
        self.supervisor = SupervisorNetwork(hidden_dim, num_layers-1).to(self.device)
        self.discriminator = DiscriminatorNetwork(hidden_dim, num_layers).to(self.device)
        
        # Optimizers
        self.e_opt = optim.Adam(self.embedder.parameters(), lr=0.001)
        self.r_opt = optim.Adam(self.recovery.parameters(), lr=0.001)
        self.g_opt = optim.Adam(self.generator.parameters(), lr=0.001)
        self.s_opt = optim.Adam(self.supervisor.parameters(), lr=0.001)
        self.d_opt = optim.Adam(self.discriminator.parameters(), lr=0.001)
        
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def _random_generator(self, batch_size):
        """Generate random noise for generator input."""
        return torch.randn(batch_size, self.seq_len, self.noise_dim).to(self.device)
    
    def fit(self, data, epochs=100, batch_size=128, verbose=True):
        """
        Train TimeGAN on data.
        
        Args:
            data: (N, T, D) numpy array of training data
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Whether to print progress
        """
        # Normalize data
        self.data_min = data.min(axis=(0, 1), keepdims=True)
        self.data_max = data.max(axis=(0, 1), keepdims=True)
        data_norm = (data - self.data_min) / (self.data_max - self.data_min + 1e-8)
        
        # Create dataloader
        dataset = TensorDataset(torch.FloatTensor(data_norm))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Phase 1: Embedding Network Training
        if verbose:
            print("   [TimeGAN] Phase 1: Training Embedding Network...")
        for epoch in range(epochs // 3):
            for batch in dataloader:
                x = batch[0].to(self.device)
                
                # Forward
                h = self.embedder(x)
                x_tilde = self.recovery(h)
                
                # Reconstruction loss
                e_loss = self.mse_loss(x_tilde, x)
                
                # Backward
                self.e_opt.zero_grad()
                self.r_opt.zero_grad()
                e_loss.backward()
                self.e_opt.step()
                self.r_opt.step()
        
        # Phase 2: Supervised Training
        if verbose:
            print("   [TimeGAN] Phase 2: Supervised Training...")
        for epoch in range(epochs // 3):
            for batch in dataloader:
                x = batch[0].to(self.device)
                
                h = self.embedder(x)
                h_hat_supervise = self.supervisor(h)
                
                # Supervised loss (predict next step)
                s_loss = self.mse_loss(h_hat_supervise[:, :-1, :], h[:, 1:, :])
                
                self.s_opt.zero_grad()
                s_loss.backward()
                self.s_opt.step()
        
        # Phase 3: Joint Training
        if verbose:
            print("   [TimeGAN] Phase 3: Joint Training...")
        for epoch in range(epochs // 3):
            for batch in dataloader:
                x = batch[0].to(self.device)
                batch_size_curr = x.size(0)
                
                # Train Generator
                z = self._random_generator(batch_size_curr)
                h = self.embedder(x)
                h_hat = self.generator(z)
                h_hat_supervise = self.supervisor(h_hat)
                x_hat = self.recovery(h_hat_supervise)
                
                # Generator losses
                y_fake = self.discriminator(h_hat_supervise)
                g_loss_u = self.bce_loss(y_fake, torch.ones_like(y_fake))
                g_loss_s = self.mse_loss(h_hat_supervise[:, :-1, :], h_hat[:, 1:, :])
                g_loss = g_loss_u + 100 * g_loss_s
                
                self.g_opt.zero_grad()
                self.s_opt.zero_grad()
                g_loss.backward()
                self.g_opt.step()
                self.s_opt.step()
                
                # Train Discriminator
                z = self._random_generator(batch_size_curr)
                h = self.embedder(x)
                h_hat = self.generator(z)
                h_hat_supervise = self.supervisor(h_hat)
                
                y_real = self.discriminator(h)
                y_fake = self.discriminator(h_hat_supervise.detach())
                
                d_loss_real = self.bce_loss(y_real, torch.ones_like(y_real))
                d_loss_fake = self.bce_loss(y_fake, torch.zeros_like(y_fake))
                d_loss = d_loss_real + d_loss_fake
                
                self.d_opt.zero_grad()
                d_loss.backward()
                self.d_opt.step()
        
        if verbose:
            print("   [TimeGAN] Training complete.")
    
    def generate(self, n_samples):
        """
        Generate synthetic time series.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            generated: (n_samples, seq_len, n_features) numpy array
        """
        self.generator.eval()
        self.supervisor.eval()
        self.recovery.eval()
        
        with torch.no_grad():
            z = self._random_generator(n_samples)
            h_hat = self.generator(z)
            h_hat_supervise = self.supervisor(h_hat)
            x_hat = self.recovery(h_hat_supervise)
            
            # Denormalize
            x_hat = x_hat.cpu().numpy()
            x_hat = x_hat * (self.data_max - self.data_min + 1e-8) + self.data_min
        
        return x_hat.squeeze()
