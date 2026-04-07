"""
TimeGAN baseline model.

This file contains the active TimeGAN implementation used by the current
`main.py` experiment pipeline.
"""

from typing import Any, Dict, Optional

import numpy as np

from models.base import GenerativeModel as TimeSeriesGenerator

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class EmbeddingNetwork(nn.Module):
        """Embedding network for TimeGAN."""

        def __init__(self, input_dim: int, hidden_dim: int, n_layers: int = 2):
            super().__init__()
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
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
                batch_first=True,
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
                batch_first=True,
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
                batch_first=True,
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
                batch_first=True,
            )
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, h: torch.Tensor) -> torch.Tensor:
            out, _ = self.rnn(h)
            return self.fc(out)


class TimeGAN(TimeSeriesGenerator):
    """TimeGAN baseline used by the active experiment pipeline."""

    MODEL_TYPE = "timegan"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for TimeGAN")

        self.hidden_dim = config.get("timegan_hidden_dim", 64)
        self.z_dim = config.get("timegan_z_dim", 32)
        self.n_layers = config.get("timegan_n_layers", 2)
        if self.n_layers < 2:
            raise ValueError("timegan_n_layers must be >= 2")
        self.epochs = config.get("timegan_epochs", 50)
        self.lr = config.get("timegan_lr", 0.001)
        self.batch_size = config.get("timegan_batch_size", 128)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.embedder = None
        self.recovery = None
        self.generator = None
        self.supervisor = None
        self.discriminator = None

        self.n_features = None
        self.seq_len = None
        self.data_mean = None
        self.data_std = None

    def fit(
        self,
        data: np.ndarray,
        time_grid: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> "TimeGAN":
        """Fit TimeGAN on windowed time-series data."""
        del time_grid

        if data.ndim == 2:
            data = data[:, :, np.newaxis]

        n_samples, seq_len, n_features = data.shape
        self.n_features = n_features
        self.seq_len = seq_len

        self.data_mean = np.mean(data)
        self.data_std = np.std(data) + 1e-8
        data_norm = (data - self.data_mean) / self.data_std
        X = torch.tensor(data_norm, dtype=torch.float32).to(self.device)

        self.embedder = EmbeddingNetwork(n_features, self.hidden_dim, self.n_layers).to(self.device)
        self.recovery = RecoveryNetwork(self.hidden_dim, n_features, self.n_layers).to(self.device)
        self.generator = GeneratorNetwork(self.z_dim, self.hidden_dim, self.n_layers).to(self.device)
        self.supervisor = SupervisorNetwork(self.hidden_dim, self.n_layers).to(self.device)
        self.discriminator = DiscriminatorNetwork(self.hidden_dim, self.n_layers).to(self.device)

        if verbose:
            print("=" * 60)
            print("TimeGAN Training")
            print("=" * 60)

        ae_optimizer = torch.optim.Adam(
            list(self.embedder.parameters()) + list(self.recovery.parameters()),
            lr=self.lr,
        )

        for epoch in range(self.epochs):
            perm = torch.randperm(n_samples)
            X = X[perm]

            total_loss = 0.0
            n_batches = 0

            for i in range(0, n_samples, self.batch_size):
                batch = X[i : i + self.batch_size]
                ae_optimizer.zero_grad()
                h = self.embedder(batch)
                x_tilde = self.recovery(h)
                loss = F.mse_loss(x_tilde, batch)
                loss.backward()
                ae_optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            if verbose and (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch + 1}/{self.epochs}, AE Loss: {total_loss / n_batches:.6f}")

        if verbose:
            print("\n[Phase 2] Supervised Training...")

        sup_optimizer = torch.optim.Adam(self.supervisor.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            perm = torch.randperm(n_samples)
            X = X[perm]

            total_loss = 0.0
            n_batches = 0

            for i in range(0, n_samples, self.batch_size):
                batch = X[i : i + self.batch_size]
                sup_optimizer.zero_grad()

                with torch.no_grad():
                    h = self.embedder(batch)

                h_hat = self.supervisor(h)
                loss = F.mse_loss(h_hat[:, :-1, :], h[:, 1:, :])
                loss.backward()
                sup_optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            if verbose and (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch + 1}/{self.epochs}, Sup Loss: {total_loss / n_batches:.6f}")

        if verbose:
            print("\n[Phase 3] Joint Training...")

        g_optimizer = torch.optim.Adam(
            list(self.generator.parameters()) + list(self.supervisor.parameters()),
            lr=self.lr,
        )
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            perm = torch.randperm(n_samples)
            X = X[perm]

            g_loss_total = 0.0
            d_loss_total = 0.0
            n_batches = 0

            for i in range(0, n_samples, self.batch_size):
                batch = X[i : i + self.batch_size]
                batch_size = len(batch)

                z = torch.randn(batch_size, seq_len, self.z_dim, device=self.device)
                h_fake = self.generator(z)
                h_fake_sup = self.supervisor(h_fake)

                with torch.no_grad():
                    h_real = self.embedder(batch)

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
                print(
                    f"    Epoch {epoch + 1}/{self.epochs}, "
                    f"G Loss: {g_loss_total / n_batches:.4f}, "
                    f"D Loss: {d_loss_total / n_batches:.4f}"
                )

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
        x0: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Generate synthetic time series samples."""
        del x0, kwargs

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
