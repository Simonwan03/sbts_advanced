import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# Helper: StandardScaler
# ==========================================
class StandardScaler:
    def __init__(self):
        self.mean = 0
        self.std = 1
        self.device = torch.device("cpu")

    def fit(self, x):
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)

    def transform(self, x):
        return (x - self.mean) / (self.std + 1e-8)

    def inverse_transform(self, x):
        return x * self.std + self.mean
    
    def transform_tensor(self, x_tensor):
        # Assumes self.mean/std are numpy, convert to tensor on fly or store as buffer
        device = x_tensor.device
        mean_t = torch.tensor(self.mean, device=device, dtype=x_tensor.dtype)
        std_t = torch.tensor(self.std, device=device, dtype=x_tensor.dtype)
        return (x_tensor - mean_t) / (std_t + 1e-8)

    def inverse_transform_tensor(self, x_tensor):
        device = x_tensor.device
        mean_t = torch.tensor(self.mean, device=device, dtype=x_tensor.dtype)
        std_t = torch.tensor(self.std, device=device, dtype=x_tensor.dtype)
        return x_tensor * std_t + mean_t

# ==========================================
# 1. Kernel Drift Estimator
# ==========================================
class KernelDriftEstimator:
    def __init__(self, bandwidth=0.1):
        self.h = bandwidth
        self.X_train = None
        self.Y_train = None 

    def fit(self, trajectories, dt):
        X_t = trajectories[:, :-1, :].reshape(-1, trajectories.shape[-1])
        X_next = trajectories[:, 1:, :].reshape(-1, trajectories.shape[-1])
        dX_dt = (X_next - X_t) / dt
        
        self.X_train = X_t
        self.Y_train = dX_dt
        return self

    def predict(self, t, x):
        if x.ndim == 1: x = x[np.newaxis, :]
        dists = np.sum((self.X_train - x)**2, axis=1)
        weights = np.exp(-0.5 * dists / (self.h ** 2))
        
        if np.sum(weights) < 1e-10:
            return np.zeros(self.Y_train.shape[1])
            
        weights = weights / np.sum(weights)
        drift = np.dot(weights, self.Y_train)
        return drift.flatten()

# ==========================================
# 2. LSTM Drift Estimator
# ==========================================
class DriftLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1, dropout_prob=0.0):
        super(DriftLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        last_out = self.dropout(last_out)
        return self.fc(last_out)

class LSTMDriftEstimator:
    def __init__(self, input_dim=1, hidden_size=32, lr=0.005, epochs=50, dt=0.01, weight_decay=1e-4, dropout=0.0):
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.lr = lr
        self.epochs = epochs
        self.dt = dt
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, trajectories, dt):
        self.dt = dt
        N, T, D = trajectories.shape
        
        data_flat = trajectories.reshape(-1, D)
        self.scaler.fit(data_flat)
        data_scaled = self.scaler.transform(trajectories)
        
        X_train = data_scaled[:, :-1, :]
        Y_train = data_scaled[:, 1:, :]
        
        X_train = X_train.reshape(-1, 1, D)
        Y_train = Y_train.reshape(-1, D)
        
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        Y_tensor = torch.FloatTensor(Y_train).to(self.device)
        
        dataset = TensorDataset(X_tensor, Y_tensor)
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
        
        self.model = DriftLSTMModel(D, self.hidden_size, D, dropout_prob=self.dropout).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()
        
        self.model.train()
        print(f"   [LSTM] Training start (wd={self.weight_decay}, drop={self.dropout})...")
        for epoch in range(self.epochs):
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
        self.model.eval()
        return self

    def predict(self, t, x):
        if x.ndim == 1: x = x[np.newaxis, :]
        x_scaled = self.scaler.transform(x)
        x_tensor = torch.FloatTensor(x_scaled).unsqueeze(1).to(self.device)
        with torch.no_grad():
            x_next_scaled = self.model(x_tensor).cpu().numpy()
        x_next_real = self.scaler.inverse_transform(x_next_scaled)
        drift = (x_next_real - x) / self.dt
        if x.ndim == 1: return drift.flatten()
        return drift

# ==============================================================================
# 3. Transformer Drift Estimator (Assets as Tokens / Attention Based)
# ==============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, t):
        if t.dim() == 2: t = t.squeeze(-1)
        sinusoid_inp = torch.ger(t, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb

class ImprovedTransformerDriftModel(nn.Module):
    """
    Architecture: "Assets as Tokens".
    Input: (Batch, N_Assets).
    Interpreted as Sequence: (Batch, Seq_Len=N_Assets, Feature=1).
    Attention mechanism learns correlation structure between assets.
    """
    def __init__(self, num_assets, d_model=32, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.3):
        super().__init__()
        self.num_assets = num_assets
        
        # 1. Project scalar asset value to d_model
        self.input_proj = nn.Linear(1, d_model)
        
        # 2. Learnable Asset Identity Embedding (to distinguish SPY from QQQ)
        self.asset_embed = nn.Parameter(torch.randn(1, num_assets, d_model))
        
        # 3. Time Embedding
        self.time_embed = SinusoidalTimeEmbedding(d_model)
        
        # 4. Transformer Encoder (Pre-LN for stability)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True,
            norm_first=True # Critical for convergence on financial data
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 5. Output Projection
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x, t):
        """
        x: (Batch, N_Assets)
        t: (Batch,)
        """
        # Treat assets as a sequence: (Batch, N_Assets, 1)
        x_seq = x.unsqueeze(-1)
        
        # Project: (Batch, N_Assets, d_model)
        x_emb = self.input_proj(x_seq)
        
        # Add Asset Identity (Broadcasting)
        x_emb = x_emb + self.asset_embed
        
        # Add Time Embedding (Broadcasting)
        # t_emb: (Batch, d_model) -> (Batch, 1, d_model)
        t_emb = self.time_embed(t).unsqueeze(1)
        x_emb = x_emb + t_emb
        
        # Transformer (Self-Attention across assets)
        # Output: (Batch, N_Assets, d_model)
        encoded = self.transformer_encoder(x_emb)
        
        # Project back to scalar drift: (Batch, N_Assets, 1)
        out = self.output_proj(encoded)
        
        # Remove feature dim: (Batch, N_Assets)
        return out.squeeze(-1)

class TransformerDriftEstimator:
    """
    Wraps the ImprovedTransformerDriftModel.
    """
    def __init__(self, input_dim=1, d_model=32, nhead=4, lr=0.001, epochs=50, dt=0.01, weight_decay=1e-3, dropout=0.3):
        self.num_assets = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.lr = lr
        self.epochs = epochs
        self.dt = dt
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, trajectories, dt):
        self.dt = dt
        N, T, D = trajectories.shape
        
        # Data Prep
        data_flat = trajectories.reshape(-1, D)
        self.scaler.fit(data_flat)
        data_scaled = self.scaler.transform(trajectories)
        
        # Input: X_t (N, T-1, D) -> Flatten to (Batch, D)
        X_train = data_scaled[:, :-1, :]
        # Target: X_{t+1} (N, T-1, D) -> Flatten to (Batch, D)
        Y_train = data_scaled[:, 1:, :]
        
        # Time steps
        time_steps = torch.arange(0, T-1).float() * dt
        T_train = time_steps.repeat(N, 1)
        
        # Flatten for Dataloader
        X_train_flat = X_train.reshape(-1, D)
        Y_train_flat = Y_train.reshape(-1, D)
        T_train_flat = T_train.reshape(-1)
        
        X_tensor = torch.FloatTensor(X_train_flat).to(self.device)
        Y_tensor = torch.FloatTensor(Y_train_flat).to(self.device)
        T_tensor = torch.FloatTensor(T_train_flat).to(self.device)
        
        dataset = TensorDataset(X_tensor, T_tensor, Y_tensor)
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
        
        # Init Model
        self.model = ImprovedTransformerDriftModel(
            num_assets=self.num_assets,
            d_model=self.d_model,
            nhead=self.nhead,
            dropout=self.dropout
        ).to(self.device)
        
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        # Using Huber Loss for robustness against outliers
        criterion = nn.HuberLoss(delta=1.0)
        
        self.model.train()
        print(f"   [Transformer] Training 'Assets-as-Tokens' (d_model={self.d_model}, wd={self.weight_decay})...")
        
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_x, batch_t, batch_y in dataloader:
                optimizer.zero_grad()
                pred = self.model(batch_x, batch_t)
                # Target for drift network: predict next state directly (Markovian)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
        self.model.eval()
        return self

    def predict(self, t, x):
        if x.ndim == 1: x = x[np.newaxis, :]
        
        x_scaled = self.scaler.transform(x)
        x_tensor = torch.FloatTensor(x_scaled).to(self.device) # (Batch, D)
        t_val = torch.full((x.shape[0],), float(t)).to(self.device)
        
        with torch.no_grad():
            x_next_scaled = self.model(x_tensor, t_val).cpu().numpy()
            
        x_next_real = self.scaler.inverse_transform(x_next_scaled)
        drift = (x_next_real - x) / self.dt
        
        if x.ndim == 1: return drift.flatten()
        return drift