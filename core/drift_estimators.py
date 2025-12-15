import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# 工具类: StandardScaler
# ==========================================
class StandardScaler:
    def __init__(self):
        self.mean = 0
        self.std = 1
        self.device = torch.device("cpu")

    def fit(self, x):
        self.mean = np.mean(x)
        self.std = np.std(x)

    def transform(self, x):
        return (x - self.mean) / (self.std + 1e-8)

    def inverse_transform(self, x):
        return x * self.std + self.mean
    
    def transform_tensor(self, x_tensor):
        # Helper for PyTorch tensors
        return (x_tensor - self.mean) / (self.std + 1e-8)
        
    def inverse_transform_tensor(self, x_tensor):
        return x_tensor * self.std + self.mean

# ==========================================
# Kernel Drift Estimator (保留原有)
# ==========================================
class KernelDriftEstimator:
    def __init__(self, bandwidth=0.1):
        self.h = bandwidth
        self.X_train = None
        self.Y_train = None # (dX / dt)

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
# LSTM Drift Estimator (新增)
# ==========================================
class DriftLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        super(DriftLSTMModel, self).__init__()
        # batch_first=True: input shape (Batch, Seq, Feature)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x: (Batch, Seq, Feature)
        out, _ = self.lstm(x)
        # 取序列最后一个时间步的输出
        last_out = out[:, -1, :]
        return self.fc(last_out)

class LSTMDriftEstimator:
    """
    Learns the drift using an LSTM.
    Drift is approximated as (Model(x_t) - x_t) / dt
    Note: Training happens in SCALED space for stability.
    """
    def __init__(self, input_dim=1, hidden_size=32, lr=0.005, epochs=50, dt=0.01):
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.lr = lr
        self.epochs = epochs
        self.dt = dt
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, trajectories, dt):
        """
        trajectories: (N, T, D)
        """
        self.dt = dt
        N, T, D = trajectories.shape
        
        # 1. Prepare Data
        # Flatten trajectories to pairs (x_t, x_{t+1})
        data_flat = trajectories.reshape(-1, D)
        
        # Fit Scaler on all data
        self.scaler.fit(data_flat)
        data_scaled = self.scaler.transform(trajectories)
        
        # Prepare Inputs (X) and Targets (Y = Next Step X)
        # LSTM input shape: (Batch, Seq_Len=1, Dim)
        X_train = data_scaled[:, :-1, :]
        Y_train = data_scaled[:, 1:, :]
        
        # Flatten for DataLoader
        X_train = X_train.reshape(-1, 1, D) # Seq len 1
        Y_train = Y_train.reshape(-1, D)    # Target is just the next vector
        
        # To Tensor
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        Y_tensor = torch.FloatTensor(Y_train).to(self.device)
        
        # DataLoader
        dataset = TensorDataset(X_tensor, Y_tensor)
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
        
        # 2. Initialize Model
        self.model = DriftLSTMModel(D, self.hidden_size, D).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = nn.MSELoss()
        
        # 3. Training Loop
        self.model.train()
        print(f"Training LSTM Drift Estimator on {self.device}...")
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch+1) % 10 == 0:
                print(f"LSTM Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(dataloader):.6f}")
                
        self.model.eval()
        return self

    def predict(self, t, x):
        """
        Returns drift in REAL units.
        drift = (InverseScale(Model(Scale(x))) - x) / dt
        """
        # x shape: (D,) or (Batch, D)
        if x.ndim == 1:
            x_in = x[np.newaxis, :]
        else:
            x_in = x
            
        # 1. Scale Input
        x_scaled = self.scaler.transform(x_in)
        
        # 2. To Tensor (Batch, Seq=1, D)
        x_tensor = torch.FloatTensor(x_scaled).unsqueeze(1).to(self.device)
        
        # 3. Predict Next Step (Scaled)
        with torch.no_grad():
            x_next_scaled = self.model(x_tensor).cpu().numpy()
            
        # 4. Inverse Scale
        x_next_real = self.scaler.inverse_transform(x_next_scaled)
        
        # 5. Calculate Drift
        drift = (x_next_real - x_in) / self.dt
        
        if x.ndim == 1:
            return drift.flatten()
        return drift # Returns (Batch, D) logic handled by solver if vectorized