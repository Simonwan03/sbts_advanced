"""
Predictive Score for Time Series Generation Evaluation

Based on the implementation from alexouadi/SBTS:
https://github.com/alexouadi/SBTS

The predictive score measures how well the generated data preserves
the predictive relationships present in the original data.

A model is trained on generated data to predict a target feature,
then evaluated on real data. Lower MAE indicates better preservation
of temporal dynamics.

Reference:
- Yoon et al., "Time-series Generative Adversarial Networks", NeurIPS 2019
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class Predictor(nn.Module):
    """
    GRU-based predictor for time series forecasting.
    """
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.RNN = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.RNN(x)
        # Use last hidden state
        y_pred = self.fc(out[:, -1, :])
        return y_pred


def predictive_score_metrics(ori_data, generated_data, col_pred=None, 
                              iterations=2000, device=torch.device('cpu')):
    """
    Compute the predictive score.
    
    The predictive score measures how well a model trained on generated
    data can predict a target feature in real data.
    
    Process:
    1. Train a predictor on generated data to predict col_pred from other features
    2. Evaluate on real data
    3. Return MAE as the predictive score
    
    Lower score indicates better preservation of predictive relationships.
    
    Args:
        ori_data: Original data (tensor or numpy array)
        generated_data: Generated data (tensor or numpy array)
        col_pred: Column index to predict (default: last column)
        iterations: Number of training iterations
        device: Torch device
        
    Returns:
        predictive_score: MAE on real data
    """
    # Convert to tensors if needed
    if isinstance(ori_data, np.ndarray):
        ori_data = torch.tensor(ori_data, dtype=torch.float32)
    if isinstance(generated_data, np.ndarray):
        generated_data = torch.tensor(generated_data, dtype=torch.float32)
    
    ori_data = ori_data.to(device)
    generated_data = generated_data.to(device)
    
    # Ensure 3D shape
    if ori_data.dim() == 2:
        ori_data = ori_data.unsqueeze(-1)
    if generated_data.dim() == 2:
        generated_data = generated_data.unsqueeze(-1)
    
    # Basic parameters
    no, seq_len, dim = ori_data.shape
    
    if col_pred is None:
        col_pred = dim - 1
    
    # Prepare training data (from generated)
    # X: all features except target at time t
    # Y: target feature at time t+1
    X_gen = generated_data[:, :-1, :]  # (N, T-1, D)
    Y_gen = generated_data[:, 1:, col_pred:col_pred+1]  # (N, T-1, 1)
    
    # Flatten for training
    X_gen_flat = X_gen.reshape(-1, X_gen.shape[-1])
    Y_gen_flat = Y_gen.reshape(-1, 1)
    
    # Prepare test data (from original)
    X_ori = ori_data[:, :-1, :]
    Y_ori = ori_data[:, 1:, col_pred:col_pred+1]
    
    X_ori_flat = X_ori.reshape(-1, X_ori.shape[-1])
    Y_ori_flat = Y_ori.reshape(-1, 1)
    
    # Build predictor
    hidden_dim = max(int(dim / 2), 4)
    num_layers = 2
    batch_size = min(128, len(X_gen_flat) // 2)
    
    predictor = Predictor(dim, hidden_dim, num_layers).to(device)
    optimizer = optim.Adam(predictor.parameters(), lr=0.001)
    criterion = nn.L1Loss()  # MAE
    
    # Create dataset for sequence prediction
    # We need to reshape back to sequences for GRU
    dataset = TensorDataset(X_gen, Y_gen[:, -1, :])  # Predict last step
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training
    predictor.train()
    for itt in range(iterations):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            y_pred = predictor(batch_x)
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()
    
    # Evaluation on original data
    predictor.eval()
    with torch.no_grad():
        y_pred_ori = predictor(X_ori)
        y_true_ori = Y_ori[:, -1, :]
        
        mae = nn.L1Loss()(y_pred_ori, y_true_ori).item()
    
    return mae
