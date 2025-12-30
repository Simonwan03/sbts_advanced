import numpy as np
from scipy.stats import entropy, wasserstein_distance
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# ==============================================================================
# 1. Existing Metrics (Preserved & Enhanced)
# ==============================================================================

def wasserstein_distance_1d(u_values, v_values):
    """
    Computes the Wasserstein distance (Earth Mover's Distance) between two 1D distributions.
    
    Args:
        u_values: Samples from distribution U.
        v_values: Samples from distribution V.
    """
    # Flattens input to ensure 1D calculation
    return wasserstein_distance(np.ravel(u_values), np.ravel(v_values))

def compute_tail_risk_metrics(data, alpha=0.95):
    """
    Returns Value-at-Risk (VaR) and Expected Shortfall (ES) at confidence level alpha.
    Vectorized to handle shape (N_samples, ...) along axis 0.
    
    Args:
        data: Array of shape (N_samples, ...)
        alpha: Confidence level (e.g., 0.95 or 0.99)
        
    Returns:
        var: Value at Risk
        es: Expected Shortfall (Conditional VaR)
    """
    # Sort data along the sample dimension (axis 0)
    sorted_data = np.sort(data, axis=0)
    n_samples = len(data)
    
    # Index for VaR (the cutoff point)
    # For alpha=0.95, we look at the worst 5%
    index = int((1 - alpha) * n_samples)
    
    # Ensure index is within bounds
    index = max(0, min(index, n_samples - 1))
    
    # VaR is the value at the cutoff index (negated for loss convention usually, 
    # here we assume returns and return the value itself or negative based on user convention.
    # The original snippet returned negative sorted value, implying Loss distribution or 
    # assuming negative returns are bad. We keep original logic.)
    var = -sorted_data[index]
    
    # ES is the mean of values worse than VaR
    if index == 0:
        es = -sorted_data[0] # Edge case
    else:
        es = -np.mean(sorted_data[:index], axis=0)
        
    return var, es

# ==============================================================================
# 2. Multi-Asset Correlation Metric (New)
# ==============================================================================

def compute_cross_correlation_metric(real_data, synthetic_data):
    """
    Measures the Frobenius norm difference between the correlation matrices 
    of real and synthetic data. Essential for Multi-Asset validation.
    
    Args:
        real_data: (N, T, D) array
        synthetic_data: (N, T, D) array
        
    Returns:
        float: The Frobenius norm of (Corr_real - Corr_synth)
    """
    N, T, D = real_data.shape
    
    # Reshape to (N*T, D) to compute correlation across all observations
    # assuming stationarity of correlation structure
    real_flat = real_data.reshape(-1, D)
    synth_flat = synthetic_data.reshape(-1, D)
    
    # Compute Correlation Matrices (D, D)
    # rowvar=False means columns are variables (assets)
    corr_real = np.corrcoef(real_flat, rowvar=False)
    corr_synth = np.corrcoef(synth_flat, rowvar=False)
    
    # Handle scalar case if D=1 (corrcoef returns scalar 1.0)
    if D == 1:
        return 0.0
        
    # Replace NaNs with 0 (in case of constant series)
    corr_real = np.nan_to_num(corr_real)
    corr_synth = np.nan_to_num(corr_synth)
    
    # Frobenius Norm difference
    diff = np.linalg.norm(corr_real - corr_synth, ord='fro')
    return diff

# ==============================================================================
# 3. Discriminative Score (PyTorch RNN Classifier)
# ==============================================================================

class DiscriminatorRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super(DiscriminatorRNN, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (Batch, Seq, Dim)
        # out shape: (Batch, Seq, Hidden)
        _, h_n = self.rnn(x) 
        # h_n shape: (1, Batch, Hidden) for GRU
        last_hidden = h_n[-1] 
        logits = self.fc(last_hidden)
        return self.sigmoid(logits)

class DiscriminativeScore:
    """
    "Turing Test" for Time Series.
    Trains a classifier to distinguish Real from Synthetic data.
    
    Score interpretation:
    - 0.5: Perfect generation (Classifier is guessing).
    - 1.0: Poor generation (Classifier easily tells apart).
    """
    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def train_and_evaluate(self, real_data, synthetic_data, epochs=10, batch_size=64, hidden_size=32):
        """
        Args:
            real_data: (N, T, D) numpy array
            synthetic_data: (N, T, D) numpy array
            
        Returns:
            accuracy (float): Test set accuracy.
        """
        # 1. Prepare Data
        # Label 1 for Real, 0 for Synthetic
        n_real = len(real_data)
        n_synth = len(synthetic_data)
        
        X = np.concatenate([real_data, synthetic_data], axis=0)
        y = np.concatenate([np.ones(n_real), np.zeros(n_synth)], axis=0)
        
        # Split Train/Test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)
        
        # Convert to Tensor
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        X_test_t = torch.FloatTensor(X_test).to(self.device)
        y_test_t = torch.FloatTensor(y_test).unsqueeze(1).to(self.device)
        
        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
        
        # 2. Initialize Model
        _, seq_len, input_dim = real_data.shape
        self.model = DiscriminatorRNN(input_dim, hidden_size).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        # 3. Train
        self.model.train()
        for epoch in range(epochs):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                
        # 4. Evaluate
        self.model.eval()
        with torch.no_grad():
            test_preds = self.model(X_test_t)
            # Threshold at 0.5
            predicted_classes = (test_preds > 0.5).float()
            correct = (predicted_classes == y_test_t).sum().item()
            accuracy = correct / len(y_test_t)
            
        return accuracy