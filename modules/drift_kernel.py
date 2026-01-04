"""
Kernel Regression Drift Estimator Module

Implements Numba-optimized Nadaraya-Watson kernel regression
for non-parametric drift estimation.

Author: Manus AI
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
from numba import njit, prange
import warnings

# Note: DriftEstimator base class is defined locally to avoid circular imports
from abc import ABC, abstractmethod

class DriftEstimator(ABC):
    """Abstract base class for drift estimators."""
    
    ESTIMATOR_TYPE: str = "base"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, data: np.ndarray, time_grid: np.ndarray) -> 'DriftEstimator':
        pass
    
    @abstractmethod
    def predict(self, t: Union[float, np.ndarray], x: np.ndarray) -> np.ndarray:
        pass


# ============================================
# Numba-Accelerated Kernel Functions
# ============================================

@njit(cache=True)
def _gaussian_kernel(u: float) -> float:
    """Standard Gaussian kernel."""
    return np.exp(-0.5 * u * u) / np.sqrt(2 * np.pi)


@njit(cache=True)
def _epanechnikov_kernel(u: float) -> float:
    """Epanechnikov kernel (optimal for MSE)."""
    if np.abs(u) <= 1:
        return 0.75 * (1 - u * u)
    return 0.0


@njit(cache=True, parallel=True)
def _nadaraya_watson_predict_numba(
    query_t: np.ndarray,
    query_x: np.ndarray,
    data_t: np.ndarray,
    data_x: np.ndarray,
    data_y: np.ndarray,
    h_t: float,
    h_x: float,
    min_weight: float = 1e-10
) -> np.ndarray:
    """
    Nadaraya-Watson kernel regression prediction.
    
    Fully Numba-optimized with parallel processing.
    
    Args:
        query_t: Query time points (n_queries,)
        query_x: Query state values (n_queries, n_features)
        data_t: Training time points (n_data,)
        data_x: Training state values (n_data, n_features)
        data_y: Training target values (n_data, n_features)
        h_t: Temporal bandwidth
        h_x: Spatial bandwidth
        min_weight: Minimum weight threshold
    
    Returns:
        Predictions (n_queries, n_features)
    """
    n_queries = len(query_t)
    n_data = len(data_t)
    n_features = query_x.shape[1]
    
    predictions = np.zeros((n_queries, n_features))
    
    for i in prange(n_queries):
        for k in range(n_features):
            numerator = 0.0
            denominator = 0.0
            
            for j in range(n_data):
                # Temporal kernel
                u_t = (query_t[i] - data_t[j]) / h_t
                w_t = _gaussian_kernel(u_t)
                
                # Spatial kernel
                u_x = (query_x[i, k] - data_x[j, k]) / h_x
                w_x = _gaussian_kernel(u_x)
                
                # Combined weight
                w = w_t * w_x
                
                numerator += w * data_y[j, k]
                denominator += w
            
            if denominator > min_weight:
                predictions[i, k] = numerator / denominator
            else:
                # Fallback to global mean
                predictions[i, k] = np.mean(data_y[:, k])
    
    return predictions


@njit(cache=True, parallel=True)
def _cross_validation_bandwidth_numba(
    data_t: np.ndarray,
    data_x: np.ndarray,
    data_y: np.ndarray,
    h_t_candidates: np.ndarray,
    h_x_candidates: np.ndarray,
    n_folds: int = 5
) -> Tuple[float, float, float]:
    """
    Cross-validation for bandwidth selection.
    
    Numba-optimized leave-one-out cross-validation.
    
    Args:
        data_t: Time points (n_data,)
        data_x: State values (n_data, n_features)
        data_y: Target values (n_data, n_features)
        h_t_candidates: Temporal bandwidth candidates
        h_x_candidates: Spatial bandwidth candidates
        n_folds: Number of CV folds
    
    Returns:
        Tuple of (best_h_t, best_h_x, best_cv_score)
    """
    n_data = len(data_t)
    n_features = data_x.shape[1]
    n_h_t = len(h_t_candidates)
    n_h_x = len(h_x_candidates)
    
    best_h_t = h_t_candidates[0]
    best_h_x = h_x_candidates[0]
    best_score = np.inf
    
    # Grid search
    for i_t in range(n_h_t):
        h_t = h_t_candidates[i_t]
        
        for i_x in range(n_h_x):
            h_x = h_x_candidates[i_x]
            
            # Leave-one-out CV
            total_error = 0.0
            
            for i in range(n_data):
                # Predict point i using all other points
                numerator = np.zeros(n_features)
                denominator = 0.0
                
                for j in range(n_data):
                    if j == i:
                        continue
                    
                    u_t = (data_t[i] - data_t[j]) / h_t
                    w_t = _gaussian_kernel(u_t)
                    
                    for k in range(n_features):
                        u_x = (data_x[i, k] - data_x[j, k]) / h_x
                        w_x = _gaussian_kernel(u_x)
                        w = w_t * w_x
                        
                        numerator[k] += w * data_y[j, k]
                    
                    denominator += w_t
                
                # Compute prediction error
                for k in range(n_features):
                    if denominator > 1e-10:
                        pred = numerator[k] / denominator
                    else:
                        pred = 0.0
                    
                    total_error += (pred - data_y[i, k]) ** 2
            
            cv_score = total_error / (n_data * n_features)
            
            if cv_score < best_score:
                best_score = cv_score
                best_h_t = h_t
                best_h_x = h_x
    
    return best_h_t, best_h_x, best_score


@njit(cache=True, parallel=True)
def _local_linear_predict_numba(
    query_t: np.ndarray,
    query_x: np.ndarray,
    data_t: np.ndarray,
    data_x: np.ndarray,
    data_y: np.ndarray,
    h_t: float,
    h_x: float
) -> np.ndarray:
    """
    Local linear regression prediction (more accurate than NW).
    
    Args:
        query_t: Query time points
        query_x: Query state values
        data_t: Training time points
        data_x: Training state values
        data_y: Training target values
        h_t: Temporal bandwidth
        h_x: Spatial bandwidth
    
    Returns:
        Predictions
    """
    n_queries = len(query_t)
    n_data = len(data_t)
    n_features = query_x.shape[1]
    
    predictions = np.zeros((n_queries, n_features))
    
    for i in prange(n_queries):
        for k in range(n_features):
            # Weighted least squares
            sum_w = 0.0
            sum_wx = 0.0
            sum_wxx = 0.0
            sum_wy = 0.0
            sum_wxy = 0.0
            
            for j in range(n_data):
                u_t = (query_t[i] - data_t[j]) / h_t
                u_x = (query_x[i, k] - data_x[j, k]) / h_x
                
                w = _gaussian_kernel(u_t) * _gaussian_kernel(u_x)
                x_diff = data_x[j, k] - query_x[i, k]
                
                sum_w += w
                sum_wx += w * x_diff
                sum_wxx += w * x_diff * x_diff
                sum_wy += w * data_y[j, k]
                sum_wxy += w * x_diff * data_y[j, k]
            
            # Solve for intercept (prediction at query point)
            det = sum_w * sum_wxx - sum_wx * sum_wx
            
            if np.abs(det) > 1e-10:
                predictions[i, k] = (sum_wxx * sum_wy - sum_wx * sum_wxy) / det
            elif sum_w > 1e-10:
                predictions[i, k] = sum_wy / sum_w
            else:
                predictions[i, k] = np.mean(data_y[:, k])
    
    return predictions


# ============================================
# Kernel Drift Estimator Class
# ============================================

class KernelDriftEstimator(DriftEstimator):
    """
    Kernel Regression Drift Estimator.
    
    Uses Nadaraya-Watson or Local Linear regression with
    Numba acceleration for fast non-parametric drift estimation.
    
    Features:
        - Automatic bandwidth selection via cross-validation
        - Numba JIT compilation for performance
        - Support for both NW and local linear methods
    """
    
    ESTIMATOR_TYPE = "kernel"
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize kernel drift estimator.
        
        Args:
            config: Configuration with keys:
                - kernel_bandwidth: Base bandwidth (default: 0.1)
                - kernel_h_t: Temporal bandwidth (optional)
                - kernel_h_x: Spatial bandwidth (optional)
                - kernel_method: 'nw' or 'local_linear' (default: 'nw')
                - kernel_use_cv: Use cross-validation for bandwidth (default: True)
                - kernel_n_pi: Number of bandwidth candidates (default: 10)
        """
        super().__init__(config)
        
        self.base_bandwidth = config.get('kernel_bandwidth', 0.1)
        self.h_t = config.get('kernel_h_t', None)
        self.h_x = config.get('kernel_h_x', None)
        self.method = config.get('kernel_method', 'nw')
        self.use_cv = config.get('kernel_use_cv', True)
        self.n_candidates = config.get('kernel_n_pi', 10)
        
        # Training data (stored for prediction)
        self.data_t = None
        self.data_x = None
        self.data_y = None
        self.n_features = None
        self.time_grid = None
    
    def fit(
        self,
        data: np.ndarray,
        time_grid: np.ndarray,
        verbose: bool = True
    ) -> 'KernelDriftEstimator':
        """
        Fit kernel drift estimator.
        
        Args:
            data: Time series data (n_samples, seq_len, n_features)
            time_grid: Time points (seq_len,)
            verbose: Whether to print progress
        
        Returns:
            self for method chaining
        """
        # Ensure 3D
        if data.ndim == 2:
            data = data[:, :, np.newaxis]
        
        n_samples, seq_len, n_features = data.shape
        self.n_features = n_features
        self.time_grid = time_grid
        
        # Compute returns (drift targets)
        returns = np.diff(data, axis=1)
        dt = time_grid[1] - time_grid[0] if len(time_grid) > 1 else 1.0
        drift_targets = returns / dt
        
        # Flatten to create training data
        # For each return, use the state at the beginning of the interval
        data_t_list = []
        data_x_list = []
        data_y_list = []
        
        for i in range(n_samples):
            for j in range(seq_len - 1):
                data_t_list.append(time_grid[j])
                data_x_list.append(data[i, j, :])
                data_y_list.append(drift_targets[i, j, :])
        
        self.data_t = np.array(data_t_list, dtype=np.float64)
        self.data_x = np.array(data_x_list, dtype=np.float64)
        self.data_y = np.array(data_y_list, dtype=np.float64)
        
        # Bandwidth selection
        if self.use_cv and len(self.data_t) > 50:
            if verbose:
                print("  [Kernel] Running cross-validation for bandwidth selection...")
            
            # Create bandwidth candidates
            x_std = np.std(self.data_x) + 1e-8
            t_range = time_grid[-1] - time_grid[0] + 1e-8
            
            h_t_candidates = np.linspace(
                0.05 * t_range,
                0.5 * t_range,
                self.n_candidates
            ).astype(np.float64)
            
            h_x_candidates = np.linspace(
                0.05 * x_std,
                0.5 * x_std,
                self.n_candidates
            ).astype(np.float64)
            
            # Subsample for CV if too much data
            if len(self.data_t) > 1000:
                idx = np.random.choice(len(self.data_t), 1000, replace=False)
                cv_data_t = self.data_t[idx]
                cv_data_x = self.data_x[idx]
                cv_data_y = self.data_y[idx]
            else:
                cv_data_t = self.data_t
                cv_data_x = self.data_x
                cv_data_y = self.data_y
            
            self.h_t, self.h_x, cv_score = _cross_validation_bandwidth_numba(
                cv_data_t,
                cv_data_x,
                cv_data_y,
                h_t_candidates,
                h_x_candidates
            )
            
            if verbose:
                print(f"  [Kernel] Selected bandwidths: h_t={self.h_t:.4f}, h_x={self.h_x:.4f}")
        else:
            # Use default bandwidths
            if self.h_t is None:
                t_range = time_grid[-1] - time_grid[0] + 1e-8
                self.h_t = self.base_bandwidth * t_range
            
            if self.h_x is None:
                x_std = np.std(self.data_x) + 1e-8
                self.h_x = self.base_bandwidth * x_std
        
        self.is_fitted = True
        return self
    
    def predict(
        self,
        t: Union[float, np.ndarray],
        x: np.ndarray
    ) -> np.ndarray:
        """
        Predict drift at given time and state.
        
        Args:
            t: Time point(s)
            x: State(s) of shape (n_samples, n_features) or (n_features,)
        
        Returns:
            Predicted drift of same shape as x
        """
        if not self.is_fitted:
            return np.zeros_like(x)
        
        # Handle scalar t
        if np.isscalar(t):
            t = np.array([t])
        else:
            t = np.asarray(t)
        
        # Ensure 2D x
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[np.newaxis, :]
            squeeze = True
        else:
            squeeze = False
        
        n_queries = x.shape[0]
        
        # Broadcast t to match x
        if len(t) == 1:
            query_t = np.full(n_queries, t[0], dtype=np.float64)
        else:
            query_t = t.astype(np.float64)
        
        query_x = x.astype(np.float64)
        
        # Predict using appropriate method
        if self.method == 'local_linear':
            predictions = _local_linear_predict_numba(
                query_t,
                query_x,
                self.data_t,
                self.data_x,
                self.data_y,
                self.h_t,
                self.h_x
            )
        else:
            predictions = _nadaraya_watson_predict_numba(
                query_t,
                query_x,
                self.data_t,
                self.data_x,
                self.data_y,
                self.h_t,
                self.h_x
            )
        
        if squeeze:
            predictions = predictions[0]
        
        return predictions
    
    def get_bandwidths(self) -> Tuple[float, float]:
        """Get selected bandwidths."""
        return self.h_t, self.h_x


# ============================================
# Factory Function
# ============================================

def get_kernel_drift_estimator(config: Dict[str, Any]) -> KernelDriftEstimator:
    """
    Factory function to create kernel drift estimator.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        KernelDriftEstimator instance
    """
    return KernelDriftEstimator(config)
