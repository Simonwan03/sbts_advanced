"""
Volatility Calibration Module

Implements local volatility surface calibration using kernel density estimation.
Supports both standard and Numba-accelerated versions.

Author: Manus AI
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
from numba import njit, prange
import warnings

# Note: VolatilityCalibrator base class is defined locally to avoid circular imports
from abc import ABC, abstractmethod

class VolatilityCalibrator(ABC):
    """Abstract base class for volatility calibration."""
    
    CALIBRATOR_TYPE: str = "base"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, data: np.ndarray, time_grid: np.ndarray) -> 'VolatilityCalibrator':
        pass
    
    @abstractmethod
    def __call__(self, t: Union[float, np.ndarray], x: np.ndarray) -> np.ndarray:
        pass


# ============================================
# Numba-Accelerated Kernel Functions
# ============================================

@njit(cache=True)
def _gaussian_kernel(u: float, bandwidth: float) -> float:
    """Gaussian kernel function."""
    return np.exp(-0.5 * (u / bandwidth) ** 2) / (bandwidth * np.sqrt(2 * np.pi))


@njit(cache=True, parallel=True)
def _compute_local_volatility_numba(
    t_grid: np.ndarray,
    x_grid: np.ndarray,
    data_t: np.ndarray,
    data_x: np.ndarray,
    data_returns: np.ndarray,
    h_t: float,
    h_x: float,
    min_vol: float = 1e-6
) -> np.ndarray:
    """
    Compute local volatility surface using Nadaraya-Watson kernel regression.
    
    Numba-accelerated with parallel processing.
    
    Args:
        t_grid: Time grid points (n_t,)
        x_grid: State grid points (n_x,)
        data_t: Time points of observations (n_obs,)
        data_x: State values at observations (n_obs,)
        data_returns: Squared returns at observations (n_obs,)
        h_t: Temporal bandwidth
        h_x: Spatial bandwidth
        min_vol: Minimum volatility floor
    
    Returns:
        Local volatility surface of shape (n_t, n_x)
    """
    n_t = len(t_grid)
    n_x = len(x_grid)
    n_obs = len(data_t)
    
    vol_surface = np.zeros((n_t, n_x))
    
    for i in prange(n_t):
        for j in range(n_x):
            t = t_grid[i]
            x = x_grid[j]
            
            numerator = 0.0
            denominator = 0.0
            
            for k in range(n_obs):
                # Kernel weights
                w_t = _gaussian_kernel(t - data_t[k], h_t)
                w_x = _gaussian_kernel(x - data_x[k], h_x)
                w = w_t * w_x
                
                numerator += w * data_returns[k]
                denominator += w
            
            if denominator > 1e-10:
                vol_surface[i, j] = np.sqrt(max(numerator / denominator, min_vol))
            else:
                vol_surface[i, j] = np.sqrt(min_vol)
    
    return vol_surface


@njit(cache=True)
def _interpolate_vol_numba(
    t: float,
    x: float,
    t_grid: np.ndarray,
    x_grid: np.ndarray,
    vol_surface: np.ndarray
) -> float:
    """
    Bilinear interpolation of volatility surface.
    
    Args:
        t: Query time
        x: Query state
        t_grid: Time grid
        x_grid: State grid
        vol_surface: Volatility surface
    
    Returns:
        Interpolated volatility
    """
    n_t = len(t_grid)
    n_x = len(x_grid)
    
    # Find indices
    i = np.searchsorted(t_grid, t)
    j = np.searchsorted(x_grid, x)
    
    # Clamp to valid range
    i = max(1, min(i, n_t - 1))
    j = max(1, min(j, n_x - 1))
    
    # Bilinear interpolation
    t0, t1 = t_grid[i-1], t_grid[i]
    x0, x1 = x_grid[j-1], x_grid[j]
    
    if t1 == t0:
        wt = 0.5
    else:
        wt = (t - t0) / (t1 - t0)
    
    if x1 == x0:
        wx = 0.5
    else:
        wx = (x - x0) / (x1 - x0)
    
    # Clamp weights
    wt = max(0.0, min(1.0, wt))
    wx = max(0.0, min(1.0, wx))
    
    v00 = vol_surface[i-1, j-1]
    v01 = vol_surface[i-1, j]
    v10 = vol_surface[i, j-1]
    v11 = vol_surface[i, j]
    
    vol = (1-wt) * (1-wx) * v00 + (1-wt) * wx * v01 + wt * (1-wx) * v10 + wt * wx * v11
    
    return vol


@njit(cache=True, parallel=True)
def _interpolate_vol_batch_numba(
    t_batch: np.ndarray,
    x_batch: np.ndarray,
    t_grid: np.ndarray,
    x_grid: np.ndarray,
    vol_surface: np.ndarray
) -> np.ndarray:
    """
    Batch interpolation of volatility surface.
    
    Args:
        t_batch: Query times (n,)
        x_batch: Query states (n, d) or (n,)
        t_grid: Time grid
        x_grid: State grid
        vol_surface: Volatility surface (n_t, n_x) or (n_t, n_x, d)
    
    Returns:
        Interpolated volatilities
    """
    n = len(t_batch)
    
    if x_batch.ndim == 1:
        result = np.zeros(n)
        for i in prange(n):
            result[i] = _interpolate_vol_numba(
                t_batch[i], x_batch[i], t_grid, x_grid, vol_surface
            )
    else:
        d = x_batch.shape[1]
        result = np.zeros((n, d))
        for i in prange(n):
            for k in range(d):
                result[i, k] = _interpolate_vol_numba(
                    t_batch[i], x_batch[i, k], t_grid, x_grid, vol_surface[:, :, k]
                )
    
    return result


# ============================================
# Local Volatility Calibrator Class
# ============================================

class LocalVolatilityCalibrator(VolatilityCalibrator):
    """
    Local Volatility Surface Calibrator using Kernel Density Estimation.
    
    Calibrates σ(t, x) from historical data using Nadaraya-Watson regression.
    Uses Numba acceleration for performance.
    """
    
    CALIBRATOR_TYPE = "local_volatility"
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize calibrator.
        
        Args:
            config: Configuration with keys:
                - vol_bandwidth: Base bandwidth (default: 0.5)
                - vol_t_bandwidth: Temporal bandwidth (optional)
                - vol_x_bandwidth: Spatial bandwidth (optional)
                - vol_n_t_grid: Number of time grid points (default: 50)
                - vol_n_x_grid: Number of state grid points (default: 100)
                - vol_min: Minimum volatility floor (default: 1e-6)
        """
        super().__init__(config)
        
        self.bandwidth = config.get('vol_bandwidth', 0.5)
        self.h_t = config.get('vol_t_bandwidth', self.bandwidth)
        self.h_x = config.get('vol_x_bandwidth', self.bandwidth)
        self.n_t_grid = config.get('vol_n_t_grid', 50)
        self.n_x_grid = config.get('vol_n_x_grid', 100)
        self.min_vol = config.get('vol_min', 1e-6)
        
        # To be fitted
        self.t_grid = None
        self.x_grid = None
        self.vol_surface = None  # Shape: (n_t, n_x) or (n_t, n_x, n_features)
        self.n_features = None
    
    def fit(
        self,
        data: np.ndarray,
        time_grid: np.ndarray,
        purified: bool = False
    ) -> 'LocalVolatilityCalibrator':
        """
        Fit local volatility surface to data.
        
        Args:
            data: Time series data of shape (n_samples, seq_len, n_features)
                  or (n_samples, seq_len) for univariate
            time_grid: Time points of shape (seq_len,)
            purified: Whether data has been purified (jumps removed)
        
        Returns:
            self for method chaining
        """
        # Ensure 3D
        if data.ndim == 2:
            data = data[:, :, np.newaxis]
        
        n_samples, seq_len, n_features = data.shape
        self.n_features = n_features
        
        # Compute returns
        returns = np.diff(data, axis=1)  # (n_samples, seq_len-1, n_features)
        
        # Create grids
        self.t_grid = np.linspace(time_grid[0], time_grid[-1], self.n_t_grid)
        
        # State grid based on data range
        x_min = np.percentile(data, 1)
        x_max = np.percentile(data, 99)
        x_range = x_max - x_min
        self.x_grid = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, self.n_x_grid)
        
        # Calibrate for each feature
        if n_features == 1:
            self.vol_surface = self._fit_single_feature(
                data[:, :, 0], returns[:, :, 0], time_grid
            )
        else:
            self.vol_surface = np.zeros((self.n_t_grid, self.n_x_grid, n_features))
            for k in range(n_features):
                self.vol_surface[:, :, k] = self._fit_single_feature(
                    data[:, :, k], returns[:, :, k], time_grid
                )
        
        self.is_fitted = True
        return self
    
    def _fit_single_feature(
        self,
        data: np.ndarray,
        returns: np.ndarray,
        time_grid: np.ndarray
    ) -> np.ndarray:
        """
        Fit volatility surface for a single feature.
        
        Args:
            data: Data of shape (n_samples, seq_len)
            returns: Returns of shape (n_samples, seq_len-1)
            time_grid: Time points
        
        Returns:
            Volatility surface of shape (n_t_grid, n_x_grid)
        """
        n_samples, seq_len = data.shape
        
        # Flatten observations
        # For each return, use the state at the beginning of the interval
        data_t = []
        data_x = []
        data_r2 = []
        
        dt = time_grid[1] - time_grid[0] if len(time_grid) > 1 else 1.0
        
        for i in range(n_samples):
            for j in range(seq_len - 1):
                data_t.append(time_grid[j])
                data_x.append(data[i, j])
                # Squared return normalized by dt
                data_r2.append((returns[i, j] ** 2) / dt)
        
        data_t = np.array(data_t, dtype=np.float64)
        data_x = np.array(data_x, dtype=np.float64)
        data_r2 = np.array(data_r2, dtype=np.float64)
        
        # Adaptive bandwidth based on data scale
        x_std = np.std(data_x)
        h_x = self.h_x * x_std if x_std > 0 else self.h_x
        
        t_range = time_grid[-1] - time_grid[0]
        h_t = self.h_t * t_range if t_range > 0 else self.h_t
        
        # Compute surface using Numba
        vol_surface = _compute_local_volatility_numba(
            self.t_grid.astype(np.float64),
            self.x_grid.astype(np.float64),
            data_t,
            data_x,
            data_r2,
            h_t,
            h_x,
            self.min_vol
        )
        
        return vol_surface
    
    def __call__(
        self,
        t: Union[float, np.ndarray],
        x: np.ndarray
    ) -> np.ndarray:
        """
        Evaluate volatility at given time and state.
        
        Args:
            t: Time point(s) - scalar or array
            x: State(s) - shape (..., n_features) or (...)
        
        Returns:
            Volatility value(s) of same shape as x
        """
        if not self.is_fitted:
            raise RuntimeError("Calibrator must be fitted before evaluation")
        
        # Handle scalar t
        if np.isscalar(t):
            t = np.array([t])
            scalar_t = True
        else:
            t = np.asarray(t)
            scalar_t = False
        
        x = np.asarray(x)
        original_shape = x.shape
        
        # Flatten for batch processing
        if x.ndim == 0:
            x = x.reshape(1)
        
        if self.n_features == 1 or x.ndim == 1:
            # Univariate case
            x_flat = x.flatten()
            t_flat = np.broadcast_to(t, x_flat.shape).flatten()
            
            result = _interpolate_vol_batch_numba(
                t_flat.astype(np.float64),
                x_flat.astype(np.float64),
                self.t_grid.astype(np.float64),
                self.x_grid.astype(np.float64),
                self.vol_surface.astype(np.float64)
            )
            
            result = result.reshape(original_shape)
        else:
            # Multivariate case
            if x.ndim == 1:
                x = x.reshape(1, -1)
            
            n_points = x.shape[0]
            n_features = x.shape[1]
            
            t_flat = np.broadcast_to(t, (n_points,)).flatten()
            
            result = np.zeros_like(x)
            for k in range(n_features):
                result[:, k] = _interpolate_vol_batch_numba(
                    t_flat.astype(np.float64),
                    x[:, k].astype(np.float64),
                    self.t_grid.astype(np.float64),
                    self.x_grid.astype(np.float64),
                    self.vol_surface[:, :, k].astype(np.float64)
                )
            
            result = result.reshape(original_shape)
        
        return result
    
    def get_surface(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the calibrated volatility surface.
        
        Returns:
            Tuple of (t_grid, x_grid, vol_surface)
        """
        if not self.is_fitted:
            raise RuntimeError("Calibrator must be fitted first")
        return self.t_grid, self.x_grid, self.vol_surface
    
    def check_smile_shape(self) -> str:
        """
        Check if volatility surface exhibits smile/skew shape.
        
        Returns:
            String describing the shape: "Smile", "Skew", "Flat", or "Inverted"
        """
        if not self.is_fitted:
            raise RuntimeError("Calibrator must be fitted first")
        
        # Take middle time slice
        mid_t_idx = self.n_t_grid // 2
        
        if self.vol_surface.ndim == 2:
            vol_slice = self.vol_surface[mid_t_idx, :]
        else:
            vol_slice = self.vol_surface[mid_t_idx, :, 0]
        
        # Check shape
        mid_x_idx = self.n_x_grid // 2
        left_vol = np.mean(vol_slice[:mid_x_idx])
        mid_vol = vol_slice[mid_x_idx]
        right_vol = np.mean(vol_slice[mid_x_idx:])
        
        if left_vol > mid_vol and right_vol > mid_vol:
            return "Smile"
        elif left_vol > mid_vol > right_vol:
            return "Skew (Left)"
        elif left_vol < mid_vol < right_vol:
            return "Skew (Right)"
        elif left_vol < mid_vol and right_vol < mid_vol:
            return "Inverted U (Warning!)"
        else:
            return "Flat"
