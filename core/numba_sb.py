"""
Numba-Accelerated Schrödinger Bridge for Time Series Generation

Based on the implementation from alexouadi/SBTS:
https://github.com/alexouadi/SBTS

This module provides a fast, Numba-compiled implementation of the 
Markovian Schrödinger Bridge kernel regression method.

Key Features:
- JIT-compiled kernel functions for speed
- Markovian weight updates for memory efficiency
- Multi-dimensional support
"""

import numpy as np
import numba as nb
import time


@nb.jit(nopython=True, cache=True)
def kernel_1d(x, h):
    """
    Epanechnikov-like kernel for 1D data.
    
    Args:
        x: Distance values
        h: Bandwidth
        
    Returns:
        Kernel weights
    """
    return np.where(np.abs(x) < h, (h ** 2 - x ** 2) ** 2, 0.0)


@nb.jit(nopython=True, cache=True)
def kernel_nd(x, h):
    """
    Epanechnikov-like kernel for multi-dimensional data.
    
    Args:
        x: (N, D) array of difference vectors
        h: Bandwidth
        
    Returns:
        (N,) array of kernel weights
    """
    x_norm = np.sqrt(np.sum(x ** 2, axis=1))
    return np.where(x_norm < h, (h ** 2 - x_norm ** 2) ** 2, 0.0)


@nb.jit(nopython=True, cache=True)
def simulate_sb_path_markovian(N, M, d, K, X, N_pi, h, deltati):
    """
    Simulate one time series path using Markovian Schrödinger Bridge.
    
    This is the core algorithm from SBTS, implementing:
    - Kernel-based drift estimation
    - Markovian weight updates (order K)
    - Euler-Maruyama discretization
    
    Args:
        N: Number of time steps (X.shape[1] - 1)
        M: Number of training samples
        d: Dimension of time series
        K: Markovian order (memory length)
        X: Training data of shape (M, N+1, d)
        N_pi: Number of Euler steps per observation interval
        h: Kernel bandwidth
        deltati: Time step between observations
        
    Returns:
        Generated path of shape (N+1, d)
    """
    # Euler scheme time grid
    time_step_Euler = deltati / N_pi
    v_time_step_Euler = np.arange(0, deltati + 1e-9, time_step_Euler)
    
    # Pre-generate Brownian increments
    num_brownian = (N * (len(v_time_step_Euler) - 1), d)
    Brownian = np.random.normal(0, 1, num_brownian)
    
    # Initialize
    X_ = X[0, 0].copy()  # Start from first training sample
    
    weights = np.ones(M)
    path = np.zeros((N + 1, d))
    path[0] = X_
    
    # Markovian history buffer
    last_K = np.empty((K, d), dtype=X.dtype)
    index_queue = 0
    brownian_idx = 0
    
    # Main simulation loop
    for i in range(N):
        # Update weights using Markovian structure
        if i > 0:
            if index_queue >= K:
                # Remove oldest weight contribution
                X_oldest = last_K[0]
                kernel_oldest = kernel_nd(X[:, i - K] - X_oldest, h)
                
                if np.any(kernel_oldest == 0):
                    # Reset weights if any kernel is zero
                    weights = np.ones(M)
                    ind_ref = i - K
                    for j in range(1, K):
                        weights *= kernel_nd(X[:, ind_ref + j] - last_K[j], h)
                else:
                    weights /= kernel_oldest
                
                # Shift history buffer
                last_K[:-1] = last_K[1:]
                last_K[-1] = X_
            else:
                last_K[index_queue] = X_
            
            index_queue += 1
            weights *= kernel_nd(X[:, i] - X_, h)
        else:
            weights[:] = 1.0 / M
        
        # Compute weights_tilde for bridge
        weights_tilde = weights.copy()
        diff = X[:, i + 1] - X_
        weights_tilde *= np.exp(np.sum(diff ** 2 / (2 * deltati), axis=1))
        
        # Euler-Maruyama steps within interval
        for k in range(len(v_time_step_Euler) - 1):
            timeprev = v_time_step_Euler[k]
            timestep = v_time_step_Euler[k + 1] - v_time_step_Euler[k]
            timestepsqrt = np.sqrt(timestep)
            
            if k == 0:
                # Initial step uses simple weights
                expec_den = np.sum(weights)
                numerator = np.sum(weights[:, np.newaxis] * (X[:, i + 1] - X_), axis=0)
            else:
                # Subsequent steps use bridge weights
                diff = X[:, i + 1] - X_
                termtoadd = np.sum(diff ** 2, axis=1)
                termtoadd = weights_tilde * np.exp(-termtoadd / (2 * (deltati - timeprev)))
                expec_den = np.sum(termtoadd)
                numerator = np.sum(termtoadd[:, np.newaxis] * (X[:, i + 1] - X_), axis=0)
            
            # Compute drift and update state
            if expec_den > 0:
                drift = (1 / (deltati - timeprev)) * (numerator / expec_den)
            else:
                drift = np.zeros(d)
            
            X_ = X_ + drift * timestep + Brownian[brownian_idx] * timestepsqrt
            brownian_idx += 1
        
        path[i + 1] = X_
    
    return path


class NumbaMarkovianSB:
    """
    Numba-accelerated Markovian Schrödinger Bridge generator.
    
    This class provides a high-level interface to the Numba-compiled
    SB simulation functions, following the SBTS implementation.
    
    Usage:
        sb = NumbaMarkovianSB(bandwidth=0.1, markov_order=3)
        generated = sb.generate(training_data, n_samples=100)
    """
    
    def __init__(self, bandwidth=0.1, markov_order=3, n_pi=10, dt=1/252):
        """
        Initialize the Numba SB generator.
        
        Args:
            bandwidth: Kernel bandwidth (h)
            markov_order: Order of Markovian approximation (K)
            n_pi: Number of Euler steps per observation interval
            dt: Time step between observations
        """
        self.h = bandwidth
        self.K = markov_order
        self.n_pi = n_pi
        self.dt = dt
        
    def generate(self, training_data, n_samples=100, verbose=True):
        """
        Generate synthetic time series using Schrödinger Bridge.
        
        Args:
            training_data: (M, T, D) array of training paths
            n_samples: Number of paths to generate
            verbose: Whether to print progress
            
        Returns:
            generated: (n_samples, T-1, D) array of generated paths
        """
        M, T, D = training_data.shape
        N = T - 1  # Number of steps to generate
        
        # Ensure data is contiguous and correct dtype
        X = np.ascontiguousarray(training_data, dtype=np.float64)
        
        # Output array
        generated = np.zeros((n_samples, T, D))
        
        if verbose:
            print(f"   [Numba-SB] Generating {n_samples} paths (h={self.h}, K={self.K})...")
            start_time = time.time()
        
        # Generate paths (could be parallelized with prange)
        for k in range(n_samples):
            generated[k] = simulate_sb_path_markovian(
                N, M, D, self.K, X, self.n_pi, self.h, self.dt
            )
            
            if verbose and k == 0:
                elapsed = time.time() - start_time
                estimated = elapsed * n_samples
                print(f"   [Numba-SB] Estimated time: {estimated:.1f}s")
        
        if verbose:
            print(f"   [Numba-SB] Generation complete in {time.time() - start_time:.1f}s")
        
        # Return without initial point (to match other methods)
        return generated[:, 1:, :]
    
    def set_bandwidth(self, h):
        """Update bandwidth parameter."""
        self.h = h
        
    def set_markov_order(self, K):
        """Update Markov order parameter."""
        self.K = K


# Compile functions on import
def _warmup():
    """Warm up Numba JIT compilation."""
    dummy_data = np.random.randn(10, 5, 2).astype(np.float64)
    sb = NumbaMarkovianSB(bandwidth=0.5, markov_order=2, n_pi=2)
    _ = sb.generate(dummy_data, n_samples=1, verbose=False)

try:
    _warmup()
except:
    pass  # Ignore warmup errors
