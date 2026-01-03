import numpy as np

class JumpDetector:
    """
    Implements Ait-Sahalia's method for jump detection using thresholding.
    
    REFACTORED (Step 1): Changed from "Clipping" to "Filter & Interpolate" strategy.
    
    Problem with Clipping:
        - Truncating returns > k*σ to k*σ artificially suppresses tail variance
        - Results in incorrect "inverted U" volatility surface (highest at mean 0)
    
    Solution (Filter & Interpolate):
        - Mark outliers (|r_t| > threshold) as jumps
        - Replace outliers with linear interpolation of neighbors for volatility estimation
        - This preserves the correct "Smile/Skew" shape in the volatility surface
    """
    def __init__(self, dt, threshold_multiplier=3.0):
        self.dt = dt
        self.c = threshold_multiplier
        self.jump_params = {
            "lambda": 0.0,      # Jump intensity (per unit time)
            "mu": 0.0,          # Mean jump size
            "sigma": 0.0,       # Jump size volatility
            "indices": []       # Indices where jumps occurred
        }
        self.is_fitted = False
        self._purified_returns = None  # Store purified returns for volatility calibration

    def fit(self, returns):
        """
        Detects jumps in historical log-returns and creates purified series.
        
        Args:
            returns: (N_samples, T_steps, D) or (N_total,) array of log-returns
            
        Returns:
            self: Fitted detector with jump parameters and purified returns
        """
        original_shape = returns.shape
        flat_returns = returns.flatten()
        
        # 1. Estimate continuous volatility (Robust to outliers using MAD)
        abs_ret = np.abs(flat_returns)
        sigma_hat = np.median(abs_ret) / 0.6745  # MAD-based robust estimator
        
        # 2. Compute threshold for jump detection
        threshold = self.c * sigma_hat * np.sqrt(self.dt)
        
        # 3. Identify jump indices
        jump_mask = np.abs(flat_returns) > threshold
        detected_jumps = flat_returns[jump_mask]
        
        # 4. Calibrate Jump Parameters (MJD: λ, μ_J, σ_J)
        n_jumps = len(detected_jumps)
        total_time = len(flat_returns) * self.dt
        
        if total_time > 0:
            self.jump_params["lambda"] = n_jumps / total_time
        else:
            self.jump_params["lambda"] = 0
            
        if n_jumps > 0:
            self.jump_params["mu"] = np.mean(detected_jumps)
            self.jump_params["sigma"] = np.std(detected_jumps)
        else:
            self.jump_params["mu"] = 0.0
            self.jump_params["sigma"] = 0.0
            
        self.jump_params["indices"] = jump_mask
        
        # 5. NEW: Create purified returns using Filter & Interpolate
        self._purified_returns = self._filter_and_interpolate(
            flat_returns.copy(), 
            jump_mask,
            original_shape
        )
        
        self.is_fitted = True
        
        print(f"   [Jumps] Detected {n_jumps} jumps. Intensity (λ): {self.jump_params['lambda']:.4f}")
        print(f"   [Jumps] Jump Size Dist: N({self.jump_params['mu']:.5f}, {self.jump_params['sigma']:.5f})")
        print(f"   [Jumps] Purification: Filter & Interpolate (preserves volatility smile)")
        
        return self

    def _filter_and_interpolate(self, flat_returns, jump_mask, original_shape):
        """
        Filter & Interpolate Strategy for Jump Purification.
        
        Instead of clipping (which distorts the volatility surface), we:
        1. Mark outliers as NaN
        2. Interpolate using neighboring values
        3. Return purified series for volatility calibration
        
        This ensures the Local Volatility Surface exhibits proper "Smile/Skew" shape.
        
        Args:
            flat_returns: Flattened return array
            jump_mask: Boolean mask indicating jump locations
            original_shape: Original shape of returns array
            
        Returns:
            purified_returns: Returns with jumps replaced by interpolated values
        """
        purified = flat_returns.copy()
        
        # Replace jumps with NaN for interpolation
        purified[jump_mask] = np.nan
        
        # Find indices of NaN values
        nan_indices = np.where(jump_mask)[0]
        valid_indices = np.where(~jump_mask)[0]
        
        if len(nan_indices) == 0 or len(valid_indices) == 0:
            # No jumps or all jumps - return original
            return flat_returns.reshape(original_shape)
        
        # Linear interpolation for each NaN
        for idx in nan_indices:
            # Find nearest valid neighbors
            left_valid = valid_indices[valid_indices < idx]
            right_valid = valid_indices[valid_indices > idx]
            
            if len(left_valid) > 0 and len(right_valid) > 0:
                # Interpolate between neighbors
                left_idx = left_valid[-1]
                right_idx = right_valid[0]
                weight = (idx - left_idx) / (right_idx - left_idx)
                purified[idx] = (1 - weight) * flat_returns[left_idx] + weight * flat_returns[right_idx]
            elif len(left_valid) > 0:
                # Use left neighbor only
                purified[idx] = flat_returns[left_valid[-1]]
            elif len(right_valid) > 0:
                # Use right neighbor only
                purified[idx] = flat_returns[right_valid[0]]
            else:
                # Fallback: use median of non-jump returns
                purified[idx] = np.median(flat_returns[~jump_mask])
        
        return purified.reshape(original_shape)

    def get_purified_returns(self):
        """
        Returns the purified return series for volatility calibration.
        
        This should be used by VolatilityCalibrator instead of raw/clipped returns
        to ensure the volatility surface exhibits proper Smile/Skew shape.
        
        Returns:
            purified_returns: Returns with jumps replaced by interpolated values
        """
        if not self.is_fitted:
            raise ValueError("JumpDetector must be fitted before getting purified returns.")
        return self._purified_returns

    def sample_jump(self, batch_size):
        """
        Samples jumps for a single time step dt.
        
        Args:
            batch_size: Number of samples to generate
            
        Returns:
            jump_sizes: (batch_size,) array of jump sizes (mostly zeros)
        """
        if not self.is_fitted:
            return np.zeros(batch_size)
            
        # 1. Poisson Arrival: Prob of jump in dt ≈ λ * dt
        prob_jump = self.jump_params["lambda"] * self.dt
        
        # Mask: 1 if jump occurs, 0 otherwise
        # Note: Assumes max 1 jump per dt (valid for small dt)
        jump_mask = np.random.random(batch_size) < prob_jump
        
        if not np.any(jump_mask):
            return np.zeros(batch_size)
            
        # 2. Jump Size: Sample from N(μ_J, σ_J)
        jump_sizes = np.random.normal(
            self.jump_params["mu"], 
            self.jump_params["sigma"], 
            size=batch_size
        )
        
        # Apply mask
        return jump_sizes * jump_mask

    def get_jump_statistics(self):
        """
        Returns calibrated jump parameters for reporting.
        
        Returns:
            dict: Jump parameters (λ, μ_J, σ_J, n_jumps)
        """
        if not self.is_fitted:
            raise ValueError("JumpDetector must be fitted first.")
        return {
            "intensity": self.jump_params["lambda"],
            "mean_size": self.jump_params["mu"],
            "size_volatility": self.jump_params["sigma"],
            "n_jumps": np.sum(self.jump_params["indices"])
        }
