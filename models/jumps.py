import numpy as np

class JumpDetector:
    """
    Implements Ait-Sahalia's method for jump detection using thresholding.
    Also handles sampling for generative modeling.
    """
    def __init__(self, dt, threshold_multiplier=3.0): # 降低一点阈值，更容易捕捉跳跃
        self.dt = dt
        self.c = threshold_multiplier
        self.jump_params = {
            "lambda": 0.0,      # Jump intensity (per unit time)
            "mu": 0.0,          # Mean jump size
            "sigma": 0.0,       # Jump size volatility
            "indices": []       # Indices where jumps occurred
        }
        self.is_fitted = False

    def fit(self, returns):
        """
        Detects jumps in historical log-returns.
        returns: (N_samples, T_steps, 1) or (N_total,)
        """
        # Flatten data to treat as one long time series for statistics
        flat_returns = returns.flatten()
        
        # 1. Estimate continuous volatility (Robust to outliers)
        # Using Median Absolute Deviation (MAD) or Bipower Variation
        abs_ret = np.abs(flat_returns)
        # Sigma_hat approx MAD / 0.6745
        sigma_hat = np.median(abs_ret) / 0.6745 
        
        # 2. Thresholding: |r| > c * sigma * sqrt(dt)
        # For daily data, dt is small, so threshold is tight
        threshold = self.c * sigma_hat * np.sqrt(self.dt)
        
        # Identify jumps
        jump_indices = np.abs(flat_returns) > threshold
        detected_jumps = flat_returns[jump_indices]
        
        # 3. Calibrate Parameters
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
            
        self.jump_params["indices"] = jump_indices
        self.is_fitted = True
        
        print(f"   [Jumps] Detected {n_jumps} jumps. Intensity (lambda): {self.jump_params['lambda']:.4f}")
        print(f"   [Jumps] Jump Size Dist: N({self.jump_params['mu']:.5f}, {self.jump_params['sigma']:.5f})")
        
        return self

    def sample_jump(self, batch_size):
        """
        Samples jumps for a single time step dt.
        Returns: (batch_size,) array of jump sizes (mostly zeros).
        """
        if not self.is_fitted:
            return np.zeros(batch_size)
            
        # 1. Poisson Arrival: Prob of jump in dt approx lambda * dt
        # Using uniform random number vs threshold
        prob_jump = self.jump_params["lambda"] * self.dt
        
        # Mask: 1 if jump occurs, 0 otherwise
        # Note: This assumes max 1 jump per dt (valid for small dt)
        jump_mask = np.random.random(batch_size) < prob_jump
        
        if not np.any(jump_mask):
            return np.zeros(batch_size)
            
        # 2. Jump Size: Sample from N(mu, sigma)
        jump_sizes = np.random.normal(
            self.jump_params["mu"], 
            self.jump_params["sigma"], 
            size=batch_size
        )
        
        # Apply mask
        return jump_sizes * jump_mask