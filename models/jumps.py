import numpy as np

class JumpDetector:
    """
    Implements Ait-Sahalia's method for jump detection in high-frequency data.
    """
    def __init__(self, threshold_multiplier=4, dt=1.0):
        self.c = threshold_multiplier
        self.dt = dt
        
    def fit_detect(self, returns):
        """
        returns: Log returns of the time series
        Returns: boolean mask of jumps, calibrated params
        """
        # Estimate continuous volatility (bipower variation or robust stats)
        # Simplified: robust sigma estimate using MAD or similar excluding extremes
        abs_ret = np.abs(returns)
        sigma_hat = np.median(abs_ret) / 0.6745 # Robust estimator assuming Gaussian core
        
        threshold = self.c * sigma_hat * np.sqrt(self.dt)
        
        jump_indices = abs_ret > threshold
        
        # Calibration
        jump_count = np.sum(jump_indices)
        total_time = len(returns) * self.dt
        intensity_lambda = jump_count / total_time
        
        detected_jumps = returns[jump_indices]
        
        if len(detected_jumps) > 0:
            jump_mean = np.mean(detected_jumps)
            jump_std = np.std(detected_jumps)
        else:
            jump_mean, jump_std = 0, 0
            
        return {
            'indices': jump_indices,
            'lambda': intensity_lambda,
            'mu_jump': jump_mean,
            'sigma_jump': jump_std
        }