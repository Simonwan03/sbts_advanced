import numpy as np
from abc import ABC, abstractmethod

class ReferenceMeasure(ABC):
    @abstractmethod
    def get_diffusion(self, t, x):
        """Returns sigma(t, x)"""
        pass
    
    @abstractmethod
    def sample_jump(self, t, dt):
        """Returns jump size dJ"""
        return 0.0

class BrownianMotion(ReferenceMeasure):
    def __init__(self, sigma=1.0):
        self.sigma = sigma
        
    def get_diffusion(self, t, x):
        return self.sigma

class LocalVolatility(ReferenceMeasure):
    """
    Proposed Alternative 1: Local Volatility Model
    dX_t = mu dt + sigma_LV(t, X_t) dW_t
    """
    def __init__(self, lv_surface_func):
        # lv_surface_func: callable (t, x) -> sigma
        self.lv_surface_func = lv_surface_func
        
    def get_diffusion(self, t, x):
        return self.lv_surface_func(t, x)

class LevyProcess(ReferenceMeasure):
    """
    Enhancement 4: Lévy Process Support
    dX_t = ... + dJ_t
    Compound Poisson + Brownian
    """
    def __init__(self, base_sigma, intensity_lambda, jump_dist_func):
        self.sigma = base_sigma
        self.lamb = intensity_lambda
        self.jump_dist = jump_dist_func # Function returning sample from G
        
    def get_diffusion(self, t, x):
        return self.sigma
        
    def sample_jump(self, t, dt):
        # Poisson arrival prob approx lambda * dt for small dt
        if np.random.random() < (self.lamb * dt):
            return self.jump_dist()
        return 0.0

def modified_drift_adjustment(reference: ReferenceMeasure, drift_kernel_est, t, x):
    """
    Phase 3 Eq: alpha* = (1/sigma_ref^2) * grad log(density)
    """
    sigma_ref = reference.get_diffusion(t, x)
    original_drift = drift_kernel_est(t, x)
    
    # Adjust magnitude based on local volatility
    # If original_drift is grad log density
    return (1.0 / (sigma_ref**2 + 1e-8)) * original_drift