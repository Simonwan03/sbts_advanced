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
        # Return scalar or broadcastable shape
        return self.sigma
    
    def sample_jump(self, t, dt):
        return 0.0

class LocalVolatilityReference(ReferenceMeasure):
    """
    Phase 3: Reference measure using calibrated Local Volatility surface.
    """
    def __init__(self, calibrator, volatility_multiplier=1.0):
        self.calibrator = calibrator
        self.multiplier = volatility_multiplier 
        
    def get_diffusion(self, t, x):
        # x shape (N, D) or (D,)
        # Predict returns (N, D) or (D,)
        vol = self.calibrator.predict(t, x)
        
        # --- FIX: Do NOT flatten. Keep (N, D) structure for multi-asset ---
        return vol * self.multiplier
        
    def sample_jump(self, t, dt):
        return 0.0

class JumpDiffusionReference(ReferenceMeasure):
    """
    Phase 4: Local Volatility + Poisson Jumps.
    Combines a volatility surface (continuous) with a jump sampler (discontinuous).
    """
    def __init__(self, vol_calibrator, jump_detector, volatility_multiplier=1.0):
        self.vol_calibrator = vol_calibrator
        self.jump_detector = jump_detector
        self.multiplier = volatility_multiplier
        
    def get_diffusion(self, t, x):
        # Continuous part: Local Volatility
        vol = self.vol_calibrator.predict(t, x)
        
        # --- FIX: Do NOT flatten. Keep (N, D) structure ---
        return vol * self.multiplier
    
    def sample_jump(self, t, dt):
        # Discontinuous part: Sample from Jump Detector
        # Note: In solver.py loop, this is called per path, 
        # so batch_size=1 returns shape (1, D) or (1,) depending on implementation.
        # If solver loop handles N paths individually, this is fine.
        # Ideally jump_detector should handle dim>1, but assuming jumps are sampled per path logic.
        jumps = self.jump_detector.sample_jump(batch_size=1)
        
        # Ensure we return a scalar or simple array, solver handles dimension expansion
        if isinstance(jumps, np.ndarray) and jumps.size == 1:
            return jumps[0]
        return jumps