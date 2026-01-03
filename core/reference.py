"""
Reference Measures for Schrödinger Bridge Time Series (SBTS)

This module defines reference measures (priors) for the SDE:
    dX_t = drift(t, X_t) dt + σ(t, X_t) dW_t + dJ_t

Reference measures include:
1. BrownianMotion: Standard Wiener process
2. LocalVolatilityReference: Calibrated local volatility surface
3. JumpDiffusionReference: Local volatility + static Poisson jumps (StaticMJD)
4. NeuralJumpDiffusionReference: Local volatility + neural time-varying jumps (NeuralMJD)

REFACTORED (Step 2): Added NeuralMJD support for "Endogenous Jumps"
"""

import numpy as np
from abc import ABC, abstractmethod


class ReferenceMeasure(ABC):
    """Abstract base class for reference measures."""
    
    @abstractmethod
    def get_diffusion(self, t, x):
        """Returns diffusion coefficient σ(t, x)."""
        pass
    
    @abstractmethod
    def sample_jump(self, t, dt, **kwargs):
        """Returns jump size dJ."""
        return 0.0
    
    def get_type(self):
        """Returns the type of reference measure."""
        return self.__class__.__name__


class BrownianMotion(ReferenceMeasure):
    """Standard Brownian motion reference measure."""
    
    def __init__(self, sigma=1.0):
        self.sigma = sigma
        
    def get_diffusion(self, t, x):
        return self.sigma
    
    def sample_jump(self, t, dt, **kwargs):
        return 0.0


class LocalVolatilityReference(ReferenceMeasure):
    """
    Reference measure using calibrated Local Volatility surface.
    
    The diffusion coefficient σ_LV(t, x) is learned from historical data
    using Kernel Ridge Regression.
    """
    def __init__(self, calibrator, volatility_multiplier=1.0):
        self.calibrator = calibrator
        self.multiplier = volatility_multiplier 
        
    def get_diffusion(self, t, x):
        vol = self.calibrator.predict(t, x)
        return vol * self.multiplier
        
    def sample_jump(self, t, dt, **kwargs):
        return 0.0


class JumpDiffusionReference(ReferenceMeasure):
    """
    Static Jump-Diffusion Reference (StaticMJD).
    
    Combines:
    - Local Volatility surface (continuous component)
    - Static Poisson jumps with constant intensity λ (discontinuous component)
    
    The jump intensity λ is calibrated from historical data and remains constant.
    """
    def __init__(self, vol_calibrator, jump_detector, volatility_multiplier=1.0):
        self.vol_calibrator = vol_calibrator
        self.jump_detector = jump_detector
        self.multiplier = volatility_multiplier
        self._mode = "static"
        
    def get_diffusion(self, t, x):
        vol = self.vol_calibrator.predict(t, x)
        return vol * self.multiplier
    
    def sample_jump(self, t, dt, **kwargs):
        """
        Sample from static MJD (constant λ).
        
        Returns:
            jump: scalar or array of jump sizes
        """
        jumps = self.jump_detector.sample_jump(batch_size=1)
        
        if isinstance(jumps, np.ndarray) and jumps.size == 1:
            return jumps[0]
        return jumps
    
    def get_type(self):
        return f"JumpDiffusion (mode={self._mode})"


class NeuralJumpDiffusionReference(ReferenceMeasure):
    """
    Neural Jump-Diffusion Reference (NeuralMJD).
    
    ADDED (Step 2): Implements "Endogenous Jumps" where jump intensity λ(t)
    is predicted by a neural network based on historical data.
    
    Key difference from JumpDiffusionReference:
    - λ is a function of hidden state h_t, not a constant
    - Requires historical sequence for intensity prediction
    
    Usage:
        ref = NeuralJumpDiffusionReference(vol_calibrator, neural_jump_detector)
        # During generation, pass x_history for intensity prediction:
        jump = ref.sample_jump(t, dt, x_history=history_seq, batch_size=n_paths)
    """
    def __init__(self, vol_calibrator, neural_jump_detector, volatility_multiplier=1.0):
        self.vol_calibrator = vol_calibrator
        self.neural_jump_detector = neural_jump_detector
        self.multiplier = volatility_multiplier
        self._mode = "neural"
        
    def get_diffusion(self, t, x):
        vol = self.vol_calibrator.predict(t, x)
        return vol * self.multiplier
    
    def sample_jump(self, t, dt, x_history=None, batch_size=1, **kwargs):
        """
        Sample from Neural MJD (time-varying λ).
        
        Args:
            t: current time
            dt: time step
            x_history: (batch, seq_len, D) historical observations for intensity prediction
            batch_size: number of samples (used if x_history is None)
            
        Returns:
            jump: array of jump sizes
        """
        if x_history is not None:
            # Use neural intensity
            jumps = self.neural_jump_detector.sample_jump(
                batch_size=x_history.shape[0] if hasattr(x_history, 'shape') else batch_size,
                x_history=x_history,
                t=t
            )
        else:
            # Fallback to static intensity
            jumps = self.neural_jump_detector.sample_jump(batch_size=batch_size, t=t)
        
        if isinstance(jumps, np.ndarray) and jumps.size == 1:
            return jumps[0]
        return jumps
    
    def get_intensity(self, t, x_history):
        """
        Get predicted jump intensity λ(t) given historical data.
        
        Args:
            t: current time
            x_history: (batch, seq_len, D) historical observations
            
        Returns:
            lambda_t: predicted intensity
        """
        return self.neural_jump_detector.get_intensity(x_history, t)
    
    def get_type(self):
        return f"NeuralJumpDiffusion (mode={self._mode})"


def create_reference_measure(vol_calibrator, jump_detector=None, 
                            use_neural_jumps=False, volatility_multiplier=1.0):
    """
    Factory function to create appropriate reference measure.
    
    Args:
        vol_calibrator: Fitted VolatilityCalibrator
        jump_detector: JumpDetector or NeuralJumpDetector (optional)
        use_neural_jumps: If True, use NeuralMJD; otherwise use StaticMJD
        volatility_multiplier: Scaling factor for volatility
        
    Returns:
        ReferenceMeasure: Appropriate reference measure instance
    """
    if jump_detector is None:
        return LocalVolatilityReference(vol_calibrator, volatility_multiplier)
    
    if use_neural_jumps:
        # Check if detector supports neural mode
        if hasattr(jump_detector, 'get_intensity'):
            return NeuralJumpDiffusionReference(
                vol_calibrator, jump_detector, volatility_multiplier
            )
        else:
            print("   [Warning] Jump detector doesn't support neural mode, using static MJD")
            return JumpDiffusionReference(
                vol_calibrator, jump_detector, volatility_multiplier
            )
    else:
        return JumpDiffusionReference(
            vol_calibrator, jump_detector, volatility_multiplier
        )
