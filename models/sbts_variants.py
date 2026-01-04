"""
JD-SBTS Model Variants

Implements the main JD-SBTS (Jump-Diffusion Schrödinger Bridge Time Series)
model variants:
    - JD-SBTS: Base model with static jumps
    - JD-SBTS-F: Feedback model with volatility clustering

Training Philosophy:
    "Decoupled Training" - Train components in isolation to prevent interference:
    1. Jump detection on raw data
    2. Volatility calibration on PURIFIED data (jumps removed)
    3. Drift estimation on PURIFIED data
    4. Combine components for generation

Author: Manus AI
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, Union, List
import warnings
from dataclasses import dataclass
import time

from models.base import GenerativeModel as TimeSeriesGenerator
from modules.volatility import LocalVolatilityCalibrator
from modules.jumps import StaticJumpDetector, NeuralJumpDetector, get_jump_detector
from modules.solver import JumpDiffusionEulerSolver
from modules.feedback import StressFactor
from modules.drift_neural import LSTMDriftEstimator, get_neural_drift_estimator
from modules.drift_kernel import KernelDriftEstimator


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    total_time: float = 0.0
    jump_detection_time: float = 0.0
    volatility_calibration_time: float = 0.0
    drift_estimation_time: float = 0.0
    n_jumps_detected: int = 0
    jump_intensity: float = 0.0
    vol_surface_shape: str = ""
    drift_loss_history: List[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_time': self.total_time,
            'jump_detection_time': self.jump_detection_time,
            'volatility_calibration_time': self.volatility_calibration_time,
            'drift_estimation_time': self.drift_estimation_time,
            'n_jumps_detected': self.n_jumps_detected,
            'jump_intensity': self.jump_intensity,
            'vol_surface_shape': self.vol_surface_shape,
            'drift_loss_history': self.drift_loss_history
        }


class JDSBTS(TimeSeriesGenerator):
    """
    JD-SBTS: Jump-Diffusion Schrödinger Bridge Time Series Generator.
    
    Base model that combines:
        - Local volatility calibration
        - Static jump detection (Merton Jump-Diffusion)
        - Neural or kernel drift estimation
    
    Key Innovation: "Filter & Interpolate" strategy for volatility calibration
    that properly separates jump and diffusion components.
    
    Usage:
        model = JDSBTS(config)
        model.fit(data, time_grid)
        generated = model.generate(n_samples, n_steps)
    """
    
    MODEL_TYPE = "jd_sbts"
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize JD-SBTS model.
        
        Args:
            config: Configuration dictionary with keys:
                - use_neural_jumps: Use neural jump detector (default: False)
                - use_neural_drift: Use neural drift estimator (default: True)
                - drift_estimator: 'lstm', 'transformer', or 'kernel' (default: 'lstm')
                - use_feedback: Use feedback mechanism (default: False for base model)
                - See individual module configs for more options
        """
        super().__init__(config)
        
        self.use_neural_jumps = config.get('use_neural_jumps', False)
        self.use_neural_drift = config.get('use_neural_drift', True)
        self.drift_type = config.get('drift_estimator', 'lstm')
        self.use_feedback = config.get('use_feedback', False)
        
        # Components (initialized during fit)
        self.jump_detector = None
        self.volatility_calibrator = None
        self.drift_estimator = None
        self.solver = None
        self.stress_factor = None
        
        # Training data
        self.time_grid = None
        self.n_features = None
        self.x0_samples = None  # Initial conditions from training data
        
        # Metrics
        self.training_metrics = TrainingMetrics()
    
    def fit(
        self,
        data: np.ndarray,
        time_grid: np.ndarray,
        verbose: bool = True
    ) -> 'JDSBTS':
        """
        Fit JD-SBTS model using decoupled training.
        
        Training Steps:
            1. Jump Detection: Detect jumps in raw data
            2. Data Purification: Remove jumps and interpolate
            3. Volatility Calibration: Fit σ(t, x) on purified data
            4. Drift Estimation: Train drift estimator on purified data
        
        Args:
            data: Time series data (n_samples, seq_len, n_features)
                  or (n_samples, seq_len) for univariate
            time_grid: Time points (seq_len,)
            verbose: Whether to print progress
        
        Returns:
            self for method chaining
        """
        start_time = time.time()
        
        # Ensure 3D
        if data.ndim == 2:
            data = data[:, :, np.newaxis]
        
        n_samples, seq_len, n_features = data.shape
        self.n_features = n_features
        self.time_grid = time_grid
        
        # Store initial conditions for generation
        self.x0_samples = data[:, 0, :].copy()
        
        if verbose:
            print("=" * 60)
            print("JD-SBTS Training (Decoupled)")
            print("=" * 60)
        
        # ========================================
        # Step 1: Jump Detection
        # ========================================
        if verbose:
            print("\n[Step 1/4] Jump Detection...")
        
        jump_start = time.time()
        
        self.jump_detector = get_jump_detector(self.config)
        self.jump_detector.fit(data)
        
        jump_mask = self.jump_detector.detect(data)
        n_jumps = np.sum(jump_mask)
        
        self.training_metrics.jump_detection_time = time.time() - jump_start
        self.training_metrics.n_jumps_detected = n_jumps
        self.training_metrics.jump_intensity = self.jump_detector.jump_intensity
        
        if verbose:
            print(f"  Detected {n_jumps} jumps")
            print(f"  Jump intensity λ = {self.jump_detector.jump_intensity:.4f}")
            print(f"  Jump mean μ_J = {self.jump_detector.jump_mean:.4f}")
            print(f"  Jump std σ_J = {self.jump_detector.jump_std:.4f}")
        
        # ========================================
        # Step 2: Data Purification
        # ========================================
        if verbose:
            print("\n[Step 2/4] Data Purification (Filter & Interpolate)...")
        
        purified_data = self.jump_detector.filter_and_interpolate(data)
        
        if verbose:
            print("  Jumps removed and interpolated")
        
        # ========================================
        # Step 3: Volatility Calibration
        # ========================================
        if verbose:
            print("\n[Step 3/4] Volatility Calibration on Purified Data...")
        
        vol_start = time.time()
        
        self.volatility_calibrator = LocalVolatilityCalibrator(self.config)
        self.volatility_calibrator.fit(purified_data, time_grid, purified=True)
        
        vol_shape = self.volatility_calibrator.check_smile_shape()
        
        self.training_metrics.volatility_calibration_time = time.time() - vol_start
        self.training_metrics.vol_surface_shape = vol_shape
        
        if verbose:
            print(f"  Volatility surface shape: {vol_shape}")
            if "Inverted" in vol_shape:
                warnings.warn("Volatility surface shows inverted U shape - check data quality")
        
        # ========================================
        # Step 4: Drift Estimation
        # ========================================
        if verbose:
            print(f"\n[Step 4/4] Drift Estimation ({self.drift_type}) on Purified Data...")
        
        drift_start = time.time()
        
        if self.use_neural_drift:
            self.drift_estimator = get_neural_drift_estimator(self.config)
        else:
            self.drift_estimator = KernelDriftEstimator(self.config)
        
        self.drift_estimator.fit(purified_data, time_grid, verbose=verbose)
        
        self.training_metrics.drift_estimation_time = time.time() - drift_start
        
        if hasattr(self.drift_estimator, 'get_training_history'):
            self.training_metrics.drift_loss_history = self.drift_estimator.get_training_history()
        
        # ========================================
        # Initialize Solver
        # ========================================
        self.solver = JumpDiffusionEulerSolver(self.config)
        
        if self.use_feedback:
            self.stress_factor = StressFactor.from_config(self.config)
        
        self.training_metrics.total_time = time.time() - start_time
        self.is_fitted = True
        
        if verbose:
            print("\n" + "=" * 60)
            print(f"Training Complete! Total time: {self.training_metrics.total_time:.2f}s")
            print("=" * 60)
        
        return self
    
    def generate(
        self,
        n_samples: int,
        n_steps: Optional[int] = None,
        x0: Optional[np.ndarray] = None,
        return_stress: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate synthetic time series.
        
        Args:
            n_samples: Number of samples to generate
            n_steps: Number of time steps (default: same as training)
            x0: Initial conditions (default: sample from training data)
            return_stress: Whether to return stress factor trajectory
        
        Returns:
            Generated paths (n_samples, n_steps, n_features)
            Optionally also stress factor (n_samples, n_steps)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before generation")
        
        if n_steps is None:
            n_steps = len(self.time_grid)
        
        # Create time grid
        t_start = self.time_grid[0]
        t_end = self.time_grid[-1]
        time_grid = np.linspace(t_start, t_end, n_steps)
        
        # Initial conditions
        if x0 is None:
            # Sample from training data initial conditions
            idx = np.random.choice(len(self.x0_samples), n_samples, replace=True)
            x0 = self.x0_samples[idx]
        
        # Ensure 2D
        if x0.ndim == 1:
            x0 = x0[:, np.newaxis]
        
        # Define drift function
        def drift_fn(t, x):
            return self.drift_estimator.predict(t, x)
        
        # Define volatility function
        def vol_fn(t, x):
            return self.volatility_calibrator(t, x)
        
        # Define jump sampler
        def jump_sampler(n, steps, dt):
            return self.jump_detector.sample_jumps(n, steps, dt)
        
        # Solve SDE
        result = self.solver.solve(
            x0=x0,
            time_grid=time_grid,
            drift_fn=drift_fn,
            volatility_fn=vol_fn,
            jump_sampler=jump_sampler,
            return_stress=return_stress
        )
        
        return result
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get training metrics."""
        return self.training_metrics.to_dict()
    
    def get_components(self) -> Dict[str, Any]:
        """Get model components for inspection."""
        return {
            'jump_detector': self.jump_detector,
            'volatility_calibrator': self.volatility_calibrator,
            'drift_estimator': self.drift_estimator,
            'solver': self.solver,
            'stress_factor': self.stress_factor
        }


class JDSBTSF(JDSBTS):
    """
    JD-SBTS-F: Jump-Diffusion Schrödinger Bridge with Feedback.
    
    Extends JD-SBTS with the "Jump-Volatility Interaction" mechanism
    that captures volatility clustering.
    
    Key Innovation: Transient Stress Factor S_t that amplifies volatility
    after jumps, then decays exponentially.
    
    Mathematical Model:
        dX_t = μ(t, X_t)dt + σ_LV(t, X_t) * √(1 + S_t) * dW_t + dJ_t
        dS_t = -κ * S_t * dt + γ * |dJ_t|
    
    Usage:
        model = JDSBTSF(config)
        model.fit(data, time_grid)
        generated, stress = model.generate(n_samples, return_stress=True)
    """
    
    MODEL_TYPE = "jd_sbts_f"
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize JD-SBTS-F model.
        
        Args:
            config: Configuration dictionary with additional keys:
                - feedback_kappa: Mean reversion speed (default: 5.0)
                - feedback_gamma: Jump impact multiplier (default: 0.5)
        """
        # Force feedback to be enabled
        config = config.copy()
        config['use_feedback'] = True
        
        super().__init__(config)
        
        self.kappa = config.get('feedback_kappa', 5.0)
        self.gamma = config.get('feedback_gamma', 0.5)
    
    def fit(
        self,
        data: np.ndarray,
        time_grid: np.ndarray,
        verbose: bool = True
    ) -> 'JDSBTSF':
        """
        Fit JD-SBTS-F model.
        
        Same as JD-SBTS but also calibrates feedback parameters.
        
        Args:
            data: Time series data
            time_grid: Time points
            verbose: Whether to print progress
        
        Returns:
            self for method chaining
        """
        # Call parent fit
        super().fit(data, time_grid, verbose=verbose)
        
        # Initialize stress factor with calibrated or configured parameters
        self.stress_factor = StressFactor(
            kappa=self.kappa,
            gamma=self.gamma
        )
        
        if verbose:
            print(f"\n[Feedback] Stress Factor Parameters:")
            print(f"  κ (mean reversion) = {self.kappa:.2f}")
            print(f"  γ (jump impact) = {self.gamma:.2f}")
            print(f"  Half-life = {self.stress_factor.half_life:.2f} time units")
        
        return self
    
    def generate(
        self,
        n_samples: int,
        n_steps: Optional[int] = None,
        x0: Optional[np.ndarray] = None,
        return_stress: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate synthetic time series with feedback mechanism.
        
        Args:
            n_samples: Number of samples to generate
            n_steps: Number of time steps
            x0: Initial conditions
            return_stress: Whether to return stress factor trajectory
        
        Returns:
            Generated paths (n_samples, n_steps, n_features)
            Optionally also stress factor (n_samples, n_steps)
        """
        # Use parent's generate with return_stress=True to get stress
        return super().generate(
            n_samples=n_samples,
            n_steps=n_steps,
            x0=x0,
            return_stress=return_stress
        )
    
    def analyze_feedback_effect(
        self,
        generated_paths: np.ndarray,
        stress_trajectory: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze the effect of feedback mechanism on generated data.
        
        Args:
            generated_paths: Generated paths from generate()
            stress_trajectory: Stress factor from generate()
        
        Returns:
            Dictionary with analysis results
        """
        from modules.feedback import analyze_volatility_clustering
        
        # Compute returns
        returns = np.diff(generated_paths, axis=1)
        if returns.ndim == 3:
            returns = returns[:, :, 0]  # Use first feature
        
        return analyze_volatility_clustering(
            returns,
            stress_trajectory,
            self.time_grid
        )


class JDSBTSNeural(JDSBTS):
    """
    JD-SBTS with Neural Jump Detection.
    
    Uses a neural network to predict time-varying jump intensity λ(t, h_t)
    instead of static parameters.
    """
    
    MODEL_TYPE = "jd_sbts_neural"
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with neural jumps enabled."""
        config = config.copy()
        config['use_neural_jumps'] = True
        super().__init__(config)


class JDSBTSFNeural(JDSBTSF):
    """
    JD-SBTS-F with Neural Jump Detection.
    
    Combines feedback mechanism with neural jump intensity prediction.
    """
    
    MODEL_TYPE = "jd_sbts_f_neural"
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with neural jumps and feedback enabled."""
        config = config.copy()
        config['use_neural_jumps'] = True
        config['use_feedback'] = True
        super().__init__(config)


# ============================================
# Factory Function
# ============================================

def get_sbts_model(config: Dict[str, Any]) -> TimeSeriesGenerator:
    """
    Factory function to create SBTS model variant.
    
    Args:
        config: Configuration with 'model_type' key:
            - 'jd_sbts': Base model
            - 'jd_sbts_f': Feedback model
            - 'jd_sbts_neural': Neural jump model
            - 'jd_sbts_f_neural': Feedback + neural model
    
    Returns:
        TimeSeriesGenerator instance
    """
    model_type = config.get('model_type', 'jd_sbts')
    
    if model_type == 'jd_sbts':
        return JDSBTS(config)
    elif model_type == 'jd_sbts_f':
        return JDSBTSF(config)
    elif model_type == 'jd_sbts_neural':
        return JDSBTSNeural(config)
    elif model_type == 'jd_sbts_f_neural':
        return JDSBTSFNeural(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
