"""
Abstract Base Classes for Generative Time Series Models

Defines the interface that all models must implement.

Author: Manus AI
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np


class GenerativeModel(ABC):
    """
    Abstract base class for all generative time series models.

    Active models in this repository should follow one simple interface:
        - fit(data, time_grid=None, verbose=True)
        - generate(n_samples, n_steps=None, x0=None, **kwargs)

    The input data convention is:
        - training windows shaped (n_samples, seq_len, n_features)
        - univariate data may also be provided as (n_samples, seq_len)

    The generation convention is:
        - return arrays shaped (n_samples, seq_len, n_features)
        - optional kwargs may be used by model-specific variants
    """
    
    # Model name following naming convention
    MODEL_NAME: str = "base_model"
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model with configuration.
        
        Args:
            config: Configuration dictionary containing hyperparameters
        """
        self.config = config
        self.is_fitted = False
        self._training_metrics = {}
    
    @abstractmethod
    def fit(
        self,
        data: np.ndarray,
        time_grid: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> 'GenerativeModel':
        """
        Fit the model to training data.
        
        Args:
            data: Training data of shape (n_samples, seq_len, n_features)
                  or (n_samples, seq_len) for univariate
            time_grid: Optional time grid for models that use it
            verbose: Whether to print training progress
        
        Returns:
            self for method chaining
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        n_samples: int,
        n_steps: Optional[int] = None,
        x0: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Generate synthetic time series samples.
        
        Args:
            n_samples: Number of samples to generate
            n_steps: Optional output length, defaults to training length
            x0: Optional seed / initial condition / prefix, depending on the model
            **kwargs: Additional generation parameters
        
        Returns:
            Generated samples of shape (n_samples, seq_len, n_features)
        """
        pass

    def sample(
        self,
        n_samples: int,
        n_steps: Optional[int] = None,
        x0: Optional[np.ndarray] = None,
        **kwargs
    ) -> np.ndarray:
        """Alias for generate() to support baseline-style sampling code."""
        return self.generate(n_samples=n_samples, n_steps=n_steps, x0=x0, **kwargs)
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Return metrics collected during training."""
        return self._training_metrics
    
    def get_config(self) -> Dict[str, Any]:
        """Return model configuration."""
        return self.config.copy()
    
    @property
    def name(self) -> str:
        """Return model name."""
        return self.MODEL_NAME
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.MODEL_NAME}, fitted={self.is_fitted})"


class DriftEstimator(ABC):
    """
    Abstract base class for drift estimators.
    
    Drift estimators learn the drift function μ(t, x) from data.
    """
    
    ESTIMATOR_TYPE: str = "base"
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize drift estimator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.is_fitted = False
    
    @abstractmethod
    def fit(
        self,
        data: np.ndarray,
        time_grid: np.ndarray
    ) -> 'DriftEstimator':
        """
        Fit the drift estimator to data.
        
        Args:
            data: Training data of shape (n_samples, seq_len, n_features)
            time_grid: Time points of shape (seq_len,)
        
        Returns:
            self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        t: Union[float, np.ndarray],
        x: np.ndarray
    ) -> np.ndarray:
        """
        Predict drift at given time and state.
        
        Args:
            t: Time point(s)
            x: State(s) of shape (..., n_features)
        
        Returns:
            Predicted drift of same shape as x
        """
        pass


class JumpDetector(ABC):
    """
    Abstract base class for jump detectors.
    
    Jump detectors identify discontinuous movements in time series
    and estimate jump parameters (intensity, size distribution).
    """
    
    DETECTOR_TYPE: str = "base"
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize jump detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.is_fitted = False
        
        # Jump parameters (to be estimated)
        self.jump_intensity = None  # λ (jumps per unit time)
        self.jump_mean = None       # μ_J (mean jump size)
        self.jump_std = None        # σ_J (jump size std)
    
    @abstractmethod
    def fit(self, data: np.ndarray) -> 'JumpDetector':
        """
        Fit jump detector and estimate parameters.
        
        Args:
            data: Time series data of shape (n_samples, seq_len, n_features)
        
        Returns:
            self for method chaining
        """
        pass
    
    @abstractmethod
    def detect(self, data: np.ndarray) -> np.ndarray:
        """
        Detect jumps in data.
        
        Args:
            data: Time series data
        
        Returns:
            Boolean mask of shape data.shape indicating jump locations
        """
        pass
    
    @abstractmethod
    def filter_and_interpolate(self, data: np.ndarray) -> np.ndarray:
        """
        Remove jumps and interpolate to create purified data.
        
        Args:
            data: Raw time series data
        
        Returns:
            Purified data with jumps removed and interpolated
        """
        pass
    
    def sample_jumps(
        self,
        n_samples: int,
        n_steps: int,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample jump occurrences and sizes.
        
        Args:
            n_samples: Number of paths
            n_steps: Number of time steps
            dt: Time step size
        
        Returns:
            Tuple of (jump_mask, jump_sizes)
        """
        if not self.is_fitted:
            raise RuntimeError("JumpDetector must be fitted before sampling")
        
        # Sample from Poisson process
        jump_probs = 1 - np.exp(-self.jump_intensity * dt)
        jump_mask = np.random.random((n_samples, n_steps)) < jump_probs
        
        # Sample jump sizes
        jump_sizes = np.random.normal(
            self.jump_mean,
            self.jump_std,
            (n_samples, n_steps)
        )
        jump_sizes = jump_sizes * jump_mask
        
        return jump_mask, jump_sizes


class VolatilityCalibrator(ABC):
    """
    Abstract base class for volatility calibration.
    
    Calibrates local volatility surface σ(t, x) from data.
    """
    
    CALIBRATOR_TYPE: str = "base"
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize volatility calibrator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.is_fitted = False
    
    @abstractmethod
    def fit(
        self,
        data: np.ndarray,
        time_grid: np.ndarray
    ) -> 'VolatilityCalibrator':
        """
        Fit volatility surface to data.
        
        Args:
            data: Training data
            time_grid: Time points
        
        Returns:
            self for method chaining
        """
        pass
    
    @abstractmethod
    def __call__(
        self,
        t: Union[float, np.ndarray],
        x: np.ndarray
    ) -> np.ndarray:
        """
        Evaluate volatility at given time and state.
        
        Args:
            t: Time point(s)
            x: State(s)
        
        Returns:
            Volatility value(s)
        """
        pass


# Alias for backward compatibility
BaseTimeSeriesGenerator = GenerativeModel
TimeSeriesGenerator = GenerativeModel


class SDESolver(ABC):
    """
    Abstract base class for SDE solvers.
    
    Solves stochastic differential equations of the form:
        dX_t = μ(t, X_t)dt + σ(t, X_t)dW_t + dJ_t
    """
    
    SOLVER_TYPE: str = "base"
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SDE solver.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    @abstractmethod
    def solve(
        self,
        x0: np.ndarray,
        time_grid: np.ndarray,
        drift_fn,
        volatility_fn,
        jump_sampler=None,
        **kwargs
    ) -> np.ndarray:
        """
        Solve the SDE.
        
        Args:
            x0: Initial conditions of shape (n_samples, n_features)
            time_grid: Time points of shape (n_steps,)
            drift_fn: Drift function μ(t, x)
            volatility_fn: Volatility function σ(t, x)
            jump_sampler: Optional jump sampler
            **kwargs: Additional solver parameters
        
        Returns:
            Solution paths of shape (n_samples, n_steps, n_features)
        """
        pass
