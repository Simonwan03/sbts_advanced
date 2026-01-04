"""
Feedback Mechanism Module

Implements the Transient Stress Factor dynamics for JD-SBTS-F.

The feedback mechanism captures "Volatility Clustering" by introducing
a Shot Noise process that amplifies volatility after jumps.

Mathematical Model:
    dS_t = -κ * S_t * dt + γ * |dJ_t|
    σ_eff(t, x) = σ_LV(t, x) * √(1 + S_t)

Author: Manus AI
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from numba import njit, prange
from dataclasses import dataclass


@dataclass
class FeedbackConfig:
    """Configuration for feedback mechanism."""
    kappa: float = 5.0      # Mean reversion speed
    gamma: float = 0.5      # Jump impact multiplier
    s0: float = 0.0         # Initial stress level
    max_stress: float = 10.0  # Maximum stress cap (for stability)


@njit(cache=True)
def _update_stress_scalar(
    s_prev: float,
    dt: float,
    jump_size: float,
    kappa: float,
    gamma: float,
    max_stress: float = 10.0
) -> float:
    """
    Update stress factor for a single step.
    
    S_t = S_{t-1} * exp(-κ*dt) + γ * |dJ|
    
    Args:
        s_prev: Previous stress value
        dt: Time step
        jump_size: Absolute jump size
        kappa: Mean reversion speed
        gamma: Jump impact multiplier
        max_stress: Maximum stress cap
    
    Returns:
        Updated stress value
    """
    decay = np.exp(-kappa * dt)
    s_new = s_prev * decay + gamma * np.abs(jump_size)
    return min(s_new, max_stress)


@njit(cache=True, parallel=True)
def _simulate_stress_trajectory(
    jump_sizes: np.ndarray,
    time_grid: np.ndarray,
    kappa: float,
    gamma: float,
    s0: float = 0.0,
    max_stress: float = 10.0
) -> np.ndarray:
    """
    Simulate stress factor trajectory given jump sizes.
    
    Args:
        jump_sizes: Jump sizes (n_samples, n_steps)
        time_grid: Time points (n_steps,)
        kappa: Mean reversion speed
        gamma: Jump impact multiplier
        s0: Initial stress level
        max_stress: Maximum stress cap
    
    Returns:
        Stress trajectory (n_samples, n_steps)
    """
    n_samples, n_steps = jump_sizes.shape
    stress = np.zeros((n_samples, n_steps))
    
    for i in prange(n_samples):
        stress[i, 0] = s0
        
        for t in range(1, n_steps):
            dt = time_grid[t] - time_grid[t-1]
            stress[i, t] = _update_stress_scalar(
                stress[i, t-1],
                dt,
                jump_sizes[i, t-1],
                kappa,
                gamma,
                max_stress
            )
    
    return stress


@njit(cache=True)
def _compute_effective_volatility(
    base_vol: np.ndarray,
    stress: np.ndarray
) -> np.ndarray:
    """
    Compute effective volatility with stress factor.
    
    σ_eff = σ_base * √(1 + S)
    
    Args:
        base_vol: Base volatility (n_samples, n_steps, n_features)
        stress: Stress factor (n_samples, n_steps)
    
    Returns:
        Effective volatility (n_samples, n_steps, n_features)
    """
    n_samples, n_steps, n_features = base_vol.shape
    eff_vol = np.zeros_like(base_vol)
    
    for i in range(n_samples):
        for t in range(n_steps):
            multiplier = np.sqrt(1.0 + stress[i, t])
            for k in range(n_features):
                eff_vol[i, t, k] = base_vol[i, t, k] * multiplier
    
    return eff_vol


class StressFactor:
    """
    Transient Stress Factor for Jump-Volatility Feedback.
    
    Models the temporary increase in volatility following jumps,
    capturing the "Volatility Clustering" phenomenon.
    
    Usage:
        stress = StressFactor(kappa=5.0, gamma=0.5)
        stress_trajectory = stress.simulate(jump_sizes, time_grid)
        eff_vol = stress.apply_to_volatility(base_vol, stress_trajectory)
    """
    
    def __init__(
        self,
        kappa: float = 5.0,
        gamma: float = 0.5,
        s0: float = 0.0,
        max_stress: float = 10.0
    ):
        """
        Initialize stress factor.
        
        Args:
            kappa: Mean reversion speed (higher = faster decay)
            gamma: Jump impact multiplier (higher = stronger impact)
            s0: Initial stress level
            max_stress: Maximum stress cap for numerical stability
        """
        self.kappa = kappa
        self.gamma = gamma
        self.s0 = s0
        self.max_stress = max_stress
        
        # Derived quantities
        self.half_life = np.log(2) / kappa if kappa > 0 else np.inf
    
    def simulate(
        self,
        jump_sizes: np.ndarray,
        time_grid: np.ndarray
    ) -> np.ndarray:
        """
        Simulate stress factor trajectory.
        
        Args:
            jump_sizes: Jump sizes (n_samples, n_steps) or (n_samples, n_steps, n_features)
            time_grid: Time points (n_steps,)
        
        Returns:
            Stress trajectory (n_samples, n_steps)
        """
        # If multivariate, sum absolute jumps across features
        if jump_sizes.ndim == 3:
            jump_sizes_agg = np.sum(np.abs(jump_sizes), axis=-1)
        else:
            jump_sizes_agg = np.abs(jump_sizes)
        
        return _simulate_stress_trajectory(
            jump_sizes_agg.astype(np.float64),
            time_grid.astype(np.float64),
            self.kappa,
            self.gamma,
            self.s0,
            self.max_stress
        )
    
    def apply_to_volatility(
        self,
        base_vol: np.ndarray,
        stress: np.ndarray
    ) -> np.ndarray:
        """
        Apply stress factor to base volatility.
        
        Args:
            base_vol: Base volatility (n_samples, n_steps, n_features)
            stress: Stress factor (n_samples, n_steps)
        
        Returns:
            Effective volatility (n_samples, n_steps, n_features)
        """
        # Ensure 3D
        if base_vol.ndim == 2:
            base_vol = base_vol[:, :, np.newaxis]
        
        return _compute_effective_volatility(
            base_vol.astype(np.float64),
            stress.astype(np.float64)
        )
    
    def get_multiplier(self, stress: np.ndarray) -> np.ndarray:
        """
        Get volatility multiplier from stress.
        
        Args:
            stress: Stress values
        
        Returns:
            Volatility multiplier √(1 + S)
        """
        return np.sqrt(1.0 + stress)
    
    def expected_stress(self, jump_intensity: float) -> float:
        """
        Compute expected long-run stress level.
        
        E[S] = γ * λ * E[|J|] / κ
        
        For a compound Poisson process with intensity λ.
        
        Args:
            jump_intensity: Jump intensity λ
        
        Returns:
            Expected stress level
        """
        # Assuming E[|J|] ≈ 1 for normalized jumps
        return self.gamma * jump_intensity / self.kappa if self.kappa > 0 else 0.0
    
    def get_config(self) -> Dict[str, float]:
        """Get configuration as dictionary."""
        return {
            'kappa': self.kappa,
            'gamma': self.gamma,
            's0': self.s0,
            'max_stress': self.max_stress,
            'half_life': self.half_life
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'StressFactor':
        """Create from configuration dictionary."""
        return cls(
            kappa=config.get('feedback_kappa', 5.0),
            gamma=config.get('feedback_gamma', 0.5),
            s0=config.get('feedback_s0', 0.0),
            max_stress=config.get('feedback_max_stress', 10.0)
        )


def calibrate_feedback_params(
    returns: np.ndarray,
    jump_mask: np.ndarray,
    time_grid: np.ndarray,
    method: str = 'moment_matching'
) -> Tuple[float, float]:
    """
    Calibrate feedback parameters from historical data.
    
    Args:
        returns: Return series (n_samples, n_steps)
        jump_mask: Boolean jump mask
        time_grid: Time points
        method: Calibration method ('moment_matching' or 'mle')
    
    Returns:
        Tuple of (kappa, gamma)
    """
    if method == 'moment_matching':
        # Simple moment matching approach
        
        # Estimate volatility around jumps vs. normal times
        vol_at_jumps = np.std(returns[jump_mask]) if np.any(jump_mask) else np.std(returns)
        vol_normal = np.std(returns[~jump_mask]) if np.any(~jump_mask) else np.std(returns)
        
        # Estimate gamma from volatility ratio
        vol_ratio = vol_at_jumps / vol_normal if vol_normal > 0 else 1.0
        gamma = max(0.1, (vol_ratio ** 2 - 1))  # From √(1 + S) = vol_ratio
        
        # Estimate kappa from autocorrelation decay
        # Use squared returns as volatility proxy
        sq_returns = returns ** 2
        
        # Compute autocorrelation at lag 1
        if len(sq_returns.flatten()) > 10:
            acf1 = np.corrcoef(sq_returns.flatten()[:-1], sq_returns.flatten()[1:])[0, 1]
            acf1 = max(0.01, min(0.99, acf1))  # Bound for stability
            
            # From AR(1): acf(1) ≈ exp(-κ*dt)
            dt = time_grid[1] - time_grid[0] if len(time_grid) > 1 else 1.0
            kappa = -np.log(acf1) / dt
            kappa = max(0.1, min(50.0, kappa))  # Reasonable bounds
        else:
            kappa = 5.0  # Default
        
        return kappa, gamma
    
    else:
        raise ValueError(f"Unknown calibration method: {method}")


def analyze_volatility_clustering(
    returns: np.ndarray,
    stress: np.ndarray,
    time_grid: np.ndarray
) -> Dict[str, Any]:
    """
    Analyze volatility clustering in generated data.
    
    Args:
        returns: Return series (n_samples, n_steps)
        stress: Stress factor trajectory (n_samples, n_steps)
        time_grid: Time points
    
    Returns:
        Dictionary with analysis results
    """
    # Compute realized volatility
    realized_vol = np.abs(returns)
    
    # Correlation between stress and realized volatility
    stress_vol_corr = np.corrcoef(
        stress.flatten(),
        realized_vol.flatten()
    )[0, 1]
    
    # Autocorrelation of squared returns (volatility clustering measure)
    sq_returns = returns ** 2
    acf_sq = []
    for lag in range(1, min(20, returns.shape[1])):
        acf = np.corrcoef(
            sq_returns[:, :-lag].flatten(),
            sq_returns[:, lag:].flatten()
        )[0, 1]
        acf_sq.append(acf)
    
    # Compute ARCH effect (variance of variance)
    vol_of_vol = np.std(realized_vol, axis=1).mean()
    
    return {
        'stress_vol_correlation': stress_vol_corr,
        'acf_squared_returns': acf_sq,
        'volatility_of_volatility': vol_of_vol,
        'mean_stress': np.mean(stress),
        'max_stress': np.max(stress),
        'stress_std': np.std(stress)
    }
