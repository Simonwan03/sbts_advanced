"""
Model Factory Module

Provides a unified interface for creating time series generation models.
Uses Factory pattern for clean model instantiation.

Author: Manus AI
"""

from typing import Dict, Any, Type, Optional
import warnings

from models.base import TimeSeriesGenerator
from models.sbts_variants import (
    JDSBTS, JDSBTSF, JDSBTSNeural, JDSBTSFNeural,
    get_sbts_model
)
from models.lightsb import LightSB, NumbaSB
from models.baselines import TimeGAN, DiffusionTS


# ============================================
# Model Registry
# ============================================

MODEL_REGISTRY: Dict[str, Type[TimeSeriesGenerator]] = {
    # JD-SBTS variants
    'jd_sbts': JDSBTS,
    'jd_sbts_f': JDSBTSF,
    'jd_sbts_neural': JDSBTSNeural,
    'jd_sbts_f_neural': JDSBTSFNeural,
    
    # Schrödinger Bridge variants
    'lightsb': LightSB,
    'numba_sb': NumbaSB,
    
    # Baselines
    'timegan': TimeGAN,
    'diffusion_ts': DiffusionTS,
}

# Aliases for convenience
MODEL_ALIASES = {
    'sbts': 'jd_sbts',
    'sbts_f': 'jd_sbts_f',
    'sbts_neural': 'jd_sbts_neural',
    'sbts_f_neural': 'jd_sbts_f_neural',
    'light_sb': 'lightsb',
    'numbasb': 'numba_sb',
    'time_gan': 'timegan',
    'diffusion': 'diffusion_ts',
}


# ============================================
# Factory Functions
# ============================================

def get_model(
    model_type: str,
    config: Optional[Dict[str, Any]] = None
) -> TimeSeriesGenerator:
    """
    Create a time series generation model.
    
    Args:
        model_type: Type of model to create. Options:
            - 'jd_sbts': Base JD-SBTS model
            - 'jd_sbts_f': JD-SBTS with feedback mechanism
            - 'jd_sbts_neural': JD-SBTS with neural jump detection
            - 'jd_sbts_f_neural': JD-SBTS with feedback and neural jumps
            - 'lightsb': Light Schrödinger Bridge
            - 'numba_sb': Numba-accelerated Markovian SB
            - 'timegan': TimeGAN baseline
            - 'diffusion_ts': Diffusion-TS baseline
        config: Configuration dictionary (optional)
    
    Returns:
        TimeSeriesGenerator instance
    
    Example:
        >>> model = get_model('jd_sbts_f', {'feedback_kappa': 5.0})
        >>> model.fit(data, time_grid)
        >>> generated = model.generate(100)
    """
    if config is None:
        config = {}
    
    # Resolve aliases
    model_type = model_type.lower()
    if model_type in MODEL_ALIASES:
        model_type = MODEL_ALIASES[model_type]
    
    # Check if model type is registered
    if model_type not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys()) + list(MODEL_ALIASES.keys())
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Available models: {available}"
        )
    
    # Add model_type to config
    config['model_type'] = model_type
    
    # Create model
    model_class = MODEL_REGISTRY[model_type]
    return model_class(config)


def list_models() -> Dict[str, str]:
    """
    List all available models with descriptions.
    
    Returns:
        Dictionary mapping model names to descriptions
    """
    descriptions = {
        'jd_sbts': 'Jump-Diffusion Schrödinger Bridge (base model)',
        'jd_sbts_f': 'JD-SBTS with Feedback mechanism (volatility clustering)',
        'jd_sbts_neural': 'JD-SBTS with Neural jump detection',
        'jd_sbts_f_neural': 'JD-SBTS with Feedback + Neural jumps',
        'lightsb': 'Light Schrödinger Bridge (variance annealing)',
        'numba_sb': 'Numba-accelerated Markovian SB (fast baseline)',
        'timegan': 'TimeGAN (Yoon et al., NeurIPS 2019)',
        'diffusion_ts': 'Diffusion model for time series',
    }
    return descriptions


def get_default_config(model_type: str) -> Dict[str, Any]:
    """
    Get default configuration for a model type.
    
    Args:
        model_type: Type of model
    
    Returns:
        Default configuration dictionary
    """
    # Resolve aliases
    model_type = model_type.lower()
    if model_type in MODEL_ALIASES:
        model_type = MODEL_ALIASES[model_type]
    
    # Base config
    base_config = {
        'model_type': model_type,
    }
    
    # Model-specific defaults
    if model_type in ['jd_sbts', 'jd_sbts_f', 'jd_sbts_neural', 'jd_sbts_f_neural']:
        base_config.update({
            # Jump detection
            'jump_threshold_std': 4.0,
            'jump_rolling_window': 20,
            'use_neural_jumps': 'neural' in model_type,
            
            # Volatility calibration
            'vol_bandwidth': 0.5,
            'vol_n_t_grid': 50,
            'vol_n_x_grid': 100,
            
            # Drift estimation
            'drift_estimator': 'lstm',
            'lstm_hidden': 128,
            'lstm_epochs': 50,
            'lstm_lr': 0.005,
            'lstm_dropout': 0.3,
            'lstm_use_huber': True,
            
            # Feedback (for _f variants)
            'use_feedback': '_f' in model_type,
            'feedback_kappa': 5.0,
            'feedback_gamma': 0.5,
            
            # Solver
            'solver_backend': 'numba',
        })
    
    elif model_type == 'lightsb':
        base_config.update({
            'lightsb_sigma_min': 0.01,
            'lightsb_sigma_max': 1.0,
            'lightsb_hidden_dim': 256,
            'lightsb_n_layers': 3,
            'lightsb_epochs': 100,
            'lightsb_lr': 0.001,
            'lightsb_batch_size': 256,
            'lightsb_ot_epsilon': 0.1,
            'lightsb_n_steps': 50,
        })
    
    elif model_type == 'numba_sb':
        base_config.update({
            'numba_sb_sigma': 0.1,
        })
    
    elif model_type == 'timegan':
        base_config.update({
            'timegan_hidden_dim': 64,
            'timegan_z_dim': 32,
            'timegan_n_layers': 2,
            'timegan_epochs': 50,
            'timegan_lr': 0.001,
            'timegan_batch_size': 128,
        })
    
    elif model_type == 'diffusion_ts':
        base_config.update({
            'diffusion_hidden_dim': 128,
            'diffusion_n_steps': 100,
            'diffusion_epochs': 100,
            'diffusion_lr': 0.001,
            'diffusion_batch_size': 64,
            'diffusion_beta_start': 0.0001,
            'diffusion_beta_end': 0.02,
        })
    
    return base_config


def create_model_comparison(
    model_types: list,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, TimeSeriesGenerator]:
    """
    Create multiple models for comparison.
    
    Args:
        model_types: List of model types to create
        config: Shared configuration (optional)
    
    Returns:
        Dictionary mapping model names to model instances
    
    Example:
        >>> models = create_model_comparison(
        ...     ['jd_sbts', 'jd_sbts_f', 'timegan'],
        ...     {'lstm_epochs': 30}
        ... )
        >>> for name, model in models.items():
        ...     model.fit(data, time_grid)
    """
    if config is None:
        config = {}
    
    models = {}
    for model_type in model_types:
        try:
            model_config = get_default_config(model_type)
            model_config.update(config)
            models[model_type] = get_model(model_type, model_config)
        except Exception as e:
            warnings.warn(f"Failed to create model '{model_type}': {e}")
    
    return models


# ============================================
# Model Information
# ============================================

def get_model_info(model_type: str) -> Dict[str, Any]:
    """
    Get detailed information about a model type.
    
    Args:
        model_type: Type of model
    
    Returns:
        Dictionary with model information
    """
    # Resolve aliases
    model_type = model_type.lower()
    if model_type in MODEL_ALIASES:
        model_type = MODEL_ALIASES[model_type]
    
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model_class = MODEL_REGISTRY[model_type]
    
    info = {
        'name': model_type,
        'class': model_class.__name__,
        'module': model_class.__module__,
        'docstring': model_class.__doc__,
        'default_config': get_default_config(model_type),
    }
    
    # Add model-specific info
    if hasattr(model_class, 'MODEL_TYPE'):
        info['model_type'] = model_class.MODEL_TYPE
    
    return info


# ============================================
# Convenience Functions
# ============================================

def create_jdsbts(config: Optional[Dict[str, Any]] = None) -> JDSBTS:
    """Create JD-SBTS model."""
    return get_model('jd_sbts', config)


def create_jdsbts_f(config: Optional[Dict[str, Any]] = None) -> JDSBTSF:
    """Create JD-SBTS-F model with feedback."""
    return get_model('jd_sbts_f', config)


def create_lightsb(config: Optional[Dict[str, Any]] = None) -> LightSB:
    """Create LightSB model."""
    return get_model('lightsb', config)


def create_timegan(config: Optional[Dict[str, Any]] = None) -> TimeGAN:
    """Create TimeGAN model."""
    return get_model('timegan', config)


def create_diffusion_ts(config: Optional[Dict[str, Any]] = None) -> DiffusionTS:
    """Create Diffusion-TS model."""
    return get_model('diffusion_ts', config)
