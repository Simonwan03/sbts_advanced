"""
SOTA Baselines for Time Series Generation

This module provides implementations of state-of-the-art
time series generation methods for benchmarking.

Available Methods:
- TimeGAN: Time-series Generative Adversarial Network (Yoon et al., 2019)
- DiffusionTS: Diffusion-based Time Series Generation
"""

from .timegan import TimeGAN
from .diffusion_ts import DiffusionTS

__all__ = ['TimeGAN', 'DiffusionTS']
