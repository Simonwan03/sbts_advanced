"""
Compatibility exports for baseline models.

The active implementations now live in dedicated modules:
    - models.timegan_baseline
    - models.diffusion_ts_baseline

This file is kept as a thin wrapper so existing imports such as
`from models.baselines import TimeGAN, DiffusionTS` continue to work.
"""

from models.timegan_baseline import TimeGAN
from models.diffusion_ts_baseline import DiffusionTS

__all__ = ["TimeGAN", "DiffusionTS"]
