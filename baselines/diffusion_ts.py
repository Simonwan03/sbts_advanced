"""
Legacy compatibility wrapper for Diffusion-TS.

The active implementation now lives in `models.diffusion_ts_baseline.DiffusionTS`.
This file is kept only so older imports keep working.
"""

from models.diffusion_ts_baseline import DiffusionTS

__all__ = ["DiffusionTS"]
