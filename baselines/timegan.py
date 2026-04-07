"""
Legacy compatibility wrapper for TimeGAN.

The active implementation now lives in `models.timegan_baseline.TimeGAN`.
This file is kept only so older imports keep working.
"""

from models.timegan_baseline import TimeGAN

__all__ = ["TimeGAN"]
