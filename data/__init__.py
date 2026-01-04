"""
Data Module

Provides data loading and preprocessing utilities for time series generation.

Features:
    - ETF data loading from Yahoo Finance
    - Synthetic data generation for testing
    - Data preprocessing and normalization
    - Train/test splitting

Author: Manus AI
"""

from data.loaders import (
    load_etf_data,
    load_synthetic_data,
    create_sliding_windows,
    normalize_data,
    denormalize_data
)

__all__ = [
    'load_etf_data',
    'load_synthetic_data',
    'create_sliding_windows',
    'normalize_data',
    'denormalize_data',
]
