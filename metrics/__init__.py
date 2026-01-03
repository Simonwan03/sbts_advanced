"""
Evaluation Metrics for Time Series Generation

This module provides comprehensive evaluation metrics for
assessing the quality of generated time series.

Metrics:
1. Discriminative Score: How well a classifier can distinguish real vs. generated
2. Predictive Score: How well generated data preserves predictive relationships
3. Statistical Metrics: Wasserstein distance, ACF, correlation structure
"""

from .discriminative_score import discriminative_score_metrics
from .predictive_score import predictive_score_metrics
from .statistical_metrics import compute_all_metrics

__all__ = [
    'discriminative_score_metrics',
    'predictive_score_metrics', 
    'compute_all_metrics'
]
