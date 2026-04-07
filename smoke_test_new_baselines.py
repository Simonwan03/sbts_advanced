#!/usr/bin/env python3
"""
Lightweight smoke test for the new baseline models.

Checks:
    - model instantiation through the factory
    - one tiny training run
    - autoregressive generation output shape
    - evaluation compatibility with the current metrics stack

Usage:
    python smoke_test_new_baselines.py --model rnn
    python smoke_test_new_baselines.py --model transformer_ar
    python smoke_test_new_baselines.py --model all
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.loaders import create_sliding_windows, load_synthetic_data
from metrics import discriminative_score_metrics, predictive_score_metrics
from metrics.numba_metrics import compute_all_metrics_numba
from models.factory import get_default_config, get_model


def build_tiny_dataset(seed: int = 123) -> np.ndarray:
    """Create a small windowed synthetic dataset for smoke testing."""
    raw_data, _, _ = load_synthetic_data(
        n_samples=12,
        n_steps=20,
        n_features=3,
        data_type="gbm_jump",
        seed=seed,
    )
    return create_sliding_windows(raw_data, window_size=12, stride=4)


def run_smoke_test(model_name: str) -> None:
    """Run a minimal end-to-end check for one model."""
    data = build_tiny_dataset()
    time_grid = np.linspace(0.0, 1.0, data.shape[1])
    n_generate = min(8, len(data))

    config = get_default_config(model_name)
    config.update(
        {
            "seed": 123,
            "verbose": False,
            "rnn_epochs": 1,
            "rnn_batch_size": 8,
            "transformer_ar_epochs": 1,
            "transformer_ar_batch_size": 8,
            "transformer_ar_max_seq_len": data.shape[1],
        }
    )

    model = get_model(model_name, config)
    model.fit(data, time_grid, verbose=False)
    generated = model.generate(n_samples=n_generate, n_steps=data.shape[1])

    if generated.shape != (n_generate, data.shape[1], data.shape[2]):
        raise RuntimeError(
            f"{model_name} generated wrong shape: {generated.shape}"
        )

    stats = compute_all_metrics_numba(
        data[:n_generate], generated, max_acf_lag=5
    )
    disc = discriminative_score_metrics(
        data[:n_generate], generated, iterations=10
    )
    pred = predictive_score_metrics(
        data[:n_generate], generated, iterations=10
    )

    print(f"[{model_name}] generated shape: {generated.shape}")
    print(
        f"[{model_name}] metrics: "
        f"WD={stats.get('wasserstein_distance', np.nan):.6f}, "
        f"disc={float(disc):.6f}, pred={float(pred):.6f}"
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Smoke test new baseline models")
    parser.add_argument(
        "--model",
        default="all",
        choices=["rnn", "transformer_ar", "all"],
        help="Which baseline to test",
    )
    return parser.parse_args()


def main() -> None:
    """Run the requested smoke tests."""
    args = parse_args()
    targets = ["rnn", "transformer_ar"] if args.model == "all" else [args.model]

    for model_name in targets:
        run_smoke_test(model_name)


if __name__ == "__main__":
    main()
