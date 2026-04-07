#!/usr/bin/env python3
"""
JD-SBTS: Jump-Diffusion Schrödinger Bridge for Time Series Generation

Main entry point for running experiments and benchmarks.

FEATURES:
- JD-SBTS-F: Feedback mechanism with Jump-Volatility Interaction
- Neural MJD: Time-varying jump intensity prediction
- Numba acceleration for CPU-intensive computations
- OOP architecture with Factory/Strategy patterns
- Comprehensive evaluation metrics
- Publication-quality visualizations

Usage:
    python main.py                          # Run with default config
    python main.py --config config.json     # Run with custom config
    python main.py --model jd_sbts_f        # Run specific model
    python main.py --model rnn              # Run RNN baseline
    python main.py --model transformer_ar   # Run causal Transformer baseline
    python main.py --benchmark              # Run full benchmark

Author: Manus AI
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================
# Default Configuration
# ============================================

DEFAULT_CONFIG = {
    # Experiment settings
    'experiment_name': 'jdsbts_experiment',
    'seed': 42,
    'verbose': True,
    
    # Data settings
    'data_source': 'etf',  # 'etf' or 'synthetic'
    'tickers': ['SPY', 'QQQ', 'IWM', 'EEM', 'GLD'],
    'start_date': '2020-01-01',
    'end_date': '2024-01-01',
    'window_size': 60,
    'stride': 10,
    'fallback_to_synthetic_on_data_error': False,
    
    # Synthetic data settings
    'synthetic_type': 'gbm_jump',
    'n_synthetic_samples': 500,
    'synthetic_total_steps': 240,
    'synthetic_return_type': 'log_returns',
    
    # Model settings
    'models_to_run': ['jd_sbts', 'jd_sbts_f'],
    
    # JD-SBTS settings
    'use_neural_jumps': False,
    'use_feedback': True,
    'jump_threshold_std': 4.0,
    'feedback_kappa': 5.0,
    'feedback_gamma': 0.5,
    
    # Drift estimation
    'drift_estimator': 'lstm',
    'lstm_hidden': 128,
    'lstm_epochs': 50,
    'lstm_lr': 0.005,
    'lstm_dropout': 0.3,
    
    # Generation settings
    'n_generate': 100,
    'n_steps': None,  # None = same as training
    
    # Evaluation settings
    'run_discriminative': True,
    'run_predictive': True,
    'discriminative_iterations': 500,
    'predictive_iterations': 500,
    'acf_max_lag': 15,
    
    # Output settings
    'output_dir': 'experiments',
    'save_models': True,
    'save_plots': True,
}

CLI_MODEL_CHOICES = [
    'jd_sbts',
    'jd_sbts_f',
    'jd_sbts_neural',
    'jd_sbts_f_neural',
    'lightsb',
    'numba_sb',
    'timegan',
    'diffusion_ts',
    'rnn',
    'transformer_ar',
    'sbts',
    'sbts_f',
    'sbts_neural',
    'sbts_f_neural',
    'light_sb',
    'numbasb',
    'time_gan',
    'diffusion',
    'rnn_baseline',
    'transformer',
    'ar_transformer',
]


# ============================================
# Import Handlers (with fallbacks)
# ============================================

def import_new_modules():
    """Import new modular architecture."""
    try:
        from models.factory import get_model, list_models, get_default_config, create_model_comparison
        from data.loaders import load_etf_data, load_synthetic_data, create_sliding_windows
        from utils.experiment_manager import ExperimentManager
        from metrics import compute_all_metrics, discriminative_score_metrics, predictive_score_metrics
        from metrics.numba_metrics import compute_all_metrics_numba, compute_stylized_facts_numba
        
        return {
            'get_model': get_model,
            'list_models': list_models,
            'get_default_config': get_default_config,
            'create_model_comparison': create_model_comparison,
            'load_etf_data': load_etf_data,
            'load_synthetic_data': load_synthetic_data,
            'create_sliding_windows': create_sliding_windows,
            'ExperimentManager': ExperimentManager,
            'compute_all_metrics': compute_all_metrics,
            'discriminative_score_metrics': discriminative_score_metrics,
            'predictive_score_metrics': predictive_score_metrics,
            'compute_all_metrics_numba': compute_all_metrics_numba,
            'compute_stylized_facts_numba': compute_stylized_facts_numba,
        }
    except ImportError as e:
        warnings.warn(f"New modules not available: {e}")
        return None


def import_visualization():
    """Import visualization modules."""
    try:
        from visualization import (
            set_style, plot_performance_metrics, plot_comprehensive_comparison,
            plot_correlation_distribution, plot_volatility_surface,
            plot_stress_factor_dynamics, plot_feedback_comparison,
            plot_model_comparison_grid, create_legend_figure
        )
        return {
            'set_style': set_style,
            'plot_performance_metrics': plot_performance_metrics,
            'plot_comprehensive_comparison': plot_comprehensive_comparison,
            'plot_correlation_distribution': plot_correlation_distribution,
            'plot_volatility_surface': plot_volatility_surface,
            'plot_stress_factor_dynamics': plot_stress_factor_dynamics,
            'plot_feedback_comparison': plot_feedback_comparison,
            'plot_model_comparison_grid': plot_model_comparison_grid,
            'create_legend_figure': create_legend_figure,
        }
    except ImportError as e:
        warnings.warn(f"Visualization not fully available: {e}")
        return None


def _extract_metric_value(result: Any, key: str, default: float) -> float:
    """Normalize metric outputs that may be dicts or scalar values."""
    if isinstance(result, dict):
        return float(result.get(key, default))
    if np.isscalar(result):
        return float(result)
    return float(default)


def _match_eval_shapes(
    real_data: np.ndarray,
    gen_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Trim real and generated batches to a shared evaluation size."""
    n_eval = min(len(real_data), len(gen_data))
    if n_eval <= 0:
        raise ValueError("Need at least one sample for evaluation")
    return real_data[:n_eval], gen_data[:n_eval]


# ============================================
# Main Experiment Runner (New Architecture)
# ============================================

def run_experiment_new(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run experiment using new modular architecture.
    
    Args:
        config: Experiment configuration
    
    Returns:
        Dictionary of results
    """
    modules = import_new_modules()
    viz = import_visualization()
    
    if modules is None:
        raise ImportError("Model modules not available in the unified architecture.")
    
    # Set random seed
    np.random.seed(config.get('seed', 42))
    
    verbose = config.get('verbose', True)
    
    # Initialize experiment manager
    models_to_run = config.get('models_to_run', ['jd_sbts'])
    if len(models_to_run) == 1:
        run_label = models_to_run[0]
    else:
        run_label = config.get('experiment_name', 'multi_model')

    exp_manager = modules['ExperimentManager'](
        model_name=run_label,
        base_dir=config.get('output_dir', 'experiments'),
        config=config,
        verbose=verbose
    )
    
    if verbose:
        print("=" * 70)
        print("JD-SBTS Experiment (New Architecture)")
        print("=" * 70)
        print(f"Experiment: {exp_manager.run_name}")
        print(f"Output dir: {exp_manager.run_dir}")
        print()
    
    # ========================================
    # Load Data
    # ========================================
    if verbose:
        print("[Phase 1] Loading Data...")
    
    data_source = config.get('data_source', 'etf')
    
    if data_source == 'etf':
        try:
            raw_data, time_grid, metadata = modules['load_etf_data'](
                tickers=config.get('tickers', ['SPY', 'QQQ', 'IWM', 'EEM', 'GLD']),
                start_date=config.get('start_date', '2020-01-01'),
                end_date=config.get('end_date', '2024-01-01'),
                return_type='log_returns'
            )
        except Exception as e:
            if config.get('fallback_to_synthetic_on_data_error', False):
                warnings.warn(f"Failed to load ETF data: {e}. Using synthetic data.")
                data_source = 'synthetic'
            else:
                raise RuntimeError(
                    "ETF data loading failed. "
                    "Set 'fallback_to_synthetic_on_data_error': True if you want "
                    "the experiment to continue on synthetic data."
                ) from e
    
    if data_source == 'synthetic':
        synthetic_total_steps = config.get(
            'synthetic_total_steps',
            max(config.get('window_size', 60) * 4, config.get('window_size', 60) + 1)
        )
        raw_data, time_grid, metadata = modules['load_synthetic_data'](
            n_samples=config.get('n_synthetic_samples', 500),
            n_steps=synthetic_total_steps,
            n_features=len(config.get('tickers', ['SPY', 'QQQ', 'IWM', 'EEM', 'GLD'])),
            data_type=config.get('synthetic_type', 'gbm_jump'),
            seed=config.get('seed', 42),
            return_type=config.get('synthetic_return_type', 'log_returns'),
        )
    
    # Create sliding windows
    window_size = config.get('window_size', 60)
    stride = config.get('stride', 10)
    
    if raw_data.shape[1] > window_size:
        data = modules['create_sliding_windows'](raw_data, window_size, stride)
    else:
        data = raw_data
    
    # Update time grid
    time_grid = np.linspace(0, 1, data.shape[1])
    
    if verbose:
        print(f"  Data shape: {data.shape}")
        print(f"  Time grid: {len(time_grid)} steps")
    
    exp_manager.log(f"Data loaded: shape={data.shape}")
    
    # ========================================
    # Train Models
    # ========================================
    if verbose:
        print("\n[Phase 2] Training Models...")
    
    trained_models = {}
    training_times = {}
    
    for model_name in models_to_run:
        if verbose:
            print(f"\n  Training {model_name}...")
        
        try:
            # Get model config
            model_config = modules['get_default_config'](model_name)
            model_config.update(config)  # Override with experiment config
            
            # Create model
            model = modules['get_model'](model_name, model_config)
            
            # Train
            start_time = time.perf_counter()
            model.fit(data, time_grid, verbose=verbose)
            train_time = time.perf_counter() - start_time
            
            trained_models[model_name] = model
            training_times[model_name] = train_time
            
            exp_manager.log(f"{model_name} trained in {train_time:.2f}s")
            
            if verbose:
                print(f"  {model_name} training time: {train_time:.2f}s")
        
        except Exception as e:
            warnings.warn(f"Failed to train {model_name}: {e}")
            exp_manager.log(f"ERROR: {model_name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # ========================================
    # Generate Samples
    # ========================================
    if verbose:
        print("\n[Phase 3] Generating Samples...")
    
    n_generate = config.get('n_generate', 100)
    n_steps = config.get('n_steps') or data.shape[1]
    
    generated_data = {}
    generation_times = {}
    stress_trajectories = {}
    
    for model_name, model in trained_models.items():
        if verbose:
            print(f"\n  Generating from {model_name}...")
        
        try:
            start_time = time.perf_counter()
            
            # Check if model supports stress factor output
            if hasattr(model, 'use_feedback') and model.use_feedback:
                gen_paths, stress = model.generate(
                    n_samples=n_generate,
                    n_steps=n_steps,
                    return_stress=True
                )
                stress_trajectories[model_name] = stress
            else:
                gen_paths = model.generate(
                    n_samples=n_generate,
                    n_steps=n_steps
                )
            
            gen_time = time.perf_counter() - start_time
            
            generated_data[model_name] = gen_paths
            generation_times[model_name] = gen_time
            
            exp_manager.log(f"{model_name} generated {n_generate} samples in {gen_time:.2f}s")
            
            if verbose:
                print(f"  {model_name} generation time: {gen_time:.2f}s")
                print(f"  Generated shape: {gen_paths.shape}")
        
        except Exception as e:
            warnings.warn(f"Failed to generate from {model_name}: {e}")
            exp_manager.log(f"ERROR: {model_name} generation failed: {e}")
            import traceback
            traceback.print_exc()
    
    # ========================================
    # Evaluate Models
    # ========================================
    if verbose:
        print("\n[Phase 4] Evaluating Models...")
    
    metrics_results = {}
    
    for model_name, gen_data in generated_data.items():
        if verbose:
            print(f"\n  Evaluating {model_name}...")
        
        try:
            real_eval, gen_eval = _match_eval_shapes(data, gen_data)

            # Statistical metrics (Numba-accelerated)
            stats = modules['compute_all_metrics_numba'](
                real_eval,
                gen_eval,
                max_acf_lag=config.get('acf_max_lag', 15)
            )
            
            # Stylized facts
            gen_returns = np.diff(gen_data, axis=1)
            if gen_returns.ndim == 3:
                gen_returns = gen_returns[:, :, 0]
            stylized = modules['compute_stylized_facts_numba'](gen_returns.flatten())
            
            metrics = {
                'train_time': training_times.get(model_name, 0),
                'gen_time': generation_times.get(model_name, 0),
                **stats,
                **{f'stylized_{k}': v for k, v in stylized.items()}
            }
            
            # Discriminative score (optional)
            if config.get('run_discriminative', True):
                try:
                    disc_result = modules['discriminative_score_metrics'](
                        real_eval,
                        gen_eval,
                        iterations=config.get('discriminative_iterations', 500),
                    )
                    metrics['discriminative_score'] = _extract_metric_value(
                        disc_result,
                        'discriminative_score',
                        0.5
                    )
                except Exception as e:
                    warnings.warn(f"Discriminative score failed: {e}")
                    metrics['discriminative_score'] = 0.5
            
            # Predictive score (optional)
            if config.get('run_predictive', True):
                try:
                    pred_result = modules['predictive_score_metrics'](
                        real_eval,
                        gen_eval,
                        iterations=config.get('predictive_iterations', 500),
                    )
                    metrics['predictive_score'] = _extract_metric_value(
                        pred_result,
                        'predictive_score',
                        0.0
                    )
                except Exception as e:
                    warnings.warn(f"Predictive score failed: {e}")
                    metrics['predictive_score'] = 0
            
            metrics_results[model_name] = metrics
            
            if verbose:
                print(f"    WD: {metrics.get('wasserstein_distance', 0):.6f}")
                print(f"    ACF MSE: {metrics.get('acf_mse', 0):.6f}")
                if 'discriminative_score' in metrics:
                    print(f"    Disc Score: {metrics['discriminative_score']:.4f}")
        
        except Exception as e:
            warnings.warn(f"Evaluation failed for {model_name}: {e}")
            exp_manager.log(f"ERROR: {model_name} evaluation failed: {e}")
    
    # ========================================
    # Save Results
    # ========================================
    if verbose:
        print("\n[Phase 5] Saving Results...")
    
    # Save metrics
    exp_manager.save_metrics(metrics_results)
    
    # Save models
    if config.get('save_models', True):
        for model_name, model in trained_models.items():
            exp_manager.save_model(model, model_name)
    
    # ========================================
    # Generate Plots
    # ========================================
    if config.get('save_plots', True) and viz is not None:
        if verbose:
            print("\n[Phase 6] Generating Plots...")
        
        try:
            # Create metrics DataFrame
            import pandas as pd
            df_metrics = pd.DataFrame(metrics_results).T
            df_metrics.index.name = 'Model'
            
            # Model comparison grid
            viz['plot_model_comparison_grid'](
                data[:min(n_generate, len(data))],
                generated_data,
                metrics_results,
                save_path=os.path.join(str(exp_manager.run_dir), 'model_comparison.png')
            )
            
            # Feedback analysis (if applicable)
            for model_name, stress in stress_trajectories.items():
                viz['plot_stress_factor_dynamics'](
                    time_grid,
                    stress,
                    save_path=os.path.join(str(exp_manager.run_dir), f'{model_name}_stress.png')
                )
            
            # Legend
            viz['create_legend_figure'](
                save_path=os.path.join(str(exp_manager.run_dir), 'legend.png')
            )
            
        except Exception as e:
            warnings.warn(f"Plot generation failed: {e}")
            exp_manager.log(f"ERROR: Plot generation failed: {e}")
    
    # ========================================
    # Summary
    # ========================================
    if verbose:
        print("\n" + "=" * 70)
        print("Experiment Complete!")
        print("=" * 70)
        print(f"\nResults saved to: {str(exp_manager.run_dir)}")
        print("\nMetrics Summary:")
        for model_name, metrics in metrics_results.items():
            print(f"\n  {model_name}:")
            print(f"    Training Time: {metrics.get('train_time', 0):.2f}s")
            print(f"    Generation Time: {metrics.get('gen_time', 0):.2f}s")
            print(f"    Wasserstein Distance: {metrics.get('wasserstein_distance', 0):.6f}")
            print(f"    ACF MSE: {metrics.get('acf_mse', 0):.6f}")
    
    return {
        'config': config,
        'metrics': metrics_results,
        'generated_data': generated_data,
        'stress_trajectories': stress_trajectories,
        'experiment_dir': str(exp_manager.run_dir)
    }


def run_benchmark(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run full benchmark comparing all available models.
    
    Args:
        config: Base configuration
    
    Returns:
        Benchmark results
    """
    modules = import_new_modules()
    if modules is None:
        raise ImportError("Model modules not available in the unified architecture.")

    # Keep benchmark coverage aligned with the active model registry.
    all_models = list(modules['list_models']().keys())

    config['models_to_run'] = all_models
    config['experiment_name'] = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return run_experiment_new(config)


# ============================================
# CLI Interface
# ============================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='JD-SBTS: Jump-Diffusion Schrödinger Bridge for Time Series'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to JSON configuration file'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        choices=CLI_MODEL_CHOICES,
        help='Specific model to run (e.g., jd_sbts_f, rnn, transformer_ar)'
    )
    
    parser.add_argument(
        '--benchmark', '-b',
        action='store_true',
        help='Run full benchmark with all models'
    )
    
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available models and exit'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='experiments',
        help='Output directory'
    )
    
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress verbose output'
    )
    
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Use synthetic data instead of ETF data'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # List models and exit
    if args.list_models:
        modules = import_new_modules()
        if modules:
            print("\nAvailable Models:")
            print("-" * 50)
            for name, desc in modules['list_models']().items():
                print(f"  {name}: {desc}")
            print()
        else:
            print("Model modules not available in the unified architecture.")
        return
    
    # Load configuration
    config = DEFAULT_CONFIG.copy()
    
    if args.config:
        with open(args.config, 'r') as f:
            user_config = json.load(f)
        config.update(user_config)
    
    # Override with CLI arguments
    if args.model:
        config['models_to_run'] = [args.model]
    
    if args.output:
        config['output_dir'] = args.output
    
    if args.seed:
        config['seed'] = args.seed
    
    if args.quiet:
        config['verbose'] = False
    
    if args.synthetic:
        config['data_source'] = 'synthetic'
    
    # Run experiment or benchmark
    try:
        if args.benchmark:
            results = run_benchmark(config)
        else:
            results = run_experiment_new(config)
        
        print(f"\nExperiment completed. Results saved to: {results['experiment_dir']}")
    
    except ImportError as e:
        print(f"\nNew architecture not available: {e}")
        print("Install the missing dependencies and rerun the unified main pipeline.")


if __name__ == '__main__':
    main()
