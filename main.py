"""
JD-SBTS: Jump-Diffusion Schrödinger Bridge Time Series
Main Entry Point with Neural MJD and SOTA Baselines

UPDATED:
- Neural Jumps enabled by default (USE_NEURAL_JUMPS=True)
- Added SOTA baselines: TimeGAN, Diffusion-TS
- Enhanced drift estimation with Numba-accelerated kernel (from SBTS)
- Comprehensive evaluation metrics: Discriminative Score, Predictive Score
- Updated visualization with SOTA comparison
"""

import numpy as np
import os
import torch
import time
import pandas as pd

# Core Modules
from core.solver import euler_maruyama_generator
from core.drift_estimators import KernelDriftEstimator, LSTMDriftEstimator, TransformerDriftEstimator
from core.lightsb import LightSBTrainer 
from core.reference import LocalVolatilityReference, JumpDiffusionReference, NeuralJumpDiffusionReference, create_reference_measure
from core.bandwidth import BandwidthSelector
from core.numba_sb import NumbaMarkovianSB  # NEW: Numba-accelerated SB from SBTS

from models.calibration import VolatilityCalibrator
from models.jumps import JumpDetector
from models.neural_jumps import NeuralJumpDetector, NeuralPointProcess

# SOTA Baselines
from baselines.timegan import TimeGAN
from baselines.diffusion_ts import DiffusionTS

# Evaluation Metrics
from metrics.discriminative_score import discriminative_score_metrics
from metrics.predictive_score import predictive_score_metrics
from metrics.statistical_metrics import compute_all_metrics

# Visualization
from utils.visualization import (
    plot_comprehensive_comparison, 
    plot_performance_metrics,
    plot_volatility_surface,
    plot_bandwidth_optimization, 
    plot_correlation_distribution,
    plot_multi_asset_jumps,
    plot_sota_comparison,  # NEW
    plot_discriminative_results,  # NEW
    set_style,
    _calc_autocorr
)
from utils.data_loader import RealDataLoader, reconstruct_prices
from utils.logger import Logger
from utils.metrics import wasserstein_distance_1d


def main():
    # =========================================================
    # 0. Configuration
    # =========================================================
    CONFIG = {
        # Data
        "USE_REAL_DATA": True,
        "FETCH_SP500": True,  
        "SP500_LIMIT": 50,  # Reduced for faster testing
        "START_DATE": "2020-01-01",
        "END_DATE": "2023-12-31",
        "SEQ_LEN": 60,
        
        # Models to Run
        "MODELS_TO_RUN": [
            "JD-SBTS-Neural",   # Our method with Neural MJD
            "JD-SBTS-Static",   # Our method with Static MJD
            "Numba-SB",         # Numba-accelerated SB (from SBTS)
            "LightSB",          # Light Schrödinger Bridge
            "TimeGAN",          # SOTA: TimeGAN baseline
            "Diffusion-TS",     # SOTA: Diffusion-based TS
        ],
        
        # Physics - Neural Jumps ENABLED by default
        "USE_JUMPS": True,
        "USE_NEURAL_JUMPS": True,  # CHANGED: Now True by default
        "JUMP_THRESHOLD_STD": 4.0,  
        "VOL_BANDWIDTH": 0.5,
        
        # Neural Jump Parameters
        "NEURAL_JUMP_HIDDEN_DIM": 64,
        "NEURAL_JUMP_LR": 0.001,
        "NEURAL_JUMP_EPOCHS": 30,
        "NEURAL_JUMP_SEQ_LEN": 10,
        
        # Drift Estimation Hyperparams
        "USE_CV_FOR_KERNEL": True,
        "BO_N_TRIALS": 10, 
        "KERNEL_TEMP_SCALE": 1.1,
        
        "LSTM_HIDDEN": 128,
        "LSTM_LR": 0.005,
        "LSTM_EPOCHS": 50,
        "LSTM_WEIGHT_DECAY": 1e-3,
        "LSTM_DROPOUT": 0.3,
        "LSTM_TEMP_SCALE": 1.1,
        "LSTM_DRIFT_DAMPENING": 0.9,
        "LSTM_USE_HUBER": True,  # NEW: Huber loss for robustness
        
        # Numba SB (from SBTS)
        "NUMBA_SB_BANDWIDTH": 0.1,
        "NUMBA_SB_MARKOV_ORDER": 3,
        "NUMBA_SB_N_PI": 10,
        
        # LightSB
        "LIGHTSB_COMPONENTS": 20,
        "LIGHTSB_LR": 0.005,
        "LIGHTSB_STEPS": 500,
        "LIGHTSB_MIN_COV": 0.01,
        "LIGHTSB_TEMP_SCALE": 1.1,
        
        # TimeGAN
        "TIMEGAN_HIDDEN": 64,
        "TIMEGAN_EPOCHS": 100,
        "TIMEGAN_BATCH_SIZE": 128,
        
        # Diffusion-TS
        "DIFFUSION_STEPS": 100,
        "DIFFUSION_EPOCHS": 50,
        
        # Generation
        "N_GEN_PATHS": 500, 
        "SYNTHETIC_SAMPLES": 10000,
        
        # Evaluation
        "EVAL_DISCRIMINATIVE": True,
        "EVAL_PREDICTIVE": True,
        "EVAL_ITERATIONS": 1000,
        "EVAL_N_RUNS": 5,
    }

    logger = Logger(base_dir="experiments")
    logger.info("=" * 60)
    logger.info("   JD-SBTS Benchmark: Neural MJD + SOTA Comparison")
    logger.info("=" * 60 + "\n")
    logger.save_config(CONFIG)
    set_style()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[Device] Using: {device}")
    
    # =========================================================
    # 1. Data Loading
    # =========================================================
    if CONFIG["USE_REAL_DATA"]:
        if CONFIG["FETCH_SP500"]:
            tickers = RealDataLoader.get_sp500_tickers(limit=CONFIG["SP500_LIMIT"])
        else:
            tickers = ["SPY", "QQQ", "IWM", "GLD", "TLT"]
        logger.info(f"[Data] Mode: Real Data ({len(tickers)} Assets)")
        loader = RealDataLoader(tickers, CONFIG["START_DATE"], CONFIG["END_DATE"])
        loader.download()
        data, mu, sigma = loader.get_sliding_windows(CONFIG["SEQ_LEN"])
        dt = 1.0 / 252.0 
        steps = CONFIG["SEQ_LEN"]
        time_grid = np.linspace(0, steps * dt, steps)
        n_assets = data.shape[-1]
        logger.info(f"[Data] Shape: {data.shape} (samples, steps, assets)")
    else:
        # Synthetic data for testing
        logger.info("[Data] Mode: Synthetic Data")
        n_samples, steps, n_assets = 1000, CONFIG["SEQ_LEN"], 5
        dt = 1.0 / 252.0
        time_grid = np.linspace(0, steps * dt, steps)
        data = np.random.randn(n_samples, steps, n_assets) * 0.02
        tickers = [f"Asset_{i}" for i in range(n_assets)]

    # =========================================================
    # 2. Calibration (Neural MJD + Volatility)
    # =========================================================
    logger.info("\n" + "=" * 40)
    logger.info("[Phase 2] Calibration")
    logger.info("=" * 40)
    
    # Initialize detectors
    static_jump_detector = JumpDetector(dt=dt, threshold_multiplier=CONFIG["JUMP_THRESHOLD_STD"])
    neural_jump_detector = None
    
    if CONFIG["USE_JUMPS"]:
        logger.info("\n[Jumps] Calibrating jump parameters...")
        
        # Always fit static detector first
        static_jump_detector.fit(data)
        
        # Fit neural detector if enabled
        if CONFIG["USE_NEURAL_JUMPS"]:
            logger.info("   [Neural MJD] Training intensity network...")
            neural_jump_detector = NeuralJumpDetector(
                dt=dt,
                input_dim=n_assets,
                hidden_dim=CONFIG["NEURAL_JUMP_HIDDEN_DIM"],
                threshold_multiplier=CONFIG["JUMP_THRESHOLD_STD"],
                lr=CONFIG["NEURAL_JUMP_LR"],
                epochs=CONFIG["NEURAL_JUMP_EPOCHS"],
                device=device
            )
            neural_jump_detector.fit(data, seq_len=CONFIG["NEURAL_JUMP_SEQ_LEN"])
        
        # Use purified returns for volatility calibration
        data_for_vol = static_jump_detector.get_purified_returns()
        
        # Visualization
        if CONFIG["USE_REAL_DATA"] and hasattr(loader, 'log_returns') and loader.log_returns is not None:
            logger.info("   [Vis] Plotting jump visualizations...")
            from utils.visualization import plot_jumps_on_price, plot_zoomed_crisis
            cont_dates = pd.to_datetime(loader.raw_prices_df.index[1:])
            plot_jumps_on_price(
                cont_dates, loader.log_returns, tickers, 
                threshold_std=CONFIG["JUMP_THRESHOLD_STD"],
                save_path=logger.get_save_path("jumps_on_price.png")
            )
    else:
        logger.info("\n[Jumps] Jump detection disabled.")
        data_for_vol = data.copy()

    # Volatility Calibration
    logger.info("\n[Volatility] Fitting Local Volatility Surface...")
    vol_calibrator = VolatilityCalibrator(dt=dt, method='kernel', bandwidth=CONFIG["VOL_BANDWIDTH"])
    vol_calibrator.fit(data, purified_trajectories=data_for_vol)
    
    # Log surface diagnostics
    diag = vol_calibrator.get_surface_diagnostics()
    if diag:
        logger.info(f"   [Surface] Shape: {'Smile/Skew ✓' if diag.get('is_smile_shape') else 'Inverted U ✗'}")
    
    # =========================================================
    # 3. Model Training & Generation
    # =========================================================
    logger.info("\n" + "=" * 40)
    logger.info("[Phase 3] Model Training & Generation")
    logger.info("=" * 40)
    
    results_store = {}
    metrics_store = {}
    
    n_test = min(CONFIG["N_GEN_PATHS"], len(data))
    test_idx = np.random.choice(len(data), n_test, replace=False)
    test_x0 = data[test_idx, 0, :] 
    real_paths_subset = data[test_idx]
    
    for model_name in CONFIG["MODELS_TO_RUN"]:
        logger.info(f"\n>>> Training: {model_name}")
        metrics_store[model_name] = {}
        
        try:
            start_t = time.time()
            gen_paths = None
            
            # ============================================
            # JD-SBTS with Neural MJD (Our Main Method)
            # ============================================
            if model_name == "JD-SBTS-Neural":
                # Reference measure with Neural MJD
                ref_proc = create_reference_measure(
                    vol_calibrator=vol_calibrator,
                    jump_detector=neural_jump_detector if neural_jump_detector else static_jump_detector,
                    use_neural_jumps=True,
                    volatility_multiplier=CONFIG["LSTM_TEMP_SCALE"]
                )
                logger.info(f"   [Ref] {ref_proc.get_type()}")
                
                # LSTM Drift Estimator with Huber loss
                estimator = LSTMDriftEstimator(
                    input_dim=n_assets, 
                    hidden_size=CONFIG["LSTM_HIDDEN"],
                    lr=CONFIG["LSTM_LR"], 
                    epochs=CONFIG["LSTM_EPOCHS"], 
                    dt=dt,
                    weight_decay=CONFIG["LSTM_WEIGHT_DECAY"], 
                    dropout=CONFIG["LSTM_DROPOUT"],
                    use_huber_loss=CONFIG["LSTM_USE_HUBER"]
                )
                estimator.fit(data, dt)
                drift_fn = lambda t, x, est=estimator: est.predict(t, x) * CONFIG["LSTM_DRIFT_DAMPENING"]
                
                metrics_store[model_name]['train_time'] = time.time() - start_t
                
                # Generation
                logger.info(f"   [Gen] Simulating {n_test} paths...")
                start_t = time.time()
                gen_paths = euler_maruyama_generator(test_x0, time_grid, drift_fn, ref_proc, n_paths=n_test)
                metrics_store[model_name]['gen_time'] = time.time() - start_t
            
            # ============================================
            # JD-SBTS with Static MJD
            # ============================================
            elif model_name == "JD-SBTS-Static":
                ref_proc = create_reference_measure(
                    vol_calibrator=vol_calibrator,
                    jump_detector=static_jump_detector,
                    use_neural_jumps=False,
                    volatility_multiplier=CONFIG["LSTM_TEMP_SCALE"]
                )
                logger.info(f"   [Ref] {ref_proc.get_type()}")
                
                estimator = LSTMDriftEstimator(
                    input_dim=n_assets, 
                    hidden_size=CONFIG["LSTM_HIDDEN"],
                    lr=CONFIG["LSTM_LR"], 
                    epochs=CONFIG["LSTM_EPOCHS"], 
                    dt=dt,
                    weight_decay=CONFIG["LSTM_WEIGHT_DECAY"], 
                    dropout=CONFIG["LSTM_DROPOUT"]
                )
                estimator.fit(data, dt)
                drift_fn = lambda t, x, est=estimator: est.predict(t, x) * CONFIG["LSTM_DRIFT_DAMPENING"]
                
                metrics_store[model_name]['train_time'] = time.time() - start_t
                
                logger.info(f"   [Gen] Simulating {n_test} paths...")
                start_t = time.time()
                gen_paths = euler_maruyama_generator(test_x0, time_grid, drift_fn, ref_proc, n_paths=n_test)
                metrics_store[model_name]['gen_time'] = time.time() - start_t
            
            # ============================================
            # Numba-accelerated SB (from SBTS)
            # ============================================
            elif model_name == "Numba-SB":
                logger.info("   [Numba-SB] Using Numba-accelerated kernel regression...")
                numba_sb = NumbaMarkovianSB(
                    bandwidth=CONFIG["NUMBA_SB_BANDWIDTH"],
                    markov_order=CONFIG["NUMBA_SB_MARKOV_ORDER"],
                    n_pi=CONFIG["NUMBA_SB_N_PI"],
                    dt=dt
                )
                
                metrics_store[model_name]['train_time'] = time.time() - start_t
                
                logger.info(f"   [Gen] Simulating {n_test} paths...")
                start_t = time.time()
                gen_paths = numba_sb.generate(data, n_samples=n_test)
                metrics_store[model_name]['gen_time'] = time.time() - start_t
            
            # ============================================
            # LightSB
            # ============================================
            elif model_name == "LightSB":
                ref_proc = LocalVolatilityReference(vol_calibrator, volatility_multiplier=CONFIG["LIGHTSB_TEMP_SCALE"])
                
                X_curr = data[:, :-1, :].reshape(-1, n_assets)
                X_next = data[:, 1:, :].reshape(-1, n_assets)
                x0_t = torch.tensor(X_curr, dtype=torch.float32)
                x1_t = torch.tensor(X_next, dtype=torch.float32)
                
                lsb = LightSBTrainer(
                    dim=n_assets, 
                    n_components=CONFIG["LIGHTSB_COMPONENTS"], 
                    lr=CONFIG["LIGHTSB_LR"], 
                    min_cov=CONFIG["LIGHTSB_MIN_COV"]
                )
                logger.info("   [LightSB] Training GMM...")
                for i in range(CONFIG["LIGHTSB_STEPS"]):
                    idx = np.random.choice(len(x0_t), min(1024, len(x0_t)))
                    lsb.train_step(x0_t[idx], x1_t[idx])
                    
                drift_fn = lambda t, x, lsb=lsb: lsb.get_drift(t, x, clip_val=5.0)
                
                metrics_store[model_name]['train_time'] = time.time() - start_t
                
                logger.info(f"   [Gen] Simulating {n_test} paths...")
                start_t = time.time()
                gen_paths = euler_maruyama_generator(test_x0, time_grid, drift_fn, ref_proc, n_paths=n_test)
                metrics_store[model_name]['gen_time'] = time.time() - start_t
            
            # ============================================
            # TimeGAN (SOTA Baseline)
            # ============================================
            elif model_name == "TimeGAN":
                logger.info("   [TimeGAN] Training GAN...")
                timegan = TimeGAN(
                    seq_len=steps,
                    n_features=n_assets,
                    hidden_dim=CONFIG["TIMEGAN_HIDDEN"],
                    device=device
                )
                timegan.fit(data, epochs=CONFIG["TIMEGAN_EPOCHS"], batch_size=CONFIG["TIMEGAN_BATCH_SIZE"])
                
                metrics_store[model_name]['train_time'] = time.time() - start_t
                
                logger.info(f"   [Gen] Generating {n_test} samples...")
                start_t = time.time()
                gen_paths = timegan.generate(n_test)
                metrics_store[model_name]['gen_time'] = time.time() - start_t
            
            # ============================================
            # Diffusion-TS (SOTA Baseline)
            # ============================================
            elif model_name == "Diffusion-TS":
                logger.info("   [Diffusion-TS] Training diffusion model...")
                diffusion = DiffusionTS(
                    seq_len=steps,
                    n_features=n_assets,
                    n_steps=CONFIG["DIFFUSION_STEPS"],
                    device=device
                )
                diffusion.fit(data, epochs=CONFIG["DIFFUSION_EPOCHS"])
                
                metrics_store[model_name]['train_time'] = time.time() - start_t
                
                logger.info(f"   [Gen] Generating {n_test} samples...")
                start_t = time.time()
                gen_paths = diffusion.generate(n_test)
                metrics_store[model_name]['gen_time'] = time.time() - start_t
            
            else:
                logger.info(f"   [Skip] Unknown model: {model_name}")
                continue
            
            # Store results
            if gen_paths is not None:
                results_store[model_name] = gen_paths
                
                # Basic metrics
                if not np.isnan(gen_paths).any():
                    wd = np.mean([wasserstein_distance_1d(real_paths_subset[:, -1, i], gen_paths[:, -1, i]) 
                                  for i in range(min(5, n_assets))])
                    
                    acf_real = _calc_autocorr(real_paths_subset)
                    acf_gen = _calc_autocorr(gen_paths)
                    acf_mse = np.mean((acf_real - acf_gen)**2)
                    
                    metrics_store[model_name]['WD'] = wd
                    metrics_store[model_name]['ACF_MSE'] = acf_mse
                    logger.info(f"   [Metrics] WD: {wd:.4f}, ACF_MSE: {acf_mse:.6f}")
                else:
                    metrics_store[model_name]['WD'] = np.nan
                    metrics_store[model_name]['ACF_MSE'] = np.nan
                    logger.info("   [Warning] Generated paths contain NaN")
                    
        except Exception as e:
            logger.info(f"   [Error] {model_name} failed: {str(e)}")
            metrics_store[model_name]['train_time'] = np.nan
            metrics_store[model_name]['gen_time'] = np.nan
            metrics_store[model_name]['WD'] = np.nan
            metrics_store[model_name]['ACF_MSE'] = np.nan

    # =========================================================
    # 4. Advanced Evaluation (Discriminative & Predictive)
    # =========================================================
    logger.info("\n" + "=" * 40)
    logger.info("[Phase 4] Advanced Evaluation")
    logger.info("=" * 40)
    
    if CONFIG["EVAL_DISCRIMINATIVE"] and len(results_store) > 0:
        logger.info("\n[Eval] Computing Discriminative Scores...")
        for model_name, gen_paths in results_store.items():
            if np.isnan(gen_paths).any():
                continue
            try:
                real_tensor = torch.tensor(real_paths_subset, dtype=torch.float32).to(device)
                gen_tensor = torch.tensor(gen_paths, dtype=torch.float32).to(device)
                
                disc_score = discriminative_score_metrics(
                    real_tensor, gen_tensor, 
                    iterations=CONFIG["EVAL_ITERATIONS"],
                    device=device
                )
                metrics_store[model_name]['Disc_Score'] = disc_score
                logger.info(f"   [{model_name}] Discriminative Score: {disc_score:.4f}")
            except Exception as e:
                logger.info(f"   [{model_name}] Discriminative eval failed: {str(e)}")
                metrics_store[model_name]['Disc_Score'] = np.nan
    
    if CONFIG["EVAL_PREDICTIVE"] and len(results_store) > 0:
        logger.info("\n[Eval] Computing Predictive Scores...")
        for model_name, gen_paths in results_store.items():
            if np.isnan(gen_paths).any():
                continue
            try:
                real_tensor = torch.tensor(real_paths_subset, dtype=torch.float32).to(device)
                gen_tensor = torch.tensor(gen_paths, dtype=torch.float32).to(device)
                
                pred_score = predictive_score_metrics(
                    real_tensor, gen_tensor,
                    col_pred=n_assets - 1,
                    iterations=CONFIG["EVAL_ITERATIONS"],
                    device=device
                )
                metrics_store[model_name]['Pred_Score'] = pred_score
                logger.info(f"   [{model_name}] Predictive Score: {pred_score:.4f}")
            except Exception as e:
                logger.info(f"   [{model_name}] Predictive eval failed: {str(e)}")
                metrics_store[model_name]['Pred_Score'] = np.nan

    # =========================================================
    # 5. Visualization & Reporting
    # =========================================================
    logger.info("\n" + "=" * 40)
    logger.info("[Phase 5] Visualization & Reporting")
    logger.info("=" * 40)
    
    # A. Performance Summary Table
    df_metrics = pd.DataFrame(metrics_store).T
    logger.info("\n--- Performance Summary ---")
    print(df_metrics.to_string())
    df_metrics.to_csv(logger.get_save_path("metrics_summary.csv"))
    
    # B. Performance Bar Charts
    plot_performance_metrics(df_metrics, save_path=logger.get_save_path("performance_metrics.png"))
    
    # C. SOTA Comparison Plot
    if len(results_store) > 1:
        plot_sota_comparison(
            real_paths_subset,
            results_store,
            metrics_store,
            save_path=logger.get_save_path("sota_comparison.png")
        )
    
    # D. Price Reconstruction
    S0 = 100.0
    real_prices = reconstruct_prices(S0, real_paths_subset)
    gen_prices_dict = {name: reconstruct_prices(S0, paths) for name, paths in results_store.items()}
    
    # E. Comprehensive Comparison
    plot_comprehensive_comparison(
        time_grid, 
        real_paths_subset, 
        results_store,
        real_prices=real_prices,
        gen_prices_dict=gen_prices_dict,
        save_path=logger.get_save_path("final_comparison.png")
    )
    
    # F. Correlation Distribution
    if n_assets > 1 and len(results_store) > 0:
        plot_correlation_distribution(
            real_paths_subset,
            results_store,
            save_path=logger.get_save_path("correlation_dist.png")
        )

    # G. Volatility Surface
    plot_volatility_surface(
        vol_calibrator, 
        data_range=data.flatten(),
        T=dt * CONFIG["SEQ_LEN"],
        save_path=logger.get_save_path("volatility_surface.png")
    )
    
    logger.info(f"\n[Done] Full Benchmark Complete. Results in: {logger.run_dir}")
    
    return df_metrics


if __name__ == "__main__":
    main()
