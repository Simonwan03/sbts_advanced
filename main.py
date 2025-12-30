import numpy as np
import os
import torch
import time
import pandas as pd

# Core Modules
from core.solver import euler_maruyama_generator
from core.drift_estimators import KernelDriftEstimator, LSTMDriftEstimator
from core.lightsb import LightSBTrainer 
from core.reference import LocalVolatilityReference, JumpDiffusionReference
from core.bandwidth import BandwidthSelector

from models.calibration import VolatilityCalibrator
from models.jumps import JumpDetector

# Visualization
from utils.visualization import (
    plot_comprehensive_comparison, 
    plot_performance_metrics, # NEW
    plot_volatility_surface,
    plot_bandwidth_optimization, 
    plot_correlation_distribution,
    plot_multi_asset_jumps,
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
        "USE_REAL_DATA": True,
        "FETCH_SP500": True,  
        "SP500_LIMIT": 100,
        "START_DATE": "2020-01-01",
        "END_DATE": "2023-12-31",
        "SEQ_LEN": 60,
        
        # Models
        "MODELS_TO_RUN": ["Kernel", "SBTS-LSTM", "LightSB"], 
        
        # Physics
        "USE_JUMPS": True,          
        "JUMP_THRESHOLD_STD": 4.0,  
        "VOL_BANDWIDTH": 0.5,
        
        # Hyperparams
        "USE_CV_FOR_KERNEL": True,
        "BO_N_TRIALS": 15, 
        "KERNEL_TEMP_SCALE": 1.1,
        
        "LSTM_HIDDEN": 128,
        "LSTM_LR": 0.005,
        "LSTM_EPOCHS": 80,
        "LSTM_WEIGHT_DECAY": 1e-3,
        "LSTM_DROPOUT": 0.3,
        "LSTM_TEMP_SCALE": 1.1,
        "LSTM_DRIFT_DAMPENING": 0.9, 
        
        "LIGHTSB_COMPONENTS": 20,
        "LIGHTSB_LR": 0.005,
        "LIGHTSB_STEPS": 1000,
        "LIGHTSB_MIN_COV": 0.01,
        "LIGHTSB_TEMP_SCALE": 1.1,
        
        "N_GEN_PATHS": 500, 
        "SYNTHETIC_SAMPLES": 10000
    }

    logger = Logger(base_dir="experiments")
    logger.info("==========================================")
    logger.info("   SBTS Benchmark: Comparison Report      ")
    logger.info("==========================================\n")
    logger.save_config(CONFIG)
    set_style()
    
    # =========================================================
    # 1. Data Loading
    # =========================================================
    if CONFIG["USE_REAL_DATA"]:
        if CONFIG["FETCH_SP500"]:
            tickers = RealDataLoader.get_sp500_tickers(limit=CONFIG["SP500_LIMIT"])
        else:
            tickers = ["SPY", "QQQ", "IWM", "GLD"]
        logger.info(f"[Data] Mode: Real Data ({len(tickers)} Assets)")
        loader = RealDataLoader(tickers, CONFIG["START_DATE"], CONFIG["END_DATE"])
        loader.download()
        data, mu, sigma = loader.get_sliding_windows(CONFIG["SEQ_LEN"])
        dt = 1.0 / 252.0 
        steps = CONFIG["SEQ_LEN"]
        time_grid = np.linspace(0, steps * dt, steps)
        n_assets = data.shape[-1]
    else:
        # Synthetic logic (simplified)
        pass

    # =========================================================
    # 2. Calibration
    # =========================================================
    jump_detector = JumpDetector(dt=dt, threshold_multiplier=CONFIG["JUMP_THRESHOLD_STD"])
    data_for_vol = data.copy()

    if CONFIG["USE_JUMPS"]:
        logger.info("\n[Jumps] Detecting jumps (Global)...")
        jump_detector.fit(data)
        
        # Purification logic...
        flat_data = data.flatten()
        sigma_robust = np.median(np.abs(flat_data)) / 0.6745
        threshold = CONFIG["JUMP_THRESHOLD_STD"] * sigma_robust * np.sqrt(dt)
        data_for_vol = np.clip(data, -threshold, threshold)
        
        if CONFIG["USE_REAL_DATA"]:
            logger.info("   [Vis] Plotting improved jump visualizations...")
            
            if loader.log_returns is not None:
                cont_returns = loader.log_returns
                # Ensure dates are proper datetime objects
                cont_dates = pd.to_datetime(loader.raw_prices_df.index[1:])
                
                # 1. Price Path View (Global View)
                from utils.visualization import plot_jumps_on_price
                plot_jumps_on_price(
                    cont_dates, 
                    cont_returns, 
                    tickers, 
                    threshold_std=CONFIG["JUMP_THRESHOLD_STD"],
                    save_path=logger.get_save_path("jumps_on_price.png")
                )
                
                # 2. Zoomed View (2020 Crisis)
                from utils.visualization import plot_zoomed_crisis
                plot_zoomed_crisis(
                    cont_dates,
                    cont_returns,
                    tickers,
                    threshold_std=CONFIG["JUMP_THRESHOLD_STD"],
                    save_path=logger.get_save_path("jumps_zoom_2020.png")
                )
        # ----------------------------
        
    else:
        logger.info("\n[Jumps] Jump detection disabled.")

    
    logger.info("\n[Calibration] Fitting Volatility...")
    vol_calibrator = VolatilityCalibrator(dt=dt, method='kernel', bandwidth=CONFIG["VOL_BANDWIDTH"])
    vol_calibrator.fit(data_for_vol)
    
    # =========================================================
    # 3. Model Loop (With Metrics Collection)
    # =========================================================
    results_store = {}
    metrics_store = {} # {Method: {train_time, gen_time, WD, ACF_MSE}}
    
    n_test = min(CONFIG["N_GEN_PATHS"], len(data))
    test_idx = np.random.choice(len(data), n_test, replace=False)
    test_x0 = data[test_idx, 0, :] 
    real_paths_subset = data[test_idx]
    
    for model_name in CONFIG["MODELS_TO_RUN"]:
        logger.info(f"\n>>> Pipeline: {model_name}")
        metrics_store[model_name] = {}
        
        # Reference Measure
        if model_name == 'Kernel': t_scale = CONFIG["KERNEL_TEMP_SCALE"]
        elif model_name == 'SBTS-LSTM': t_scale = CONFIG["LSTM_TEMP_SCALE"]
        else: t_scale = CONFIG["LIGHTSB_TEMP_SCALE"]
            
        if CONFIG["USE_JUMPS"]:
            ref_proc = JumpDiffusionReference(vol_calibrator, jump_detector, volatility_multiplier=t_scale)
        else:
            ref_proc = LocalVolatilityReference(vol_calibrator, volatility_multiplier=t_scale)
            
        # Training
        drift_fn = None
        start_t = time.time()
        
        if model_name == 'Kernel':
            best_h = 0.05
            if CONFIG["USE_CV_FOR_KERNEL"]:
                logger.info("   [Opt] Bandwidth BayesOpt...")
                ref_cv = LocalVolatilityReference(vol_calibrator, 1.0)
                def cv_sim(start, drift):
                    return euler_maruyama_generator(start, time_grid, drift, ref_cv, len(start))
                selector = BandwidthSelector(n_trials=CONFIG["BO_N_TRIALS"], n_splits=3)
                best_h = selector.fit(data, KernelDriftEstimator, cv_sim, dt)
                plot_bandwidth_optimization(selector.get_history(), best_h, save_path=logger.get_save_path("bandwidth_bayes_opt.png"))
            
            estimator = KernelDriftEstimator(bandwidth=best_h)
            estimator.fit(data, dt)
            drift_fn = lambda t,x: estimator.predict(t, x)
            
        elif model_name == 'SBTS-LSTM':
            estimator = LSTMDriftEstimator(
                input_dim=n_assets, hidden_size=CONFIG["LSTM_HIDDEN"],
                lr=CONFIG["LSTM_LR"], epochs=CONFIG["LSTM_EPOCHS"], dt=dt,
                weight_decay=CONFIG["LSTM_WEIGHT_DECAY"], dropout=CONFIG["LSTM_DROPOUT"]
            )
            estimator.fit(data, dt)
            drift_fn = lambda t,x: estimator.predict(t, x) * CONFIG["LSTM_DRIFT_DAMPENING"]
            
        elif model_name == 'LightSB':
            X_curr = data[:, :-1, :].reshape(-1, n_assets)
            X_next = data[:, 1:, :].reshape(-1, n_assets)
            x0_t = torch.tensor(X_curr, dtype=torch.float32)
            x1_t = torch.tensor(X_next, dtype=torch.float32)
            
            lsb = LightSBTrainer(dim=n_assets, n_components=CONFIG["LIGHTSB_COMPONENTS"], lr=CONFIG["LIGHTSB_LR"], min_cov=CONFIG["LIGHTSB_MIN_COV"])
            logger.info("   [LightSB] Training GMM...")
            for i in range(CONFIG["LIGHTSB_STEPS"]):
                idx = np.random.choice(len(x0_t), 1024)
                lsb.train_step(x0_t[idx], x1_t[idx])
                
            drift_fn = lambda t,x: lsb.get_drift(t, x, clip_val=5.0)
            
        metrics_store[model_name]['train_time'] = time.time() - start_t
        
        # Generation
        logger.info(f"   [Gen] Simulating {n_test} paths...")
        start_t = time.time()
        gen_paths = euler_maruyama_generator(test_x0, time_grid, drift_fn, ref_proc, n_paths=n_test)
        metrics_store[model_name]['gen_time'] = time.time() - start_t
        
        results_store[model_name] = gen_paths
        
        # Eval
        if not np.isnan(gen_paths).any():
            wd = np.mean([wasserstein_distance_1d(real_paths_subset[:, -1, i], gen_paths[:, -1, i]) 
                          for i in range(min(5, n_assets))])
            
            acf_real = _calc_autocorr(real_paths_subset)
            acf_gen = _calc_autocorr(gen_paths)
            acf_mse = np.mean((acf_real - acf_gen)**2)
            
            metrics_store[model_name]['WD'] = wd
            metrics_store[model_name]['ACF_MSE'] = acf_mse
        else:
            metrics_store[model_name]['WD'] = np.nan
            metrics_store[model_name]['ACF_MSE'] = np.nan

    # =========================================================
    # 4. Visualization & Reporting
    # =========================================================
    logger.info("\n[Visualization] Generating Reports...")
    
    # A. Performance Bar Charts (The 1x4 Plot)
    df_metrics = pd.DataFrame(metrics_store).T
    print("\n--- Performance Summary ---")
    print(df_metrics)
    from utils.visualization import plot_performance_metrics
    plot_performance_metrics(df_metrics, save_path=logger.get_save_path("performance_metrics.png"))
    
    # B. Price Reconstruction
    S0 = 100.0
    real_prices = reconstruct_prices(S0, real_paths_subset)
    gen_prices_dict = {name: reconstruct_prices(S0, paths) for name, paths in results_store.items()}
    
    # C. Comprehensive Comparison (The 2x3 Plot)
    plot_comprehensive_comparison(
        time_grid, 
        real_paths_subset, 
        results_store,
        real_prices=real_prices,
        gen_prices_dict=gen_prices_dict,
        save_path=logger.get_save_path("final_comparison.png")
    )
    
    if n_assets > 1:
        plot_correlation_distribution(
            real_paths_subset,
            results_store,
            save_path=logger.get_save_path("final_correlation_dist.png")
        )

    logger.info(f"\n[Done] Results saved in: {logger.run_dir}")

    # D. Correlation Distribution
    if n_assets > 1:
        plot_correlation_distribution(
            real_paths_subset,
            results_store,
            save_path=logger.get_save_path("final_correlation_dist.png")
        )

    # --- Volatility Surface Visualization ---
    logger.info("   [Vis] Plotting Volatility Surface...")
    # Use the flatten data range to define the grid
    plot_volatility_surface(
        vol_calibrator, 
        data_range=data.flatten(), # Use full data range for x-axis limits
        T=dt*CONFIG["SEQ_LEN"],    # Physical time horizon
        save_path=logger.get_save_path("volatility_surface.png")
    )
    
    logger.info(f"\n[Done] Full Benchmark Complete. Results in: {logger.run_dir}")

if __name__ == "__main__":
    main()