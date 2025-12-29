import numpy as np
import os

# Imports (Relative to root)
from core.solver import euler_maruyama_generator
from core.drift_estimators import KernelDriftEstimator, LSTMDriftEstimator
from core.reference import LocalVolatilityReference, JumpDiffusionReference
from core.bandwidth import BandwidthSelector

from models.calibration import VolatilityCalibrator
from models.jumps import JumpDetector

from utils.visualization import plot_method_comparison, plot_price_reconstruction, plot_bandwidth_optimization, set_style
from utils.data_loader import RealDataLoader, reconstruct_prices
from utils.logger import Logger

def main():
    # =========================================================
    # 0. Configuration
    # =========================================================
    CONFIG = {
        "USE_REAL_DATA": True,
        "TICKER": "SPY",
        "START_DATE": "2020-01-01",
        "END_DATE": "2023-12-31",
        "SEQ_LEN": 60,
        
        # Jump Detection & Calibration
        "USE_JUMPS": True,          # Enable Phase 4
        "JUMP_THRESHOLD_STD": 4.0,  # Threshold for jump detection (higher = fewer jumps)
        
        # Volatility Calibration
        "VOL_BANDWIDTH_REAL": 0.5,
        "VOL_BANDWIDTH_SYNTH": 0.5,
        
        # Kernel Estimator (Bayesian Opt Config)
        "USE_CV_FOR_KERNEL": True, 
        "BO_N_TRIALS": 20,              
        "KERNEL_BANDWIDTH_DEFAULT": 0.05,
        "KERNEL_TEMP_SCALE": 1.1,
        
        # LSTM Estimator
        "LSTM_HIDDEN_SIZE": 128,        
        "LSTM_LR": 0.005,
        "LSTM_EPOCHS": 100,
        "LSTM_WEIGHT_DECAY": 1e-3,
        "LSTM_DROPOUT": 0.3,
        "LSTM_TEMP_SCALE": 1.1,         # Slightly increased to compensate for removed jumps in sigma
        "LSTM_DRIFT_DAMPENING": 0.8,
        
        "N_GEN_PATHS": 1000,
        "SYNTHETIC_SAMPLES": 10000
    }

    logger = Logger(base_dir="experiments")
    logger.info("==========================================")
    logger.info("   SBTS Advanced: Jump-Diffusion Pipeline ")
    logger.info("   (With Volatility Purification)         ")
    logger.info("==========================================\n")
    logger.save_config(CONFIG)
    set_style()
    
    # =========================================================
    # 1. Data Loading
    # =========================================================
    if CONFIG["USE_REAL_DATA"]:
        logger.info(f"[Data] Mode: Real Data ({CONFIG['TICKER']})")
        loader = RealDataLoader(CONFIG["TICKER"], CONFIG["START_DATE"], CONFIG["END_DATE"])
        loader.download()
        data, mu, sigma = loader.get_sliding_windows(CONFIG["SEQ_LEN"])
        dt = 1.0 / 252.0 
        steps = CONFIG["SEQ_LEN"]
        time_grid = np.linspace(0, steps * dt, steps)
        logger.info(f"[Stats] Mean: {mu:.5f}, Vol: {sigma:.5f}")
    else:
        logger.info("[Data] Mode: Synthetic OU Data")
        T, N_samples = 1.0, CONFIG["SYNTHETIC_SAMPLES"]
        steps = 50
        time_grid = np.linspace(0, T, steps)
        dt = T / (steps - 1)
        data = np.zeros((N_samples, steps, 1))
        for i in range(N_samples):
            x = np.random.normal(0, 0.5)
            for t_idx in range(1, steps):
                dx = -2.0 * x * dt + 0.5 * np.sqrt(dt) * np.random.randn()
                x += dx
                data[i, t_idx, 0] = x

    # =========================================================
    # 2. Phase 4: Jump Detection & Data Purification
    # =========================================================
    jump_detector = JumpDetector(dt=dt, threshold_multiplier=CONFIG["JUMP_THRESHOLD_STD"])
    
    # Create a clean dataset for volatility calibration
    # Default is original data (if jumps are disabled)
    data_for_vol_calibration = data.copy() 

    if CONFIG["USE_JUMPS"]:
        logger.info("\n[Jumps] Detecting jumps in training data...")
        jump_detector.fit(data)
        
        # --- CRITICAL: Purify Data ---
        logger.info("   [Purification] Removing jumps from data for volatility calibration...")
        
        flat_data = data.flatten()
        # Robust sigma estimate
        sigma_robust = np.median(np.abs(flat_data)) / 0.6745
        threshold = CONFIG["JUMP_THRESHOLD_STD"] * sigma_robust * np.sqrt(dt)
        
        # Clip data to threshold to remove extreme jumps
        data_for_vol_calibration = np.clip(data, -threshold, threshold)
        
        logger.info(f"   [Purification] Data clipped to range [{ -threshold:.4f}, {threshold:.4f}]")
    else:
        logger.info("\n[Jumps] Jump detection disabled.")

    # =========================================================
    # 3. Phase 3: Volatility Calibration (On Purified Data)
    # =========================================================
    logger.info("\n[Calibration] Fitting Local Volatility Surface...")
    bw = CONFIG["VOL_BANDWIDTH_REAL"] if CONFIG["USE_REAL_DATA"] else CONFIG["VOL_BANDWIDTH_SYNTH"]
    
    vol_calibrator = VolatilityCalibrator(dt=dt, method='kernel', bandwidth=bw)
    # Fit on CLEAN data to avoid double counting variance
    vol_calibrator.fit(data_for_vol_calibration) 
    
    # Reference for CV (use simple LV, no jumps for bandwidth tuning)
    ref_for_cv = LocalVolatilityReference(vol_calibrator, volatility_multiplier=1.0)

    # =========================================================
    # 4. Drift Estimation (Kernel & LSTM)
    # =========================================================
    # Note: Drift estimators are trained on ORIGINAL data (with jumps)
    # so they learn the correct mean-reversion force after a jump.
    
    logger.info("\n[Estimator 1] Kernel Drift Estimator")
    best_h = CONFIG["KERNEL_BANDWIDTH_DEFAULT"]
    
    if CONFIG["USE_CV_FOR_KERNEL"]:
        logger.info("   Running Bayesian Optimization for Bandwidth...")
        
        def cv_simulator(start_points, drift_fn):
            return euler_maruyama_generator(
                start_points, time_grid, drift_fn, ref_for_cv, n_paths=len(start_points)
            )
            
        selector = BandwidthSelector(n_trials=CONFIG["BO_N_TRIALS"], n_splits=3)
        best_h = selector.fit(
            trajectories=data, 
            drift_estimator_cls=KernelDriftEstimator, 
            simulator_func=cv_simulator,
            dt=dt
        )
        
        plot_bandwidth_optimization(
            selector.get_history(), best_h, 
            save_path=logger.get_save_path("bandwidth_bayes_opt.png")
        )
        logger.info(f"   Optimization Plot saved.")
    
    kernel_est = KernelDriftEstimator(bandwidth=best_h) 
    kernel_est.fit(data, dt)
    
    # --- B. LSTM ---
    logger.info("\n[Estimator 2] Training LSTM Drift...")
    lstm_est = LSTMDriftEstimator(
        input_dim=1, 
        hidden_size=CONFIG["LSTM_HIDDEN_SIZE"], 
        lr=CONFIG["LSTM_LR"],         
        epochs=CONFIG["LSTM_EPOCHS"],       
        dt=dt,
        weight_decay=CONFIG["LSTM_WEIGHT_DECAY"], 
        dropout=CONFIG["LSTM_DROPOUT"]        
    )
    lstm_est.fit(data, dt)

    # =========================================================
    # 5. Generation (With Jumps)
    # =========================================================
    logger.info("\n[Generation] Simulating paths...")
    
    n_test = min(CONFIG["N_GEN_PATHS"], len(data))
    
    # Dynamic Sampling logic
    test_idx = np.random.choice(len(data), n_test, replace=False)
    test_x0 = data[test_idx, 0, :] 
    real_paths_subset = data[test_idx]
    
    logger.info(f"   Generating {n_test} test paths...")
    
    # --- Construct Reference Measures (Mixed or Simple) ---
    if CONFIG["USE_JUMPS"]:
        ref_kernel = JumpDiffusionReference(vol_calibrator, jump_detector, volatility_multiplier=CONFIG["KERNEL_TEMP_SCALE"])
        ref_lstm   = JumpDiffusionReference(vol_calibrator, jump_detector, volatility_multiplier=CONFIG["LSTM_TEMP_SCALE"])
    else:
        ref_kernel = LocalVolatilityReference(vol_calibrator, volatility_multiplier=CONFIG["KERNEL_TEMP_SCALE"])
        ref_lstm   = LocalVolatilityReference(vol_calibrator, volatility_multiplier=CONFIG["LSTM_TEMP_SCALE"])

    # Kernel Gen
    paths_kernel = euler_maruyama_generator(
        test_x0, time_grid, lambda t,x: kernel_est.predict(t,x), ref_kernel, n_paths=n_test
    )
    
    # LSTM Gen
    dampening = CONFIG["LSTM_DRIFT_DAMPENING"]
    paths_lstm = euler_maruyama_generator(
        test_x0, time_grid, lambda t,x: lstm_est.predict(t,x)*dampening, ref_lstm, n_paths=n_test
    )
    
    # =========================================================
    # 6. Visualization
    # =========================================================
    logger.info("\n[Visualization] Producing Plots...")
    
    save_path_returns = logger.get_save_path("returns_comparison.png")
    plot_method_comparison(time_grid, real_paths_subset, paths_kernel, paths_lstm, save_path=save_path_returns)
    logger.info(f"   Saved returns plot to: {save_path_returns}")
    
    if CONFIG["USE_REAL_DATA"]:
        S0 = 100.0
        real_prices = reconstruct_prices(S0, real_paths_subset)
        lstm_prices = reconstruct_prices(S0, paths_lstm)
        
        save_path_prices = logger.get_save_path("price_reconstruction.png")
        plot_price_reconstruction(
            real_prices, 
            lstm_prices, 
            "LSTM (Jump-Diff)", 
            CONFIG["TICKER"], 
            save_path=save_path_prices
        )
        logger.info(f"   Saved price plot to: {save_path_prices}")

    logger.info(f"\n[Done] All results saved in: {logger.run_dir}")

if __name__ == "__main__":
    main()