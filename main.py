import numpy as np
import os

# Imports (Relative to root)
from core.solver import euler_maruyama_generator
from core.drift_estimators import KernelDriftEstimator, LSTMDriftEstimator
from core.reference import LocalVolatilityReference
# Import the NEW Bayesian Selector
from core.bandwidth import BandwidthSelector

from models.calibration import VolatilityCalibrator
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
        
        # Volatility Calibration
        "VOL_BANDWIDTH_REAL": 0.5,
        "VOL_BANDWIDTH_SYNTH": 0.5,
        
        # Kernel Estimator (Bayesian Opt Config)
        "USE_CV_FOR_KERNEL": True, 
        "BO_N_TRIALS": 20,              # <--- NEW: 贝叶斯优化的尝试次数
        "KERNEL_BANDWIDTH_DEFAULT": 0.05,
        "KERNEL_TEMP_SCALE": 1.1,
        
        # LSTM Estimator
        "LSTM_HIDDEN_SIZE": 128,        # Slightly larger for real data
        "LSTM_LR": 0.005,
        "LSTM_EPOCHS": 100,
        "LSTM_WEIGHT_DECAY": 1e-3,
        "LSTM_DROPOUT": 0.3,
        "LSTM_TEMP_SCALE": 1.0,         # Reduced based on previous findings
        "LSTM_DRIFT_DAMPENING": 0.8,
        
        "N_GEN_PATHS": 1000,
        "SYNTHETIC_SAMPLES": 10000
    }

    logger = Logger(base_dir="experiments")
    logger.info("==========================================")
    logger.info("   SBTS Advanced: Bayesian Pipeline       ")
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
                sigma_x = 0.3 + 0.2 * np.abs(x)
                dx = -2.0 * x * dt + sigma_x * np.sqrt(dt) * np.random.randn()
                x += dx
                data[i, t_idx, 0] = x

    # =========================================================
    # 2. Volatility Calibration
    # =========================================================
    logger.info("\n[Calibration] Fitting Local Volatility Surface...")
    bw = CONFIG["VOL_BANDWIDTH_REAL"] if CONFIG["USE_REAL_DATA"] else CONFIG["VOL_BANDWIDTH_SYNTH"]
    vol_calibrator = VolatilityCalibrator(dt=dt, method='kernel', bandwidth=bw)
    vol_calibrator.fit(data)
    
    ref_for_cv = LocalVolatilityReference(vol_calibrator, volatility_multiplier=1.0)

    # =========================================================
    # 3. Drift Estimation (Bayesian Optimization)
    # =========================================================
    
    logger.info("\n[Estimator 1] Kernel Drift Estimator")
    best_h = CONFIG["KERNEL_BANDWIDTH_DEFAULT"]
    
    if CONFIG["USE_CV_FOR_KERNEL"]:
        logger.info("   Running Bayesian Optimization for Bandwidth...")
        
        # 修复 n_paths 问题
        def cv_simulator(start_points, drift_fn):
            return euler_maruyama_generator(
                start_points, time_grid, drift_fn, ref_for_cv, n_paths=len(start_points)
            )
            
        # 初始化贝叶斯选择器
        selector = BandwidthSelector(n_trials=CONFIG["BO_N_TRIALS"], n_splits=3)
        best_h = selector.fit(
            trajectories=data, 
            drift_estimator_cls=KernelDriftEstimator, 
            simulator_func=cv_simulator,
            dt=dt
        )
        
        # Optuna 的历史记录不是网格，而是散点。
        # 我们的 plot_bandwidth_optimization 函数依然兼容 (dict: h -> mse)
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
    # 4. Generation
    # =========================================================
    logger.info("\n[Generation] Simulating paths...")
    
    n_test = min(CONFIG["N_GEN_PATHS"], len(data))
    test_idx = np.random.choice(len(data), n_test, replace=False)
    test_x0 = data[test_idx, 0, :] 
    real_paths_subset = data[test_idx]
    
    logger.info(f"   Generating {n_test} test paths...")
    
    # Kernel Gen
    ref_kernel = LocalVolatilityReference(vol_calibrator, volatility_multiplier=CONFIG["KERNEL_TEMP_SCALE"])
    paths_kernel = euler_maruyama_generator(
        test_x0, time_grid, lambda t,x: kernel_est.predict(t,x), ref_kernel, n_paths=n_test
    )
    
    # LSTM Gen
    ref_lstm = LocalVolatilityReference(vol_calibrator, volatility_multiplier=CONFIG["LSTM_TEMP_SCALE"])
    dampening = CONFIG["LSTM_DRIFT_DAMPENING"]
    paths_lstm = euler_maruyama_generator(
        test_x0, time_grid, lambda t,x: lstm_est.predict(t,x)*dampening, ref_lstm, n_paths=n_test
    )
    
    # =========================================================
    # 5. Visualization
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
            "LSTM", 
            CONFIG["TICKER"], 
            save_path=save_path_prices
        )
        logger.info(f"   Saved price plot to: {save_path_prices}")

    logger.info(f"\n[Done] All results saved in: {logger.run_dir}")

if __name__ == "__main__":
    main()