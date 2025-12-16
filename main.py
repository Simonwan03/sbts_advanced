import numpy as np
import os

# Core Modules
from core.solver import euler_maruyama_generator
from core.drift_estimators import KernelDriftEstimator, LSTMDriftEstimator
from core.reference import LocalVolatilityReference
from models.calibration import VolatilityCalibrator
from utils.visualization import plot_method_comparison, set_style

def main():
    set_style()
    print("==========================================")
    print("   SBTS Advanced: Optimized Pipeline      ")
    print("==========================================\n")
    
    os.makedirs("outputs", exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. Data Generation
    # ---------------------------------------------------------
    T, N_samples = 1.0, 10000 
    steps = 50
    time_grid = np.linspace(0, T, steps)
    dt = T / (steps - 1)
    
    print(f"Generating {N_samples} Synthetic Trajectories...")
    data = np.zeros((N_samples, steps, 1))
    
    # Generate OU process with state-dependent volatility
    # This tests both drift recovery and volatility calibration
    for i in range(N_samples):
        x = np.random.normal(0, 0.5)
        for t_idx in range(1, steps):
            # Sigma increases away from mean (smile)
            sigma_x = 0.3 + 0.2 * np.abs(x)
            dx = -2.0 * x * dt + sigma_x * np.sqrt(dt) * np.random.randn()
            x += dx
            data[i, t_idx, 0] = x
            
    print(f"Data Shape: {data.shape}")

    # ---------------------------------------------------------
    # 2. Volatility Calibration (Phase 3)
    # ---------------------------------------------------------
    print("\n[Calibration] Fitting Local Volatility Surface...")
    vol_calibrator = VolatilityCalibrator(dt=dt, method='kernel', bandwidth=0.5)
    vol_calibrator.fit(data)

    # ---------------------------------------------------------
    # 3. Drift Estimation
    # ---------------------------------------------------------
    
    # A. Kernel
    print("\n[Estimator 1] Fitting Kernel Drift...")
    kernel_est = KernelDriftEstimator(bandwidth=0.1)
    kernel_est.fit(data, dt)
    
    # B. LSTM (Optimized for Distribution Matching)
    print("\n[Estimator 2] Training LSTM Drift...")
    lstm_est = LSTMDriftEstimator(
        input_dim=1, 
        hidden_size=64, 
        lr=0.01, 
        epochs=80,        # Increased epochs due to dropout
        dt=dt,
        weight_decay=1e-4, 
        dropout=0.2       # Added Dropout
    )
    lstm_est.fit(data, dt)

    # ---------------------------------------------------------
    # 4. Generation (With Drift Dampening & Temp Scaling)
    # ---------------------------------------------------------
    print("\n[Generation] Simulating paths...")
    
    test_x0 = data[:1000, 0, :]
    
    # --- Configuration for Kernel ---
    # Kernel fits variance well, so Temp Scale 1.4 is usually sufficient
    ref_kernel = LocalVolatilityReference(vol_calibrator, volatility_multiplier=1.4)
    def kernel_drift_func(t, x): 
        return kernel_est.predict(t, x)
    
    paths_kernel = euler_maruyama_generator(
        test_x0, time_grid, kernel_drift_func, ref_kernel, n_paths=1000
    )
    
    # --- Configuration for LSTM ---
    # LSTM tends to pull too hard (high kurtosis). 
    # Fix: Higher Temp Scale + Drift Dampening
    ref_lstm = LocalVolatilityReference(vol_calibrator, volatility_multiplier=1.6)
    
    # Dampening factor to fix the steep ACF slope
    DRIFT_DAMPENING = 0.9 
    
    def lstm_drift_func(t, x): 
        raw_drift = lstm_est.predict(t, x)
        return raw_drift * DRIFT_DAMPENING

    paths_lstm = euler_maruyama_generator(
        test_x0, time_grid, lstm_drift_func, ref_lstm, n_paths=1000
    )

    # ---------------------------------------------------------
    # 5. Visualization
    # ---------------------------------------------------------
    print("\n[Visualization] Saving Final Comparison...")
    
    plot_method_comparison(
        time_grid, 
        data, 
        paths_kernel, 
        paths_lstm, 
        save_path="outputs/final_corrected.png"
    )
    
    print("\nDone! Check 'outputs/final_corrected.png'.")

if __name__ == "__main__":
    main()