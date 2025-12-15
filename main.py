import numpy as np
import os

# Core Modules
from core.reference import LevyProcess
from core.solver import euler_maruyama_generator
from core.drift_estimators import KernelDriftEstimator, LSTMDriftEstimator
from models.jumps import JumpDetector
# 引入新的统一绘图接口
from utils.visualization import (
    plot_bandwidth_optimization,
    plot_jump_detection,
    plot_method_comparison
)

def main():
    print("==========================================")
    print("   SBTS Advanced: Final Comparison        ")
    print("==========================================\n")
    
    # ---------------------------------------------------------
    # 1. Data Generation (High Density)
    # ---------------------------------------------------------
    T, N_samples = 1.0, 10000 
    steps = 50
    time_grid = np.linspace(0, T, steps)
    dt = T / (steps - 1)
    
    print(f"Generating {N_samples} Synthetic OU Trajectories...")
    data = np.zeros((N_samples, steps, 1))
    for i in range(N_samples):
        x = np.random.normal(0, 0.4)
        for t_idx in range(1, steps):
            # OU Process: dX = -2X dt + 0.5 dW + Jumps
            dx = -2.0 * x * dt + 0.5 * np.sqrt(dt) * np.random.randn()
            if np.random.random() < (0.5 * dt): 
                dx += np.random.choice([0.6, -0.6])
            x += dx
            data[i, t_idx, 0] = x
            
    # ---------------------------------------------------------
    # 2. Estimator Training
    # ---------------------------------------------------------
    
    # --- A. Kernel ---
    print("\n[Estimator 1] Fitting Kernel Drift...")
    kernel_est = KernelDriftEstimator(bandwidth=0.1)
    kernel_est.fit(data, dt)
    
    # --- B. LSTM ---
    print("\n[Estimator 2] Training LSTM Drift...")
    lstm_est = LSTMDriftEstimator(input_dim=1, hidden_size=64, lr=0.01, epochs=50, dt=dt)
    lstm_est.fit(data, dt)

    # ---------------------------------------------------------
    # 3. Generation (with Temperature Scaling)
    # ---------------------------------------------------------
    print("\n[Generation] Simulating paths...")
    SIGMA_SCALE = 1.4
    
    class ScaledRef(LevyProcess):
        def get_diffusion(self, t, x):
            return self.sigma * SIGMA_SCALE

    ref_process = ScaledRef(0.5, 0, lambda:0) # Base sigma matches data (0.5)
    test_x0 = data[:1000, 0, :]
    
    # Wrapper functions for solver
    def kernel_func(t, x): return kernel_est.predict(t, x)
    def lstm_func(t, x): return lstm_est.predict(t, x)
    
    paths_kernel = euler_maruyama_generator(
        test_x0, time_grid, kernel_func, ref_process, n_paths=1000
    )
    
    paths_lstm = euler_maruyama_generator(
        test_x0, time_grid, lstm_func, ref_process, n_paths=1000
    )

    # ---------------------------------------------------------
    # 4. Visualization (Unified)
    # ---------------------------------------------------------
    print("\n[Visualization] Producing Reports...")
    
    # Example 1: Jump Detection Plot (Using just the first trace)
    detector = JumpDetector(dt=dt, threshold_multiplier=3.0)
    returns = np.diff(data[0, :, 0])
    jump_res = detector.fit_detect(returns)
    plot_jump_detection(
        time_grid, data[0,:,0], jump_res['indices'], 
        save_path="outputs/analysis_jump_detection.png"
    )
    
    # Example 2: Method Comparison (The Big Plot)
    plot_method_comparison(
        time_grid, 
        data, 
        paths_kernel, 
        paths_lstm, 
        save_path="outputs/final_method_comparison.png"
    )
    
    print("\nDone. All plots saved to 'outputs/' directory.")

if __name__ == "__main__":
    main()