import numpy as np
import torch
from core.bandwidth import BandwidthSelector
from core.lightsb import LightSBTrainer
from core.reference import LocalVolatility, LevyProcess, modified_drift_adjustment
from core.solver import euler_maruyama_generator
from models.jumps import JumpDetector

import numpy as np
import torch
import os

# Importing Core Modules
from core.bandwidth import BandwidthSelector
from core.lightsb import LightSBTrainer
from core.reference import LocalVolatility, LevyProcess, modified_drift_adjustment
from core.solver import euler_maruyama_generator
from models.jumps import JumpDetector

# Importing Visualization
from utils.visualization import (
    plot_trajectories, 
    plot_bandwidth_optimization, 
    plot_marginal_density,
    plot_jump_detection
)

def main():
    print("==========================================")
    print("   Advanced SBTS Framework Initialized    ")
    print("==========================================\n")
    
    # Output directory for plots
    os.makedirs("outputs", exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. Data Loading (Mock Synthetic Data: OU Process with Jumps)
    # ---------------------------------------------------------
    T, N_samples = 1.0, 1000
    steps = 252
    time_grid = np.linspace(0, T, steps)
    dt = T / (steps - 1)
    
    # Create Synthetic Data: OU dX = -2X dt + dW + Jumps
    print(f"Loading {N_samples} sample trajectories...")
    data = np.zeros((N_samples, steps, 1))
    for i in range(N_samples):
        x = 0.0
        for t_idx in range(1, steps):
            # Normal OU
            dx = -2.0 * x * dt + np.sqrt(dt) * np.random.randn()
            # Random sparse jumps
            if np.random.random() < (5.0 * dt): # Lambda ~ 5
                dx += np.random.normal(0, 0.5) 
            x += dx
            data[i, t_idx, 0] = x
            
    print(f"Data Shape: {data.shape}")

    # ---------------------------------------------------------
    # Enhancement 1: Adaptive Bandwidth Selection
    # ---------------------------------------------------------
    print("\n[Phase 1] Tuning Bandwidth (Cross-Validation)...")
    selector = BandwidthSelector(candidates=np.logspace(-2, 0, 20), n_splits=3)
    
    # Mock definitions for selector to work quickly without full training
    def mock_drift_est(d, h): return lambda t, x: -2.0*x # Known ground truth for mock
    def mock_sim(start, drift): return start * np.exp(-2.0 * 1.0) # Decay
    
    best_h = selector.fit(data, mock_drift_est, mock_sim)
    
    # Plot Bandwidth Optimization
    plot_bandwidth_optimization(
        selector.mse_history, 
        best_h, 
        save_path="outputs/1_bandwidth_cv.png"
    )
    
    # ---------------------------------------------------------
    # Enhancement 2: LightSB (Long Sequences)
    # ---------------------------------------------------------
    print("\n[Phase 2] Training LightSB GMM Potential...")
    lsb_trainer = LightSBTrainer(dim=1, n_components=10)
    
    # Convert data ends to Torch for training
    x0_torch = torch.tensor(data[:, 0, :], dtype=torch.float32)
    x1_torch = torch.tensor(data[:, -1, :], dtype=torch.float32)
    
    # Quick Training Loop
    losses = []
    for _ in range(50): # 50 steps for demo
        loss = lsb_trainer.train_step(x0_torch, x1_torch)
        losses.append(loss)
        
    print(f"Final LightSB Loss: {losses[-1]:.4f}")
    
    # ---------------------------------------------------------
    # Enhancement 4: Jump Handling
    # ---------------------------------------------------------
    print("\n[Phase 4] Detecting Jumps & Calibration...")
    # Analyze the first path for visualization
    sample_returns = np.diff(data[0, :, 0])
    detector = JumpDetector(dt=dt, threshold_multiplier=3.0)
    jump_params = detector.fit_detect(sample_returns)
    
    print(f"Detected Jump Intensity (Lambda): {jump_params['lambda']:.4f}")
    
    # Plot Jump Detection
    plot_jump_detection(
        time_grid, 
        data[0, :, 0], 
        jump_params['indices'], 
        save_path="outputs/2_jump_detection.png"
    )
    
    # ---------------------------------------------------------
    # Generation & Visualization
    # ---------------------------------------------------------
    print("\n[Generation] Generating New Trajectories...")
    
    # Setup Levy Reference Process
    def jump_dist_sampler(): 
        return np.random.normal(jump_params['mu_jump'], max(0.1, jump_params['sigma_jump']))
        
    ref_process = LevyProcess(
        base_sigma=1.0, 
        intensity_lambda=jump_params['lambda'], 
        jump_dist_func=jump_dist_sampler
    )
    
    # Define Drift Function (combining LightSB + Phase 3 Adjustment)
    def drift_fn(t, x):
        # 1. Get Gradient from LightSB
        grad_log_v = lsb_trainer.get_drift(t, x)
        # 2. Adjust using Reference (Phase 3 logic)
        return modified_drift_adjustment(ref_process, lambda t_in, x_in: grad_log_v, t, x)
    
    # Generate 100 paths for valid distribution comparison
    gen_paths = euler_maruyama_generator(
        x0=data[:100, 0, :], # Start from same initial distribution
        time_grid=time_grid, 
        drift_func=drift_fn, 
        reference_process=ref_process, 
        n_paths=100
    )
    
    print(f"Generated Shape: {gen_paths.shape}")
    
    # ---------------------------------------------------------
    # Final Plots
    # ---------------------------------------------------------
    print("\n[Visualization] Saving Comparison Plots...")
    
    plot_trajectories(
        time_grid, 
        data[:100], 
        gen_paths, 
        save_path="outputs/3_trajectory_comparison.png"
    )
    
    plot_marginal_density(
        data, 
        gen_paths, 
        t_step=-1, 
        save_path="outputs/4_terminal_density.png"
    )
    
    print("\nAll tasks completed successfully. Check 'outputs/' folder.")

if __name__ == "__main__":
    main()