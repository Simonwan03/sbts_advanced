"""
Euler-Maruyama Solver for Jump-Diffusion Schrödinger Bridge

This module implements the path generation for the SDE:
    dX_t = α(t, X_t) dt + σ(t, X_t) dW_t + dJ_t

REFACTORED (Step 3):
- Jumps are treated as EXTERNAL component added to diffusion step
- This ensures gradient flow for drift parameter α is correct
- Drift network doesn't try to learn the jumps

Key Design Principle:
    dX = dX_continuous + dJ_external
    
    where:
    - dX_continuous = α(t,x)dt + σ(t,x)dW (learned by drift network)
    - dJ_external = jump component (sampled from calibrated distribution)
    
This separation is crucial because:
1. Jumps are discontinuous and shouldn't affect drift gradient
2. The drift α* should only capture continuous dynamics
3. Jump statistics are calibrated separately from drift learning
"""

import numpy as np


def euler_maruyama_generator(x0, time_grid, drift_func, reference_process, n_paths=1,
                             x_history=None, history_seq_len=10):
    """
    Generates trajectories using the Advanced SBTS SDE with Explicit Jump Diffusion.
    
    Model:
        dX_t = dX_cont + dJ_t
        dX_cont = drift(t, X_t) * dt + sigma(t, X_t) * dW_t
        dJ_t = Jump component (Poisson/Compound) - EXTERNAL to drift
    
    IMPORTANT (Step 3 Fix):
    Jumps are added as an external component AFTER the continuous increment.
    This ensures:
    1. Drift network learns only continuous dynamics
    2. Gradient flow for α is not corrupted by jump discontinuities
    3. The SB matching loss remains valid
    
    Args:
        x0: Initial states. Shape (dim,) or (n_paths, dim).
        time_grid: Array of time steps.
        drift_func: Callable returning drift vector. Should support batched inputs.
        reference_process: Instance of ReferenceMeasure (provides diffusion and jumps).
        n_paths: Number of trajectories to generate.
        x_history: Optional (n_paths, seq_len, dim) history for neural jumps.
        history_seq_len: Sequence length for neural jump intensity.
        
    Returns:
        paths: Generated trajectories of shape (n_paths, len(time_grid), dim).
    """
    N_steps = len(time_grid)
    dt = time_grid[1] - time_grid[0]
    
    # --- 1. Dimension & Shape Handling ---
    x0 = np.asarray(x0)
    
    if x0.ndim == 1:
        dim = x0.shape[0]
        current_x = np.tile(x0, (n_paths, 1))
    else:
        dim = x0.shape[-1]
        if x0.shape[0] != n_paths:
            n_paths = x0.shape[0]
        current_x = x0.copy()
    
    # Initialize path storage
    paths = np.zeros((n_paths, N_steps, dim))
    paths[:, 0, :] = current_x
    
    # Initialize history buffer for neural jumps (if needed)
    use_neural_jumps = hasattr(reference_process, 'get_intensity')
    if use_neural_jumps:
        # Maintain rolling history for intensity prediction
        history_buffer = np.zeros((n_paths, history_seq_len, dim))
        history_buffer[:, -1, :] = current_x
    
    # --- 2. Generation Loop ---
    for i in range(N_steps - 1):
        t = time_grid[i]
        
        # ============================================
        # A. CONTINUOUS COMPONENT (Drift + Diffusion)
        # ============================================
        
        # A1. Compute Drift (Vectorized)
        try:
            drift = drift_func(t, current_x)
        except (TypeError, ValueError, RuntimeError):
            drift = np.array([drift_func(t, x) for x in current_x])
            
        if drift.ndim == 1 and n_paths > 1:
            drift = drift.reshape(n_paths, -1)

        # A2. Compute Diffusion (Sigma)
        try:
            sigma = reference_process.get_diffusion(t, current_x)
        except (TypeError, ValueError, AttributeError):
            sigma = np.array([reference_process.get_diffusion(t, x) for x in current_x])
            
        if sigma.ndim == 1: 
            sigma = sigma[:, np.newaxis]
        
        # A3. Brownian Motion (dW)
        dW = np.random.normal(0, np.sqrt(dt), size=(n_paths, dim))
        
        # A4. Continuous Increment
        # This is what the drift network learns to predict
        dX_cont = drift * dt + sigma * dW
        
        # ============================================
        # B. JUMP COMPONENT (External - NOT learned by drift)
        # ============================================
        # 
        # CRITICAL (Step 3): Jumps are sampled independently and added
        # as an external component. The drift network never sees these
        # discontinuities during training, ensuring correct gradient flow.
        #
        
        if use_neural_jumps:
            # Neural MJD: intensity depends on history
            dJ_list = []
            for path_idx in range(n_paths):
                path_history = history_buffer[path_idx:path_idx+1]  # (1, seq_len, dim)
                jump = reference_process.sample_jump(t, dt, x_history=path_history, batch_size=1)
                dJ_list.append(jump)
            dJ = np.array(dJ_list).reshape(n_paths, -1)
        else:
            # Static MJD: constant intensity
            dJ_list = [reference_process.sample_jump(t, dt) for _ in range(n_paths)]
            dJ = np.array(dJ_list)
        
        # Ensure dJ shape (N, D)
        if dJ.ndim == 1: 
            dJ = dJ[:, np.newaxis]
        if dJ.shape[-1] == 1 and dim > 1:
            # Broadcast scalar jump to all dimensions (common for single-factor jumps)
            dJ = np.tile(dJ, (1, dim))
        
        # ============================================
        # C. UPDATE STATE
        # ============================================
        # 
        # Explicit Jump Diffusion: X_new = X_old + dX_cont + dJ
        # 
        # The separation ensures:
        # 1. dX_cont captures continuous dynamics (learned by drift)
        # 2. dJ captures discontinuous shocks (calibrated from data)
        # 3. No gradient leakage between the two components
        #
        current_x = current_x + dX_cont + dJ
        
        # Store
        paths[:, i+1, :] = current_x
        
        # Update history buffer for neural jumps
        if use_neural_jumps:
            history_buffer = np.roll(history_buffer, -1, axis=1)
            history_buffer[:, -1, :] = current_x
        
    return paths


def euler_maruyama_with_drift_dampening(x0, time_grid, drift_func, reference_process, 
                                        n_paths=1, dampening_factor=0.9):
    """
    Euler-Maruyama solver with drift dampening for stability.
    
    Drift dampening prevents explosive paths by scaling down the drift:
        α_damped = α * dampening_factor
    
    This is particularly useful when:
    1. The drift network overfits to training data
    2. Generated paths tend to diverge
    3. Multi-asset correlations cause instability
    
    Args:
        x0: Initial states
        time_grid: Time grid
        drift_func: Original drift function
        reference_process: Reference measure
        n_paths: Number of paths
        dampening_factor: Scaling factor for drift (0 < factor ≤ 1)
        
    Returns:
        paths: Generated trajectories
    """
    # Wrap drift function with dampening
    def dampened_drift(t, x):
        return drift_func(t, x) * dampening_factor
    
    return euler_maruyama_generator(
        x0, time_grid, dampened_drift, reference_process, n_paths
    )
