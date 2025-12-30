import numpy as np

def euler_maruyama_generator(x0, time_grid, drift_func, reference_process, n_paths=1):
    """
    Generates trajectories using the Advanced SBTS SDE with Explicit Jump Diffusion.
    
    Model:
        dX_t = dX_cont + dJ_t
        dX_cont = drift(t, X_t) * dt + sigma(t, X_t) * dW_t
        dJ_t = Jump component (Poisson/Compound)
    
    Args:
        x0: Initial states. Shape (dim,) or (n_paths, dim).
        time_grid: Array of time steps.
        drift_func: Callable returning drift vector. Should support batched inputs.
        reference_process: Instance of ReferenceMeasure (provides diffusion and jumps).
        n_paths: Number of trajectories to generate.
        
    Returns:
        paths: Generated trajectories of shape (n_paths, len(time_grid), dim).
    """
    N_steps = len(time_grid)
    dt = time_grid[1] - time_grid[0]
    
    # --- 1. Dimension & Shape Handling (Vectorization) ---
    x0 = np.asarray(x0)
    
    # Determine dimension (D) and ensure x0 is (n_paths, D)
    if x0.ndim == 1:
        # Case: x0 is a single point (D,) -> Broadcast to (n_paths, D)
        dim = x0.shape[0]
        current_x = np.tile(x0, (n_paths, 1))
    else:
        # Case: x0 is a batch (n_paths, D)
        # Ensure n_paths matches or trust input
        dim = x0.shape[-1]
        if x0.shape[0] != n_paths:
            # If mismatch, prioritize input shape if it looks like a batch
            n_paths = x0.shape[0]
        current_x = x0.copy()
    
    # Initialize full path tensor
    paths = np.zeros((n_paths, N_steps, dim))
    paths[:, 0, :] = current_x
    
    # --- 2. Generation Loop ---
    for i in range(N_steps - 1):
        t = time_grid[i]
        
        # A. Compute Drift (Vectorized)
        # Try passing batch first (N, D), fallback to loop if func doesn't support it
        try:
            drift = drift_func(t, current_x)
        except (TypeError, ValueError, RuntimeError):
            # Fallback for simple scalar functions
            drift = np.array([drift_func(t, x) for x in current_x])
            
        # Ensure drift shape (N, D)
        if drift.ndim == 1 and n_paths > 1:
             # Handle case where drift returns 1D array for batch inputs (rare but possible)
             drift = drift.reshape(n_paths, -1)

        # B. Compute Diffusion (Sigma)
        # Reference process should return sigma for the batch
        try:
            sigma = reference_process.get_diffusion(t, current_x)
        except (TypeError, ValueError, AttributeError):
            sigma = np.array([reference_process.get_diffusion(t, x) for x in current_x])
            
        # Reshape sigma for broadcasting: (N, D) or (N, 1)
        if sigma.ndim == 1: 
            sigma = sigma[:, np.newaxis]
        
        # C. Brownian Motion (dW)
        # Standard Gaussian noise
        dW = np.random.normal(0, np.sqrt(dt), size=(n_paths, dim))
        
        # D. Continuous Increment
        dX_cont = drift * dt + sigma * dW
        
        # E. Jump Increment (dJ)
        # Sample jumps for each path. 
        # Note: Optimization opportunity here if reference_process supports batch sampling.
        # For now, we keep the loop to ensure compatibility with generic reference classes.
        dJ_list = [reference_process.sample_jump(t, dt) for _ in range(n_paths)]
        dJ = np.array(dJ_list)
        
        # Ensure dJ shape (N, D)
        if dJ.ndim == 1: 
            dJ = dJ[:, np.newaxis]
            
        # F. Update State
        # Explicit Jump Diffusion: X_new = X_old + dX_cont + dJ
        current_x = current_x + dX_cont + dJ
        
        # Store
        paths[:, i+1, :] = current_x
        
    return paths   