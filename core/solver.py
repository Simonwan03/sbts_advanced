import numpy as np

def euler_maruyama_generator(x0, time_grid, drift_func, reference_process, n_paths=1):
    """
    Generates trajectories using the Advanced SBTS SDE.
    dX_t = alpha_t dt + sigma_ref(t, X_t) dW_t + dJ_t
    """
    N = len(time_grid)
    dt = time_grid[1] - time_grid[0]
    
    # --- FIX START ---
    # Determine dimension correctly. 
    # x0 can be shape (dim,) or (n_paths, dim)
    if x0.ndim > 1:
        dim = x0.shape[-1]
    else:
        dim = x0.shape[0]
    # --- FIX END ---
    
    paths = np.zeros((n_paths, N, dim))
    
    # Assign start points (broadcasting handles both single point and batch cases)
    paths[:, 0, :] = x0
    
    for i in range(N - 1):
        t = time_grid[i]
        current_x = paths[:, i, :]
        
        # 1. Compute Drift
        # Iterating over current_x gives individual vectors of shape (dim,)
        drift = np.array([drift_func(t, x) for x in current_x])
        
        # 2. Diffusion
        sigma = np.array([reference_process.get_diffusion(t, x) for x in current_x])
        # Reshape sigma for broadcasting if scalar
        if sigma.ndim == 1: sigma = sigma[:, np.newaxis]
        
        dW = np.random.normal(0, np.sqrt(dt), size=(n_paths, dim))
        
        # 3. Jumps
        dJ = np.array([reference_process.sample_jump(t, dt) for _ in range(n_paths)])
        if dJ.ndim == 1: dJ = dJ[:, np.newaxis]
        
        # Update
        paths[:, i+1, :] = current_x + drift * dt + sigma * dW + dJ
        
    return paths