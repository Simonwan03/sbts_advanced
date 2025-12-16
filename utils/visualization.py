import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- Nature-Style Configuration ---
def set_style():
    """Configures Matplotlib for Nature-journal quality."""
    plt.style.use('seaborn-v0_8-paper')
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("ticks")
    
    # Nature Colors: Blue, Red, Green, Purple
    nature_colors = ['#4DBBD5', '#E64B35', '#00A087', '#3C5488']
    sns.set_palette(nature_colors)
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.linewidth': 1.0,
        'axes.edgecolor': '#333333',
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'figure.dpi': 300,
        'legend.frameon': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'grid.color': '#F0F0F0',
        'grid.linestyle': '-',
        'grid.linewidth': 0.8,
    })

def _ensure_dir(path):
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

def _calc_autocorr(paths, max_lag=20):
    """Helper: Calculate auto-correlation with NaN handling."""
    if paths.ndim == 3:
        series = paths[:, :, 0]
    else:
        series = paths
        
    acfs = []
    for i in range(max_lag):
        if i >= series.shape[1] - 1:
            acfs.append(0)
            continue
        
        val_t = series[:, :-i-1].flatten()
        val_t_plus_k = series[:, i+1:].flatten()
        
        # Remove NaNs
        mask = ~np.isnan(val_t) & ~np.isnan(val_t_plus_k)
        clean_t = val_t[mask]
        clean_t_k = val_t_plus_k[mask]
        
        if len(clean_t) < 2:
            acfs.append(0)
        else:
            acfs.append(np.corrcoef(clean_t, clean_t_k)[0, 1])
    return np.array(acfs)

# --- Plotting Functions ---

def plot_bandwidth_optimization(mse_history, selected_h, save_path="outputs/bandwidth_tuning.png"):
    """Visualizes Cross-Validation MSE vs Bandwidth."""
    set_style()
    _ensure_dir(save_path)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    hs = sorted(list(mse_history.keys()))
    mses = [mse_history[h] for h in hs]
    
    ax.plot(hs, mses, marker='o', markersize=5, color='#3C5488', linewidth=1.5, label='CV MSE')
    ax.axvline(x=selected_h, color='#E64B35', linestyle='--', linewidth=2, label=f'Optimal h={selected_h:.3f}')
    
    ax.set_xscale('log')
    ax.set_xlabel("Bandwidth (h)")
    ax.set_ylabel("CV MSE (Terminal)")
    ax.set_title("Adaptive Bandwidth Selection", fontweight='bold', pad=15)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    sns.despine()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[Plot Saved] {save_path}")
    plt.close()

def plot_jump_detection(time_grid, path, jump_indices, save_path="outputs/jump_detection.png"):
    """Highlights detected jumps on a trajectory."""
    set_style()
    _ensure_dir(save_path)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time_grid, path, color='#333333', linewidth=1.0, label='Trajectory')
    
    jump_times = time_grid[1:][jump_indices]
    jump_values = path[1:][jump_indices]
    
    if len(jump_times) > 0:
        ax.scatter(jump_times, jump_values, color='#E64B35', s=40, zorder=5, label='Detected Jumps')
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title("Regime Shift / Jump Detection", fontweight='bold', pad=15)
    ax.legend()
    sns.despine()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[Plot Saved] {save_path}")
    plt.close()

def plot_method_comparison(time_grid, real_data, kernel_paths, lstm_paths, save_path="outputs/comparison.png"):
    """
    Comparison Plot: Real vs Kernel vs LSTM.
    Includes NaN safety checks to prevent blank plots.
    """
    set_style()
    _ensure_dir(save_path)
    
    # --- SAFETY: Filter NaNs ---
    def clean_paths(paths):
        if np.isnan(paths).any():
            print(f"[Warning] NaNs detected in generated paths. Filtering corrupted trajectories...")
            # Keep only paths that have NO NaNs across all time steps
            mask = ~np.isnan(paths).any(axis=(1, 2))
            cleaned = paths[mask]
            print(f"   Dropped {len(paths) - len(cleaned)} bad paths. Remaining: {len(cleaned)}")
            return cleaned
        return paths

    kernel_paths = clean_paths(kernel_paths)
    lstm_paths = clean_paths(lstm_paths)
    
    if len(kernel_paths) == 0 or len(lstm_paths) == 0:
        print("[Error] All paths corrupted. Cannot plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # --- 1. Trajectory Mean ---
    # Use nanmean for extra safety
    real_mean = np.nanmean(real_data[:,:,0], axis=0)
    kernel_mean = np.nanmean(kernel_paths[:,:,0], axis=0)
    lstm_mean = np.nanmean(lstm_paths[:,:,0], axis=0)
    
    # Plot a few transparency paths
    axes[0].plot(time_grid, kernel_paths[0:15,:,0].T, color='#4DBBD5', alpha=0.15)
    
    axes[0].plot(time_grid, real_mean, 'k--', linewidth=2.5, label='Real')
    axes[0].plot(time_grid, kernel_mean, color='#4DBBD5', linewidth=2.5, label='Kernel')
    axes[0].plot(time_grid, lstm_mean, color='#E64B35', linewidth=2.5, linestyle='-.', label='LSTM')
    
    axes[0].set_title("Trajectory Reconstruction", fontweight='bold')
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("State X_t")
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.2)

    # --- 2. Terminal Distribution ---
    real_term = real_data[:, -1, 0]
    kernel_term = kernel_paths[:, -1, 0]
    lstm_term = lstm_paths[:, -1, 0]
    
    # Filter NaNs for KDE
    sns.kdeplot(real_term[~np.isnan(real_term)], ax=axes[1], color='gray', fill=True, alpha=0.3, label='Real')
    sns.kdeplot(kernel_term[~np.isnan(kernel_term)], ax=axes[1], color='#4DBBD5', linewidth=2.5, label='Kernel')
    sns.kdeplot(lstm_term[~np.isnan(lstm_term)], ax=axes[1], color='#E64B35', linewidth=2.5, linestyle='--', label='LSTM')
    
    axes[1].set_title("Terminal Distribution", fontweight='bold')
    axes[1].set_xlabel("Value")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    axes[1].grid(False)

    # --- 3. Auto-correlation ---
    max_lag = 15
    lag_grid = np.arange(max_lag)
    
    acf_real = _calc_autocorr(real_data, max_lag)
    acf_kernel = _calc_autocorr(kernel_paths, max_lag)
    acf_lstm = _calc_autocorr(lstm_paths, max_lag)
    
    axes[2].plot(lag_grid, acf_real, 'ko-', linewidth=1.5, markersize=6, label='Real')
    axes[2].plot(lag_grid, acf_kernel, 's-', color='#4DBBD5', linewidth=1.5, markersize=6, label='Kernel')
    axes[2].plot(lag_grid, acf_lstm, '^--', color='#E64B35', linewidth=1.5, markersize=6, label='LSTM')
    
    axes[2].set_title("Temporal Dynamics (ACF)", fontweight='bold')
    axes[2].set_xlabel("Lag (steps)")
    axes[2].set_ylabel("Correlation")
    axes[2].legend()
    axes[2].grid(True, alpha=0.2)

    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[Plot Saved] {save_path}")
    plt.close()