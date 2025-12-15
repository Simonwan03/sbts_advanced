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
    # Used for: Generated(Blue/Red), Real(Gray/Black)
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
    """Helper: Calculate auto-correlation for plotting."""
    # paths: (N, T, D) or (N, T)
    if paths.ndim == 3:
        series = paths[:, :, 0]
    else:
        series = paths
        
    acfs = []
    for i in range(max_lag):
        if i >= series.shape[1] - 1:
            acfs.append(0)
            continue
        # Flatten to calc correlation across all paths simultaneously
        val_t = series[:, :-i-1].flatten()
        val_t_plus_k = series[:, i+1:].flatten()
        if len(val_t) < 2:
            acfs.append(0)
        else:
            acfs.append(np.corrcoef(val_t, val_t_plus_k)[0, 1])
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
    3 Subplots: Trajectory Mean, Terminal Density, Auto-correlation.
    """
    set_style()
    _ensure_dir(save_path)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # --- 1. Trajectory Reconstruction (Mean) ---
    real_mean = np.mean(real_data[:,:,0], axis=0)
    kernel_mean = np.mean(kernel_paths[:,:,0], axis=0)
    lstm_mean = np.mean(lstm_paths[:,:,0], axis=0)
    
    # Plot faint individual paths for context (Kernel)
    axes[0].plot(time_grid, kernel_paths[0:15,:,0].T, color='#4DBBD5', alpha=0.15)
    
    # Means
    axes[0].plot(time_grid, real_mean, 'k--', linewidth=2.5, label='Real')
    axes[0].plot(time_grid, kernel_mean, color='#4DBBD5', linewidth=2.5, label='Kernel')
    axes[0].plot