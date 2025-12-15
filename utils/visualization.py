import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import os

# --- Nature-Style Plotting Configuration ---
def set_style():
    """
    Configures Matplotlib to produce Nature-journal quality plots.
    """
    plt.style.use('seaborn-v0_8-paper')
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("ticks")
    
    # Nature standardized colors (approximate)
    # Red, Blue, Green, Purple, Orange, Gray
    nature_colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#7E6148']
    sns.set_palette(nature_colors)
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
        'axes.linewidth': 1.0,
        'axes.edgecolor': '#333333',
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'legend.frameon': False,
    })

def plot_trajectories(time_grid, real_paths, gen_paths, save_path="trajectory_comparison.png"):
    """
    Compares Real vs Generated Trajectories.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot a subset of real paths
    for i in range(min(50, real_paths.shape[0])):
        ax.plot(time_grid, real_paths[i, :, 0], color='gray', alpha=0.2, linewidth=0.8)
    
    # Plot a subset of generated paths (using Nature Blue)
    for i in range(min(10, gen_paths.shape[0])):
        ax.plot(time_grid, gen_paths[i, :, 0], color='#4DBBD5', alpha=0.8, linewidth=1.2)
        
    # Plot Mean paths
    real_mean = np.mean(real_paths[:, :, 0], axis=0)
    gen_mean = np.mean(gen_paths[:, :, 0], axis=0)
    
    ax.plot(time_grid, real_mean, color='black', linestyle='--', linewidth=1.5, label='Real Mean')
    ax.plot(time_grid, gen_mean, color='#E64B35', linewidth=1.5, label='Generated Mean') # Nature Red

    ax.set_xlabel("Time (t)")
    ax.set_ylabel("State (X_t)")
    ax.set_title("Trajectory Reconstruction", fontweight='bold')
    ax.legend()
    sns.despine(offset=10, trim=True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"[Plot Saved] {save_path}")
    plt.show()

def plot_bandwidth_optimization(mse_history, selected_h, save_path="bandwidth_tuning.png"):
    """
    Visualizes the Cross-Validation Error vs Bandwidth.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(5, 4))
    
    hs = sorted(list(mse_history.keys()))
    mses = [mse_history[h] for h in hs]
    
    ax.plot(hs, mses, marker='o', markersize=4, color='#3C5488', linewidth=1.5)
    
    # Highlight selected h
    ax.axvline(x=selected_h, color='#E64B35', linestyle='--', label=f'Optimal h={selected_h:.3f}')
    
    ax.set_xscale('log')
    ax.set_xlabel("Bandwidth (h)")
    ax.set_ylabel("CV MSE (Terminal)")
    ax.set_title("Adaptive Bandwidth Selection", fontweight='bold')
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.1)
    sns.despine()
    
    if save_path:
        plt.savefig(save_path)
        print(f"[Plot Saved] {save_path}")
    plt.show()

def plot_marginal_density(real_data, gen_data, t_step=-1, save_path="marginal_density.png"):
    """
    Compares the marginal distribution at a specific time step (default: terminal).
    """
    set_style()
    fig, ax = plt.subplots(figsize=(5, 4))
    
    real_vals = real_data[:, t_step, 0]
    gen_vals = gen_data[:, t_step, 0]
    
    sns.kdeplot(real_vals, color='gray', fill=True, alpha=0.3, label='Real', ax=ax)
    sns.kdeplot(gen_vals, color='#E64B35', fill=False, linewidth=2, label='Generated', ax=ax)
    
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title(f"Terminal Distribution Comparison", fontweight='bold')
    ax.legend()
    sns.despine()
    
    if save_path:
        plt.savefig(save_path)
        print(f"[Plot Saved] {save_path}")
    plt.show()

def plot_jump_detection(time_grid, path, jump_indices, save_path="jump_detection.png"):
    """
    Highlights detected jumps on a single trajectory.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(8, 3))
    
    ax.plot(time_grid, path, color='#333333', linewidth=1.0, label='Trajectory')
    
    # Highlight jumps
    # jump_indices corresponds to returns, so indices map to time[1:]
    jump_times = time_grid[1:][jump_indices]
    jump_values = path[1:][jump_indices]
    
    if len(jump_times) > 0:
        ax.scatter(jump_times, jump_values, color='#E64B35', s=30, zorder=5, label='Detected Jumps')
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Asset Price / Value")
    ax.set_title("Regime Shift / Jump Detection", fontweight='bold')
    ax.legend()
    sns.despine()
    
    if save_path:
        plt.savefig(save_path)
        print(f"[Plot Saved] {save_path}")
    plt.show()