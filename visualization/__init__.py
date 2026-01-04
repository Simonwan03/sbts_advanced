"""
Visualization Module

Provides comprehensive visualization tools for time series generation analysis.

Modules:
    - feedback_plots: Specialized plots for JD-SBTS-F feedback mechanism
    - comparison_plots: Model comparison and benchmark visualizations
    - diagnostic_plots: Training diagnostics and model inspection

Author: Manus AI
"""

from visualization.feedback_plots import (
    plot_stress_factor_dynamics,
    plot_feedback_comparison,
    plot_jump_volatility_interaction,
    plot_model_comparison_grid,
    create_legend_figure,
    set_publication_style,
    STYLE_CONFIG
)

__all__ = [
    # Feedback visualization
    'plot_stress_factor_dynamics',
    'plot_feedback_comparison',
    'plot_jump_volatility_interaction',
    
    # Model comparison
    'plot_model_comparison_grid',
    'create_legend_figure',
    
    # Style utilities
    'set_publication_style',
    'STYLE_CONFIG',
]
