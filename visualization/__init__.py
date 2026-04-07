"""
Visualization Module

Provides comprehensive visualization tools for time series generation analysis.

Modules:
    - general: General experiment and benchmark plots
    - feedback_plots: Specialized plots for JD-SBTS-F feedback mechanism

Author: Manus AI
"""

from visualization.general import (
    set_style,
    plot_performance_metrics,
    plot_comprehensive_comparison,
    plot_bandwidth_optimization,
    plot_correlation_distribution,
    plot_price_index_comparison,
    plot_volatility_surface,
    plot_multi_asset_jumps,
    plot_zoomed_crisis,
    plot_jumps_on_price,
    plot_sota_comparison,
    plot_discriminative_results,
    plot_method_ranking,
    _calc_autocorr,
)
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
    # General visualization
    'set_style',
    'plot_performance_metrics',
    'plot_comprehensive_comparison',
    'plot_bandwidth_optimization',
    'plot_correlation_distribution',
    'plot_price_index_comparison',
    'plot_volatility_surface',
    'plot_multi_asset_jumps',
    'plot_zoomed_crisis',
    'plot_jumps_on_price',
    'plot_sota_comparison',
    'plot_discriminative_results',
    'plot_method_ranking',
    '_calc_autocorr',

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
