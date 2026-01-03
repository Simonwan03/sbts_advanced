"""
Statistical Metrics for Time Series Generation Evaluation

This module provides statistical metrics for comparing real and
generated time series distributions.

Metrics:
1. Wasserstein Distance: Measures distribution distance
2. ACF Error: Autocorrelation function difference
3. Correlation Matrix Error: Cross-asset correlation preservation
4. Marginal Statistics: Mean, std, skewness, kurtosis
"""

import numpy as np
from scipy import stats
from scipy.stats import wasserstein_distance


def compute_wasserstein_distance(real_data, gen_data, n_features=None):
    """
    Compute average Wasserstein distance across features.
    
    Args:
        real_data: (N, T, D) real time series
        gen_data: (N, T, D) generated time series
        n_features: Number of features to evaluate (default: all)
        
    Returns:
        avg_wd: Average Wasserstein distance
        per_feature_wd: Per-feature Wasserstein distances
    """
    if n_features is None:
        n_features = real_data.shape[-1]
    
    n_features = min(n_features, real_data.shape[-1], gen_data.shape[-1])
    
    per_feature_wd = []
    for i in range(n_features):
        # Use terminal values
        real_terminal = real_data[:, -1, i]
        gen_terminal = gen_data[:, -1, i]
        
        wd = wasserstein_distance(real_terminal, gen_terminal)
        per_feature_wd.append(wd)
    
    avg_wd = np.mean(per_feature_wd)
    return avg_wd, per_feature_wd


def compute_acf(data, max_lag=20):
    """
    Compute average autocorrelation function.
    
    Args:
        data: (N, T, D) time series
        max_lag: Maximum lag to compute
        
    Returns:
        acf: (max_lag,) average ACF
    """
    N, T, D = data.shape
    max_lag = min(max_lag, T - 1)
    
    acf_all = []
    for i in range(D):
        feature_data = data[:, :, i].flatten()
        acf_feature = []
        for lag in range(max_lag):
            if lag == 0:
                acf_feature.append(1.0)
            else:
                corr = np.corrcoef(feature_data[:-lag], feature_data[lag:])[0, 1]
                acf_feature.append(corr if not np.isnan(corr) else 0.0)
        acf_all.append(acf_feature)
    
    return np.mean(acf_all, axis=0)


def compute_acf_error(real_data, gen_data, max_lag=20):
    """
    Compute ACF error between real and generated data.
    
    Args:
        real_data: (N, T, D) real time series
        gen_data: (N, T, D) generated time series
        max_lag: Maximum lag
        
    Returns:
        acf_mse: Mean squared error of ACF
        acf_real: Real data ACF
        acf_gen: Generated data ACF
    """
    acf_real = compute_acf(real_data, max_lag)
    acf_gen = compute_acf(gen_data, max_lag)
    
    acf_mse = np.mean((acf_real - acf_gen) ** 2)
    
    return acf_mse, acf_real, acf_gen


def compute_correlation_matrix(data):
    """
    Compute average correlation matrix across samples.
    
    Args:
        data: (N, T, D) time series
        
    Returns:
        corr_matrix: (D, D) correlation matrix
    """
    N, T, D = data.shape
    
    # Compute correlation for each sample and average
    corr_matrices = []
    for i in range(N):
        sample = data[i]  # (T, D)
        corr = np.corrcoef(sample.T)  # (D, D)
        if not np.any(np.isnan(corr)):
            corr_matrices.append(corr)
    
    if len(corr_matrices) == 0:
        return np.eye(D)
    
    return np.mean(corr_matrices, axis=0)


def compute_correlation_error(real_data, gen_data):
    """
    Compute correlation matrix error.
    
    Args:
        real_data: (N, T, D) real time series
        gen_data: (N, T, D) generated time series
        
    Returns:
        corr_error: Frobenius norm of correlation difference
        corr_real: Real correlation matrix
        corr_gen: Generated correlation matrix
    """
    corr_real = compute_correlation_matrix(real_data)
    corr_gen = compute_correlation_matrix(gen_data)
    
    corr_error = np.linalg.norm(corr_real - corr_gen, 'fro')
    
    return corr_error, corr_real, corr_gen


def compute_marginal_statistics(data):
    """
    Compute marginal statistics of time series.
    
    Args:
        data: (N, T, D) time series
        
    Returns:
        stats_dict: Dictionary of statistics
    """
    flat_data = data.reshape(-1, data.shape[-1])
    
    return {
        'mean': np.mean(flat_data, axis=0),
        'std': np.std(flat_data, axis=0),
        'skewness': stats.skew(flat_data, axis=0),
        'kurtosis': stats.kurtosis(flat_data, axis=0),
        'min': np.min(flat_data, axis=0),
        'max': np.max(flat_data, axis=0),
        'percentile_1': np.percentile(flat_data, 1, axis=0),
        'percentile_99': np.percentile(flat_data, 99, axis=0),
    }


def compute_marginal_error(real_data, gen_data):
    """
    Compute marginal statistics error.
    
    Args:
        real_data: (N, T, D) real time series
        gen_data: (N, T, D) generated time series
        
    Returns:
        error_dict: Dictionary of per-statistic errors
    """
    real_stats = compute_marginal_statistics(real_data)
    gen_stats = compute_marginal_statistics(gen_data)
    
    error_dict = {}
    for key in real_stats:
        error_dict[f'{key}_error'] = np.mean(np.abs(real_stats[key] - gen_stats[key]))
    
    return error_dict


def compute_all_metrics(real_data, gen_data, n_features=None, max_lag=20):
    """
    Compute all statistical metrics.
    
    Args:
        real_data: (N, T, D) real time series
        gen_data: (N, T, D) generated time series
        n_features: Number of features for WD (default: all)
        max_lag: Maximum lag for ACF
        
    Returns:
        metrics: Dictionary of all metrics
    """
    metrics = {}
    
    # Wasserstein Distance
    avg_wd, per_feature_wd = compute_wasserstein_distance(real_data, gen_data, n_features)
    metrics['wasserstein_distance'] = avg_wd
    metrics['wasserstein_per_feature'] = per_feature_wd
    
    # ACF Error
    acf_mse, acf_real, acf_gen = compute_acf_error(real_data, gen_data, max_lag)
    metrics['acf_mse'] = acf_mse
    metrics['acf_real'] = acf_real
    metrics['acf_gen'] = acf_gen
    
    # Correlation Error
    corr_error, corr_real, corr_gen = compute_correlation_error(real_data, gen_data)
    metrics['correlation_error'] = corr_error
    metrics['correlation_real'] = corr_real
    metrics['correlation_gen'] = corr_gen
    
    # Marginal Statistics Error
    marginal_errors = compute_marginal_error(real_data, gen_data)
    metrics.update(marginal_errors)
    
    return metrics
