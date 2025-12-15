import numpy as np
from scipy.stats import entropy

def discriminative_score(real, synthetic):
    """
    Placeholder for classifier accuracy distinguishing real vs synthetic.
    Target < 0.15 for Energy datasets.
    """
    # Requires training a simple RNN/MLP classifier
    pass

def wasserstein_distance_1d(u_values, v_values):
    from scipy.stats import wasserstein_distance
    return wasserstein_distance(u_values, v_values)

def compute_tail_risk_metrics(data, alpha=0.95):
    """
    Returns VaR and ES (Expected Shortfall)
    """
    sorted_data = np.sort(data, axis=0)
    index = int((1-alpha) * len(data))
    var = -sorted_data[index]
    es = -np.mean(sorted_data[:index], axis=0)
    return var, es