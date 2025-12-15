import numpy as np
from sklearn.model_selection import KFold
import logging

class BandwidthSelector:
    """
    Implements Adaptive Bandwidth Selection via Cross-Validation.
    Objectives: Replace Silverman's rule, minimize prediction error on test sets.
    """
    def __init__(self, candidates=None, n_splits=5):
        # As per PDF: np.logspace(-2, 1, 50)
        self.candidates = candidates if candidates is not None else np.logspace(-2, 1, 50)
        self.n_splits = n_splits
        self.best_h = None
        self.mse_history = {}

    def compute_terminal_mse(self, generated_terminal, actual_terminal):
        """
        Metric: Minimize Mean Squared Error (MSE) of terminal values.
        MSE_h = (1/Q) * Sum_q [(1/L) * Sum_l (Y_hat - Y)^2]
        """
        # generated_terminal shape: (n_validation_samples, n_generated_per_sample)
        # actual_terminal shape: (n_validation_samples,)
        
        # Expand actual to match generated shape for vectorized calc
        actual_expanded = actual_terminal[:, np.newaxis]
        
        squared_diffs = (generated_terminal - actual_expanded) ** 2
        mse = np.mean(squared_diffs)
        return mse

    def fit(self, trajectories, drift_estimator_func, simulator_func):
        """
        Grid search over bandwidth space.
        
        trajectories: (N, T, D) array of training paths
        drift_estimator_func: function(data, h) -> drift_fn
        simulator_func: function(start_points, drift_fn) -> terminal_values
        """
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        # Flatten trajectories for splitting if necessary, or split by ID
        n_trajs = trajectories.shape[0]
        indices = np.arange(n_trajs)
        
        scores = {h: [] for h in self.candidates}

        for train_idx, val_idx in kf.split(indices):
            train_data = trajectories[train_idx]
            val_data = trajectories[val_idx]
            
            val_start_points = val_data[:, 0, :]
            val_terminal_actual = val_data[:, -1, :]

            for h in self.candidates:
                # 1. Build drift with candidate h
                drift_fn = drift_estimator_func(train_data, h)
                
                # 2. Simulate trajectories using validation start points
                # Returns shape (n_val, n_samples_per_traj, D)
                gen_terminal = simulator_func(val_start_points, drift_fn)
                
                # 3. Compute MSE
                mse = self.compute_terminal_mse(gen_terminal, val_terminal_actual)
                scores[h].append(mse)

        # Average scores and find min
        avg_scores = {h: np.mean(val) for h, val in scores.items()}
        self.best_h = min(avg_scores, key=avg_scores.get)
        self.mse_history = avg_scores
        
        logging.info(f"Optimal Bandwidth selected: {self.best_h:.4f}")
        return self.best_h