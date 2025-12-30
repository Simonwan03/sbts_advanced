import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import warnings

class VolatilityCalibrator:
    """
    Implements Local Volatility Calibration (Phase 3).
    Fits a surface sigma(t, x) based on realized volatility of training paths.
    """
    def __init__(self, dt, method='kernel', bandwidth=0.1):
        self.dt = dt
        self.method = method
        self.bandwidth = bandwidth 
        self.model = None
        self.scaler_X = StandardScaler() 
        self.min_vol = 1e-4

    def _compute_instantaneous_vol(self, trajectories):
        X_current = trajectories[:, :-1, :]
        X_next = trajectories[:, 1:, :]
        dX = X_next - X_current
        
        realized_var = (dX ** 2) / self.dt
        
        N, T_minus_1, D = realized_var.shape
        t_grid = np.linspace(0, self.dt * T_minus_1, T_minus_1)
        
        X_features = []
        Y_target = []
        
        for i in range(T_minus_1):
            t = t_grid[i]
            current_x = X_current[:, i, :]
            current_var = realized_var[:, i, :]
            
            t_col = np.full((N, 1), t)
            
            feat = np.hstack([t_col, current_x])
            X_features.append(feat)
            Y_target.append(current_var)
            
        return np.vstack(X_features), np.vstack(Y_target)

    def fit(self, trajectories):
        print("   [Calibration] Computing realized volatility surface...")
        X_train, Y_train = self._compute_instantaneous_vol(trajectories)
        
        Y_vol = np.sqrt(np.maximum(Y_train, 1e-8))
        
        limit = 20000
        if len(X_train) > limit:
            idx = np.random.choice(len(X_train), limit, replace=False)
            X_train = X_train[idx]
            Y_vol = Y_vol[idx]

        # Standardize features (t, x)
        X_train_scaled = self.scaler_X.fit_transform(X_train)

        if self.method == 'kernel':
            gamma = 1.0 / (2 * self.bandwidth ** 2)
            self.model = KernelRidge(kernel='rbf', gamma=gamma, alpha=0.1)
        elif self.method == 'knn':
            n_neighbors = int(max(5, 100 * self.bandwidth))
            self.model = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X_train_scaled, Y_vol)
            
        print("   [Calibration] Local Volatility Surface fitted (Normalized).")
        return self

    def predict(self, t, x):
        """
        Predicts volatility sigma(t, x).
        x: (N, D)
        t: scalar OR array of shape (N,)
        """
        if self.model is None:
            raise ValueError("Calibrator not fitted.")
            
        if x.ndim == 1:
            x = x[np.newaxis, :]
            
        # --- FIX: Handle Vectorized Time Input ---
        t = np.asarray(t)
        if t.ndim == 0:
            # Scalar case: Broadcast to (N, 1)
            t_col = np.full((x.shape[0], 1), t.item())
        else:
            # Vector case: Reshape to (N, 1)
            if t.shape[0] != x.shape[0]:
                raise ValueError(f"Time dimension {t.shape} does not match State dimension {x.shape}")
            t_col = t.reshape(-1, 1)
        # -----------------------------------------

        query = np.hstack([t_col, x])
        
        query_scaled = self.scaler_X.transform(query)
        pred_vol = self.model.predict(query_scaled)
        
        return np.maximum(pred_vol, self.min_vol)