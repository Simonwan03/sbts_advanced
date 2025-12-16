import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import warnings

class VolatilityCalibrator:
    """
    Implements Local Volatility Calibration (Phase 3).
    Fits a surface sigma(t, x) based on realized volatility of training paths.
    
    [Update]: Includes input feature standardization to prevent numerical instability (NaNs).
    """
    def __init__(self, dt, method='kernel', bandwidth=0.1):
        self.dt = dt
        self.method = method
        self.bandwidth = bandwidth # acts as gamma (kernel) or n_neighbors factor (knn)
        self.model = None
        self.scaler_X = StandardScaler() # Added: Normalize inputs (t, x)
        self.min_vol = 1e-4

    def _compute_instantaneous_vol(self, trajectories):
        """
        Estimate instantaneous volatility squared from discrete paths.
        sigma^2(t) approx (dX_t)^2 / dt
        """
        # trajectories: (N, T, D)
        # Calculate increments: dX = X_{t+1} - X_t
        X_current = trajectories[:, :-1, :]
        X_next = trajectories[:, 1:, :]
        dX = X_next - X_current
        
        # Realized Variance approx: (dX)^2 / dt
        # Shape: (N, T-1, D)
        realized_var = (dX ** 2) / self.dt
        
        # Flatten for regression: Inputs (t, x), Target (sigma^2)
        N, T_minus_1, D = realized_var.shape
        
        # Create time grid for T-1 steps
        t_grid = np.linspace(0, self.dt * T_minus_1, T_minus_1)
        
        # Prepare training arrays
        X_features = []
        Y_target = []
        
        for i in range(T_minus_1):
            t = t_grid[i]
            # Current states at time t: (N, D)
            current_x = X_current[:, i, :]
            # Variances at time t: (N, D)
            current_var = realized_var[:, i, :]
            
            # Feature vector: [t, x1, x2...]
            t_col = np.full((N, 1), t)
            
            feat = np.hstack([t_col, current_x])
            X_features.append(feat)
            Y_target.append(current_var)
            
        return np.vstack(X_features), np.vstack(Y_target)

    def fit(self, trajectories):
        """
        Fits the Local Volatility model sigma(t, x).
        """
        print("   [Calibration] Computing realized volatility surface...")
        X_train, Y_train = self._compute_instantaneous_vol(trajectories)
        
        # Target: Volatility (sqrt of variance)
        # Prevent sqrt of negative numbers due to numerical noise
        Y_vol = np.sqrt(np.maximum(Y_train, 1e-8))
        
        # Subsample for speed if data is too large
        limit = 20000
        if len(X_train) > limit:
            idx = np.random.choice(len(X_train), limit, replace=False)
            X_train = X_train[idx]
            Y_vol = Y_vol[idx]

        # --- CRITICAL FIX: Scaling ---
        # Standardize features (t, x) so KernelRidge works correctly
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        # -----------------------------

        if self.method == 'kernel':
            # Gamma depends on the scale. With normalized data, 
            # gamma approx 1/(2*sigma^2) where sigma is bandwidth
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
        x: (D,) or (1, D)
        """
        if self.model is None:
            raise ValueError("Calibrator not fitted.")
            
        if x.ndim == 1:
            x = x[np.newaxis, :]
            
        # Construct query: [t, x]
        t_col = np.full((x.shape[0], 1), t)
        query = np.hstack([t_col, x])
        
        # --- CRITICAL FIX: Predict on Scaled Data ---
        query_scaled = self.scaler_X.transform(query)
        pred_vol = self.model.predict(query_scaled)
        # --------------------------------------------
        
        # Ensure positive volatility
        return np.maximum(pred_vol, self.min_vol)