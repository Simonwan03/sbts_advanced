import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import warnings

class VolatilityCalibrator:
    """
    Implements Local Volatility Calibration (Phase 3).
    Fits a surface σ_LV(t, x) based on realized volatility of training paths.
    
    REFACTORED (Step 1): 
    - Now accepts purified returns from JumpDetector
    - Uses Filter & Interpolate purification instead of Clipping
    - Results in proper "Smile/Skew" volatility surface shape
    
    The volatility surface should exhibit:
    - Higher volatility at extreme values (tails)
    - Lower volatility near the mean
    - NOT an "inverted U" shape (which indicates incorrect purification)
    """
    def __init__(self, dt, method='kernel', bandwidth=0.1):
        self.dt = dt
        self.method = method
        self.bandwidth = bandwidth 
        self.model = None
        self.scaler_X = StandardScaler() 
        self.min_vol = 1e-4
        self._surface_diagnostics = {}  # Store diagnostics for validation

    def _compute_instantaneous_vol(self, trajectories):
        """
        Compute instantaneous realized volatility from trajectory data.
        
        Args:
            trajectories: (N, T, D) array of price/return paths
            
        Returns:
            X_features: (N*(T-1), 1+D) array of (t, x) features
            Y_target: (N*(T-1), D) array of realized volatility
        """
        X_current = trajectories[:, :-1, :]
        X_next = trajectories[:, 1:, :]
        dX = X_next - X_current
        
        # Realized variance: (dX)^2 / dt
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

    def fit(self, trajectories, purified_trajectories=None):
        """
        Fit the Local Volatility surface using Kernel Ridge Regression.
        
        IMPORTANT: For correct volatility smile/skew, use purified_trajectories
        from JumpDetector.get_purified_returns() instead of raw data.
        
        Args:
            trajectories: (N, T, D) array of original paths (for feature extraction)
            purified_trajectories: (N, T, D) array of purified paths (for vol estimation)
                                   If None, uses trajectories directly
                                   
        Returns:
            self: Fitted calibrator
        """
        # Use purified data for volatility estimation if provided
        data_for_vol = purified_trajectories if purified_trajectories is not None else trajectories
        
        print("   [Calibration] Computing realized volatility surface...")
        if purified_trajectories is not None:
            print("   [Calibration] Using Filter & Interpolate purified returns")
        
        X_train, Y_train = self._compute_instantaneous_vol(data_for_vol)
        
        # Convert variance to volatility (standard deviation)
        Y_vol = np.sqrt(np.maximum(Y_train, 1e-8))
        
        # Subsample for efficiency
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
        
        # Store diagnostics for validation
        self._compute_surface_diagnostics(X_train, Y_vol)
            
        print("   [Calibration] Local Volatility Surface fitted (Normalized).")
        return self

    def _compute_surface_diagnostics(self, X_train, Y_vol):
        """
        Compute diagnostics to verify volatility surface shape.
        
        A correct surface should show:
        - Higher volatility at extreme x values (smile/skew)
        - NOT highest volatility at x=0 (inverted U)
        
        Args:
            X_train: Feature array (t, x)
            Y_vol: Volatility targets
        """
        # Extract x values (assuming 1D for simplicity)
        x_values = X_train[:, 1] if X_train.shape[1] > 1 else X_train[:, 0]
        
        # Compute volatility by x-quantile
        quantiles = np.percentile(x_values, [10, 25, 50, 75, 90])
        vol_by_quantile = []
        
        for i in range(len(quantiles) - 1):
            mask = (x_values >= quantiles[i]) & (x_values < quantiles[i+1])
            if np.any(mask):
                vol_by_quantile.append(np.mean(Y_vol[mask]))
            else:
                vol_by_quantile.append(np.nan)
        
        # Check for inverted U shape (center > edges)
        if len(vol_by_quantile) >= 3:
            center_vol = vol_by_quantile[len(vol_by_quantile)//2]
            edge_vol = (vol_by_quantile[0] + vol_by_quantile[-1]) / 2
            
            self._surface_diagnostics = {
                "center_volatility": center_vol,
                "edge_volatility": edge_vol,
                "is_smile_shape": edge_vol > center_vol,
                "vol_by_quantile": vol_by_quantile
            }
            
            if edge_vol > center_vol:
                print("   [Calibration] ✓ Volatility surface exhibits Smile/Skew shape")
            else:
                print("   [Calibration] ⚠ Warning: Surface may have inverted U shape")

    def predict(self, t, x):
        """
        Predicts volatility σ_LV(t, x).
        
        Args:
            x: (N, D) array of state values
            t: scalar OR array of shape (N,) for time
            
        Returns:
            vol: (N, D) array of predicted volatilities
        """
        if self.model is None:
            raise ValueError("Calibrator not fitted.")
            
        if x.ndim == 1:
            x = x[np.newaxis, :]
            
        # Handle Vectorized Time Input
        t = np.asarray(t)
        if t.ndim == 0:
            # Scalar case: Broadcast to (N, 1)
            t_col = np.full((x.shape[0], 1), t.item())
        else:
            # Vector case: Reshape to (N, 1)
            if t.shape[0] != x.shape[0]:
                raise ValueError(f"Time dimension {t.shape} does not match State dimension {x.shape}")
            t_col = t.reshape(-1, 1)

        query = np.hstack([t_col, x])
        
        query_scaled = self.scaler_X.transform(query)
        pred_vol = self.model.predict(query_scaled)
        
        return np.maximum(pred_vol, self.min_vol)

    def get_surface_diagnostics(self):
        """
        Returns diagnostics about the fitted volatility surface.
        
        Returns:
            dict: Surface shape diagnostics
        """
        return self._surface_diagnostics
