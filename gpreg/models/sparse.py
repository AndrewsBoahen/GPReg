"""Sparse Gaussian Process Regression via the FITC approximation.

Implements the Fully Independent Training Conditional (FITC) sparse GP
of Snelson & Ghahramani (2006). Reduces complexity from O(n^3) to
O(n*m^2), where m is the number of inducing points.

Math summary (notation: f = latent function values, u = inducing values,
K_nm = cross-covariance between training and inducing inputs):

    Q_nn = K_nm @ K_mm^-1 @ K_mn   (low-rank approximation to K_nn)
    Λ = diag(K_nn - Q_nn) + σ²I    (FITC's diagonal correction)
    Approximated likelihood: y ~ N(0, Q_nn + Λ)

The diagonal Λ preserves marginal variances exactly while making the
off-diagonal structure low-rank — that's what gives FITC better
calibration than the simpler DTC approximation.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..kernels import RBF, White
from ..utils.linalg import stable_cholesky, cholesky_solve, log_det_from_cholesky
from ..utils.exceptions import (
    NotFittedError,
    ConvergenceError,
    KernelDimensionError,
)


class SparseGPRegressor:
    """Sparse Gaussian Process Regression via FITC.
    
    Suitable for datasets where the standard exact GP would be too
    slow (typically n > 5000). Trades a small amount of accuracy for
    a large speedup.
    
    Parameters
    ----------
    kernel : Kernel, default=RBF() + White(0.1)
        Covariance kernel. Should include a White noise component.
    n_inducing : int, default=100
        Number of inducing points (m). More inducing points -> closer
        to exact GP, but slower. Rule of thumb: 100-500 works well
        for most problems.
    inducing_strategy : {'kmeans', 'random', 'subset'}, default='kmeans'
        How to initialize inducing point locations:
        - 'kmeans': cluster centers of the training inputs (recommended)
        - 'random': random subset of training inputs
        - 'subset': first n_inducing training points (for reproducibility)
    optimize_inducing : bool, default=False
        Whether to also optimize inducing point locations along with
        kernel hyperparameters. More flexible but much slower.
    normalize_y : bool, default=True
    n_restarts : int, default=3
    random_state : int or None, default=None
    """
    
    def __init__(self, kernel=None, n_inducing=100,
                 inducing_strategy="kmeans", optimize_inducing=False,
                 normalize_y=True, n_restarts=3, random_state=None):
        self.kernel = kernel if kernel is not None else RBF() + White(noise_var=0.1)
        self.n_inducing = n_inducing
        self.inducing_strategy = inducing_strategy
        self.optimize_inducing = optimize_inducing
        self.normalize_y = normalize_y
        self.n_restarts = n_restarts
        self.random_state = random_state
        self._fitted = False
    
    def _to_array(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X = X.values
        else:
            self.feature_names_ = None
            X = np.atleast_2d(np.asarray(X, dtype=float))
            if X.ndim == 1:
                X = X.reshape(-1, 1)
        
        if y is not None:
            if isinstance(y, (pd.Series, pd.DataFrame)):
                y = y.values
            y = np.asarray(y, dtype=float).ravel()
            return X, y
        
        return X
    
    def _init_inducing_points(self, X):
        """Pick initial inducing point locations from the training data."""
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        m = min(self.n_inducing, n)
        
        if self.inducing_strategy == "kmeans":
            # Use k-means cluster centers. We import sklearn lazily
            # to keep it an optional dependency for users who only
            # want random/subset initialization.
            try:
                from sklearn.cluster import KMeans
                km = KMeans(n_clusters=m, random_state=self.random_state, n_init=3)
                km.fit(X)
                return km.cluster_centers_.astype(float)
            except ImportError:
                # Fall back to random if sklearn isn't available
                idx = rng.choice(n, size=m, replace=False)
                return X[idx].astype(float).copy()
        elif self.inducing_strategy == "random":
            idx = rng.choice(n, size=m, replace=False)
            return X[idx].astype(float).copy()
        elif self.inducing_strategy == "subset":
            return X[:m].astype(float).copy()
        else:
            raise ValueError(
                f"Unknown inducing_strategy {self.inducing_strategy!r}. "
                f"Choose from 'kmeans', 'random', 'subset'."
            )
    
    def _extract_noise_var(self):
        """Pull the noise variance out of the kernel.
        
        We assume the kernel has the form (something) + White(noise_var)
        and need the White component separately because FITC's diagonal
        correction is built around it.
        
        If the user supplies a kernel without a White component, we add
        a small floor noise.
        """
        # Walk the kernel tree looking for a White kernel
        from ..kernels.standard import White
        from ..kernels.base import SumKernel
        
        def find_white(k):
            if isinstance(k, White):
                return k
            if isinstance(k, SumKernel):
                w = find_white(k.k1)
                if w is not None:
                    return w
                return find_white(k.k2)
            return None
        
        white = find_white(self.kernel)
        if white is None:
            return 1e-4  # default noise floor
        return white.noise_var
    
    def _signal_kernel(self, X1, X2=None):
        """Evaluate the kernel WITHOUT the white-noise diagonal contribution.
        
        FITC needs the noiseless prior covariance for the inducing-point
        math; the noise enters explicitly through the diagonal correction.
        """
        from ..kernels.standard import White
        from ..kernels.base import SumKernel
        
        def eval_no_white(k, X1, X2):
            if isinstance(k, White):
                # White contributes nothing to off-diagonal cross-covariance
                if X2 is None or (X1.shape == X2.shape and np.array_equal(X1, X2)):
                    return np.zeros((X1.shape[0], X1.shape[0]))
                return np.zeros((X1.shape[0], X2.shape[0]))
            if isinstance(k, SumKernel):
                return eval_no_white(k.k1, X1, X2) + eval_no_white(k.k2, X1, X2)
            # Non-white component evaluates normally
            return k(X1, X2)
        
        return eval_no_white(self.kernel, X1, X2)
    
    def _fitc_components(self, X, Z):
        """Compute the matrices FITC needs: K_mm, K_nm, diagonal of Q_nn.
        
        Returns
        -------
        K_mm : (m, m) inducing-inducing covariance (signal only)
        K_nm : (n, m) cross-covariance between training and inducing
        Knn_diag : (n,) diagonal of the full training kernel (signal only)
        sigma2 : float, noise variance
        """
        K_mm = self._signal_kernel(Z)
        K_nm = self._signal_kernel(X, Z)
        # Just the diagonal of K_nn — much cheaper than the full matrix
        # For most kernels the diagonal equals signal_var; we compute it
        # generically as the diagonal of K(x_i, x_i).
        Knn_diag = np.array([self._signal_kernel(X[i:i+1])[0, 0] for i in range(X.shape[0])])
        sigma2 = self._extract_noise_var()
        return K_mm, K_nm, Knn_diag, sigma2
    
    def _neg_log_marginal_likelihood(self, theta, X, y, Z):
        """FITC marginal log-likelihood (Snelson & Ghahramani, Eq. 8).
        
        log q(y) = -0.5 y^T (Q_nn + Λ)^-1 y
                   -0.5 log|Q_nn + Λ|
                   -n/2 log(2π)
        
        We use the matrix inversion lemma to evaluate this without
        forming the n×n matrix Q_nn + Λ explicitly.
        """
        try:
            self.kernel.set_param_vector(theta)
        except Exception:
            return np.inf
        
        try:
            K_mm, K_nm, Knn_diag, sigma2 = self._fitc_components(X, Z)
            
            # Cholesky of K_mm
            L_mm, _ = stable_cholesky(K_mm)
            
            # Q_nn diagonal: q_ii = k_im^T K_mm^-1 k_im
            # Compute V = L_mm^-1 K_mn^T  (m, n).  Then Q_nn diag = sum(V**2, axis=0).
            V = np.linalg.solve(L_mm, K_nm.T)  # shape (m, n)
            Qnn_diag = np.sum(V ** 2, axis=0)
            
            # Λ = diag(K_nn - Q_nn) + σ²
            Lambda = Knn_diag - Qnn_diag + sigma2
            # Floor to avoid numerical issues
            Lambda = np.maximum(Lambda, 1e-8)
            
            # Apply Woodbury / matrix inversion lemma:
            # (Λ + V^T V)^-1 = Λ^-1 - Λ^-1 V^T (I + V Λ^-1 V^T)^-1 V Λ^-1
            # Build the m×m matrix B = I + V Λ^-1 V^T
            V_scaled = V / np.sqrt(Lambda)  # (m, n)
            B = np.eye(V.shape[0]) + V_scaled @ V_scaled.T
            L_B, _ = stable_cholesky(B)
            
            # Quadratic form: y^T (Q+Λ)^-1 y
            y_scaled = y / np.sqrt(Lambda)
            c = V_scaled @ y_scaled  # (m,)
            quad = y_scaled @ y_scaled - c @ cholesky_solve(L_B, c)
            
            # log determinant: log|Q+Λ| = log|Λ| + log|B|
            log_det = np.sum(np.log(Lambda)) + log_det_from_cholesky(L_B)
            
            log_lik = -0.5 * quad - 0.5 * log_det - 0.5 * len(y) * np.log(2 * np.pi)
            return -log_lik
        except Exception:
            return np.inf
    
    def fit(self, X, y):
        """Fit the sparse GP by optimizing kernel hyperparameters.
        
        Inducing point locations are initialized once and held fixed
        unless optimize_inducing=True (not yet implemented in this version).
        """
        X, y = self._to_array(X, y)
        
        if self.normalize_y:
            self.y_mean_ = float(y.mean())
            y_centered = y - self.y_mean_
        else:
            self.y_mean_ = 0.0
            y_centered = y
        
        # Initialize inducing points
        self.Z_ = self._init_inducing_points(X)
        
        if self.optimize_inducing:
            # Not implemented in this minimal version — would require
            # adding Z to the parameter vector and computing gradients
            # through the FITC objective.
            raise NotImplementedError(
                "Optimizing inducing point locations is not yet supported. "
                "Set optimize_inducing=False and rely on the kmeans/random "
                "initialization."
            )
        
        rng = np.random.default_rng(self.random_state)
        starts = [self.kernel.get_param_vector()]
        for _ in range(self.n_restarts):
            starts.append(rng.normal(0.0, 1.5, size=self.kernel.n_params))
        
        best_result = None
        best_nll = np.inf
        
        for theta0 in starts:
            try:
                result = minimize(
                    self._neg_log_marginal_likelihood,
                    theta0,
                    args=(X, y_centered, self.Z_),
                    method="L-BFGS-B",
                )
                if result.fun < best_nll:
                    best_nll = result.fun
                    best_result = result
            except Exception:
                continue
        
        if best_result is None:
            raise ConvergenceError(
                "All hyperparameter optimization restarts failed for sparse GP."
            )
        
        self.kernel.set_param_vector(best_result.x)
        
        # Cache prediction-time matrices
        K_mm, K_nm, Knn_diag, sigma2 = self._fitc_components(X, self.Z_)
        self._L_mm, _ = stable_cholesky(K_mm)
        V = np.linalg.solve(self._L_mm, K_nm.T)
        Qnn_diag = np.sum(V ** 2, axis=0)
        Lambda = np.maximum(Knn_diag - Qnn_diag + sigma2, 1e-8)
        
        # Pre-compute alpha_m and Σ_m for prediction
        # Posterior over u: q(u) = N(u | μ_u, Σ_u) with
        # Σ_u^-1 = K_mm^-1 + K_mm^-1 K_mn Λ^-1 K_nm K_mm^-1
        # μ_u   = Σ_u K_mm^-1 K_mn Λ^-1 y
        V_scaled = V / np.sqrt(Lambda)
        B = np.eye(V.shape[0]) + V_scaled @ V_scaled.T
        self._L_B, _ = stable_cholesky(B)
        # Effective alpha for predictive mean (in K_mm-space)
        rhs = V @ (y_centered / Lambda)
        self._alpha_u = np.linalg.solve(self._L_mm.T, cholesky_solve(self._L_B, rhs))
        
        self.X_train_ = X
        self.y_train_ = y_centered
        self.log_marginal_likelihood_ = -best_nll
        self._sigma2 = sigma2
        self._fitted = True
        
        return self
    
    def predict(self, X, return_std=False):
        """Predict mean and std at test inputs using the FITC posterior."""
        if not self._fitted:
            raise NotFittedError("Sparse GP must be fitted before prediction.")
        
        X = self._to_array(X)
        if X.shape[1] != self.X_train_.shape[1]:
            raise KernelDimensionError(
                f"Test data has {X.shape[1]} features but model was fit on "
                f"{self.X_train_.shape[1]}."
            )
        
        K_starm = self._signal_kernel(X, self.Z_)  # (n_test, m)
        
        # Predictive mean: K_*m @ alpha_u
        y_mean = K_starm @ self._alpha_u + self.y_mean_
        
        if not return_std:
            return y_mean
        
        # Predictive variance under FITC:
        # var(f_*) = K_** - K_*m K_mm^-1 K_m* + K_*m Σ_u K_m*
        # where Σ_u = K_mm^-1 + K_mm^-1 K_mn Λ^-1 K_nm K_mm^-1 (carefully)
        # The clean form using our cached Cholesky factors:
        Kss_diag = np.array([
            self._signal_kernel(X[i:i+1])[0, 0] for i in range(X.shape[0])
        ])
        # A = L_mm^-1 K_m*  -> shape (m, n_test)
        A = np.linalg.solve(self._L_mm, K_starm.T)
        # Term subtracted: Q_** diagonal
        prior_term = np.sum(A ** 2, axis=0)
        # Term added back from posterior over u:
        C = cholesky_solve(self._L_B, A)
        post_term = np.sum(A * C, axis=0)
        
        var = Kss_diag - prior_term + post_term + self._sigma2
        var = np.maximum(var, 1e-12)
        return y_mean, np.sqrt(var)
    
    def score(self, X, y):
        X, y = self._to_array(X, y)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1.0 - ss_res / ss_tot
    
    def __repr__(self):
        status = "fitted" if self._fitted else "unfitted"
        return (f"SparseGPRegressor(kernel={self.kernel!r}, "
                f"n_inducing={self.n_inducing}, status={status})")
