"""Exact Gaussian Process Regression.

Implements the standard exact GP using Cholesky decomposition for fitting
and prediction. O(n^3) in training size; suitable for n < ~5000.

For larger datasets, see SparseGP in this package.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..kernels import RBF, White
from ..kernels.base import Kernel
from ..utils.linalg import stable_cholesky, cholesky_solve, log_det_from_cholesky
from ..utils.exceptions import (
    NotFittedError,
    ConvergenceError,
    KernelDimensionError,
)


class GaussianProcessRegressor:
    """Exact Gaussian Process Regression.
    
    Parameters
    ----------
    kernel : Kernel, default=RBF() + White(0.1)
        The covariance kernel. White noise is typically added so the
        model has a noise variance to fit.
    normalize_y : bool, default=True
        Whether to subtract the mean of y before fitting. Almost always
        helpful since the GP prior assumes zero mean.
    n_restarts : int, default=5
        Number of random restarts for hyperparameter optimization.
        The marginal log-likelihood is non-convex, so multiple starts
        help avoid bad local optima.
    optimizer : str, default='L-BFGS-B'
        Optimizer for hyperparameter tuning. Options:
        - 'L-BFGS-B' (default): SciPy's L-BFGS-B, fast for small problems
        - any other scipy.optimize.minimize method name
        - 'pytorch': use PyTorch autograd + Adam (requires torch installed)
    random_state : int or None, default=None
        Seed for reproducible random restarts.
    
    Attributes
    ----------
    X_train_ : ndarray of shape (n, d)
        Training inputs (after any DataFrame conversion).
    y_train_ : ndarray of shape (n,)
        Training targets (after mean subtraction if normalize_y=True).
    y_mean_ : float
        Mean of training y, subtracted during fitting.
    L_ : ndarray of shape (n, n)
        Cholesky factor of the kernel matrix at the fitted hyperparameters.
    alpha_ : ndarray of shape (n,)
        Solution to K alpha = y, used in prediction.
    log_marginal_likelihood_ : float
        Log marginal likelihood at the fitted hyperparameters.
    """
    
    def __init__(self, kernel=None, normalize_y=True, n_restarts=5,
                 optimizer="L-BFGS-B", random_state=None):
        self.kernel = kernel if kernel is not None else RBF() + White(noise_var=0.1)
        self.normalize_y = normalize_y
        self.n_restarts = n_restarts
        self.optimizer = optimizer
        self.random_state = random_state
        self._fitted = False
    
    def _to_array(self, X, y=None):
        """Convert pandas DataFrames/Series to numpy arrays.
        
        Inputs must be continuous (numeric). Non-numeric DataFrame
        columns are rejected with an informative error rather than
        silently coerced.
        """
        if isinstance(X, pd.DataFrame):
            from pandas.api.types import is_numeric_dtype
            non_numeric = [c for c in X.columns if not is_numeric_dtype(X[c])]
            if non_numeric:
                raise ValueError(
                    f"GPReg only accepts continuous (numeric) inputs. "
                    f"Non-numeric columns: {non_numeric}. "
                    f"Convert them to numeric or drop them before fitting."
                )
            self.feature_names_ = list(X.columns)
            X = X.values.astype(float)
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
    
    def _neg_log_marginal_likelihood(self, theta, X, y):
        """Negative log marginal likelihood (the optimization objective).
        
        Following Rasmussen & Williams Eq. 2.30:
            log p(y|X) = -0.5 y^T K^-1 y - 0.5 log|K| - n/2 log(2*pi)
        
        We negate it to turn maximization into minimization.
        """
        try:
            self.kernel.set_param_vector(theta)
        except Exception:
            return np.inf
        
        try:
            K = self.kernel(X)
            L, _ = stable_cholesky(K)
        except Exception:
            return np.inf
        
        alpha = cholesky_solve(L, y)
        
        # Rasmussen & Williams Eq. 2.30:
        # log p(y|X) = -0.5 y^T K^-1 y - 0.5 log|K| - n/2 log(2*pi)
        log_lik = (
            -0.5 * y @ alpha
            - 0.5 * log_det_from_cholesky(L)
            - 0.5 * len(y) * np.log(2 * np.pi)
        )
        
        return -log_lik
    
    def fit(self, X, y):
        """Fit the GP by optimizing kernel hyperparameters via marginal likelihood.
        
        Parameters
        ----------
        X : array-like or DataFrame of shape (n, d)
            Training inputs.
        y : array-like or Series of shape (n,)
            Training targets.
        
        Returns
        -------
        self : GaussianProcessRegressor
            Fitted estimator.
        """
        X, y = self._to_array(X, y)
        
        if self.normalize_y:
            self.y_mean_ = float(y.mean())
            y_centered = y - self.y_mean_
        else:
            self.y_mean_ = 0.0
            y_centered = y
        
        rng = np.random.default_rng(self.random_state)
        
        if self.optimizer == "pytorch":
            # Delegate to the autograd-based optimizer
            from .torch_backend import torch_optimize
            best_theta, best_nll = torch_optimize(
                self.kernel, X, y_centered,
                n_restarts=self.n_restarts,
                random_state=self.random_state,
            )
            self.kernel.set_param_vector(best_theta)
        else:
            # SciPy path (default)
            best_result = None
            best_nll = np.inf
            
            # First restart from current hyperparameters
            starts = [self.kernel.get_param_vector()]
            # Additional random restarts in log-space
            for _ in range(self.n_restarts):
                starts.append(rng.normal(0.0, 1.5, size=self.kernel.n_params))
            
            for theta0 in starts:
                try:
                    result = minimize(
                        self._neg_log_marginal_likelihood,
                        theta0,
                        args=(X, y_centered),
                        method=self.optimizer,
                    )
                    if result.fun < best_nll:
                        best_nll = result.fun
                        best_result = result
                except Exception:
                    continue
            
            if best_result is None:
                raise ConvergenceError(
                    "All hyperparameter optimization restarts failed. "
                    "Try a different kernel or check your data for issues."
                )
            
            # Set hyperparameters to the best found
            self.kernel.set_param_vector(best_result.x)
        
        # Cache values needed for prediction
        K = self.kernel(X)
        self.L_, _ = stable_cholesky(K)
        self.alpha_ = cholesky_solve(self.L_, y_centered)
        
        self.X_train_ = X
        self.y_train_ = y_centered
        self.log_marginal_likelihood_ = -best_nll
        self._fitted = True
        
        return self
    
    def predict(self, X, return_std=False, return_cov=False):
        """Predict mean (and optionally std or full covariance) at new inputs.
        
        Parameters
        ----------
        X : array-like or DataFrame of shape (m, d)
            Test inputs.
        return_std : bool, default=False
            If True, also return predictive standard deviations.
        return_cov : bool, default=False
            If True, also return the full predictive covariance matrix.
            Cannot be combined with return_std.
        
        Returns
        -------
        y_mean : ndarray of shape (m,)
            Predictive means.
        y_std : ndarray of shape (m,), optional
            Predictive standard deviations (if return_std=True).
        y_cov : ndarray of shape (m, m), optional
            Predictive covariance (if return_cov=True).
        """
        if not self._fitted:
            raise NotFittedError(
                "Model must be fitted before predicting. Call .fit(X, y) first."
            )
        if return_std and return_cov:
            raise ValueError("Cannot set both return_std and return_cov.")
        
        X = self._to_array(X)
        
        if X.shape[1] != self.X_train_.shape[1]:
            raise KernelDimensionError(
                f"Test data has {X.shape[1]} features but model was fit on "
                f"{self.X_train_.shape[1]} features."
            )
        
        # Cross-covariance and predictive mean (R&W Eq. 2.25)
        K_star = self.kernel(self.X_train_, X)  # (n, m)
        y_mean = K_star.T @ self.alpha_ + self.y_mean_
        
        if not (return_std or return_cov):
            return y_mean
        
        # Predictive covariance (R&W Eq. 2.26)
        # v = L^-1 K_star, then K** - v^T v
        v = cholesky_solve(self.L_, K_star)
        K_starstar = self.kernel(X)
        y_cov = K_starstar - K_star.T @ v
        
        if return_std:
            # Numerical floor to avoid tiny negative variances
            y_var = np.diag(y_cov).copy()
            y_var = np.maximum(y_var, 1e-12)
            return y_mean, np.sqrt(y_var)
        
        return y_mean, y_cov
    
    def sample_y(self, X, n_samples=1, random_state=None):
        """Draw samples from the predictive distribution at X.
        
        Useful for visualizing the range of plausible functions.
        """
        rng = np.random.default_rng(random_state)
        y_mean, y_cov = self.predict(X, return_cov=True)
        # Add jitter to make sampling stable
        y_cov_jittered = y_cov + 1e-8 * np.eye(y_cov.shape[0])
        return rng.multivariate_normal(y_mean, y_cov_jittered, size=n_samples).T
    
    def score(self, X, y):
        """R^2 score on test data (for sklearn-style API compatibility)."""
        X, y = self._to_array(X, y)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1.0 - ss_res / ss_tot
    
    def __repr__(self):
        status = "fitted" if self._fitted else "unfitted"
        return f"GaussianProcessRegressor(kernel={self.kernel!r}, status={status})"
