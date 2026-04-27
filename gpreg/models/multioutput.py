"""Multi-output Gaussian Process regression.

Wraps any single-output GP (exact or sparse) to handle multivariate
targets by training one independent GP per output dimension.

This is the simplest sensible approach. It assumes the output dimensions
are conditionally independent given the inputs — a strong assumption,
but a reasonable starting point. For correlated outputs, more advanced
methods (Linear Model of Coregionalization, multi-task kernels) would
share information across dimensions; those are not implemented here.

Each output gets its own copy of the kernel, so they fit different
hyperparameters and adapt independently.
"""

import copy

import numpy as np
import pandas as pd

from ..utils.exceptions import NotFittedError, KernelDimensionError


class MultiOutputGP:
    """Wrap a single-output GP for multivariate target prediction.
    
    Fits one independent GP per output column. All outputs share the
    same input space and the same kernel structure (with separately
    optimized hyperparameters per output).
    
    Parameters
    ----------
    base_gp : GaussianProcessRegressor or SparseGPRegressor
        Template GP. A deep copy is made for each output column so
        hyperparameters are fit independently.
    
    Attributes
    ----------
    estimators_ : list
        One fitted GP per output dimension.
    n_outputs_ : int
        Number of output dimensions seen during fit.
    output_names_ : list of str or None
        Column names if Y was a DataFrame, otherwise None.
    
    Examples
    --------
    >>> import numpy as np
    >>> from gpreg import GaussianProcessRegressor, RBF, White, MultiOutputGP
    >>> X = np.linspace(-3, 3, 30).reshape(-1, 1)
    >>> Y = np.column_stack([np.sin(X).ravel(), np.cos(X).ravel()])
    >>> base = GaussianProcessRegressor(kernel=RBF() + White(0.1), random_state=0)
    >>> mgp = MultiOutputGP(base)
    >>> mgp.fit(X, Y)
    >>> Y_mean, Y_std = mgp.predict(X, return_std=True)
    >>> Y_mean.shape, Y_std.shape
    ((30, 2), (30, 2))
    """
    
    def __init__(self, base_gp):
        self.base_gp = base_gp
        self._fitted = False
    
    def _to_array_y(self, Y):
        """Convert Y to a 2D array, recording output column names if any."""
        if isinstance(Y, pd.DataFrame):
            self.output_names_ = list(Y.columns)
            return Y.values.astype(float)
        elif isinstance(Y, pd.Series):
            self.output_names_ = [Y.name] if Y.name else None
            arr = Y.values.astype(float).reshape(-1, 1)
            return arr
        
        Y_arr = np.asarray(Y, dtype=float)
        if Y_arr.ndim == 1:
            Y_arr = Y_arr.reshape(-1, 1)
        self.output_names_ = None
        return Y_arr
    
    def fit(self, X, Y):
        """Fit one GP per output dimension.
        
        Parameters
        ----------
        X : array-like or DataFrame of shape (n, d)
            Input features.
        Y : array-like or DataFrame of shape (n, p)
            Output targets. If 1D, treated as a single-output problem
            (still wrapped, for API consistency).
        
        Returns
        -------
        self
        """
        Y = self._to_array_y(Y)
        self.n_outputs_ = Y.shape[1]
        
        # Train an independent GP per output dimension. We deep-copy
        # the base GP so each output gets its own hyperparameter values
        # without leaking state between fits.
        self.estimators_ = []
        for j in range(self.n_outputs_):
            gp_j = copy.deepcopy(self.base_gp)
            gp_j.fit(X, Y[:, j])
            self.estimators_.append(gp_j)
        
        self._fitted = True
        return self
    
    def predict(self, X, return_std=False):
        """Predict each output dimension independently.
        
        Returns
        -------
        Y_mean : ndarray of shape (m, p)
        Y_std : ndarray of shape (m, p), if return_std=True
        """
        if not self._fitted:
            raise NotFittedError(
                "MultiOutputGP must be fitted before predicting."
            )
        
        means = []
        stds = []
        for gp_j in self.estimators_:
            if return_std:
                m, s = gp_j.predict(X, return_std=True)
                means.append(m)
                stds.append(s)
            else:
                means.append(gp_j.predict(X))
        
        Y_mean = np.column_stack(means)
        if return_std:
            return Y_mean, np.column_stack(stds)
        return Y_mean
    
    def score(self, X, Y):
        """Average R^2 across output dimensions.
        
        Returns the mean R^2 over outputs. Per-output scores are
        available via the individual estimators in self.estimators_.
        """
        Y = self._to_array_y(Y)
        scores = [
            self.estimators_[j].score(X, Y[:, j])
            for j in range(self.n_outputs_)
        ]
        return float(np.mean(scores))
    
    def per_output_scores(self, X, Y):
        """Return individual R^2 scores for each output."""
        Y = self._to_array_y(Y)
        return np.array([
            self.estimators_[j].score(X, Y[:, j])
            for j in range(self.n_outputs_)
        ])
    
    @property
    def log_marginal_likelihood_(self):
        """Sum of log-marginal-likelihoods across outputs.
        
        Since outputs are independent under this model, the joint
        log marginal likelihood is just the sum.
        """
        if not self._fitted:
            raise NotFittedError("Model must be fitted first.")
        return sum(gp.log_marginal_likelihood_ for gp in self.estimators_)
    
    def __repr__(self):
        if self._fitted:
            return (f"MultiOutputGP(n_outputs={self.n_outputs_}, "
                    f"base={type(self.base_gp).__name__})")
        return f"MultiOutputGP(base={type(self.base_gp).__name__}, fitted=False)"
