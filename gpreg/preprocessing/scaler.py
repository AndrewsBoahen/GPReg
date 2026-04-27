"""Preprocessing transformers for GP pipelines.

Each transformer follows the sklearn-style API:
- fit(X): learn parameters from training data
- transform(X): apply the learned transformation
- fit_transform(X): convenience for both at once

Transformers are designed to be composed via the Pipeline class.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

from ..utils.exceptions import NotFittedError


class Transformer(ABC):
    """Abstract base for all transformers.
    
    Subclasses must implement _fit() and _transform(). The public
    fit/transform/fit_transform methods handle DataFrame conversion
    and the fitted-state check uniformly.
    """
    
    def __init__(self):
        self._fitted = False
    
    def fit(self, X, y=None):
        """Learn transformation parameters from X."""
        X = self._to_array(X)
        self._fit(X, y)
        self._fitted = True
        return self
    
    def transform(self, X):
        """Apply the fitted transformation to new data."""
        if not self._fitted:
            raise NotFittedError(
                f"{self.__class__.__name__} must be fitted before transforming."
            )
        X = self._to_array(X)
        return self._transform(X)
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def _to_array(self, X):
        """Convert input to a 2D numpy array, preserving column names."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
            return X.values.astype(float)
        else:
            arr = np.atleast_2d(np.asarray(X, dtype=float))
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            self.feature_names_in_ = [f"x{i}" for i in range(arr.shape[1])]
            return arr
    
    @abstractmethod
    def _fit(self, X, y):
        pass
    
    @abstractmethod
    def _transform(self, X):
        pass


class StandardScaler(Transformer):
    """Standardize features by removing the mean and scaling to unit variance.
    
    GPs are extremely sensitive to input scales — if one feature has range
    [0, 1] and another has range [0, 10000], the kernel's length-scale
    becomes meaningless. Always scale your inputs before fitting a GP.
    
    Attributes
    ----------
    mean_ : ndarray of shape (n_features,)
    scale_ : ndarray of shape (n_features,)
    """
    
    def _fit(self, X, y):
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        # Guard against zero-variance features (constant columns)
        self.scale_[self.scale_ < 1e-10] = 1.0
    
    def _transform(self, X):
        return (X - self.mean_) / self.scale_
    
    def inverse_transform(self, X):
        if not self._fitted:
            raise NotFittedError("StandardScaler must be fitted first.")
        return X * self.scale_ + self.mean_
    
    def __repr__(self):
        return f"StandardScaler(fitted={self._fitted})"
