"""Principal Component Analysis for GP feature reduction.

GPs suffer from the curse of dimensionality as it becomes hard to estimate hyperparameters.
PCA can help by projecting to a lower-dimensional subspace that retains most of the variance.

Implementation note: we compute PCA via SVD rather than the covariance
eigendecomposition, because I learnt that SVD is more numerically stable when some
features are nearly collinear (a common situation after dummy coding).
"""

import numpy as np

from .scaler import Transformer
from ..utils.exceptions import NotFittedError, InvalidHyperparameterError


class PCA(Transformer):
    """Principal Component Analysis for dimension reduction.
    
    Parameters
    ----------
    n_components : int or float, default=None
        - If int: keep exactly this many components.
        - If float in (0, 1]: keep enough components to explain at least
          this fraction of variance.
        - If None: keep all components (no reduction; useful for diagnostics).
    
    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Principal axes (rows are PCs in feature space).
    explained_variance_ : ndarray of shape (n_components,)
        Variance explained by each component.
    explained_variance_ratio_ : ndarray of shape (n_components,)
        Fraction of total variance explained by each component.
    mean_ : ndarray of shape (n_features,)
        Mean of training data, subtracted before projection.
    n_components_ : int
        Resolved number of components after fitting.
    """
    
    def __init__(self, n_components=None):
        super().__init__()
        if n_components is not None:
            if isinstance(n_components, float):
                if not 0 < n_components <= 1:
                    raise InvalidHyperparameterError(
                        f"n_components as float must be in (0, 1], got {n_components}."
                    )
            elif isinstance(n_components, int):
                if n_components < 1:
                    raise InvalidHyperparameterError(
                        f"n_components must be >= 1, got {n_components}."
                    )
            else:
                raise InvalidHyperparameterError(
                    f"n_components must be int or float, got {type(n_components)}."
                )
        self.n_components = n_components
    
    def _fit(self, X, y):
        n_samples, n_features = X.shape
        
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        
        # SVD: X_centered = U @ diag(s) @ Vt
        # The principal components are the rows of Vt; the variance
        # explained by component i is s_i^2 / (n_samples - 1).
        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Total variance and variance ratios
        total_var = (s ** 2).sum() / (n_samples - 1)
        explained_variance = (s ** 2) / (n_samples - 1)
        explained_ratio = explained_variance / total_var
        
        # Resolve n_components
        if self.n_components is None:
            n_components_ = len(s)
        elif isinstance(self.n_components, float):
            # Keep enough components to reach the target cumulative ratio
            cum_ratio = np.cumsum(explained_ratio)
            n_components_ = int(np.searchsorted(cum_ratio, self.n_components) + 1)
            n_components_ = min(n_components_, len(s))
        else:
            n_components_ = min(self.n_components, len(s))
        
        self.components_ = Vt[:n_components_]
        self.explained_variance_ = explained_variance[:n_components_]
        self.explained_variance_ratio_ = explained_ratio[:n_components_]
        self.n_components_ = n_components_
    
    def _transform(self, X):
        X_centered = X - self.mean_
        return X_centered @ self.components_.T
    
    def inverse_transform(self, X_proj):
        """Project back to the original feature space (lossy if reduced)."""
        if not self._fitted:
            raise NotFittedError("PCA must be fitted first.")
        return X_proj @ self.components_ + self.mean_
    
    def __repr__(self):
        if self._fitted:
            total_explained = self.explained_variance_ratio_.sum()
            return (f"PCA(n_components={self.n_components_}, "
                    f"variance_explained={total_explained:.3f})")
        return f"PCA(n_components={self.n_components}, fitted=False)"
