"""Linear algebra utilities for stable GP computations.

Uses Cholesky decomposition rather than direct inversion for numerical
stability. Adds adaptive jitter when matrices are near-singular.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from .exceptions import NonPSDMatrixError


def stable_cholesky(K, jitter=1e-8, max_tries=6):
    """Compute Cholesky decomposition with adaptive jitter.
    
    Adds a small value to the diagonal to ensure positive definiteness.
    If the initial jitter is insufficient, increases it by 10x up to
    max_tries times.
    
    Parameters
    ----------
    K : ndarray of shape (n, n)
        Symmetric matrix to decompose.
    jitter : float, default=1e-8
        Initial value added to the diagonal.
    max_tries : int, default=6
        Maximum number of jitter increases before giving up.
    
    Returns
    -------
    L : ndarray of shape (n, n)
        Lower triangular Cholesky factor such that L @ L.T = K + jitter*I.
    jitter_used : float
        Final jitter value that succeeded.
    
    Raises
    ------
    NonPSDMatrixError
        If decomposition fails even with maximum jitter.
    """
    n = K.shape[0]
    K_sym = (K + K.T) / 2  # Force symmetry to handle numerical asymmetry
    
    current_jitter = jitter
    for attempt in range(max_tries):
        try:
            L = np.linalg.cholesky(K_sym + current_jitter * np.eye(n))
            return L, current_jitter
        except np.linalg.LinAlgError:
            current_jitter *= 10
    
    raise NonPSDMatrixError(
        f"Kernel matrix is not positive definite even with jitter={current_jitter:.2e}. "
        f"This often indicates duplicate or near-duplicate input points, or "
        f"poorly chosen kernel hyperparameters."
    )


def cholesky_solve(L, b):
    """Solve K @ x = b given the Cholesky factor L of K.
    
    Equivalent to solving L @ L.T @ x = b via two triangular solves,
    which is more stable than computing K^-1 @ b directly.
    
    Parameters
    ----------
    L : ndarray of shape (n, n)
        Lower triangular Cholesky factor.
    b : ndarray of shape (n,) or (n, k)
        Right-hand side vector(s).
    
    Returns
    -------
    x : ndarray
        Solution to K @ x = b.
    """
    return cho_solve((L, True), b)


def log_det_from_cholesky(L):
    """Compute log|K| from its Cholesky factor.
    
    Since K = L @ L.T, |K| = |L|^2 = (prod of diagonal entries)^2,
    so log|K| = 2 * sum(log(diag(L))).
    
    This avoids overflow that would occur from computing the determinant
    directly for large matrices.
    """
    return 2.0 * np.sum(np.log(np.diag(L)))
