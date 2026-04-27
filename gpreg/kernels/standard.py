"""Concrete kernel implementations.

Includes the most commonly used kernels for GP regression:
- RBF (Squared Exponential): smooth, infinitely differentiable
- Matern: parameterized smoothness via nu (1/2, 3/2, 5/2)
- Linear: for modeling linear trends
- White: pure noise, used as an additive component
"""

import numpy as np
from scipy.spatial.distance import cdist

from .base import Kernel, _validate_positive
from ..utils.exceptions import InvalidHyperparameterError


class RBF(Kernel):
    """Radial Basis Function (Squared Exponential) kernel.
    
    k(x, x') = signal_var * exp(-||x - x'||^2 / (2 * length_scale^2))
    
    Produces very smooth functions. The default choice when you don't
    have a strong prior reason to pick something else.
    
    Parameters
    ----------
    length_scale : float, default=1.0
        Controls how quickly the function varies. Smaller values produce
        more wiggly functions; larger values produce smoother ones.
    signal_var : float, default=1.0
        Overall variance of the function (vertical scale).
    """
    
    def __init__(self, length_scale=1.0, signal_var=1.0):
        self.length_scale = _validate_positive(length_scale, "length_scale")
        self.signal_var = _validate_positive(signal_var, "signal_var")
    
    def _compute(self, X1, X2):
        # Use cdist for efficient pairwise squared distances
        sq_dist = cdist(X1 / self.length_scale, X2 / self.length_scale,
                        metric="sqeuclidean")
        return self.signal_var * np.exp(-0.5 * sq_dist)
    
    def get_params(self):
        return {"length_scale": self.length_scale, "signal_var": self.signal_var}
    
    def set_params(self, **params):
        if "length_scale" in params:
            self.length_scale = _validate_positive(params["length_scale"], "length_scale")
        if "signal_var" in params:
            self.signal_var = _validate_positive(params["signal_var"], "signal_var")
    
    @property
    def n_params(self):
        return 2


class Matern(Kernel):
    """Matern kernel with parameter nu controlling smoothness.
    
    Less smooth than RBF (which is the limit as nu -> infinity), often
    a better choice for real-world data. Common values:
    - nu=1/2: equivalent to exponential kernel (very rough, like Brownian motion)
    - nu=3/2: once-differentiable functions
    - nu=5/2: twice-differentiable (most popular default)
    
    Parameters
    ----------
    length_scale : float, default=1.0
        Same role as in RBF.
    signal_var : float, default=1.0
        Same role as in RBF.
    nu : float, default=2.5
        Smoothness parameter. Must be one of {0.5, 1.5, 2.5} for the
        closed-form expressions used here.
    """
    
    def __init__(self, length_scale=1.0, signal_var=1.0, nu=2.5):
        self.length_scale = _validate_positive(length_scale, "length_scale")
        self.signal_var = _validate_positive(signal_var, "signal_var")
        if nu not in (0.5, 1.5, 2.5):
            raise InvalidHyperparameterError(
                f"nu must be 0.5, 1.5, or 2.5, got {nu}. "
                f"(Other values require the modified Bessel function and are "
                f"not implemented.)"
            )
        self.nu = nu
    
    def _compute(self, X1, X2):
        dist = cdist(X1 / self.length_scale, X2 / self.length_scale,
                     metric="euclidean")
        
        if self.nu == 0.5:
            # Exponential kernel
            K = np.exp(-dist)
        elif self.nu == 1.5:
            sqrt3_d = np.sqrt(3) * dist
            K = (1.0 + sqrt3_d) * np.exp(-sqrt3_d)
        elif self.nu == 2.5:
            sqrt5_d = np.sqrt(5) * dist
            K = (1.0 + sqrt5_d + (5.0 / 3.0) * dist**2) * np.exp(-sqrt5_d)
        
        return self.signal_var * K
    
    def get_params(self):
        # nu is fixed (not optimized); only length_scale and signal_var go in
        return {"length_scale": self.length_scale, "signal_var": self.signal_var}
    
    def set_params(self, **params):
        if "length_scale" in params:
            self.length_scale = _validate_positive(params["length_scale"], "length_scale")
        if "signal_var" in params:
            self.signal_var = _validate_positive(params["signal_var"], "signal_var")
    
    @property
    def n_params(self):
        return 2  # nu is fixed, not optimized


class Linear(Kernel):
    """Linear kernel: k(x, x') = signal_var * (x . x').
    
    Equivalent to Bayesian linear regression. Useful as an additive
    component for modeling linear trends combined with nonlinear
    structure from another kernel.
    """
    
    def __init__(self, signal_var=1.0):
        self.signal_var = _validate_positive(signal_var, "signal_var")
    
    def _compute(self, X1, X2):
        return self.signal_var * (X1 @ X2.T)
    
    def get_params(self):
        return {"signal_var": self.signal_var}
    
    def set_params(self, **params):
        if "signal_var" in params:
            self.signal_var = _validate_positive(params["signal_var"], "signal_var")
    
    @property
    def n_params(self):
        return 1


class White(Kernel):
    """White noise kernel: k(x, x') = noise_var * I(x == x').
    
    Adds independent noise to each observation. Typically combined with
    a smooth kernel via addition: e.g., RBF() + White(0.1).
    """
    
    def __init__(self, noise_var=1.0):
        self.noise_var = _validate_positive(noise_var, "noise_var")
    
    def _compute(self, X1, X2):
        # Identity only when X1 and X2 are the same set of points
        if X1.shape == X2.shape and np.array_equal(X1, X2):
            return self.noise_var * np.eye(X1.shape[0])
        else:
            return np.zeros((X1.shape[0], X2.shape[0]))
    
    def get_params(self):
        return {"noise_var": self.noise_var}
    
    def set_params(self, **params):
        if "noise_var" in params:
            self.noise_var = _validate_positive(params["noise_var"], "noise_var")
    
    @property
    def n_params(self):
        return 1
