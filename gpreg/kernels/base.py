"""Base kernel class.

Defines the interface all kernels must implement and supports composition
via + and * operators. Subclasses override _compute() and provide their
own hyperparameters.
"""

from abc import ABC, abstractmethod
import numpy as np

from ..utils.exceptions import InvalidHyperparameterError, KernelDimensionError


class Kernel(ABC):
    """Abstract base class for all kernels.
    
    Subclasses must implement:
    - _compute(X1, X2): the actual covariance computation
    - get_params(): return current hyperparameters as a dict
    - set_params(**params): update hyperparameters
    - n_params property: number of hyperparameters (for optimizer)
    
    Supports composition via:
    - k1 + k2 -> SumKernel
    - k1 * k2 -> ProductKernel
    """
    
    def __call__(self, X1, X2=None):
        """Compute the kernel matrix K(X1, X2).
        
        If X2 is None, computes K(X1, X1) (the symmetric case).
        """
        X1 = np.atleast_2d(X1)
        if X2 is None:
            X2 = X1
        else:
            X2 = np.atleast_2d(X2)
        
        if X1.shape[1] != X2.shape[1]:
            raise KernelDimensionError(
                f"Input dimensions don't match: X1 has {X1.shape[1]} features, "
                f"X2 has {X2.shape[1]} features."
            )
        
        return self._compute(X1, X2)
    
    @abstractmethod
    def _compute(self, X1, X2):
        """Compute the kernel matrix. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def get_params(self):
        """Return hyperparameters as a dict."""
        pass
    
    @abstractmethod
    def set_params(self, **params):
        """Update hyperparameters from kwargs."""
        pass
    
    @property
    @abstractmethod
    def n_params(self):
        """Number of hyperparameters (for the optimizer)."""
        pass
    
    def get_param_vector(self):
        """Return hyperparameters as a flat array (for optimization).
        
        Uses log-transform for positive parameters so the optimizer can
        work in unconstrained space.
        """
        params = self.get_params()
        return np.log(np.array(list(params.values())))
    
    def set_param_vector(self, theta):
        """Set hyperparameters from a flat log-space array."""
        keys = list(self.get_params().keys())
        values = np.exp(theta)
        self.set_params(**dict(zip(keys, values)))
    
    def __add__(self, other):
        return SumKernel(self, other)
    
    def __mul__(self, other):
        return ProductKernel(self, other)
    
    def __repr__(self):
        params = self.get_params()
        param_str = ", ".join(f"{k}={v:.4g}" for k, v in params.items())
        return f"{self.__class__.__name__}({param_str})"


class SumKernel(Kernel):
    """Sum of two kernels: k(x, x') = k1(x, x') + k2(x, x')."""
    
    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2
    
    def _compute(self, X1, X2):
        return self.k1(X1, X2) + self.k2(X1, X2)
    
    def get_params(self):
        p1 = {f"k1__{k}": v for k, v in self.k1.get_params().items()}
        p2 = {f"k2__{k}": v for k, v in self.k2.get_params().items()}
        return {**p1, **p2}
    
    def set_params(self, **params):
        k1_params = {k[4:]: v for k, v in params.items() if k.startswith("k1__")}
        k2_params = {k[4:]: v for k, v in params.items() if k.startswith("k2__")}
        if k1_params:
            self.k1.set_params(**k1_params)
        if k2_params:
            self.k2.set_params(**k2_params)
    
    @property
    def n_params(self):
        return self.k1.n_params + self.k2.n_params
    
    def __repr__(self):
        return f"({self.k1!r} + {self.k2!r})"


class ProductKernel(Kernel):
    """Product of two kernels: k(x, x') = k1(x, x') * k2(x, x')."""
    
    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2
    
    def _compute(self, X1, X2):
        return self.k1(X1, X2) * self.k2(X1, X2)
    
    def get_params(self):
        p1 = {f"k1__{k}": v for k, v in self.k1.get_params().items()}
        p2 = {f"k2__{k}": v for k, v in self.k2.get_params().items()}
        return {**p1, **p2}
    
    def set_params(self, **params):
        k1_params = {k[4:]: v for k, v in params.items() if k.startswith("k1__")}
        k2_params = {k[4:]: v for k, v in params.items() if k.startswith("k2__")}
        if k1_params:
            self.k1.set_params(**k1_params)
        if k2_params:
            self.k2.set_params(**k2_params)
    
    @property
    def n_params(self):
        return self.k1.n_params + self.k2.n_params
    
    def __repr__(self):
        return f"({self.k1!r} * {self.k2!r})"


def _validate_positive(value, name):
    """Helper to validate that a hyperparameter is positive."""
    if value <= 0:
        raise InvalidHyperparameterError(
            f"{name} must be positive, got {value}."
        )
    return float(value)
