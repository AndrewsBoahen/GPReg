"""GP regression models."""

from .exact import GaussianProcessRegressor
from .sparse import SparseGPRegressor
from .multioutput import MultiOutputGP

__all__ = ["GaussianProcessRegressor", "SparseGPRegressor", "MultiOutputGP"]
