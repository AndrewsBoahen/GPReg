"""Utility functions and exceptions for GPReg."""

from .exceptions import (
    GPRegError,
    KernelDimensionError,
    NonPSDMatrixError,
    ConvergenceError,
    NotFittedError,
    InvalidHyperparameterError,
)
from .linalg import stable_cholesky, cholesky_solve, log_det_from_cholesky
from .io import save_model, load_model

__all__ = [
    "GPRegError",
    "KernelDimensionError",
    "NonPSDMatrixError",
    "ConvergenceError",
    "NotFittedError",
    "InvalidHyperparameterError",
    "stable_cholesky",
    "cholesky_solve",
    "log_det_from_cholesky",
    "save_model",
    "load_model",
]
