"""Custom exceptions for the GPReg package.

Provides informative errors instead of cryptic NumPy/SciPy tracebacks.
"""


class GPRegError(Exception):
    """Base exception for all GPReg errors."""
    pass


class KernelDimensionError(GPRegError):
    """Raised when input dimensions don't match kernel expectations."""
    pass


class NonPSDMatrixError(GPRegError):
    """Raised when the kernel matrix is not positive semi-definite,
    even after adding jitter."""
    pass


class ConvergenceError(GPRegError):
    """Raised when hyperparameter optimization fails to converge."""
    pass


class NotFittedError(GPRegError):
    """Raised when predict() is called on an unfitted model."""
    pass


class InvalidHyperparameterError(GPRegError):
    """Raised when hyperparameters are outside valid ranges
    (e.g., negative length-scale)."""
    pass
