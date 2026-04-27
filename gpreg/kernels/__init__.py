"""Kernel functions for Gaussian Process regression.

Kernels can be composed via addition (+) and multiplication (*):

    >>> from gpreg.kernels import RBF, White, Linear
    >>> k = RBF(length_scale=1.0) + White(noise_var=0.1)
    >>> k_complex = RBF() * Linear() + Matern(nu=1.5)
"""

from .base import Kernel, SumKernel, ProductKernel
from .standard import RBF, Matern, Linear, White

__all__ = [
    "Kernel",
    "SumKernel",
    "ProductKernel",
    "RBF",
    "Matern",
    "Linear",
    "White",
]
