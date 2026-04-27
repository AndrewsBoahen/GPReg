"""GPReg: Gaussian Process Regression in Python.

A from-scratch GP regression package with:
- Exact and sparse (FITC) GP regression
- Composable kernels (RBF, Matern, Linear, White) with + and * operators
- Diagnostic suite (RMSE, NLPD, LOO-CV, calibration plots)
- Preprocessing pipeline (scaling, dummy coding, PCA)

Quick start
-----------
>>> import numpy as np
>>> from gpreg import GaussianProcessRegressor, RBF, White
>>>
>>> X = np.linspace(-3, 3, 30).reshape(-1, 1)
>>> y = np.sin(X).ravel() + 0.1 * np.random.randn(30)
>>>
>>> gp = GaussianProcessRegressor(kernel=RBF() + White(0.1))
>>> gp.fit(X, y)
>>> y_mean, y_std = gp.predict(np.linspace(-4, 4, 100).reshape(-1, 1), return_std=True)
"""

__version__ = "0.1.0"

from .models import GaussianProcessRegressor, SparseGPRegressor, MultiOutputGP
from .kernels import Kernel, RBF, Matern, Linear, White
from .preprocessing import (
    StandardScaler,
    CategoricalEncoder,
    PCA,
    Pipeline,
    make_gp_pipeline,
)
from .diagnostics import rmse, nlpd, loo_cv, calibration_curve
from .utils import save_model, load_model

__all__ = [
    "GaussianProcessRegressor",
    "SparseGPRegressor",
    "MultiOutputGP",
    "Kernel",
    "RBF",
    "Matern",
    "Linear",
    "White",
    "StandardScaler",
    "CategoricalEncoder",
    "PCA",
    "Pipeline",
    "make_gp_pipeline",
    "rmse",
    "nlpd",
    "loo_cv",
    "calibration_curve",
    "save_model",
    "load_model",
]
