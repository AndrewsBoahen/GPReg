"""Diagnostic tools for assessing GP model quality."""

from .metrics import rmse, nlpd, loo_cv
from .plots import (
    plot_predictions_1d,
    plot_predictions_2d,
    plot_residuals,
    plot_kernel_heatmap,
    plot_pair,
)

__all__ = [
    "rmse",
    "nlpd",
    "loo_cv",
    "plot_predictions_1d",
    "plot_predictions_2d",
    "plot_residuals",
    "plot_kernel_heatmap",
    "plot_pair",
]
