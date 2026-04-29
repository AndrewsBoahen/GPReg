"""Preprocessing transformers and pipelines for GP regression.

Includes:
- StandardScaler: feature scaling (essential for GPs)
- PCA: dimension reduction for high-dim inputs
- Pipeline: chain preprocessing steps with a GP model
- make_gp_pipeline: convenience factory for the standard pipeline
"""

from .scaler import Transformer, StandardScaler
from .pca import PCA
from .pipeline import Pipeline, make_gp_pipeline

__all__ = [
    "Transformer",
    "StandardScaler",
    "PCA",
    "Pipeline",
    "make_gp_pipeline",
]
