"""Preprocessing transformers and pipelines for GP regression.

Includes:
- StandardScaler: feature scaling (essential for GPs)
- CategoricalEncoder: automatic dummy coding for object/category columns
- PCA: dimension reduction for high-dim inputs
- Pipeline: chain preprocessing steps with a GP model
- make_gp_pipeline: convenience factory for the standard pipeline
"""

from .scaler import Transformer, StandardScaler
from .categorical import CategoricalEncoder
from .pca import PCA
from .pipeline import Pipeline, make_gp_pipeline

__all__ = [
    "Transformer",
    "StandardScaler",
    "CategoricalEncoder",
    "PCA",
    "Pipeline",
    "make_gp_pipeline",
]
