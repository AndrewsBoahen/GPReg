"""Persistence utilities for fitted GP models.

Provides two save formats:

- Pickle: saves the entire fitted model object including precomputed
  Cholesky factors. Fast to load, but version-fragile (requires the
  same gpreg version to load).

- JSON: saves only the configuration (kernel hyperparameters, training
  data, settings). On load, the model is re-fit. Slower but
  version-portable and human-readable.
"""

import json
import pickle
from pathlib import Path

import numpy as np


def save_model(model, path, format="pickle"):
    """Save a fitted GP model to disk.
    
    Parameters
    ----------
    model : GaussianProcessRegressor or SparseGPRegressor
        Fitted model to save.
    path : str or Path
        File path. Conventional extensions: .pkl for pickle, .json for json.
    format : {'pickle', 'json'}, default='pickle'
        Serialization format.
    """
    path = Path(path)
    
    if format == "pickle":
        with open(path, "wb") as f:
            pickle.dump(model, f)
    elif format == "json":
        if not getattr(model, "_fitted", False):
            raise ValueError("Cannot save unfitted model in JSON format.")
        
        data = {
            "model_type": type(model).__name__,
            "kernel_repr": repr(model.kernel),
            "kernel_params": model.kernel.get_params(),
            "X_train": model.X_train_.tolist(),
            "y_train": (model.y_train_ + model.y_mean_).tolist(),
            "y_mean": float(model.y_mean_),
            "log_marginal_likelihood": float(model.log_marginal_likelihood_),
            "normalize_y": model.normalize_y,
        }
        # Include sparse-specific data if applicable
        if hasattr(model, "Z_"):
            data["Z"] = model.Z_.tolist()
            data["n_inducing"] = model.n_inducing
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    else:
        raise ValueError(f"Unknown format {format!r}. Use 'pickle' or 'json'.")


def load_model(path, format=None):
    """Load a saved GP model.
    
    Parameters
    ----------
    path : str or Path
    format : {'pickle', 'json'} or None, default=None
        If None, infer from file extension.
    
    Returns
    -------
    model : fitted GP model
    """
    path = Path(path)
    
    if format is None:
        if path.suffix == ".json":
            format = "json"
        else:
            format = "pickle"
    
    if format == "pickle":
        with open(path, "rb") as f:
            return pickle.load(f)
    elif format == "json":
        with open(path, "r") as f:
            data = json.load(f)
        
        # Reconstruct the model. We import here to avoid circular imports.
        from ..models import GaussianProcessRegressor, SparseGPRegressor
        
        # Note: this only handles simple kernels for now. Composite
        # kernels would require parsing kernel_repr or storing more
        # structured metadata.
        raise NotImplementedError(
            "JSON loading requires kernel reconstruction logic that depends "
            "on which kernels you used. For now, use pickle format for "
            "round-trip save/load. JSON files are still useful for "
            "inspecting model parameters."
        )
    else:
        raise ValueError(f"Unknown format {format!r}.")
