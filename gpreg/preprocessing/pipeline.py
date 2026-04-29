"""Pipeline for chaining preprocessing steps with a GP model.

Mirrors the sklearn Pipeline API:
    pipe = Pipeline([
        ('encode', CategoricalEncoder()),
        ('scale', StandardScaler()),
        ('pca', PCA(n_components=0.95)),
        ('gp', GaussianProcessRegressor()),
    ])
    pipe.fit(X_train, y_train)
    pipe.predict(X_test)

The final step must be a GP model (something with fit/predict). All
prior steps must be transformers with fit/transform.
"""

import numpy as np

from ..utils.exceptions import NotFittedError


class Pipeline:
    """Chain transformers with a final GP estimator.
    
    Parameters
    ----------
    steps : list of (name, transformer/estimator) tuples
        All but the last step must be transformers; the last must be
        an estimator with fit/predict methods.
    
    Attributes
    ----------
    named_steps : dict
        Convenient access to individual steps by name, e.g.
        pipe.named_steps['scale'].
    """
    
    def __init__(self, steps):
        if len(steps) < 1:
            raise ValueError("Pipeline must have at least one step.")
        
        # Check all step names are unique
        names = [name for name, _ in steps]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate step names: {names}")
        
        self.steps = steps
        self.named_steps = {name: obj for name, obj in steps}
        self._fitted = False
    
    @property
    def _transformers(self):
        return self.steps[:-1]
    
    @property
    def _estimator_step(self):
        return self.steps[-1]
    
    @property
    def estimator(self):
        """Convenience accessor for the final estimator."""
        return self._estimator_step[1]
    
    def fit(self, X, y):
        """Fit each transformer in sequence, then fit the final estimator."""
        X_current = X
        for name, transformer in self._transformers:
            if not (hasattr(transformer, "fit") and hasattr(transformer, "transform")):
                raise TypeError(
                    f"Step {name!r} is not a transformer "
                    f"(must have fit and transform methods)."
                )
            X_current = transformer.fit_transform(X_current, y)
        
        # Final step: the estimator
        name, est = self._estimator_step
        if not (hasattr(est, "fit") and hasattr(est, "predict")):
            raise TypeError(
                f"Final step {name!r} is not an estimator "
                f"(must have fit and predict methods)."
            )
        est.fit(X_current, y)
        
        self._fitted = True
        return self
    
    def _transform(self, X):
        """Apply all transformers in sequence."""
        X_current = X
        for name, transformer in self._transformers:
            X_current = transformer.transform(X_current)
        return X_current
    
    def predict(self, X, return_std=False):
        """Pass X through all transformers and into the GP for prediction."""
        if not self._fitted:
            raise NotFittedError("Pipeline must be fitted before predicting.")
        
        X_transformed = self._transform(X)
        
        if return_std:
            return self.estimator.predict(X_transformed, return_std=True)
        return self.estimator.predict(X_transformed)
    
    def score(self, X, y):
        if not self._fitted:
            raise NotFittedError("Pipeline must be fitted before scoring.")
        X_transformed = self._transform(X)
        return self.estimator.score(X_transformed, y)
    
    def __repr__(self):
        steps_str = ",\n  ".join(f"({name!r}, {obj!r})" for name, obj in self.steps)
        return f"Pipeline([\n  {steps_str}\n])"


def make_gp_pipeline(model, scale=True, pca_components=None):
    """Convenience factory for the standard GP preprocessing pipeline.
    
    Parameters
    ----------
    model : GaussianProcessRegressor or SparseGPRegressor
        The final GP estimator.
    scale : bool, default=True
        Whether to standardize features.
    pca_components : int, float, or None, default=None
        If set, add a PCA step with this many components (or variance ratio).
    
    Returns
    -------
    pipeline : Pipeline
    """
    from .scaler import StandardScaler
    from .pca import PCA
    
    steps = []
    if scale:
        steps.append(("scale", StandardScaler()))
    if pca_components is not None:
        steps.append(("pca", PCA(n_components=pca_components)))
    steps.append(("gp", model))
    
    return Pipeline(steps)
