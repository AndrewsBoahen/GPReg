"""Categorical variable encoding for GP pipelines.

GPs treat all inputs as continuous. To handle categorical features
(e.g., 'red', 'blue', 'green'), we dummy-code them into 0/1 columns.

Important caveat: GPs aren't ideal for categorical data. With dummy
codes, the kernel only "sees" categories as 0/1 points in space, so
the length-scale governs how strongly different categories influence
each other. For features with many unique categories or important
non-linear interactions among them, consider categorical-aware kernels
(not implemented here).
"""

import numpy as np
import pandas as pd

from .scaler import Transformer
from ..utils.exceptions import NotFittedError


class CategoricalEncoder(Transformer):
    """Dummy-code categorical columns; pass through numeric columns.
    
    Uses pandas under the hood, which is faster than rolling our own
    one-hot encoder and handles unseen categories gracefully (they
    encode as all-zeros).
    
    Parameters
    ----------
    categorical_columns : list of str or list of int, optional
        Column names (for DataFrames) or indices (for arrays) to treat
        as categorical. If None, automatically detects:
        - For DataFrames: any column with dtype 'object', 'category', or 'bool'
        - For arrays: no columns (raises a hint to specify them)
    drop_first : bool, default=True
        Drop the first category to avoid the dummy variable trap. Set
        False if you want all categories represented (rarely needed for GPs).
    
    Attributes
    ----------
    categorical_columns_ : list
        Resolved list of categorical column names/indices.
    categories_ : dict
        Mapping from column -> list of seen categories during fit.
    feature_names_out_ : list of str
        Names of the columns produced by transform().
    """
    
    def __init__(self, categorical_columns=None, drop_first=True):
        super().__init__()
        self.categorical_columns = categorical_columns
        self.drop_first = drop_first
    
    def fit(self, X, y=None):
        # Override fit because we need access to the DataFrame BEFORE
        # converting to array (to detect dtypes).
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
            self.feature_names_in_ = list(X_df.columns)
            self._is_dataframe = True
        else:
            X_arr = np.atleast_2d(np.asarray(X))
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            self.feature_names_in_ = [f"x{i}" for i in range(X_arr.shape[1])]
            X_df = pd.DataFrame(X_arr, columns=self.feature_names_in_)
            self._is_dataframe = False
        
        # Resolve which columns are categorical
        if self.categorical_columns is None:
            # Detect non-numeric columns. Using is_numeric_dtype is more
            # robust than checking dtype names, since pandas reports
            # different names across versions ('object', 'str', 'category').
            from pandas.api.types import is_numeric_dtype
            self.categorical_columns_ = [
                c for c in X_df.columns if not is_numeric_dtype(X_df[c])
            ]
        else:
            # Allow ints (indices) for array input
            if all(isinstance(c, int) for c in self.categorical_columns):
                self.categorical_columns_ = [
                    self.feature_names_in_[i] for i in self.categorical_columns
                ]
            else:
                self.categorical_columns_ = list(self.categorical_columns)
        
        # Remember the categories seen for each column so we can apply
        # the same encoding to new data even if it has different categories
        self.categories_ = {
            col: sorted(X_df[col].dropna().unique().tolist())
            for col in self.categorical_columns_
        }
        
        # Compute output column names (do a dry-run encode on the training data)
        encoded = self._encode(X_df)
        self.feature_names_out_ = list(encoded.columns)
        
        self._fitted = True
        return self
    
    def _encode(self, X_df):
        """Apply the encoding (used internally for both fit and transform)."""
        # Reset the index to avoid alignment issues during concat —
        # if the input DataFrame has duplicate or non-sequential indices,
        # pd.concat can produce wrong-shaped output.
        result = X_df.reset_index(drop=True).copy()
        
        for col in self.categorical_columns_:
            if col not in result.columns:
                continue
            # Convert to categorical with the categories seen during fit,
            # so unseen categories become NaN -> all-zero dummies.
            cat_series = pd.Categorical(
                result[col],
                categories=self.categories_[col],
            )
            dummies = pd.get_dummies(
                cat_series,
                prefix=col,
                drop_first=self.drop_first,
                dummy_na=False,
            ).astype(float)
            # Replace original column with dummies
            result = result.drop(columns=[col])
            result = pd.concat([result.reset_index(drop=True),
                                dummies.reset_index(drop=True)], axis=1)
        
        return result
    
    def transform(self, X):
        if not self._fitted:
            raise NotFittedError("CategoricalEncoder must be fitted first.")
        
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_arr = np.atleast_2d(np.asarray(X))
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            X_df = pd.DataFrame(X_arr, columns=self.feature_names_in_)
        
        encoded = self._encode(X_df)
        # Ensure column order matches what we saw during fit, filling
        # any missing columns with zeros
        for col in self.feature_names_out_:
            if col not in encoded.columns:
                encoded[col] = 0.0
        encoded = encoded[self.feature_names_out_]
        
        return encoded.values.astype(float)
    
    # The base class abstractmethods are not used because we override fit/transform
    def _fit(self, X, y):
        pass
    
    def _transform(self, X):
        pass
    
    def __repr__(self):
        return (f"CategoricalEncoder(categorical_columns={self.categorical_columns_ if self._fitted else self.categorical_columns}, "
                f"drop_first={self.drop_first}, fitted={self._fitted})")
