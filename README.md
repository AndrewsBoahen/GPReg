# GPReg

A from-scratch Gaussian Process Regression package in Python.

## Features

- **Exact and sparse GP regression** — exact GP for small data, FITC sparse GP for n > 5000
- **Multivariate inputs and outputs** — DataFrames with mixed types in; multiple targets out via `MultiOutputGP`
- **Composable kernels** — RBF, Matérn (ν = 1/2, 3/2, 5/2), Linear, White, with `+` and `*` operators
- **Diagnostics suite** — RMSE, NLPD, leave-one-out CV (closed-form), calibration plots, residual plots
- **Preprocessing pipeline** — automatic dummy coding, standard scaling, PCA, all chainable
- **DataFrame-native** — pass pandas DataFrames in directly; categoricals auto-detected
- **Two optimizer backends** — SciPy L-BFGS-B (default, fast) or PyTorch Adam (autograd, flexible)
- **Persistence** — save and load fitted models via pickle
- **Streamlit app** — interactive frontend for non-coders

## Installation

```bash
pip install numpy scipy pandas matplotlib seaborn
# Optional:
pip install scikit-learn  # enables k-means inducing-point initialization
pip install torch          # enables PyTorch optimizer backend
pip install streamlit      # enables the interactive app
```

Then put the `gpreg` folder on your Python path.

## Quick start

```python
import numpy as np
from gpreg import GaussianProcessRegressor, RBF, White

X = np.linspace(-3, 3, 30).reshape(-1, 1)
y = np.sin(X).ravel() + 0.1 * np.random.randn(30)

gp = GaussianProcessRegressor(kernel=RBF() + White(noise_var=0.1))
gp.fit(X, y)

X_test = np.linspace(-4, 4, 100).reshape(-1, 1)
y_mean, y_std = gp.predict(X_test, return_std=True)
```

## Mixed DataFrames

```python
import pandas as pd
from gpreg import make_gp_pipeline, GaussianProcessRegressor, RBF, White

df = pd.DataFrame({
    'temperature': [...],
    'humidity':    [...],
    'season':      ['spring', 'summer', ...],   # categorical, auto-detected
})
pipe = make_gp_pipeline(GaussianProcessRegressor(kernel=RBF() + White(0.1)))
pipe.fit(df, target_array)
pipe.predict(df_new, return_std=True)
```

## Multivariate outputs

```python
import pandas as pd
from gpreg import GaussianProcessRegressor, MultiOutputGP, RBF, White

# Y can be a DataFrame (preserves column names) or a (n, p) array
Y = pd.DataFrame({'temp': [...], 'humidity': [...], 'pressure': [...]})

base = GaussianProcessRegressor(kernel=RBF() + White(0.1))
mgp = MultiOutputGP(base)
mgp.fit(X, Y)

Y_mean, Y_std = mgp.predict(X_new, return_std=True)   # shape (m, 3) each
mgp.per_output_scores(X, Y)                            # array of R^2 per output
```

## Sparse GP for large data

```python
from gpreg import SparseGPRegressor

sparse = SparseGPRegressor(
    kernel=RBF() + White(0.1),
    n_inducing=100,
    inducing_strategy='kmeans',
)
sparse.fit(X, y)   # works for n in the tens of thousands
```

## Running the app

```bash
streamlit run app.py
```

## Package layout

```
gpreg/
├── kernels/         RBF, Matern, Linear, White; SumKernel/ProductKernel
├── models/          GaussianProcessRegressor, SparseGPRegressor, torch_backend
├── preprocessing/   StandardScaler, CategoricalEncoder, PCA, Pipeline
├── diagnostics/     metrics (RMSE, NLPD, LOO-CV, calibration) and plots
└── utils/           Cholesky helpers, custom exceptions, save/load
```

## References

- Rasmussen & Williams, *Gaussian Processes for Machine Learning* (2006)
- Snelson & Ghahramani, *Sparse Gaussian Processes using Pseudo-inputs* (NIPS 2006) — FITC approximation
