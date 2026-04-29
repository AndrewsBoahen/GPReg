# GPReg

A from-scratch Gaussian Process Regression package in Python.

📖 **[Full documentation](DOCUMENTATION.md)** | 📓 **[Demo notebook](examples/demo.ipynb)**

## Features

- **Exact and sparse GP regression** — exact GP for small data, FITC sparse GP for n > 5000
- **Multivariate inputs and outputs** — continuous (numeric) only, with support for multiple targets via `MultiOutputGP`
- **Composable kernels** — RBF, Matérn (ν = 1/2, 3/2, 5/2), Linear, White, with `+` and `*` operators
- **Diagnostic suite** — RMSE, NLPD, leave-one-out CV (closed-form), residual plots
- **Preprocessing pipeline** — standard scaling and PCA, chainable with the model
- **DataFrame-friendly** — pass numeric pandas DataFrames in directly
- **Two optimizer backends** — SciPy L-BFGS-B (default, fast) or PyTorch Adam (autograd, flexible)
- **Persistence** — save and load fitted models via pickle
- **Streamlit app** — interactive frontend for non-coders

> **Note:** GPReg accepts continuous (numeric) inputs and outputs only. Categorical features must be transformed to numeric (e.g., target encoding, embeddings) before being passed in.

## Installation

```bash
pip install -r requirements.txt
```

Or for a development install:

```bash
pip install -e .
```

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

## Numeric DataFrames

```python
import pandas as pd
from gpreg import make_gp_pipeline, GaussianProcessRegressor, RBF, White

df = pd.DataFrame({
    'temperature': [...],
    'humidity':    [...],
    'pressure':    [...],
})
pipe = make_gp_pipeline(GaussianProcessRegressor(kernel=RBF() + White(0.1)))
pipe.fit(df, target_array)
y_mean, y_std = pipe.predict(df_new, return_std=True)
```

## Multivariate outputs

```python
import pandas as pd
from gpreg import GaussianProcessRegressor, MultiOutputGP, RBF, White

# Y can be a numeric DataFrame (column names preserved) or a (n, p) array
Y = pd.DataFrame({'temp': [...], 'humidity': [...], 'pressure': [...]})

base = GaussianProcessRegressor(kernel=RBF() + White(0.1))
mgp = MultiOutputGP(base)
mgp.fit(X, Y)

Y_mean, Y_std = mgp.predict(X_new, return_std=True)
mgp.per_output_scores(X, Y)
```

## Sparse GP for large data

```python
from gpreg import SparseGPRegressor

sparse = SparseGPRegressor(
    kernel=RBF() + White(0.1),
    n_inducing=100,
    inducing_strategy='kmeans',
)
sparse.fit(X, y)
```

## Running the app

```bash
streamlit run app.py
```

## Package layout

```
gpreg/
├── kernels/         RBF, Matern, Linear, White; SumKernel/ProductKernel
├── models/          GaussianProcessRegressor, SparseGPRegressor, MultiOutputGP, torch_backend
├── preprocessing/   StandardScaler, PCA, Pipeline
├── diagnostics/     metrics (RMSE, NLPD, LOO-CV) and plots
└── utils/           Cholesky helpers, custom exceptions, save/load
```

## References

- Rasmussen & Williams, *Gaussian Processes for Machine Learning* (2006)
- Snelson & Ghahramani, *Sparse Gaussian Processes using Pseudo-inputs* (NIPS 2006) — FITC approximation
