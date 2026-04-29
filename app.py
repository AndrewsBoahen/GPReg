"""GPReg interactive Streamlit app.

Run with:
    streamlit run app.py

Features:
- Upload a CSV or use built-in demo datasets
- Pick target column and feature columns
- Choose kernel (RBF, Matern 1/2, 3/2, 5/2)
- Toggle exact vs sparse GP
- See predictions, calibration, residuals, and diagnostics
"""

import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from gpreg import (
    GaussianProcessRegressor,
    SparseGPRegressor,
    RBF,
    Matern,
    White,
    make_gp_pipeline,
    rmse,
    nlpd,
    loo_cv,
)
from gpreg.diagnostics import (
    plot_predictions_1d,
    plot_predictions_2d,
    plot_calibration,
    plot_residuals,
)


st.set_page_config(page_title="GPReg Demo", layout="wide")

st.title("GPReg — Gaussian Process Regression")
st.caption(
    "Interactive demo of the GPReg package. Upload a dataset or pick a demo, "
    "configure the kernel, and watch the GP fit in real time."
)


# ============================================================================
# Sidebar: data and configuration
# ============================================================================
st.sidebar.header("1. Data")

data_source = st.sidebar.radio(
    "Source",
    ["Demo: 1D sine", "Demo: 2D peaks", "Demo: mixed-types DataFrame", "Upload CSV"],
)


@st.cache_data
def load_demo_1d(seed=42, n=40):
    rng = np.random.default_rng(seed)
    x = np.linspace(-3, 3, n)
    y = np.sin(x) + 0.5 * np.cos(0.3 * x) + 0.15 * rng.standard_normal(n)
    return pd.DataFrame({"x": x, "y": y})


@st.cache_data
def load_demo_2d(seed=42, n=120):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-3, 3, n)
    x2 = rng.uniform(-3, 3, n)
    y = np.exp(-(x1 ** 2 + x2 ** 2) / 4) * np.sin(2 * x1) + 0.1 * rng.standard_normal(n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


@st.cache_data
def load_demo_mixed(seed=42, n=200):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "temperature": rng.uniform(10, 30, n),
        "humidity": rng.uniform(0.2, 0.9, n),
        "season": rng.choice(["spring", "summer", "fall", "winter"], n),
    })
    season_effect = df["season"].map(
        {"spring": 1.0, "summer": 2.5, "fall": 0.5, "winter": -1.0}
    )
    df["y"] = (
        0.1 * df["temperature"]
        - 1.5 * df["humidity"]
        + season_effect
        + 0.2 * rng.standard_normal(n)
    )
    return df


df = None
if data_source == "Demo: 1D sine":
    df = load_demo_1d()
elif data_source == "Demo: 2D peaks":
    df = load_demo_2d()
elif data_source == "Demo: mixed-types DataFrame":
    df = load_demo_mixed()
else:
    upload = st.sidebar.file_uploader("CSV file", type=["csv"])
    if upload is not None:
        df = pd.read_csv(upload)

if df is None:
    st.info("← Pick a demo dataset or upload a CSV in the sidebar to get started.")
    st.stop()

st.subheader("Data preview")
st.dataframe(df.head(8), use_container_width=True)

# Target / feature selection
all_columns = list(df.columns)
multi_target = st.sidebar.checkbox("Multivariate outputs (predict several targets)", value=False)
if multi_target:
    target_cols = st.sidebar.multiselect(
        "Target columns",
        all_columns,
        default=[all_columns[-1]],
    )
    if not target_cols:
        st.error("Pick at least one target column.")
        st.stop()
    target_col = target_cols[0]  # for compatibility with downstream UI checks
else:
    target_col = st.sidebar.selectbox(
        "Target column",
        all_columns,
        index=len(all_columns) - 1,  # default: last column
    )
    target_cols = [target_col]
feature_cols = st.sidebar.multiselect(
    "Feature columns",
    [c for c in all_columns if c not in target_cols],
    default=[c for c in all_columns if c not in target_cols],
)

if not feature_cols:
    st.error("Pick at least one feature column.")
    st.stop()


# ============================================================================
# Sidebar: kernel + model config
# ============================================================================
st.sidebar.header("2. Model")

kernel_choice = st.sidebar.selectbox(
    "Kernel",
    ["RBF", "Matern 1/2", "Matern 3/2", "Matern 5/2"],
)
init_length_scale = st.sidebar.slider(
    "Initial length-scale", 0.1, 5.0, 1.0, 0.1
)
init_noise = st.sidebar.slider(
    "Initial noise variance", 0.001, 1.0, 0.1, 0.01
)

model_type = st.sidebar.radio(
    "Model type",
    ["Exact GP", "Sparse GP (FITC)"],
)
if model_type == "Sparse GP (FITC)":
    n_inducing = st.sidebar.slider(
        "Inducing points", 10, 200, 50, 10
    )

n_restarts = st.sidebar.slider("Optimization restarts", 0, 10, 3)
random_state = st.sidebar.number_input("Random seed", value=0)


# ============================================================================
# Build the model and fit
# ============================================================================
def build_kernel():
    if kernel_choice == "RBF":
        base = RBF(length_scale=init_length_scale, signal_var=1.0)
    else:
        nu = {"Matern 1/2": 0.5, "Matern 3/2": 1.5, "Matern 5/2": 2.5}[kernel_choice]
        base = Matern(length_scale=init_length_scale, signal_var=1.0, nu=nu)
    return base + White(noise_var=init_noise)


def build_model():
    kernel = build_kernel()
    if model_type == "Exact GP":
        gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts=n_restarts, random_state=int(random_state)
        )
    else:
        gp = SparseGPRegressor(
            kernel=kernel, n_inducing=n_inducing,
            n_restarts=n_restarts, random_state=int(random_state),
        )
    
    # If user picked multiple targets, wrap in MultiOutputGP.
    # Otherwise, use the single-output pipeline as before.
    if len(target_cols) > 1:
        from gpreg import MultiOutputGP
        return MultiOutputGP(make_gp_pipeline(gp))
    else:
        return make_gp_pipeline(gp)


fit_button = st.sidebar.button("Fit model", type="primary")

if "fitted_pipeline" not in st.session_state:
    st.session_state["fitted_pipeline"] = None
    st.session_state["fit_config"] = None

if fit_button:
    with st.spinner("Fitting the GP..."):
        try:
            X_df = df[feature_cols]
            if len(target_cols) > 1:
                y = df[target_cols]  # DataFrame
            else:
                y = df[target_cols[0]].values.astype(float)
            pipe = build_model()
            pipe.fit(X_df, y)
            st.session_state["fitted_pipeline"] = pipe
            st.session_state["fit_X"] = X_df
            st.session_state["fit_y"] = y
            st.session_state["fit_config"] = {
                "kernel": kernel_choice,
                "model_type": model_type,
                "feature_cols": feature_cols,
                "target_cols": target_cols,
                "multi_target": len(target_cols) > 1,
            }
            st.success("Model fitted successfully!")
        except Exception as e:
            st.error(f"Fitting failed: {e}")

pipe = st.session_state["fitted_pipeline"]


# ============================================================================
# Results display
# ============================================================================
if pipe is not None:
    cfg = st.session_state["fit_config"]
    X_df = st.session_state["fit_X"]
    y = st.session_state["fit_y"]
    
    st.header("Fitted model")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training samples", len(X_df))
        st.metric("Output dimensions", len(cfg["target_cols"]))
    with col2:
        try:
            st.metric("Log marginal likelihood",
                      f"{pipe.log_marginal_likelihood_:.2f}")
        except Exception:
            try:
                st.metric("Log marginal likelihood",
                          f"{pipe.estimator.log_marginal_likelihood_:.2f}")
            except Exception:
                pass
    
    # Display fitted kernels (one per output for multi-output)
    if cfg["multi_target"]:
        st.subheader("Fitted kernels per output")
        for name, sub_pipe in zip(cfg["target_cols"], pipe.estimators_):
            st.code(f"{name}: {sub_pipe.estimator.kernel!r}", language="text")
    else:
        st.code(repr(pipe.estimator.kernel), language="text")
    
    # Predictions
    y_pred, y_std = pipe.predict(X_df, return_std=True)
    
    st.header("Predictions")
    
    if cfg["multi_target"]:
        # Show a predicted-vs-actual plot for each output
        y_arr = y.values if hasattr(y, "values") else np.asarray(y)
        
        n_out = len(cfg["target_cols"])
        fig, axes = plt.subplots(1, n_out, figsize=(5 * n_out, 5), squeeze=False)
        for j, (name, ax) in enumerate(zip(cfg["target_cols"], axes[0])):
            actual = y_arr[:, j]
            predicted = y_pred[:, j]
            std = y_std[:, j]
            lo, hi = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
            ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y = x")
            ax.errorbar(actual, predicted, yerr=2 * std, fmt="o",
                        alpha=0.5, markersize=4, capsize=2)
            ax.set_xlabel(f"Actual {name}")
            ax.set_ylabel("Predicted")
            ax.set_title(f"Output: {name}")
            ax.legend()
            ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
    
    elif len(cfg["feature_cols"]) == 1:
        # 1D: plot prediction curve
        col_name = cfg["feature_cols"][0]
        x_min, x_max = X_df[col_name].min(), X_df[col_name].max()
        margin = 0.15 * (x_max - x_min)
        x_plot = np.linspace(x_min - margin, x_max + margin, 200)
        X_plot_df = pd.DataFrame({col_name: x_plot})
        
        y_plot, y_plot_std = pipe.predict(X_plot_df, return_std=True)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.fill_between(x_plot, y_plot - 2 * y_plot_std, y_plot + 2 * y_plot_std,
                        alpha=0.25, color="C0", label="±2σ band")
        ax.plot(x_plot, y_plot, color="C0", linewidth=2, label="GP mean")
        ax.scatter(X_df[col_name], y, color="black", s=20, zorder=5,
                   label="Training data")
        ax.set_xlabel(col_name)
        ax.set_ylabel(cfg["target_cols"][0])
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
    
    elif len(cfg["feature_cols"]) == 2 and all(
        pd.api.types.is_numeric_dtype(X_df[c]) for c in cfg["feature_cols"]
    ):
        # 2D numeric: heatmap
        c1, c2 = cfg["feature_cols"]
        x1_range = (X_df[c1].min(), X_df[c1].max())
        x2_range = (X_df[c2].min(), X_df[c2].max())
        
        n_grid = 40
        x1_g = np.linspace(*x1_range, n_grid)
        x2_g = np.linspace(*x2_range, n_grid)
        X1, X2 = np.meshgrid(x1_g, x2_g)
        grid_df = pd.DataFrame({c1: X1.ravel(), c2: X2.ravel()})
        Z_mean = pipe.predict(grid_df).reshape(X1.shape)
        _, Z_std = pipe.predict(grid_df, return_std=True)
        Z_std = Z_std.reshape(X1.shape)
        
        col_a, col_b = st.columns(2)
        with col_a:
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(Z_mean, extent=[*x1_range, *x2_range],
                           origin="lower", aspect="auto", cmap="viridis")
            plt.colorbar(im, ax=ax)
            ax.scatter(X_df[c1], X_df[c2], c="white", edgecolors="black", s=30)
            ax.set_xlabel(c1)
            ax.set_ylabel(c2)
            ax.set_title("Predictive mean")
            st.pyplot(fig)
            plt.close(fig)
        with col_b:
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(Z_std, extent=[*x1_range, *x2_range],
                           origin="lower", aspect="auto", cmap="magma")
            plt.colorbar(im, ax=ax)
            ax.scatter(X_df[c1], X_df[c2], c="white", edgecolors="black", s=30)
            ax.set_xlabel(c1)
            ax.set_ylabel(c2)
            ax.set_title("Predictive std")
            st.pyplot(fig)
            plt.close(fig)
    else:
        # Higher dim or mixed types: show predicted vs actual
        fig, ax = plt.subplots(figsize=(7, 6))
        lo, hi = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y = x")
        ax.errorbar(y, y_pred, yerr=2 * y_std, fmt="o", alpha=0.5,
                    markersize=4, capsize=2)
        ax.set_xlabel(f"Actual {cfg['target_cols'][0]}")
        ax.set_ylabel("Predicted")
        ax.set_title("Predicted vs Actual (training data)")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
    
    # ============================================================================
    # Diagnostics
    # ============================================================================
    st.header("Diagnostics")
    
    if cfg["multi_target"]:
        # Show per-output metrics in a table
        y_arr = y.values if hasattr(y, "values") else np.asarray(y)
        rows = []
        for j, name in enumerate(cfg["target_cols"]):
            rows.append({
                "Output": name,
                "RMSE": f"{rmse(y_arr[:, j], y_pred[:, j]):.4f}",
                "NLPD": f"{nlpd(y_arr[:, j], y_pred[:, j], y_std[:, j]):.4f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.caption(
            "ℹ️ Each output has its own independent GP with its own hyperparameters. "
            "Diagnostic plots (calibration, residuals) are shown for the first output below."
        )
        # Show calibration/residuals only for first output to keep the UI tidy
        first_idx = 0
        plot_col1, plot_col2 = st.columns(2)
        with plot_col1:
            fig, _ = plot_calibration(y_arr[:, first_idx], y_pred[:, first_idx],
                                      y_std[:, first_idx], n_bins=10)
            fig.suptitle(f"Calibration: {cfg['target_cols'][first_idx]}")
            st.pyplot(fig)
            plt.close(fig)
        with plot_col2:
            fig, _ = plot_residuals(y_arr[:, first_idx], y_pred[:, first_idx],
                                    y_std[:, first_idx])
            fig.suptitle(f"Residuals: {cfg['target_cols'][first_idx]}")
            st.pyplot(fig)
            plt.close(fig)
    else:
        diag_col1, diag_col2, diag_col3 = st.columns(3)
        train_rmse = rmse(y, y_pred)
        train_nlpd = nlpd(y, y_pred, y_std)
        
        with diag_col1:
            st.metric("Training RMSE", f"{train_rmse:.4f}")
        with diag_col2:
            st.metric("Training NLPD", f"{train_nlpd:.4f}")
        with diag_col3:
            if cfg["model_type"] == "Exact GP":
                try:
                    loo_results = loo_cv(pipe.estimator)
                    st.metric("LOO-CV RMSE", f"{loo_results['loo_rmse']:.4f}")
                except Exception:
                    st.metric("LOO-CV RMSE", "N/A")
            else:
                st.metric("LOO-CV RMSE", "N/A (sparse)")
        
        # Calibration + residuals
        plot_col1, plot_col2 = st.columns(2)
        with plot_col1:
            fig, _ = plot_calibration(y, y_pred, y_std, n_bins=10)
            st.pyplot(fig)
            plt.close(fig)
        with plot_col2:
            fig, _ = plot_residuals(y, y_pred, y_std)
            st.pyplot(fig)
            plt.close(fig)
        
        st.caption(
            "ℹ️ Calibration above the diagonal = under-confident (intervals too wide). "
            "Below = over-confident (intervals too narrow, which is the bigger problem). "
            "Standardized residuals should mostly fall within ±2 with no obvious pattern."
        )

else:
    st.info("Configure the model in the sidebar and click **Fit model** to begin.")
