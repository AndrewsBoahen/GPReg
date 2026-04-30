"""GPReg interactive Streamlit app.

Run with:
    streamlit run app.py

Features:
- Upload a CSV (numeric columns only) or use built-in demo datasets
- Pick target column(s) and feature columns (all must be continuous)
- Choose kernel (RBF, Matern 1/2, 3/2, 5/2)
- Toggle exact vs sparse GP
- See predictions, residuals, and diagnostic metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from gpreg import (
    GaussianProcessRegressor,
    SparseGPRegressor,
    MultiOutputGP,
    RBF,
    Matern,
    White,
    make_gp_pipeline,
    rmse,
    nlpd,
    loo_cv,
)
from gpreg.diagnostics import (
    plot_predictions_2d,
    plot_residuals,
)


st.set_page_config(page_title="GPReg Demo", layout="wide")

st.title("GPReg — Gaussian Process Regression")
st.caption(
    "Interactive demo of the GPReg package. Upload a numeric dataset or pick "
    "a demo, configure the kernel, and watch the GP fit in real time. "
    "GPReg accepts continuous (numeric) inputs and outputs only."
)


# ============================================================================
# Sidebar: data and configuration
# ============================================================================
st.sidebar.header("1. Data")

data_source = st.sidebar.radio(
    "Source",
    ["Demo: 1D sine", "Demo: 2D peaks", "Demo: multi-output", "Upload CSV"],
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
def load_demo_multi(seed=42, n=80):
    rng = np.random.default_rng(seed)
    x = np.linspace(-3, 3, n)
    return pd.DataFrame({
        "x": x,
        "sin":      np.sin(x) + 0.1 * rng.standard_normal(n),
        "cos":      np.cos(x) + 0.1 * rng.standard_normal(n),
        "parabola": 0.5 * x ** 2 - 1.0 + 0.1 * rng.standard_normal(n),
    })


df = None
if data_source == "Demo: 1D sine":
    df = load_demo_1d()
elif data_source == "Demo: 2D peaks":
    df = load_demo_2d()
elif data_source == "Demo: multi-output":
    df = load_demo_multi()
else:
    upload = st.sidebar.file_uploader("CSV file (numeric columns only)", type=["csv"])
    if upload is not None:
        df = pd.read_csv(upload)

if df is None:
    st.info("← Pick a demo dataset or upload a CSV in the sidebar to get started.")
    st.stop()

# Check that uploaded data is fully numeric
from pandas.api.types import is_numeric_dtype
non_numeric = [c for c in df.columns if not is_numeric_dtype(df[c])]
if non_numeric:
    st.error(
        f"GPReg accepts continuous (numeric) data only. The following columns are "
        f"non-numeric and cannot be used: {non_numeric}. Please convert them to "
        f"numeric or drop them from your CSV."
    )
    st.stop()

st.subheader("Data preview")
st.dataframe(df.head(8), use_container_width=True)

# ============================================================================
# Exploratory Data Analysis (EDA) — collapsed by default
# ============================================================================
with st.expander("🔍 Exploratory data analysis (optional)", expanded=False):
    st.caption(
        "Inspect your data before fitting the GP. Use this to spot outliers, "
        "check distributions, and see whether features are correlated with the target."
    )
    
    # Summary statistics for the entire dataset
    st.markdown("**Summary statistics**")
    st.dataframe(df.describe().round(4), use_container_width=True)
    
    eda_col1, eda_col2 = st.columns([1, 2])
    
    with eda_col1:
        plot_type = st.radio(
            "Plot type",
            ["Histogram", "Box plot", "Scatter (vs another variable)",
             "Correlation heatmap"],
            key="eda_plot_type",
        )
    
    with eda_col2:
        if plot_type in ("Histogram", "Box plot"):
            eda_var = st.selectbox(
                "Variable to inspect",
                df.columns.tolist(),
                key="eda_var",
            )
        elif plot_type == "Scatter (vs another variable)":
            eda_x = st.selectbox(
                "X-axis variable", df.columns.tolist(), key="eda_x"
            )
            eda_y = st.selectbox(
                "Y-axis variable", df.columns.tolist(),
                index=min(1, len(df.columns) - 1), key="eda_y"
            )
    
    # Render the chosen plot
    if plot_type == "Histogram":
        fig, ax = plt.subplots(figsize=(8, 4))
        values = df[eda_var].dropna()
        ax.hist(values, bins=min(30, max(10, int(np.sqrt(len(values))))),
                edgecolor="black", alpha=0.75, color="C0")
        ax.axvline(values.mean(), color="red", linestyle="--",
                   linewidth=1.5, label=f"mean = {values.mean():.3f}")
        ax.axvline(values.median(), color="orange", linestyle=":",
                   linewidth=1.5, label=f"median = {values.median():.3f}")
        ax.set_xlabel(eda_var)
        ax.set_ylabel("Frequency")
        ax.set_title(f"Distribution of {eda_var}")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
        # Quick interpretation hints
        skewness = values.skew()
        if abs(skewness) > 1:
            st.caption(
                f"⚠️ Skewness = {skewness:.2f} (highly skewed). "
                f"Consider log-transforming this variable before fitting."
            )
        else:
            st.caption(f"Skewness = {skewness:.2f} (roughly symmetric).")
    
    elif plot_type == "Box plot":
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(df[eda_var].dropna(), vert=True, patch_artist=True,
                   boxprops=dict(facecolor="C0", alpha=0.5))
        ax.set_ylabel(eda_var)
        ax.set_title(f"Box plot of {eda_var}")
        ax.grid(alpha=0.3, axis="y")
        st.pyplot(fig)
        plt.close(fig)
        # Outlier hint using IQR rule
        q1, q3 = df[eda_var].quantile([0.25, 0.75])
        iqr = q3 - q1
        n_outliers = ((df[eda_var] < q1 - 1.5 * iqr) | (df[eda_var] > q3 + 1.5 * iqr)).sum()
        if n_outliers > 0:
            st.caption(
                f"⚠️ {n_outliers} potential outliers (beyond 1.5×IQR). "
                f"GPs are sensitive to outliers — consider removing or down-weighting them."
            )
        else:
            st.caption("No outliers detected by the 1.5×IQR rule.")
    
    elif plot_type == "Scatter (vs another variable)":
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(df[eda_x], df[eda_y], alpha=0.6, s=30, color="C0",
                   edgecolors="black", linewidths=0.5)
        ax.set_xlabel(eda_x)
        ax.set_ylabel(eda_y)
        ax.set_title(f"{eda_y} vs {eda_x}")
        ax.grid(alpha=0.3)
        # Linear correlation
        corr = df[[eda_x, eda_y]].corr().iloc[0, 1]
        ax.text(0.05, 0.95, f"Pearson r = {corr:.3f}",
                transform=ax.transAxes, fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        st.pyplot(fig)
        plt.close(fig)
        if abs(corr) > 0.7:
            st.caption(
                f"Strong linear relationship (|r| = {abs(corr):.2f}). "
                f"A linear kernel component (or RBF with long length-scale) may capture this."
            )
        elif abs(corr) < 0.1:
            st.caption(
                f"Weak linear relationship (|r| = {abs(corr):.2f}). "
                f"This doesn't rule out nonlinear structure that a GP could still pick up."
            )
        else:
            st.caption(f"Moderate linear relationship (|r| = {abs(corr):.2f}).")
    
    elif plot_type == "Correlation heatmap":
        import seaborn as sns
        corr_matrix = df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(min(10, 1 + len(corr_matrix)),
                                        min(8, 1 + len(corr_matrix))))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, vmin=-1, vmax=1, square=True,
                    cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title("Pairwise correlations")
        st.pyplot(fig)
        plt.close(fig)
        # Find highly correlated feature pairs (excluding the diagonal)
        high_corr_pairs = []
        cols = corr_matrix.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if abs(corr_matrix.iloc[i, j]) > 0.85:
                    high_corr_pairs.append(
                        (cols[i], cols[j], corr_matrix.iloc[i, j])
                    )
        if high_corr_pairs:
            msg = "⚠️ Highly correlated feature pairs (|r| > 0.85): "
            msg += ", ".join(f"{a}↔{b} ({r:.2f})" for a, b, r in high_corr_pairs)
            st.caption(msg + ". Consider dropping one or using PCA to reduce redundancy.")

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
else:
    target_col = st.sidebar.selectbox(
        "Target column",
        all_columns,
        index=len(all_columns) - 1,
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
init_length_scale = st.sidebar.slider("Initial length-scale", 0.1, 5.0, 1.0, 0.1)
init_noise = st.sidebar.slider("Initial noise variance", 0.001, 1.0, 0.1, 0.01)

model_type = st.sidebar.radio("Model type", ["Exact GP", "Sparse GP (FITC)"])
if model_type == "Sparse GP (FITC)":
    n_inducing = st.sidebar.slider("Inducing points", 10, 200, 50, 10)

n_restarts = st.sidebar.slider("Optimization restarts", 0, 10, 3)
random_state = st.sidebar.number_input("Random seed", value=0)

st.sidebar.header("3. Train/test split")
test_fraction = st.sidebar.slider(
    "Fraction of data held out for testing", 0.1, 0.5, 0.25, 0.05,
    help="Data is split into training (used to fit the GP) and test (used to evaluate). "
         "GPs interpolate training data nearly perfectly, so test metrics are the honest measure."
)


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
    
    if len(target_cols) > 1:
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
            # Train/test split before fitting
            n_total = len(df)
            n_test = max(1, int(round(test_fraction * n_total)))
            n_train = n_total - n_test
            if n_train < 5:
                st.error(
                    f"Not enough training data: only {n_train} samples remain after the "
                    f"test split. Increase the dataset size or reduce the test fraction."
                )
                st.stop()
            
            rng = np.random.default_rng(int(random_state))
            shuffled = rng.permutation(n_total)
            train_idx = shuffled[:n_train]
            test_idx = shuffled[n_train:]
            
            df_train = df.iloc[train_idx].reset_index(drop=True)
            df_test = df.iloc[test_idx].reset_index(drop=True)
            
            X_train_df = df_train[feature_cols]
            X_test_df = df_test[feature_cols]
            
            if len(target_cols) > 1:
                y_train = df_train[target_cols]
                y_test = df_test[target_cols]
            else:
                y_train = df_train[target_cols[0]].values.astype(float)
                y_test = df_test[target_cols[0]].values.astype(float)
            
            pipe = build_model()
            pipe.fit(X_train_df, y_train)
            
            st.session_state["fitted_pipeline"] = pipe
            st.session_state["fit_X_train"] = X_train_df
            st.session_state["fit_y_train"] = y_train
            st.session_state["fit_X_test"] = X_test_df
            st.session_state["fit_y_test"] = y_test
            st.session_state["fit_config"] = {
                "kernel": kernel_choice,
                "model_type": model_type,
                "feature_cols": feature_cols,
                "target_cols": target_cols,
                "multi_target": len(target_cols) > 1,
                "n_train": n_train,
                "n_test": n_test,
                "test_fraction": test_fraction,
            }
            st.success(
                f"Model fitted on {n_train} training samples; "
                f"{n_test} held out for testing."
            )
        except Exception as e:
            st.error(f"Fitting failed: {e}")

pipe = st.session_state["fitted_pipeline"]


# ============================================================================
# Results display
# ============================================================================
if pipe is not None:
    cfg = st.session_state["fit_config"]
    X_train_df = st.session_state["fit_X_train"]
    y_train = st.session_state["fit_y_train"]
    X_test_df = st.session_state["fit_X_test"]
    y_test = st.session_state["fit_y_test"]
    
    st.header("Fitted model")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training samples", cfg["n_train"])
    with col2:
        st.metric("Test samples", cfg["n_test"])
    with col3:
        try:
            st.metric("Log marginal likelihood", f"{pipe.log_marginal_likelihood_:.2f}")
        except Exception:
            try:
                st.metric("Log marginal likelihood",
                          f"{pipe.estimator.log_marginal_likelihood_:.2f}")
            except Exception:
                pass
    
    if cfg["multi_target"]:
        st.subheader("Fitted kernels per output")
        for name, sub_pipe in zip(cfg["target_cols"], pipe.estimators_):
            st.code(f"{name}: {sub_pipe.estimator.kernel!r}", language="text")
    else:
        st.code(repr(pipe.estimator.kernel), language="text")
    
    
    # Get predictions on the TEST set for evaluation
    y_pred, y_std = pipe.predict(X_test_df, return_std=True)
    
    st.header("Predictions")
    
    if cfg["multi_target"]:
        # Multi-output: predicted vs actual on the test set, one panel per output
        y_test_arr = y_test.values if hasattr(y_test, "values") else np.asarray(y_test)
        n_out = len(cfg["target_cols"])
        fig, axes = plt.subplots(1, n_out, figsize=(5 * n_out, 5), squeeze=False)
        for j, (name, ax) in enumerate(zip(cfg["target_cols"], axes[0])):
            actual = y_test_arr[:, j]
            predicted = y_pred[:, j]
            std = y_std[:, j]
            lo, hi = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
            ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y = x")
            ax.errorbar(actual, predicted, yerr=2 * std, fmt="o",
                        alpha=0.5, markersize=4, capsize=2)
            ax.set_xlabel(f"Actual {name}")
            ax.set_ylabel("Predicted")
            ax.set_title(f"Output: {name} (test data)")
            ax.legend()
            ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
    
    elif len(cfg["feature_cols"]) == 1:
        # 1D plot: show GP curve over the full input range, with both train and test points
        col_name = cfg["feature_cols"][0]
        x_min = min(X_train_df[col_name].min(), X_test_df[col_name].min())
        x_max = max(X_train_df[col_name].max(), X_test_df[col_name].max())
        margin = 0.15 * (x_max - x_min)
        x_plot = np.linspace(x_min - margin, x_max + margin, 200)
        X_plot_df = pd.DataFrame({col_name: x_plot})
        
        y_plot, y_plot_std = pipe.predict(X_plot_df, return_std=True)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.fill_between(x_plot, y_plot - 2 * y_plot_std, y_plot + 2 * y_plot_std,
                        alpha=0.25, color="C0", label="±2σ band")
        ax.plot(x_plot, y_plot, color="C0", linewidth=2, label="GP mean")
        ax.scatter(X_train_df[col_name], y_train, color="black", s=20, zorder=5,
                   label="Training data")
        ax.scatter(X_test_df[col_name], y_test, color="red", s=30, marker="x",
                   zorder=5, label="Test data")
        ax.set_xlabel(col_name)
        ax.set_ylabel(cfg["target_cols"][0])
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
    
    elif len(cfg["feature_cols"]) == 2 and all(
        pd.api.types.is_numeric_dtype(X_train_df[c]) for c in cfg["feature_cols"]
    ):
        c1, c2 = cfg["feature_cols"]
        x1_min = min(X_train_df[c1].min(), X_test_df[c1].min())
        x1_max = max(X_train_df[c1].max(), X_test_df[c1].max())
        x2_min = min(X_train_df[c2].min(), X_test_df[c2].min())
        x2_max = max(X_train_df[c2].max(), X_test_df[c2].max())
        x1_range = (x1_min, x1_max)
        x2_range = (x2_min, x2_max)
        
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
            ax.scatter(X_train_df[c1], X_train_df[c2], c="white",
                       edgecolors="black", s=30, label="Train")
            ax.scatter(X_test_df[c1], X_test_df[c2], c="red",
                       edgecolors="black", s=40, marker="x", label="Test")
            ax.set_xlabel(c1)
            ax.set_ylabel(c2)
            ax.set_title("Predictive mean")
            ax.legend(loc="upper right", fontsize=8)
            st.pyplot(fig)
            plt.close(fig)
        with col_b:
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(Z_std, extent=[*x1_range, *x2_range],
                           origin="lower", aspect="auto", cmap="magma")
            plt.colorbar(im, ax=ax)
            ax.scatter(X_train_df[c1], X_train_df[c2], c="white",
                       edgecolors="black", s=30)
            ax.scatter(X_test_df[c1], X_test_df[c2], c="red",
                       edgecolors="black", s=40, marker="x")
            ax.set_xlabel(c1)
            ax.set_ylabel(c2)
            ax.set_title("Predictive std")
            st.pyplot(fig)
            plt.close(fig)
    else:
        # Higher dim: show predicted vs actual on test set
        fig, ax = plt.subplots(figsize=(7, 6))
        lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y = x")
        ax.errorbar(y_test, y_pred, yerr=2 * y_std, fmt="o", alpha=0.5,
                    markersize=4, capsize=2)
        ax.set_xlabel(f"Actual {cfg['target_cols'][0]}")
        ax.set_ylabel("Predicted")
        ax.set_title("Predicted vs Actual (test data)")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
    
    # ============================================================================
    # Diagnostics — based on the held-out test set
    # ============================================================================
    st.header("Diagnostics (test set)")
    
    if cfg["multi_target"]:
        y_test_arr = y_test.values if hasattr(y_test, "values") else np.asarray(y_test)
        rows = []
        for j, name in enumerate(cfg["target_cols"]):
            rows.append({
                "Output": name,
                "Test RMSE": f"{rmse(y_test_arr[:, j], y_pred[:, j]):.4f}",
                "Test NLPD": f"{nlpd(y_test_arr[:, j], y_pred[:, j], y_std[:, j]):.4f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.caption(
            "ℹ️ Each output has its own independent GP. "
            "Residual plot below is shown for the first output only."
        )
        first_idx = 0
        fig, _ = plot_residuals(y_test_arr[:, first_idx], y_pred[:, first_idx],
                                y_std[:, first_idx])
        fig.suptitle(f"Residuals: {cfg['target_cols'][first_idx]} (test set)")
        st.pyplot(fig)
        plt.close(fig)
    else:
        diag_col1, diag_col2, diag_col3 = st.columns(3)
        test_rmse = rmse(y_test, y_pred)
        test_nlpd = nlpd(y_test, y_pred, y_std)
        
        with diag_col1:
            st.metric("Test RMSE", f"{test_rmse:.4f}")
        with diag_col2:
            st.metric("Test NLPD", f"{test_nlpd:.4f}")
        with diag_col3:
            if cfg["model_type"] == "Exact GP":
                try:
                    loo_results = loo_cv(pipe.estimator)
                    st.metric("LOO-CV RMSE", f"{loo_results['loo_rmse']:.4f}")
                except Exception:
                    st.metric("LOO-CV RMSE", "N/A")
            else:
                st.metric("LOO-CV RMSE", "N/A (sparse)")
        
        fig, _ = plot_residuals(y_test, y_pred, y_std)
        st.pyplot(fig)
        plt.close(fig)
        
        st.caption(
            "ℹ️ Test RMSE/NLPD are evaluated on data the model has never seen — "
            "the honest measure of generalization. LOO-CV provides an additional "
            "estimate using the training set alone. "
            "Standardized residuals should mostly fall within ±2 with no obvious pattern."
        )

else:
    st.info("Configure the model in the sidebar and click **Fit model** to begin.")
