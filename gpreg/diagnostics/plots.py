"""Plotting utilities for GP regression.

All plotting functions return (figure, axes) so users can customize
styling, save figures, or add to subplots. They never call plt.show()
internally — that's the caller's responsibility.

Uses matplotlib for the core plots and seaborn for multivariate
visualizations (pair plots, kernel heatmaps).
"""

import numpy as np
import matplotlib.pyplot as plt

# Seaborn is imported lazily inside the functions that use it, so users
# without seaborn installed can still use the matplotlib-based plots.

from .metrics import calibration_curve
from ..utils.exceptions import NotFittedError


def plot_predictions_1d(gp, X_test, y_train_obs=None, X_train=None,
                        n_samples=0, ax=None, ci=2.0,
                        true_fn=None, random_state=None):
    """Plot GP predictions for 1D inputs with uncertainty bands.
    
    The signature plot for GP regression: predictive mean as a line,
    confidence band (default ±2σ ≈ 95%) as a shaded region, optionally
    overlaid with training points and posterior samples.
    
    Parameters
    ----------
    gp : GaussianProcessRegressor
        Fitted model. Must have 1D input.
    X_test : array-like of shape (m, 1) or (m,)
        Test points to predict at. Should be sorted for clean plotting.
    y_train_obs : array-like, optional
        Original training targets to overlay as scatter points.
    X_train : array-like, optional
        Training inputs to overlay. If None, uses gp.X_train_.
    n_samples : int, default=0
        Number of posterior function samples to draw and overlay.
    ax : matplotlib axis, optional
        Axis to draw on. If None, a new figure is created.
    ci : float, default=2.0
        Width of confidence band in standard deviations (2.0 ≈ 95%).
    true_fn : callable, optional
        If provided, plot the true function f(X_test) for comparison.
    random_state : int, optional
        Seed for sample drawing.
    
    Returns
    -------
    fig, ax : matplotlib figure and axis
    """
    if not gp._fitted:
        raise NotFittedError("GP must be fitted before plotting.")
    
    X_test = np.asarray(X_test)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)
    
    if X_test.shape[1] != 1:
        raise ValueError(
            f"plot_predictions_1d requires 1D input, got {X_test.shape[1]}D. "
            f"Use plot_predictions_2d or plot_pair for higher dimensions."
        )
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.figure
    
    y_mean, y_std = gp.predict(X_test, return_std=True)
    x_flat = X_test.ravel()
    
    # Confidence band
    ax.fill_between(
        x_flat,
        y_mean - ci * y_std,
        y_mean + ci * y_std,
        alpha=0.25,
        color="C0",
        label=f"±{ci:.0f}σ predictive interval",
    )
    
    # Predictive mean
    ax.plot(x_flat, y_mean, color="C0", linewidth=2, label="GP mean")
    
    # Posterior samples
    if n_samples > 0:
        samples = gp.sample_y(X_test, n_samples=n_samples, random_state=random_state)
        for i in range(n_samples):
            ax.plot(x_flat, samples[:, i], color="C0", alpha=0.3, linewidth=0.8)
    
    # True function overlay
    if true_fn is not None:
        ax.plot(x_flat, true_fn(x_flat), "k--", linewidth=1.5, label="True function")
    
    # Training points
    if X_train is None:
        X_train = gp.X_train_
    if y_train_obs is None:
        y_train_obs = gp.y_train_ + gp.y_mean_
    
    ax.scatter(np.asarray(X_train).ravel(), np.asarray(y_train_obs).ravel(),
               color="black", s=30, zorder=5, label="Training data")
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Gaussian Process Regression")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    
    return fig, ax


def plot_predictions_2d(gp, x1_range, x2_range, n_grid=50, ax=None,
                        plot_type="mean", contour=True):
    """Plot GP predictions over a 2D input grid.
    
    Parameters
    ----------
    gp : GaussianProcessRegressor
        Fitted model with 2D input.
    x1_range, x2_range : tuple of (min, max)
        Range of each input dimension.
    n_grid : int, default=50
        Resolution of the prediction grid.
    ax : matplotlib axis, optional
    plot_type : {'mean', 'std'}, default='mean'
        Whether to visualize the predictive mean or standard deviation.
    contour : bool, default=True
        Whether to overlay contour lines.
    
    Returns
    -------
    fig, ax : matplotlib figure and axis
    """
    if not gp._fitted:
        raise NotFittedError("GP must be fitted before plotting.")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure
    
    x1 = np.linspace(*x1_range, n_grid)
    x2 = np.linspace(*x2_range, n_grid)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.column_stack([X1.ravel(), X2.ravel()])
    
    y_mean, y_std = gp.predict(X_grid, return_std=True)
    
    if plot_type == "mean":
        Z = y_mean.reshape(X1.shape)
        cmap = "viridis"
        title = "Predictive mean"
    elif plot_type == "std":
        Z = y_std.reshape(X1.shape)
        cmap = "magma"
        title = "Predictive std"
    else:
        raise ValueError(f"plot_type must be 'mean' or 'std', got {plot_type!r}")
    
    im = ax.imshow(Z, extent=[*x1_range, *x2_range], origin="lower",
                   cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax, label=title)
    
    if contour:
        ax.contour(X1, X2, Z, colors="white", alpha=0.4, linewidths=0.8)
    
    # Overlay training points
    ax.scatter(gp.X_train_[:, 0], gp.X_train_[:, 1],
               c="white", edgecolors="black", s=40, label="Training")
    
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title)
    ax.legend(loc="upper right")
    
    return fig, ax


def plot_calibration(y_true, y_mean, y_std, n_bins=10, ax=None):
    """Plot observed vs expected coverage for the predictive distribution.
    
    A perfectly calibrated GP sits on the diagonal. Above-diagonal points
    indicate under-confidence (the GP's intervals are wider than they
    need to be). Below-diagonal points indicate over-confidence (intervals
    too narrow — the bigger problem).
    
    Parameters
    ----------
    y_true : array-like
        True values from a held-out set.
    y_mean : array-like
        Predicted means.
    y_std : array-like
        Predicted standard deviations.
    n_bins : int, default=10
        Number of confidence levels to evaluate.
    ax : matplotlib axis, optional
    
    Returns
    -------
    fig, ax : matplotlib figure and axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure
    
    expected, observed = calibration_curve(y_true, y_mean, y_std, n_bins=n_bins)
    
    # Reference diagonal
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    
    # Actual calibration
    ax.plot(expected, observed, "o-", color="C0", markersize=8,
            linewidth=2, label="Observed")
    
    # Shade region of under/over confidence
    ax.fill_between([0, 1], [0, 1], 1, alpha=0.05, color="green",
                    label="Under-confident")
    ax.fill_between([0, 1], 0, [0, 1], alpha=0.05, color="red",
                    label="Over-confident")
    
    ax.set_xlabel("Expected coverage")
    ax.set_ylabel("Observed coverage")
    ax.set_title("Calibration diagram")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")
    
    return fig, ax


def plot_residuals(y_true, y_mean, y_std=None, ax=None):
    """Plot standardized residuals to diagnose model fit.
    
    Standardized residuals = (y_true - y_pred) / y_std. If the model is
    well-specified, these should look like draws from a standard normal:
    most within ±2, no obvious patterns vs. the prediction.
    
    Parameters
    ----------
    y_true, y_mean : array-like
        True and predicted values.
    y_std : array-like, optional
        Predictive standard deviations. If provided, plots standardized
        residuals; otherwise plots raw residuals.
    ax : matplotlib axis, optional
    
    Returns
    -------
    fig, ax : matplotlib figure and axis
    """
    y_true = np.asarray(y_true).ravel()
    y_mean = np.asarray(y_mean).ravel()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure
    
    if y_std is not None:
        y_std = np.asarray(y_std).ravel()
        residuals = (y_true - y_mean) / y_std
        ylabel = "Standardized residual"
        # Reference lines for ±2σ
        ax.axhline(2, color="red", linestyle=":", alpha=0.5)
        ax.axhline(-2, color="red", linestyle=":", alpha=0.5)
    else:
        residuals = y_true - y_mean
        ylabel = "Residual"
    
    ax.scatter(y_mean, residuals, alpha=0.6, s=30)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Predicted value")
    ax.set_ylabel(ylabel)
    ax.set_title("Residuals vs. predicted")
    ax.grid(alpha=0.3)
    
    return fig, ax


def plot_kernel_heatmap(kernel, X, ax=None):
    """Visualize the kernel matrix as a heatmap (uses seaborn).
    
    Useful for understanding what a composite kernel is doing or
    diagnosing why fitting failed (e.g., near-duplicate rows).
    
    Parameters
    ----------
    kernel : Kernel
        Any kernel from gpreg.kernels.
    X : array-like
        Inputs to compute the kernel matrix on.
    ax : matplotlib axis, optional
    
    Returns
    -------
    fig, ax
    """
    import seaborn as sns
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure
    
    K = kernel(X)
    sns.heatmap(K, ax=ax, cmap="viridis", cbar_kws={"label": "k(x_i, x_j)"})
    ax.set_title(f"Kernel matrix: {kernel!r}")
    ax.set_xlabel("Index j")
    ax.set_ylabel("Index i")
    
    return fig, ax


def plot_pair(gp, X_test, feature_names=None):
    """Pairwise visualization for multivariate GP predictions (uses seaborn).
    
    Plots predictive mean vs. each input feature, holding others at their
    median. Useful for understanding feature effects in higher-dim models.
    
    Parameters
    ----------
    gp : GaussianProcessRegressor
        Fitted model.
    X_test : array-like of shape (n, d)
        Reference points (we use their range and median).
    feature_names : list of str, optional
        Names for each feature. Defaults to ['x1', 'x2', ...].
    
    Returns
    -------
    fig, axes
    """
    import seaborn as sns
    
    X_test = np.atleast_2d(np.asarray(X_test))
    n_features = X_test.shape[1]
    medians = np.median(X_test, axis=0)
    
    if feature_names is None:
        if hasattr(gp, "feature_names_") and gp.feature_names_ is not None:
            feature_names = gp.feature_names_
        else:
            feature_names = [f"x{i+1}" for i in range(n_features)]
    
    fig, axes = plt.subplots(1, n_features, figsize=(4 * n_features, 4),
                             squeeze=False)
    axes = axes[0]
    
    for i, ax in enumerate(axes):
        # Vary feature i over its range; hold others at median
        x_grid = np.linspace(X_test[:, i].min(), X_test[:, i].max(), 100)
        X_grid = np.tile(medians, (100, 1))
        X_grid[:, i] = x_grid
        
        y_mean, y_std = gp.predict(X_grid, return_std=True)
        
        ax.fill_between(x_grid, y_mean - 2 * y_std, y_mean + 2 * y_std,
                        alpha=0.25, color="C0")
        ax.plot(x_grid, y_mean, color="C0", linewidth=2)
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel("Predicted y")
        ax.set_title(f"Effect of {feature_names[i]}\n(others at median)")
        ax.grid(alpha=0.3)
    
    fig.tight_layout()
    return fig, axes
