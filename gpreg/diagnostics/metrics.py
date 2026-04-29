"""Diagnostic metrics for GP regression.

Implements three core diagnostics:
- RMSE: point prediction accuracy
- NLPD: probabilistic accuracy (uses predictive distribution)
- LOO-CV: leave-one-out cross-validation (closed-form for GPs)
"""

import numpy as np
from scipy.stats import norm

from ..utils.linalg import cholesky_solve
from ..utils.exceptions import NotFittedError


def rmse(y_true, y_pred):
    """Root mean squared error.
    
    Standard regression metric for point predictions. Lower is better.
    Uses the predictive mean only; it does not assess uncertainty quality.
    
    Parameters
    ----------
    y_true : array-like of shape (n,)
        True target values.
    y_pred : array-like of shape (n,)
        Predicted means.
    
    Returns
    -------
    float
        RMSE value (in the same units as y).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def nlpd(y_true, y_mean, y_std):
    """Negative log predictive density.
    
    Evaluates the full predictive distribution against the true values.
    Penalizes both inaccurate means AND poorly calibrated uncertainty:
    a confidently-wrong prediction is punished more than an uncertain
    wrong one.
    
    Lower is better. Reported per data point (averaged), so values are
    comparable across datasets of different sizes.
    
    Parameters
    ----------
    y_true : array-like of shape (n,)
        True target values.
    y_mean : array-like of shape (n,)
        Predictive means.
    y_std : array-like of shape (n,)
        Predictive standard deviations.
    
    Returns
    -------
    float
        Average NLPD per data point.
    """
    y_true = np.asarray(y_true).ravel()
    y_mean = np.asarray(y_mean).ravel()
    y_std = np.asarray(y_std).ravel()
    
    if np.any(y_std <= 0):
        raise ValueError("All predictive standard deviations must be positive.")
    
    log_pdfs = norm.logpdf(y_true, loc=y_mean, scale=y_std)
    return float(-np.mean(log_pdfs))


def loo_cv(gp_model):
    """Leave-one-out cross-validation for a fitted GP.
    
    Uses the closed-form solution from Rasmussen & Williams (Eq. 5.12),
    which avoids actually refitting the model n times. The cost is
    O(n^2) given the already-computed Cholesky factor, vs. O(n^4) for
    naive LOO.
    
    For each training point i, we get the predictive mean and variance
    that we WOULD have obtained had we left point i out. From those,
    we compute the LOO-RMSE and LOO-NLPD.
    
    Parameters
    ----------
    gp_model : GaussianProcessRegressor
        A fitted GP model.
    
    Returns
    -------
    dict
        Dict with keys:
        - 'loo_means': (n,) predictive means with each point left out
        - 'loo_vars': (n,) predictive variances with each point left out
        - 'loo_rmse': float, RMSE of LOO predictions
        - 'loo_nlpd': float, NLPD of LOO predictions
    """
    if not gp_model._fitted:
        raise NotFittedError("LOO-CV requires a fitted model.")
    
    L = gp_model.L_
    alpha = gp_model.alpha_
    n = len(alpha)
    
    # Compute K^-1 from the Cholesky factor.
    # We need only the diagonal in principle, but having the full inverse
    # makes the formulas clearer and we already pay O(n^2) memory.
    K_inv = cholesky_solve(L, np.eye(n))
    
    # R&W Eq. 5.12:
    # mu_i = y_i - alpha_i / [K^-1]_{ii}
    # var_i = 1 / [K^-1]_{ii}
    # The "predicted" value at point i (with i held out) is the training
    # value minus the residual that the inverse would have to absorb.
    K_inv_diag = np.diag(K_inv)
    loo_var = 1.0 / K_inv_diag
    loo_residual = alpha / K_inv_diag  # = y_i - mu_loo_i (in centered space)
    
    # Reconstruct LOO means in original (uncentered) y-space
    y_train_uncentered = gp_model.y_train_ + gp_model.y_mean_
    loo_means = y_train_uncentered - loo_residual
    
    loo_std = np.sqrt(loo_var)
    
    return {
        "loo_means": loo_means,
        "loo_vars": loo_var,
        "loo_rmse": rmse(y_train_uncentered, loo_means),
        "loo_nlpd": nlpd(y_train_uncentered, loo_means, loo_std),
    }
