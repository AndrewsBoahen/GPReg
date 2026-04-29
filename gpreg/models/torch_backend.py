"""PyTorch-based hyperparameter optimization.

An alternative to the SciPy L-BFGS-B optimizer used by default in
GaussianProcessRegressor.fit(). Uses PyTorch's autograd to compute
gradients of the marginal log-likelihood automatically, avoiding the
need to derive them by hand or rely on finite-difference approximations.

This is the same approach GPyTorch takes; autograd makes it trivial
to support arbitrary new kernel compositions. To use:

    gp = GaussianProcessRegressor(kernel=..., optimizer="pytorch")
    gp.fit(X, y)

Performance note: for small problems (n < 1000) and our standard
kernels, SciPy's analytical L-BFGS-B is typically faster because
PyTorch has more per-iteration overhead. The autograd version becomes
preferable when (a) you've defined a custom kernel that's hard to
differentiate analytically, or (b) you're running on a GPU.

This module is optional  if PyTorch isn't installed, importing it
raises a clean error rather than crashing the whole package.
"""

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from ..utils.exceptions import ConvergenceError


def _check_torch():
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is not installed. Install it with `pip install torch` "
            "to use the autograd-based optimizer, or use the default "
            "SciPy optimizer instead."
        )


def _torch_kernel(kernel, X1, X2, theta):
    """Re-evaluate the kernel using PyTorch tensors for autograd.
    
    This is the trickiest piece: we need a kernel computation that
    differentiates through theta. Rather than rewriting every kernel
    in PyTorch, we mirror the math for the kernels we know about.
    
    Currently supports composites of RBF, Matern (nu in {0.5, 1.5, 2.5}),
    Linear, and White via SumKernel/ProductKernel. Falls back to an
    error for unknown kernels.
    """
    from ..kernels.base import SumKernel, ProductKernel
    from ..kernels.standard import RBF, Matern, Linear, White
    
    # We need an iterator over theta values matching the order in
    # kernel.get_param_vector(). The simplest correct approach: walk
    # the kernel tree in the same order as get_params(), pulling
    # parameters off the front of theta as we go.
    
    def consume_and_eval(k, theta_iter):
        """Evaluate kernel k by consuming the next n_params entries from theta_iter."""
        if isinstance(k, RBF):
            log_ls = next(theta_iter)
            log_sv = next(theta_iter)
            ls = torch.exp(log_ls)
            sv = torch.exp(log_sv)
            # Squared distance with broadcasting: (n, 1, d) - (1, m, d)
            X1_s = X1 / ls
            X2_s = X2 / ls
            sq_dist = ((X1_s.unsqueeze(1) - X2_s.unsqueeze(0)) ** 2).sum(-1)
            return sv * torch.exp(-0.5 * sq_dist)
        
        elif isinstance(k, Matern):
            log_ls = next(theta_iter)
            log_sv = next(theta_iter)
            ls = torch.exp(log_ls)
            sv = torch.exp(log_sv)
            X1_s = X1 / ls
            X2_s = X2 / ls
            sq_dist = ((X1_s.unsqueeze(1) - X2_s.unsqueeze(0)) ** 2).sum(-1)
            # Add tiny floor to avoid sqrt(0) gradient issues at coincident points
            dist = torch.sqrt(sq_dist + 1e-12)
            
            if k.nu == 0.5:
                K = torch.exp(-dist)
            elif k.nu == 1.5:
                sqrt3_d = np.sqrt(3) * dist
                K = (1.0 + sqrt3_d) * torch.exp(-sqrt3_d)
            elif k.nu == 2.5:
                sqrt5_d = np.sqrt(5) * dist
                K = (1.0 + sqrt5_d + (5.0 / 3.0) * sq_dist) * torch.exp(-sqrt5_d)
            else:
                raise ValueError(f"Unsupported Matern nu={k.nu}")
            return sv * K
        
        elif isinstance(k, Linear):
            log_sv = next(theta_iter)
            sv = torch.exp(log_sv)
            return sv * (X1 @ X2.T)
        
        elif isinstance(k, White):
            log_nv = next(theta_iter)
            nv = torch.exp(log_nv)
            # Identity only when X1 IS X2 (same tensor). We check via shape
            # and equality on the underlying data.
            if X1.shape == X2.shape and torch.equal(X1, X2):
                return nv * torch.eye(X1.shape[0], dtype=X1.dtype, device=X1.device)
            else:
                return torch.zeros((X1.shape[0], X2.shape[0]),
                                   dtype=X1.dtype, device=X1.device)
        
        elif isinstance(k, SumKernel):
            return consume_and_eval(k.k1, theta_iter) + consume_and_eval(k.k2, theta_iter)
        
        elif isinstance(k, ProductKernel):
            return consume_and_eval(k.k1, theta_iter) * consume_and_eval(k.k2, theta_iter)
        
        else:
            raise NotImplementedError(
                f"PyTorch optimizer doesn't yet support kernel type {type(k).__name__}. "
                f"Use the default SciPy optimizer."
            )
    
    theta_iter = iter(theta)
    return consume_and_eval(kernel, theta_iter)


def torch_optimize(kernel, X, y, n_iters=200, lr=0.05, n_restarts=3,
                    verbose=False, random_state=None):
    """Optimize kernel hyperparameters using PyTorch autograd.
    
    Maximizes the GP marginal log-likelihood w.r.t. kernel hyperparameters
    (in log-space) using Adam. Adam is more robust to bad initialization
    than L-BFGS-B for this objective, at the cost of needing more
    iterations.
    
    Parameters
    ----------
    kernel : Kernel
        Kernel whose hyperparameters will be optimized in-place.
    X : ndarray of shape (n, d)
        Training inputs.
    y : ndarray of shape (n,)
        Training targets (already centered).
    n_iters : int, default=200
        Number of Adam steps per restart.
    lr : float, default=0.05
        Adam learning rate.
    n_restarts : int, default=3
    verbose : bool, default=False
        Print loss every 50 iterations.
    random_state : int, optional
    
    Returns
    -------
    best_theta : ndarray
        Optimal hyperparameters (in log-space).
    best_loss : float
        Negative log marginal likelihood at the optimum.
    """
    _check_torch()
    
    rng = np.random.default_rng(random_state)
    if random_state is not None:
        torch.manual_seed(random_state)
    
    X_t = torch.as_tensor(X, dtype=torch.float64)
    y_t = torch.as_tensor(y, dtype=torch.float64)
    n = len(y)
    
    def neg_log_marginal(theta_tensor):
        K = _torch_kernel(kernel, X_t, X_t, theta_tensor)
        # Add a small jitter for numerical stability
        K = K + 1e-6 * torch.eye(n, dtype=K.dtype)
        L = torch.linalg.cholesky(K)
        alpha = torch.cholesky_solve(y_t.unsqueeze(-1), L).squeeze(-1)
        log_det = 2.0 * torch.sum(torch.log(torch.diag(L)))
        log_lik = (
            -0.5 * (y_t @ alpha)
            - 0.5 * log_det
            - 0.5 * n * np.log(2 * np.pi)
        )
        return -log_lik
    
    # Run restarts
    starts = [kernel.get_param_vector()]
    for _ in range(n_restarts):
        starts.append(rng.normal(0.0, 1.5, size=kernel.n_params))
    
    best_theta = None
    best_loss = float("inf")
    
    for restart_idx, theta0 in enumerate(starts):
        theta = torch.tensor(theta0, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([theta], lr=lr)
        
        try:
            for it in range(n_iters):
                optimizer.zero_grad()
                loss = neg_log_marginal(theta)
                loss.backward()
                optimizer.step()
                if verbose and (it + 1) % 50 == 0:
                    print(f"  Restart {restart_idx}, iter {it+1}: loss={loss.item():.4f}")
            
            final_loss = loss.item()
            if final_loss < best_loss:
                best_loss = final_loss
                best_theta = theta.detach().numpy().copy()
        except Exception as e:
            if verbose:
                print(f"  Restart {restart_idx} failed: {e}")
            continue
    
    if best_theta is None:
        raise ConvergenceError(
            "All PyTorch optimization restarts failed. Try the SciPy "
            "optimizer (default), or check your data."
        )
    
    return best_theta, best_loss
