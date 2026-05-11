"""
Empirical Natural Gradient Descent (ENGD) for Black-Scholes PINNs.

Port / re-implementation of the JAX version from:
    Zeinhofer, M. et al.  "Natural Gradient PINNs", ICML 2023.
    https://github.com/MariusZeinhofer/Natural-Gradient-PINNs-ICML23

Mathematical background
-----------------------
For a PINN with Black-Scholes PDE residual $F[u_\\theta](s,t)$ and terminal
condition residual $u_\\theta(s,T) - \\varphi(s)$, the Gram matrix is

    $G(\\theta) = \\frac{\\lambda_F}{N_F} \\sum_i J_F(x_i) J_F(x_i)^\\top
               + \\frac{\\lambda_{TC}}{N_{TC}} \\sum_j J_{TC}(x_j) J_{TC}(x_j)^\\top
               + \\varepsilon I$

where $J_F(x) = \\partial_\\theta F[u_\\theta](x)$ and
$J_{TC}(x) = \\partial_\\theta u_\\theta(x)$ are the Jacobians of the
measurement functions w.r.t. the network parameters $\\theta$.

The natural gradient direction $\\delta$ satisfies $G(\\theta)\\delta = \\nabla_\\theta L$
and the parameters are updated as $\\theta \\leftarrow \\theta - \\alpha\\delta$,
with $\\alpha$ chosen by grid line search.

Implementation notes
--------------------
* ``torch.func.jacrev`` + ``vmap`` compute per-sample Jacobians efficiently.
* The functional BSM residual ``_bsm_scalar`` uses ``torch.func.grad``
  (not ``torch.autograd.grad``) so that it composes correctly with ``jacrev``.
* Solving $G\\delta = g$ uses Conjugate Gradient without forming $G$
  explicitly, avoiding the $O(n_\\text{params}^2)$ memory cost.
* ``n_gram`` interior Gram points (typically 64–256) provide a good
  approximation at a fraction of the full-collocation cost.

Limitations
-----------
* Only the standard model ``forward()`` is used for the Gram Jacobian.
  ``BermudaETCNN.forward_pde()`` (operator bypass) is not yet wired in.
* Models with stochastic layers (Dropout, etc.) should be put in ``eval()``
  before calling ``ENGDOptimizer.step()``.
"""
from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
from torch.func import functional_call
from torch.func import grad as func_grad
from torch.func import jacfwd, jacrev, vmap


# ---------------------------------------------------------------------------
# Parameter vector helpers
# ---------------------------------------------------------------------------

def flat_params(model: nn.Module) -> torch.Tensor:
    """Concatenated flat copy of all model parameters (detached)."""
    return torch.cat([p.data.flatten() for p in model.parameters()])


def set_flat_params(model: nn.Module, flat: torch.Tensor) -> None:
    """Write a flat parameter vector back into the model in-place."""
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[offset : offset + n].reshape(p.shape))
        offset += n


def flat_grad(model: nn.Module) -> torch.Tensor:
    """Concatenated flat gradient of all parameters (zeros if grad is None)."""
    device = next(model.parameters()).device
    return torch.cat(
        [
            p.grad.flatten()
            if p.grad is not None
            else torch.zeros(p.numel(), device=device)
            for p in model.parameters()
        ]
    )


# ---------------------------------------------------------------------------
# Functional BSM residual — composable with torch.func transforms
# ---------------------------------------------------------------------------

def _bsm_scalar(
    params_dict: dict,
    model: nn.Module,
    s: torch.Tensor,
    t: torch.Tensor,
    r: float,
    q: float,
    sigma: float,
) -> torch.Tensor:
    """$F[u_\\theta](s,t)$ at a single SCALAR point $(s,t)$.

    Uses ``torch.func.grad`` (not ``torch.autograd.grad``) so that
    it composes correctly with ``jacrev`` and ``vmap``.

    Parameters
    ----------
    params_dict:
        Current model parameters as a ``{name: tensor}`` dict
        (should be detached clones for correct gradient isolation).
    model:
        ``nn.Module`` — architecture only; weights come from ``params_dict``.
    s, t:
        0-dim tensors — a single (asset-price, time) point.
    r, q, sigma:
        Black-Scholes parameters.
    """

    def u_fn(s_: torch.Tensor, t_: torch.Tensor) -> torch.Tensor:
        x = torch.stack([s_, t_]).unsqueeze(0)  # (1, 2)
        return functional_call(model, params_dict, x).squeeze()

    v = u_fn(s, t)
    dv_dt = func_grad(lambda t_: u_fn(s, t_))(t)
    dv_ds = func_grad(lambda s_: u_fn(s_, t))(s)
    d2v_ds2 = func_grad(func_grad(lambda s_: u_fn(s_, t)))(s)

    return dv_dt + 0.5 * sigma**2 * s**2 * d2v_ds2 + (r - q) * s * dv_ds - r * v


def _tc_scalar(
    params_dict: dict,
    model: nn.Module,
    s: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """$u_\\theta(s,t)$ at a single scalar point — measurement for terminal Gram."""
    x = torch.stack([s, t]).unsqueeze(0)  # (1, 2)
    return functional_call(model, params_dict, x).squeeze()


# ---------------------------------------------------------------------------
# Vectorised per-sample Jacobians
# ---------------------------------------------------------------------------

def compute_jacobians(
    model: nn.Module,
    s_f: torch.Tensor,
    t_f: torch.Tensor,
    s_tc: torch.Tensor,
    t_tc: torch.Tensor,
    r: float,
    q: float,
    sigma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute flat per-sample Jacobians for the Gram matrix.

    Parameters
    ----------
    model:
        The PINN model (standard ``forward()`` is used).
    s_f, t_f:
        Interior Gram points, shape ``(N_gram,)``.
    s_tc, t_tc:
        Terminal Gram points, shape ``(N_gram_tc,)``.
    r, q, sigma:
        Black-Scholes parameters.

    Returns
    -------
    J_F:
        Tensor of shape ``(N_gram, n_params)`` —
        $\\partial_\\theta F[u_\\theta](x_i)$ for each interior point.
    J_TC:
        Tensor of shape ``(N_gram_tc, n_params)`` —
        $\\partial_\\theta u_\\theta(x_j)$ for each terminal point.
    """
    # Freeze a snapshot of current parameters — detached so jacrev
    # differentiates w.r.t. these values, not the live model buffers.
    params_dict = {k: v.detach().clone() for k, v in model.named_parameters()}

    # -- PDE Jacobian (interior) --
    def jac_F_flat(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        jac = jacrev(_bsm_scalar, argnums=0)(params_dict, model, s, t, r, q, sigma)
        return torch.cat([v.flatten() for v in jac.values()])

    J_F = vmap(jac_F_flat)(s_f.detach(), t_f.detach())

    # -- Terminal Jacobian --
    def jac_TC_flat(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        jac = jacrev(_tc_scalar, argnums=0)(params_dict, model, s, t)
        return torch.cat([v.flatten() for v in jac.values()])

    J_TC = vmap(jac_TC_flat)(s_tc.detach(), t_tc.detach())

    return J_F, J_TC


# ---------------------------------------------------------------------------
# Generic Jacobian builder — for arbitrary measurement functions
# ---------------------------------------------------------------------------


def measurement_jacobian(
    measurement_fn: Callable,
    params_dict: dict,
    *args,
) -> torch.Tensor:
    """Compute the Jacobian of a *vector-valued* measurement w.r.t. parameters.

    Use this when your measurement function returns a *vector* of residuals
    (e.g. weak-form residuals $R_{i,k}$ in a VPINN).  For *scalar* per-point
    measurements (standard PINN), use :func:`compute_jacobians` which adds
    a ``vmap`` over the batch dimension.

    Parameters
    ----------
    measurement_fn:
        Callable ``(params_dict, *args) -> Tensor`` of shape ``(M,)``.
        Must be composable with ``torch.func.jacrev`` (i.e., use
        ``functional_call`` / ``torch.func.grad`` rather than the live
        model and ``torch.autograd.grad``).
    params_dict:
        Detached parameter snapshot — the variable jacrev differentiates against.
    *args:
        Additional fixed arguments forwarded to ``measurement_fn``.

    Returns
    -------
    J : Tensor of shape ``(M, n_params)`` — Jacobian rows are flattened over
        the parameter pytree (consistent with ``flat_grad``/``flat_params``).
    """
    jac_pytree = jacrev(measurement_fn, argnums=0)(params_dict, *args)
    # jac_pytree[name] has shape (M, *params_dict[name].shape)
    return torch.cat([j.flatten(start_dim=1) for j in jac_pytree.values()], dim=1)


def measurement_jacobian_fwd(
    measurement_fn: Callable,
    params_dict: dict,
    *args,
) -> torch.Tensor:
    """Same as :func:`measurement_jacobian` but uses **forward-mode** AD (``jacfwd``).

    Prefer this when ``n_params < n_measurements`` (the regime of the ICML 2023
    paper).  ``jacfwd`` does ``n_params`` JVPs instead of ``n_measurements`` VJPs,
    which is cheaper when the Jacobian matrix is tall (M ≫ n).

    Parameters / Returns: identical to :func:`measurement_jacobian`.
    """
    jac_pytree = jacfwd(measurement_fn, argnums=0)(params_dict, *args)
    return torch.cat([j.flatten(start_dim=1) for j in jac_pytree.values()], dim=1)


# ---------------------------------------------------------------------------
# Conjugate Gradient solver — implicit Gram-vector products
# ---------------------------------------------------------------------------

def _gram_matvec(
    v: torch.Tensor,
    J_F: torch.Tensor,
    J_TC: torch.Tensor,
    lam_f: float,
    lam_tc: float,
    reg: float,
) -> torch.Tensor:
    """$G v$ without forming $G$ explicitly.

    $G = \\frac{\\lambda_F}{N_F} J_F^\\top J_F
       + \\frac{\\lambda_{TC}}{N_{TC}} J_{TC}^\\top J_{TC}
       + \\varepsilon I$

    Empty Jacobians (``shape[0] == 0``) and zero weights are handled
    gracefully so that, e.g., an ETCNN with hard-enforced terminal BC
    can be used with ``lam_tc = 0``.
    """
    Gv = reg * v
    if J_F.shape[0] > 0 and lam_f != 0.0:
        Gv = Gv + (lam_f / J_F.shape[0]) * (J_F.T @ (J_F @ v))
    if J_TC.shape[0] > 0 and lam_tc != 0.0:
        Gv = Gv + (lam_tc / J_TC.shape[0]) * (J_TC.T @ (J_TC @ v))
    return Gv


def solve_cg(
    g: torch.Tensor,
    J_F: torch.Tensor,
    J_TC: torch.Tensor,
    lam_f: float,
    lam_tc: float,
    reg: float,
    n_iters: int = 50,
    tol: float = 1e-7,
) -> torch.Tensor:
    """Solve $G\\delta = g$ by Conjugate Gradient.

    The Gram matrix $G$ is never formed explicitly; only matrix-vector
    products $J_F^\\top (J_F v)$ etc. are evaluated.

    Parameters
    ----------
    g:
        Right-hand side (standard flat gradient $\\nabla_\\theta L$),
        shape ``(n_params,)``.
    J_F, J_TC:
        Per-sample Jacobian matrices from :func:`compute_jacobians`.
    lam_f, lam_tc:
        Loss weights — must match those used in the PINN loss.
    reg:
        Tikhonov regularisation $\\varepsilon > 0$.
    n_iters:
        Maximum CG iterations.
    tol:
        Stop when $\\|r\\|_2 < $ ``tol``.

    Returns
    -------
    delta:
        Natural gradient direction, shape ``(n_params,)``.
    """

    def Av(v: torch.Tensor) -> torch.Tensor:
        return _gram_matvec(v, J_F, J_TC, lam_f, lam_tc, reg)

    x = torch.zeros_like(g)
    r = g - Av(x)
    p = r.clone()
    rsold = torch.dot(r, r)

    for _ in range(n_iters):
        Ap = Av(p)
        denom = torch.dot(p, Ap)
        # Defensive: if denom <= 0 then G is not SPD (numerical noise from
        # very small reg, or a Jacobian with collinear rows). Bail out
        # with the current iterate; caller can either accept the partial
        # solution or increase ``reg``.
        if denom <= 0.0 or not torch.isfinite(denom):
            break
        alpha = rsold / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.dot(r, r)
        if rsnew.sqrt() < tol:
            break
        beta = rsnew / rsold.clamp(min=1e-30)
        p = r + beta * p
        rsold = rsnew

    return x


# ---------------------------------------------------------------------------
# Grid line search
# ---------------------------------------------------------------------------

def grid_line_search(
    model: nn.Module,
    loss_fn: Callable[[], torch.Tensor],
    nat_grad: torch.Tensor,
    n_steps: int = 30,
    step_max: float = 1.0,
) -> float:
    """Search $\\alpha \\in \\{\\alpha_0 \\cdot 2^{-k}\\}$ for the best step.

    Evaluates ``loss_fn()`` at each candidate step
    $\\theta \\leftarrow \\theta - \\alpha\\delta$ without backpropagating,
    then restores the original parameters.

    Parameters
    ----------
    model:
        The PINN model.
    loss_fn:
        Callable that returns the total PINN loss scalar (no ``.backward()``
        needed). Should use the same collocation batch as the current step.
    nat_grad:
        Natural gradient direction $\\delta$, shape ``(n_params,)``.
    n_steps:
        Number of step sizes to evaluate.
    step_max:
        Largest step size $\\alpha_0$.

    Returns
    -------
    best_step : float
    """
    steps = step_max * (
        0.5 ** torch.arange(n_steps, dtype=nat_grad.dtype, device=nat_grad.device)
    )
    flat0 = flat_params(model).clone()

    best_loss = float("inf")
    best_step = steps[-1].item()

    prev_training = model.training
    model.eval()
    for alpha in steps:
        set_flat_params(model, flat0 - alpha * nat_grad)
        # enable_grad is required because loss_fn may call bsm_operator,
        # which uses torch.autograd.grad internally w.r.t. collocation inputs.
        with torch.enable_grad():
            loss_val = loss_fn().item()
        model.zero_grad()  # discard any parameter gradients built above
        # Skip NaN / Inf candidates (can occur for very large step sizes).
        if not math.isfinite(loss_val):
            continue
        if loss_val < best_loss:
            best_loss = loss_val
            best_step = alpha.item()

    set_flat_params(model, flat0)
    model.train(prev_training)
    return best_step


# ---------------------------------------------------------------------------
# High-level optimizer
# ---------------------------------------------------------------------------


class ENGDOptimizer:
    """Empirical Natural Gradient Descent (ENGD) for Black-Scholes PINNs.

    Computes per-sample Jacobians of the PDE residual and terminal condition,
    assembles the Gram matrix implicitly, solves for the natural gradient via
    Conjugate Gradient, and applies a grid line search for the step size.

    Parameters
    ----------
    model:
        The PINN model (any ``nn.Module`` compatible with ``functional_call``).
    r, q, sigma:
        Black-Scholes parameters — must match those used in the loss.
    lam_f:
        PDE loss weight — must match the training loss.
    lam_tc:
        Terminal condition loss weight. Set to ``0.0`` for models with
        hard-enforced terminal conditions (e.g. ETCNN) since the terminal
        Gram vanishes by construction.
    reg:
        Tikhonov regularisation $\\varepsilon$ added to the Gram diagonal.
        Larger values improve conditioning but bias the natural gradient.
    cg_iters:
        Maximum Conjugate Gradient iterations.
    ls_steps:
        Number of step sizes in the grid line search.
    ls_step_max:
        Largest step size $\\alpha_0$ in the line search.

    Notes
    -----
    ``BermudaETCNN.forward_pde()`` (operator bypass) is not yet supported;
    the standard ``forward()`` is used for the Gram Jacobian computation.

    Example
    -------
    See ``experiments/python_scripts/exp1/phase3_engd.py`` for a complete
    training loop.
    """

    def __init__(
        self,
        model: nn.Module,
        r: float,
        q: float,
        sigma: float,
        lam_f: float = 1.0,
        lam_tc: float = 1.0,
        reg: float = 1e-4,
        cg_iters: int = 50,
        ls_steps: int = 30,
        ls_step_max: float = 1.0,
    ) -> None:
        self.model = model
        self.r = r
        self.q = q
        self.sigma = sigma
        self.lam_f = lam_f
        self.lam_tc = lam_tc
        self.reg = reg
        self.cg_iters = cg_iters
        self.ls_steps = ls_steps
        self.ls_step_max = ls_step_max

    def step(
        self,
        g: torch.Tensor,
        s_gram: torch.Tensor,
        t_gram: torch.Tensor,
        s_tc_gram: torch.Tensor,
        t_tc_gram: torch.Tensor,
        loss_fn: Callable[[], torch.Tensor],
    ) -> dict:
        """One ENGD update: compute natural gradient and apply with line search.

        Parameters
        ----------
        g:
            Flat standard gradient $\\nabla_\\theta L$ of shape ``(n_params,)``.
            Obtain via ``loss.backward(); g = flat_grad(model)``.
        s_gram, t_gram:
            Interior Gram points, shape ``(N_gram,)``.
            A subset of the current collocation batch is recommended
            (e.g. ``s_gram = s_f[:n_gram].detach()``).
        s_tc_gram, t_tc_gram:
            Terminal Gram points, shape ``(N_tc_gram,)``.
        loss_fn:
            Callable ``() -> scalar`` returning the total PINN loss without
            backpropagation, evaluated at the *current collocation batch*.
            Used only during the line search.

        Returns
        -------
        dict with keys:

        ``'step_size'``
            The step size $\\alpha$ chosen by line search.
        ``'cg_residual_norm'``
            $\\|G\\delta - g\\|_2$ — CG convergence diagnostic.
        ``'J_F_norm'``
            Frobenius norm of the PDE Jacobian (scale indicator).
        """
        J_F, J_TC = compute_jacobians(
            self.model,
            s_gram,
            t_gram,
            s_tc_gram,
            t_tc_gram,
            self.r,
            self.q,
            self.sigma,
        )

        delta = solve_cg(
            g,
            J_F,
            J_TC,
            self.lam_f,
            self.lam_tc,
            self.reg,
            self.cg_iters,
        )

        step_size = grid_line_search(
            self.model,
            loss_fn,
            delta,
            n_steps=self.ls_steps,
            step_max=self.ls_step_max,
        )

        # Apply the accepted step
        flat0 = flat_params(self.model)
        set_flat_params(self.model, flat0 - step_size * delta)

        # Diagnostics
        Gdelta = _gram_matvec(delta, J_F, J_TC, self.lam_f, self.lam_tc, self.reg)
        cg_res = (g - Gdelta).norm().item()

        return {
            "step_size": step_size,
            "cg_residual_norm": cg_res,
            "J_F_norm": J_F.norm().item(),
        }
