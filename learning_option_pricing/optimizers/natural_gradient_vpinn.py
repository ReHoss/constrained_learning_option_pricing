"""
Empirical Natural Gradient Descent (ENGD) for *Variational* PINNs.

The standard PINN ENGD treats the **pointwise** PDE residual as the
"measurement" function $f_x(\\theta) = F[u_\\theta](x)$ and assembles the
Gram matrix from per-sample Jacobians at quadrature points $x_i$.

In a VPINN, the residual is **integrated against test functions**
$\\phi_k$ (Galerkin / weak form).  After integration by parts the
weak residual at time $\\tau_i$ is

$$
R_{i,k}(\\theta) \\;=\\; \\int_{-x_{\\max}}^{x_{\\max}}
\\Big[ \\partial_\\tau u_\\theta\\,\\phi_k
+ \\tfrac{\\sigma^2}{2}\\,\\partial_x u_\\theta\\,\\phi'_k
- \\mu\\,\\partial_x u_\\theta\\,\\phi_k
+ r\\,u_\\theta\\,\\phi_k \\Big]\\,\\mathrm{d}x,
$$

where the integrals are evaluated by Gauss-Legendre quadrature.  The
*natural* set of measurements is therefore the matrix
$\\{R_{i,k}(\\theta)\\}_{i=1..N_\\tau,\\,k=1..K}$, whose Jacobian
$J_R \\in \\mathbb{R}^{(N_\\tau K) \\times n_\\theta}$ is what enters the Gram:

$$
G(\\theta) = \\frac{1}{N_\\tau K}\\,J_R^\\top J_R + \\varepsilon I.
$$

The CG solver and line search are unchanged from
:mod:`learning_option_pricing.optimizers.natural_gradient`.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torch.func import functional_call, grad as func_grad

from .natural_gradient import (
    flat_grad,
    flat_params,
    grid_line_search,
    measurement_jacobian,
    set_flat_params,
    solve_cg,
)


# ---------------------------------------------------------------------------
# Functional weak residual — composable with torch.func.jacrev
# ---------------------------------------------------------------------------


def _vpinn_residuals(
    params_dict: dict,
    model: nn.Module,
    tau_batch: torch.Tensor,
    x_nodes: torch.Tensor,
    phi_w: torch.Tensor,
    dphi_w: torch.Tensor,
    sigma: float,
    mu: float,
    r: float,
) -> torch.Tensor:
    """Vector of weak residuals ``R[i, k]`` flattened to shape ``(N_tau * K,)``.

    This is the functional re-implementation of
    :meth:`learning_option_pricing.vpinn.loss.VPINNLoss.forward` that uses
    ``torch.func.grad`` (instead of ``torch.autograd.grad``) so that the
    function composes with :func:`torch.func.jacrev`.

    Parameters
    ----------
    params_dict : dict
        ``{name: Tensor}`` snapshot of the parameters.  ``jacrev`` will
        differentiate w.r.t. this dict.
    model : nn.Module
        Architecture (weights are pulled from ``params_dict`` via
        ``functional_call``).
    tau_batch : Tensor (N_tau,)
        Time collocation points $\\tau_i \\in (0, T]$.
    x_nodes : Tensor (N_q,)
        Gauss-Legendre quadrature nodes in the spatial domain.
    phi_w, dphi_w : Tensor (K, N_q)
        Pre-weighted test-function values $\\phi_k(x_j) w_j$ and
        derivatives $\\phi_k'(x_j) w_j$.
    sigma, mu, r : float
        BSM parameters in log-moneyness coordinates.

    Returns
    -------
    R_flat : Tensor (N_tau * K,)
        Flattened residuals $R_{i,k}$ in row-major order
        (i.e. ``R_flat[i*K + k]``).
    """
    # u_fn(tau, x) -> scalar  (auto-differentiable via torch.func.grad)
    def u_fn(tau: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        inp = torch.stack([tau, x]).unsqueeze(0)  # (1, 2)
        return functional_call(model, params_dict, inp).squeeze()

    # Vectorised evaluation: u, du/dtau, du/dx at all (tau_i, x_j)
    # Shapes: (N_tau, N_q)
    def _u_at(tau, x):
        return u_fn(tau, x)

    def _du_dtau_at(tau, x):
        return func_grad(lambda t: u_fn(t, x))(tau)

    def _du_dx_at(tau, x):
        return func_grad(lambda y: u_fn(tau, y))(x)

    # We avoid an O(N_tau * N_q) double loop by using torch.vmap-like nesting.
    # torch.func.vmap composes; we map over tau (outer) then x (inner).
    from torch.func import vmap  # local import to keep top-level light

    u_vals = vmap(vmap(_u_at, in_dims=(None, 0)), in_dims=(0, None))(tau_batch, x_nodes)
    u_tau = vmap(vmap(_du_dtau_at, in_dims=(None, 0)), in_dims=(0, None))(tau_batch, x_nodes)
    u_x = vmap(vmap(_du_dx_at, in_dims=(None, 0)), in_dims=(0, None))(tau_batch, x_nodes)

    # Integrand coefficients (N_tau, N_q)
    f_phi = u_tau - mu * u_x + r * u_vals
    f_dphi = (sigma**2 / 2.0) * u_x

    # R[i, k] = sum_j (f_phi[i, j] * phi_w[k, j] + f_dphi[i, j] * dphi_w[k, j])
    R = f_phi @ phi_w.T + f_dphi @ dphi_w.T  # (N_tau, K)
    return R.reshape(-1)  # (N_tau * K,)


# ---------------------------------------------------------------------------
# High-level optimizer
# ---------------------------------------------------------------------------


class VPINNENGDOptimizer:
    """Empirical Natural Gradient Descent for VPINNs.

    The Gram matrix uses the Jacobian of the *weak* residuals
    $R_{i,k}(\\theta)$ rather than the pointwise residuals, matching the
    geometry of the VPINN loss
    $\\mathcal{L} = \\operatorname{mean}_{i,k} R_{i,k}^2$.

    Parameters
    ----------
    model : nn.Module
        $u_\\theta(\\tau, x)$ (input columns: $\\tau$ then $x$).
    vpinn_loss :
        An instance of
        :class:`learning_option_pricing.vpinn.loss.VPINNLoss` — supplies
        the precomputed quadrature nodes and weighted test functions.
    reg : float
        Tikhonov regularisation $\\varepsilon$.
    cg_iters, ls_steps, ls_step_max :
        Same role as in
        :class:`learning_option_pricing.optimizers.natural_gradient.ENGDOptimizer`.

    Notes
    -----
    The boundary contributions are absorbed by integration by parts
    (the test functions $\\phi_k$ vanish at $\\pm x_{\\max}$), so there
    is no separate "terminal Gram" term — the entire Gram is built from
    $J_R$ alone.

    Initial conditions (terminal payoff) must be enforced *outside* this
    optimizer, e.g. via an exact-BC ansatz (ETCNN) or an additional MSE
    term in a separate optimizer step.
    """

    def __init__(
        self,
        model: nn.Module,
        vpinn_loss,
        reg: float = 1e-4,
        cg_iters: int = 50,
        ls_steps: int = 30,
        ls_step_max: float = 1.0,
    ) -> None:
        self.model = model
        self.vpinn_loss = vpinn_loss
        self.reg = reg
        self.cg_iters = cg_iters
        self.ls_steps = ls_steps
        self.ls_step_max = ls_step_max

    def _compute_jacobian(self, tau_gram: torch.Tensor) -> torch.Tensor:
        """Build $J_R \\in \\mathbb{R}^{(N_\\tau K)\\times n_\\theta}$ for the
        current parameter values, evaluated at ``tau_gram``."""
        params_dict = {k: v.detach().clone() for k, v in self.model.named_parameters()}
        return measurement_jacobian(
            _vpinn_residuals,
            params_dict,
            self.model,
            tau_gram.detach(),
            self.vpinn_loss.x_nodes,
            self.vpinn_loss.phi_w,
            self.vpinn_loss.dphi_w,
            self.vpinn_loss.sigma,
            self.vpinn_loss.mu,
            self.vpinn_loss.r,
        )

    @staticmethod
    def _gram_matvec(v: torch.Tensor, J: torch.Tensor, reg: float) -> torch.Tensor:
        return (1.0 / J.shape[0]) * (J.T @ (J @ v)) + reg * v

    def _solve(self, g: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
        """Solve $G\\delta = g$ via CG without forming $G$."""
        # Reuse the generic CG solver: pass J as J_F, an empty J_TC,
        # lam_f=1, lam_tc=0.
        empty_TC = torch.zeros((0, J.shape[1]), dtype=J.dtype, device=J.device)
        return solve_cg(
            g, J, empty_TC,
            lam_f=1.0, lam_tc=0.0, reg=self.reg,
            n_iters=self.cg_iters,
        )

    def step(
        self,
        g: torch.Tensor,
        tau_gram: torch.Tensor,
        loss_fn: Callable[[], torch.Tensor],
    ) -> dict:
        """One ENGD step: build $J_R$, solve $G\\delta = g$, line search, update.

        Parameters
        ----------
        g : Tensor (n_params,)
            Standard flat gradient $\\nabla_\\theta L$ of the VPINN loss
            (after ``loss.backward()``).
        tau_gram : Tensor (N_tau_gram,)
            Time-collocation points used to build the Gram matrix.  These
            should be deterministic / fixed across iterations.
        loss_fn : Callable
            Closure ``() -> scalar`` returning the VPINN loss for the
            *current parameter setting* (no backprop).  Used during line
            search.

        Returns
        -------
        dict with keys ``step_size``, ``cg_residual_norm``, ``J_norm``.
        """
        J = self._compute_jacobian(tau_gram)
        delta = self._solve(g, J)

        step_size = grid_line_search(
            self.model, loss_fn, delta,
            n_steps=self.ls_steps, step_max=self.ls_step_max,
        )

        flat0 = flat_params(self.model)
        set_flat_params(self.model, flat0 - step_size * delta)

        Gdelta = self._gram_matvec(delta, J, self.reg)
        cg_res = (g - Gdelta).norm().item()
        return {
            "step_size": step_size,
            "cg_residual_norm": cg_res,
            "J_norm": J.norm().item(),
        }


# ---------------------------------------------------------------------------
# Convenience: build a Jacobian for inspection / unit tests
# ---------------------------------------------------------------------------


def vpinn_jacobian(
    model: nn.Module,
    vpinn_loss,
    tau_batch: torch.Tensor,
) -> torch.Tensor:
    """Standalone Jacobian builder — same code path as
    :meth:`VPINNENGDOptimizer._compute_jacobian` but exposed as a free
    function for tests and diagnostics."""
    params_dict = {k: v.detach().clone() for k, v in model.named_parameters()}
    return measurement_jacobian(
        _vpinn_residuals,
        params_dict,
        model,
        tau_batch.detach(),
        vpinn_loss.x_nodes,
        vpinn_loss.phi_w,
        vpinn_loss.dphi_w,
        vpinn_loss.sigma,
        vpinn_loss.mu,
        vpinn_loss.r,
    )


# Re-export for convenience
__all__ = [
    "VPINNENGDOptimizer",
    "vpinn_jacobian",
    "flat_grad",
    "flat_params",
    "set_flat_params",
]
