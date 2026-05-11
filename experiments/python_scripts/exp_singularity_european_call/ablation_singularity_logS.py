"""Ablation study — European call singularity, coordinates (x=ln(S), t).

Same three-method comparison as ablation_singularity.py but the network input
is x = ln(S) instead of S.  The BSM PDE then has *constant* coefficients:

    F[V] = dV/dt + sigma^2/2 * d^2V/dx^2 + (r - sigma^2/2)*dV/dx - r*V = 0

No S^2 factor — autograd operates on a uniformly-conditioned operator.
The singularity is now at (x = ln(K), t = T) instead of (S = K, t = T).

Usage (from repo root):
    # Smoke test — 200 iters, GPU:
    python3 experiments/python_scripts/exp_singularity_european_call/ablation_singularity_logS.py \\
        --iters 200 --device cuda

    # Full 3-method comparison:
    python3 experiments/python_scripts/exp_singularity_european_call/ablation_singularity_logS.py \\
        --iters 30000 --device cuda

    # Sensitivity to epsilon / beta / IS:
    python3 ... --iters 30000 --device cuda --mode ablation-eps
    python3 ... --iters 30000 --device cuda --mode ablation-beta
    python3 ... --iters 30000 --device cuda --mode ablation-is

    # Regenerate plots without retraining:
    python3 ... --replot data/exp_singularity_european_call/<run_dir>_logS_iters<N>

    # Add a single new variant (e.g. VPINN) to an existing ablation folder:
    python3 ... --add-variant vpinn:data/exp_singularity_european_call/<run_dir>_logS_iters<N> \\
        --device cuda
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp1"))

import phase3_training as p3
from learning_option_pricing.models.etcnn import PINN, InputNormalization
from learning_option_pricing.models.resnet import ResNet
from learning_option_pricing.pricing.terminal import black_scholes_put
from learning_option_pricing.vpinn import GaussLegendreQuadrature
from learning_option_pricing.optimizers import (
    flat_grad, flat_params, set_flat_params,
    grid_line_search, measurement_jacobian, measurement_jacobian_fwd, solve_cg,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameters (same as exp1 / ablation_singularity.py)
# ---------------------------------------------------------------------------
K, r, sigma, T, q = p3.K, p3.r, p3.sigma, p3.T, p3.q

# Domain in x = ln(S) space
S_LO,      S_HI      = 20.0,  160.0
S_EVAL_LO, S_EVAL_HI = 60.0,  140.0
X_LO      = math.log(S_LO)       # ≈ 3.00
X_HI      = math.log(S_HI)       # ≈ 5.08
X_EVAL_LO = math.log(S_EVAL_LO)  # ≈ 4.09
X_EVAL_HI = math.log(S_EVAL_HI)  # ≈ 4.94
X_ATM     = math.log(K)          # ≈ 4.61  — ATM point in log space


# ---------------------------------------------------------------------------
# BSM operator in (x = ln(S), t) coordinates — constant coefficients
# ---------------------------------------------------------------------------

def bsm_operator_logS(
    V: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
    r: float,
    sigma: float,
) -> torch.Tensor:
    """BSM PDE residual with x = ln(S).

    F[V] = dV/dt + sigma^2/2 * d^2V/dx^2 + (r - sigma^2/2)*dV/dx - r*V

    All coefficients are constant — no S^2 factor.
    x and t must be leaf tensors with requires_grad=True.
    """
    (dV_dx,) = torch.autograd.grad(
        V, x, grad_outputs=torch.ones_like(V), create_graph=True
    )
    (dV_dt,) = torch.autograd.grad(
        V, t, grad_outputs=torch.ones_like(V), create_graph=True
    )
    (d2V_dx2,) = torch.autograd.grad(
        dV_dx, x, grad_outputs=torch.ones_like(dV_dx), create_graph=True
    )
    return dV_dt + 0.5 * sigma**2 * d2V_dx2 + (r - 0.5 * sigma**2) * dV_dx - r * V


# ---------------------------------------------------------------------------
# VPINN weak-form loss in (x=ln S, t) forward-time coordinates
# ---------------------------------------------------------------------------

class _VPINNLossForwardLogS(torch.nn.Module):
    r"""Weak-form PDE residual for BSM in $(x=\ln S,\,t)$ forward-time coordinates.

    Starting from the strong-form PDE
        $\partial_t V + \frac{\sigma^2}{2}\partial_{xx}V + \mu\,\partial_x V - rV = 0$
    (with $\mu = r - \sigma^2/2$), multiplying by $\phi_k$ and applying IBP on
    the second-order term (boundary terms vanish because $\phi_k(X_{lo})=\phi_k(X_{hi})=0$):

        $R_{i,k} = \int_{X_{lo}}^{X_{hi}}
            \bigl[\partial_t V\cdot\phi_k
             - \tfrac{\sigma^2}{2}\,\partial_x V\cdot\phi_k'
             + \mu\,\partial_x V\cdot\phi_k
             - r\,V\cdot\phi_k\bigr]\,dx = 0.$

    Test functions: $\phi_k(x)=\sin\!\left(\tfrac{k\pi(x-X_{lo})}{X_{hi}-X_{lo}}\right)$.
    Spatial integral approximated by Gauss-Legendre quadrature.
    Loss: $\mathcal{L}_f = \mathrm{mean}_{i,k}\,R_{i,k}^2$.
    """

    phi_w:        torch.Tensor
    dphi_w:       torch.Tensor
    x_nodes:      torch.Tensor
    weights:      torch.Tensor

    def __init__(
        self,
        sigma: float,
        r: float,
        x_lo: float,
        x_hi: float,
        K_test: int = 20,
        n_quad:  int = 100,
    ):
        super().__init__()
        self.sigma = sigma
        self.r     = r
        self.mu    = r - 0.5 * sigma**2

        quad    = GaussLegendreQuadrature(n_quad, x_lo, x_hi, dtype=torch.float32)
        x_nodes = quad.nodes    # (N_q,)
        weights = quad.weights  # (N_q,)

        # φ_k(x) = sin(k π (x − x_lo) / L),  k = 1 … K_test
        L       = x_hi - x_lo
        k_idx   = torch.arange(1, K_test + 1, dtype=torch.float32).unsqueeze(1)  # (K, 1)
        freq    = k_idx * math.pi / L                                             # (K, 1)
        x_shift = (x_nodes - x_lo).unsqueeze(0)                                  # (1, N_q)
        phi     = torch.sin(freq * x_shift)                                       # (K, N_q)
        dphi    = freq * torch.cos(freq * x_shift)                               # (K, N_q)

        self.domain_length = L
        self.register_buffer("phi_w",   phi  * weights)   # (K, N_q)
        self.register_buffer("dphi_w",  dphi * weights)   # (K, N_q)
        self.register_buffer("x_nodes", x_nodes)          # (N_q,)
        self.register_buffer("weights", weights)           # (N_q,)

    def forward(self, model: torch.nn.Module, t_batch: torch.Tensor) -> torch.Tensor:
        """Compute VPINN PDE residual loss.

        Args:
            model:   Network V(x, t), input shape (..., 2).
            t_batch: (N_t,) collocation time values.
        """
        N_t = t_batch.shape[0]
        N_q = self.x_nodes.shape[0]

        # Flat (N_t × N_q, 2) grid — model convention is (x, t)
        x_rep = self.x_nodes.unsqueeze(0).expand(N_t, N_q).reshape(-1, 1)
        t_rep = t_batch.unsqueeze(1).expand(N_t, N_q).reshape(-1, 1)
        x_rep = x_rep.detach().requires_grad_(True)
        t_rep = t_rep.detach().requires_grad_(True)

        V = model(torch.cat([x_rep, t_rep], dim=1))  # (N_t*N_q, 1)

        dV_dt, dV_dx = torch.autograd.grad(
            V, [t_rep, x_rep],
            grad_outputs=torch.ones_like(V),
            create_graph=True,
        )

        V_vals = V.squeeze(1).reshape(N_t, N_q)
        V_t    = dV_dt.squeeze(1).reshape(N_t, N_q)
        V_x    = dV_dx.squeeze(1).reshape(N_t, N_q)

        # Weak residual integrand (forward-time BSM after IBP)
        f_phi  = V_t + self.mu * V_x - self.r * V_vals   # (N_t, N_q)
        f_dphi = -(self.sigma**2 / 2.0) * V_x             # (N_t, N_q)

        # R_{i,k} = Σ_j [f_phi_{i,j}·φ_w_{k,j} + f_dphi_{i,j}·dφ_w_{k,j}]
        R = f_phi @ self.phi_w.T + f_dphi @ self.dphi_w.T  # (N_t, K)
        return R.pow(2).mean()

    def ic_loss_quad(self, model: torch.nn.Module, payoff_fn) -> torch.Tensor:
        r"""IC loss at t=T via quadrature: $\frac{1}{|\Omega|}\sum_q w_q|u_\theta(T,x_q)-h(x_q)|^2$.

        Uses the same spatial nodes as the PDE residual — consistent with the
        weak L² formulation.  Normalised by domain length so the scale matches
        a Monte-Carlo MSE estimator.
        """
        x_q = self.x_nodes.detach()
        t_T = torch.full_like(x_q, T)
        with torch.no_grad():
            h_q = payoff_fn(x_q)
        u_T = model(torch.stack([x_q, t_T], dim=1)).squeeze()
        return ((u_T - h_q).pow(2) * self.weights).sum() / self.domain_length


# ---------------------------------------------------------------------------
# Loss in log coordinates (replaces p3.compute_losses)
# ---------------------------------------------------------------------------

def compute_losses_logS(model, x_f, t_f, x_tc, t_tc, payoff_fn):
    """PDE + terminal losses in (x=ln(S), t) coordinates."""
    u_f = model(torch.stack([x_f, t_f], dim=1)).squeeze()
    F_u = bsm_operator_logS(u_f, x_f, t_f, r, sigma)
    loss_f = (F_u ** 2).mean()

    with torch.no_grad():
        phi = payoff_fn(x_tc)
    u_tc = model(torch.stack([x_tc, t_tc], dim=1)).squeeze()
    loss_tc = ((u_tc - phi) ** 2).mean()

    total = p3.LAMBDA_F * loss_f + p3.LAMBDA_TC * loss_tc
    return total, loss_f.item(), loss_tc.item()


# ---------------------------------------------------------------------------
# Reference: Black-Scholes call in (x, t) coordinates
# ---------------------------------------------------------------------------

def bs_call_logS(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """C^BS via put-call parity, inputs in (x=ln(S), t) space."""
    tau = (T - t).clamp(min=1e-8)
    S   = x.exp()
    P   = black_scholes_put(S, K, r, sigma, tau)
    return S - K * torch.exp(-r * tau) + P


# ---------------------------------------------------------------------------
# Payoffs in x = ln(S) space
# ---------------------------------------------------------------------------

def payoff_exact_logS(x: torch.Tensor) -> torch.Tensor:
    """(e^x - K)^+  — exact call payoff in log-S space."""
    return torch.clamp(x.exp() - K, min=0.0)


def make_payoff_smooth_logS(beta: float):
    """Softplus approximation of (e^x - K)^+ centred at x=ln(K)."""
    log2 = math.log(2.0)
    def _payoff(x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x.exp() - K, beta=beta) - log2 / beta
    return _payoff


# ---------------------------------------------------------------------------
# Sampler factories in (x, t) space
# ---------------------------------------------------------------------------

def make_sampler_naive_logS(
    n_f: int,
    n_tc: int,
    generator: torch.Generator | None = None,
):
    """Naive uniform sampler for (x, t) collocation and (x, T) terminal points.

    Args:
        n_f: number of PDE collocation points per batch.
        n_tc: number of terminal-condition collocation points per batch.
        generator: explicit ``torch.Generator`` driving the random draws.  When
            ``None``, falls back to the global RNG state — kept only for
            backward compatibility, do not rely on it for new code.  The
            generator must live on the same device as ``p3.DEVICE``.
    """
    def _sample():
        device = p3.DEVICE
        x_f = (torch.rand(n_f, generator=generator, device=device) * (X_HI - X_LO) + X_LO).requires_grad_(True)
        t_f = (torch.rand(n_f, generator=generator, device=device) * T).requires_grad_(True)
        x_tc = torch.rand(n_tc, generator=generator, device=device) * (X_HI - X_LO) + X_LO
        t_tc = torch.full((n_tc,), T, device=device)
        return x_f, t_f, x_tc, t_tc
    return _sample


def make_sampler_truncated_logS(
    n_f: int,
    n_tc: int,
    eps: float,
    generator: torch.Generator | None = None,
):
    """PDE points in t ∈ [0, T-eps]; terminal condition still at t=T exact.

    ``generator`` semantics: see :func:`make_sampler_naive_logS`.
    """
    def _sample():
        device = p3.DEVICE
        x_f = (torch.rand(n_f, generator=generator, device=device) * (X_HI - X_LO) + X_LO).requires_grad_(True)
        t_f = (torch.rand(n_f, generator=generator, device=device) * (T - eps)).requires_grad_(True)
        x_tc = torch.rand(n_tc, generator=generator, device=device) * (X_HI - X_LO) + X_LO
        t_tc = torch.full((n_tc,), T, device=device)
        return x_f, t_f, x_tc, t_tc
    return _sample


def make_sampler_importance_logS(
    n_f: int,
    n_tc: int,
    sigma_is: float,
    mix: float = 0.5,
    eps: float = 0.0,
    generator: torch.Generator | None = None,
):
    """Mix of uniform + Gaussian concentrated around x = ln(K) (ATM).

    ``generator`` semantics: see :func:`make_sampler_naive_logS`.
    """
    def _sample():
        device = p3.DEVICE
        n_focal   = int(n_f * mix)
        n_uniform = n_f - n_focal
        x_uniform = torch.rand(n_uniform, generator=generator, device=device) * (X_HI - X_LO) + X_LO
        x_focal   = (X_ATM + torch.randn(n_focal, generator=generator, device=device) * sigma_is).clamp(X_LO, X_HI)
        x_f = torch.cat([x_uniform, x_focal]).requires_grad_(True)
        t_f = (torch.rand(n_f, generator=generator, device=device) * (T - eps)).requires_grad_(True)
        x_tc = torch.rand(n_tc, generator=generator, device=device) * (X_HI - X_LO) + X_LO
        t_tc = torch.full((n_tc,), T, device=device)
        return x_f, t_f, x_tc, t_tc
    return _sample


def make_sampler_vpinn_logS(
    n_tau: int,
    eps: float = 0.01 * T,
    generator: torch.Generator | None = None,
):
    """Return t_batch (time collocation points for the VPINN weak residual).

    The IC at t=T is enforced via quadrature inside ``_VPINNLossForwardLogS.ic_loss_quad``,
    so no separate TC points are needed here.

    Args:
        n_tau: number of time points per batch.
        eps: temporal truncation — PDE collocation points are drawn from [0, T−eps].
        generator: explicit ``torch.Generator`` (see :func:`make_sampler_naive_logS`).
    """
    def _sample():
        return torch.rand(n_tau, generator=generator, device=p3.DEVICE) * (T - eps)
    return _sample


def make_sampler_vpinn_logS_is_tau(
    n_tau: int,
    alpha: float = 0.3,
    eps: float = 0.001 * T,
    generator: torch.Generator | None = None,
):
    """VPINN sampler biased toward τ → 0 (near maturity).

    Draws τ = T · U^(1/α) with U ~ Uniform(0, 1), then sets t = T − τ.

    Choice of α (where "biased" means "concentration of points near τ=0"):
        α = 1   → classical uniform (no bias)
        α = 0.5 → moderate bias    (half the points fall in [0, T/4])
        α = 0.3 → strong bias      (half the points fall in [0, T·0.099])
        α = 0.2 → very strong bias (half the points fall in [0, T·0.031])

    Expected effect on γ near τ=0
    -----------------------------
    The PDE residual is harder to satisfy near maturity (γ is near-singular
    there).  Concentrating samples in that zone forces the optimizer to spend
    more network capacity there, at the cost of slightly degrading the fit
    elsewhere.

    No importance sampling correction
    ---------------------------------
    Intentionally BIASED: we minimise E_{τ~q}[L_f(T − τ)] rather than
    E_{τ~U(0,T)}[L_f(T − τ)].  That is exactly what we want when the goal is
    to put more weight on the difficult region.  Re-weighting each residual
    by p_uniform(τ) / q(τ) would recover the unbiased estimator of the
    uniform integral — but then the benefit collapses to variance reduction,
    not a targeted fit.

    Args:
        n_tau: number of time points per batch.
        alpha: power-law exponent.  α < 1 ⟹ concentration near τ=0.
        eps: lower clamp on τ to avoid drawing τ=0 exactly (the residual is
             ill-defined there; the pure-payoff IC term handles t=T already).
        generator: explicit ``torch.Generator`` (see :func:`make_sampler_naive_logS`).
    """
    def _sample():
        u = torch.rand(n_tau, generator=generator, device=p3.DEVICE).clamp(min=1e-6)
        tau = T * u.pow(1.0 / alpha)          # density ∝ τ^(α-1)/T^α
        tau = tau.clamp(min=eps)               # guard against τ=0 exactly
        return T - tau                          # back to t coordinates
    return _sample


# ---------------------------------------------------------------------------
# Derivative norm monitoring
# ---------------------------------------------------------------------------

_DERIV_TAU_PROBES    = [0.01 * T, 0.05 * T, 0.25 * T, T]
_DERIV_N_PTS         = 64   # spatial evaluation points for norm computation
_DERIV_SPATIAL_N     = 200  # spatial evaluation points for distribution plots


def _compute_deriv_norms(model: torch.nn.Module) -> tuple[list[float], list[float]]:
    """RMS of ∂_x V and ∂_xx V at each τ in _DERIV_TAU_PROBES over [X_EVAL_LO, X_EVAL_HI].

    Returns two lists (dx_rms, d2x_rms), one entry per tau probe.
    Gradients are taken w.r.t. the spatial input — not w.r.t. model parameters.
    """
    was_training = model.training
    model.eval()
    device = p3.DEVICE
    dx_rms, d2x_rms = [], []
    for tau_val in _DERIV_TAU_PROBES:
        t_val = T - tau_val
        x_p = torch.linspace(X_EVAL_LO, X_EVAL_HI, _DERIV_N_PTS,
                              device=device).requires_grad_(True)
        t_p = torch.full((_DERIV_N_PTS,), t_val, device=device).requires_grad_(True)
        V_p = model(torch.stack([x_p, t_p], dim=1)).squeeze()
        (dV_dx,)   = torch.autograd.grad(V_p.sum(), x_p, create_graph=True)
        (d2V_dx2,) = torch.autograd.grad(dV_dx.sum(), x_p, create_graph=False)
        dx_rms.append(dV_dx.detach().pow(2).mean().sqrt().item())
        d2x_rms.append(d2V_dx2.detach().pow(2).mean().sqrt().item())
    if was_training:
        model.train()
    return dx_rms, d2x_rms


def compute_losses_vpinn_logS(
    model,
    t_batch: torch.Tensor,
    vpinn_module: _VPINNLossForwardLogS,
    payoff_fn,
    lam_f: float | None = None,
):
    """Total loss for VPINN: weak PDE residual + variational (quadrature) IC loss.

    The IC is evaluated at the Gauss-Legendre quadrature nodes — same as the
    PDE residual — giving the true discrete L² penalty:
        L_ic = (1/|Ω|) Σ_q w_q |u_θ(T, x_q) − h(x_q)|²

    The VPINN weak residual L_f is structurally ~10x smaller than the
    strong-form collocation L_f; pass lam_f explicitly to rescale.
    """
    lambda_f = lam_f if lam_f is not None else p3.LAMBDA_F
    loss_f  = vpinn_module(model, t_batch)
    loss_ic = vpinn_module.ic_loss_quad(model, payoff_fn)
    total   = lambda_f * loss_f + p3.LAMBDA_TC * loss_ic
    return total, loss_f.item(), loss_ic.item()


# ---------------------------------------------------------------------------
# Checkpoint helpers — reproducibility on resume
# ---------------------------------------------------------------------------

def _capture_rng_state(sampler_gen: torch.Generator | None = None) -> dict:
    """Snapshot stochastic state for a checkpoint.

    Captures three things, all optional:

    * ``sampler_gen_state`` — the explicit per-variant ``torch.Generator`` used
      by the sampler (preferred path: bit-for-bit reproducible, isolated from
      the global RNG).
    * ``torch_rng`` / ``torch_cuda_rng`` — the *global* CPU and CUDA RNG
      states.  Kept as a fallback to make legacy code paths (e.g. anything
      that still relies on ``torch.rand(...)`` without ``generator=``) also
      survive a resume.

    Captured states are CPU tensors so they can be serialised across devices.
    """
    state: dict = {"torch_rng": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["torch_cuda_rng"] = torch.cuda.get_rng_state_all()
    if sampler_gen is not None:
        state["sampler_gen_state"] = sampler_gen.get_state().cpu()
    return state


def _restore_rng_state(
    ckpt: dict,
    label: str,
    sampler_gen: torch.Generator | None = None,
) -> bool:
    """Restore stochastic state from a checkpoint.

    If ``sampler_gen`` is provided and the checkpoint carries a saved
    ``sampler_gen_state``, the dedicated per-variant generator is restored —
    that is the bit-for-bit reproducible path for new code.  The global CPU
    and CUDA RNG states are also restored when present, for legacy compatibility.

    Returns True if any kind of stochastic state was restored, False otherwise.
    A warning is logged when the checkpoint has neither kind of state (so the
    user knows the resume is statistically equivalent but not bit-for-bit).
    """
    restored_any = False
    if sampler_gen is not None and "sampler_gen_state" in ckpt:
        sampler_gen.set_state(ckpt["sampler_gen_state"].cpu())
        logger.info(
            f"[{label}] Restored dedicated sampler torch.Generator state from checkpoint"
            " — resume is bit-for-bit reproducible vs an equivalent continuous run"
        )
        restored_any = True
    if "torch_rng" in ckpt:
        torch.set_rng_state(ckpt["torch_rng"])
        if "torch_cuda_rng" in ckpt and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt["torch_cuda_rng"])
        if not restored_any:
            logger.info(
                f"[{label}] Restored global RNG state (CPU"
                f"{' + CUDA' if 'torch_cuda_rng' in ckpt else ''}) from checkpoint"
                " — legacy fallback path (consider migrating to a propagated"
                " torch.Generator for stronger guarantees)"
            )
        restored_any = True
    if not restored_any:
        logger.warning(
            f"[{label}] ⚠ Checkpoint has no stochastic state — the resume will draw"
            " from a fresh random state.  The post-resume trajectory will diverge"
            " from a continuous run (statistically equivalent, not bit-for-bit)."
        )
    return restored_any


class BestModelTracker:
    """Keep the model parameters that achieved the lowest loss during training.

    Why: stochastic optimizers (Adam, stochastic L-BFGS) do NOT monotonically
    improve the loss — they can spike up by an order of magnitude near the end
    of training due to a bad mini-batch or a momentum-driven overshoot.
    The last iterate is therefore not always the best one.  We track the best
    state seen so far and restore it at the end of training, which makes the
    saved model independent of the stopping point.

    Usage
    -----
    ```python
    best = BestModelTracker()
    for it in range(start_iter, total_iters + 1):
        loss_val = ...  # current scalar loss
        best.update(model, loss_val, it)
        if checkpoint_path is not None and it % save_every == 0:
            torch.save({..., "best_tracker": best.state_dict()}, checkpoint_path)
    best.restore(model)
    logger.info(f"[label] restored best model from iter {best.best_iter}")
    ```

    Resume support
    --------------
    Serialise via ``state_dict()`` and reload via ``load_state_dict()``.  The
    best state survives interruptions: if you stop mid-training and resume,
    the tracker still knows the best iter from the previous session.

    Cost
    ----
    Each ``update()`` only clones tensors when a strict improvement is observed.
    Memory overhead: one extra copy of the model parameters (~88 KB for our
    PINN ResNet), negligible.
    """

    def __init__(self) -> None:
        self.best_loss: float = float("inf")
        self.best_iter: int = -1
        self._best_state: dict[str, torch.Tensor] | None = None

    def update(self, model: torch.nn.Module, loss: float, it: int) -> bool:
        """Snapshot the model if its current loss strictly improves the best.

        Skips NaN / inf losses defensively — those can appear in stochastic
        L-BFGS during line-search failures and would otherwise corrupt the
        ``best_loss`` comparison (NaN < anything returns False but we want
        an explicit guard).

        Returns True if a new best was recorded.
        """
        if not math.isfinite(loss):
            return False
        if loss < self.best_loss:
            self.best_loss = float(loss)
            self.best_iter = int(it)
            # Detach + clone + move to CPU so the saved state survives device
            # changes and does not pin GPU memory.
            self._best_state = {
                k: v.detach().clone().cpu() for k, v in model.state_dict().items()
            }
            return True
        return False

    def restore(self, model: torch.nn.Module) -> bool:
        """Load the best model parameters into ``model`` (in place).

        Returns True if a state was available to restore (i.e. at least one
        successful ``update`` happened), False otherwise.
        """
        if self._best_state is None:
            return False
        device = next(model.parameters()).device
        model.load_state_dict({k: v.to(device) for k, v in self._best_state.items()})
        return True

    def state_dict(self) -> dict:
        """Serialise the tracker into a checkpoint-friendly dict."""
        return {
            "best_loss":  self.best_loss,
            "best_iter":  self.best_iter,
            "best_state": self._best_state,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore tracker fields from a previously saved ``state_dict()``."""
        self.best_loss   = state.get("best_loss",  float("inf"))
        self.best_iter   = state.get("best_iter",  -1)
        self._best_state = state.get("best_state", None)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_variant(
    model: torch.nn.Module,
    total_iters: int,
    sampler_fn,
    payoff_fn,
    label: str,
    log_every: int | None = None,
    sampler_gen: torch.Generator | None = None,
) -> dict:
    """Adam training loop for strong-form variants.

    ``sampler_gen`` is the explicit ``torch.Generator`` driving ``sampler_fn``.
    It is unused inside this function (this loop has no checkpointing yet) but
    accepted for API uniformity across training functions.
    """
    if log_every is None:
        log_every = p3._adaptive_log_every(total_iters)
    model.to(p3.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, p3.build_lr_lambda(total_iters))
    history: dict = {"loss": [], "loss_f": [], "loss_tc": [], "iter": [], "grad_norm": [], "lr": [],
                     "dx_rms": [], "d2x_rms": []}
    best_tracker = BestModelTracker()
    last_loss = float("nan")
    model.train()
    t0 = time.time()

    for it in range(1, total_iters + 1):
        optimizer.zero_grad()
        x_f, t_f, x_tc, t_tc = sampler_fn()
        loss, lf, ltc = compute_losses_logS(model, x_f, t_f, x_tc, t_tc, payoff_fn)
        loss.backward()

        total_norm = sum(
            p.grad.detach().norm(2).item() ** 2
            for p in model.parameters() if p.grad is not None
        ) ** 0.5

        optimizer.step()
        scheduler.step()

        last_loss = loss.item()
        # Track the best (lowest-loss) model state so the saved model does not
        # depend on whether the last iter happened to be a good or bad one.
        best_tracker.update(model, last_loss, it)

        if it % log_every == 0 or it == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            dx_rms, d2x_rms = _compute_deriv_norms(model)
            history["loss"].append(last_loss)
            history["loss_f"].append(lf)
            history["loss_tc"].append(ltc)
            history["grad_norm"].append(total_norm)
            history["lr"].append(lr_now)
            history["iter"].append(it)
            history["dx_rms"].append(dx_rms)
            history["d2x_rms"].append(d2x_rms)
            logger.info(
                f"[{label}] iter {it:>6d}/{total_iters}  "
                f"loss={last_loss:.4e}  Lf={lf:.4e}  Ltc={ltc:.4e}  "
                f"|g|={total_norm:.2e}  lr={lr_now:.5f}  "
                f"dx_rms(τ={_DERIV_TAU_PROBES[0]:.2f})={dx_rms[0]:.2e}  "
                f"({time.time()-t0:.1f}s)"
            )

    if best_tracker.restore(model):
        logger.info(
            f"[{label}] Restored best model from iter {best_tracker.best_iter} "
            f"(loss={best_tracker.best_loss:.4e}; last iter loss was {last_loss:.4e})"
        )
    model.eval()
    logger.info(f"[{label}] Training done in {time.time()-t0:.1f}s")
    return history


def train_variant_vpinn(
    model: torch.nn.Module,
    vpinn_module: _VPINNLossForwardLogS,
    total_iters: int,
    sampler_fn,
    payoff_fn,
    label: str,
    lam_f: float | None = None,
    log_every: int | None = None,
    sampler_gen: torch.Generator | None = None,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    save_every: int = 500,
) -> dict:
    """Training loop for the VPINN variant (weak-form PDE loss).

    Supports checkpoint save / resume to survive interruptions (the laptop
    closing, an OOM kill, etc.) without losing progress.  The checkpoint
    captures the full state needed for a faithful resume:

    * model parameters,
    * Adam optimizer internal state (per-parameter first- and second-moment
      estimates ``m_t``, ``v_t``),
    * LR scheduler internal state (``last_epoch`` and friends),
    * the running ``history`` dict (so the post-resume training curve is
      continuous),
    * the ``BestModelTracker`` state (best iter + parameter snapshot so far),
    * the explicit sampler ``torch.Generator`` state, when provided, plus
      the global CPU/CUDA RNG states as a fallback.

    ``save_every`` defaults to 500 because a single Adam iter is cheap (~30 ms
    on GPU) — saving every iter would dominate wall-clock; 500 strikes a good
    balance.  Adjust if the checkpoint write becomes a bottleneck.
    """
    if log_every is None:
        log_every = p3._adaptive_log_every(total_iters)
    model.to(p3.DEVICE)
    vpinn_module.to(p3.DEVICE)
    lambda_f  = lam_f if lam_f is not None else p3.LAMBDA_F
    lambda_tc = p3.LAMBDA_TC
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, p3.build_lr_lambda(total_iters))
    history: dict = {"loss": [], "loss_f": [], "loss_tc": [], "iter": [], "grad_norm": [], "lr": [],
                     "dx_rms": [], "d2x_rms": []}
    best_tracker = BestModelTracker()
    last_loss = float("nan")
    start_iter = 1
    if resume and checkpoint_path is not None and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=p3.DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if "history" in ckpt:
            history = ckpt["history"]
        if "best_tracker" in ckpt:
            best_tracker.load_state_dict(ckpt["best_tracker"])
        start_iter = ckpt["iter"] + 1
        logger.info(
            f"[{label}] ── Resumed from checkpoint at iter {ckpt['iter']}/{total_iters} "
            f"(model + Adam moments + LR scheduler + history + best-tracker restored)"
        )
        _restore_rng_state(ckpt, label, sampler_gen=sampler_gen)
    model.train()
    t0 = time.time()
    logger.info(
        f"[{label}] loss = λ_f·Lf + λ_tc·L_ic  "
        f"with λ_f={lambda_f}, λ_tc={lambda_tc}  (L_ic = variational quadrature IC)"
    )

    for it in range(start_iter, total_iters + 1):
        optimizer.zero_grad()
        t_batch = sampler_fn()
        loss, lf, ltc = compute_losses_vpinn_logS(
            model, t_batch, vpinn_module, payoff_fn, lam_f=lam_f
        )
        loss.backward()

        total_norm = sum(
            p.grad.detach().norm(2).item() ** 2
            for p in model.parameters() if p.grad is not None
        ) ** 0.5

        optimizer.step()
        scheduler.step()
        last_loss = loss.item()
        best_tracker.update(model, last_loss, it)

        if it % log_every == 0 or it == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            dx_rms, d2x_rms = _compute_deriv_norms(model)
            history["loss"].append(last_loss)
            history["loss_f"].append(lf)
            history["loss_tc"].append(ltc)
            history["grad_norm"].append(total_norm)
            history["lr"].append(lr_now)
            history["iter"].append(it)
            history["dx_rms"].append(dx_rms)
            history["d2x_rms"].append(d2x_rms)
            logger.info(
                f"[{label}] iter {it:>6d}/{total_iters}  "
                f"loss={last_loss:.4e}  "
                f"(λ_f·Lf={lambda_f * lf:.4e}  λ_tc·L_ic={lambda_tc * ltc:.4e})  "
                f"Lf={lf:.4e}  L_ic={ltc:.4e}  "
                f"|g|={total_norm:.2e}  lr={lr_now:.5f}  "
                f"dx_rms(τ={_DERIV_TAU_PROBES[0]:.2f})={dx_rms[0]:.2e}  "
                f"({time.time()-t0:.1f}s)"
            )

        if checkpoint_path is not None and it % save_every == 0:
            torch.save({
                "iter": it,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "history": history,
                "best_tracker": best_tracker.state_dict(),
                **_capture_rng_state(sampler_gen=sampler_gen),
            }, checkpoint_path)

    if best_tracker.restore(model):
        logger.info(
            f"[{label}] Restored best model from iter {best_tracker.best_iter} "
            f"(loss={best_tracker.best_loss:.4e}; last iter loss was {last_loss:.4e})"
        )
    model.eval()
    logger.info(f"[{label}] Training done in {time.time()-t0:.1f}s")
    return history


def _vpinn_residuals_forward_logS(
    params_dict: dict,
    model: torch.nn.Module,
    t_batch: torch.Tensor,
    x_nodes: torch.Tensor,
    phi_w: torch.Tensor,
    dphi_w: torch.Tensor,
    sigma_: float,
    mu_: float,
    r_: float,
) -> torch.Tensor:
    """Forward-time weak residuals R_{i,k} for the Gram-matrix Jacobian.

    Mirrors ``_VPINNLossForwardLogS.forward`` but uses ``torch.func`` primitives
    (``functional_call`` + ``vmap`` + ``func_grad``) so that ``jacrev`` can
    differentiate through it w.r.t. ``params_dict``.

    Model input convention: ``[x, t]`` (spatial first, time second) — forward time.
    PDE residual (IBP on ∂_{xx}):
        R_{i,k} = ∫ [(∂_t V + μ ∂_x V − r V) φ_k − (σ²/2) ∂_x V ∂_x φ_k] dx
    """
    from torch.func import vmap, grad as func_grad, functional_call as fc

    def u_fn(t_val: torch.Tensor, x_val: torch.Tensor) -> torch.Tensor:
        inp = torch.stack([x_val, t_val]).unsqueeze(0)  # (1, 2): x first, t second
        return fc(model, params_dict, inp).squeeze()

    def _u_at(t, x):
        return u_fn(t, x)

    def _du_dt_at(t, x):
        return func_grad(lambda tv: u_fn(tv, x))(t)

    def _du_dx_at(t, x):
        return func_grad(lambda xv: u_fn(t, xv))(x)

    u_vals = vmap(vmap(_u_at,    in_dims=(None, 0)), in_dims=(0, None))(t_batch, x_nodes)
    u_t    = vmap(vmap(_du_dt_at, in_dims=(None, 0)), in_dims=(0, None))(t_batch, x_nodes)
    u_x    = vmap(vmap(_du_dx_at, in_dims=(None, 0)), in_dims=(0, None))(t_batch, x_nodes)

    # Forward-time BSM residual integrand (same as _VPINNLossForwardLogS)
    f_phi  = u_t + mu_ * u_x - r_ * u_vals           # (N_t, N_q)
    f_dphi = -(sigma_**2 / 2.0) * u_x                 # (N_t, N_q)

    R = f_phi @ phi_w.T + f_dphi @ dphi_w.T  # (N_t, K)
    return R.reshape(-1)  # (N_t * K,)


def train_variant_vpinn_engd(
    model: torch.nn.Module,
    vpinn_module: _VPINNLossForwardLogS,
    total_iters: int,
    sampler_fn,
    payoff_fn,
    label: str = "vpinn_engd",
    lam_f: float | None = None,
    log_every: int | None = None,
    engd_reg: float = 1e-3,
    engd_cg_iters: int = 20,
    engd_n_tau_gram: int = 4,
    engd_K_test_gram: int = 8,
    engd_n_quad_gram: int = 32,
    engd_n_tau_ls: int = 32,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    save_every: int = 50,
    sampler_gen: torch.Generator | None = None,
) -> dict:
    """VPINN training with ENGD (natural gradient).

    ``sampler_gen``: explicit ``torch.Generator`` driving the sampler.  Its
    state is saved in the checkpoint so resumes are bit-for-bit reproducible
    vs an equivalent continuous run.

    The Gram matrix is built from the Jacobian of the forward-time weak residuals
    ``_vpinn_residuals_forward_logS``, matching the ``_VPINNLossForwardLogS``
    geometry.  IC loss is folded into the gradient before the ENGD step so that
    a single natural-gradient update handles both terms simultaneously.

    Performance notes
    -----------------
    * The Gram uses a *smaller* VPINN module (K_test=8, n_quad=32) and fewer
      tau points (n_tau_gram=4) to keep Jacobian computation ≈ 0.7s on CPU
      (J ∈ ℝ^{32 × n_params}).
    * The line-search closure uses a fixed small t_ls grid (n_tau_ls=32 points)
      instead of the 512-point random training batch, reducing LS from ~10s to
      ~0.3s per ENGD step.  Total per-iteration cost ≈ 2s on CPU.
    """
    if log_every is None:
        log_every = p3._adaptive_log_every(total_iters)
    model.to(p3.DEVICE)
    vpinn_module.to(p3.DEVICE)
    lambda_f  = lam_f if lam_f is not None else p3.LAMBDA_F
    lambda_tc = p3.LAMBDA_TC

    # ── Dedicated small VPINN module for the Gram Jacobian ─────────────────
    vpinn_gram = _VPINNLossForwardLogS(
        sigma=vpinn_module.sigma, r=vpinn_module.r,
        x_lo=X_LO, x_hi=X_HI,
        K_test=engd_K_test_gram, n_quad=engd_n_quad_gram,
    ).to(p3.DEVICE)

    # Fixed grids: one for Gram, one for line-search (avoids 512-point LS cost)
    t_gram = torch.linspace(1e-3, T - 1e-3, engd_n_tau_gram, device=p3.DEVICE)
    t_ls   = torch.linspace(1e-3, T - 1e-3, engd_n_tau_ls,   device=p3.DEVICE)

    history: dict = {"loss": [], "loss_f": [], "loss_tc": [], "iter": [],
                     "grad_norm": [], "lr": [], "dx_rms": [], "d2x_rms": []}
    best_tracker = BestModelTracker()
    last_loss = float("nan")
    start_iter = 1
    if resume and checkpoint_path is not None and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=p3.DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        start_iter = ckpt["iter"] + 1
        history    = ckpt["history"]
        if "best_tracker" in ckpt:
            best_tracker.load_state_dict(ckpt["best_tracker"])
        logger.info(f"[{label}] ── Resumed from checkpoint at iter {ckpt['iter']}/{total_iters}")
        _restore_rng_state(ckpt, label, sampler_gen=sampler_gen)

    model.train()
    t0 = time.time()
    logger.info(
        f"[{label}] VPINN+ENGD — reg={engd_reg}  cg_iters={engd_cg_iters}  "
        f"n_tau_gram={engd_n_tau_gram}  K_test_gram={engd_K_test_gram}  "
        f"n_quad_gram={engd_n_quad_gram}  n_tau_ls={engd_n_tau_ls}  "
        f"λ_f={lambda_f}  λ_tc={lambda_tc}"
    )

    for it in range(start_iter, total_iters + 1):
        t_batch = sampler_fn()

        # ── 1. Compute combined gradient g = λ_f ∇L_f + λ_tc ∇L_ic ──────
        model.zero_grad()
        lf_val  = vpinn_module(model, t_batch)
        ltc_val = vpinn_module.ic_loss_quad(model, payoff_fn)
        loss    = lambda_f * lf_val + lambda_tc * ltc_val
        loss.backward()
        g_total = flat_grad(model)

        # ── 2. Gram Jacobian from the small dedicated VPINN module ────────
        params_snap = {k: v.detach().clone() for k, v in model.named_parameters()}
        J = measurement_jacobian(
            _vpinn_residuals_forward_logS,
            params_snap, model,
            t_gram.detach(),
            vpinn_gram.x_nodes,
            vpinn_gram.phi_w,
            vpinn_gram.dphi_w,
            vpinn_gram.sigma,
            vpinn_gram.mu,
            vpinn_gram.r,
        )

        # ── 3. Solve G δ = g via CG ───────────────────────────────────────
        empty_TC = torch.zeros((0, J.shape[1]), dtype=J.dtype, device=J.device)
        delta = solve_cg(
            g_total, J, empty_TC,
            lam_f=1.0, lam_tc=0.0, reg=engd_reg, n_iters=engd_cg_iters,
        )

        # ── 4. Grid line search with fixed small t_ls (fast evaluation) ───
        def _loss_fn():
            return (lambda_f * vpinn_module(model, t_ls)
                    + lambda_tc * vpinn_module.ic_loss_quad(model, payoff_fn))

        step_size = grid_line_search(model, _loss_fn, delta, n_steps=30, step_max=1.0)
        flat0 = flat_params(model)
        set_flat_params(model, flat0 - step_size * delta)

        last_loss = loss.item()
        best_tracker.update(model, last_loss, it)

        if it % log_every == 0 or it == 1:
            total_norm = g_total.norm().item()
            dx_rms, d2x_rms = _compute_deriv_norms(model)
            history["loss"].append(last_loss)
            history["loss_f"].append(lf_val.item())
            history["loss_tc"].append(ltc_val.item())
            history["grad_norm"].append(total_norm)
            history["lr"].append(float(step_size))
            history["iter"].append(it)
            history["dx_rms"].append(dx_rms)
            history["d2x_rms"].append(d2x_rms)
            logger.info(
                f"[{label}] iter {it:>5d}/{total_iters}  "
                f"loss={last_loss:.4e}  Lf={lf_val.item():.4e}  L_ic={ltc_val.item():.4e}  "
                f"alpha={step_size:.2e}  |g|={total_norm:.2e}  "
                f"dx_rms(τ={_DERIV_TAU_PROBES[0]:.2f})={dx_rms[0]:.2e}  "
                f"({time.time()-t0:.1f}s)"
            )

        if checkpoint_path is not None and it % save_every == 0:
            torch.save({"iter": it, "model_state": model.state_dict(),
                        "history": history,
                        "best_tracker": best_tracker.state_dict(),
                        **_capture_rng_state(sampler_gen=sampler_gen)}, checkpoint_path)

    if best_tracker.restore(model):
        logger.info(
            f"[{label}] Restored best model from iter {best_tracker.best_iter} "
            f"(loss={best_tracker.best_loss:.4e}; last iter loss was {last_loss:.4e})"
        )
    model.eval()
    logger.info(f"[{label}] Training done in {time.time()-t0:.1f}s")
    return history


def _strong_residuals_logS(
    params_dict: dict,
    model: torch.nn.Module,
    x_batch: torch.Tensor,
    t_batch: torch.Tensor,
) -> torch.Tensor:
    """Strong-form BSM PDE residuals F[V](x_i, t_i) for the Gram-matrix Jacobian.

    F[V] = dV/dt + σ²/2 d²V/dx² + μ dV/dx - r V   (constant-coefficient BSM in log-S)

    Uses ``torch.func`` primitives so that ``measurement_jacobian`` (which calls
    ``jacrev`` over params_dict) can differentiate through this function.
    Second derivatives in x are computed via nested ``func_grad`` (reverse-over-reverse).
    """
    from torch.func import vmap, grad as func_grad, functional_call as fc

    sigma_, mu_, r_ = sigma, r - 0.5 * sigma**2, r

    def V_fn(x_val: torch.Tensor, t_val: torch.Tensor) -> torch.Tensor:
        inp = torch.stack([x_val, t_val]).unsqueeze(0)  # [x, t] convention
        return fc(model, params_dict, inp).squeeze()

    def residual_at(x_val: torch.Tensor, t_val: torch.Tensor) -> torch.Tensor:
        V       = V_fn(x_val, t_val)
        dV_dt   = func_grad(lambda tv: V_fn(x_val, tv))(t_val)
        dV_dx_fn = func_grad(lambda xv: V_fn(xv, t_val))
        dV_dx   = dV_dx_fn(x_val)
        d2V_dx2 = func_grad(dV_dx_fn)(x_val)
        return dV_dt + 0.5 * sigma_**2 * d2V_dx2 + mu_ * dV_dx - r_ * V

    return vmap(residual_at)(x_batch, t_batch)  # (N,)


def _terminal_values_logS(
    params_dict: dict,
    model: torch.nn.Module,
    x_tc: torch.Tensor,
    t_tc: torch.Tensor,
) -> torch.Tensor:
    """Network output at terminal-condition points V_θ(x_i, T). Shape: (N_tc,).

    Used to build the terminal-condition Jacobian J_TC for the ENGD Gram matrix.
    Model input convention: [x, t] (spatial first, time second).
    """
    from torch.func import functional_call as fc
    inp = torch.stack([x_tc, t_tc], dim=1)  # (N_tc, 2)
    return fc(model, params_dict, inp).squeeze(1)  # (N_tc,)


def train_variant_engd(
    model: torch.nn.Module,
    total_iters: int,
    sampler_fn,   # kept for API compatibility; not used (fixed deterministic grid)
    payoff_fn,
    label: str = "engd",
    log_every: int | None = None,
    n_grid: int = 30,       # N for (N-2)^2 interior × (N-1) terminal grid
    n_tc_grid: int | None = None,  # override # of terminal pts (default: n_grid-1)
    n_ls_steps: int = 30,   # halving steps in line search
    tikhonov_rel: float = 1e-6,  # Tikhonov reg as fraction of ||G||_op (0 disables)
    lam_f_override: float | None = None,
    lam_tc_override: float | None = None,
    preconditioner_mode: str = "joint",   # "joint" | "alt" — G uses both Jacobians or alternates
    checkpoint_path: Path | None = None,
    resume: bool = False,
    save_every: int = 50,
    sampler_gen: torch.Generator | None = None,
) -> dict:
    """Strong-form PINN training with paper-faithful ENGD (lstsq + fixed grid).

    ``sampler_gen``: explicit ``torch.Generator`` driving the sampler.  Its
    state is saved in the checkpoint so resumes are bit-for-bit reproducible
    vs an equivalent continuous run.

    Follows Zeinhofer et al. (ICML 2023) closely:
    - Fixed deterministic grid: (n_grid-2)^2 interior + (n_grid-1) terminal points
    - jacfwd Jacobian for interior (M >> n_params regime)
    - jacrev Jacobian for terminal condition (N_tc < n_params)
    - Gram G = (λ_f/N_int) J_F^T J_F + (λ_tc/N_tc) J_TC^T J_TC
    - Tikhonov regularisation ε‖G‖₂ I (default 1e-6, paper uses 0)
    - Natural gradient direction δ = lstsq(G + ε‖G‖₂ I, g) via SVD pseudoinverse
    - Grid line search with halving step sizes on the same fixed grid

    Requires a small network so that N_int >> n_params (paper regime).
    Use ``_build_engd_pinn()`` (129 params) with n_grid=30 (784 interior).

    Diagnostics (logged at every ``log_every`` step):
    - ``cond(G)`` — Gram condition number (largest / smallest positive eigenvalue)
    - ``|δ|``      — natural-gradient direction norm
    - ``cos(g,δ)`` — cosine angle between ordinary and natural gradient
    """
    if log_every is None:
        log_every = p3._adaptive_log_every(total_iters)
    device = p3.DEVICE
    model.to(device)

    # ── Fixed deterministic grids ────────────────────────────────────────────
    x_all     = torch.linspace(X_LO, X_HI, n_grid, device=device)
    t_all     = torch.linspace(0.0,  T,     n_grid, device=device)
    x_int_1d  = x_all[1:-1]                            # (n_grid-2,)
    t_int_1d  = t_all[1:-1]                            # (n_grid-2,)
    grid_int  = torch.cartesian_prod(x_int_1d, t_int_1d)   # ((n_grid-2)^2, 2)
    x_int     = grid_int[:, 0].contiguous()            # (N_int,)
    t_int     = grid_int[:, 1].contiguous()            # (N_int,)
    if n_tc_grid is not None:
        x_tc  = torch.linspace(X_LO, X_HI, n_tc_grid, device=device)
    else:
        x_tc  = x_all[:-1]                             # (n_grid-1,)
    t_tc      = torch.full((x_tc.shape[0],), T, device=device)
    N_int     = x_int.shape[0]
    N_tc      = x_tc.shape[0]
    lam_f     = p3.LAMBDA_F  if lam_f_override  is None else float(lam_f_override)
    lam_tc    = p3.LAMBDA_TC if lam_tc_override is None else float(lam_tc_override)

    history: dict = {"loss": [], "loss_f": [], "loss_tc": [], "iter": [],
                     "grad_norm": [], "lr": [], "dx_rms": [], "d2x_rms": [],
                     "cond_G": [], "delta_norm": [], "cos_g_delta": [],
                     "lam_max_G": [], "lam_min_G": [], "J_F_norm": [], "J_TC_norm": []}
    best_tracker = BestModelTracker()
    last_loss = float("nan")
    start_iter = 1
    if resume and checkpoint_path is not None and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        start_iter = ckpt["iter"] + 1
        history    = ckpt["history"]
        if "best_tracker" in ckpt:
            best_tracker.load_state_dict(ckpt["best_tracker"])
        logger.info(f"[{label}] ── Resumed from checkpoint at iter {ckpt['iter']}/{total_iters}")
        _restore_rng_state(ckpt, label, sampler_gen=sampler_gen)

    model.train()
    t0 = time.time()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"[{label}] Strong-form ENGD (lstsq, fixed grid)  "
        f"n_grid={n_grid}  N_int={N_int}  N_tc={N_tc}  n_params={n_params}  "
        f"M/n_params={N_int/n_params:.1f}  λ_f={lam_f}  λ_tc={lam_tc}  "
        f"tikhonov_rel={tikhonov_rel:.0e}"
    )

    for it in range(start_iter, total_iters + 1):
        # ── 1. Full gradient on fixed grid ───────────────────────────────────
        model.zero_grad()
        x_f_in = x_int.detach().requires_grad_(True)
        t_f_in = t_int.detach().requires_grad_(True)
        V_int_ = model(torch.stack([x_f_in, t_f_in], dim=1)).squeeze()
        F_int  = bsm_operator_logS(V_int_, x_f_in, t_f_in, r, sigma)
        loss_f = F_int.pow(2).mean()

        V_tc_  = model(torch.stack([x_tc, t_tc], dim=1)).squeeze()
        phi_tc = payoff_fn(x_tc).detach()
        loss_tc = (V_tc_ - phi_tc).pow(2).mean()

        loss = lam_f * loss_f + lam_tc * loss_tc
        loss.backward()
        g_total = flat_grad(model)

        # ── 2. Interior Jacobian via jacfwd (N_int >> n_params) ──────────────
        params_snap = {k: v.detach().clone() for k, v in model.named_parameters()}
        J_F = measurement_jacobian_fwd(
            _strong_residuals_logS,
            params_snap, model,
            x_int.detach(), t_int.detach(),
        )  # (N_int, n_params)

        # ── 3. Terminal Jacobian via jacrev (N_tc < n_params) ────────────────
        J_TC = measurement_jacobian(
            _terminal_values_logS,
            params_snap, model,
            x_tc.detach(), t_tc.detach(),
        )  # (N_tc, n_params)

        # ── 4. Gram: joint (default) or alternating preconditioner ──────────
        G_F  = (lam_f  / N_int) * (J_F.T  @ J_F)
        G_TC = (lam_tc / N_tc ) * (J_TC.T @ J_TC)
        if preconditioner_mode == "joint":
            G = G_F + G_TC
            precond_tag = "J"
        elif preconditioner_mode == "alt":
            # Even (1-indexed) → use J_F only; odd → use J_TC only.
            if (it - 1) % 2 == 0:
                G = G_F
                precond_tag = "F"
            else:
                G = G_TC
                precond_tag = "T"
        else:
            raise ValueError(f"Unknown preconditioner_mode: {preconditioner_mode!r}")

        # ── 5. Natural gradient via lstsq with Tikhonov ε‖G‖₂ I ──────────────
        # gelsd (SVD) is CPU-only; Gram is small (n_params × n_params) so the
        # host round-trip is negligible.  Tikhonov bounds the amplification of
        # gradient components along small-eigenvalue directions of G —
        # essential when the gradient is dominated by terms that the Gram
        # cannot resolve well (e.g. terminal-condition residuals here).
        G_cpu = G.cpu()
        if tikhonov_rel > 0:
            G_op_norm = torch.linalg.matrix_norm(G_cpu, ord=2)
            G_solve = G_cpu + (tikhonov_rel * G_op_norm) * torch.eye(
                G_cpu.shape[0], dtype=G_cpu.dtype
            )
        else:
            G_solve = G_cpu
        delta = torch.linalg.lstsq(
            G_solve, g_total.cpu().unsqueeze(1), driver="gelsd"
        ).solution.squeeze(1).to(device)

        # ── 6. Grid line search on fixed grid ────────────────────────────────
        def _loss_fn():
            xf_ = x_int.detach().requires_grad_(True)
            tf_ = t_int.detach().requires_grad_(True)
            V_ = model(torch.stack([xf_, tf_], dim=1)).squeeze()
            F_ = bsm_operator_logS(V_, xf_, tf_, r, sigma)
            lf_ = F_.pow(2).mean()
            V_tc__ = model(torch.stack([x_tc, t_tc], dim=1)).squeeze()
            ltc_   = (V_tc__ - phi_tc).pow(2).mean()
            return lam_f * lf_ + lam_tc * ltc_

        step_size = grid_line_search(model, _loss_fn, delta, n_steps=n_ls_steps, step_max=1.0)
        flat0 = flat_params(model)
        set_flat_params(model, flat0 - step_size * delta)

        last_loss = loss.item()
        best_tracker.update(model, last_loss, it)

        if it % log_every == 0 or it == 1:
            total_norm = g_total.norm().item()
            dx_rms, d2x_rms = _compute_deriv_norms(model)
            # Diagnostics: spectrum of G (pre-Tikhonov), |δ|, angle(g, δ),
            # and the *residual recovery ratio* ρ — fraction of the stacked
            # residual ‖r_full‖² that lies in col(J_full).  ρ → 1 means a single
            # Gauss-Newton step can zero out the residual (good representability);
            # ρ ≪ 1 means the residual has an irreducible component perpendicular
            # to col(J_full) — no parameter update can fix it.
            with torch.no_grad():
                eigs = torch.linalg.eigvalsh(G_cpu)
                lam_max = float(eigs[-1].item())
                pos = eigs[eigs > 0]
                lam_min_pos = float(pos.min().item()) if pos.numel() > 0 else 0.0
                cond_G = lam_max / lam_min_pos if lam_min_pos > 0 else float("inf")
                delta_norm = float(delta.norm().item())
                g_norm = total_norm
                denom = g_norm * delta_norm + 1e-30
                cos_g_delta = float((g_total @ delta).item() / denom)
                JF_norm  = float(J_F.norm().item())
                JTC_norm = float(J_TC.norm().item())

                # Residual recovery ratio ρ — solves an independent GN system
                # using the *joint* J_full irrespective of preconditioner_mode.
                a_sq = lam_f  / N_int
                b_sq = lam_tc / N_tc
                F_det     = F_int.detach()
                r_tc_det  = V_tc_.detach() - phi_tc
                r_sq      = float(a_sq * (F_det @ F_det).item()
                                  + b_sq * (r_tc_det @ r_tc_det).item())
                JTr       = a_sq * (J_F.T @ F_det) + b_sq * (J_TC.T @ r_tc_det)
                G_joint   = G_F + G_TC
                G_joint_c = G_joint.cpu()
                G_joint_r = G_joint_c + (
                    max(tikhonov_rel, 1e-10)
                    * torch.linalg.matrix_norm(G_joint_c, ord=2)
                ) * torch.eye(G_joint_c.shape[0], dtype=G_joint_c.dtype)
                proj_coef = torch.linalg.lstsq(
                    G_joint_r, JTr.cpu().unsqueeze(1), driver="gelsd"
                ).solution.squeeze(1)
                proj_sq   = float((JTr.cpu() @ proj_coef).item())
                rho       = proj_sq / r_sq if r_sq > 1e-30 else float("nan")
            history["loss"].append(loss.item())
            history["loss_f"].append(loss_f.item())
            history["loss_tc"].append(loss_tc.item())
            history["grad_norm"].append(total_norm)
            history["lr"].append(float(step_size))
            history["iter"].append(it)
            history["dx_rms"].append(dx_rms)
            history["d2x_rms"].append(d2x_rms)
            history["cond_G"].append(cond_G)
            history["delta_norm"].append(delta_norm)
            history["cos_g_delta"].append(cos_g_delta)
            history["lam_max_G"].append(lam_max)
            history["lam_min_G"].append(lam_min_pos)
            history["J_F_norm"].append(JF_norm)
            history["J_TC_norm"].append(JTC_norm)
            history.setdefault("rho_recovery", []).append(rho)
            logger.info(
                f"[{label}] iter {it:>5d}/{total_iters}  G={precond_tag}  "
                f"loss={loss.item():.4e}  Lf={loss_f.item():.4e}  "
                f"Ltc={loss_tc.item():.4e}  "
                f"alpha={step_size:.2e}  |g|={g_norm:.2e}  |δ|={delta_norm:.2e}  "
                f"cos(g,δ)={cos_g_delta:+.3f}  ρ={rho:.4f}  "
                f"cond(G)={cond_G:.2e}  "
                f"|J_F|={JF_norm:.2e}  |J_TC|={JTC_norm:.2e}  "
                f"({time.time()-t0:.1f}s)"
            )

        if checkpoint_path is not None and it % save_every == 0:
            torch.save({"iter": it, "model_state": model.state_dict(),
                        "history": history,
                        "best_tracker": best_tracker.state_dict(),
                        **_capture_rng_state(sampler_gen=sampler_gen)}, checkpoint_path)

    if best_tracker.restore(model):
        logger.info(
            f"[{label}] Restored best model from iter {best_tracker.best_iter} "
            f"(loss={best_tracker.best_loss:.4e}; last iter loss was {last_loss:.4e})"
        )
    model.eval()
    logger.info(f"[{label}] Training done in {time.time()-t0:.1f}s")
    return history


def train_variant_vpinn_lbfgs(
    model: torch.nn.Module,
    vpinn_module: _VPINNLossForwardLogS,
    total_iters: int,
    sampler_fn,
    payoff_fn,
    label: str = "vpinn_lbfgs",
    lam_f: float | None = None,
    log_every: int | None = None,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    save_every: int = 50,
    stochastic_batch: bool = True,
    sampler_gen: torch.Generator | None = None,
) -> dict:
    """VPINN training with L-BFGS (quasi-Newton, limited-memory Hessian approx.).

    ``sampler_gen``: explicit ``torch.Generator`` driving the sampler.  Its
    state is saved in the checkpoint so resumes are bit-for-bit reproducible
    vs an equivalent continuous run.

    Each outer L-BFGS step may call the closure multiple times for the
    strong-Wolfe line search.  One 'iteration' here corresponds to one outer
    L-BFGS step (not one function evaluation).

    Parameters
    ----------
    total_iters : int
        Number of outer L-BFGS steps (each step ≈ 5–20 function evaluations).
    stochastic_batch : bool
        If True (default), resample t_batch at every outer step — the objective
        changes between steps, violating the L-BFGS secant condition and causing
        curvature history to accumulate stochastic noise.
        If False, sample t_batch *once* before the loop and reuse it for all
        outer steps, making the objective deterministic and the curvature
        estimates consistent.
    """
    if log_every is None:
        log_every = p3._adaptive_log_every(total_iters)
    model.to(p3.DEVICE)
    vpinn_module.to(p3.DEVICE)
    lambda_f  = lam_f if lam_f is not None else p3.LAMBDA_F
    lambda_tc = p3.LAMBDA_TC

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=20,
        history_size=100,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
    )

    history: dict = {"loss": [], "loss_f": [], "loss_tc": [], "iter": [],
                     "grad_norm": [], "lr": [], "dx_rms": [], "d2x_rms": []}
    best_tracker = BestModelTracker()
    last_loss = float("nan")
    start_iter = 1
    t_batch_fixed: torch.Tensor | None = None
    if resume and checkpoint_path is not None and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=p3.DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_iter = ckpt["iter"] + 1
        history    = ckpt["history"]
        if "best_tracker" in ckpt:
            best_tracker.load_state_dict(ckpt["best_tracker"])
        if not stochastic_batch and "t_batch_fixed" in ckpt:
            # Restore the exact same time points so the objective is identical
            # to what built the curvature history — secant condition stays valid.
            t_batch_fixed = ckpt["t_batch_fixed"].to(p3.DEVICE)
            logger.info(
                f"[{label}] ── Resumed from checkpoint at iter {ckpt['iter']}/{total_iters} "
                f"— curvature history + fixed t_batch restored"
            )
        else:
            logger.info(f"[{label}] ── Resumed from checkpoint at iter {ckpt['iter']}/{total_iters}"
                        " (L-BFGS curvature history restored)")
        _restore_rng_state(ckpt, label, sampler_gen=sampler_gen)

    model.train()
    t0 = time.time()
    logger.info(
        f"[{label}] VPINN+LBFGS — lr=1.0  max_iter=20  history=100  "
        f"line_search=strong_wolfe  λ_f={lambda_f}  λ_tc={lambda_tc}  "
        f"stochastic_batch={stochastic_batch}"
    )

    # Fixed batch: sample once (or restore from checkpoint), reuse for all outer
    # steps so the objective is deterministic → valid L-BFGS secant condition.
    if not stochastic_batch and t_batch_fixed is None:
        t_batch_fixed = sampler_fn()
    if not stochastic_batch:
        logger.info(
            f"[{label}] Fixed t_batch: {len(t_batch_fixed)} time points reused for all steps"  # type: ignore[arg-type]
        )

    _lf_last, _ltc_last = [0.0], [0.0]

    for it in range(start_iter, total_iters + 1):
        t_batch = sampler_fn() if stochastic_batch else t_batch_fixed  # type: ignore[assignment]

        # Snapshot params before the step so we can roll back on NaN
        params_before = flat_params(model).clone()

        def closure():
            optimizer.zero_grad()
            loss, lf, ltc = compute_losses_vpinn_logS(
                model, t_batch, vpinn_module, payoff_fn, lam_f=lam_f
            )
            loss.backward()
            _lf_last[0]  = lf
            _ltc_last[0] = ltc
            return loss

        loss = optimizer.step(closure)

        # NaN guard: roll back params AND reset optimizer state (corrupted curvature)
        if loss is None or not torch.isfinite(torch.tensor(loss.item())):
            set_flat_params(model, params_before)
            old_lr = optimizer.param_groups[0]["lr"]
            new_lr = old_lr * 0.5
            optimizer.__init__(model.parameters(), lr=new_lr, max_iter=20,
                               history_size=100, line_search_fn="strong_wolfe",
                               tolerance_grad=1e-7, tolerance_change=1e-9)
            logger.warning(f"[{label}] iter {it}: NaN — rolled back + reset optimizer, lr {old_lr:.2e}→{new_lr:.2e}")
            continue

        last_loss = loss.item() if loss is not None else float("nan")
        best_tracker.update(model, last_loss, it)

        if it % log_every == 0 or it == 1:
            total_norm = sum(
                p.grad.detach().norm(2).item() ** 2
                for p in model.parameters() if p.grad is not None
            ) ** 0.5
            dx_rms, d2x_rms = _compute_deriv_norms(model)
            loss_val = last_loss
            history["loss"].append(loss_val)
            history["loss_f"].append(_lf_last[0])
            history["loss_tc"].append(_ltc_last[0])
            history["grad_norm"].append(total_norm)
            history["lr"].append(1.0)
            history["iter"].append(it)
            history["dx_rms"].append(dx_rms)
            history["d2x_rms"].append(d2x_rms)
            logger.info(
                f"[{label}] iter {it:>5d}/{total_iters}  "
                f"loss={loss_val:.4e}  "
                f"(λ_f·Lf={lambda_f * _lf_last[0]:.4e}  "
                f"λ_tc·L_ic={lambda_tc * _ltc_last[0]:.4e})  "
                f"Lf={_lf_last[0]:.4e}  L_ic={_ltc_last[0]:.4e}  "
                f"|g|={total_norm:.2e}  "
                f"dx_rms(τ={_DERIV_TAU_PROBES[0]:.2f})={dx_rms[0]:.2e}  "
                f"({time.time()-t0:.1f}s)"
            )

        if checkpoint_path is not None and it % save_every == 0:
            ckpt_data: dict = {"iter": it, "model_state": model.state_dict(),
                               "optimizer_state": optimizer.state_dict(),
                               "history": history,
                               "best_tracker": best_tracker.state_dict(),
                               **_capture_rng_state(sampler_gen=sampler_gen)}
            if t_batch_fixed is not None:
                ckpt_data["t_batch_fixed"] = t_batch_fixed.cpu()
            torch.save(ckpt_data, checkpoint_path)

    if best_tracker.restore(model):
        logger.info(
            f"[{label}] Restored best model from iter {best_tracker.best_iter} "
            f"(loss={best_tracker.best_loss:.4e}; last iter loss was {last_loss:.4e})"
        )
    model.eval()
    logger.info(f"[{label}] Training done in {time.time()-t0:.1f}s")
    return history


def train_variant_vpinn_lbfgs_epoch(
    model: torch.nn.Module,
    vpinn_module: _VPINNLossForwardLogS,
    total_iters: int,
    sampler_fn,
    payoff_fn,
    label: str = "vpinn_lbfgs_epoch",
    lam_f: float | None = None,
    log_every: int | None = None,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    save_every: int = 50,
    epoch_size: int = 20,
    sampler_gen: torch.Generator | None = None,
) -> dict:
    """VPINN training with epoch-based L-BFGS (fixed batch within an epoch).

    ``sampler_gen``: explicit ``torch.Generator`` driving the sampler.  Its
    state is saved in the checkpoint so resumes are bit-for-bit reproducible
    vs an equivalent continuous run.

    Principle
    ---------
    Training is split into "epochs" of ``epoch_size`` consecutive L-BFGS outer
    steps.  Within each epoch ``t_batch`` is drawn once and kept fixed; the
    L-BFGS curvature history is cleared at the start of every new epoch.

    Why this fixes the corrupted-curvature problem
    ----------------------------------------------
    L-BFGS estimates curvature through the secant condition

        y_k = ∇f(x_{k+1}) − ∇f(x_k)  ≈  H · s_k

    which is only valid when both gradients are taken on the *same* objective
    ``f``.  Resampling the batch at every step computes
    ``∇f_{B_{k+1}}(x_{k+1}) − ∇f_{B_k}(x_k)`` — that is noise, not curvature.
    Pinning the batch across a whole epoch makes every (s_k, y_k) pair stored
    in the history mutually consistent.

    A fresh batch is drawn at every new epoch so that the time integral over
    [0, T] is still well covered across the run.

    Parameters
    ----------
    epoch_size : int
        Number of L-BFGS outer steps per epoch.  Must be ≤ history_size
        (which we set equal to epoch_size here) so the buffer fills exactly
        once per epoch.  Recommended value: 20.
    """
    if log_every is None:
        log_every = p3._adaptive_log_every(total_iters)
    model.to(p3.DEVICE)
    vpinn_module.to(p3.DEVICE)
    lambda_f  = lam_f if lam_f is not None else p3.LAMBDA_F
    lambda_tc = p3.LAMBDA_TC

    def _make_optimizer():
        """Build a fresh L-BFGS optimizer with an empty curvature history."""
        return torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=20,
            history_size=epoch_size,   # buffer sized for exactly one epoch
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
        )

    optimizer = _make_optimizer()

    history: dict = {"loss": [], "loss_f": [], "loss_tc": [], "iter": [],
                     "grad_norm": [], "lr": [], "dx_rms": [], "d2x_rms": []}
    best_tracker = BestModelTracker()
    last_loss = float("nan")
    start_iter = 1
    t_batch: torch.Tensor = sampler_fn()   # current epoch's batch
    if resume and checkpoint_path is not None and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=p3.DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_iter = ckpt["iter"] + 1
        history    = ckpt["history"]
        if "best_tracker" in ckpt:
            best_tracker.load_state_dict(ckpt["best_tracker"])
        if "t_batch_epoch" in ckpt:
            t_batch = ckpt["t_batch_epoch"].to(p3.DEVICE)
        logger.info(
            f"[{label}] ── Resumed from checkpoint at iter {ckpt['iter']}/{total_iters} "
            f"(L-BFGS curvature history + epoch t_batch restored)"
        )
        _restore_rng_state(ckpt, label, sampler_gen=sampler_gen)

    model.train()
    t0 = time.time()
    n_epochs_total = math.ceil(total_iters / epoch_size)
    logger.info(
        f"[{label}] Epoch-based VPINN + L-BFGS — "
        f"epoch_size={epoch_size}  history_size={epoch_size}  "
        f"total_iters={total_iters}  n_epochs≈{n_epochs_total}  "
        f"lr=1.0  line_search=strong_wolfe  λ_f={lambda_f}  λ_tc={lambda_tc}"
    )

    _lf_last, _ltc_last = [0.0], [0.0]
    current_epoch = (start_iter - 1) // epoch_size

    for it in range(start_iter, total_iters + 1):
        # ── Start of a new epoch ───────────────────────────────────────────
        new_epoch = (it - 1) // epoch_size
        if new_epoch != current_epoch:
            current_epoch = new_epoch
            # Fresh draw: cover a new slice of [0, T] for this epoch.
            t_batch = sampler_fn()
            # Clear curvature history: (s, y) pairs from the previous epoch
            # are no longer consistent with the new objective f_{t_batch}.
            optimizer = _make_optimizer()
            logger.info(
                f"[{label}] ── Epoch {current_epoch + 1}/{n_epochs_total} "
                f"(iter {it}): drew a fresh t_batch and reset the L-BFGS history"
            )

        # Snapshot params so we can roll back on a NaN
        params_before = flat_params(model).clone()

        def closure():
            optimizer.zero_grad()
            loss, lf, ltc = compute_losses_vpinn_logS(
                model, t_batch, vpinn_module, payoff_fn, lam_f=lam_f
            )
            loss.backward()
            _lf_last[0]  = lf
            _ltc_last[0] = ltc
            return loss

        loss = optimizer.step(closure)

        # NaN guard: roll back parameters and force a new epoch
        if loss is None or not torch.isfinite(torch.tensor(loss.item())):
            set_flat_params(model, params_before)
            logger.warning(
                f"[{label}] iter {it}: NaN detected — rolling back + "
                f"forcing a new epoch at next step"
            )
            current_epoch = -1
            continue

        last_loss = loss.item()
        best_tracker.update(model, last_loss, it)

        if it % log_every == 0 or it == 1:
            total_norm = sum(
                p.grad.detach().norm(2).item() ** 2
                for p in model.parameters() if p.grad is not None
            ) ** 0.5
            dx_rms, d2x_rms = _compute_deriv_norms(model)
            loss_val = last_loss
            history["loss"].append(loss_val)
            history["loss_f"].append(_lf_last[0])
            history["loss_tc"].append(_ltc_last[0])
            history["grad_norm"].append(total_norm)
            history["lr"].append(1.0)
            history["iter"].append(it)
            history["dx_rms"].append(dx_rms)
            history["d2x_rms"].append(d2x_rms)
            logger.info(
                f"[{label}] iter {it:>5d}/{total_iters}  epoch {current_epoch + 1}/{n_epochs_total}  "
                f"loss={loss_val:.4e}  "
                f"(λ_f·Lf={lambda_f * _lf_last[0]:.4e}  "
                f"λ_tc·L_ic={lambda_tc * _ltc_last[0]:.4e})  "
                f"Lf={_lf_last[0]:.4e}  L_ic={_ltc_last[0]:.4e}  "
                f"|g|={total_norm:.2e}  "
                f"dx_rms(τ={_DERIV_TAU_PROBES[0]:.2f})={dx_rms[0]:.2e}  "
                f"({time.time()-t0:.1f}s)"
            )

        if checkpoint_path is not None and it % save_every == 0:
            torch.save({
                "iter": it,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "t_batch_epoch": t_batch.cpu(),
                "history": history,
                "best_tracker": best_tracker.state_dict(),
                **_capture_rng_state(sampler_gen=sampler_gen),
            }, checkpoint_path)

    if best_tracker.restore(model):
        logger.info(
            f"[{label}] Restored best model from iter {best_tracker.best_iter} "
            f"(loss={best_tracker.best_loss:.4e}; last iter loss was {last_loss:.4e})"
        )
    model.eval()
    logger.info(
        f"[{label}] Training done in {time.time()-t0:.1f}s  "
        f"({n_epochs_total} epochs of {epoch_size} steps)"
    )
    return history


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(model: torch.nn.Module, hist: dict) -> dict:
    device = p3.DEVICE
    model.eval()

    # ── Full price grid in (x, t) space ─────────────────────────────────
    n_x, n_t = 80, 50
    t_max  = T - 1e-2          # avoid t=T (tau=0 singularity)
    x_vals = torch.linspace(X_EVAL_LO, X_EVAL_HI, n_x, device=device)
    t_vals = torch.linspace(0.0, t_max, n_t, device=device)
    X_grid, T_grid = torch.meshgrid(x_vals, t_vals, indexing="ij")  # (n_x, n_t)

    with torch.no_grad():
        x_in  = torch.stack([X_grid.reshape(-1), T_grid.reshape(-1)], dim=1)
        V_pred = model(x_in).squeeze().reshape(n_x, n_t)
        V_ref  = bs_call_logS(X_grid, T_grid)

    err    = V_pred - V_ref
    rel_l2 = ((err ** 2).sum() / (V_ref ** 2).sum()).sqrt().item()

    # ATM band: x ∈ [ln(0.9K), ln(1.1K)]
    x_lo_atm = math.log(0.9 * K)
    x_hi_atm = math.log(1.1 * K)
    mask_atm = (X_grid >= x_lo_atm) & (X_grid <= x_hi_atm)
    rel_l2_atm = (
        (err[mask_atm] ** 2).sum() / (V_ref[mask_atm] ** 2).sum()
    ).sqrt().item()

    # ── Greeks at tau = T/2 (t = T/2) ───────────────────────────────────
    t_fix  = T / 2.0
    tau_fix = T - t_fix
    n_greek = 100
    x_1d = torch.linspace(X_EVAL_LO, X_EVAL_HI, n_greek, device=device).requires_grad_(True)
    t_1d = torch.full((n_greek,), t_fix, device=device).requires_grad_(True)
    V_1d = model(torch.stack([x_1d, t_1d], dim=1)).squeeze()

    # dV/dx and d^2V/dx^2
    (dV_dx,) = torch.autograd.grad(V_1d.sum(), x_1d, create_graph=True)
    (d2V_dx2,) = torch.autograd.grad(dV_dx.sum(), x_1d, create_graph=False)

    # Convert to Delta_S and Gamma_S:
    #   Delta_S = dV/dS = dV/dx * 1/S = e^{-x} * dV/dx
    #   Gamma_S = d^2V/dS^2 = e^{-2x}*(d^2V/dx^2 - dV/dx)
    with torch.no_grad():
        x_d   = x_1d.detach()
        S_1d  = x_d.exp()
        tau_t = torch.full((n_greek,), tau_fix, device=device)
        d1    = (x_d - math.log(K) + (r + 0.5*sigma**2)*tau_t) / (sigma*tau_t.sqrt())
        sqrt2 = math.sqrt(2.0)
        delta_ref = 0.5 * torch.erfc(-d1 / sqrt2)               # N(d1) = Delta_S
        gamma_ref = (
            torch.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
            / (S_1d * sigma * tau_t.sqrt())
        )

    delta_pred = (dV_dx.detach()  * (-x_d).exp())             # e^{-x} * dV/dx
    gamma_pred = ((d2V_dx2.detach() - dV_dx.detach()) * (-2*x_d).exp())

    rel_l2_delta = (((delta_pred - delta_ref)**2).sum() / (delta_ref**2).sum()).sqrt().item()
    rel_l2_gamma = (((gamma_pred - gamma_ref)**2).sum() / (gamma_ref**2).sum()).sqrt().item()

    # ── GEI ─────────────────────────────────────────────────────────────
    norms  = np.array(hist["grad_norm"])
    cutoff = max(1, int(len(norms) * 2 / 3))
    norms_early = norms[:cutoff]
    gei = float(norms_early.max() / (np.median(norms_early) + 1e-10))

    # ── PDE residual profile along x = ln(K) slice ──────────────────────
    n_tau_profile = 25
    tau_profile = torch.linspace(1e-2, T, n_tau_profile, device=device)
    res_profile = []
    for tau_val in tau_profile:
        t_val = T - tau_val.item()
        n_pts = 50
        x_p = torch.full((n_pts,), X_ATM, device=device).requires_grad_(True)
        t_p = torch.full((n_pts,), t_val, device=device).requires_grad_(True)
        V_p = model(torch.stack([x_p, t_p], dim=1)).squeeze()
        F_p = bsm_operator_logS(V_p, x_p, t_p, r, sigma)
        res_profile.append(F_p.detach().abs().mean().item())

    return {
        "rel_l2":        rel_l2,
        "rel_l2_atm":    rel_l2_atm,
        "rel_l2_delta":  rel_l2_delta,
        "rel_l2_gamma":  rel_l2_gamma,
        "gei":           gei,
        "pde_residual_tau": {
            "tau":      tau_profile.cpu().tolist(),
            "residual": res_profile,
        },
    }


# ---------------------------------------------------------------------------
# Ground-truth comparison slices
# ---------------------------------------------------------------------------

_GT_TAU_SLICES = [0.02 * T, T / 4, T / 2, 3 * T / 4, T]  # near-singularity → t=0
_GT_N_X = 120
# τ grid for weak-residual profile (denser near singularity)
_WEAK_RES_TAU = [0.01*T, 0.02*T, 0.05*T, 0.10*T, T/4, T/2, 3*T/4, T]


def _compute_weak_residual_profile(model: torch.nn.Module) -> np.ndarray:
    """Evaluate mean weak-form PDE residual at each τ in _WEAK_RES_TAU.

    Uses the same GL quadrature and sine test functions as _VPINNLossForwardLogS,
    applied to any variant — strong-form PINNs included.
    Returns an array of shape (len(_WEAK_RES_TAU),).
    """
    device = p3.DEVICE
    vpinn_eval = _VPINNLossForwardLogS(
        sigma, r, X_LO, X_HI, K_test=20, n_quad=100
    ).to(device)
    model.eval()
    vals = []
    for tau_val in _WEAK_RES_TAU:
        t_val = T - tau_val
        t_batch = torch.tensor([t_val], device=device)
        with torch.enable_grad():
            lf = vpinn_eval(model, t_batch).item()
        vals.append(lf)
    return np.array(vals)


def _compute_gt_slices(model: torch.nn.Module) -> dict:
    """Compute price and greek slices vs Black-Scholes for ground-truth comparison."""
    device = p3.DEVICE
    model.eval()
    x_vals = torch.linspace(X_EVAL_LO, X_EVAL_HI, _GT_N_X, device=device)

    V_pred_slices, V_ref_slices = [], []
    for tau_val in _GT_TAU_SLICES:
        t_val = T - tau_val
        t_vec = torch.full((_GT_N_X,), t_val, device=device)
        inp = torch.stack([x_vals, t_vec], dim=1)
        with torch.no_grad():
            V_pred = model(inp).squeeze()
            V_ref  = bs_call_logS(x_vals, t_vec)
        V_pred_slices.append(V_pred.cpu().numpy())
        V_ref_slices.append(V_ref.cpu().numpy())

    # Greeks at every tau in _GT_TAU_SLICES (so we can see near-singularity behaviour)
    delta_pred_slices, delta_ref_slices = [], []
    gamma_pred_slices, gamma_ref_slices = [], []
    for tau_val in _GT_TAU_SLICES:
        t_fix = T - tau_val
        x_1d = torch.linspace(X_EVAL_LO, X_EVAL_HI, _GT_N_X, device=device).requires_grad_(True)
        t_1d = torch.full((_GT_N_X,), t_fix, device=device).requires_grad_(True)
        V_1d = model(torch.stack([x_1d, t_1d], dim=1)).squeeze()
        (dV_dx,)   = torch.autograd.grad(V_1d.sum(), x_1d, create_graph=True)
        (d2V_dx2,) = torch.autograd.grad(dV_dx.sum(), x_1d, create_graph=False)
        with torch.no_grad():
            x_d   = x_1d.detach()
            S_1d  = x_d.exp()
            tau_t = torch.full((_GT_N_X,), tau_val, device=device)
            d1    = (x_d - math.log(K) + (r + 0.5*sigma**2)*tau_t) / (sigma*tau_t.sqrt())
            sqrt2 = math.sqrt(2.0)
            delta_ref = 0.5 * torch.erfc(-d1 / sqrt2)
            gamma_ref = (
                torch.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
                / (S_1d * sigma * tau_t.sqrt())
            )
            delta_pred = dV_dx.detach()  * (-x_d).exp()
            gamma_pred = (d2V_dx2.detach() - dV_dx.detach()) * (-2 * x_d).exp()
        delta_pred_slices.append(delta_pred.cpu().numpy())
        delta_ref_slices.append(delta_ref.cpu().numpy())
        gamma_pred_slices.append(gamma_pred.cpu().numpy())
        gamma_ref_slices.append(gamma_ref.cpu().numpy())

    # Raw ∂_x V̂ and ∂_xx V̂ on a fine grid at _DERIV_TAU_PROBES (for spatial distribution plots)
    x_sp = torch.linspace(X_EVAL_LO, X_EVAL_HI, _DERIV_SPATIAL_N, device=device)
    dx_pred_sp, d2x_pred_sp, dx_ref_sp, d2x_ref_sp = [], [], [], []
    sqrt2 = math.sqrt(2.0)
    for tau_val in _DERIV_TAU_PROBES:
        t_fix = T - tau_val
        x_1d = x_sp.detach().clone().requires_grad_(True)
        t_1d = torch.full((_DERIV_SPATIAL_N,), t_fix, device=device).requires_grad_(True)
        V_1d = model(torch.stack([x_1d, t_1d], dim=1)).squeeze()
        (dV_dx,)   = torch.autograd.grad(V_1d.sum(), x_1d, create_graph=True)
        (d2V_dx2,) = torch.autograd.grad(dV_dx.sum(), x_1d, create_graph=False)
        with torch.no_grad():
            x_d_sp  = x_1d.detach()
            tau_t   = torch.full((_DERIV_SPATIAL_N,), tau_val, device=device)
            d1      = (x_d_sp - math.log(K) + (r + 0.5*sigma**2)*tau_t) / (sigma*tau_t.sqrt())
            Nd1     = 0.5 * torch.erfc(-d1 / sqrt2)
            phid1   = torch.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
            dx_ref  = x_d_sp.exp() * Nd1
            d2x_ref = x_d_sp.exp() * (Nd1 + phid1 / (sigma * tau_t.sqrt()))
        dx_pred_sp.append(dV_dx.detach().cpu().numpy())
        d2x_pred_sp.append(d2V_dx2.detach().cpu().numpy())
        dx_ref_sp.append(dx_ref.cpu().numpy())
        d2x_ref_sp.append(d2x_ref.cpu().numpy())

    return {
        "x_vals":             x_vals.cpu().numpy(),
        "tau_slices":         np.array(_GT_TAU_SLICES),
        "V_pred_slices":      np.array(V_pred_slices),
        "V_ref_slices":       np.array(V_ref_slices),
        "x_greek":            x_d.cpu().numpy(),
        "delta_pred_slices":  np.array(delta_pred_slices),
        "delta_ref_slices":   np.array(delta_ref_slices),
        "gamma_pred_slices":  np.array(gamma_pred_slices),
        "gamma_ref_slices":   np.array(gamma_ref_slices),
        "x_deriv_spatial":    x_sp.cpu().numpy(),
        "dx_pred_spatial":    np.array(dx_pred_sp),
        "d2x_pred_spatial":   np.array(d2x_pred_sp),
        "dx_ref_spatial":     np.array(dx_ref_sp),
        "d2x_ref_spatial":    np.array(d2x_ref_sp),
        "weak_residual_tau":  np.array(_WEAK_RES_TAU),
        "weak_residual":      _compute_weak_residual_profile(model),
    }


# ---------------------------------------------------------------------------
# Persistence  (identical structure to ablation_singularity.py)
# ---------------------------------------------------------------------------

def _to_mpl_ls(ls):
    """Convert nested list linestyle to matplotlib tuple (recursive).

    matplotlib requires ``(offset, (on, off, ...))`` tuples for complex dash
    patterns.  YAML ``safe_load`` and Python list literals both produce lists,
    so we normalise to tuples here before passing to any plotting call.
    """
    if isinstance(ls, list):
        return tuple(_to_mpl_ls(x) if isinstance(x, list) else x for x in ls)
    return ls

def _save_variant(res: dict, vdir: Path) -> None:
    hist = res["hist"]
    np.savez_compressed(vdir / "hist.npz", **{k: np.array(v) for k, v in hist.items()})
    m = res["metrics"]
    np.savez_compressed(
        vdir / "metrics.npz",
        rel_l2=np.array([m["rel_l2"]]),
        rel_l2_atm=np.array([m["rel_l2_atm"]]),
        rel_l2_delta=np.array([m["rel_l2_delta"]]),
        rel_l2_gamma=np.array([m["rel_l2_gamma"]]),
        gei=np.array([m["gei"]]),
        pde_tau=np.array(m["pde_residual_tau"]["tau"]),
        pde_residual=np.array(m["pde_residual_tau"]["residual"]),
    )
    if res.get("gt_slices") is not None:
        np.savez_compressed(vdir / "gt_comparison.npz", **res["gt_slices"])


def _load_variant(vdir: Path, summary_entry: dict) -> dict:
    h = np.load(vdir / "hist.npz")
    hist = {k: h[k].tolist() for k in h.files}
    m = np.load(vdir / "metrics.npz")
    metrics = {
        "rel_l2":       float(m["rel_l2"][0]),
        "rel_l2_atm":   float(m["rel_l2_atm"][0]),
        "rel_l2_delta": float(m["rel_l2_delta"][0]),
        "rel_l2_gamma": float(m["rel_l2_gamma"][0]),
        "gei":          float(m["gei"][0]),
        "pde_residual_tau": {
            "tau":      m["pde_tau"].tolist(),
            "residual": m["pde_residual"].tolist(),
        },
    }
    gt_path = vdir / "gt_comparison.npz"
    gt_slices = None
    if gt_path.exists():
        g = np.load(gt_path)
        gt_slices = {k: g[k] for k in g.files}
    return {
        **summary_entry,
        "linestyle": _to_mpl_ls(summary_entry.get("linestyle", "-")),
        "hist": hist, "metrics": metrics, "gt_slices": gt_slices,
    }


# ---------------------------------------------------------------------------
# Formula annotations
# ---------------------------------------------------------------------------

_BOX_STYLE = dict(boxstyle="round,pad=0.6", facecolor="lightyellow", edgecolor="gray", alpha=0.9)


def _add_formula_box(fig, text: str, bottom_margin: float = 0.12) -> None:
    """Anchor an annotation box strictly *below* the figure body.

    The text is placed at y = -0.04 (in figure coords) with vertical anchor
    "top" — so it sits beneath the plotting area.  When the figure is saved
    with ``bbox_inches="tight"``, matplotlib expands the saved bitmap to
    include the box, producing a clear visible gap between the plots and
    the formula text.  ``bottom_margin`` still pads the body so subplot
    labels do not overlap the legend or x-axis ticks.
    """
    fig.text(0.5, -0.04, text, ha="center", va="top", fontsize=8,
             bbox=_BOX_STYLE, linespacing=1.6)
    fig.subplots_adjust(bottom=bottom_margin)


_FORMULA_OP = (
    r"BSM operator ($x=\ln S$, constant coefficients):  "
    r"$\mathcal{F}[V] = \partial_t V + \frac{\sigma^2}{2}\partial_{xx}V"
    r" + (r-\frac{\sigma^2}{2})\partial_x V - rV$"
)
_FORMULA_LF = "\n".join([
    r"$\mathcal{L}_f = \frac{1}{N_f}\sum_i \mathcal{F}[\hat{V}](x_i, t_i)^2$",
    _FORMULA_OP,
])
_FORMULA_LTC = "\n".join([
    r"$\mathcal{L}_{tc} = \frac{1}{N_{tc}}\sum_i(\hat{V}(x_i,T)-\Phi(x_i))^2$   with $x=\ln S$",
    r"Naïf / trunc.:  $\Phi(x)=(e^x-K)^+$",
    r"Smooth:  $\tilde{\Phi}_\beta(x)=\frac{1}{\beta}\ln(1+e^{\beta(e^x-K)})-\frac{\ln 2}{\beta}$",
])
_FORMULA_GRAD = "\n".join([
    r"$\|\nabla_\theta\mathcal{L}\|_2 = \sqrt{\sum_l \|\nabla_{\theta_l}\mathcal{L}\|_2^2}$",
    r"$\mathcal{L} = \lambda_f\,\mathcal{L}_f + \lambda_{tc}\,\mathcal{L}_{tc}$"
    rf"   with $\lambda_f={p3.LAMBDA_F}$,  $\lambda_{{tc}}={p3.LAMBDA_TC}$",
])
_FORMULA_PDE_TAU = "\n".join([
    r"$\bar{F}(\tau) = \frac{1}{N}\sum_i |\mathcal{F}[\hat{V}](x=\ln K,\, T-\tau)|$   (50 pts at ATM)",
    _FORMULA_OP,
])
_FORMULA_METRICS = "\n".join([
    r"$\varepsilon_{L^2}=\|\hat{V}-C^{\mathrm{BS}}\|_2/\|C^{\mathrm{BS}}\|_2$"
    r"   (grid $x\in[\ln 60,\ln 140]$, $t\in[0,\,T-0.01]$)",
    r"$\varepsilon_{L^2}^{\mathrm{ATM}}$: same restricted to $x\in[\ln(0.9K),\ln(1.1K)]$",
    r"$\varepsilon_\Delta$: rel. $L^2$ of $e^{-x}\partial_x\hat{V}$ vs $N(d_1)$ at $\tau=T/2$"
    r"     $\varepsilon_\Gamma$: rel. $L^2$ of $e^{-2x}(\partial_{xx}\hat{V}-\partial_x\hat{V})$"
    r" vs $N'(d_1)/(S\sigma\sqrt{\tau})$ at $\tau=T/2$",
    r"$\mathrm{GEI}=\max\|\nabla_\theta\mathcal{L}\|/\mathrm{median}\|\nabla_\theta\mathcal{L}\|$"
    r"   (first 2/3 of training)"
    r"     $C^{\mathrm{BS}}=S-Ke^{-r\tau}+P^{\mathrm{BS}}$,  $d_1=(x-\ln K+(r+\sigma^2/2)\tau)/(\sigma\sqrt{\tau})$",
])
_FORMULA_PRICE = "\n".join([
    r"$\hat{V}$: PINN prediction in $(x=\ln S,\,t)$ coords, displayed vs $S=e^x$",
    r"$C^{\mathrm{BS}}=S\,N(d_1)-Ke^{-r\tau}N(d_2)$,"
    r"  $d_1=\frac{x-\ln K+(r+\sigma^2/2)\tau}{\sigma\sqrt{\tau}}$,  $d_2=d_1-\sigma\sqrt{\tau}$",
])
_FORMULA_GREEKS_CMP = "\n".join([
    r"$\hat{\Delta}=e^{-x}\partial_x\hat{V}$ (chain rule: $\partial_S V=e^{-x}\partial_x V$)"
    r"     $\hat{\Gamma}=e^{-2x}(\partial_{xx}\hat{V}-\partial_x\hat{V})$"
    r" (chain rule: $\partial_{SS}V=e^{-2x}(\partial_{xx}V-\partial_x V)$)",
    r"BS refs: $\Delta^{\mathrm{BS}}=N(d_1)$,  $\Gamma^{\mathrm{BS}}=N'(d_1)/(S\sigma\sqrt{\tau})$"
    r"  at $\tau=T/2$",
])
_FORMULA_LF_VPINN = "\n".join([
    r"$\mathcal{L}_f=\frac{1}{N_t K}\sum_{i,k}R_{i,k}^2$  —  VPINN weak residual",
    r"$R_{i,k}=\int_{X_{lo}}^{X_{hi}}"
    r"[\partial_t\hat{V}\,\phi_k"
    r" - \frac{\sigma^2}{2}\partial_x\hat{V}\,\partial_x\phi_k"
    r" + \mu\,\partial_x\hat{V}\,\phi_k"
    r" - r\hat{V}\,\phi_k]\,dx$,  $\mu=r-\sigma^2/2$",
    r"$\phi_k(x)=\sin\left(\frac{k\pi(x-X_{lo})}{X_{hi}-X_{lo}}\right)$"
    r"  — IBP on $\partial_{xx}\hat{V}$ eliminates 2nd-order autograd",
])
_FORMULA_IC_QUAD = "\n".join([
    r"$\mathcal{L}_{ic}^{var}(\theta)="
    r"\frac{\lambda_{tc}}{|\Omega|}\sum_{q=1}^{N_q}w_q\,|\hat{u}(T,x_q)-h(x_q)|^2"
    r"\approx\frac{\lambda_{tc}}{|\Omega|}\|\hat{u}(T,\cdot)-h\|^2_{L^2(\Omega)}$",
    r"Uses Gauss-Legendre nodes $\{x_q,w_q\}$ (same as PDE residual)"
    r" — consistent $L^2$ penalization, no extra random samples.",
])
_FORMULA_DX_NORM = "\n".join([
    r"$\mathrm{RMS}_\tau(\partial_x\hat{V})"
    r"=\left(\frac{1}{N}\sum_{i=1}^N|\partial_x\hat{V}(x_i,\,T-\tau)|^2\right)^{1/2}$"
    r"  —  $x_i$ uniform on $[x_{lo},x_{hi}]$, $N=" + str(_DERIV_N_PTS) + r"$",
    r"Near $\tau=0$: singularity in $\partial_x\hat{V}$ (discontinuous payoff slope at $x=\ln K$)."
    r"  $\mathrm{RMS}(\partial_{xx}\hat{V})$ amplifies this further.",
])
_FORMULA_DERIV_SPATIAL = "\n".join([
    r"$\partial_x V^{\mathrm{BS}} = e^x N(d_1)$"
    r",  $\partial_{xx} V^{\mathrm{BS}} = e^x\!\left[N(d_1)+\frac{\varphi(d_1)}{\sigma\sqrt{\tau}}\right]$"
    r"  where $\varphi = \frac{e^{-d_1^2/2}}{\sqrt{2\pi}}$,  $d_1=\frac{x-\ln K+(r+\sigma^2/2)\tau}{\sigma\sqrt{\tau}}$",
    r"Near $\tau\to 0$: $\partial_x V^{\mathrm{BS}}\to e^x H(x-\ln K)$ (step),"
    r"  $\partial_{xx} V^{\mathrm{BS}}\sim e^x\,\delta(x-\ln K)/(\sigma^2\tau)$ (diverges)."
    r"  Dashed line = BS reference.",
])
_FORMULA_WEAK_RES = "\n".join([
    r"$\mathcal{L}_f^{var}(\hat{V},\tau)=\frac{1}{K}\sum_{k=1}^{K}R_k(\tau)^2$,"
    r"  $R_k=\int\left[\partial_t\hat{V}\,\phi_k"
    r"-\frac{\sigma^2}{2}\partial_x\hat{V}\,\partial_x\phi_k"
    r"+\mu\,\partial_x\hat{V}\,\phi_k-r\hat{V}\,\phi_k\right]dx$  (IBP on $\partial_{xx}$)",
    r"$K=20$ sine test functions, GL quadrature — same metric applied to ALL variants."
    r"  Left: strong-form $|\mathcal{F}[\hat{V}]|$ along $x=\ln K$.",
])


# ---------------------------------------------------------------------------
# Variant catalogue  (same grid as original script)
# ---------------------------------------------------------------------------

_EPS_GRID   = [0.005*T, 0.01*T, 0.02*T, 0.05*T, 0.10*T]
_BETA_GRID  = [10, 50, 100, 500, 1000]
_IS_CONFIGS = [(2.0, 0.5), (5.0, 0.5), (10.0, 0.5)]
_COLORS     = ["tab:blue", "tab:orange", "tab:green", "tab:red",
               "tab:purple", "tab:brown", "tab:pink", "tab:gray",
               "tab:olive", "tab:cyan", "steelblue", "coral"]


def _build_variants(mode: str) -> list[dict]:
    naive_cfg = dict(
        name="naive", label="Naïf (control)",
        sampler_type="naive", payoff_type="exact",
        eps=0.0, beta=None, sigma_is=None, mix=0.0,
        color="tab:blue", linestyle="-", linewidth=2.5,
    )

    if mode == "compare-boundary-singularity-european-call":
        return [
            naive_cfg,
            dict(name="truncated", label=r"$\varepsilon$-trunc. ($\varepsilon=1\%T$)",
                 sampler_type="truncated", payoff_type="exact",
                 eps=0.01*T, beta=None, sigma_is=None, mix=0.0,
                 color="tab:orange", linestyle="--", linewidth=2.0),
            dict(name="smooth", label=r"Smooth ($\beta=100$)",
                 sampler_type="naive", payoff_type="smooth",
                 eps=0.0, beta=100, sigma_is=None, mix=0.0,
                 color="tab:green", linestyle="-.", linewidth=2.0),
            dict(name="vpinn", label="VPINN (weak form)",
                 sampler_type="vpinn", payoff_type="exact",
                 eps=0.0, beta=None, sigma_is=None, mix=0.0,
                 # n_tau=512: more temporal coverage (was 128 → 32x undersampled vs n_f=4096)
                 # lam_f=200: VPINN Lf is ~10x smaller than strong-form Lf; rescale so
                 #   PDE contributes ~50% of the gradient (was 24% with lam_f=20)
                 # eps=0.0: no temporal truncation — train on full [0,T] including singularity
                 n_tau=512, K_test=20, n_quad=100, lam_f=200.0,
                 color="tab:red", linestyle=":", linewidth=2.0),
            dict(name="vpinn_50k", label="VPINN — Adam 50k iters (long run)",
                 sampler_type="vpinn", payoff_type="exact",
                 eps=0.0, beta=None, sigma_is=None, mix=0.0,
                 n_tau=512, K_test=20, n_quad=100, lam_f=200.0,
                 # iters_override : force 50 000 iters Adam, ignore --iters et max_iters.
                 # À 20k iters la pente log-log de la loss est ~-2.8 → encore en
                 # descente rapide.  Adam préserve mieux la singularité du γ près de
                 # τ=0 que L-BFGS (effet de moyennage stochastique + pas de
                 # lissage par optimisation précise).
                 iters_override=50000,
                 color="darkred", linestyle=(0, (5, 1, 1, 1, 1, 1)), linewidth=2.5),
            dict(name="vpinn_engd", label="VPINN + ENGD (nat. grad.)",
                 sampler_type="vpinn_engd", payoff_type="exact",
                 eps=0.0, beta=None, sigma_is=None, mix=0.0,
                 n_tau=512, K_test=20, n_quad=100, lam_f=200.0,
                 # max_iters: ENGD steps are ~100× more expensive per iteration;
                 #   1000 natural-gradient steps ≈ 20k Adam steps wall-clock.
                 max_iters=1000,
                 color="tab:purple", linestyle=[0, [3, 2, 1, 2]], linewidth=2.0),
            dict(name="engd", label="Strong-form + ENGD (paper-faithful, lstsq)",
                 sampler_type="engd", payoff_type="exact",
                 eps=0.0, beta=None, sigma_is=None, mix=0.0,
                 # Paper-faithful: small network (129 params), fixed (N-2)^2=784
                 # interior + N-1=29 terminal points, lstsq solve (no Tikhonov).
                 # M/n_params ≈ 6 — same regime as Zeinhofer et al. ICML 2023.
                 n_grid=30,
                 tikhonov_rel=1e-6,
                 max_iters=1000,
                 color="tab:brown", linestyle=[0, [5, 2]], linewidth=2.0),
            # Note: two failed variants explored during diagnostics —
            #   `engd_tc_dense` (N_tc=200 vs 29)        : marginal, same trap
            #   `engd_alt` (alternating G_F / G_TC)     : never reaches J^T r=0
            # Documented in documents/methodology/engd_singularity_diagnostic.md
            # and removed from the catalog.
            dict(name="vpinn_lbfgs", label="VPINN + L-BFGS (stoch. batch)",
                 sampler_type="vpinn_lbfgs", payoff_type="exact",
                 eps=0.0, beta=None, sigma_is=None, mix=0.0,
                 n_tau=512, K_test=20, n_quad=100, lam_f=200.0,
                 # One outer L-BFGS step ≈ 2–3s on GPU → cap at 1000 steps ≈ 15 min on GPU.
                 # Loss à iter 500 = 0.040 et descend encore (|g|=1.7) → 1000 steps offre
                 # ~30% de gain supplémentaire sans plateau visible.
                 max_iters=1000,
                 color="tab:pink", linestyle=[0, [1, 1]], linewidth=2.0),
            dict(name="vpinn_lbfgs_is_tau",
                 label=r"VPINN + L-BFGS (biased $\tau\to 0$ sampling, $\alpha=0.3$)",
                 sampler_type="vpinn_lbfgs_is_tau", payoff_type="exact",
                 eps=0.001 * T, beta=None, sigma_is=None, mix=0.0,
                 n_tau=512, K_test=20, n_quad=100, lam_f=200.0,
                 # τ = T·U^(1/0.3) with U~U(0,1) — concentrates time samples near
                 # the maturity singularity. The estimator is intentionally biased
                 # (no IS correction) so that γ near τ=0 carries more weight.
                 is_tau_alpha=0.3,
                 max_iters=1000,
                 color="tab:cyan", linestyle=[0, [3, 1, 1, 1]], linewidth=2.0),
        ]

    if mode == "ablation-eps":
        variants = [naive_cfg]
        for i, eps in enumerate(_EPS_GRID):
            pct = int(round(eps / T * 100))
            variants.append(dict(
                name=f"trunc_{pct}pct", label=rf"$\varepsilon={pct}\%T$",
                sampler_type="truncated", payoff_type="exact",
                eps=eps, beta=None, sigma_is=None, mix=0.0,
                color=_COLORS[i+1], linestyle="--", linewidth=1.8,
            ))
        return variants

    if mode == "ablation-beta":
        variants = [naive_cfg]
        for i, beta in enumerate(_BETA_GRID):
            variants.append(dict(
                name=f"smooth_b{beta}", label=rf"$\beta={beta}$",
                sampler_type="naive", payoff_type="smooth",
                eps=0.0, beta=beta, sigma_is=None, mix=0.0,
                color=_COLORS[i+1], linestyle="-.", linewidth=1.8,
            ))
        return variants

    if mode == "ablation-is":
        variants = [
            naive_cfg,
            dict(name="trunc_1pct", label=r"Trunc. unif.",
                 sampler_type="truncated", payoff_type="exact",
                 eps=0.01*T, beta=None, sigma_is=None, mix=0.0,
                 color="tab:orange", linestyle="--", linewidth=2.0),
        ]
        for i, (sig, mix) in enumerate(_IS_CONFIGS):
            variants.append(dict(
                name=f"is_sig{int(sig)}_mix{int(mix*100)}",
                label=rf"IS $\sigma_x={sig}$, mix={mix}",
                sampler_type="importance", payoff_type="exact",
                eps=0.01*T, beta=None, sigma_is=sig, mix=mix,
                color=_COLORS[i+2], linestyle=":", linewidth=1.8,
            ))
        return variants

    raise ValueError(f"Unknown mode: {mode!r}")


def _build_sampler(
    cfg: dict,
    n_f: int,
    n_tc: int,
    generator: torch.Generator | None = None,
):
    """Build the per-variant sampling closure, propagating an explicit RNG.

    The ``generator`` argument should be a ``torch.Generator`` created on
    ``p3.DEVICE`` and owned by the caller (typically :func:`_train_one_variant`).
    Passing it here makes the sampler's stochasticity independent from the
    global ``torch`` RNG and lets the caller checkpoint / restore the state
    deterministically across resumes.
    """
    t = cfg["sampler_type"]
    if t in ("naive", "engd"):
        return make_sampler_naive_logS(n_f, n_tc, generator=generator)
    if t == "truncated":
        return make_sampler_truncated_logS(n_f, n_tc, eps=cfg["eps"], generator=generator)
    if t == "importance":
        return make_sampler_importance_logS(n_f, n_tc,
                                            sigma_is=cfg["sigma_is"],
                                            mix=cfg["mix"], eps=cfg["eps"],
                                            generator=generator)
    if t in ("vpinn", "vpinn_engd", "vpinn_lbfgs", "vpinn_lbfgs_epoch"):
        return make_sampler_vpinn_logS(cfg.get("n_tau", 512),
                                       eps=cfg.get("eps", 0.01 * T),
                                       generator=generator)
    if t == "vpinn_lbfgs_is_tau":
        # Biased τ → 0 sampling to better resolve γ near maturity
        return make_sampler_vpinn_logS_is_tau(
            cfg.get("n_tau", 512),
            alpha=cfg.get("is_tau_alpha", 0.3),
            eps=cfg.get("eps", 0.001 * T),
            generator=generator,
        )
    raise ValueError(f"Unknown sampler_type: {t!r}")


def _build_payoff(cfg: dict):
    if cfg["payoff_type"] == "exact":
        return payoff_exact_logS
    if cfg["payoff_type"] == "smooth":
        return make_payoff_smooth_logS(cfg["beta"])
    raise ValueError(f"Unknown payoff_type: {cfg['payoff_type']!r}")


def _build_pinn() -> PINN:
    # No InputNormalization — x = ln(S) is already well-scaled
    return PINN(resnet=ResNet(d_in=2, d_out=1, n=50, M=4, L=2))


def _build_engd_pinn(hidden: int = 32) -> torch.nn.Module:
    """Small MLP [2 → hidden → 1] with Tanh activation — paper-faithful ENGD network.

    Linear(2, hidden) → Tanh → Linear(hidden, 1)
    With hidden=32: 2×32 + 32 + 32×1 + 1 = 129 parameters.
    This matches the architecture in Zeinhofer et al. (ICML 2023) and ensures
    N_int >> n_params (784 >> 129) so that lstsq gives a well-determined solution.
    """
    return torch.nn.Sequential(
        torch.nn.Linear(2, hidden),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden, 1),
    )


def _build_vpinn_loss(cfg: dict) -> _VPINNLossForwardLogS:
    """Build the VPINN loss module from a variant config dict."""
    return _VPINNLossForwardLogS(
        sigma=sigma, r=r,
        x_lo=X_LO, x_hi=X_HI,
        K_test=cfg.get("K_test", 20),
        n_quad=cfg.get("n_quad", 100),
    ).to(p3.DEVICE)


# ---------------------------------------------------------------------------
# Per-variant plots
# ---------------------------------------------------------------------------

_SUPTITLE = (
    rf"European call — PINN, $x=\ln S$, $K={K}$, $r={r}$, $\sigma={sigma}$, $T={T}$"
)


def _plot_variant(res: dict, vdir: Path) -> None:
    out = vdir / "training_metrics"
    out.mkdir(exist_ok=True)
    h, label = res["hist"], res["label"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 7))
    axes[0].semilogy(h["iter"], h["loss"], color="tab:blue")
    axes[0].set_title("Total loss"); axes[0].set_xlabel("Iteration"); axes[0].grid(True, alpha=0.3)
    axes[1].semilogy(h["iter"], h["loss_f"],  color="tab:orange", label=r"$\mathcal{L}_f$")
    axes[1].semilogy(h["iter"], h["loss_tc"], color="tab:red",    label=r"$\mathcal{L}_{tc}$")
    axes[1].set_title("PDE vs TC loss"); axes[1].set_xlabel("Iteration")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[2].semilogy(h["iter"], h["grad_norm"], color="tab:purple")
    axes[2].set_title(r"Gradient norm $\|\nabla_\theta\mathcal{L}\|_2$")
    axes[2].set_xlabel("Iteration"); axes[2].grid(True, alpha=0.3)
    fig.suptitle(f"{label}\n{_SUPTITLE}", fontsize=10)
    fig.tight_layout()
    _vpinn_like = res.get("sampler_type") in ("vpinn", "vpinn_engd", "vpinn_lbfgs", "vpinn_lbfgs_epoch", "vpinn_lbfgs_is_tau")
    lf_formula  = _FORMULA_LF_VPINN if _vpinn_like else _FORMULA_LF
    ltc_formula = _FORMULA_IC_QUAD  if _vpinn_like else _FORMULA_LTC
    _add_formula_box(fig, lf_formula + "\n" + ltc_formula + "\n" + _FORMULA_GRAD,
                     bottom_margin=0.44)
    fig.savefig(out / "training_curves.png", dpi=150)
    plt.close(fig)

    pde = res["metrics"]["pde_residual_tau"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(pde["tau"], pde["residual"], color=res["color"],
                linestyle=res["linestyle"], linewidth=res["linewidth"], marker="o", ms=4)
    ax.axvline(0.0, color="k", linestyle=":", linewidth=0.8)
    ax.set_xlabel(r"$\tau = T - t$")
    ax.set_ylabel(r"$\mathbb{E}_{x=\ln K}[|\mathcal{F}[\hat{V}]|]$")
    ax.set_title(r"PDE residual along $x=\ln K$ (ATM)")
    ax.grid(True, alpha=0.3)
    fig.suptitle(f"{label}\n{_SUPTITLE}", fontsize=10)
    fig.tight_layout()
    _add_formula_box(fig, _FORMULA_PDE_TAU, bottom_margin=0.16)
    fig.savefig(out / "pde_residual_tau.png", dpi=150)
    plt.close(fig)

    # ── Spatial derivative norms vs training ─────────────────────────────
    if res["hist"].get("dx_rms"):
        _colors_probe = ["tab:red", "tab:orange", "tab:blue", "tab:gray"]
        tau_labels = [rf"$\tau={p:.2f}$" for p in _DERIV_TAU_PROBES]
        dx_arr  = np.array(res["hist"]["dx_rms"])   # (n_log, n_probes)
        d2x_arr = np.array(res["hist"]["d2x_rms"])  # (n_log, n_probes)
        iters   = res["hist"]["iter"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        for k in range(dx_arr.shape[1]):
            c = _colors_probe[k % len(_colors_probe)]
            ax1.semilogy(iters, dx_arr[:, k],  label=tau_labels[k], color=c)
            ax2.semilogy(iters, d2x_arr[:, k], label=tau_labels[k], color=c)
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel(r"$\mathrm{RMS}(\partial_x\hat{V})$")
        ax1.set_title(r"Spatial gradient norm $\|\partial_x\hat{V}\|$ per $\tau$")
        ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel(r"$\mathrm{RMS}(\partial_{xx}\hat{V})$")
        ax2.set_title(r"Second derivative norm $\|\partial_{xx}\hat{V}\|$ per $\tau$")
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
        fig.suptitle(f"{label}\n{_SUPTITLE}", fontsize=10)
        fig.tight_layout()
        _add_formula_box(fig, _FORMULA_DX_NORM, bottom_margin=0.20)
        fig.savefig(out / "deriv_norms.png", dpi=150)
        plt.close(fig)

    _plot_gt_per_variant(res, vdir)


def _plot_gt_per_variant(res: dict, vdir: Path) -> None:
    """Price slices, absolute error slices, and greeks vs Black-Scholes for one variant."""
    gt = res.get("gt_slices")
    if gt is None:
        return
    out = vdir / "training_metrics"
    out.mkdir(exist_ok=True)
    label     = res["label"]
    color     = res["color"]
    ls        = res["linestyle"]
    lw        = res["linewidth"]
    tau_slices = gt["tau_slices"]
    S_vals    = np.exp(gt["x_vals"])
    n_tau     = len(tau_slices)

    # ── Price slices ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, n_tau, figsize=(5 * n_tau, 5))
    if n_tau == 1:
        axes = [axes]
    for j, tau_val in enumerate(tau_slices):
        ax = axes[j]
        ax.plot(S_vals, gt["V_ref_slices"][j], "k--", linewidth=1.5,
                label=r"$C^{\mathrm{BS}}$", zorder=10)
        ax.plot(S_vals, gt["V_pred_slices"][j], color=color, linestyle=ls, linewidth=lw,
                label=r"$\hat{V}$")
        ax.axvline(K, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel(r"$S$"); ax.set_ylabel(r"$V$")
        ax.set_title(rf"$\tau = {tau_val:.2f}$")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle(f"{label} — Price vs BS\n{_SUPTITLE}", fontsize=9)
    fig.tight_layout()
    _add_formula_box(fig, _FORMULA_PRICE, bottom_margin=0.20)
    fig.savefig(out / "price_slices.png", dpi=150)
    plt.close(fig)

    # ── Absolute error slices ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, n_tau, figsize=(5 * n_tau, 5))
    if n_tau == 1:
        axes = [axes]
    for j, tau_val in enumerate(tau_slices):
        ax = axes[j]
        err = np.abs(gt["V_pred_slices"][j] - gt["V_ref_slices"][j])
        ax.semilogy(S_vals, err, color=color, linestyle=ls, linewidth=lw)
        ax.axvline(K, color="k", linestyle=":", linewidth=0.8, label=rf"$K={K}$")
        ax.set_xlabel(r"$S$"); ax.set_ylabel(r"$|\hat{V} - C^{\mathrm{BS}}|$")
        ax.set_title(rf"$\tau = {tau_val:.2f}$")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle(f"{label} — Absolute error vs BS\n{_SUPTITLE}", fontsize=9)
    fig.tight_layout()
    _add_formula_box(fig, r"$|\hat{V}(S,\tau) - C^{\mathrm{BS}}(S,\tau)|$  pointwise absolute error",
                     bottom_margin=0.20)
    fig.savefig(out / "price_error_slices.png", dpi=150)
    plt.close(fig)

    # ── Delta & Gamma — one column per tau slice ──────────────────────────
    S_greek    = np.exp(gt["x_greek"])
    tau_slices = gt["tau_slices"]
    n_tau      = len(tau_slices)
    fig, axes = plt.subplots(2, n_tau, figsize=(5 * n_tau, 9))
    if n_tau == 1:
        axes = axes.reshape(2, 1)
    for j, tau_val in enumerate(tau_slices):
        ax_d, ax_g = axes[0, j], axes[1, j]
        ax_d.plot(S_greek, gt["delta_ref_slices"][j],  "k--", linewidth=1.5,
                  label=r"$\Delta^{\mathrm{BS}}$", zorder=10)
        ax_d.plot(S_greek, gt["delta_pred_slices"][j], color=color, linestyle=ls, linewidth=lw,
                  label=r"$\hat{\Delta}$")
        ax_d.axvline(K, color="gray", linestyle=":", linewidth=0.8)
        ax_d.set_ylabel(r"$\Delta$")
        ax_d.set_title(rf"$\tau = {tau_val:.2f}$")
        ax_d.legend(fontsize=8); ax_d.grid(True, alpha=0.3)

        ax_g.plot(S_greek, gt["gamma_ref_slices"][j],  "k--", linewidth=1.5,
                  label=r"$\Gamma^{\mathrm{BS}}$", zorder=10)
        ax_g.plot(S_greek, gt["gamma_pred_slices"][j], color=color, linestyle=ls, linewidth=lw,
                  label=r"$\hat{\Gamma}$")
        ax_g.axvline(K, color="gray", linestyle=":", linewidth=0.8)
        ax_g.set_xlabel(r"$S$"); ax_g.set_ylabel(r"$\Gamma$")
        ax_g.legend(fontsize=8); ax_g.grid(True, alpha=0.3)

    fig.suptitle(f"{label} — Greeks vs BS\n{_SUPTITLE}", fontsize=9)
    fig.tight_layout()
    _add_formula_box(fig, _FORMULA_GREEKS_CMP, bottom_margin=0.12)
    fig.savefig(out / "greeks_comparison.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Comparison plots
# ---------------------------------------------------------------------------

def _plot_comparison(results: list[dict], ablation_dir: Path, iters: int, mode: str):
    comp_dir = ablation_dir / "comparison"
    comp_dir.mkdir(exist_ok=True)
    colors     = [r["color"]     for r in results]
    linestyles = [r["linestyle"] for r in results]
    labels     = [r["label"]     for r in results]
    linewidths = [r["linewidth"] for r in results]

    def _apply_outside_legend(fig):
        """Replace every per-axis legend with one shared figure-level legend.

        Walks *all* axes (not just the first one) and collects every
        labelled artist, deduplicating by label so a variant that appears
        in several axes shows up only once.  This matters for multi-row
        figures where each row plots a disjoint subset of variants — the
        figure-level legend then carries the union of all subsets, not
        just whichever row matplotlib happened to look at first.
        """
        all_handles, all_labels = [], []
        seen = set()
        for ax in fig.axes:
            h, lbl = ax.get_legend_handles_labels()
            for hh, ll in zip(h, lbl):
                if ll not in seen:
                    seen.add(ll)
                    all_handles.append(hh)
                    all_labels.append(ll)
        for ax in fig.axes:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
        if all_handles:
            fig.legend(all_handles, all_labels,
                       loc="center left",
                       bbox_to_anchor=(0.83, 0.5),
                       fontsize=9, frameon=True)
            fig.subplots_adjust(right=0.80)

    def _savefig(fig, name, formula, bottom=0.10, legend_outside=False):
        """Save ``fig`` under ``comp_dir`` with an optional formula box.

        Set ``legend_outside=True`` when several axes share the same set of
        variants — a per-axis legend is then redundant and crowds the plot
        area.  See ``_apply_outside_legend``.
        """
        fig.tight_layout()
        if legend_outside:
            _apply_outside_legend(fig)
        _add_formula_box(fig, formula, bottom_margin=bottom)
        fig.savefig(comp_dir / name, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Loss Lf — split into two rows (strong-form vs weak/VPINN) because the two
    # losses live in different norms and are not directly comparable on a shared
    # axis.  Each row shows only the variants of its formulation; when only one
    # formulation is present, the figure degenerates to a single row.
    _vpinn_like_types = ("vpinn", "vpinn_engd", "vpinn_lbfgs",
                          "vpinn_lbfgs_epoch", "vpinn_lbfgs_is_tau")
    has_vpinn  = any(r.get("sampler_type") in _vpinn_like_types for r in results)
    strong_idx = [i for i, r in enumerate(results)
                  if r.get("sampler_type") not in _vpinn_like_types]
    weak_idx   = [i for i, r in enumerate(results)
                  if r.get("sampler_type") in _vpinn_like_types]

    rows = []
    if strong_idx:
        rows.append((
            strong_idx,
            r"Strong-form PDE residual  $\mathcal{L}_f = \frac{1}{N_f}\sum_i \mathcal{F}[\hat V](x_i,t_i)^2$",
        ))
    if weak_idx:
        rows.append((
            weak_idx,
            r"Weak-form PDE residual (VPINN)  $\mathcal{L}_f^{var} = \frac{1}{N_t K}\sum_{i,k} R_{i,k}^2$",
        ))
    n_rows = max(len(rows), 1)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 5 * n_rows), squeeze=False)
    for row, (idx, title) in enumerate(rows):
        ax = axes[row, 0]
        for i in idx:
            res = results[i]
            ax.semilogy(res["hist"]["iter"], res["hist"]["loss_f"],
                        label=labels[i], color=colors[i],
                        linestyle=linestyles[i], linewidth=linewidths[i])
        ax.set_xlabel("Iteration"); ax.set_ylabel(r"$\mathcal{L}_f$")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.suptitle(
        _SUPTITLE
        + "\nThe two formulations carry DIFFERENT norms and are split into separate rows."
        + "  See pde_residual_by_tau.png for a single-norm strong-form comparison.",
        fontsize=9,
    )
    lf_formula_cmp = "\n".join([
        _FORMULA_LF,
        _FORMULA_LF_VPINN,
        r"Note: the two formulations are NOT directly comparable on a shared axis"
        r" (different norms). pde_residual_by_tau.png evaluates the same strong-form"
        r" residual on every trained model for a fair comparison.",
    ])
    _savefig(fig, "loss_pde.png", lf_formula_cmp, bottom=0.20, legend_outside=True)

    # Gradient norm
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, res in enumerate(results):
        ax.semilogy(res["hist"]["iter"], res["hist"]["grad_norm"],
                    label=labels[i], color=colors[i],
                    linestyle=linestyles[i], linewidth=linewidths[i], alpha=0.8)
    ax.set_xlabel("Iteration"); ax.set_ylabel(r"$\|\nabla_\theta \mathcal{L}\|_2$")
    ax.set_title("Gradient norm — singularity instability signature")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.suptitle(_SUPTITLE, fontsize=10)
    _savefig(fig, "grad_norm.png", _FORMULA_GRAD, legend_outside=True)

    # PDE residual profile — FAIR comparison (strong-form residual, post-training)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, res in enumerate(results):
        pde = res["metrics"]["pde_residual_tau"]
        ax.semilogy(pde["tau"], pde["residual"],
                    label=labels[i], color=colors[i],
                    linestyle=linestyles[i], linewidth=linewidths[i], marker="o", ms=3)
    ax.axvline(0.0, color="k", linestyle=":", linewidth=0.8, label=r"$\tau=0$ (singular)")
    ax.set_xlabel(r"$\tau = T - t$")
    ax.set_ylabel(r"$\mathbb{E}_{x=\ln K}[|\mathcal{F}[\hat{V}]|]$")
    pde_title = r"Mean strong-form PDE residual along $x=\ln K$ vs $\tau$"
    if has_vpinn:
        pde_title += "\n[Fair comparison — strong-form $\\mathcal{F}[\\hat{V}]$ evaluated post-training for all variants]"
    ax.set_title(pde_title, fontsize=9)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.suptitle(_SUPTITLE, fontsize=10)
    _savefig(fig, "pde_residual_by_tau.png", _FORMULA_PDE_TAU, legend_outside=True)

    # Fair-vs-unfair overview (side-by-side panel for quick reference)
    if has_vpinn:
        fig2, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 6))
        for i, res in enumerate(results):
            ax_l.semilogy(res["hist"]["iter"], res["hist"]["loss_f"],
                          label=labels[i], color=colors[i],
                          linestyle=linestyles[i], linewidth=linewidths[i])
        ax_l.set_xlabel("Iteration"); ax_l.set_ylabel(r"$\mathcal{L}_f$")
        ax_l.set_title(
            r"Training loss $\mathcal{L}_f$" + "\n"
            r"[!] VPINN uses integrated weak-form residual — different norm",
            fontsize=9,
        )
        ax_l.legend(fontsize=9); ax_l.grid(True, alpha=0.3)
        ax_l.text(0.98, 0.98, "Unfair to compare\nacross methods",
                  transform=ax_l.transAxes, ha="right", va="top", fontsize=8,
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffeeba", edgecolor="#e6a817", alpha=0.9))
        for i, res in enumerate(results):
            pde = res["metrics"]["pde_residual_tau"]
            ax_r.semilogy(pde["tau"], pde["residual"],
                          label=labels[i], color=colors[i],
                          linestyle=linestyles[i], linewidth=linewidths[i], marker="o", ms=3)
        ax_r.axvline(0.0, color="k", linestyle=":", linewidth=0.8, label=r"$\tau=0$ (singular)")
        ax_r.set_xlabel(r"$\tau = T - t$")
        ax_r.set_ylabel(r"$\mathbb{E}_{x=\ln K}[|\mathcal{F}[\hat{V}]|]$")
        ax_r.set_title(
            r"Strong-form PDE residual (post-training)" + "\n"
            r"Same $\mathcal{F}[\hat{V}]$ operator applied to all variants",
            fontsize=9,
        )
        ax_r.legend(fontsize=9); ax_r.grid(True, alpha=0.3)
        ax_r.text(0.98, 0.98, "Fair comparison\n(same metric for all)",
                  transform=ax_r.transAxes, ha="right", va="top", fontsize=8,
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="#d4edda", edgecolor="#28a745", alpha=0.9))
        fig2.suptitle(
            "Training loss vs post-training PDE residual — fair/unfair comparison  |  " + _SUPTITLE,
            fontsize=9,
        )
        fig2.tight_layout()
        _apply_outside_legend(fig2)
        _add_formula_box(fig2,
            "\n".join([
                r"Left: $\mathcal{L}_f$ during training — "
                r"VPINN uses weak norm $\Vert R \Vert_{L^2_t \times \ell^2_k}$, "
                r"strong-form uses pointwise $\Vert \mathcal{F}[\hat{V}] \Vert_{L^2_{x,t}}$",
                r"Right (fair): $\bar{F}(\tau)=\frac{1}{N}\sum_i|\mathcal{F}[\hat{V}](x=\ln K,\,T-\tau)|$"
                r" — same strong-form operator for all methods",
            ]),
            bottom_margin=0.14,
        )
        fig2.savefig(comp_dir / "fair_comparison_overview.png", dpi=150, bbox_inches="tight")
        plt.close(fig2)

    # Metric bar chart  (3 rows: global metrics / εΔ per τ / εΓ per τ)
    # Three representative tau slices: near-singularity, mid, full maturity
    _MB_TAU_IDX  = [0, 2, 4]   # indices into _GT_TAU_SLICES: 0.02T, T/2, T
    _mb_tau_vals = [_GT_TAU_SLICES[i] for i in _MB_TAU_IDX]

    metric_keys  = ["rel_l2", "rel_l2_atm", "gei"]
    metric_names = [
        r"$\varepsilon_{L^2}$",
        r"$\varepsilon_{L^2}^{\mathrm{ATM}}$",
        r"GEI",
    ]
    fig, axes = plt.subplots(3, 3, figsize=(13, 14))

    # Row 0: global scalar metrics
    for j, (mk, mn) in enumerate(zip(metric_keys, metric_names)):
        vals = [res["metrics"][mk] for res in results]
        bars = axes[0, j].bar(range(len(results)), vals, color=colors)
        axes[0, j].set_xticks(range(len(results)))
        axes[0, j].set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        axes[0, j].set_title(mn, fontsize=10)
        axes[0, j].set_yscale("log")
        axes[0, j].grid(axis="y", alpha=0.3)
        for br, val in zip(bars, vals):
            axes[0, j].text(br.get_x() + br.get_width()/2, val * 1.1,
                            f"{val:.2e}", ha="center", va="bottom", fontsize=7)

    # Rows 1–2: εΔ and εΓ at three representative τ values (one col per τ)
    for row_idx, (pred_key, ref_key, ylbl) in enumerate([
        ("delta_pred_slices", "delta_ref_slices", r"$\varepsilon_{\Delta}(\tau)$"),
        ("gamma_pred_slices", "gamma_ref_slices", r"$\varepsilon_{\Gamma}(\tau)$"),
    ], start=1):
        for k, (i_tau, tau_v) in enumerate(zip(_MB_TAU_IDX, _mb_tau_vals)):
            ax = axes[row_idx, k]
            vals = []
            for res in results:
                gs = res.get("gt_slices")
                if gs is not None:
                    gp = gs[pred_key][i_tau]
                    gr = gs[ref_key][i_tau]
                    eps = float(np.linalg.norm(gp - gr) / (np.linalg.norm(gr) + 1e-12))
                else:
                    eps = float("nan")
                vals.append(eps)
            bars = ax.bar(range(len(results)), vals, color=colors)
            ax.set_xticks(range(len(results)))
            ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
            ax.set_title(rf"{ylbl}    $\tau={tau_v:.2f}$", fontsize=10)
            ax.set_yscale("log")
            ax.grid(axis="y", alpha=0.3)
            for br, val in zip(bars, vals):
                if not np.isnan(val):
                    ax.text(br.get_x() + br.get_width()/2, val * 1.1,
                            f"{val:.2e}", ha="center", va="bottom", fontsize=7)

    fig.suptitle(f"Metric comparison — mode={mode}, {iters} iters\n{_SUPTITLE}", fontsize=10)
    _formula_metrics_bar = "\n".join([
        r"$\varepsilon_{L^2}=\|\hat{V}-C^{\mathrm{BS}}\|_2/\|C^{\mathrm{BS}}\|_2$"
        r"   (grid $x\in[\ln 60,\ln 140]$, $t\in[0,\,T-0.01]$)",
        r"$\varepsilon_{L^2}^{\mathrm{ATM}}$: same restricted to $x\in[\ln(0.9K),\ln(1.1K)]$     "
        r"$\mathrm{GEI}=\max\|\nabla_\theta\mathcal{L}\|/\mathrm{median}\|\nabla_\theta\mathcal{L}\|$"
        r"   (first 2/3 of training)",
        r"Rows 1–2: $\varepsilon_\Delta(\tau)$, $\varepsilon_\Gamma(\tau)$ — rel. $L^2$ error"
        r" over $S\in[60,140]$ at $\tau\in\{"
        + ", ".join(f"{v:.2f}" for v in _mb_tau_vals)
        + r"\}$ (near-singularity / mid / full maturity)",
        r"$\hat{\Delta}=e^{-x}\partial_x\hat{V}$,  $\hat{\Gamma}=e^{-2x}(\partial_{xx}\hat{V}-\partial_x\hat{V})$"
        r"     $C^{\mathrm{BS}}=S-Ke^{-r\tau}+P^{\mathrm{BS}}$,  $d_1=(x-\ln K+(r+\sigma^2/2)\tau)/(\sigma\sqrt{\tau})$",
    ])
    fig.subplots_adjust(bottom=0.30, top=0.92, wspace=0.40, hspace=0.85)
    fig.text(0.5, 0.01, _formula_metrics_bar, ha="center", va="bottom",
             fontsize=7.5, bbox=_BOX_STYLE)
    fig.savefig(comp_dir / "metrics_bar.png", dpi=150)
    plt.close(fig)

    # Payoff comparison — exact vs smooth softplus (analytical, no model needed)
    smooth_results = [r for r in results
                      if r.get("payoff_type") == "smooth" and r.get("beta") is not None]
    if smooth_results:
        with torch.no_grad():
            x_fine    = torch.linspace(X_EVAL_LO, X_EVAL_HI, 600)
            phi_exact = payoff_exact_logS(x_fine).numpy()
        S_fine = x_fine.exp().numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ax1.plot(S_fine, phi_exact, color="k", linewidth=2.5,
                 label=r"Exact: $(S-K)^+$", zorder=10)
        for sv in smooth_results:
            beta = sv["beta"]
            with torch.no_grad():
                phi_sm = make_payoff_smooth_logS(beta)(x_fine).numpy()
            ax1.plot(S_fine, phi_sm, color=sv["color"], linestyle=sv["linestyle"],
                     linewidth=sv["linewidth"],
                     label=rf"$\tilde{{\Phi}}_{{\beta={beta}}}$  ({sv['label']})")
        ax1.axvline(K, color="gray", linestyle=":", linewidth=1.0, label=rf"$K={K:.0f}$ (ATM)")
        ax1.set_xlabel(r"$S = e^x$"); ax1.set_ylabel(r"Payoff $\Phi(x)$")
        ax1.set_title("Terminal condition — exact vs smooth payoff")
        ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

        ax2.set_title(r"Smoothing error $|\tilde{\Phi}_\beta(x) - (S-K)^+|$")
        for sv in smooth_results:
            beta = sv["beta"]
            with torch.no_grad():
                phi_sm = make_payoff_smooth_logS(beta)(x_fine).numpy()
            err = np.abs(phi_sm - phi_exact)
            ax2.plot(S_fine, err, color=sv["color"], linestyle=sv["linestyle"],
                     linewidth=sv["linewidth"],
                     label=rf"$\beta={beta}$,  max$={err.max():.2e}$")
        ax2.axvline(K, color="gray", linestyle=":", linewidth=1.0, label=rf"$K={K:.0f}$ (ATM)")
        ax2.set_xlabel(r"$S = e^x$"); ax2.set_ylabel("Absolute error")
        ax2.set_yscale("log")
        ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

        fig.suptitle(f"Payoff smoothing  |  {_SUPTITLE}", fontsize=10)
        _formula_payoff = "\n".join([
            r"Exact:  $\Phi(x) = (e^x-K)^+$   — discontinuous slope at $x=\ln K$"
            r" (source of the terminal-condition singularity)",
            r"Smooth: $\tilde{\Phi}_\beta(x) = \frac{1}{\beta}\ln(1+e^{\beta(e^x-K)})"
            r" - \frac{\ln 2}{\beta}$   (softplus, centré en $\tilde{\Phi}_\beta(\ln K)=0$)",
            r"Max error: $\max_x|\tilde{\Phi}_\beta-\Phi| = \frac{\ln 2}{\beta}$"
            r"   atteint en $x=\ln K$ (ATM)",
        ])
        _savefig(fig, "payoff_comparison.png", _formula_payoff, bottom=0.20)

    # Terminal-condition loss — split into two rows like loss_pde.png because the
    # strong-form variants use a Monte-Carlo MSE at randomly sampled (x_i, T)
    # points while the VPINN variants use a deterministic Gauss-Legendre
    # quadrature.  Both estimate the same L²(Ω) norm of the payoff-fit error,
    # but the discrete values differ (MC noise vs deterministic rule) so we
    # avoid sitting them on a shared axis.
    rows_tc = []
    if strong_idx:
        rows_tc.append((
            strong_idx,
            r"Strong-form terminal-condition loss  "
            r"$\mathcal{L}_{tc} = \frac{1}{N_{tc}}\sum_i (\hat V(x_i,T) - \Phi(x_i))^2$"
            r"   (Monte-Carlo)",
        ))
    if weak_idx:
        rows_tc.append((
            weak_idx,
            r"Variational terminal-condition loss  "
            r"$\mathcal{L}_{ic}^{var} = \frac{1}{|\Omega|}\sum_q w_q |\hat u(T,x_q)-h(x_q)|^2$"
            r"   (Gauss-Legendre quadrature)",
        ))
    n_rows_tc = max(len(rows_tc), 1)
    fig, axes = plt.subplots(n_rows_tc, 1, figsize=(10, 5 * n_rows_tc), squeeze=False)
    for row, (idx, title) in enumerate(rows_tc):
        ax = axes[row, 0]
        for i in idx:
            res = results[i]
            ax.semilogy(res["hist"]["iter"], res["hist"]["loss_tc"],
                        label=labels[i], color=colors[i],
                        linestyle=linestyles[i], linewidth=linewidths[i])
        ax.set_xlabel("Iteration"); ax.set_ylabel(r"$\mathcal{L}_{tc}$")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.suptitle(
        _SUPTITLE
        + "\nMC and quadrature estimators of the same L²(Ω) payoff-fit error"
        + " — split into separate rows to keep the comparison fair within each estimator.",
        fontsize=9,
    )
    _savefig(fig, "loss_tc.png", _FORMULA_LTC + "\n" + _FORMULA_IC_QUAD, bottom=0.22,
             legend_outside=True)

    # Derivative norm comparison — aggregated grid: 2 rows (∂_x, ∂_xx) × N_probes cols
    valid_dx = [r for r in results if r["hist"].get("dx_rms")]
    if valid_dx:
        n_probes = len(_DERIV_TAU_PROBES)
        fig, axes = plt.subplots(2, n_probes, figsize=(4.5 * n_probes, 9),
                                 sharex=True)
        for k, tau_val in enumerate(_DERIV_TAU_PROBES):
            ax_dx, ax_d2x = axes[0, k], axes[1, k]
            for res in valid_dx:
                dx_arr  = np.array(res["hist"]["dx_rms"])
                d2x_arr = np.array(res["hist"]["d2x_rms"])
                kw = dict(label=res["label"], color=res["color"],
                          linestyle=res["linestyle"], linewidth=res["linewidth"])
                ax_dx.semilogy(res["hist"]["iter"],  dx_arr[:, k],  **kw)
                ax_d2x.semilogy(res["hist"]["iter"], d2x_arr[:, k], **kw)
            tau_str = rf"$\tau={tau_val:.2f}$"
            ax_dx.set_title(tau_str, fontsize=10)
            ax_dx.grid(True, alpha=0.3)
            ax_d2x.set_xlabel("Iteration"); ax_d2x.grid(True, alpha=0.3)
            if k == 0:
                ax_dx.set_ylabel(r"$\mathrm{RMS}(\partial_x\hat{V})$")
                ax_d2x.set_ylabel(r"$\mathrm{RMS}(\partial_{xx}\hat{V})$")
            if k == n_probes - 1:
                ax_dx.legend(fontsize=8)
                ax_d2x.legend(fontsize=8)
        # Add BS reference lines (analytical RMS values, precomputed)
        try:
            bs_dx  = [74.3, 72.9, 69.5, 65.2]   # precomputed RMS(∂_x V^BS)
            bs_d2x = [387., 277., 211., 169.]     # precomputed RMS(∂_xx V^BS)
            for k in range(n_probes):
                axes[0, k].axhline(bs_dx[k],  color="k", linestyle=":", lw=1.0, alpha=0.6)
                axes[1, k].axhline(bs_d2x[k], color="k", linestyle=":", lw=1.0, alpha=0.6)
            axes[0, n_probes - 1].plot([], [], color="k", linestyle=":", lw=1.0,
                                       alpha=0.6, label=r"BS ref")
            axes[0, n_probes - 1].legend(fontsize=8)
        except Exception:
            pass
        fig.suptitle(
            r"Spatial derivative norms vs training — all variants  |  " + _SUPTITLE,
            fontsize=9,
        )
        fig.tight_layout()
        _apply_outside_legend(fig)
        _add_formula_box(fig, _FORMULA_DX_NORM, bottom_margin=0.10)
        fig.savefig(comp_dir / "deriv_norms_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Spatial derivative distribution — ∂_x V̂(x) and ∂_xx V̂(x) vs x at τ probes
    valid_sp = [r for r in results
                if r.get("gt_slices") and "dx_pred_spatial" in (r["gt_slices"] or {})]
    if valid_sp:
        n_probes = len(_DERIV_TAU_PROBES)
        fig, axes = plt.subplots(2, n_probes, figsize=(4.5 * n_probes, 9))
        x_sp = valid_sp[0]["gt_slices"]["x_deriv_spatial"]
        for k, tau_val in enumerate(_DERIV_TAU_PROBES):
            ax_dx  = axes[0, k]
            ax_d2x = axes[1, k]
            # BS reference (same for all variants)
            ax_dx.plot(x_sp, valid_sp[0]["gt_slices"]["dx_ref_spatial"][k],
                       "k--", lw=1.5, label=r"$\partial_x V^{\mathrm{BS}}$", zorder=10)
            ax_d2x.plot(x_sp, valid_sp[0]["gt_slices"]["d2x_ref_spatial"][k],
                        "k--", lw=1.5, label=r"$\partial_{xx} V^{\mathrm{BS}}$", zorder=10)
            for res in valid_sp:
                gt = res["gt_slices"]
                kw = dict(label=res["label"], color=res["color"],
                          linestyle=res["linestyle"], linewidth=res["linewidth"])
                ax_dx.plot(x_sp,  gt["dx_pred_spatial"][k],  **kw)
                ax_d2x.plot(x_sp, gt["d2x_pred_spatial"][k], **kw)
            ax_dx.axvline(math.log(K), color="gray", linestyle=":", lw=0.8)
            ax_d2x.axvline(math.log(K), color="gray", linestyle=":", lw=0.8)
            ax_dx.set_title(rf"$\tau={tau_val:.2f}$", fontsize=10)
            ax_dx.grid(True, alpha=0.3)
            ax_d2x.set_xlabel(r"$x = \ln S$"); ax_d2x.grid(True, alpha=0.3)
            if k == 0:
                ax_dx.set_ylabel(r"$\partial_x \hat{V}$")
                ax_d2x.set_ylabel(r"$\partial_{xx} \hat{V}$")
            if k == n_probes - 1:
                ax_dx.legend(fontsize=8)
                ax_d2x.legend(fontsize=8)
        fig.suptitle(
            r"Spatial derivative distribution — all variants  |  " + _SUPTITLE,
            fontsize=9,
        )
        fig.tight_layout()
        _apply_outside_legend(fig)
        _add_formula_box(fig, _FORMULA_DERIV_SPATIAL, bottom_margin=0.13)
        fig.savefig(comp_dir / "deriv_spatial_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Weak-form residual profile — fair comparison for all variants
    valid_wr = [r for r in results
                if r.get("gt_slices") and "weak_residual" in (r["gt_slices"] or {})]
    if valid_wr:
        fig, (ax_strong, ax_weak) = plt.subplots(1, 2, figsize=(12, 5))
        for res in valid_wr:
            kw = dict(label=res["label"], color=res["color"],
                      linestyle=res["linestyle"], linewidth=res["linewidth"],
                      marker="o", markersize=4)
            pde = res["metrics"]["pde_residual_tau"]
            ax_strong.semilogy(pde["tau"], pde["residual"], **kw)
            gt = res["gt_slices"]
            ax_weak.semilogy(gt["weak_residual_tau"], gt["weak_residual"], **kw)
        for ax, title in [
            (ax_strong, r"Strong-form residual $|\mathcal{F}[\hat{V}]|$ along $x=\ln K$"),
            (ax_weak,   r"Weak-form residual $\mathcal{L}_f^{var}(\hat{V},\tau)$  [all variants, same metric]"),
        ]:
            ax.set_xlabel(r"$\tau$"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
            ax.set_title(title, fontsize=9)
        ax_strong.set_ylabel(r"$|\mathcal{F}[\hat{V}]|$")
        ax_weak.set_ylabel(r"$\mathcal{L}_f^{var}$")
        fig.suptitle(r"Residual comparison — strong vs weak form  |  " + _SUPTITLE, fontsize=9)
        fig.tight_layout()
        _apply_outside_legend(fig)
        _add_formula_box(fig, _FORMULA_WEAK_RES, bottom_margin=0.14)
        fig.savefig(comp_dir / "weak_residual_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # L-BFGS comparison: stochastic batch vs epoch-based batch (when both are present)
    lbfgs_stoch = next((r for r in results if r.get("sampler_type") == "vpinn_lbfgs"), None)
    lbfgs_epoch = next((r for r in results if r.get("sampler_type") == "vpinn_lbfgs_epoch"), None)
    if lbfgs_stoch is not None and lbfgs_epoch is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for res, ls_label in [
            (lbfgs_stoch, "Stochastic L-BFGS\n(fresh batch every step)"),
            (lbfgs_epoch, "Epoch L-BFGS\n(same batch for 20 steps, then resample)"),
        ]:
            kw = dict(color=res["color"], linestyle=res["linestyle"],
                      linewidth=res["linewidth"], label=ls_label)
            axes[0].semilogy(res["hist"]["iter"], res["hist"]["loss"],      **kw)
            axes[1].semilogy(res["hist"]["iter"], res["hist"]["loss_f"],    **kw)
            axes[2].semilogy(res["hist"]["iter"], res["hist"]["grad_norm"], **kw)
        for ax, title in [
            (axes[0], "Total loss"),
            (axes[1], r"PDE (weak-form) loss $\mathcal{L}_f$"),
            (axes[2], r"Gradient norm $\|\nabla_\theta\mathcal{L}\|_2$"),
        ]:
            ax.set_xlabel("Outer L-BFGS step"); ax.set_title(title, fontsize=9)
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        m_s = lbfgs_stoch["metrics"]; m_e = lbfgs_epoch["metrics"]
        fig.suptitle(
            f"L-BFGS: stochastic vs epoch-based batch  |  {_SUPTITLE}\n"
            f"Stochastic: rel_L2={m_s['rel_l2']:.3e}  |  "
            f"Epoch: rel_L2={m_e['rel_l2']:.3e}",
            fontsize=9,
        )
        fig.tight_layout()
        _apply_outside_legend(fig)
        _add_formula_box(fig,
            r"Stochastic: $t_{\rm batch}\sim U(0,T)$ at every step "
            r"— $y_k=\nabla f_{B_{k+1}}(x_{k+1})-\nabla f_{B_k}(x_k)$ mixes two objectives "
            r"(curvature noise, NaN instabilities)."
            "\n"
            r"Epoch-based: the same $t_{\rm batch}$ is kept for $N=20$ steps "
            r"— $y_k=\nabla f_B(x_{k+1})-\nabla f_B(x_k)$ is a true curvature estimate. "
            r"The L-BFGS history is cleared at each new epoch to avoid stale (s,y) pairs.",
            bottom_margin=0.16,
        )
        fig.savefig(comp_dir / "lbfgs_stoch_vs_epoch.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("L-BFGS comparison plot saved → lbfgs_stoch_vs_epoch.png")

    _plot_gt_comparison(results, comp_dir)
    logger.info(f"Comparison plots saved to {comp_dir}/")


def _plot_gt_comparison(results: list[dict], comp_dir: Path) -> None:
    """Cross-variant price slices, error slices, and greeks vs Black-Scholes."""
    valid = [r for r in results if r.get("gt_slices") is not None]
    if not valid:
        return

    tau_slices = valid[0]["gt_slices"]["tau_slices"]
    S_vals     = np.exp(valid[0]["gt_slices"]["x_vals"])
    n_tau      = len(tau_slices)

    def _outside_legend(fig):
        """Same as the helper in _plot_comparison — replace per-axis legends
        with one figure-level legend on the right.  Collects labels from
        every axis (deduplicated) so multi-axis figures still show every
        variant in the shared legend.
        """
        all_handles, all_labels = [], []
        seen = set()
        for ax in fig.axes:
            h, lbl = ax.get_legend_handles_labels()
            for hh, ll in zip(h, lbl):
                if ll not in seen:
                    seen.add(ll)
                    all_handles.append(hh)
                    all_labels.append(ll)
        for ax in fig.axes:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
        if all_handles:
            fig.legend(all_handles, all_labels,
                       loc="center left",
                       bbox_to_anchor=(0.83, 0.5),
                       fontsize=9, frameon=True)
            fig.subplots_adjust(right=0.80)

    def _save(fig, name, formula, bottom=0.12, legend_outside=True):
        fig.tight_layout()
        if legend_outside:
            _outside_legend(fig)
        _add_formula_box(fig, formula, bottom_margin=bottom)
        fig.savefig(comp_dir / name, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── Price comparison ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, n_tau, figsize=(5 * n_tau, 5))
    if n_tau == 1:
        axes = [axes]
    for j, tau_val in enumerate(tau_slices):
        ax = axes[j]
        ax.plot(S_vals, valid[0]["gt_slices"]["V_ref_slices"][j],
                "k--", linewidth=1.5, label=r"$C^{\mathrm{BS}}$", zorder=10)
        for r in valid:
            ax.plot(S_vals, r["gt_slices"]["V_pred_slices"][j],
                    color=r["color"], linestyle=r["linestyle"], linewidth=r["linewidth"],
                    label=r["label"])
        ax.axvline(K, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel(r"$S$"); ax.set_ylabel(r"$V$")
        ax.set_title(rf"$\tau = {tau_val:.2f}$")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle(f"Price vs BS — all variants  |  {_SUPTITLE}", fontsize=9)
    _save(fig, "price_slices.png", _FORMULA_PRICE)

    # ── Error comparison ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, n_tau, figsize=(5 * n_tau, 5))
    if n_tau == 1:
        axes = [axes]
    for j, tau_val in enumerate(tau_slices):
        ax = axes[j]
        for r in valid:
            err = np.abs(r["gt_slices"]["V_pred_slices"][j] - r["gt_slices"]["V_ref_slices"][j])
            ax.semilogy(S_vals, err,
                        color=r["color"], linestyle=r["linestyle"], linewidth=r["linewidth"],
                        label=r["label"])
        ax.axvline(K, color="k", linestyle=":", linewidth=0.8, label=rf"$K={K}$")
        ax.set_xlabel(r"$S$"); ax.set_ylabel(r"$|\hat{V} - C^{\mathrm{BS}}|$")
        ax.set_title(rf"$\tau = {tau_val:.2f}$")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle(f"Absolute error vs BS — all variants  |  {_SUPTITLE}", fontsize=9)
    _save(fig, "price_error_slices.png",
          r"$|\hat{V}(S,\tau) - C^{\mathrm{BS}}(S,\tau)|$  pointwise absolute error")

    # ── Greeks comparison — one column per tau slice ──────────────────────
    S_greek    = np.exp(valid[0]["gt_slices"]["x_greek"])
    tau_slices_g = valid[0]["gt_slices"]["tau_slices"]
    n_tau_g    = len(tau_slices_g)
    fig, axes = plt.subplots(2, n_tau_g, figsize=(5 * n_tau_g, 9))
    if n_tau_g == 1:
        axes = axes.reshape(2, 1)
    for j, tau_val in enumerate(tau_slices_g):
        ax_d, ax_g = axes[0, j], axes[1, j]
        ax_d.plot(S_greek, valid[0]["gt_slices"]["delta_ref_slices"][j],
                  "k--", linewidth=1.5, label=r"$\Delta^{\mathrm{BS}}$", zorder=10)
        ax_g.plot(S_greek, valid[0]["gt_slices"]["gamma_ref_slices"][j],
                  "k--", linewidth=1.5, label=r"$\Gamma^{\mathrm{BS}}$", zorder=10)
        for rv in valid:
            ax_d.plot(S_greek, rv["gt_slices"]["delta_pred_slices"][j],
                      color=rv["color"], linestyle=rv["linestyle"], linewidth=rv["linewidth"],
                      label=rv["label"])
            ax_g.plot(S_greek, rv["gt_slices"]["gamma_pred_slices"][j],
                      color=rv["color"], linestyle=rv["linestyle"], linewidth=rv["linewidth"],
                      label=rv["label"])
        ax_d.axvline(K, color="gray", linestyle=":", linewidth=0.8)
        ax_g.axvline(K, color="gray", linestyle=":", linewidth=0.8)
        ax_d.set_ylabel(r"$\Delta$")
        ax_d.set_title(rf"$\tau = {tau_val:.2f}$")
        ax_d.legend(fontsize=7); ax_d.grid(True, alpha=0.3)
        ax_g.set_xlabel(r"$S$"); ax_g.set_ylabel(r"$\Gamma$")
        ax_g.legend(fontsize=7); ax_g.grid(True, alpha=0.3)
    fig.suptitle(f"Greeks vs BS — all variants  |  {_SUPTITLE}", fontsize=9)
    _save(fig, "greeks_comparison.png", _FORMULA_GREEKS_CMP, bottom=0.12)


# ---------------------------------------------------------------------------
# Replot
# ---------------------------------------------------------------------------

def _build_model_for_variant(v: dict) -> torch.nn.Module:
    """Instantiate the correct architecture for a variant config."""
    if v["sampler_type"] == "engd":
        return _build_engd_pinn(hidden=v.get("hidden", 32))
    return _build_pinn()


def _load_model_for_variant(ablation_dir: Path, variant_name: str,
                             v: dict | None = None) -> torch.nn.Module | None:
    """Load a saved model for a variant; return None if missing OR architecture mismatch.

    Tolerant aux modèles obsolètes laissés par d'anciens runs (architecture
    Sequential vs PINN actuelle) — émet juste un avertissement et continue,
    plutôt que de crasher la régénération de tous les graphes.
    """
    model_path = ablation_dir / f"variant_{variant_name}" / "models" / "pinn.pt"
    if not model_path.exists():
        return None
    state = torch.load(model_path, map_location=p3.DEVICE, weights_only=True)
    # Try the current architecture first; fall back to legacy engd_pinn if needed
    candidates = [_build_pinn] if not variant_name.startswith("engd") else [_build_engd_pinn, _build_pinn]
    if variant_name.startswith("engd"):
        candidates = [_build_engd_pinn, _build_pinn]
    for build_fn in candidates:
        model = build_fn()
        try:
            model.load_state_dict(state)
            model.to(p3.DEVICE)
            return model
        except RuntimeError:
            continue
    logger.warning(
        f"⚠ Modèle pinn.pt de '{variant_name}' incompatible avec les architectures "
        f"connues — variant ignoré pour le recalcul des GT slices (probablement "
        f"un dossier orphelin d'un run antérieur)."
    )
    return None


def _replot(ablation_dir: Path) -> None:
    with open(ablation_dir / "summary.yaml") as f:
        summary = yaml.safe_load(f)
    with open(ablation_dir / "metadata.yaml") as f:
        meta = yaml.safe_load(f)
    results = [_load_variant(ablation_dir / f"variant_{e['name']}", e)
               for e in summary["variants"]]

    # Always recompute GT slices from the saved model so that changes to
    # _GT_TAU_SLICES (e.g. adding a near-singularity slice) are picked up.
    for res, entry in zip(results, summary["variants"]):
        model = _load_model_for_variant(ablation_dir, entry["name"])
        if model is not None:
            gt_slices = _compute_gt_slices(model)
            res["gt_slices"] = gt_slices
            np.savez_compressed(
                ablation_dir / f"variant_{entry['name']}" / "gt_comparison.npz",
                **gt_slices,
            )
            logger.info(f"GT slices recomputed for variant {entry['name']}")
        else:
            logger.warning(f"No model found for variant {entry['name']} — GT plots skipped")

    for res, entry in zip(results, summary["variants"]):
        _plot_variant(res, ablation_dir / f"variant_{entry['name']}")
    _plot_comparison(results, ablation_dir, meta["iters"], meta["mode"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_INIT_SEED_OFFSET    = 0x5EED_1117   # arbitrary disambiguation between roles
_SAMPLER_SEED_OFFSET = 0x5A11_9100


def _variant_seed(variant_name: str, salt: int) -> int:
    """Deterministic 64-bit seed derived from a variant name and a role salt.

    Built from a stable hash so that two runs with the same variant produce
    the same initial model weights and the same sampler trajectory, while
    different variants are decorrelated by construction.
    """
    import hashlib
    h = hashlib.blake2b(variant_name.encode("utf-8"), digest_size=8).digest()
    raw = int.from_bytes(h, "big") ^ (salt & ((1 << 64) - 1))
    # torch.Generator.manual_seed expects a positive 64-bit int
    return raw & ((1 << 63) - 1)


def _train_one_variant(
    v: dict,
    vdir: Path,
    n_f: int,
    n_tc: int,
    total_iters: int,
    resume: bool = False,
) -> dict:
    """Build, train, evaluate, and save one variant; return the result dict.

    Random-number-generator hygiene
    -------------------------------
    Two roles need decorrelated, deterministic RNGs:

    1. *Model initialisation* — done once before training.  Currently the
       PyTorch ``nn.Module`` parameter init helpers (Kaiming, etc.) all read
       from the *global* RNG, so we set ``torch.manual_seed(...)`` once
       deterministically right before ``_build_model_for_variant(v)``.  On
       resume the model weights are loaded from the checkpoint and this
       seed has no observable effect — it is only useful for reproducing
       a fresh run from scratch.

    2. *Stochastic sampling during training* — happens at every step.  We
       create a dedicated ``torch.Generator`` on ``p3.DEVICE`` and propagate
       it explicitly to the sampler closure via :func:`_build_sampler` and
       to the training function (for checkpoint save/restore).  This is the
       modern best practice: it avoids relying on hidden global state,
       which is brittle (cross-library pollution, non-thread-safe) and
       opaque (no inspectable seed).

    The two seeds are derived from the variant name through different salts
    so different variants — and the two roles within a variant — are
    decorrelated.
    """
    # Three possible semantics for the effective iteration count:
    #   - iters_override: forces this exact value, ignoring --iters and max_iters
    #     → for variants that need more iters than --iters provides
    #       (e.g. Adam 50k to better resolve a singularity).
    #   - max_iters: upper cap, take min(max_iters, total_iters)
    #     → variants expensive per step that must stay under total_iters
    #       (ENGD, L-BFGS).
    #   - otherwise: follow total_iters (--iters).
    if "iters_override" in v:
        effective_iters = v["iters_override"]
    else:
        effective_iters = min(v.get("max_iters", total_iters), total_iters)

    # Deterministic model init (only matters for from-scratch runs; on resume
    # the weights are loaded from the checkpoint).
    torch.manual_seed(_variant_seed(v["name"], _INIT_SEED_OFFSET))

    # Dedicated sampler RNG, propagated explicitly — independent of global state.
    sampler_gen = torch.Generator(device=p3.DEVICE)
    sampler_gen.manual_seed(_variant_seed(v["name"], _SAMPLER_SEED_OFFSET))

    model      = _build_model_for_variant(v)
    sampler_fn = _build_sampler(v, n_f, n_tc, generator=sampler_gen)
    payoff_fn  = _build_payoff(v)
    ckpt_path  = vdir / "checkpoint.pt"

    if v["sampler_type"] == "vpinn":
        vpinn_module = _build_vpinn_loss(v)
        hist = train_variant_vpinn(model, vpinn_module, effective_iters,
                                   sampler_fn, payoff_fn, v["name"],
                                   lam_f=v.get("lam_f"),
                                   sampler_gen=sampler_gen,
                                   checkpoint_path=ckpt_path, resume=resume)
    elif v["sampler_type"] == "vpinn_engd":
        vpinn_module = _build_vpinn_loss(v)
        hist = train_variant_vpinn_engd(model, vpinn_module, effective_iters,
                                        sampler_fn, payoff_fn, v["name"],
                                        lam_f=v.get("lam_f"),
                                        checkpoint_path=ckpt_path, resume=resume,
                                        sampler_gen=sampler_gen)
    elif v["sampler_type"] == "engd":
        hist = train_variant_engd(model, effective_iters,
                                  sampler_fn, payoff_fn, v["name"],
                                  n_grid=v.get("n_grid", 30),
                                  n_tc_grid=v.get("n_tc_grid"),
                                  tikhonov_rel=v.get("tikhonov_rel", 1e-6),
                                  lam_f_override=v.get("lam_f_override"),
                                  lam_tc_override=v.get("lam_tc_override"),
                                  preconditioner_mode=v.get("preconditioner_mode", "joint"),
                                  checkpoint_path=ckpt_path, resume=resume,
                                  sampler_gen=sampler_gen)
    elif v["sampler_type"] in ("vpinn_lbfgs", "vpinn_lbfgs_is_tau"):
        # Same training function (stochastic L-BFGS with NaN-guard); the
        # difference is the sampler_fn built upstream (uniform vs τ→0 biased).
        # The expected benefit of "is_tau" is on the local γ quality near
        # maturity, not on the optimization procedure itself.
        vpinn_module = _build_vpinn_loss(v)
        hist = train_variant_vpinn_lbfgs(model, vpinn_module, effective_iters,
                                         sampler_fn, payoff_fn, v["name"],
                                         lam_f=v.get("lam_f"),
                                         checkpoint_path=ckpt_path, resume=resume,
                                         stochastic_batch=True,
                                         sampler_gen=sampler_gen)
    elif v["sampler_type"] == "vpinn_lbfgs_epoch":
        vpinn_module = _build_vpinn_loss(v)
        hist = train_variant_vpinn_lbfgs_epoch(model, vpinn_module, effective_iters,
                                               sampler_fn, payoff_fn, v["name"],
                                               lam_f=v.get("lam_f"),
                                               epoch_size=v.get("epoch_size", 20),
                                               checkpoint_path=ckpt_path, resume=resume,
                                               sampler_gen=sampler_gen)
    else:
        hist = train_variant(model, effective_iters, sampler_fn, payoff_fn, v["name"],
                             sampler_gen=sampler_gen)

    metrics   = compute_metrics(model, hist)
    gt_slices = _compute_gt_slices(model)

    torch.save(model.state_dict(), vdir / "models" / "pinn.pt")
    res = {**v, "linestyle": _to_mpl_ls(v.get("linestyle", "-")),
           "hist": hist, "metrics": metrics, "gt_slices": gt_slices}
    _save_variant(res, vdir)
    return res


def _ls_to_yaml(ls):
    """Recursively convert tuples → lists so yaml.safe_dump never writes !!python/tuple."""
    if isinstance(ls, (tuple, list)):
        return [_ls_to_yaml(x) for x in ls]
    return ls


def _summary_entry(v: dict, metrics: dict) -> dict:
    """Build the summary.yaml entry for one variant."""
    return {
        "name": v["name"], "label": v["label"],
        "color": v["color"], "linestyle": _ls_to_yaml(v["linestyle"]), "linewidth": v["linewidth"],
        "sampler_type": v["sampler_type"], "payoff_type": v["payoff_type"],
        "eps": v["eps"], "beta": v["beta"], "sigma_is": v["sigma_is"], "mix": v["mix"],
        **{k: val for k, val in metrics.items() if k != "pde_residual_tau"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation — European call PINN, x=ln(S) coordinates"
    )
    parser.add_argument("--iters",  type=int, default=200)
    parser.add_argument("--mode",   type=str,
                        default="compare-boundary-singularity-european-call",
                        choices=["compare-boundary-singularity-european-call",
                                 "ablation-eps", "ablation-beta", "ablation-is"])
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--n-tc",   type=int, default=None)
    parser.add_argument("--n-f",    type=int, default=None)
    parser.add_argument("--replot", type=str, default=None, metavar="DIR",
                        help="Regenerate all plots from saved data in DIR")
    parser.add_argument("--add-variant", type=str, default=None,
                        metavar="NAME:DIR",
                        help="Train variant NAME and append results to existing ablation DIR")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint.pt in the variant dir (for ENGD/LBFGS)")
    args = parser.parse_args()

    # ── Replot only ───────────────────────────────────────────────────────────
    if args.replot is not None:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                            datefmt="%H:%M:%S")
        logging.getLogger("matplotlib.mathtext").setLevel(logging.WARNING)
        _replot(Path(args.replot))
        return

    # ── Add a single variant to an existing run ───────────────────────────────
    if args.add_variant is not None:
        if ":" not in args.add_variant:
            raise SystemExit("--add-variant expects NAME:DIR  (e.g. vpinn:data/...)")
        variant_name, ablation_dir_str = args.add_variant.split(":", 1)
        ablation_dir = Path(ablation_dir_str)

        with open(ablation_dir / "metadata.yaml") as f:
            meta = yaml.safe_load(f)

        p3._apply_device_arg(args.device)
        n_tc = args.n_tc if args.n_tc is not None else meta.get("n_tc", p3.N_TC)
        n_f  = args.n_f  if args.n_f  is not None else meta.get("n_f",  p3.N_F)

        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
            handlers=[logging.StreamHandler(),
                      logging.FileHandler(ablation_dir / "ablation.log", mode="a")],
        )
        logging.getLogger("matplotlib.mathtext").setLevel(logging.WARNING)

        variants = _build_variants(meta["mode"])
        matching = [vv for vv in variants if vv["name"] == variant_name]
        if not matching:
            raise SystemExit(
                f"Variant {variant_name!r} not found in mode {meta['mode']!r}. "
                f"Available: {[vv['name'] for vv in variants]}"
            )
        v = matching[0]

        vdir = ablation_dir / f"variant_{variant_name}"
        for sub in ("training_metrics", "models"):
            (vdir / sub).mkdir(parents=True, exist_ok=True)

        logger.info(f"\n{'='*60}\n  Adding variant: {v['name']} — {v['label']}\n{'='*60}")
        logger.info(f"cmdline: {' '.join(sys.argv)}")

        res = _train_one_variant(v, vdir, n_f, n_tc, meta["iters"], resume=args.resume)
        _plot_variant(res, vdir)

        m = res["metrics"]
        logger.info(
            f"[{v['name']}]  rel_L2={m['rel_l2']:.3e}  "
            f"rel_L2_ATM={m['rel_l2_atm']:.3e}  "
            f"eps_Delta={m['rel_l2_delta']:.3e}  "
            f"eps_Gamma={m['rel_l2_gamma']:.3e}  "
            f"GEI={m['gei']:.2f}"
        )

        # Append to summary.yaml (replace if same name, append otherwise)
        with open(ablation_dir / "summary.yaml") as f:
            summary = yaml.safe_load(f)
        existing_names = {e["name"] for e in summary["variants"]}
        new_entry = _summary_entry(v, m)
        if variant_name in existing_names:
            summary["variants"] = [
                new_entry if e["name"] == variant_name else e
                for e in summary["variants"]
            ]
            logger.info(f"Updated existing entry for variant {variant_name!r} in summary.yaml")
        else:
            summary["variants"].append(new_entry)
            logger.info(f"Appended variant {variant_name!r} to summary.yaml")
        with open(ablation_dir / "summary.yaml", "w") as f:
            yaml.safe_dump(summary, f, allow_unicode=True)

        # Regenerate all comparison plots with the complete variant set
        _replot(ablation_dir)
        logger.info(f"\nVariant {variant_name!r} added — results in {ablation_dir}")
        return

    # ── Full ablation run ─────────────────────────────────────────────────────
    p3._apply_device_arg(args.device)
    n_tc = args.n_tc if args.n_tc is not None else p3.N_TC
    n_f  = args.n_f  if args.n_f  is not None else p3.N_F

    timestamp    = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    ablation_dir = (
        Path("data/exp_singularity_european_call")
        / f"{timestamp}_{args.mode}_logS_iters{args.iters}"
    )
    ablation_dir.mkdir(parents=True, exist_ok=True)
    (ablation_dir / "comparison").mkdir(exist_ok=True)

    variants = _build_variants(args.mode)
    for v in variants:
        for sub in ("training_metrics", "models"):
            (ablation_dir / f"variant_{v['name']}" / sub).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(ablation_dir / "ablation.log")],
    )
    logging.getLogger("matplotlib.mathtext").setLevel(logging.WARNING)

    logger.info(f"exp_singularity_european_call  coords=logS  mode={args.mode}  iters={args.iters}")
    logger.info(f"cmdline: {' '.join(sys.argv)}")
    logger.info(f"device={p3.DEVICE}  N_TC={n_tc}  N_F={n_f}")
    logger.info(f"x_lo={X_LO:.3f}  x_hi={X_HI:.3f}  x_atm={X_ATM:.3f}")
    logger.info(f"x_eval_lo={X_EVAL_LO:.3f}  x_eval_hi={X_EVAL_HI:.3f}")
    logger.info(f"output: {ablation_dir}")

    with open(ablation_dir / "metadata.yaml", "w") as f:
        yaml.safe_dump({
            "cmdline":     sys.argv,
            "mode":        args.mode,
            "iters":       args.iters,
            "coords":      "logS",
            "device":      str(p3.DEVICE),
            "n_tc":        n_tc,
            "n_f":         n_f,
            "K":           K,  "r": r, "sigma": sigma, "T": T,
            "x_lo":        X_LO,      "x_hi":      X_HI,      "x_atm":      X_ATM,
            "x_eval_lo":   X_EVAL_LO, "x_eval_hi": X_EVAL_HI,
        }, f)

    results, summary_variants = [], []

    for v in variants:
        vdir = ablation_dir / f"variant_{v['name']}"
        logger.info(f"\n{'='*60}\n  Variant: {v['name']} — {v['label']}\n{'='*60}")

        res = _train_one_variant(v, vdir, n_f, n_tc, args.iters)
        _plot_variant(res, vdir)
        results.append(res)

        m = res["metrics"]
        logger.info(
            f"[{v['name']}]  rel_L2={m['rel_l2']:.3e}  "
            f"rel_L2_ATM={m['rel_l2_atm']:.3e}  "
            f"eps_Delta={m['rel_l2_delta']:.3e}  "
            f"eps_Gamma={m['rel_l2_gamma']:.3e}  "
            f"GEI={m['gei']:.2f}"
        )
        summary_variants.append(_summary_entry(v, m))

    with open(ablation_dir / "summary.yaml", "w") as f:
        yaml.safe_dump({"variants": summary_variants}, f, allow_unicode=True)

    _plot_comparison(results, ablation_dir, args.iters, args.mode)
    logger.info(f"\nAll done — results in {ablation_dir}")


if __name__ == "__main__":
    main()
