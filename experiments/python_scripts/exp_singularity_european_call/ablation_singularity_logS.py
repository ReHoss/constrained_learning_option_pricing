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

    # Cheaper smoke test — single variant (naive), 20k iters on GPU:
    python3 experiments/python_scripts/exp_singularity_european_call/ablation_singularity_logS.py \\
        --variant naive --iters 20000 --device cuda

    # Full 3-method comparison:
    python3 experiments/python_scripts/exp_singularity_european_call/ablation_singularity_logS.py \\
        --iters 30000 --device cuda

    # Sensitivity to epsilon / beta / IS:
    python3 ... --iters 30000 --device cuda --mode ablation-eps
    python3 ... --iters 30000 --device cuda --mode ablation-beta
    python3 ... --iters 30000 --device cuda --mode ablation-is

    # Regenerate plots without retraining:
    python3 ... --replot data/ablation_singularity_logS/<mode>/<run_dir>_logS_iters<N>

    # Add a single new variant (e.g. VPINN) to an existing ablation folder:
    python3 ... --add-variant vpinn:data/ablation_singularity_logS/<mode>/<run_dir>_logS_iters<N> \\
        --device cuda
"""
from __future__ import annotations

# ── Init-only fast path ─────────────────────────────────────────────────────
# Detect --init-only in argv BEFORE importing torch.  Loading torch from a
# Lustre-backed venv costs several seconds; the init phase only writes YAML
# files so it has no reason to pay that cost.  When --init-only is present we
# defer to the torch-free `_ablation_catalogue.handle_init_only_cli`, which
# reproduces the in-function branch byte-for-byte.  The in-function branch in
# `main()` is kept as a safety net for programmatic callers (e.g. tests) that
# import this module rather than invoking it from the CLI.
import sys as _sys_for_init_only
if "--init-only" in _sys_for_init_only.argv and __name__ == "__main__":
    from pathlib import Path as _Path
    _sys_for_init_only.path.insert(0, str(_Path(__file__).resolve().parent))
    from _ablation_catalogue import handle_init_only_cli as _handle_init_only
    _handle_init_only()
    _sys_for_init_only.exit(0)
del _sys_for_init_only

import argparse
import contextlib
import fcntl
import logging
import math
import os
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import phase3_training as p3
from learning_option_pricing.models.etcnn import PINN, ETCNN, InputNormalization
from learning_option_pricing.models.resnet import ResNet
from learning_option_pricing.pricing.terminal import black_scholes_put
from learning_option_pricing.vpinn import GaussLegendreQuadrature
from learning_option_pricing.optimizers import (
    flat_grad, flat_params, set_flat_params,
    grid_line_search, measurement_jacobian, measurement_jacobian_fwd, solve_cg,
)
# Torch-free catalogue: variant list, plot-exclusions, mode→folder routing.
# Re-imported under the local-module names below so existing references stay
# unchanged; a runtime assertion checks that the catalogue mirrors p3's
# numerical constants so the two cannot drift silently.
import _ablation_catalogue as _cat

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameters (same as exp1 / ablation_singularity.py)
# ---------------------------------------------------------------------------
K, r, sigma, T, q = p3.K, p3.r, p3.sigma, p3.T, p3.q

# Catalogue mirrors these constants; assert parity so a drift fails loudly
# instead of producing a metadata.yaml whose numbers don't match the run.
assert (_cat.K, _cat.r, _cat.sigma, _cat.T, _cat.q) == (K, r, sigma, T, q), (
    "_ablation_catalogue constants drift from phase3_training:"
    f"  catalogue=(K={_cat.K}, r={_cat.r}, sigma={_cat.sigma}, T={_cat.T}, q={_cat.q})"
    f"  phase3_training=(K={K}, r={r}, sigma={sigma}, T={T}, q={q}). "
    "Update _ablation_catalogue.py to match."
)

# Per-script data-folder convention (CLAUDE.md): every run lands under
# data/<script_stem>/.  ``_ablation_catalogue.data_root_for_mode`` hard-codes
# the runner stem (kept torch-free), so this assertion fires loudly if the two
# files are ever moved / renamed independently.
assert _cat.RUNNER_SCRIPT_STEM == Path(__file__).stem, (
    f"_ablation_catalogue.RUNNER_SCRIPT_STEM={_cat.RUNNER_SCRIPT_STEM!r} but "
    f"this runner is {Path(__file__).stem!r}.  Update _ablation_catalogue.py "
    f"or this file so the per-script data folder ('data/<runner>/') stays "
    f"in sync between the two."
)

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
# Environment / config logging helpers
# ---------------------------------------------------------------------------

def _log_environment() -> None:
    """Log Python version, PyTorch version, CUDA toolkit version, and GPU info.

    Called once at script startup so every log file is self-contained and
    reproducible issues can be traced back to the exact software environment.
    """
    import platform
    logger.info(f"Python      : {platform.python_version()}  ({sys.executable})")
    logger.info(f"PyTorch     : {torch.__version__}")
    if torch.cuda.is_available():
        logger.info(f"CUDA toolkit: {torch.version.cuda}")
        for gpu_index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(gpu_index)
            total_memory_gib = props.total_memory / (1024 ** 3)
            logger.info(
                f"  GPU {gpu_index}: {props.name}  "
                f"{total_memory_gib:.1f} GiB  "
                f"compute capability {props.major}.{props.minor}"
            )
    else:
        logger.info("CUDA        : not available — running on CPU")


def _log_variant_config(v: dict, effective_iters: int,
                        model: torch.nn.Module,
                        master_seed: int,
                        init_seed: int, sampler_seed: int) -> None:
    """Log the full configuration of a variant before training starts.

    Logs every key in the variant dict, the effective iteration count, the
    model parameter count, the ablation-wide master seed, and the derived
    per-role seeds — so the log file alone is sufficient to reproduce or
    audit a run without consulting the source code.
    """
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  effective_iters : {effective_iters}")
    logger.info(f"  model_parameters: {n_params:,} total  ({n_trainable:,} trainable)")
    logger.info(f"  master_seed     : {master_seed}")
    logger.info(f"  init_seed       : {init_seed}")
    logger.info(f"  sampler_seed    : {sampler_seed}")
    skip_keys = {"color", "linestyle", "linewidth", "label"}
    for key, value in sorted(v.items()):
        if key not in skip_keys:
            logger.info(f"  {key:<20}: {value}")


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
        # Always restore on CPU — map_location='cuda' would have moved this
        # ByteTensor to GPU, but torch.set_rng_state requires a CPU tensor.
        torch.set_rng_state(ckpt["torch_rng"].cpu())
        if "torch_cuda_rng" in ckpt and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["torch_cuda_rng"]])
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
    elapsed = time.time() - t0
    logger.info(
        f"[{label}] Training done — {total_iters} iters in {elapsed:.1f}s "
        f"({elapsed / max(total_iters, 1):.2f}s/iter)  "
        f"best_loss={best_tracker.best_loss:.4e} at iter {best_tracker.best_iter}"
    )
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
            logger.info(f"[{label}] Checkpoint saved at iter {it}/{total_iters} → {checkpoint_path}")

    if best_tracker.restore(model):
        logger.info(
            f"[{label}] Restored best model from iter {best_tracker.best_iter} "
            f"(loss={best_tracker.best_loss:.4e}; last iter loss was {last_loss:.4e})"
        )
    model.eval()
    elapsed = time.time() - t0
    logger.info(
        f"[{label}] Training done — {total_iters} iters in {elapsed:.1f}s "
        f"({elapsed / max(total_iters, 1):.2f}s/iter)  "
        f"best_loss={best_tracker.best_loss:.4e} at iter {best_tracker.best_iter}"
    )
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
            logger.info(f"[{label}] Checkpoint saved at iter {it}/{total_iters} → {checkpoint_path}")

    if best_tracker.restore(model):
        logger.info(
            f"[{label}] Restored best model from iter {best_tracker.best_iter} "
            f"(loss={best_tracker.best_loss:.4e}; last iter loss was {last_loss:.4e})"
        )
    model.eval()
    elapsed = time.time() - t0
    logger.info(
        f"[{label}] Training done — {total_iters} iters in {elapsed:.1f}s "
        f"({elapsed / max(total_iters, 1):.2f}s/iter)  "
        f"best_loss={best_tracker.best_loss:.4e} at iter {best_tracker.best_iter}"
    )
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
            logger.info(f"[{label}] Checkpoint saved at iter {it}/{total_iters} → {checkpoint_path}")

    if best_tracker.restore(model):
        logger.info(
            f"[{label}] Restored best model from iter {best_tracker.best_iter} "
            f"(loss={best_tracker.best_loss:.4e}; last iter loss was {last_loss:.4e})"
        )
    model.eval()
    elapsed = time.time() - t0
    logger.info(
        f"[{label}] Training done — {total_iters} iters in {elapsed:.1f}s "
        f"({elapsed / max(total_iters, 1):.2f}s/iter)  "
        f"best_loss={best_tracker.best_loss:.4e} at iter {best_tracker.best_iter}"
    )
    return history


def train_variant_vpinn_lbfgs(
    model: torch.nn.Module,
    vpinn_module: _VPINNLossForwardLogS,
    total_iters: int,
    sampler_fn,
    payoff_fn,
    label: str = "vpinn_lbfgs",
    lam_f: float | None = None,
    lam_tc: float | None = None,
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
    lambda_f  = lam_f  if lam_f  is not None else p3.LAMBDA_F
    lambda_tc = lam_tc if lam_tc is not None else p3.LAMBDA_TC

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
        start_iter = ckpt["iter"] + 1
        history    = ckpt["history"]
        if "best_tracker" in ckpt:
            best_tracker.load_state_dict(ckpt["best_tracker"])
        if not stochastic_batch and "t_batch_fixed" in ckpt:
            # Restore optimizer state only when the batch is fixed: the curvature
            # history was built on exactly these time points, so the secant
            # condition remains valid across the resume boundary.
            optimizer.load_state_dict(ckpt["optimizer_state"])
            t_batch_fixed = ckpt["t_batch_fixed"].to(p3.DEVICE)
            logger.info(
                f"[{label}] ── Resumed from checkpoint at iter {ckpt['iter']}/{total_iters} "
                f"— curvature history + fixed t_batch restored"
            )
        else:
            # Stochastic batch: the curvature history was built on batches drawn
            # from a different RNG sequence; restoring it would misguide the line
            # search and cause NaN cascades.  Start L-BFGS fresh (lr=1.0) from
            # the saved model weights instead.
            logger.info(
                f"[{label}] ── Resumed from checkpoint at iter {ckpt['iter']}/{total_iters} "
                f"— model weights restored, L-BFGS curvature history reset "
                f"(stochastic batch: old history invalid across resume boundary)"
            )
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
    # Tracks consecutive NaN steps to detect and break the lr-halving spiral:
    # after _NAN_STREAK_HARD_RESET consecutive NaN, lr is reset to a moderate
    # value so the optimizer can recover rather than decaying to underflow.
    _NAN_STREAK_HARD_RESET = 15
    _nan_streak = [0]

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

        # NaN guard: roll back params AND reset optimizer state (corrupted curvature).
        # Hard recovery after _NAN_STREAK_HARD_RESET consecutive NaN steps: reset
        # lr to 1e-3 to break the exponential-halving spiral that would otherwise
        # drive lr to floating-point underflow.
        if loss is None or not torch.isfinite(torch.tensor(loss.item())):
            _nan_streak[0] += 1
            set_flat_params(model, params_before)
            old_lr = optimizer.param_groups[0]["lr"]
            if _nan_streak[0] >= _NAN_STREAK_HARD_RESET:
                new_lr = 1e-3
                logger.warning(
                    f"[{label}] iter {it}: {_nan_streak[0]} consecutive NaN steps — "
                    f"hard-reset lr {old_lr:.2e}→{new_lr:.2e} to break halving spiral"
                )
                _nan_streak[0] = 0
            else:
                new_lr = max(old_lr * 0.5, 1e-8)
                logger.warning(
                    f"[{label}] iter {it}: NaN (streak={_nan_streak[0]}) — "
                    f"rolled back + reset optimizer, lr {old_lr:.2e}→{new_lr:.2e}"
                )
            optimizer.__init__(model.parameters(), lr=new_lr, max_iter=20,
                               history_size=100, line_search_fn="strong_wolfe",
                               tolerance_grad=1e-7, tolerance_change=1e-9)
            continue

        _nan_streak[0] = 0
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
            logger.info(f"[{label}] Checkpoint saved at iter {it}/{total_iters} → {checkpoint_path}")

    if best_tracker.restore(model):
        logger.info(
            f"[{label}] Restored best model from iter {best_tracker.best_iter} "
            f"(loss={best_tracker.best_loss:.4e}; last iter loss was {last_loss:.4e})"
        )
    model.eval()
    elapsed = time.time() - t0
    logger.info(
        f"[{label}] Training done — {total_iters} iters in {elapsed:.1f}s "
        f"({elapsed / max(total_iters, 1):.2f}s/iter)  "
        f"best_loss={best_tracker.best_loss:.4e} at iter {best_tracker.best_iter}"
    )
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
    elapsed = time.time() - t0
    logger.info(
        f"[{label}] Training done — {total_iters} iters in {elapsed:.1f}s "
        f"({elapsed / max(total_iters, 1):.2f}s/iter)  "
        f"{n_epochs_total} epochs of {epoch_size} steps  "
        f"best_loss={best_tracker.best_loss:.4e} at iter {best_tracker.best_iter}"
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


def _compute_strong_residual_domain_profile(
    model: torch.nn.Module, n_x: int = 200,
) -> np.ndarray:
    """Strong-form PDE residual MEAN over the FULL spatial domain at each τ.

    Companion to :func:`_compute_weak_residual_profile`: the weak-form
    residual integrates over the whole spatial domain [X_LO, X_HI], so
    a fair strong-form comparison must do the same — not just probe a
    single ATM point.  Returns mean_x |F[V̂](x, T-τ)| for each τ in
    ``_WEAK_RES_TAU`` (the same τ grid as the weak-form profile), using
    ``n_x`` evenly spaced collocation points across [X_LO, X_HI].
    """
    device = p3.DEVICE
    model.eval()
    vals = []
    for tau_val in _WEAK_RES_TAU:
        t_val = T - tau_val
        x_p = torch.linspace(X_LO, X_HI, n_x, device=device).requires_grad_(True)
        t_p = torch.full((n_x,), t_val, device=device).requires_grad_(True)
        with torch.enable_grad():
            V_p = model(torch.stack([x_p, t_p], dim=1)).squeeze()
            F_p = bsm_operator_logS(V_p, x_p, t_p, r, sigma)
        vals.append(F_p.detach().abs().mean().item())
    return np.array(vals)


class _VPINNLossForwardLogS_ATM(torch.nn.Module):
    r"""Weak-form PDE residual with a single LOCALIZED test function at x = ln K.

    Companion to :class:`_VPINNLossForwardLogS`: that class uses 20 global
    sine modes spanning the whole domain, so it measures a "globally
    integrated" weak residual.  This variant replaces the basis with a
    single Gaussian bump centred at ``x_atm = ln K`` with bandwidth ``h``,
    L² -normalised over [X_LO, X_HI].  The result is a localized weak
    residual that probes the kink/ATM region only, providing the fair
    weak-form analog of the at-ATM strong-form residual.

    Bandwidth ``h`` is chosen small enough that the boundary terms from
    integration-by-parts are negligible (Gaussian tail ≪ machine epsilon
    at the domain boundaries).
    """

    phi_w:   torch.Tensor
    dphi_w:  torch.Tensor
    x_nodes: torch.Tensor
    weights: torch.Tensor

    def __init__(
        self,
        sigma: float,
        r: float,
        x_lo: float,
        x_hi: float,
        x_atm: float,
        h: float = 0.1,
        n_quad: int = 200,
    ):
        super().__init__()
        self.sigma = sigma
        self.r     = r
        self.mu    = r - 0.5 * sigma**2
        self.x_atm = x_atm
        self.h     = h

        quad    = GaussLegendreQuadrature(n_quad, x_lo, x_hi, dtype=torch.float32)
        x_nodes = quad.nodes
        weights = quad.weights

        # Localized test function: Gaussian bump centred at x_atm, std h.
        # Normalised so that ‖phi‖_{L²([X_LO,X_HI])} ≈ 1 (under quadrature),
        # i.e. its overall scale is comparable to one of the unit-norm sine
        # modes used in the global basis.
        dx       = x_nodes - x_atm
        phi      = torch.exp(-0.5 * (dx / h) ** 2)
        norm_sq  = (phi.pow(2) * weights).sum().clamp_min(1e-30)
        phi      = phi / norm_sq.sqrt()
        dphi     = -(dx / h**2) * phi

        self.register_buffer("phi_w",   phi  * weights)
        self.register_buffer("dphi_w",  dphi * weights)
        self.register_buffer("x_nodes", x_nodes)
        self.register_buffer("weights", weights)

    def forward(self, model: torch.nn.Module, t_batch: torch.Tensor) -> torch.Tensor:
        N_t = t_batch.shape[0]
        N_q = self.x_nodes.shape[0]
        x_rep = self.x_nodes.unsqueeze(0).expand(N_t, N_q).reshape(-1, 1)
        t_rep = t_batch.unsqueeze(1).expand(N_t, N_q).reshape(-1, 1)
        x_rep = x_rep.detach().requires_grad_(True)
        t_rep = t_rep.detach().requires_grad_(True)
        V = model(torch.cat([x_rep, t_rep], dim=1))
        dV_dt, dV_dx = torch.autograd.grad(
            V, [t_rep, x_rep],
            grad_outputs=torch.ones_like(V),
            create_graph=True,
        )
        V_vals = V.squeeze(1).reshape(N_t, N_q)
        V_t    = dV_dt.squeeze(1).reshape(N_t, N_q)
        V_x    = dV_dx.squeeze(1).reshape(N_t, N_q)
        f_phi  = V_t + self.mu * V_x - self.r * V_vals
        f_dphi = -(self.sigma**2 / 2.0) * V_x
        # Single test function → R is a scalar per τ (sum of quadrature
        # weighted integrand contributions).
        R = (f_phi * self.phi_w).sum(dim=1) + (f_dphi * self.dphi_w).sum(dim=1)
        return R.pow(2).mean()


def _compute_weak_residual_atm_profile(
    model: torch.nn.Module, h: float = 0.1,
) -> np.ndarray:
    """Weak-form PDE residual using a localized Gaussian-bump test function at ATM.

    Mirrors :func:`_compute_weak_residual_profile` but with a single test
    function centred at ``x = ln K`` instead of 20 global sine modes.
    Returns one value per τ in ``_WEAK_RES_TAU``.  ``h`` controls the
    bandwidth of the bump (default 0.1, i.e. ~5% of a typical
    [X_LO, X_HI] span).
    """
    device = p3.DEVICE
    vpinn_eval = _VPINNLossForwardLogS_ATM(
        sigma, r, X_LO, X_HI, X_ATM, h=h, n_quad=200,
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


# Diagnostic-only constants — kept separate from the canonical τ-grids
# so they can be tuned without touching the canonical comparison plots.
_DIAGNOSTIC_TAU_SLICES = [0.05 * T, T / 2, 0.98 * T]
_DIAGNOSTIC_K_MAX      = 200
_DIAGNOSTIC_N_QUAD     = 1024


def _compute_residual_spectrum_profile(
    model: torch.nn.Module,
    K_max: int = _DIAGNOSTIC_K_MAX,
    n_quad: int = _DIAGNOSTIC_N_QUAD,
) -> dict:
    r"""Project the strong-form PDE residual onto the sine basis at multiple τ.

    For each $\tau \in$ ``_DIAGNOSTIC_TAU_SLICES`` this computes the squared
    Fourier sine coefficients

    .. math::

        |\hat{\mathcal{F}}_k(\tau)|^2 \;=\;
        \left(\int_{X_{lo}}^{X_{hi}}\!\sin\!\left(\tfrac{k\pi(x-X_{lo})}{L}\right)
        \mathcal{F}[\hat V](x, T-\tau)\,dx\right)^{\!2}, \quad k = 1, \ldots, K_{\max}

    using Gauss-Legendre quadrature with ``n_quad`` nodes.

    Purpose: exposes the spatial-frequency structure of the trained network's
    residual.  Empirically (hard-IC ansatz, hard_ic_vpinn vs hard_ic_naive):

      * VPINN: spectrum is roughly FLAT at $|\hat{\mathcal{F}}_k|^2 \sim O(1)$
        across all $k$ — i.e. broadband, low-and-high alike.  Initially we
        expected a "cliff" at $k = K_{\rm test}^{\rm train}$ (the idea being
        that VPINN only minimises against the first $K_{\rm test}$ sine modes
        and ignores higher ones), but the data falsifies that prediction.
        The flat spectrum is consistent with a near-Dirac contribution to
        $\partial_{xx} V$ at $x = \ln K$: the hard-IC ansatz
        $V = g_1\,\mathrm{resnet} + g_2$ with $g_2 = (\mathrm{e}^x - K)^+$
        is $C^0$ only — the kink in $g_2$ propagates into $V$ at every $t$,
        and a Dirac in $\partial_{xx} V$ has a flat Fourier spectrum.
        The weak form sidesteps this via IBP (it never touches
        $\partial_{xx} V$), so VPINN can train successfully despite the
        underlying strong-form residual being broadband-large.

      * Strong-form PINN (naive / truncated / smooth): spectrum is flat AND
        small ($\sim 10^{-6}$) across all $k$, because minimising the
        pointwise $|\mathcal{F}[V]|^2$ forces the network to fight the
        Dirac contribution explicitly — at the cost of distorting the
        global solution (rel_L2 = 0.18 vs VPINN's 0.026).

    This is a *diagnostic-only* measurement: results live in
    ``comparison_diagnostics/`` (NOT in the canonical comparison folder).

    Returns a dict carrying three flat arrays so it can be unpacked into
    ``gt_comparison.npz`` via ``**spectrum``:

      * ``residual_spectrum_tau``      shape (n_tau,)
      * ``residual_spectrum_k_idx``    shape (K_max,)
      * ``residual_spectrum_F_hat_sq`` shape (n_tau, K_max)
    """
    device = p3.DEVICE
    model.eval()

    quad    = GaussLegendreQuadrature(n_quad, X_LO, X_HI, dtype=torch.float32)
    x_nodes = quad.nodes.to(device)
    weights = quad.weights.to(device)
    L       = X_HI - X_LO
    k_idx   = torch.arange(1, K_max + 1, dtype=torch.float32, device=device)
    # phi[k-1, q] = sin(k π (x_q - X_LO) / L)  — shape (K_max, n_quad)
    phi = torch.sin(
        k_idx.unsqueeze(1) * math.pi * (x_nodes - X_LO).unsqueeze(0) / L
    )

    F_hat_sq_all = []
    for tau_val in _DIAGNOSTIC_TAU_SLICES:
        t_val = T - tau_val
        x_in  = x_nodes.detach().clone().requires_grad_(True)
        t_in  = torch.full_like(x_in, t_val).requires_grad_(True)
        with torch.enable_grad():
            V = model(torch.stack([x_in, t_in], dim=1)).squeeze()
            F_pde = bsm_operator_logS(V, x_in, t_in, r, sigma)
        F_det = F_pde.detach()  # (n_quad,)
        # F̂_k = Σ_q w_q · φ_k(x_q) · F(x_q),   shape (K_max,)
        F_hat = (phi * (weights * F_det).unsqueeze(0)).sum(dim=1)
        F_hat_sq_all.append(F_hat.pow(2).cpu().numpy())

    return {
        "residual_spectrum_tau":      np.asarray(_DIAGNOSTIC_TAU_SLICES, dtype=float),
        "residual_spectrum_k_idx":    k_idx.cpu().numpy().astype(int),
        "residual_spectrum_F_hat_sq": np.asarray(F_hat_sq_all),
    }


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
        "weak_residual_tau":         np.array(_WEAK_RES_TAU),
        "weak_residual":             _compute_weak_residual_profile(model),
        # Companion measures for the fair 2x2 strong-vs-weak comparison:
        # both forms evaluated on (a) the full domain and (b) at ATM only.
        # All four use the same τ grid (_WEAK_RES_TAU) for direct overlay.
        "strong_residual_domain":    _compute_strong_residual_domain_profile(model),
        "weak_residual_atm":         _compute_weak_residual_atm_profile(model),
        # Diagnostic-only — see _compute_residual_spectrum_profile.  Plotted
        # by _plot_diagnostics into the comparison_diagnostics/ subfolder.
        **_compute_residual_spectrum_profile(model),
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
    r"Naive / trunc.:  $\Phi(x)=(e^x-K)^+$",
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
_FORMULA_RESIDUAL_SPECTRUM = "\n".join([
    r"$|\hat{\mathcal{F}}_k(\tau)|^2 = \left("
    r"\int_{X_{lo}}^{X_{hi}}\!"
    r"\sin\!\left(\frac{k\pi(x-X_{lo})}{X_{hi}-X_{lo}}\right)\,"
    r"\mathcal{F}[\hat V](x,\,T-\tau)\,dx"
    r"\right)^{\!2}$",
    _FORMULA_OP,
    r"Integration over the FULL domain $x\in[X_{lo},X_{hi}]$ by Gauss-Legendre quadrature ($n_{quad}=1024$).",
    r"With the C$^0$ hard-IC ansatz ($g_2$ has a kink at $x=\ln K$), $\partial_{xx}\hat V$ contains a near-Dirac at $\ln K$,"
    r" whose sine coefficient is $\sin(k\pi\alpha)$ with $\alpha=(\ln K-X_{lo})/(X_{hi}-X_{lo})$"
    r" — gives the $\sin^2(k\pi\alpha)$ oscillation envelope on top of a smooth resnet baseline.",
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
# Variant catalogue
# ---------------------------------------------------------------------------
# Defined in the torch-free `_ablation_catalogue` module so the init-only
# fast path at the top of this file can use them without importing torch.
# Re-exported under the local-module names for backward compatibility.
_EPS_GRID       = _cat._EPS_GRID
_BETA_GRID      = _cat._BETA_GRID
_IS_CONFIGS     = _cat._IS_CONFIGS
_COLORS         = _cat._COLORS
_build_variants = _cat._build_variants



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
    if t in ("vpinn", "vpinn_engd", "vpinn_lbfgs", "vpinn_lbfgs_epoch",
             "vpinn_lbfgs_full_batch"):
        return make_sampler_vpinn_logS(cfg.get("n_tau", 512),
                                       eps=cfg.get("eps", 0.01 * T),
                                       generator=generator)
    if t in ("vpinn_lbfgs_is_tau", "vpinn_lbfgs_is_tau_full_batch"):
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


def _build_hard_ic_ansatz_pinn(payoff_type: str = "exact",
                               beta: float | None = None) -> ETCNN:
    r"""Hard-IC ansatz: $V(x,t) = \frac{T-t}{T}\,\mathrm{NN}(x,t) + g_2(x)$.

    The linear ramp $g_1(t) = (T-t)/T$ vanishes at maturity, completely
    suppressing the network output at $t = T$, so $V(\cdot, T) = g_2(\cdot)$
    holds by construction.  No $\mathcal{L}_{ic}$ soft penalty is required:
    setting ``lam_tc=0`` on the variant skips the (already-zero) IC loss
    computation entirely.

    Parameters
    ----------
    payoff_type :
        ``"exact"`` (default) sets $g_2(x) = (e^x - K)^+$ — the exact
        European-call payoff with a kink at $x = \ln K$.  ``"smooth"`` sets
        $g_2(x) = \mathrm{softplus}(e^x - K, \beta) - \log 2 / \beta$, the
        $C^\infty$ approximation used by the ``smooth`` variant; pick its
        sharpness via ``beta`` (default $\beta = 100$).
    beta :
        Sharpness parameter for ``payoff_type="smooth"``.  Ignored for
        ``"exact"``.  Must be provided when ``payoff_type="smooth"``.

    Backbone is identical to :func:`_build_pinn` (ResNet d_in=2, d_out=1,
    n=50, M=4, L=2) to keep the parameter count comparable.
    """
    def g1(x_in: torch.Tensor, t_in: torch.Tensor) -> torch.Tensor:
        # (T - t) / T  — vanishes at t = T (the maturity / initial condition)
        return (T - t_in) / T

    if payoff_type == "exact":
        def g2(x_in: torch.Tensor, t_in: torch.Tensor) -> torch.Tensor:
            return torch.clamp(x_in.exp() - K, min=0.0)
    elif payoff_type == "smooth":
        if beta is None:
            raise ValueError(
                "_build_hard_ic_ansatz_pinn(payoff_type='smooth') requires beta"
            )
        log2 = math.log(2.0)
        beta_val = float(beta)

        def g2(x_in: torch.Tensor, t_in: torch.Tensor) -> torch.Tensor:
            return F.softplus(x_in.exp() - K, beta=beta_val) - log2 / beta_val
    else:
        raise ValueError(
            f"_build_hard_ic_ansatz_pinn: unknown payoff_type {payoff_type!r}"
        )

    resnet = ResNet(d_in=2, d_out=1, n=50, M=4, L=2)
    return ETCNN(resnet=resnet, g1=g1, g2=g2)


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
    _vpinn_like = res.get("sampler_type") in ("vpinn", "vpinn_engd", "vpinn_lbfgs", "vpinn_lbfgs_epoch", "vpinn_lbfgs_is_tau", "vpinn_lbfgs_full_batch", "vpinn_lbfgs_is_tau_full_batch")
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

# Variants to exclude from all comparison figures.  Data and checkpoints are
# still saved (training runs normally) — only the visual output is suppressed.
# Single source of truth lives in `_ablation_catalogue` so the torch-free
# init/launcher tooling sees the same exclusion list.
_PLOT_EXCLUDED_VARIANTS = _cat._PLOT_EXCLUDED_VARIANTS


def _plot_comparison(results: list[dict], ablation_dir: Path, iters: int, mode: str,
                     *, output_subdir: str = "comparison"):
    """Generate the comparison figures for ``results`` under ``ablation_dir``.

    The default ``output_subdir="comparison"`` writes to the canonical
    location populated by every full ablation run.  Pass a different name
    (e.g. ``comparison_excl_hard_ic_smooth``) to produce an alternative
    figure set without overwriting the canonical one — used by the
    ``--replot --exclude-variant`` workflow to inspect what the comparison
    looks like once a pathological run is dropped.
    """
    results = [r for r in results if r.get("name") not in _PLOT_EXCLUDED_VARIANTS]
    comp_dir = ablation_dir / output_subdir
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
                          "vpinn_lbfgs_epoch", "vpinn_lbfgs_is_tau",
                          "vpinn_lbfgs_full_batch", "vpinn_lbfgs_is_tau_full_batch")
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
        + "  See weak_residual_comparison.png for a strong-vs-weak post-hoc comparison.",
        fontsize=9,
    )
    lf_formula_cmp = "\n".join([
        _FORMULA_LF,
        _FORMULA_LF_VPINN,
        r"Note: the two formulations are NOT directly comparable on a shared axis"
        r" (different norms). weak_residual_comparison.png evaluates strong and weak"
        r" residuals on the same trained models, at both full-domain and ATM support.",
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

    # NOTE: a stand-alone "pde_residual_by_tau.png" (strong-form residual at ATM
    # vs τ, single panel) used to live here.  It is now subsumed by the bottom-
    # left panel of weak_residual_comparison.png, which plots the exact same
    # quantity inside the 2x2 strong/weak × full-domain/ATM grid and therefore
    # offers a fair comparison context.  Keeping a second copy in isolation was
    # redundant and reinforced the "single-axis" reading the 2x2 plot was
    # explicitly designed to discourage.

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
            r" - \frac{\ln 2}{\beta}$   (softplus, centered at $\tilde{\Phi}_\beta(\ln K)=0$)",
            r"Max error: $\max_x|\tilde{\Phi}_\beta-\Phi| = \frac{\ln 2}{\beta}$"
            r"   attained at $x=\ln K$ (ATM)",
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

    # Strong vs weak residual comparison — 2x2 fair-comparison layout.
    # The original 1x2 layout was misleading: the strong-form panel was
    # evaluated at a SINGLE spatial point (x = ln K), while the weak-form
    # panel integrated over the WHOLE domain.  Putting them side by side
    # invited reading them as "the same quantity in two forms" — they were
    # not.  The 2x2 layout below disentangles the two axes of comparison:
    #   * columns        = formulation  (strong  |  weak)
    #   * rows           = spatial support of the residual measure
    #                      top row     = full domain integration
    #                      bottom row  = localized at ATM (x = ln K)
    # Each panel within a row uses the SAME spatial support across the
    # two formulations, so the strong vs weak read is now apples-to-apples.
    valid_wr = [r for r in results
                if r.get("gt_slices") and "weak_residual" in (r["gt_slices"] or {})]
    if valid_wr:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        ax_dom_strong, ax_dom_weak = axes[0, 0], axes[0, 1]
        ax_atm_strong, ax_atm_weak = axes[1, 0], axes[1, 1]
        for res in valid_wr:
            kw = dict(label=res["label"], color=res["color"],
                      linestyle=res["linestyle"], linewidth=res["linewidth"],
                      marker="o", markersize=4)
            gt = res["gt_slices"]
            # Row 0 — full-domain support
            if "strong_residual_domain" in gt:
                ax_dom_strong.semilogy(
                    gt["weak_residual_tau"], gt["strong_residual_domain"], **kw,
                )
            ax_dom_weak.semilogy(
                gt["weak_residual_tau"], gt["weak_residual"], **kw,
            )
            # Row 1 — at-ATM support
            pde = res["metrics"]["pde_residual_tau"]
            ax_atm_strong.semilogy(pde["tau"], pde["residual"], **kw)
            if "weak_residual_atm" in gt:
                ax_atm_weak.semilogy(
                    gt["weak_residual_tau"], gt["weak_residual_atm"], **kw,
                )

        for ax, title in [
            (ax_dom_strong,
             r"Strong, full-domain: $\langle |\mathcal{F}[\hat V](\cdot,\tau)| \rangle_{x\in[X_{lo},X_{hi}]}$"),
            (ax_dom_weak,
             r"Weak, full-domain: $\frac{1}{K}\sum_k\!\left(\!\int \varphi_k\,\mathcal{F}[\hat V]\,dx\!\right)^2$  ($K{=}20$ sine modes)"),
            (ax_atm_strong,
             r"Strong, at ATM: $|\mathcal{F}[\hat V](x{=}\ln K,\tau)|$"),
            (ax_atm_weak,
             r"Weak, at ATM: $\left(\int \varphi_{\rm bump}\,\mathcal{F}[\hat V]\,dx\right)^2$  (Gaussian bump at $\ln K$)"),
        ]:
            ax.set_xlabel(r"$\tau$")
            ax.legend(fontsize=7, loc="best")
            ax.grid(True, alpha=0.3)
            ax.set_title(title, fontsize=8.5)
        ax_dom_strong.set_ylabel(r"$\langle |\mathcal{F}[\hat V]| \rangle$")
        ax_dom_weak.set_ylabel(r"$\mathcal{L}_f^{var}$")
        ax_atm_strong.set_ylabel(r"$|\mathcal{F}[\hat V]|_{x=\ln K}$")
        ax_atm_weak.set_ylabel(r"$\mathcal{L}_f^{var,\,\mathrm{ATM}}$")
        fig.suptitle(
            r"Residual comparison — strong vs weak $\times$ full-domain vs ATM  |  "
            + _SUPTITLE,
            fontsize=9,
        )
        fig.tight_layout()
        _apply_outside_legend(fig)
        _add_formula_box(fig, _FORMULA_WEAK_RES, bottom_margin=0.12)
        fig.savefig(comp_dir / "weak_residual_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # L-BFGS comparison: stochastic / epoch-based / full-batch (whichever are present)
    lbfgs_stoch      = next((r for r in results if r.get("sampler_type") == "vpinn_lbfgs"), None)
    lbfgs_epoch      = next((r for r in results if r.get("sampler_type") == "vpinn_lbfgs_epoch"), None)
    lbfgs_full_batch = next((r for r in results if r.get("sampler_type") == "vpinn_lbfgs_full_batch"), None)
    _lbfgs_candidates = [(lbfgs_stoch, "Stochastic L-BFGS\n(fresh batch every step)"),
                         (lbfgs_epoch, "Epoch L-BFGS\n(same batch for 20 steps, then resample)"),
                         (lbfgs_full_batch, "Full-batch L-BFGS\n(same fixed batch for entire run)")]
    _lbfgs_present = [(r, lbl) for r, lbl in _lbfgs_candidates if r is not None]
    if len(_lbfgs_present) >= 2:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for res, ls_label in _lbfgs_present:
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
        metrics_summary = "  |  ".join(
            f"{lbl.split(chr(10))[0]}: rel_L2={r['metrics']['rel_l2']:.3e}"
            for r, lbl in _lbfgs_present
        )
        fig.suptitle(
            f"L-BFGS batch strategy comparison  |  {_SUPTITLE}\n{metrics_summary}",
            fontsize=9,
        )
        fig.tight_layout()
        _apply_outside_legend(fig)
        _add_formula_box(fig,
            r"Stochastic: $t_{\rm batch}\sim U(0,T)$ at every step "
            r"— $y_k=\nabla f_{B_{k+1}}(x_{k+1})-\nabla f_{B_k}(x_k)$ mixes two objectives "
            r"(curvature noise, NaN instabilities)."
            "\n"
            r"Epoch-based: the same $t_{\rm batch}$ is kept for $N=20$ steps; "
            r"Full-batch: the same $t_{\rm batch}$ is kept for the entire run "
            r"— $y_k=\nabla f_B(x_{k+1})-\nabla f_B(x_k)$ is a true curvature estimate.",
            bottom_margin=0.16,
        )
        fig.savefig(comp_dir / "lbfgs_batch_strategy_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("L-BFGS comparison plot saved → lbfgs_batch_strategy_comparison.png")

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
    """Instantiate the correct architecture for a variant config.

    Dispatch order:
      * ``model_type == "hard_ic_ansatz"``  → :func:`_build_hard_ic_ansatz_pinn`
        (forces V(·,T) = g2(·) by construction; pass through ``payoff_type``
        and ``beta`` so the ``smooth`` analogue can use a softplus payoff in
        $g_2$ — analogous to how the original ``smooth`` variant uses a
        softplus in the IC loss).
      * ``sampler_type == "engd"``          → :func:`_build_engd_pinn`
        (paper-faithful small MLP for natural-gradient experiments)
      * default                             → :func:`_build_pinn`
    """
    if v.get("model_type") == "hard_ic_ansatz":
        return _build_hard_ic_ansatz_pinn(
            payoff_type=v.get("payoff_type", "exact"),
            beta=v.get("beta"),
        )
    if v["sampler_type"] == "engd":
        return _build_engd_pinn(hidden=v.get("hidden", 32))
    return _build_pinn()


def _load_model_for_variant(ablation_dir: Path, variant_name: str,
                             v: dict | None = None) -> torch.nn.Module | None:
    """Load a saved model for a variant; return None if missing OR architecture mismatch.

    Tolerant of obsolete models left by old runs (Sequential vs current PINN
    architecture) — emits a warning and continues, rather than crashing the
    regeneration of all plots.
    """
    model_path = ablation_dir / f"variant_{variant_name}" / "models" / "pinn.pt"
    if not model_path.exists():
        return None
    state = torch.load(model_path, map_location=p3.DEVICE, weights_only=True)
    # Try the current architecture first; fall back to legacy engd_pinn if needed
    candidates = [_build_pinn] if not variant_name.startswith("engd") else [_build_engd_pinn, _build_pinn]
    if variant_name.startswith("engd"):
        candidates = [_build_engd_pinn, _build_pinn]
    # Hard-IC ansatz variants store ETCNN weights with a different key layout
    # than the plain PINN; try the ansatz builder first so load_state_dict
    # succeeds.  The closures bind any per-variant payoff parameters
    # (smooth softplus β) so that the architecture instantiated here matches
    # the one that produced the saved weights.
    if v is not None and v.get("model_type") == "hard_ic_ansatz":
        payoff_type = v.get("payoff_type", "exact")
        payoff_beta = v.get("beta")

        def _build_ansatz_for_load() -> torch.nn.Module:
            return _build_hard_ic_ansatz_pinn(payoff_type=payoff_type,
                                              beta=payoff_beta)

        candidates = [_build_ansatz_for_load, _build_pinn]
    for build_fn in candidates:
        model = build_fn()
        try:
            model.load_state_dict(state)
            model.to(p3.DEVICE)
            return model
        except RuntimeError:
            continue
    logger.warning(
        f"pinn.pt model of '{variant_name}' is incompatible with known "
        f"architectures — variant skipped for GT slices recomputation "
        f"(likely an orphan directory from a prior run)."
    )
    return None


def _plot_diagnostics(results: list[dict], ablation_dir: Path) -> None:
    """Investigation-only plots, written into a sibling ``comparison_diagnostics/``
    folder so they do NOT contaminate the canonical ``comparison/`` set.

    Per the CLAUDE.md folder-organisation rule: any plot that probes a specific
    hypothesis (rather than reporting the canonical ablation result) lives in
    its own subfolder.  This keeps the canonical set citation-ready and makes
    it trivial to drop a diagnostic via ``rm -rf comparison_diagnostics/``.

    Currently produces:

    * ``residual_spectrum.png`` — squared Fourier sine coefficients
      $|\\hat{\\mathcal{F}}_k(\\tau)|^2$ of the strong-form PDE residual, plotted
      on log-y as a function of mode index $k$, one panel per τ slice in
      ``_DIAGNOSTIC_TAU_SLICES``.  A vertical line at $k=20$ marks the VPINN
      training basis size.  Variants whose residual spectrum sits at floor
      for $k \\le 20$ but jumps sharply for $k > 20$ are the ones VPINN
      "couldn't see" during training.
    """
    diag_dir = ablation_dir / "comparison_diagnostics"
    valid = [r for r in results
             if r.get("gt_slices") is not None
             and "residual_spectrum_F_hat_sq" in (r["gt_slices"] or {})]
    if not valid:
        return
    diag_dir.mkdir(exist_ok=True)

    first_gt = valid[0]["gt_slices"]
    tau_arr  = np.asarray(first_gt["residual_spectrum_tau"])
    k_idx    = np.asarray(first_gt["residual_spectrum_k_idx"])
    n_tau    = len(tau_arr)

    fig, axes = plt.subplots(1, n_tau, figsize=(5 * n_tau, 5), sharey=True)
    if n_tau == 1:
        axes = [axes]
    for j, tau_val in enumerate(tau_arr):
        ax = axes[j]
        for r in valid:
            gt = r["gt_slices"]
            spec = np.asarray(gt["residual_spectrum_F_hat_sq"])[j]
            # Floor tiny / non-positive values for log display
            spec_plot = np.maximum(spec, 1e-30)
            ax.semilogy(k_idx, spec_plot,
                        label=r["label"], color=r["color"],
                        linestyle=r["linestyle"], linewidth=r["linewidth"])
        ax.axvline(20, color="black", linestyle=":", linewidth=1.0,
                   label=r"$K_{\rm test}^{\rm train}=20$")
        ax.set_xlabel(r"sine mode index $k$")
        if j == 0:
            ax.set_ylabel(r"$|\hat{\mathcal{F}}_k(\tau)|^2$")
        ax.set_title(rf"$\tau = {float(tau_val):.3f}$", fontsize=10)
        ax.grid(True, alpha=0.3, which="both")
        if j == n_tau - 1:
            ax.legend(fontsize=7, loc="best")
    fig.suptitle(
        r"Diagnostic: Fourier sine spectrum of $\mathcal{F}[\hat V]$, "
        r"integrated over the FULL spatial domain $x\in[X_{lo},X_{hi}]$"
        "\n"
        r"Flat baseline $\Rightarrow$ near-Dirac $\partial_{xx}V$ at the kink; "
        r"$\sin^2(k\pi\alpha)$ envelope $\Rightarrow$ Dirac position $\ln K$ relative to domain"
        "\n" + _SUPTITLE,
        fontsize=9,
    )
    fig.tight_layout()
    _add_formula_box(fig, _FORMULA_RESIDUAL_SPECTRUM, bottom_margin=0.20)
    fig.savefig(diag_dir / "residual_spectrum.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Diagnostic plot saved → {diag_dir}/residual_spectrum.png")


def _replot(ablation_dir: Path, *, extra_exclude: list[str] | None = None) -> None:
    """Regenerate plots for an existing ablation directory.

    Default behaviour (``extra_exclude`` is None or empty) regenerates every
    per-variant figure, recomputes ground-truth Greek slices from the saved
    model, and rewrites the canonical ``comparison/`` folder — the workflow
    used after a run completes (e.g. in the SLURM FINALIZE phase) or when
    plotting code has changed.

    When ``extra_exclude`` is non-empty the function switches to an
    inspection-only mode: per-variant figures and ``gt_comparison.npz`` are
    left untouched, the listed variant names are dropped on top of the
    catalogue-wide ``_PLOT_EXCLUDED_VARIANTS``, and the comparison figures
    are written to ``comparison_excl_<sorted-names-joined-by-_>/`` so the
    canonical folder is preserved.  Useful for inspecting how the ablation
    reads once a pathological variant is removed, without losing the full
    figure set.
    """
    extra_exclude = list(extra_exclude or [])
    excluded = set(_PLOT_EXCLUDED_VARIANTS) | set(extra_exclude)

    with open(ablation_dir / "summary.yaml", encoding="utf-8") as f:
        summary = yaml.safe_load(f)
    with open(ablation_dir / "metadata.yaml", encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    visible_entries = [e for e in summary["variants"]
                       if e["name"] not in excluded]
    results = [_load_variant(ablation_dir / f"variant_{e['name']}", e)
               for e in visible_entries]

    if not extra_exclude:
        # Canonical replot — recompute GT slices and per-variant figures so
        # any change to _GT_TAU_SLICES or per-variant plotting code is
        # picked up.
        # CRITICAL: hard-IC ansatz variants must be instantiated via the
        # wrapping ``HardICAnsatzPINN`` (i.e. ``V = g1·resnet + g2``) rather
        # than the bare ``PINN`` (just ``resnet``) — both share state_dict
        # key names because the ansatz contains a resnet submodule with
        # identical layer naming, so ``load_state_dict`` silently succeeds
        # against the wrong architecture and every post-hoc metric is then
        # computed on the unwrapped resnet, NOT the trained function (off
        # by 7+ orders of magnitude — see commit msg for diagnostic numbers).
        # The summary.yaml entry omits ``model_type``, so we rebuild the
        # variant configs from the mode-keyed catalogue and pass those.
        full_variants_by_name = {v["name"]: v for v in _build_variants(meta["mode"])}
        for res, entry in zip(results, visible_entries):
            v_full = full_variants_by_name.get(entry["name"], entry)
            model = _load_model_for_variant(ablation_dir, entry["name"], v=v_full)
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
        for res, entry in zip(results, visible_entries):
            _plot_variant(res, ablation_dir / f"variant_{entry['name']}")
        output_subdir = "comparison"
    else:
        # Filtered replot — comparison figures only, never overwrite the
        # canonical artefacts.  The subdir name encodes the exclusion list
        # so the user can keep multiple filtered views side by side.
        output_subdir = "comparison_excl_" + "_".join(sorted(extra_exclude))
        logger.info(
            f"Filtered replot: excluding {sorted(extra_exclude)} on top of "
            f"the catalogue-wide exclusion list; writing comparison figures "
            f"to {output_subdir}/.  Per-variant figures and gt_comparison.npz "
            f"left untouched."
        )

    # Older metadata may still carry the legacy ``iters`` field; new runs
    # write ``num_iterations`` (which can be null when each variant uses its
    # own default).  Fall back to 0 so ``_plot_comparison`` always receives
    # an int.
    iters_for_title = (
        meta.get("num_iterations")
        if "num_iterations" in meta else meta.get("iters", 0)
    ) or 0
    _plot_comparison(results, ablation_dir, iters_for_title, meta["mode"],
                     output_subdir=output_subdir)

    # Diagnostics live in their own sibling folder so the canonical
    # ``comparison/`` set stays clean.  Only emitted on canonical replots —
    # filtered ``--exclude-variant`` views deliberately do not touch them.
    if not extra_exclude:
        _plot_diagnostics(results, ablation_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _derive_seed(master_seed: int, role_tag: str) -> int:
    """Deterministic 63-bit seed derived from a master seed and a role tag.

    The master seed is the single source of randomness for the whole run; it
    is supplied by the user via the ``--seed`` CLI flag (default 0) and shared
    across every variant of an ablation, so that two variants with identical
    hyperparameters produce identical initial weights and identical sampler
    trajectories — the only differences observable between variants reflect
    the methodological change, not RNG noise.

    Per-role decorrelation (model init vs. sampler vs. evaluation, …) is
    obtained by hashing the role tag and XORing it into the master seed.
    The role tag is internal plumbing — it must never carry presentation
    metadata such as the variant's display name, label, color, or iteration
    budget, otherwise reproducibility breaks across cosmetic variant edits.
    """
    import hashlib
    h = hashlib.blake2b(role_tag.encode("utf-8"), digest_size=8).digest()
    raw = int.from_bytes(h, "big") ^ (master_seed & ((1 << 64) - 1))
    # torch.Generator.manual_seed expects a positive 64-bit int
    return raw & ((1 << 63) - 1)


def _train_one_variant(
    v: dict,
    vdir: Path,
    n_f: int,
    n_tc: int,
    total_iters: int | None,
    master_seed: int,
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

    Both per-role seeds are derived from ``master_seed`` (the ablation-wide
    seed supplied via ``--seed``) through different role tags, so the two
    roles are decorrelated *within* a variant while every variant of the
    ablation sees the same per-role seed at fixed ``master_seed``.  This is
    the seeding policy required by ``CLAUDE.md``: variants sharing the same
    hyperparameters produce the same initial weights and the same sampler
    trajectory regardless of cosmetic differences (name, label, iteration
    budget).  For variance estimates, sweep ``--seed`` over several values.
    """
    # Effective iteration count: each variant declares its own natural budget
    # via ``default_num_iterations``; the user can override the whole ablation
    # at the CLI with ``--num-iterations N`` (typically for smoke tests).  No
    # min/max gymnastics — one number per variant, one global override.
    # ``total_iters`` is the resolved CLI override or ``None`` (= use each
    # variant's default).
    if total_iters is not None:
        effective_iters = total_iters
    else:
        effective_iters = v["default_num_iterations"]

    # Deterministic model init (only matters for from-scratch runs; on resume
    # the weights are loaded from the checkpoint).
    init_seed    = _derive_seed(master_seed, "init")
    sampler_seed = _derive_seed(master_seed, "sampler")
    torch.manual_seed(init_seed)

    # Dedicated sampler RNG, propagated explicitly — independent of global state.
    sampler_gen = torch.Generator(device=p3.DEVICE)
    sampler_gen.manual_seed(sampler_seed)

    model      = _build_model_for_variant(v)
    sampler_fn = _build_sampler(v, n_f, n_tc, generator=sampler_gen)
    payoff_fn  = _build_payoff(v)
    ckpt_path  = vdir / "checkpoint.pt"

    _log_variant_config(v, effective_iters, model, master_seed, init_seed, sampler_seed)

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
    elif v["sampler_type"] in ("vpinn_lbfgs_full_batch", "vpinn_lbfgs_is_tau_full_batch"):
        # Full-batch L-BFGS: the n_tau time points are drawn once before the loop
        # and held fixed for every outer step.  The objective is deterministic,
        # so the L-BFGS secant condition is always consistent — curvature history
        # accumulates reliable second-order information.  Unlike the stochastic
        # variant, the optimizer state (including curvature history) can be safely
        # restored across checkpoint resumes.
        # vpinn_lbfgs_is_tau_full_batch additionally uses IS τ→0 biased sampling
        # (τ = T·U^(1/α), α=0.3) to concentrate the fixed batch near the maturity
        # singularity while keeping the deterministic L-BFGS objective.
        vpinn_module = _build_vpinn_loss(v)
        hist = train_variant_vpinn_lbfgs(model, vpinn_module, effective_iters,
                                         sampler_fn, payoff_fn, v["name"],
                                         lam_f=v.get("lam_f"),
                                         lam_tc=v.get("lam_tc"),
                                         checkpoint_path=ckpt_path, resume=resume,
                                         stochastic_batch=False,
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


@contextlib.contextmanager
def _summary_yaml_lock(ablation_dir: Path):
    """Exclusive file-lock for the summary.yaml read-modify-write section.

    When several ``--add-variant`` jobs run in parallel (e.g. SLURM array on a
    cluster), they each call this block to append their entry to
    ``summary.yaml``.  Without a lock, two concurrent processes would race
    on the read-then-write sequence and the last writer would silently clobber
    the previous one's entry.

    The lock lives at ``<ablation_dir>/summary.yaml.lock`` and is held for the
    full duration of the wrapped block.  ``fcntl.flock`` is advisory and
    POSIX-only — fine on Linux/Jean Zay, would need a different primitive on
    Windows but the cluster setup is Linux-only.
    """
    lock_path = ablation_dir / "summary.yaml.lock"
    lock_path.touch(exist_ok=True)
    fd = os.open(str(lock_path), os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _summary_entry(v: dict, metrics: dict) -> dict:
    """Build the summary.yaml entry for one variant.

    ``model_type`` is preserved when present so that downstream consumers
    (``--replot``, ``--add-variant``) can dispatch to the correct
    architecture in :func:`_load_model_for_variant` without having to
    rebuild the catalogue first.  Variants that don't set ``model_type``
    in the catalogue (the bare PINN sweeps) keep behaving as before.
    """
    entry = {
        "name": v["name"], "label": v["label"],
        "color": v["color"], "linestyle": _ls_to_yaml(v["linestyle"]), "linewidth": v["linewidth"],
        "sampler_type": v["sampler_type"], "payoff_type": v["payoff_type"],
        "eps": v["eps"], "beta": v["beta"], "sigma_is": v["sigma_is"], "mix": v["mix"],
        **{k: val for k, val in metrics.items() if k != "pde_residual_tau"},
    }
    if "model_type" in v:
        entry["model_type"] = v["model_type"]
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation — European call PINN, x=ln(S) coordinates"
    )
    parser.add_argument("--num-iterations", dest="num_iterations",
                        type=int, default=None,
                        help=("Global override for every variant's "
                              "default_num_iterations.  Omit (the default) to "
                              "let each variant use its own natural budget "
                              "declared in the catalogue — recommended for "
                              "real ablations.  Useful for smoke tests "
                              "(combine with --debug)."))
    parser.add_argument("--seed", dest="seed", type=int, default=0,
                        help=("Ablation-wide master seed (default 0). Every "
                              "variant of the ablation uses the same master "
                              "seed; per-role RNGs (model init, sampler, …) "
                              "are derived deterministically from it via "
                              "role-tagged salts (see _derive_seed). Bump "
                              "--seed to obtain a fresh, fully decorrelated "
                              "repeat of the whole ablation (variance "
                              "estimates: sweep --seed 0,1,2,…)."))
    parser.add_argument("--mode",   type=str,
                        default="compare-boundary-singularity-european-call",
                        choices=["compare-boundary-singularity-european-call",
                                 "ablation-eps", "ablation-beta", "ablation-is",
                                 "hard-ic-ansatz-european-call"])
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--n-tc",   type=int, default=None)
    parser.add_argument("--n-f",    type=int, default=None)
    parser.add_argument("--replot", type=str, default=None, metavar="DIR",
                        help="Regenerate all plots from saved data in DIR")
    parser.add_argument("--exclude-variant", dest="exclude_variant",
                        action="append", default=[], metavar="NAME",
                        help=("Variant name to exclude from the comparison "
                              "figures. Repeatable (pass multiple times to "
                              "exclude several variants).  Only meaningful "
                              "alongside --replot: filtered figures are "
                              "written to "
                              "<DIR>/comparison_excl_<sorted-names-joined-by-_>/, "
                              "leaving the canonical <DIR>/comparison/ "
                              "untouched.  Use it for inspection-only "
                              "regenerations (e.g. \"what does the ablation "
                              "look like if I drop a divergent run?\") "
                              "without rewriting per-variant figures or "
                              "ground-truth slices."))
    parser.add_argument("--add-variant", type=str, default=None,
                        metavar="NAME:DIR",
                        help="Train variant NAME and append results to existing ablation DIR")
    parser.add_argument("--variant", type=str, default=None, metavar="NAME",
                        help=("Train a single variant NAME standalone in a fresh "
                              "ablation directory (skips multi-variant comparison "
                              "plots). Useful for smoke tests."))
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint.pt in the variant dir (for ENGD/LBFGS)")
    parser.add_argument("--debug", action="store_true",
                        help=("Mark this run as a test/smoke run.  The "
                              "timestamped ablation directory is prefixed "
                              "with '_debug_' so it is visually separated "
                              "from real runs in `ls` (leading underscore "
                              "sorts first) and can be cleaned in bulk via:"
                              "  find data -type d -name '_debug_*' -prune "
                              "-exec rm -rf {} +"))
    parser.add_argument("--init-only", action="store_true",
                        help=("Create the timestamped ablation directory + "
                              "metadata.yaml + empty summary.yaml and exit, "
                              "without training any variant. Designed for "
                              "SLURM-array workflows where the directory must "
                              "be ready before parallel --add-variant jobs are "
                              "submitted. Prints the absolute directory path "
                              "on stdout (last line) so the launcher can "
                              "capture it."))
    parser.add_argument("--config-dir", type=str, default=None, metavar="DIR",
                        help=("Folder containing per-variant YAML configs (one "
                              "file per array task).  Must be used together "
                              "with --config-name.  Matches the convention "
                              "expected by bash_scripts/.../job_array_batch_xp.slurm "
                              "so that this Python script can be plugged into "
                              "the existing Jean Zay job-array worker without "
                              "modification."))
    parser.add_argument("--config-name", type=str, default=None, metavar="NAME",
                        help=("Basename (without .yaml) of the config file "
                              "inside --config-dir to load.  The YAML must "
                              "contain at least 'mode', 'variant_name' and "
                              "'ablation_dir'; effect is equivalent to "
                              "--add-variant <variant_name>:<ablation_dir> for "
                              "the specified mode."))
    args = parser.parse_args()

    # ── Config-driven entry point (YAML) ─────────────────────────────────────
    # This mirrors what the Jean Zay worker job_array_batch_xp.slurm passes:
    #     python script.py --config-dir DIR --config-name NAME
    # We translate the YAML's fields into the legacy --add-variant invocation
    # so the rest of the main() body stays untouched.
    if args.config_dir is not None or args.config_name is not None:
        if args.config_dir is None or args.config_name is None:
            raise SystemExit(
                "--config-dir and --config-name must be provided together."
            )
        config_path = Path(args.config_dir) / f"{args.config_name}.yaml"
        if not config_path.exists():
            raise SystemExit(f"Config file does not exist: {config_path}")
        with open(config_path, encoding="utf-8") as f:
            cfg_yaml = yaml.safe_load(f) or {}
        required = ("mode", "variant_name", "ablation_dir")
        missing = [k for k in required if k not in cfg_yaml]
        if missing:
            raise SystemExit(
                f"Config {config_path} is missing required keys: {missing}"
            )
        args.mode = cfg_yaml["mode"]
        args.add_variant = f"{cfg_yaml['variant_name']}:{cfg_yaml['ablation_dir']}"
        if "num_iterations" in cfg_yaml and cfg_yaml["num_iterations"] is not None:
            args.num_iterations = int(cfg_yaml["num_iterations"])
        if "device" in cfg_yaml and args.device == "auto":
            args.device = cfg_yaml["device"]
        if cfg_yaml.get("resume", False):
            args.resume = True
        # ``master_seed`` is the only legitimate override path for the
        # ablation-wide seed in the YAML — kept optional so existing config
        # files without it keep working (they fall back to the metadata.yaml
        # of the ablation directory, see --add-variant below).
        if "master_seed" in cfg_yaml and cfg_yaml["master_seed"] is not None:
            args.seed = int(cfg_yaml["master_seed"])

    # ── Replot only ───────────────────────────────────────────────────────────
    if args.replot is not None:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                            datefmt="%H:%M:%S")
        logging.getLogger("matplotlib.mathtext").setLevel(logging.WARNING)
        _replot(Path(args.replot), extra_exclude=args.exclude_variant)
        return

    # ── Add a single variant to an existing run ───────────────────────────────
    if args.add_variant is not None:
        if ":" not in args.add_variant:
            raise SystemExit("--add-variant expects NAME:DIR  (e.g. vpinn:data/...)")
        variant_name, ablation_dir_str = args.add_variant.split(":", 1)
        ablation_dir = Path(ablation_dir_str)

        with open(ablation_dir / "metadata.yaml", encoding="utf-8") as f:
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
        _log_environment()
        logger.info(f"device={p3.DEVICE}  N_TC={n_tc}  N_F={n_f}")

        # The CLI flag wins over what's recorded in metadata.yaml when both
        # are provided.  When --num-iterations is omitted, fall back to the
        # value the init step persisted (which itself may be null = use
        # per-variant defaults).
        num_iterations_override = args.num_iterations
        if num_iterations_override is None:
            num_iterations_override = meta.get("num_iterations")
        # ``master_seed`` is an ablation-wide property recorded by the
        # init-only step into ``metadata.yaml``; every --add-variant job
        # for the same ablation directory must reuse exactly that value so
        # variants stay comparable.  Pre-existing ablations created before
        # the master-seed mechanism existed do not carry the field — we
        # fall back to 0 there for forward-compatibility, but new runs
        # always have it explicitly persisted.
        master_seed = int(meta.get("master_seed", 0))
        res = _train_one_variant(
            v, vdir, n_f, n_tc, num_iterations_override, master_seed,
            resume=args.resume,
        )
        _plot_variant(res, vdir)

        m = res["metrics"]
        logger.info(
            f"[{v['name']}]  rel_L2={m['rel_l2']:.3e}  "
            f"rel_L2_ATM={m['rel_l2_atm']:.3e}  "
            f"eps_Delta={m['rel_l2_delta']:.3e}  "
            f"eps_Gamma={m['rel_l2_gamma']:.3e}  "
            f"GEI={m['gei']:.2f}"
        )

        # Append to summary.yaml (replace if same name, append otherwise).
        # Wrapped in an advisory file-lock so concurrent SLURM-array jobs do
        # not clobber each other's entry on the read-modify-write sequence.
        new_entry = _summary_entry(v, m)
        with _summary_yaml_lock(ablation_dir):
            with open(ablation_dir / "summary.yaml", encoding="utf-8") as f:
                summary = yaml.safe_load(f)
            existing_names = {e["name"] for e in summary["variants"]}
            if variant_name in existing_names:
                summary["variants"] = [
                    new_entry if e["name"] == variant_name else e
                    for e in summary["variants"]
                ]
                logger.info(
                    f"Updated existing entry for variant {variant_name!r} in summary.yaml"
                )
            else:
                summary["variants"].append(new_entry)
                logger.info(f"Appended variant {variant_name!r} to summary.yaml")
            with open(ablation_dir / "summary.yaml", "w", encoding="utf-8") as f:
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
    variant_suffix = f"_variant_{args.variant}" if args.variant else ""
    # ``--debug`` prepends ``_debug_`` to the timestamped folder.  The
    # underscore sorts after digits in C/UTF-8 locale, so debug runs land in
    # a contiguous block at the bottom of ``ls`` — visually separated from
    # real runs and wipeable in one command (see --help).  Keep this
    # consistent with the torch-free fast path in
    # ``_ablation_catalogue.handle_init_only_cli``.
    debug_prefix = "_debug_" if args.debug else ""
    # The folder name carries an explicit iteration count only when the user
    # provided a global ``--num-iterations`` override; otherwise each variant
    # uses its own ``default_num_iterations`` and a single global number in
    # the path would be misleading.
    num_iterations_tag = (
        f"_num_iterations_{args.num_iterations}"
        if args.num_iterations is not None else ""
    )
    # data_root_for_mode lives in _ablation_catalogue so the init-only fast
    # path and the launcher's YAML generation use the same routing logic.
    data_root = Path(_cat.data_root_for_mode(args.mode))
    ablation_dir = (
        data_root
        / f"{debug_prefix}{timestamp}_{args.mode}_logS{num_iterations_tag}{variant_suffix}"
    )
    ablation_dir.mkdir(parents=True, exist_ok=True)
    (ablation_dir / "comparison").mkdir(exist_ok=True)

    variants = _build_variants(args.mode)
    if args.variant is not None:
        matching = [vv for vv in variants if vv["name"] == args.variant]
        if not matching:
            raise SystemExit(
                f"--variant {args.variant!r} not found in mode {args.mode!r}. "
                f"Available: {[vv['name'] for vv in variants]}"
            )
        variants = matching
    for v in variants:
        for sub in ("training_metrics", "models"):
            (ablation_dir / f"variant_{v['name']}" / sub).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(ablation_dir / "ablation.log")],
    )
    logging.getLogger("matplotlib.mathtext").setLevel(logging.WARNING)

    logger.info(
        f"{Path(__file__).stem}  coords=logS  mode={args.mode}  "
        f"num_iterations={args.num_iterations!r} (None = per-variant default)"
    )
    logger.info(f"cmdline: {' '.join(sys.argv)}")
    _log_environment()
    logger.info(f"device={p3.DEVICE}  N_TC={n_tc}  N_F={n_f}")
    logger.info(f"sigma={sigma}  r={r}  K={K}  T={T}")
    logger.info(f"x_lo={X_LO:.3f}  x_hi={X_HI:.3f}  x_atm={X_ATM:.3f}")
    logger.info(f"x_eval_lo={X_EVAL_LO:.3f}  x_eval_hi={X_EVAL_HI:.3f}")
    logger.info(f"master_seed={args.seed}  (shared across every variant of this ablation)")
    logger.info(f"output: {ablation_dir}")

    with open(ablation_dir / "metadata.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({
            "cmdline":         sys.argv,
            "mode":            args.mode,
            "num_iterations":  args.num_iterations,
            "master_seed":     args.seed,
            "coords":          "logS",
            "device":          str(p3.DEVICE),
            "n_tc":            n_tc,
            "n_f":             n_f,
            "K":               K,  "r": r, "sigma": sigma, "T": T,
            "x_lo":            X_LO,      "x_hi":      X_HI,      "x_atm":      X_ATM,
            "x_eval_lo":       X_EVAL_LO, "x_eval_hi": X_EVAL_HI,
        }, f)

    # ── Init-only short-circuit ──────────────────────────────────────────────
    # When parallel SLURM-array jobs need a shared ablation directory, this
    # mode creates the directory + metadata.yaml + empty summary.yaml, prints
    # the absolute path on stdout (last line), and exits *without training*.
    # The launcher can then capture the path and submit one --add-variant job
    # per variant against it.
    if args.init_only:
        empty_summary: dict = {"variants": []}
        with open(ablation_dir / "summary.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(empty_summary, f, allow_unicode=True)
        logger.info(
            f"--init-only: created ablation dir, metadata.yaml, and empty "
            f"summary.yaml. No variant trained."
        )
        # Print absolute path on the *last* stdout line so a bash launcher can
        # do  EXPDIR=$(python ... --init-only | tail -n1)
        print(str(ablation_dir.resolve()))
        return

    results, summary_variants = [], []

    for v in variants:
        vdir = ablation_dir / f"variant_{v['name']}"
        logger.info(f"\n{'='*60}\n  Variant: {v['name']} — {v['label']}\n{'='*60}")

        res = _train_one_variant(v, vdir, n_f, n_tc, args.num_iterations, args.seed)
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

    with open(ablation_dir / "summary.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"variants": summary_variants}, f, allow_unicode=True)

    if len(results) > 1:
        # _plot_comparison still receives a single iters value for its title;
        # pass the user override or 0 as a placeholder (the title text isn't
        # critical — the per-variant captions carry their actual iter counts).
        _plot_comparison(results, ablation_dir,
                         args.num_iterations if args.num_iterations is not None else 0,
                         args.mode)
    else:
        logger.info("Single-variant run — skipping comparison plots.")
    logger.info(f"\nAll done — results in {ablation_dir}")


if __name__ == "__main__":
    main()
