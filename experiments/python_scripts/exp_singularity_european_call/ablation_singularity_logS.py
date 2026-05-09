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

    phi_w:   torch.Tensor
    dphi_w:  torch.Tensor
    x_nodes: torch.Tensor

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

        self.register_buffer("phi_w",   phi  * weights)   # (K, N_q)
        self.register_buffer("dphi_w",  dphi * weights)   # (K, N_q)
        self.register_buffer("x_nodes", x_nodes)          # (N_q,)

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

def make_sampler_naive_logS(n_f: int, n_tc: int):
    def _sample():
        device = p3.DEVICE
        x_f = (torch.rand(n_f, device=device) * (X_HI - X_LO) + X_LO).requires_grad_(True)
        t_f = (torch.rand(n_f, device=device) * T).requires_grad_(True)
        x_tc = torch.rand(n_tc, device=device) * (X_HI - X_LO) + X_LO
        t_tc = torch.full((n_tc,), T, device=device)
        return x_f, t_f, x_tc, t_tc
    return _sample


def make_sampler_truncated_logS(n_f: int, n_tc: int, eps: float):
    """PDE points in t ∈ [0, T-eps]; terminal condition still at t=T exact."""
    def _sample():
        device = p3.DEVICE
        x_f = (torch.rand(n_f, device=device) * (X_HI - X_LO) + X_LO).requires_grad_(True)
        t_f = (torch.rand(n_f, device=device) * (T - eps)).requires_grad_(True)
        x_tc = torch.rand(n_tc, device=device) * (X_HI - X_LO) + X_LO
        t_tc = torch.full((n_tc,), T, device=device)
        return x_f, t_f, x_tc, t_tc
    return _sample


def make_sampler_importance_logS(n_f: int, n_tc: int,
                                  sigma_is: float, mix: float = 0.5,
                                  eps: float = 0.0):
    """Mix of uniform + Gaussian concentrated around x = ln(K) (ATM)."""
    def _sample():
        device = p3.DEVICE
        n_focal   = int(n_f * mix)
        n_uniform = n_f - n_focal
        x_uniform = torch.rand(n_uniform, device=device) * (X_HI - X_LO) + X_LO
        x_focal   = (X_ATM + torch.randn(n_focal, device=device) * sigma_is).clamp(X_LO, X_HI)
        x_f = torch.cat([x_uniform, x_focal]).requires_grad_(True)
        t_f = (torch.rand(n_f, device=device) * (T - eps)).requires_grad_(True)
        x_tc = torch.rand(n_tc, device=device) * (X_HI - X_LO) + X_LO
        t_tc = torch.full((n_tc,), T, device=device)
        return x_f, t_f, x_tc, t_tc
    return _sample


def make_sampler_vpinn_logS(n_tau: int, n_tc: int, eps: float = 0.01 * T):
    """Return t_batch (collocation times for the VPINN) + TC points.

    The spatial quadrature is handled inside _VPINNLossForwardLogS; the sampler
    only needs to draw time collocation points and terminal-condition points.
    Returns (t_batch, x_tc, t_tc).

    Args:
        eps: Temporal truncation — PDE collocation points are drawn from
             [0, T - eps] to avoid the near-terminal singularity zone where
             ∂V/∂t is large (same rationale as the truncated variant).
    """
    def _sample():
        device = p3.DEVICE
        t_batch = torch.rand(n_tau, device=device) * (T - eps)  # (n_tau,) in [0, T-eps]
        x_tc = torch.rand(n_tc, device=device) * (X_HI - X_LO) + X_LO
        t_tc = torch.full((n_tc,), T, device=device)
        return t_batch, x_tc, t_tc
    return _sample


def compute_losses_vpinn_logS(
    model,
    t_batch: torch.Tensor,
    vpinn_module: _VPINNLossForwardLogS,
    x_tc: torch.Tensor,
    t_tc: torch.Tensor,
    payoff_fn,
    lam_f: float | None = None,
):
    """Total loss for VPINN: weak PDE loss + terminal-condition MSE.

    The VPINN weak residual L_f is structurally ~10x smaller than the
    strong-form collocation L_f (because it is an integral average over K
    test functions rather than a pointwise residual).  Pass lam_f explicitly
    to override p3.LAMBDA_F and restore the ~50/50 PDE/TC balance that the
    strong-form variants enjoy.
    """
    lambda_f = lam_f if lam_f is not None else p3.LAMBDA_F
    loss_f = vpinn_module(model, t_batch)
    with torch.no_grad():
        phi = payoff_fn(x_tc)
    u_tc   = model(torch.stack([x_tc, t_tc], dim=1)).squeeze()
    loss_tc = ((u_tc - phi) ** 2).mean()
    total   = lambda_f * loss_f + p3.LAMBDA_TC * loss_tc
    return total, loss_f.item(), loss_tc.item()


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
) -> dict:
    if log_every is None:
        log_every = p3._adaptive_log_every(total_iters)
    model.to(p3.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, p3.build_lr_lambda(total_iters))
    history: dict = {"loss": [], "loss_f": [], "loss_tc": [], "iter": [], "grad_norm": [], "lr": []}
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

        if it % log_every == 0 or it == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            history["loss"].append(loss.item())
            history["loss_f"].append(lf)
            history["loss_tc"].append(ltc)
            history["grad_norm"].append(total_norm)
            history["lr"].append(lr_now)
            history["iter"].append(it)
            logger.info(
                f"[{label}] iter {it:>6d}/{total_iters}  "
                f"loss={loss.item():.4e}  Lf={lf:.4e}  Ltc={ltc:.4e}  "
                f"|g|={total_norm:.2e}  lr={lr_now:.5f}  ({time.time()-t0:.1f}s)"
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
) -> dict:
    """Training loop for the VPINN variant (weak-form PDE loss)."""
    if log_every is None:
        log_every = p3._adaptive_log_every(total_iters)
    model.to(p3.DEVICE)
    vpinn_module.to(p3.DEVICE)
    lambda_f  = lam_f if lam_f is not None else p3.LAMBDA_F
    lambda_tc = p3.LAMBDA_TC
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, p3.build_lr_lambda(total_iters))
    history: dict = {"loss": [], "loss_f": [], "loss_tc": [], "iter": [], "grad_norm": [], "lr": []}
    model.train()
    t0 = time.time()
    logger.info(
        f"[{label}] loss = λ_f·Lf + λ_tc·Ltc  "
        f"with λ_f={lambda_f}, λ_tc={lambda_tc}"
    )

    for it in range(1, total_iters + 1):
        optimizer.zero_grad()
        t_batch, x_tc, t_tc = sampler_fn()
        loss, lf, ltc = compute_losses_vpinn_logS(
            model, t_batch, vpinn_module, x_tc, t_tc, payoff_fn, lam_f=lam_f
        )
        loss.backward()

        total_norm = sum(
            p.grad.detach().norm(2).item() ** 2
            for p in model.parameters() if p.grad is not None
        ) ** 0.5

        optimizer.step()
        scheduler.step()

        if it % log_every == 0 or it == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            history["loss"].append(loss.item())
            history["loss_f"].append(lf)
            history["loss_tc"].append(ltc)
            history["grad_norm"].append(total_norm)
            history["lr"].append(lr_now)
            history["iter"].append(it)
            logger.info(
                f"[{label}] iter {it:>6d}/{total_iters}  "
                f"loss={loss.item():.4e}  "
                f"(λ_f·Lf={lambda_f * lf:.4e}  λ_tc·Ltc={lambda_tc * ltc:.4e})  "
                f"Lf={lf:.4e}  Ltc={ltc:.4e}  "
                f"|g|={total_norm:.2e}  lr={lr_now:.5f}  ({time.time()-t0:.1f}s)"
            )

    model.eval()
    logger.info(f"[{label}] Training done in {time.time()-t0:.1f}s")
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

_GT_TAU_SLICES = [0.02 * T, T / 4, T / 2, 3 * T / 4]   # includes near-singularity slice
_GT_N_X = 120


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

    # Greeks at tau = T/2
    tau_greek = T / 2.0
    t_fix = T - tau_greek
    x_1d = torch.linspace(X_EVAL_LO, X_EVAL_HI, _GT_N_X, device=device).requires_grad_(True)
    t_1d = torch.full((_GT_N_X,), t_fix, device=device).requires_grad_(True)
    V_1d = model(torch.stack([x_1d, t_1d], dim=1)).squeeze()
    (dV_dx,) = torch.autograd.grad(V_1d.sum(), x_1d, create_graph=True)
    (d2V_dx2,) = torch.autograd.grad(dV_dx.sum(), x_1d, create_graph=False)

    with torch.no_grad():
        x_d   = x_1d.detach()
        S_1d  = x_d.exp()
        tau_t = torch.full((_GT_N_X,), tau_greek, device=device)
        d1    = (x_d - math.log(K) + (r + 0.5*sigma**2)*tau_t) / (sigma*tau_t.sqrt())
        sqrt2 = math.sqrt(2.0)
        delta_ref = 0.5 * torch.erfc(-d1 / sqrt2)
        gamma_ref = (
            torch.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
            / (S_1d * sigma * tau_t.sqrt())
        )
        delta_pred = dV_dx.detach()  * (-x_d).exp()
        gamma_pred = (d2V_dx2.detach() - dV_dx.detach()) * (-2 * x_d).exp()

    return {
        "x_vals":        x_vals.cpu().numpy(),
        "tau_slices":    np.array(_GT_TAU_SLICES),
        "V_pred_slices": np.array(V_pred_slices),
        "V_ref_slices":  np.array(V_ref_slices),
        "x_greek":       x_d.cpu().numpy(),
        "delta_pred":    delta_pred.cpu().numpy(),
        "delta_ref":     delta_ref.cpu().numpy(),
        "gamma_pred":    gamma_pred.cpu().numpy(),
        "gamma_ref":     gamma_ref.cpu().numpy(),
    }


# ---------------------------------------------------------------------------
# Persistence  (identical structure to ablation_singularity.py)
# ---------------------------------------------------------------------------

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
    return {**summary_entry, "hist": hist, "metrics": metrics, "gt_slices": gt_slices}


# ---------------------------------------------------------------------------
# Formula annotations
# ---------------------------------------------------------------------------

_BOX_STYLE = dict(boxstyle="round,pad=0.6", facecolor="lightyellow", edgecolor="gray", alpha=0.9)


def _add_formula_box(fig, text: str, bottom_margin: float = 0.20) -> None:
    fig.text(0.5, 0.01, text, ha="center", va="bottom", fontsize=8,
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


def _build_sampler(cfg: dict, n_f: int, n_tc: int):
    t = cfg["sampler_type"]
    if t == "naive":
        return make_sampler_naive_logS(n_f, n_tc)
    if t == "truncated":
        return make_sampler_truncated_logS(n_f, n_tc, eps=cfg["eps"])
    if t == "importance":
        return make_sampler_importance_logS(n_f, n_tc,
                                            sigma_is=cfg["sigma_is"],
                                            mix=cfg["mix"], eps=cfg["eps"])
    if t == "vpinn":
        return make_sampler_vpinn_logS(cfg.get("n_tau", 512), n_tc,
                                       eps=cfg.get("eps", 0.01 * T))
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
    lf_formula = _FORMULA_LF_VPINN if res.get("sampler_type") == "vpinn" else _FORMULA_LF
    _add_formula_box(fig, lf_formula + "\n" + _FORMULA_LTC + "\n" + _FORMULA_GRAD,
                     bottom_margin=0.52)
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
    _add_formula_box(fig, _FORMULA_PDE_TAU, bottom_margin=0.24)
    fig.savefig(out / "pde_residual_tau.png", dpi=150)
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

    # ── Delta & Gamma ─────────────────────────────────────────────────────
    S_greek = np.exp(gt["x_greek"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(S_greek, gt["delta_ref"],  "k--", linewidth=1.5,
                 label=r"$\Delta^{\mathrm{BS}}=N(d_1)$", zorder=10)
    axes[0].plot(S_greek, gt["delta_pred"], color=color, linestyle=ls, linewidth=lw,
                 label=r"$\hat{\Delta}$")
    axes[0].axvline(K, color="gray", linestyle=":", linewidth=0.8)
    axes[0].set_xlabel(r"$S$"); axes[0].set_ylabel(r"$\Delta$")
    axes[0].set_title(r"$\Delta$ at $\tau = T/2$")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    axes[1].plot(S_greek, gt["gamma_ref"],  "k--", linewidth=1.5,
                 label=r"$\Gamma^{\mathrm{BS}}=N'(d_1)/(S\sigma\sqrt{\tau})$", zorder=10)
    axes[1].plot(S_greek, gt["gamma_pred"], color=color, linestyle=ls, linewidth=lw,
                 label=r"$\hat{\Gamma}$")
    axes[1].axvline(K, color="gray", linestyle=":", linewidth=0.8)
    axes[1].set_xlabel(r"$S$"); axes[1].set_ylabel(r"$\Gamma$")
    axes[1].set_title(r"$\Gamma$ at $\tau = T/2$")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"{label} — Greeks vs BS\n{_SUPTITLE}", fontsize=9)
    fig.tight_layout()
    _add_formula_box(fig, _FORMULA_GREEKS_CMP, bottom_margin=0.24)
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

    def _savefig(fig, name, formula, bottom=0.15):
        fig.tight_layout()
        _add_formula_box(fig, formula, bottom_margin=bottom)
        fig.savefig(comp_dir / name, dpi=150)
        plt.close(fig)

    # Loss Lf
    has_vpinn = any(r.get("sampler_type") == "vpinn" for r in results)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, res in enumerate(results):
        ax.semilogy(res["hist"]["iter"], res["hist"]["loss_f"],
                    label=labels[i], color=colors[i],
                    linestyle=linestyles[i], linewidth=linewidths[i])
    ax.set_xlabel("Iteration"); ax.set_ylabel(r"$\mathcal{L}_f$")
    lf_title = r"PDE residual loss $\mathcal{L}_f$  ($x=\ln S$)"
    if has_vpinn:
        lf_title += ("\n"
                     r"[!] VPINN $\mathcal{L}_f$ = integrated weak-form residual "
                     r"— different norm, NOT directly comparable to others")
    ax.set_title(lf_title, fontsize=9)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.suptitle(_SUPTITLE, fontsize=10)
    lf_formula_cmp = _FORMULA_LF
    if has_vpinn:
        lf_formula_cmp = "\n".join([
            _FORMULA_LF,
            r"[!] VPINN uses a weak-form residual integrated over $x$ (different norm)."
            r"  See pde_residual_by_tau.png for a fair (strong-form) comparison.",
        ])
    _savefig(fig, "loss_pde.png", lf_formula_cmp)

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
    _savefig(fig, "grad_norm.png", _FORMULA_GRAD)

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
    _savefig(fig, "pde_residual_by_tau.png", _FORMULA_PDE_TAU)

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
        _add_formula_box(fig2,
            "\n".join([
                r"Left: $\mathcal{L}_f$ during training — "
                r"VPINN uses weak norm $\Vert R \Vert_{L^2_t \times \ell^2_k}$, "
                r"strong-form uses pointwise $\Vert \mathcal{F}[\hat{V}] \Vert_{L^2_{x,t}}$",
                r"Right (fair): $\bar{F}(\tau)=\frac{1}{N}\sum_i|\mathcal{F}[\hat{V}](x=\ln K,\,T-\tau)|$"
                r" — same strong-form operator for all methods",
            ]),
            bottom_margin=0.22,
        )
        fig2.savefig(comp_dir / "fair_comparison_overview.png", dpi=150)
        plt.close(fig2)

    # Metric bar chart
    metric_keys  = ["rel_l2", "rel_l2_atm", "rel_l2_delta", "rel_l2_gamma", "gei"]
    metric_names = [r"$\varepsilon_{L^2}$", r"$\varepsilon_{L^2}^{\mathrm{ATM}}$",
                    r"$\varepsilon_{\Delta}$", r"$\varepsilon_{\Gamma}$", r"GEI"]
    fig, axes = plt.subplots(1, 5, figsize=(20, 6))
    for j, (mk, mn) in enumerate(zip(metric_keys, metric_names)):
        vals = [res["metrics"][mk] for res in results]
        bars = axes[j].bar(range(len(results)), vals, color=colors)
        axes[j].set_xticks(range(len(results)))
        axes[j].set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        axes[j].set_title(mn, fontsize=10); axes[j].set_yscale("log")
        axes[j].grid(axis="y", alpha=0.3)
        for br, val in zip(bars, vals):
            axes[j].text(br.get_x() + br.get_width()/2, val*1.1,
                         f"{val:.2e}", ha="center", va="bottom", fontsize=7)
    fig.suptitle(f"Metric comparison — mode={mode}, {iters} iters\n{_SUPTITLE}", fontsize=10)
    fig.subplots_adjust(bottom=0.44, top=0.88, wspace=0.35)
    fig.text(0.5, 0.01, _FORMULA_METRICS, ha="center", va="bottom",
             fontsize=7.5, bbox=_BOX_STYLE)
    fig.savefig(comp_dir / "metrics_bar.png", dpi=150)
    plt.close(fig)

    # TC loss
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, res in enumerate(results):
        ax.semilogy(res["hist"]["iter"], res["hist"]["loss_tc"],
                    label=labels[i], color=colors[i],
                    linestyle=linestyles[i], linewidth=linewidths[i])
    ax.set_xlabel("Iteration"); ax.set_ylabel(r"$\mathcal{L}_{tc}$")
    ax.set_title(r"Terminal-condition loss $\mathcal{L}_{tc}$")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.suptitle(_SUPTITLE, fontsize=10)
    _savefig(fig, "loss_tc.png", _FORMULA_LTC)

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

    def _save(fig, name, formula, bottom=0.20):
        fig.tight_layout()
        _add_formula_box(fig, formula, bottom_margin=bottom)
        fig.savefig(comp_dir / name, dpi=150)
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

    # ── Greeks comparison ─────────────────────────────────────────────────
    S_greek = np.exp(valid[0]["gt_slices"]["x_greek"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(S_greek, valid[0]["gt_slices"]["delta_ref"], "k--", linewidth=1.5,
                 label=r"$\Delta^{\mathrm{BS}}$", zorder=10)
    axes[1].plot(S_greek, valid[0]["gt_slices"]["gamma_ref"], "k--", linewidth=1.5,
                 label=r"$\Gamma^{\mathrm{BS}}$", zorder=10)
    for r in valid:
        axes[0].plot(S_greek, r["gt_slices"]["delta_pred"],
                     color=r["color"], linestyle=r["linestyle"], linewidth=r["linewidth"],
                     label=r["label"])
        axes[1].plot(S_greek, r["gt_slices"]["gamma_pred"],
                     color=r["color"], linestyle=r["linestyle"], linewidth=r["linewidth"],
                     label=r["label"])
    for ax, ylabel, title in zip(axes,
                                  [r"$\Delta$", r"$\Gamma$"],
                                  [r"$\Delta$ at $\tau=T/2$", r"$\Gamma$ at $\tau=T/2$"]):
        ax.axvline(K, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel(r"$S$"); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle(f"Greeks vs BS — all variants  |  {_SUPTITLE}", fontsize=9)
    _save(fig, "greeks_comparison.png", _FORMULA_GREEKS_CMP, bottom=0.24)


# ---------------------------------------------------------------------------
# Replot
# ---------------------------------------------------------------------------

def _load_model_for_variant(ablation_dir: Path, variant_name: str) -> torch.nn.Module | None:
    """Load a saved PINN for a variant; return None if the checkpoint is missing."""
    model_path = ablation_dir / f"variant_{variant_name}" / "models" / "pinn.pt"
    if not model_path.exists():
        return None
    model = _build_pinn()
    model.load_state_dict(torch.load(model_path, map_location=p3.DEVICE, weights_only=True))
    model.to(p3.DEVICE)
    return model


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

def _train_one_variant(
    v: dict,
    vdir: Path,
    n_f: int,
    n_tc: int,
    total_iters: int,
) -> dict:
    """Build, train, evaluate, and save one variant; return the result dict."""
    model      = _build_pinn()
    sampler_fn = _build_sampler(v, n_f, n_tc)
    payoff_fn  = _build_payoff(v)

    if v["sampler_type"] == "vpinn":
        vpinn_module = _build_vpinn_loss(v)
        hist = train_variant_vpinn(model, vpinn_module, total_iters,
                                   sampler_fn, payoff_fn, v["name"],
                                   lam_f=v.get("lam_f"))
    else:
        hist = train_variant(model, total_iters, sampler_fn, payoff_fn, v["name"])

    metrics   = compute_metrics(model, hist)
    gt_slices = _compute_gt_slices(model)

    torch.save(model.state_dict(), vdir / "models" / "pinn.pt")
    res = {**v, "hist": hist, "metrics": metrics, "gt_slices": gt_slices}
    _save_variant(res, vdir)
    return res


def _summary_entry(v: dict, metrics: dict) -> dict:
    """Build the summary.yaml entry for one variant."""
    return {
        "name": v["name"], "label": v["label"],
        "color": v["color"], "linestyle": v["linestyle"], "linewidth": v["linewidth"],
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

        res = _train_one_variant(v, vdir, n_f, n_tc, meta["iters"])
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
            yaml.dump(summary, f, allow_unicode=True)

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
        yaml.dump({
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
        yaml.dump({"variants": summary_variants}, f, allow_unicode=True)

    _plot_comparison(results, ablation_dir, args.iters, args.mode)
    logger.info(f"\nAll done — results in {ablation_dir}")


if __name__ == "__main__":
    main()
