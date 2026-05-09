"""Ablation study — European call singularity at tau=0, S=K.

Three cases (all use a plain PINN, no g1/g2 ansatz, for fair comparison):

    naive        payoff = clamp(S-K, 0),  PDE on full [0, T]
    truncated    payoff = clamp(S-K, 0),  PDE on [0, T-eps]  (excludes singular zone)
    smooth       payoff = Softplus(S-K, beta) - log(2)/beta,  PDE on full [0, T]

Ablation modes (--mode):
    simple        3 representative variants: naive / truncated(eps=1%T) / smooth(beta=100)
    ablation-eps  naive + 5 truncated variants sweeping eps
    ablation-beta naive + 5 smooth variants sweeping beta
    ablation-is   naive + truncated(eps=1%T) + 3 importance-sampling variants

Reference: exact Black-Scholes call via put-call parity  C = S - K*exp(-r*tau) + P^BS.

Usage (from repo root):
    # Smoke test — primary 3-method comparison, 200 iters, GPU:
    python3 experiments/python_scripts/exp_singularity_european_call/ablation_singularity.py \\
        --iters 200 --device cuda

    # Full 3-method comparison (main result):
    python3 experiments/python_scripts/exp_singularity_european_call/ablation_singularity.py \\
        --iters 30000 --device cuda

    # Sensitivity to epsilon (truncation zone size):
    python3 experiments/python_scripts/exp_singularity_european_call/ablation_singularity.py \\
        --iters 30000 --device cuda --mode ablation-eps

    # Sensitivity to beta (smoothing temperature):
    python3 experiments/python_scripts/exp_singularity_european_call/ablation_singularity.py \\
        --iters 30000 --device cuda --mode ablation-beta

    # Importance-sampling vs uniform:
    python3 experiments/python_scripts/exp_singularity_european_call/ablation_singularity.py \\
        --iters 30000 --device cuda --mode ablation-is

    # Regenerate plots from a saved run (no retraining):
    python3 experiments/python_scripts/exp_singularity_european_call/ablation_singularity.py \\
        --replot data/exp_singularity_european_call/20260509_120000_compare-boundary-singularity-european-call_iters200
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
from learning_option_pricing.pricing.terminal import bsm_operator, black_scholes_put

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model parameters (same as exp1 for comparability)
# ---------------------------------------------------------------------------
K, r, sigma, T, q = p3.K, p3.r, p3.sigma, p3.T, p3.q
S_LO, S_HI = 20.0, 160.0
S_EVAL_LO, S_EVAL_HI = 60.0, 140.0


# ---------------------------------------------------------------------------
# Black-Scholes call reference  (put-call parity, q=0)
# ---------------------------------------------------------------------------

def bs_call(s: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """C = S - K*exp(-r*tau) + P^BS.  tau must be > 0 (clamp externally)."""
    P = black_scholes_put(s, K, r, sigma, tau)
    return s - K * torch.exp(-r * tau) + P


# ---------------------------------------------------------------------------
# Payoff factories
# ---------------------------------------------------------------------------

def payoff_exact(s: torch.Tensor) -> torch.Tensor:
    return torch.clamp(s - K, min=0.0)


def make_payoff_smooth(beta: float):
    log2 = math.log(2.0)
    def _payoff(s: torch.Tensor) -> torch.Tensor:
        # Softplus centred so that Softplus(K, beta) = 0
        return F.softplus(s - K, beta=beta) - log2 / beta
    return _payoff


# ---------------------------------------------------------------------------
# Sampler factories — return callables () -> (s_f, t_f, s_tc, t_tc)
# ---------------------------------------------------------------------------

def make_sampler_naive(n_f: int, n_tc: int):
    """Uniform on full [0, T] — the naive control."""
    def _sample():
        device = p3.DEVICE
        s_f = (torch.rand(n_f, device=device) * (S_HI - S_LO) + S_LO).requires_grad_(True)
        t_f = (torch.rand(n_f, device=device) * T).requires_grad_(True)
        s_tc = torch.rand(n_tc, device=device) * (S_HI - S_LO) + S_LO
        t_tc = torch.full((n_tc,), T, device=device)
        return s_f, t_f, s_tc, t_tc
    return _sample


def make_sampler_truncated(n_f: int, n_tc: int, eps: float):
    """PDE points in [0, T-eps]; terminal condition still at tau=0 exact."""
    def _sample():
        device = p3.DEVICE
        t_max_pde = T - eps
        s_f = (torch.rand(n_f, device=device) * (S_HI - S_LO) + S_LO).requires_grad_(True)
        t_f = (torch.rand(n_f, device=device) * t_max_pde).requires_grad_(True)
        s_tc = torch.rand(n_tc, device=device) * (S_HI - S_LO) + S_LO
        t_tc = torch.full((n_tc,), T, device=device)
        return s_f, t_f, s_tc, t_tc
    return _sample


def make_sampler_importance(n_f: int, n_tc: int, sigma_is: float,
                            mix: float = 0.5, eps: float = 0.0):
    """Mix of uniform + Gaussian concentrated around S=K.

    Args:
        sigma_is: Standard deviation of the Gaussian proposal around K.
        mix:      Fraction of points drawn from the Gaussian proposal.
        eps:      Optional temporal truncation applied simultaneously.
    """
    def _sample():
        device = p3.DEVICE
        n_focal   = int(n_f * mix)
        n_uniform = n_f - n_focal
        s_uniform = torch.rand(n_uniform, device=device) * (S_HI - S_LO) + S_LO
        s_focal   = (K + torch.randn(n_focal, device=device) * sigma_is).clamp(S_LO, S_HI)
        s_f = torch.cat([s_uniform, s_focal]).requires_grad_(True)
        t_max_pde = T - eps
        t_f = (torch.rand(n_f, device=device) * t_max_pde).requires_grad_(True)
        s_tc = torch.rand(n_tc, device=device) * (S_HI - S_LO) + S_LO
        t_tc = torch.full((n_tc,), T, device=device)
        return s_f, t_f, s_tc, t_tc
    return _sample


# ---------------------------------------------------------------------------
# Custom training loop (same as p3.train_model but sampler_fn is injected)
# ---------------------------------------------------------------------------

def train_variant(
    model: torch.nn.Module,
    total_iters: int,
    sampler_fn,
    payoff_fn,
    label: str,
    log_every: int | None = None,
) -> dict:
    """Train a PINN with Adam + two-stage LR, injectable sampler and payoff."""
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
        s_f, t_f, s_tc, t_tc = sampler_fn()
        loss, lf, ltc = p3.compute_losses(
            model, s_f, t_f, s_tc, t_tc, payoff_fn,
            p3.LAMBDA_F, p3.LAMBDA_TC,
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
                f"loss={loss.item():.4e}  Lf={lf:.4e}  Ltc={ltc:.4e}  "
                f"|g|={total_norm:.2e}  lr={lr_now:.5f}  ({time.time()-t0:.1f}s)"
            )

    model.eval()
    logger.info(f"[{label}] Training done in {time.time()-t0:.1f}s")
    return history


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(model: torch.nn.Module, hist: dict) -> dict:
    """Compute evaluation metrics after training.

    Returns a dict with:
        rel_l2          relative L2 error on full evaluation grid
        rel_l2_atm      relative L2 error restricted to S in [0.9K, 1.1K]
        rel_l2_delta    relative L2 error on Delta at tau=0.5
        rel_l2_gamma    relative L2 error on Gamma at tau=0.5
        gei             Gradient Explosion Index = max(|g|) / median(|g|)
        pde_residual_tau list of (tau, mean|F|) along the S=K slice
    """
    device = p3.DEVICE
    model.eval()

    # ── Full price grid ──────────────────────────────────────────────────
    n_s, n_t = 80, 50
    tau_min = 1e-2  # avoid tau=0 where analytical Gamma diverges
    s_vals = torch.linspace(S_EVAL_LO, S_EVAL_HI, n_s, device=device)
    # t in [0, T - tau_min]  →  tau in [tau_min, T]
    t_vals = torch.linspace(0.0, T - tau_min, n_t, device=device)
    S_grid, T_grid = torch.meshgrid(s_vals, t_vals, indexing="ij")   # (n_s, n_t)
    tau_grid = T - T_grid

    with torch.no_grad():
        x_eval = torch.stack([S_grid.reshape(-1), T_grid.reshape(-1)], dim=1)
        V_pred = model(x_eval).squeeze().reshape(n_s, n_t)
        V_ref  = bs_call(S_grid, tau_grid.clamp(min=1e-8))

    err = V_pred - V_ref
    rel_l2 = ((err ** 2).sum() / (V_ref ** 2).sum()).sqrt().item()

    # ATM band ±10 %
    mask_atm = (S_grid >= K * 0.9) & (S_grid <= K * 1.1)
    rel_l2_atm = (
        (err[mask_atm] ** 2).sum() / (V_ref[mask_atm] ** 2).sum()
    ).sqrt().item()

    # ── Greeks at tau = T/2 ──────────────────────────────────────────────
    tau_fix = T / 2.0
    t_fix   = T - tau_fix
    n_greek = 100
    s_1d = torch.linspace(S_EVAL_LO, S_EVAL_HI, n_greek, device=device).requires_grad_(True)
    t_1d = torch.full((n_greek,), t_fix, device=device).requires_grad_(True)
    x_1d = torch.stack([s_1d, t_1d], dim=1)
    V_1d = model(x_1d).squeeze()

    (delta_pred,) = torch.autograd.grad(V_1d.sum(), s_1d, create_graph=True)
    (gamma_pred,) = torch.autograd.grad(delta_pred.sum(), s_1d, create_graph=False)

    with torch.no_grad():
        s_d   = s_1d.detach()
        tau_t = torch.full((n_greek,), tau_fix, device=device)
        d1    = (torch.log(s_d / K) + (r + 0.5 * sigma ** 2) * tau_t) / (sigma * tau_t.sqrt())
        sqrt2 = torch.tensor(2.0, device=device).sqrt()
        delta_ref = 0.5 * torch.erfc(-d1 / sqrt2)                            # N(d1)
        gamma_ref = (
            torch.exp(-0.5 * d1 ** 2) / (2 * math.pi) ** 0.5
            / (s_d * sigma * tau_t.sqrt())
        )

    dp, dp_ref = delta_pred.detach(), delta_ref
    gp, gp_ref = gamma_pred.detach(), gamma_ref
    rel_l2_delta = (((dp - dp_ref) ** 2).sum() / (dp_ref ** 2).sum()).sqrt().item()
    rel_l2_gamma = (((gp - gp_ref) ** 2).sum() / (gp_ref ** 2).sum()).sqrt().item()

    # ── GEI (early training phase, first 2/3 of history) ────────────────
    norms = np.array(hist["grad_norm"])
    cutoff = max(1, int(len(norms) * 2 / 3))
    norms_early = norms[:cutoff]
    gei = float(norms_early.max() / (np.median(norms_early) + 1e-10))

    # ── PDE residual profile along S=K slice ────────────────────────────
    n_tau_profile = 25
    tau_profile = torch.linspace(tau_min, T, n_tau_profile, device=device)
    res_profile = []
    for tau_val in tau_profile:
        t_val = (T - tau_val.item())
        n_pts = 50
        s_p = torch.full((n_pts,), K, device=device).requires_grad_(True)
        t_p = torch.full((n_pts,), t_val, device=device).requires_grad_(True)
        x_p = torch.stack([s_p, t_p], dim=1)
        V_p = model(x_p).squeeze()
        F_p = bsm_operator(V_p, s_p, t_p, r, q, sigma)
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
# Persistence
# ---------------------------------------------------------------------------

def _save_variant(res: dict, vdir: Path) -> None:
    hist = res["hist"]
    np.savez_compressed(
        vdir / "hist.npz",
        **{k: np.array(v) for k, v in hist.items()},
    )
    m = res["metrics"]
    pde_tau  = np.array(m["pde_residual_tau"]["tau"])
    pde_res  = np.array(m["pde_residual_tau"]["residual"])
    np.savez_compressed(vdir / "metrics.npz",
        rel_l2=np.array([m["rel_l2"]]),
        rel_l2_atm=np.array([m["rel_l2_atm"]]),
        rel_l2_delta=np.array([m["rel_l2_delta"]]),
        rel_l2_gamma=np.array([m["rel_l2_gamma"]]),
        gei=np.array([m["gei"]]),
        pde_tau=pde_tau, pde_residual=pde_res,
    )


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
    return {**summary_entry, "hist": hist, "metrics": metrics}


# ---------------------------------------------------------------------------
# Variant catalogue
# ---------------------------------------------------------------------------

_EPS_GRID  = [0.005 * T, 0.01 * T, 0.02 * T, 0.05 * T, 0.10 * T]
_BETA_GRID = [10, 50, 100, 500, 1000]
_IS_CONFIGS = [  # (sigma_is, mix)
    (2.0, 0.5),
    (5.0, 0.5),
    (10.0, 0.5),
]
_COLORS    = ["tab:blue", "tab:orange", "tab:green", "tab:red",
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
        # Primary comparison: the 3 methods with representative default parameters.
        # eps = 1 % T  (avoids singular Gamma region; small enough not to distort the domain)
        # beta = 100   (payoff error < 0.7 % at ATM; bounded Gamma <= 25)
        return [
            naive_cfg,
            dict(name="truncated", label=r"$\varepsilon$-trunc. ($\varepsilon=1\%T$)",
                 sampler_type="truncated", payoff_type="exact",
                 eps=0.01 * T, beta=None, sigma_is=None, mix=0.0,
                 color="tab:orange", linestyle="--", linewidth=2.0),
            dict(name="smooth", label=r"Smooth ($\beta=100$)",
                 sampler_type="naive", payoff_type="smooth",
                 eps=0.0, beta=100, sigma_is=None, mix=0.0,
                 color="tab:green", linestyle="-.", linewidth=2.0),
        ]

    if mode == "ablation-eps":
        variants = [naive_cfg]
        for i, eps in enumerate(_EPS_GRID):
            pct = int(round(eps / T * 100))
            variants.append(dict(
                name=f"trunc_{pct}pct",
                label=rf"$\varepsilon={pct}\%T$",
                sampler_type="truncated", payoff_type="exact",
                eps=eps, beta=None, sigma_is=None, mix=0.0,
                color=_COLORS[i + 1], linestyle="--", linewidth=1.8,
            ))
        return variants

    if mode == "ablation-beta":
        variants = [naive_cfg]
        for i, beta in enumerate(_BETA_GRID):
            variants.append(dict(
                name=f"smooth_b{beta}",
                label=rf"$\beta={beta}$",
                sampler_type="naive", payoff_type="smooth",
                eps=0.0, beta=beta, sigma_is=None, mix=0.0,
                color=_COLORS[i + 1], linestyle="-.", linewidth=1.8,
            ))
        return variants

    if mode == "ablation-is":
        variants = [
            naive_cfg,
            dict(name="trunc_1pct", label=r"Trunc. unif.",
                 sampler_type="truncated", payoff_type="exact",
                 eps=0.01 * T, beta=None, sigma_is=None, mix=0.0,
                 color="tab:orange", linestyle="--", linewidth=2.0),
        ]
        for i, (sig, mix) in enumerate(_IS_CONFIGS):
            variants.append(dict(
                name=f"is_sig{int(sig)}_mix{int(mix*100)}",
                label=rf"IS $\sigma={sig}$, mix={mix}",
                sampler_type="importance", payoff_type="exact",
                eps=0.01 * T, beta=None, sigma_is=sig, mix=mix,
                color=_COLORS[i + 2], linestyle=":", linewidth=1.8,
            ))
        return variants

    raise ValueError(f"Unknown mode: {mode!r}")


def _build_sampler(cfg: dict, n_f: int, n_tc: int):
    t = cfg["sampler_type"]
    if t == "naive":
        return make_sampler_naive(n_f, n_tc)
    if t == "truncated":
        return make_sampler_truncated(n_f, n_tc, eps=cfg["eps"])
    if t == "importance":
        return make_sampler_importance(n_f, n_tc,
                                       sigma_is=cfg["sigma_is"],
                                       mix=cfg["mix"],
                                       eps=cfg["eps"])
    raise ValueError(f"Unknown sampler_type: {t!r}")


def _build_payoff(cfg: dict):
    if cfg["payoff_type"] == "exact":
        return payoff_exact
    if cfg["payoff_type"] == "smooth":
        return make_payoff_smooth(cfg["beta"])
    raise ValueError(f"Unknown payoff_type: {cfg['payoff_type']!r}")


def _build_pinn() -> PINN:
    return PINN(
        resnet=ResNet(d_in=2, d_out=1, n=50, M=4, L=2),
        normalizer=InputNormalization(K),
    )


# ---------------------------------------------------------------------------
# Formula annotations
# ---------------------------------------------------------------------------

_BOX_STYLE = dict(boxstyle="round,pad=0.6", facecolor="lightyellow", edgecolor="gray", alpha=0.9)

def _add_formula_box(fig, text: str, bottom_margin: float = 0.15) -> None:
    """Add a formula text box at the bottom of the figure."""
    fig.text(0.5, 0.01, text, ha="center", va="bottom", fontsize=8,
             bbox=_BOX_STYLE, linespacing=1.6)
    fig.subplots_adjust(bottom=bottom_margin)


_FORMULA_LF = "\n".join([
    r"$\mathcal{L}_f = \frac{1}{N_f}\sum_{i} \mathcal{F}[\hat{V}](S_i,\,t_i)^2$",
    r"BSM operator:  $\mathcal{F}[V] = \partial_t V + \frac{\sigma^2}{2}S^2\,\partial_{SS}V + rS\,\partial_S V - rV$",
])
_FORMULA_LTC = "\n".join([
    r"$\mathcal{L}_{tc} = \frac{1}{N_{tc}}\sum_{i}(\hat{V}(S_i,T)-\Phi(S_i))^2$",
    r"Naïf / trunc.:  $\Phi(S)=(S-K)^{+}$",
    r"Smooth:  $\tilde{\Phi}_\beta(S)=\frac{1}{\beta}\ln(1+e^{\beta(S-K)})-\frac{\ln 2}{\beta}$   (bounded Gamma $\leq\beta/4$)",
])
_FORMULA_GRAD = "\n".join([
    r"$\|\nabla_\theta\mathcal{L}\|_2 = \sqrt{\sum_l \|\nabla_{\theta_l}\mathcal{L}\|_2^2}$",
    r"Total loss:  $\mathcal{L} = \lambda_f\,\mathcal{L}_f + \lambda_{tc}\,\mathcal{L}_{tc}$" +
    rf"   with $\lambda_f={p3.LAMBDA_F}$,  $\lambda_{{tc}}={p3.LAMBDA_TC}$",
])
_FORMULA_PDE_TAU = "\n".join([
    r"$\bar{F}(\tau) = \frac{1}{N}\sum_{i=1}^{N}|\mathcal{F}[\hat{V}](K,\,T-\tau)|$   (50 points at $S=K$ per slice)",
    r"BSM operator:  $\mathcal{F}[V] = \partial_t V + \frac{\sigma^2}{2}S^2\,\partial_{SS}V + rS\,\partial_S V - rV$",
])
_FORMULA_METRICS = "\n".join([
    r"$\varepsilon_{L^2}=\|\hat{V}-C^{\mathrm{BS}}\|_2 / \|C^{\mathrm{BS}}\|_2$   (grid $S\in[60,140]$, $\tau\in[0.01,T]$)",
    r"$\varepsilon_{L^2}^{\mathrm{ATM}}$: same restricted to $S\in[0.9K,\,1.1K]$",
    r"$\varepsilon_\Delta$: rel. $L^2$ of $\partial_S\hat{V}$ vs $\Delta^{\mathrm{BS}}=N(d_1)$ at $\tau=T/2$",
    r"$\varepsilon_\Gamma$: rel. $L^2$ of $\partial_{SS}\hat{V}$ vs $\Gamma^{\mathrm{BS}}=N'(d_1)/(S\sigma\sqrt{\tau})$ at $\tau=T/2$",
    r"$\mathrm{GEI} = \max\|\nabla_\theta\mathcal{L}\| / \mathrm{median}\|\nabla_\theta\mathcal{L}\|$   (first 2/3 of training)",
    r"Reference:  $C^{\mathrm{BS}}=S - Ke^{-r\tau}+P^{\mathrm{BS}}$,   $d_1=[\ln(S/K)+(r+\sigma^2/2)\tau]\,/\,(\sigma\sqrt{\tau})$",
])


# ---------------------------------------------------------------------------
# Per-variant plots
# ---------------------------------------------------------------------------

def _plot_variant(res: dict, vdir: Path) -> None:
    """Save training diagnostic plots for a single variant."""
    out = vdir / "training_metrics"
    out.mkdir(exist_ok=True)
    h     = res["hist"]
    label = res["label"]

    # ── Loss curves ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].semilogy(h["iter"], h["loss"],    color="tab:blue")
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
    _add_formula_box(fig, _FORMULA_LF + "\n" + _FORMULA_LTC + "\n" + _FORMULA_GRAD, bottom_margin=0.30)
    fig.savefig(out / "training_curves.png", dpi=150)
    plt.close(fig)

    # ── PDE residual profile ─────────────────────────────────────────────
    pde = res["metrics"]["pde_residual_tau"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(pde["tau"], pde["residual"], color=res["color"],
                linestyle=res["linestyle"], linewidth=res["linewidth"], marker="o", ms=4)
    ax.axvline(0.0, color="k", linestyle=":", linewidth=0.8)
    ax.set_xlabel(r"$\tau = T - t$")
    ax.set_ylabel(r"$\mathbb{E}_{S=K}[|\mathcal{F}[\hat{V}]|]$")
    ax.set_title(r"PDE residual along $S=K$")
    ax.grid(True, alpha=0.3)
    fig.suptitle(f"{label}\n{_SUPTITLE}", fontsize=10)
    fig.tight_layout()
    _add_formula_box(fig, _FORMULA_PDE_TAU, bottom_margin=0.18)
    fig.savefig(out / "pde_residual_tau.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Comparison plots
# ---------------------------------------------------------------------------

_SUPTITLE = (
    rf"European call — PINN baseline, $K={K}$, $r={r}$, $\sigma={sigma}$, $T={T}$"
)


def _plot_comparison(results: list[dict], ablation_dir: Path, iters: int, mode: str):
    comp_dir = ablation_dir / "comparison"
    comp_dir.mkdir(exist_ok=True)

    colors     = [r["color"]     for r in results]
    linestyles = [r["linestyle"] for r in results]
    labels     = [r["label"]     for r in results]
    linewidths = [r["linewidth"] for r in results]

    # ── Plot 1: PDE residual loss curves ────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, res in enumerate(results):
        h = res["hist"]
        ax.semilogy(h["iter"], h["loss_f"],
                    label=labels[i], color=colors[i],
                    linestyle=linestyles[i], linewidth=linewidths[i])
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\mathcal{L}_f$")
    ax.set_title(r"PDE residual loss $\mathcal{L}_f$")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.suptitle(_SUPTITLE, fontsize=10)
    fig.tight_layout()
    _add_formula_box(fig, _FORMULA_LF)
    fig.savefig(comp_dir / "loss_pde.png", dpi=150)
    plt.close(fig)

    # ── Plot 2: Gradient norm ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, res in enumerate(results):
        h = res["hist"]
        ax.semilogy(h["iter"], h["grad_norm"],
                    label=labels[i], color=colors[i],
                    linestyle=linestyles[i], linewidth=linewidths[i], alpha=0.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\|\nabla_\theta \mathcal{L}\|_2$")
    ax.set_title("Gradient norm — signature of singularity instability")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.suptitle(_SUPTITLE, fontsize=10)
    fig.tight_layout()
    _add_formula_box(fig, _FORMULA_GRAD)
    fig.savefig(comp_dir / "grad_norm.png", dpi=150)
    plt.close(fig)

    # ── Plot 3: PDE residual profile along tau (S=K slice) ──────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, res in enumerate(results):
        pde = res["metrics"]["pde_residual_tau"]
        ax.semilogy(pde["tau"], pde["residual"],
                    label=labels[i], color=colors[i],
                    linestyle=linestyles[i], linewidth=linewidths[i], marker="o", ms=3)
    ax.axvline(0.0, color="k", linestyle=":", linewidth=0.8, label=r"$\tau=0$ (singular)")
    ax.set_xlabel(r"$\tau = T - t$")
    ax.set_ylabel(r"$\mathbb{E}_{S=K}[|\mathcal{F}[\hat{V}]|]$")
    ax.set_title(r"Mean PDE residual along $S=K$ as a function of $\tau$")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.suptitle(_SUPTITLE, fontsize=10)
    fig.tight_layout()
    _add_formula_box(fig, _FORMULA_PDE_TAU)
    fig.savefig(comp_dir / "pde_residual_by_tau.png", dpi=150)
    plt.close(fig)

    # ── Plot 4: Metric bar chart ─────────────────────────────────────────
    metric_keys  = ["rel_l2", "rel_l2_atm", "rel_l2_delta", "rel_l2_gamma", "gei"]
    metric_names = [
        r"$\varepsilon_{L^2}$ (global)",
        r"$\varepsilon_{L^2}^{\mathrm{ATM}}$",
        r"$\varepsilon_{\Delta}$",
        r"$\varepsilon_{\Gamma}$",
        r"GEI",
    ]
    n_metrics = len(metric_keys)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 6))
    for j, (mk, mn) in enumerate(zip(metric_keys, metric_names)):
        vals = [res["metrics"][mk] for res in results]
        bars = axes[j].bar(range(len(results)), vals, color=colors)
        axes[j].set_xticks(range(len(results)))
        axes[j].set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        axes[j].set_title(mn, fontsize=10)
        axes[j].set_yscale("log")
        axes[j].grid(axis="y", alpha=0.3)
        for bar_rect, val in zip(bars, vals):
            axes[j].text(
                bar_rect.get_x() + bar_rect.get_width() / 2,
                val * 1.1, f"{val:.2e}", ha="center", va="bottom", fontsize=7,
            )
    fig.suptitle(f"Metric comparison — mode={mode}, {iters} iters\n{_SUPTITLE}", fontsize=10)
    fig.subplots_adjust(bottom=0.38, top=0.88, wspace=0.35)
    fig.text(0.5, 0.01, _FORMULA_METRICS,
             ha="center", va="bottom", fontsize=7.5, bbox=_BOX_STYLE)
    fig.savefig(comp_dir / "metrics_bar.png", dpi=150)
    plt.close(fig)

    # ── Plot 5: TC loss curves ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, res in enumerate(results):
        h = res["hist"]
        ax.semilogy(h["iter"], h["loss_tc"],
                    label=labels[i], color=colors[i],
                    linestyle=linestyles[i], linewidth=linewidths[i])
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\mathcal{L}_{tc}$")
    ax.set_title(r"Terminal-condition loss $\mathcal{L}_{tc}$")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.suptitle(_SUPTITLE, fontsize=10)
    fig.tight_layout()
    _add_formula_box(fig, _FORMULA_LTC)
    fig.savefig(comp_dir / "loss_tc.png", dpi=150)
    plt.close(fig)

    logger.info(f"Comparison plots saved to {comp_dir}/")


# ---------------------------------------------------------------------------
# Replot mode
# ---------------------------------------------------------------------------

def _replot(ablation_dir: Path) -> None:
    summary_path = ablation_dir / "summary.yaml"
    meta_path    = ablation_dir / "metadata.yaml"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.yaml not found in {ablation_dir}")
    with open(summary_path) as f:
        summary = yaml.safe_load(f)
    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    results = []
    for entry in summary["variants"]:
        vdir = ablation_dir / f"variant_{entry['name']}"
        results.append(_load_variant(vdir, entry))
    _plot_comparison(results, ablation_dir, meta["iters"], meta["mode"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation — European call PINN singularity study"
    )
    parser.add_argument("--iters",  type=int,   default=200,
                        help="Training iterations per variant (default 200 — smoke test)")
    parser.add_argument("--mode",   type=str,   default="compare-boundary-singularity-european-call",
                        choices=["compare-boundary-singularity-european-call", "ablation-eps", "ablation-beta", "ablation-is"],
                        help="Which variant set to run (default: compare-boundary-singularity-european-call — primary 3-method comparison)")
    parser.add_argument("--device", type=str,   default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--n-tc",   type=int,   default=None, help="Override N_TC")
    parser.add_argument("--n-f",    type=int,   default=None, help="Override N_F")
    parser.add_argument("--replot", type=str,   default=None, metavar="DIR",
                        help="Regenerate plots from an existing ablation directory.")
    args = parser.parse_args()

    if args.replot is not None:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                            datefmt="%H:%M:%S")
        _replot(Path(args.replot))
        return

    p3._apply_device_arg(args.device)
    n_tc = args.n_tc if args.n_tc is not None else p3.N_TC
    n_f  = args.n_f  if args.n_f  is not None else p3.N_F

    timestamp    = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    ablation_dir = (
        Path("data/exp_singularity_european_call")
        / f"{timestamp}_{args.mode}_iters{args.iters}"
    )
    ablation_dir.mkdir(parents=True, exist_ok=True)
    (ablation_dir / "comparison").mkdir(exist_ok=True)

    variants = _build_variants(args.mode)
    for v in variants:
        for sub in ("training_metrics", "models"):
            (ablation_dir / f"variant_{v['name']}" / sub).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(ablation_dir / "ablation.log"),
        ],
    )
    logging.getLogger("matplotlib.mathtext").setLevel(logging.WARNING)

    logger.info(f"exp_singularity_european_call  mode={args.mode}  iters={args.iters}")
    logger.info(f"device={p3.DEVICE}  N_TC={n_tc}  N_F={n_f}")
    logger.info(f"output: {ablation_dir}")
    logger.info(f"variants: {[v['name'] for v in variants]}")

    # Save metadata
    with open(ablation_dir / "metadata.yaml", "w") as f:
        yaml.dump({"mode": args.mode, "iters": args.iters,
                   "device": str(p3.DEVICE), "n_tc": n_tc, "n_f": n_f,
                   "K": K, "r": r, "sigma": sigma, "T": T}, f)

    results = []
    summary_variants = []

    for v in variants:
        vdir   = ablation_dir / f"variant_{v['name']}"
        logger.info(f"\n{'='*60}\n  Variant: {v['name']} — {v['label']}\n{'='*60}")

        model      = _build_pinn()
        sampler_fn = _build_sampler(v, n_f, n_tc)
        payoff_fn  = _build_payoff(v)

        hist    = train_variant(model, args.iters, sampler_fn, payoff_fn, v["name"])
        metrics = compute_metrics(model, hist)

        torch.save(model.state_dict(), vdir / "models" / "pinn.pt")

        res = {**v, "hist": hist, "metrics": metrics}
        _save_variant(res, vdir)
        _plot_variant(res, vdir)
        results.append(res)

        m = metrics
        logger.info(
            f"[{v['name']}]  rel_L2={m['rel_l2']:.3e}  "
            f"rel_L2_ATM={m['rel_l2_atm']:.3e}  "
            f"eps_Delta={m['rel_l2_delta']:.3e}  "
            f"eps_Gamma={m['rel_l2_gamma']:.3e}  "
            f"GEI={m['gei']:.2f}"
        )
        summary_variants.append({
            "name":          v["name"],
            "label":         v["label"],
            "color":         v["color"],
            "linestyle":     v["linestyle"],
            "linewidth":     v["linewidth"],
            "sampler_type":  v["sampler_type"],
            "payoff_type":   v["payoff_type"],
            "eps":           v["eps"],
            "beta":          v["beta"],
            "sigma_is":      v["sigma_is"],
            "mix":           v["mix"],
            **{k: v for k, v in m.items() if k != "pde_residual_tau"},
        })

    with open(ablation_dir / "summary.yaml", "w") as f:
        yaml.dump({"variants": summary_variants}, f, allow_unicode=True)

    _plot_comparison(results, ablation_dir, args.iters, args.mode)
    logger.info(f"\nAll done — results in {ablation_dir}")


if __name__ == "__main__":
    main()
