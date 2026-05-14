"""Ablation study — Bermuda put, terminal-condition enforcement method.

This script isolates the effect of *how* the terminal condition (TC) at t=t1
is enforced in Stage B, keeping the ansatz design fixed.

Fixed settings for all variants:
    Stage A:     hard-enforced ETCNN on [t1, T]  (g2_type="bs")
    Stage B ansatz: NO singularity extraction (put_ansatz=False, pchip interpolation)
                    This keeps the two ablations (ansatz vs formulation) orthogonal.
    LAMBDA_F:    20.0  (PDE loss weight, same as ablation_bermudan.py)

Ablation axis (Stage B):
    TC enforcement method — hard (ETCNN architecture) vs soft (penalty term)

Variants:
    hard_etcnn         ETCNN ansatz: TC enforced exactly via g1(s,t)=(t1-t) factor
    soft_pinn_lam10    Plain PINN + penalty  lambda_tc_soft = 10
    soft_pinn_lam100   Plain PINN + penalty  lambda_tc_soft = 100
    soft_pinn_lam1000  Plain PINN + penalty  lambda_tc_soft = 1000

The ``hard_etcnn`` variant uses ``bermudan_problem()`` from phase3_training,
which also trains Stage A and saves etcnn_a.pt.  All soft-penalty variants
then load the *same* etcnn_a.pt so that Stage A is shared and only Stage B
differs.

For the soft PINN Stage B:
  - Model: plain PINN (ResNet + InputNormalization, no ETCNN wrapper)
  - Loss:  LAMBDA_F * L_f  +  lambda_tc_soft * L_tc
      L_f  = (1/N_f) Σ |F[V](s_i, t_i)|²   BSM residual on [0, t1]
      L_tc = (1/N_tc) Σ |V(s_j, t1) - V_target(s_j)|²   soft BC at t=t1
      V_target(s) = max(payoff(s), etcnn_a(s, t1))   (Bermudean TC)

Usage (from repo root):
    # Smoke test — 50 iters each stage:
    python3 experiments/python_scripts/exp1/ablation_bermudan_formulation.py

    # Full run — 2000 iters each stage (recommended):
    python3 experiments/python_scripts/exp1/ablation_bermudan_formulation.py \\
        --iters-a 2000 --iters-b 2000 --device cuda

    # Reuse a pre-trained Stage A to save time:
    python3 experiments/python_scripts/exp1/ablation_bermudan_formulation.py \\
        --iters-b 2000 --load-stage-a data/ablation_bermudan/<run>/variant_baseline/models/etcnn_a.pt

    # Regenerate all plots from a saved run (no retraining):
    python3 experiments/python_scripts/exp1/ablation_bermudan_formulation.py \\
        --replot data/ablation_bermudan_formulation/<timestamp>_itersA...
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import phase3_training as p3
from phase3_training import bermudan_problem
from learning_option_pricing.models.etcnn import PINN, InputNormalization
from learning_option_pricing.models.resnet import ResNet
from learning_option_pricing.utils.run_context import script_data_dir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

# Each variant carries an explicit ``mode`` field used to partition the
# catalogue into independent ablation regimes:
#   * "tc-enforcement"  — hard ETCNN ansatz vs soft PINN penalty (the
#                         original axis of this script).
#   * "mollifiers"      — analytical-Stage-A bermudan study testing how
#                         the bermudan TC max((K-s)+, V^E(s, t1)) is
#                         mollified into a smooth g_2^B(s, t).  Stage A is
#                         the exact Black-Scholes European put — no NN.
# The CLI flag --mode selects which subset is iterated over.
VARIANTS: list[dict] = [
    # ── Mode: tc-enforcement ─────────────────────────────────────────────────
    {
        "name":           "hard_etcnn",
        "label":          "Hard BC (ETCNN ansatz)",
        "tc_type":        "hard",
        "lambda_tc_soft": None,
        "mode":           "tc-enforcement",
        "color":          "tab:blue",
        "linestyle":      "-",
        "linewidth":      2.5,
    },
    {
        "name":           "soft_pinn_lam10",
        "label":          r"Soft BC (PINN, $\lambda_{tc}=10$)",
        "tc_type":        "soft",
        "lambda_tc_soft": 10.0,
        "mode":           "tc-enforcement",
        "color":          "tab:orange",
        "linestyle":      "--",
        "linewidth":      2.0,
    },
    {
        "name":           "soft_pinn_lam100",
        "label":          r"Soft BC (PINN, $\lambda_{tc}=100$)",
        "tc_type":        "soft",
        "lambda_tc_soft": 100.0,
        "mode":           "tc-enforcement",
        "color":          "tab:green",
        "linestyle":      "-.",
        "linewidth":      2.0,
    },
    {
        "name":           "soft_pinn_lam1000",
        "label":          r"Soft BC (PINN, $\lambda_{tc}=1000$)",
        "tc_type":        "soft",
        "lambda_tc_soft": 1000.0,
        "mode":           "tc-enforcement",
        "color":          "tab:red",
        "linestyle":      ":",
        "linewidth":      2.0,
    },
    # ── Mode: mollifiers ─────────────────────────────────────────────────────
    # Companion family to the European-call study in
    # exp_singularity_european_call/_ablation_catalogue.py
    # (hard-ic-ansatz-european-call mode).  Each Stage B variant is an
    # ETCNN with V_theta(s, t) = (t1 - t)/t1 * NN(s, t) + g_2^B(s, t),
    # where g_2^B mollifies the Bermudan TC
    #     V_target(s) = max((K-s)+, V^E_BS(s, t1))
    # with a chosen mollifier family.  Stage A is the exact Black-Scholes
    # European put (no NN trained); tc_type "mollifier_*" triggers
    # train_stage_b_mollifier_ansatz (see below).
    {
        "name":           "bermudan_naive_max",
        "label":          r"Mollifier ansatz — Naïve $\max(\Phi, V^E)$",
        "tc_type":        "mollifier_naive",
        "lambda_tc_soft": None,
        "mode":           "mollifiers",
        "color":          "#0d47a1",  # blue-900 — naive baseline (cf hard_ic_naive)
        "linestyle":      "-",
        "linewidth":      2.5,
    },
    {
        "name":           "bermudan_softplus",
        "label":          r"Mollifier ansatz — Softplus ($\beta=100$)",
        "tc_type":        "mollifier_softplus",
        "lambda_tc_soft": None,
        "mode":           "mollifiers",
        "moll_beta":      100.0,
        "color":          "#42a5f5",  # blue-400 (cf hard_ic_smooth)
        "linestyle":      "-",
        "linewidth":      2.0,
    },
    {
        "name":           "bermudan_cm_static",
        "label":          r"Mollifier ansatz — Chen–Mangasarian static ($\varepsilon=1$)",
        "tc_type":        "mollifier_cm_static",
        "lambda_tc_soft": None,
        "mode":           "mollifiers",
        "moll_eps":       1.0,
        "color":          "#c2185b",  # pink-700 (cf hard_ic_cm_static)
        "linestyle":      "-",
        "linewidth":      2.0,
    },
    {
        "name":           "bermudan_cm_time",
        "label":          r"Mollifier ansatz — Chen–Mangasarian time-dep ($\varepsilon_0=1$, linear)",
        "tc_type":        "mollifier_cm_time",
        "lambda_tc_soft": None,
        "mode":           "mollifiers",
        "moll_eps":       1.0,
        "color":          "#388e3c",  # green-700 (cf hard_ic_cm_time)
        "linestyle":      "-",
        "linewidth":      2.0,
    },
    {
        "name":           "bermudan_cm_time_noisy",
        "label":          r"Mollifier ansatz — CM time-dep, noisy $V^E$ ($\sigma=1\%$)",
        "tc_type":        "mollifier_cm_time_noisy",
        "lambda_tc_soft": None,
        "mode":           "mollifiers",
        "moll_eps":       1.0,
        "noise_sigma_frac": 0.01,  # smooth Gaussian random field amplitude as fraction of |V^E|_max
        "noise_n_modes":  8,        # number of Fourier modes
        "noise_seed":     0,        # fixed seed so the perturbation is reproducible across runs
        "color":          "#7b1fa2",  # purple-700 — distinct from the noise-free cm_time green
        "linestyle":      "--",
        "linewidth":      2.0,
    },
]

_STYLE_BY_NAME: dict[str, dict] = {v["name"]: v for v in VARIANTS}
_FALLBACK_COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:red",
    "tab:purple", "tab:brown", "tab:pink", "tab:gray",
]

_SUPTITLE_PARAMS = (
    r"Bermudan put — $g_2^{(A)}=V^{\mathrm{BS}}$, no extraction ansatz, "
    r"$K=100$, $r=0.02$, $\sigma=0.25$, $T=1$, $t_1=0.5$, $\lambda_f=20$"
)


# ---------------------------------------------------------------------------
# Formula annotations
# ---------------------------------------------------------------------------

_BOX_STYLE = dict(
    boxstyle="round,pad=0.6", facecolor="lightyellow",
    edgecolor="gray", alpha=0.9,
)

_FORMULA_LF_B = "\n".join([
    r"$\mathcal{L}_f = \frac{1}{N_f}\sum_{i} \mathcal{F}[V_\theta](S_i,\,t_i)^2$",
    r"BSM operator:  $\mathcal{F}[V] = \partial_t V + \frac{\sigma^2}{2}S^2\,\partial_{SS}V + rS\,\partial_S V - rV$",
    r"Collocation: $(S_i, t_i) \sim \mathrm{Uniform}([S_{\min}, S_{\max}] \times [0, t_1])$",
])
_FORMULA_TC = "\n".join([
    r"Hard BC (ETCNN): $V_\theta(S, t_1) \equiv V_{\mathrm{target}}(S)$ exactly via ansatz $g_1(S,t)=(t_1-t)$",
    r"Soft BC (PINN):  $\mathcal{L}_{tc} = \frac{1}{N_{tc}}\sum_j (V_\theta(S_j, t_1) - V_{\mathrm{target}}(S_j))^2$",
    r"$V_{\mathrm{target}}(S) = \max(\Phi(S),\, V^{(A)}_{\bar{\theta}}(S, t_1))$   (Bermudean exercise condition at $t_1$)",
    r"Total soft loss: $\mathcal{L} = \lambda_f\,\mathcal{L}_f + \lambda_{tc}\,\mathcal{L}_{tc}$",
])
_FORMULA_GRAD = "\n".join([
    r"$\|\nabla_\theta\mathcal{L}\|_2 = \sqrt{\sum_l \|\nabla_{\theta_l}\mathcal{L}\|_2^2}$",
    rf"$\lambda_f = {p3.LAMBDA_F}$ (PDE weight, fixed for all variants)",
    r"$\lambda_{tc}$ varies per soft variant; hard variant: $\lambda_{tc} \to \infty$ (exact)",
])
_FORMULA_METRICS = "\n".join([
    r"$\varepsilon_{L^2} = \|V_\theta(\cdot,0) - V^{\mathrm{BT}}(\cdot,0)\|_2\,/\,\|V^{\mathrm{BT}}(\cdot,0)\|_2$  (grid $S\in[60,120]$)",
    r"$\varepsilon_{L^2}^{\mathrm{ATM}}$: same restricted to $S\in[0.9K,\,1.1K]$",
    r"$\varepsilon_\Delta$: rel.\ $L^2$ of $\partial_S V_\theta(\cdot,0)$ vs finite-diff BT Delta",
    r"$\mathrm{GEI} = \max\|\nabla_\theta\mathcal{L}\| / \mathrm{median}\|\nabla_\theta\mathcal{L}\|$   (first 2/3 of Stage B training)",
])
_FORMULA_PDE_PROFILE = "\n".join([
    r"$\bar{F}(t) = \frac{1}{N}\sum_{i=1}^{N}|\mathcal{F}[V_\theta](K,\,t)|$   ($N=50$ points at $S=K$ per slice)",
    r"Profile computed along the $S=K$ slice for $t\in[0,\,t_1]$",
])
_FORMULA_TC_ERROR = "\n".join([
    r"TC error at $t=t_1$: $\frac{1}{N}\sum_j |V_\theta(S_j, t_1) - V_{\mathrm{target}}(S_j)|$",
    r"Evaluated on the training grid at $t=t_1$ (same points as $\mathcal{L}_{tc}$ but mean absolute, not squared).",
    r"Hard BC: error $\equiv 0$ by construction.  Soft BC: error decreases with $\lambda_{tc}$.",
])


def _add_formula_box(fig, text: str, bottom_margin: float = 0.18) -> None:
    fig.text(0.5, 0.01, text, ha="center", va="bottom", fontsize=8,
             bbox=_BOX_STYLE, linespacing=1.6)
    fig.subplots_adjust(bottom=bottom_margin)


# ---------------------------------------------------------------------------
# Build a plain PINN model for soft Stage B variants
# ---------------------------------------------------------------------------

def _build_soft_pinn() -> PINN:
    """Construct the same-capacity PINN as the ETCNN Stage B ResNet backbone."""
    return PINN(
        resnet=ResNet(d_in=2, d_out=1, n=p3.n, M=p3.M, L=p3.L_BLOCK),
        normalizer=InputNormalization(p3.K),
    )


# ---------------------------------------------------------------------------
# Mollifier helpers for the "mollifiers" mode (Stage A analytical).
# These define g_2^B(s, t) for the Bermudan TC max((K-s)+, V^E(s, t1)) with
# different mollification strategies — the bermudan analogue of the European-
# call hard-IC variants in exp_singularity_european_call/_ablation_catalogue.py.
# ---------------------------------------------------------------------------

# ε_safe (price^2 units) added under every sqrt that could see a zero radicand
# at (s, t) = (s*, t1).  Same value as the European hard_ic_smooth_t / cm_time
# variants so the two studies remain fair (price-domain bias of magnitude
# eps_safe / 2 = 0.5 at the kink, decaying to <0.01 once |a - b| > 1).
_MOLL_EPS_SAFE = 1.0


def _bs_european_put_torch(s: torch.Tensor, t1: float) -> torch.Tensor:
    """Analytic Black-Scholes European put at (s, t1), maturity p3.T."""
    K  = float(p3.K)
    r  = float(p3.r)
    sigma = float(p3.sigma)
    tau = float(p3.T) - t1  # remaining maturity, scalar > 0
    # Vectorised over s; broadcast tau.
    sqrt_tau = math.sqrt(tau)
    d1 = (torch.log(s / K) + (r + 0.5 * sigma * sigma) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    sqrt2 = math.sqrt(2.0)
    Nm_d1 = 0.5 * (1.0 - torch.erf(d1 / sqrt2))
    Nm_d2 = 0.5 * (1.0 - torch.erf(d2 / sqrt2))
    return K * math.exp(-r * tau) * Nm_d2 - s * Nm_d1


def _smooth_gaussian_field(noise_seed: int, n_modes: int, sigma_frac: float,
                            v_e_max: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a smooth Gaussian random field xi(s) = sigma * sum_k a_k sin(k_k s + phi_k).

    A single realisation is drawn once at the seed and frozen for the run, so
    g_2^B stays smooth in s (deterministic perturbation per query, not Monte
    Carlo).  The bandwidth scales with the asset-price domain so the
    perturbation has typical wavelength ~ (S_HI - S_LO) / max_k k.

    Args:
        noise_seed:  RNG seed (variant-level, fixed per variant for repro).
        n_modes:     Number of sinusoidal modes summed.
        sigma_frac:  Target RMS amplitude as a fraction of |V^E|_max.
        v_e_max:     Scale used to anchor sigma to V^E magnitudes.

    Returns:
        Callable s -> xi(s) (tensor in same dtype/device as s).
    """
    rng = np.random.default_rng(noise_seed)
    # Frequencies span low end of the spectrum: wavelengths from ~ (full domain)
    # down to ~ (full domain) / n_modes.  Domain length on which we anchor.
    S_LO = float(p3.S_TRAIN_LO)
    S_HI = float(p3.S_TRAIN_HI)
    L = S_HI - S_LO
    base_k = 2.0 * math.pi / L  # fundamental
    ks_np     = base_k * (1.0 + np.arange(n_modes))
    phases_np = rng.uniform(0.0, 2.0 * math.pi, size=n_modes)
    amps_np   = rng.normal(loc=0.0, scale=1.0, size=n_modes)
    # Normalise so the empirical std of xi on a dense grid is sigma = sigma_frac * v_e_max.
    s_dense = np.linspace(S_LO, S_HI, 4000)
    xi_dense = np.zeros_like(s_dense)
    for a, k, ph in zip(amps_np, ks_np, phases_np):
        xi_dense += a * np.sin(k * s_dense + ph)
    std_pre = xi_dense.std() if xi_dense.std() > 1e-12 else 1.0
    target_sigma = sigma_frac * v_e_max
    amps_np = amps_np * (target_sigma / std_pre)
    logger.info(
        f"  noise field: n_modes={n_modes}  sigma_frac={sigma_frac}  "
        f"sigma_target={target_sigma:.3e}  seed={noise_seed}  "
        f"k_range=[{ks_np[0]:.3f}, {ks_np[-1]:.3f}]"
    )

    ks_t = torch.tensor(ks_np, dtype=torch.get_default_dtype())
    ph_t = torch.tensor(phases_np, dtype=torch.get_default_dtype())
    am_t = torch.tensor(amps_np, dtype=torch.get_default_dtype())

    def xi(s: torch.Tensor) -> torch.Tensor:
        # Broadcast: s (N,) and modes (M,) -> (N, M) -> (N,)
        ks  = ks_t.to(s.device, dtype=s.dtype)
        ph  = ph_t.to(s.device, dtype=s.dtype)
        am  = am_t.to(s.device, dtype=s.dtype)
        return (am * torch.sin(ks * s.unsqueeze(-1) + ph)).sum(dim=-1)

    return xi


def _build_mollifier_g2(variant: dict) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build g_2^B(s, t) for one of the ``mollifier_*`` Stage B variants.

    Common structure across all four mollifier families:
        g_2^B(s, t) = M_epsilon(Phi(s, t),  V^E_noisy(s))
    where
        Phi(s, t)   — the put payoff (K - s)+, optionally mollified in s
        V^E_noisy   — analytic Black-Scholes European put at t1, optionally
                       perturbed by a smooth Gaussian random field xi(s).
        M_epsilon   — smooth max:
                       naive      : exact max
                       softplus   : (1/beta) log(e^(beta a) + e^(beta b))
                       cm_static  : 0.5 (a+b + sqrt((a-b)^2 + eps^2))
                       cm_time    : 0.5 (a+b + sqrt((a-b)^2 + eps(t)^2 + eps_safe))

    Returns a function (s, t) -> g_2^B (both inputs (N,) tensors).
    """
    tc_type = variant["tc_type"]
    K       = float(p3.K)
    t1      = float(p3.t1)
    eps0    = float(variant.get("moll_eps", 1.0))
    beta    = float(variant.get("moll_beta", 100.0))
    eps_safe = _MOLL_EPS_SAFE

    # ── V^E_noisy(s) ─────────────────────────────────────────────────────
    if tc_type == "mollifier_cm_time_noisy":
        # Realise the smooth random field once.  Anchor amplitude to the
        # ATM European put value so sigma_frac is dimensionless-meaningful.
        s_atm = torch.tensor([K], dtype=torch.get_default_dtype())
        v_e_atm = float(_bs_european_put_torch(s_atm, t1).item())
        xi_fn = _smooth_gaussian_field(
            noise_seed=int(variant.get("noise_seed", 0)),
            n_modes=int(variant.get("noise_n_modes", 8)),
            sigma_frac=float(variant.get("noise_sigma_frac", 0.01)),
            v_e_max=max(abs(v_e_atm), 1e-6),
        )

        def v_e_noisy(s: torch.Tensor) -> torch.Tensor:
            return _bs_european_put_torch(s, t1) + xi_fn(s)
    else:
        def v_e_noisy(s: torch.Tensor) -> torch.Tensor:
            return _bs_european_put_torch(s, t1)

    # ── Phi(s, t) — put payoff, optionally smoothed in s ─────────────────
    # For the bermudan max((K-s)+, V^E), the (K-s)+ kink at s = K is hidden
    # beneath V^E (V^E(K, t1) > 0 = (K-s)+(K)), so we leave (K-s)+ as-is for
    # every variant.  The only kink visible in V_target is the exercise
    # boundary s = s*, which is what the M_epsilon smoothing addresses.
    def Phi(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.clamp(K - s, min=0.0)

    # ── M_epsilon — the smoothed max ─────────────────────────────────────
    if tc_type == "mollifier_naive":
        def g2_b(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return torch.maximum(Phi(s, t), v_e_noisy(s))
    elif tc_type == "mollifier_softplus":
        # Stable log-sum-exp form of the soft-max:
        #     M_beta(a, b) = (1/beta) log(e^{beta a} + e^{beta b})
        #                 = M + (1/beta) log(e^{beta (a-M)} + e^{beta (b-M)})
        def g2_b(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            a = Phi(s, t)
            b = v_e_noisy(s)
            M = torch.maximum(a, b)
            return M + torch.log(
                torch.exp(beta * (a - M)) + torch.exp(beta * (b - M))
            ) / beta
    elif tc_type == "mollifier_cm_static":
        eps_sq = eps0 * eps0

        def g2_b(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            a = Phi(s, t)
            b = v_e_noisy(s)
            diff = a - b
            return 0.5 * (a + b + torch.sqrt(diff * diff + eps_sq))
    elif tc_type in ("mollifier_cm_time", "mollifier_cm_time_noisy"):
        def g2_b(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            a = Phi(s, t)
            b = v_e_noisy(s)
            eps_t = eps0 * (t1 - t) / t1
            diff = a - b
            radicand = diff * diff + eps_t * eps_t + eps_safe
            return 0.5 * (a + b + torch.sqrt(radicand.clamp(min=0.0)))
    else:
        raise ValueError(
            f"_build_mollifier_g2: unknown tc_type {tc_type!r}"
        )

    return g2_b


# ---------------------------------------------------------------------------
# Stage B training for mollifier ansatze (no soft TC penalty — hard IC via
# ETCNN construction).  Mirrors the Adam + cosine-LR schedule of
# train_stage_b_soft_pinn so wall-clock and metrics stay comparable.
# ---------------------------------------------------------------------------

def train_stage_b_mollifier_ansatz(
    variant: dict,
    total_iters: int,
    log_every: int | None = None,
) -> tuple[torch.nn.Module, dict]:
    """Train an ETCNN Stage B model with a mollified g_2^B(s, t).

    The ansatz is V_theta(s, t) = (t1 - t)/t1 * NN(s, t) + g_2^B(s, t),
    so the Bermudan TC is enforced exactly (up to the mollifier bias) at
    t = t1 and the network only sees a PDE-residual loss on [0, t1].
    """
    from learning_option_pricing.models.etcnn import ETCNN

    if log_every is None:
        log_every = p3._adaptive_log_every(total_iters)

    label = variant["name"]
    t1    = float(p3.t1)

    # ── Ansatz construction ─────────────────────────────────────────────────
    def g1(s, t):
        return (t1 - t) / t1

    g2_b = _build_mollifier_g2(variant)
    resnet = ResNet(d_in=2, d_out=1, n=p3.n, M=p3.M, L=p3.L_BLOCK)
    normalizer = InputNormalization(p3.K)
    model = ETCNN(resnet=resnet, g1=g1, g2=g2_b, normalizer=normalizer)
    model.to(p3.DEVICE)

    # ── Optimiser + LR schedule (same as soft variants) ─────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, p3.build_lr_lambda(total_iters)
    )

    history: dict = {
        "loss": [], "loss_f": [], "loss_tc": [],
        "iter": [], "grad_norm": [], "lr": [],
        "tc_enforced": True,
    }
    best_loss = float("inf")
    best_state: dict | None = None
    best_iter = 0

    model.train()
    t0 = time.time()
    t_prev = t0

    for iteration in range(1, total_iters + 1):
        optimizer.zero_grad()

        s_f = (
            torch.rand(p3.N_F, device=p3.DEVICE) * (p3.S_TRAIN_HI - p3.S_TRAIN_LO)
            + p3.S_TRAIN_LO
        ).requires_grad_(True)
        t_f = (
            torch.rand(p3.N_F, device=p3.DEVICE) * t1
        ).requires_grad_(True)

        x_f = torch.stack([s_f, t_f], dim=1)
        V_f = model(x_f).squeeze()
        F_f = p3.bsm_operator(V_f, s_f, t_f, p3.r, p3.q, p3.sigma)
        loss_f = (F_f ** 2).mean()
        loss = p3.LAMBDA_F * loss_f

        loss.backward()
        total_norm = sum(
            p.grad.detach().norm(2).item() ** 2
            for p in model.parameters() if p.grad is not None
        ) ** 0.5
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss and math.isfinite(loss.item()):
            best_loss = loss.item()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_iter = iteration

        if iteration % log_every == 0 or iteration == 1:
            lr_now    = optimizer.param_groups[0]["lr"]
            t_now     = time.time()
            elapsed   = t_now - t0
            iter_rate = (t_now - t_prev) / log_every if iteration > 1 else float("nan")
            t_prev    = t_now

            history["loss"].append(loss.item())
            history["loss_f"].append(loss_f.item())
            history["loss_tc"].append(0.0)  # exactly zero by construction
            history["grad_norm"].append(total_norm)
            history["lr"].append(lr_now)
            history["iter"].append(iteration)

            logger.info(
                f"[{label}] iter {iteration:>6d}/{total_iters}  "
                f"loss={loss.item():.4e}  Lf={loss_f.item():.4e}  "
                f"Ltc=0 (hard)  |g|={total_norm:.2e}  "
                f"lr={lr_now:.5f}  ({elapsed:.1f}s, {iter_rate:.3f}s/iter)"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(
            f"[{label}] Restored best model from iter {best_iter} "
            f"(best_loss={best_loss:.4e}; last iter loss was {loss.item():.4e})"
        )
    model.eval()
    total_elapsed = time.time() - t0
    per_iter = total_elapsed / total_iters
    logger.info(
        f"[{label}] Training done — total={total_elapsed:.1f}s  "
        f"({per_iter:.3f}s/iter)  best_loss={best_loss:.4e} at iter {best_iter}"
    )
    return model, history


# ---------------------------------------------------------------------------
# Load Stage A and compute Bermudean TC target at t=t1
# ---------------------------------------------------------------------------

def _build_analytical_vtarget():
    """Build V_target(s) at t=t1 using the EXACT Black-Scholes formula.

    Companion to :func:`_load_etcnn_a_and_build_vtarget` for runs that use
    ``--analytical-stage-a``.  V_target(s) = max(payoff(s), V_BS_eur(s, t1))
    is constructed without training any Stage A network — Stage A is the
    closed-form European put with remaining maturity T - t1.

    Returns:
        v_target_fn:    Callable querying V_target at arbitrary asset prices.
        v_target_dense: (s_dense, vtarget_dense) numpy arrays for diagnostics.
    """
    from learning_option_pricing.models.etcnn import AnalyticalEuropeanPut
    etcnn_a = AnalyticalEuropeanPut(K=p3.K, r=p3.r, sigma=p3.sigma, T=p3.T)
    etcnn_a.to(p3.DEVICE).eval()

    s_dense  = torch.linspace(p3.S_TRAIN_LO - 10, p3.S_TRAIN_HI + 10, 2000, device=p3.DEVICE)
    t1_dense = torch.full_like(s_dense, p3.t1)
    x_t1     = torch.stack([s_dense, t1_dense], dim=1)
    with torch.no_grad():
        hold_val = etcnn_a(x_t1).squeeze()
    exercise_val = p3.payoff_put(s_dense, p3.K)
    v_t1_vals    = torch.maximum(exercise_val, hold_val)

    v_interp = p3.PchipInterpolator(s_dense.cpu(), v_t1_vals.cpu())

    def v_target_fn(s_batch: torch.Tensor) -> torch.Tensor:
        return v_interp(s_batch)

    return v_target_fn, (s_dense.cpu().numpy(), v_t1_vals.cpu().numpy())


def _compute_s_star_bermudan_analytical() -> float:
    r"""Locate the exercise boundary $s^\star$ at $t=t_1$ in analytical Stage A mode.

    $s^\star$ is the unique asset price where the put payoff meets the
    Black-Scholes European hold value, so the Bermudan terminal target at
    $t_1$ -- $V_{\mathrm{target}}(s) = \max(\Phi(s), V^{\mathrm{eur}}(s,t_1))$ --
    switches from one branch to the other:

    .. math::
        K - s^\star \; = \; V^{\mathrm{eur}}_{\mathrm{BS}}(s^\star,\,K,\,r,\,\sigma,\,T-t_1).

    The LHS decreases linearly from $K$ at $s=0$ down to $0$ at $s=K$ while the
    RHS is a positive smooth function strictly below $K-s$ on $(0,K)$, so the
    equation has a single root, located by bisection on $[10^{-6},\,K]$.

    Returns:
        s_star: Exercise-boundary asset price, in $(0, K)$.  Used as a visual
        reference (vertical line) on every $S$-axis comparison figure.
    """
    from learning_option_pricing.pricing.terminal import black_scholes_put

    tau = float(p3.T - p3.t1)
    K_  = float(p3.K)
    r_  = float(p3.r)
    sg_ = float(p3.sigma)

    def _diff(s_val: float) -> float:
        s_t   = torch.tensor([s_val], dtype=torch.get_default_dtype())
        tau_t = torch.tensor([tau],   dtype=torch.get_default_dtype())
        eur   = float(black_scholes_put(s_t, K_, r_, sg_, tau_t).item())
        return (K_ - s_val) - eur

    a, b = 1e-6, K_
    for _ in range(80):
        if abs(b - a) < 1e-8:
            break
        mid = 0.5 * (a + b)
        if _diff(a) * _diff(mid) < 0:
            b = mid
        else:
            a = mid
    return 0.5 * (a + b)


def _add_s_star_line(ax, s_star, *, with_label: bool = True) -> None:
    r"""Draw the exercise-boundary vertical line at $S = s^\star$ on an $S$-axis plot.

    $s^\star$ is the corner of the intermediate terminal payoff at $t_1$:

    .. math::
        V_{\mathrm{target}}(s) = \max(\Phi(s),\, V^{\mathrm{eur}}(s, t_1)),

    where the two branches of the $\max$ meet -- the only kink of the Bermudan
    value function on the Stage-B time interval, and a useful visual cue when
    comparing prices / errors / Greeks at $t=0$.  Silently no-ops when
    ``s_star`` is missing, NaN, or non-finite (so the helper is safe to call
    unconditionally from every $S$-axis plot, including in ``--replot`` mode
    on older folders that pre-date the metadata field).

    Args:
        ax:         Matplotlib axis to draw on.
        s_star:     Exercise-boundary asset price.  ``None``, ``"nan"`` and
                    non-finite floats are treated as "missing" and skipped.
        with_label: When ``True`` (default) attach a legend label of the form
                    ``"$s^\\star \\approx VALUE$ (exercise boundary at $t_1$)"``;
                    pass ``False`` on the second axis of a paired figure
                    (e.g. Greeks) to avoid the duplicate legend entry.
    """
    if s_star is None or s_star == "nan":
        return
    try:
        s_star_f = float(s_star)
    except (TypeError, ValueError):
        return
    if not np.isfinite(s_star_f):
        return
    label = (rf"$s^\star \approx {s_star_f:.2f}$ (exercise boundary at $t_1$)"
             if with_label else None)
    ax.axvline(s_star_f, color="tab:green", linestyle="-.", linewidth=1.0,
               label=label)


def _load_etcnn_a_and_build_vtarget(etcnn_a_path: Path):
    """Load a saved etcnn_a checkpoint and return a callable V_target(s) at t=t1.

    Args:
        etcnn_a_path: Path to etcnn_a.pt saved by bermudan_problem.

    Returns:
        v_target_fn:  Callable that takes a 1-D torch.Tensor of asset prices and
                      returns V_target(s) = max(payoff(s), etcnn_a(s, t1)).
        v_target_dense: (s_dense, vtarget_dense) numpy arrays for diagnostics.
    """
    from learning_option_pricing.models.etcnn import AmericanPutETCNN
    etcnn_a = AmericanPutETCNN(
        K=p3.K, r=p3.r, sigma=p3.sigma, T=p3.T,
        normalize_input=True, g2_type="bs",
    )
    etcnn_a.load_state_dict(torch.load(etcnn_a_path, map_location=p3.DEVICE))
    etcnn_a.eval().to(p3.DEVICE)

    s_dense  = torch.linspace(p3.S_TRAIN_LO - 10, p3.S_TRAIN_HI + 10, 2000, device=p3.DEVICE)
    t1_dense = torch.full_like(s_dense, p3.t1)
    x_t1     = torch.stack([s_dense, t1_dense], dim=1)
    with torch.no_grad():
        hold_val     = etcnn_a(x_t1).squeeze()
    exercise_val = p3.payoff_put(s_dense, p3.K)
    v_t1_vals    = torch.maximum(exercise_val, hold_val)

    # Build a PCHIP interpolant so v_target_fn can be queried at arbitrary S
    v_interp = p3.PchipInterpolator(s_dense.cpu(), v_t1_vals.cpu())

    def v_target_fn(s_batch: torch.Tensor) -> torch.Tensor:
        return v_interp(s_batch)

    return v_target_fn, (s_dense.cpu().numpy(), v_t1_vals.cpu().numpy())


# ---------------------------------------------------------------------------
# Custom Stage B training loop for soft BC variants
# ---------------------------------------------------------------------------

def train_stage_b_soft_pinn(
    model: torch.nn.Module,
    total_iters: int,
    v_target_fn,
    lambda_tc_soft: float,
    label: str,
    log_every: int | None = None,
) -> dict:
    """Train a plain PINN for Stage B with soft terminal-condition penalty.

    Args:
        model:          Plain PINN (no hard BC encoding in architecture).
        total_iters:    Number of gradient steps.
        v_target_fn:    Callable s_batch -> V_target(s_batch) for the TC penalty.
        lambda_tc_soft: Weight on the soft TC penalty term.
        label:          Short name for log messages.
        log_every:      Logging interval (default: adaptive).

    Returns:
        Training history dict with keys iter, loss, loss_f, loss_tc, grad_norm, lr.
    """
    if log_every is None:
        log_every = p3._adaptive_log_every(total_iters)

    model.to(p3.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, p3.build_lr_lambda(total_iters))

    history: dict = {
        "loss":      [], "loss_f": [], "loss_tc": [],
        "iter":      [], "grad_norm": [], "lr": [],
        "tc_enforced": False,
    }
    model.train()
    t0    = time.time()
    t_prev = t0

    for iteration in range(1, total_iters + 1):
        optimizer.zero_grad()

        # --- Interior PDE collocation points in [0, t1] ----------------------
        s_f = (
            torch.rand(p3.N_F, device=p3.DEVICE) * (p3.S_TRAIN_HI - p3.S_TRAIN_LO)
            + p3.S_TRAIN_LO
        ).requires_grad_(True)
        t_f = (torch.rand(p3.N_F, device=p3.DEVICE) * p3.t1).requires_grad_(True)

        # --- TC points at t=t1 -----------------------------------------------
        s_tc = (
            torch.rand(p3.N_TC, device=p3.DEVICE) * (p3.S_TRAIN_HI - p3.S_TRAIN_LO)
            + p3.S_TRAIN_LO
        )
        t_tc = torch.full((p3.N_TC,), p3.t1, device=p3.DEVICE)

        # --- PDE loss -----------------------------------------------------------
        x_f = torch.stack([s_f, t_f], dim=1)
        V_f = model(x_f).squeeze()
        F_f = p3.bsm_operator(V_f, s_f, t_f, p3.r, p3.q, p3.sigma)
        loss_f = (F_f ** 2).mean()

        # --- TC penalty loss ---------------------------------------------------
        x_tc  = torch.stack([s_tc, t_tc], dim=1)
        V_tc  = model(x_tc).squeeze()
        with torch.no_grad():
            V_target_vals = v_target_fn(s_tc)
        loss_tc = ((V_tc - V_target_vals) ** 2).mean()

        loss = p3.LAMBDA_F * loss_f + lambda_tc_soft * loss_tc
        loss.backward()

        total_norm = sum(
            p.grad.detach().norm(2).item() ** 2
            for p in model.parameters() if p.grad is not None
        ) ** 0.5

        optimizer.step()
        scheduler.step()

        if iteration % log_every == 0 or iteration == 1:
            lr_now    = optimizer.param_groups[0]["lr"]
            t_now     = time.time()
            elapsed   = t_now - t0
            iter_rate = (t_now - t_prev) / log_every if iteration > 1 else float("nan")
            t_prev    = t_now

            history["loss"].append(loss.item())
            history["loss_f"].append(loss_f.item())
            history["loss_tc"].append(loss_tc.item())
            history["grad_norm"].append(total_norm)
            history["lr"].append(lr_now)
            history["iter"].append(iteration)

            logger.info(
                f"[{label}] iter {iteration:>6d}/{total_iters}  "
                f"loss={loss.item():.4e}  Lf={loss_f.item():.4e}  "
                f"Ltc={loss_tc.item():.4e}  |g|={total_norm:.2e}  "
                f"lr={lr_now:.5f}  ({elapsed:.1f}s, {iter_rate:.3f}s/iter)"
            )

    model.eval()
    total_elapsed = time.time() - t0
    per_iter = total_elapsed / total_iters
    logger.info(
        f"[{label}] Training done — "
        f"total={total_elapsed:.1f}s  ({per_iter:.3f}s/iter)"
    )
    return history


# ---------------------------------------------------------------------------
# Rich metric computation (shared with ablation_bermudan.py)
# ---------------------------------------------------------------------------

def _compute_slice_evaluations(
    model: torch.nn.Module,
    s_eval_arr: np.ndarray,
    t_slices: tuple[float, ...],
) -> dict:
    r"""Evaluate $V_\theta$, $\partial_S V_\theta$, $\partial_{SS} V_\theta$ at
    multiple time slices on a fixed $S$-grid.

    Used by both training-time (via :func:`compute_metrics_stage_b`) and replot
    (via :func:`_regenerate_slices_from_saved_model`) so the multi-time price /
    Greeks figures (``form_prices_by_t.png``, ``form_greeks_by_t.png``) share
    one source of truth.  Greeks are taken via ``torch.autograd`` rather than
    finite differences so the prediction curves stay smooth even for the soft-
    PINN variants whose surfaces can be noisy.

    Args:
        model:      Trained Stage B network in eval mode.  Caller is
                    responsible for placing it on the right device.
        s_eval_arr: 1-D asset-price evaluation grid (numpy).  Reused at every
                    slice so the columns of the returned arrays line up.
        t_slices:   Tuple of times in $[0, t_1]$ at which to evaluate.  The
                    canonical choice is ``(0.0, t_1/2, t_1)``; the ends pin
                    the Stage-B initial / terminal sides and the midpoint
                    exposes any backward-time leakage of the $t_1$ kink.

    Returns:
        Dict with shape ``(len(t_slices), len(s_eval_arr))``:
        - ``"t_slices"``       — list of evaluation times (length ``n_t``).
        - ``"price_slices"``   — $V_\theta(S, t_k)$.
        - ``"delta_slices"``   — $\partial_S V_\theta(S, t_k)$ (autograd).
        - ``"gamma_slices"``   — $\partial_{SS} V_\theta(S, t_k)$ (autograd).
    """
    device = next(model.parameters()).device
    n_t    = len(t_slices)
    n_s    = len(s_eval_arr)
    price_slices = np.zeros((n_t, n_s), dtype=np.float64)
    delta_slices = np.zeros((n_t, n_s), dtype=np.float64)
    gamma_slices = np.zeros((n_t, n_s), dtype=np.float64)
    s_base = torch.tensor(s_eval_arr, dtype=torch.get_default_dtype(), device=device)
    for k, t_val in enumerate(t_slices):
        s_d = s_base.detach().clone().requires_grad_(True)
        t_d = torch.full_like(s_d, float(t_val), requires_grad=True)
        x_d = torch.stack([s_d, t_d], dim=1)
        V_d = model(x_d).squeeze()
        price_slices[k] = V_d.detach().cpu().numpy()
        try:
            (delta_d,) = torch.autograd.grad(V_d.sum(), s_d, create_graph=True)
            (gamma_d,) = torch.autograd.grad(delta_d.sum(), s_d, create_graph=False)
            delta_slices[k] = delta_d.detach().cpu().numpy()
            gamma_slices[k] = gamma_d.detach().cpu().numpy()
        except Exception as exc:
            logger.warning(
                f"_compute_slice_evaluations: autograd failed at t={t_val} "
                f"({exc}); slice filled with NaN."
            )
            delta_slices[k] = np.nan
            gamma_slices[k] = np.nan
    return {
        "t_slices":     [float(t) for t in t_slices],
        "price_slices": price_slices,
        "delta_slices": delta_slices,
        "gamma_slices": gamma_slices,
    }


# Canonical Bermudan Stage-B time probes for multi-time comparison figures.
# Three slices: the Stage-B initial side, midpoint, and the $t=t_1$ terminal
# side where the analytical $V_{\mathrm{target}}$ reference is known.
_T_SLICES_BERMUDAN = (0.0, p3.t1 / 2.0, p3.t1)


def compute_metrics_stage_b(
    model: torch.nn.Module,
    hist_b: dict,
    bt_prices: np.ndarray,
    s_eval_arr: np.ndarray,
    v_target_fn=None,
) -> dict:
    """Compute Stage B evaluation metrics.

    Args:
        model:        Trained Stage B model.
        hist_b:       Training history (must contain "grad_norm").
        bt_prices:    Binomial-tree prices at s_eval_arr, t=0.
        s_eval_arr:   1-D asset-price evaluation grid.
        v_target_fn:  Optional callable for TC error at t=t1 (soft variants only).

    Returns:
        Dict with rel_l2_bt, rel_l2_atm, rel_l2_delta, gei, tc_mae, pde_residual_t.
    """
    device = p3.DEVICE
    model.eval()

    # --- Price comparison at t=0 -------------------------------------------
    s_tensor = torch.tensor(s_eval_arr, dtype=torch.get_default_dtype(), device=device)
    t_zero   = torch.zeros_like(s_tensor)
    x_eval   = torch.stack([s_tensor, t_zero], dim=1)
    with torch.no_grad():
        nn_prices = model(x_eval).squeeze().cpu().numpy()

    err       = nn_prices - bt_prices
    rel_l2_bt = float(np.linalg.norm(err) / (np.linalg.norm(bt_prices) + 1e-10))

    atm_mask   = np.abs(s_eval_arr - p3.K) <= 0.1 * p3.K
    rel_l2_atm = float(
        np.linalg.norm(err[atm_mask]) / (np.linalg.norm(bt_prices[atm_mask]) + 1e-10)
    )

    # --- Delta and Gamma comparison ----------------------------------------
    # Reference Greeks come from finite differences on the binomial-tree price
    # curve.  Gamma is the second FD of BT prices and is noisy near the kink,
    # but the relative L² ratio is still informative when comparing variants
    # under the same reference.
    try:
        s_d = torch.tensor(
            s_eval_arr, dtype=torch.get_default_dtype(), device=device
        ).requires_grad_(True)
        t_d = torch.zeros(len(s_eval_arr), device=device, requires_grad=True)
        x_d = torch.stack([s_d, t_d], dim=1)
        V_d = model(x_d).squeeze()
        (nn_delta,) = torch.autograd.grad(V_d.sum(), s_d, create_graph=True)
        (nn_gamma,) = torch.autograd.grad(nn_delta.sum(), s_d, create_graph=False)
        nn_delta_np  = nn_delta.detach().cpu().numpy()
        nn_gamma_np  = nn_gamma.detach().cpu().numpy()
        bt_delta_np  = np.gradient(bt_prices, s_eval_arr)
        bt_gamma_np  = np.gradient(bt_delta_np, s_eval_arr)
        rel_l2_delta = float(
            np.linalg.norm(nn_delta_np - bt_delta_np)
            / (np.linalg.norm(bt_delta_np) + 1e-10)
        )
        rel_l2_gamma = float(
            np.linalg.norm(nn_gamma_np - bt_gamma_np)
            / (np.linalg.norm(bt_gamma_np) + 1e-10)
        )
        greeks_curves = {
            "s":        np.asarray(s_eval_arr),
            "nn_delta": nn_delta_np,
            "bt_delta": bt_delta_np,
            "nn_gamma": nn_gamma_np,
            "bt_gamma": bt_gamma_np,
        }
    except Exception as exc:
        logger.warning(f"compute_metrics_stage_b: Greeks failed ({exc})")
        rel_l2_delta = float("nan")
        rel_l2_gamma = float("nan")
        greeks_curves = None

    # --- GEI ---------------------------------------------------------------
    norms = np.array(hist_b.get("grad_norm", []))
    if len(norms) > 0:
        cutoff = max(1, int(len(norms) * 2 / 3))
        n_early = norms[:cutoff]
        gei     = float(n_early.max() / (np.median(n_early) + 1e-10))
    else:
        gei = float("nan")

    # --- TC mean absolute error at t=t1 (for soft variants) ----------------
    if v_target_fn is not None:
        try:
            s_tc_eval = torch.tensor(s_eval_arr, dtype=torch.get_default_dtype(), device=device)
            t_tc_eval = torch.full_like(s_tc_eval, p3.t1)
            x_tc      = torch.stack([s_tc_eval, t_tc_eval], dim=1)
            with torch.no_grad():
                V_tc_pred = model(x_tc).squeeze().cpu().numpy()
                V_tc_tgt  = v_target_fn(s_tc_eval).cpu().numpy()
            tc_mae = float(np.abs(V_tc_pred - V_tc_tgt).mean())
        except Exception as exc:
            logger.warning(f"compute_metrics_stage_b: TC MAE failed ({exc})")
            tc_mae = float("nan")
    else:
        tc_mae = 0.0  # hard BC: exactly 0 by construction

    # --- PDE residual profile along S=K for t in [0, t1] -------------------
    n_profile = 25
    t_profile_tensor = torch.linspace(1e-3, p3.t1 - 1e-3, n_profile, device=device)
    pde_residuals    = []
    try:
        for t_val in t_profile_tensor:
            n_pts = 50
            s_p   = torch.full((n_pts,), p3.K, device=device).requires_grad_(True)
            t_p   = torch.full((n_pts,), t_val.item(), device=device).requires_grad_(True)
            x_p   = torch.stack([s_p, t_p], dim=1)
            V_p   = model(x_p).squeeze()
            F_p   = p3.bsm_operator(V_p, s_p, t_p, p3.r, p3.q, p3.sigma)
            pde_residuals.append(F_p.detach().abs().mean().item())
    except Exception as exc:
        logger.warning(f"compute_metrics_stage_b: PDE profile failed ({exc})")
        pde_residuals = [float("nan")] * n_profile

    # --- Multi-time slices for form_prices_by_t.png + form_greeks_by_t.png --
    # Evaluate V_theta, Delta, Gamma at every probe in _T_SLICES_BERMUDAN on
    # the same s_eval_arr grid, plus the reference curves where they exist
    # (BT at t=0, V_target at t=t1 — none in between).  Numerical
    # finite-difference Greeks on the references match what the existing
    # form_greeks.png plot uses, so the multi-t version is just a horizontal
    # tiling of the t=0 column.
    try:
        slices = _compute_slice_evaluations(model, s_eval_arr, _T_SLICES_BERMUDAN)
        ref_price_t0 = np.asarray(bt_prices)
        ref_delta_t0 = np.gradient(ref_price_t0, s_eval_arr)
        ref_gamma_t0 = np.gradient(ref_delta_t0, s_eval_arr)
        if v_target_fn is not None:
            s_tens     = torch.tensor(s_eval_arr, dtype=torch.get_default_dtype(), device=device)
            with torch.no_grad():
                ref_price_t1 = v_target_fn(s_tens).detach().cpu().numpy()
            ref_delta_t1 = np.gradient(ref_price_t1, s_eval_arr)
            ref_gamma_t1 = np.gradient(ref_delta_t1, s_eval_arr)
        else:
            ref_price_t1 = None
            ref_delta_t1 = None
            ref_gamma_t1 = None
        slices.update({
            "ref_price_t0": ref_price_t0,
            "ref_delta_t0": ref_delta_t0,
            "ref_gamma_t0": ref_gamma_t0,
            "ref_price_t1": ref_price_t1,
            "ref_delta_t1": ref_delta_t1,
            "ref_gamma_t1": ref_gamma_t1,
        })
    except Exception as exc:
        logger.warning(f"compute_metrics_stage_b: slice evaluation failed ({exc})")
        slices = None

    return {
        "rel_l2_bt":     rel_l2_bt,
        "rel_l2_atm":    rel_l2_atm,
        "rel_l2_delta":  rel_l2_delta,
        "rel_l2_gamma":  rel_l2_gamma,
        "gei":           gei,
        "tc_mae":        tc_mae,
        "greeks":        greeks_curves,
        "pde_residual_t": {
            "t":        t_profile_tensor.cpu().tolist(),
            "residual": pde_residuals,
        },
        "slices":        slices,
    }


# ---------------------------------------------------------------------------
# Evaluate a trained model vs binomial tree
# ---------------------------------------------------------------------------

def _evaluate_vs_binomial_tree(model: torch.nn.Module) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Evaluate model at t=0 and compute MAE and relative L2 vs BT.

    Returns:
        (s_eval_arr, bt_prices, etcnn_b_prices, mae, rel_l2)
    """
    from learning_option_pricing.solvers.binomial_tree import bermuda_put_binomial_tree

    logger.info("Computing binomial tree reference prices ...")
    s_eval_arr = np.linspace(60.0, 140.0, 81)
    bt_prices  = np.array([
        bermuda_put_binomial_tree(float(s), p3.K, p3.r, p3.sigma, p3.T, [p3.t1], N=2000)
        for s in s_eval_arr
    ])
    logger.info("  Binomial tree reference prices computed.")

    s_tensor = torch.tensor(s_eval_arr, dtype=torch.get_default_dtype(), device=p3.DEVICE)
    t_zero   = torch.zeros_like(s_tensor)
    x_eval   = torch.stack([s_tensor, t_zero], dim=1)
    with torch.no_grad():
        nn_prices = model(x_eval).squeeze().cpu().numpy()

    err       = nn_prices - bt_prices
    mae       = float(np.abs(err).mean())
    rel_l2    = float(np.linalg.norm(err) / (np.linalg.norm(bt_prices) + 1e-10))
    logger.info(f"  Evaluation — MAE={mae:.4e}  rel_L2={rel_l2:.4e}")

    return s_eval_arr, bt_prices, nn_prices, mae, rel_l2


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _save_variant_results(res: dict, vdir: Path) -> None:
    hist = res["hist_b"]
    np.savez_compressed(
        vdir / "hist_b.npz",
        iter=np.array(hist["iter"]),
        loss=np.array(hist["loss"]),
        loss_f=np.array(hist["loss_f"]),
        loss_tc=np.array(hist["loss_tc"]),
        grad_norm=np.array(hist["grad_norm"]),
        lr=np.array(hist["lr"]),
        tc_enforced=np.array([hist.get("tc_enforced", False)]),
    )
    np.savez_compressed(
        vdir / "prices.npz",
        etcnn_b_prices=np.array(res["etcnn_b_prices"]),
        bt_prices=np.array(res["bt_prices"]),
        s_eval_arr=np.array(res["s_eval_arr"]),
    )
    metrics = res.get("metrics")
    if metrics is not None:
        payload = {
            "rel_l2_bt":    np.array([metrics["rel_l2_bt"]]),
            "rel_l2_atm":   np.array([metrics["rel_l2_atm"]]),
            "rel_l2_delta": np.array([metrics["rel_l2_delta"]]),
            "rel_l2_gamma": np.array([metrics.get("rel_l2_gamma", float("nan"))]),
            "gei":          np.array([metrics["gei"]]),
            "tc_mae":       np.array([metrics["tc_mae"]]),
            "pde_t":        np.array(metrics["pde_residual_t"]["t"]),
            "pde_residual": np.array(metrics["pde_residual_t"]["residual"]),
        }
        greeks = metrics.get("greeks")
        if greeks is not None:
            payload.update({
                "greeks_s":        np.asarray(greeks["s"]),
                "greeks_nn_delta": np.asarray(greeks["nn_delta"]),
                "greeks_bt_delta": np.asarray(greeks["bt_delta"]),
                "greeks_nn_gamma": np.asarray(greeks["nn_gamma"]),
                "greeks_bt_gamma": np.asarray(greeks["bt_gamma"]),
            })
        slices = metrics.get("slices")
        if slices is not None:
            payload.update({
                "slice_t":            np.asarray(slices["t_slices"]),
                "slice_price":        np.asarray(slices["price_slices"]),
                "slice_delta":        np.asarray(slices["delta_slices"]),
                "slice_gamma":        np.asarray(slices["gamma_slices"]),
                "slice_ref_price_t0": np.asarray(slices["ref_price_t0"]),
                "slice_ref_delta_t0": np.asarray(slices["ref_delta_t0"]),
                "slice_ref_gamma_t0": np.asarray(slices["ref_gamma_t0"]),
            })
            if slices.get("ref_price_t1") is not None:
                payload.update({
                    "slice_ref_price_t1": np.asarray(slices["ref_price_t1"]),
                    "slice_ref_delta_t1": np.asarray(slices["ref_delta_t1"]),
                    "slice_ref_gamma_t1": np.asarray(slices["ref_gamma_t1"]),
                })
        np.savez_compressed(vdir / "metrics.npz", **payload)


def _load_variant_results(vdir: Path, summary_entry: dict, style: dict) -> dict:
    hist_npz   = np.load(vdir / "hist_b.npz")
    prices_npz = np.load(vdir / "prices.npz")
    hist = {
        "iter":        hist_npz["iter"].tolist(),
        "loss":        hist_npz["loss"].tolist(),
        "loss_f":      hist_npz["loss_f"].tolist(),
        "loss_tc":     hist_npz["loss_tc"].tolist(),
        "grad_norm":   hist_npz["grad_norm"].tolist(),
        "lr":          hist_npz["lr"].tolist(),
        "tc_enforced": bool(hist_npz["tc_enforced"][0]) if "tc_enforced" in hist_npz else False,
    }
    metrics_path = vdir / "metrics.npz"
    if metrics_path.exists():
        m = np.load(metrics_path)
        metrics = {
            "rel_l2_bt":     float(m["rel_l2_bt"][0]),
            "rel_l2_atm":    float(m["rel_l2_atm"][0]),
            "rel_l2_delta":  float(m["rel_l2_delta"][0]),
            "rel_l2_gamma":  float(m["rel_l2_gamma"][0]) if "rel_l2_gamma" in m.files else float("nan"),
            "gei":           float(m["gei"][0]),
            "tc_mae":        float(m["tc_mae"][0]) if "tc_mae" in m.files else float("nan"),
            "pde_residual_t": {
                "t":        m["pde_t"].tolist(),
                "residual": m["pde_residual"].tolist(),
            },
        }
        if "greeks_s" in m.files:
            metrics["greeks"] = {
                "s":        m["greeks_s"],
                "nn_delta": m["greeks_nn_delta"],
                "bt_delta": m["greeks_bt_delta"],
                "nn_gamma": m["greeks_nn_gamma"],
                "bt_gamma": m["greeks_bt_gamma"],
            }
        else:
            metrics["greeks"] = None
        if "slice_t" in m.files:
            slices = {
                "t_slices":     m["slice_t"].tolist(),
                "price_slices": m["slice_price"],
                "delta_slices": m["slice_delta"],
                "gamma_slices": m["slice_gamma"],
                "ref_price_t0": m["slice_ref_price_t0"],
                "ref_delta_t0": m["slice_ref_delta_t0"],
                "ref_gamma_t0": m["slice_ref_gamma_t0"],
            }
            if "slice_ref_price_t1" in m.files:
                slices.update({
                    "ref_price_t1": m["slice_ref_price_t1"],
                    "ref_delta_t1": m["slice_ref_delta_t1"],
                    "ref_gamma_t1": m["slice_ref_gamma_t1"],
                })
            else:
                slices.update({"ref_price_t1": None,
                               "ref_delta_t1": None,
                               "ref_gamma_t1": None})
            metrics["slices"] = slices
        else:
            metrics["slices"] = None
    else:
        metrics = None
    return {
        **style,
        **summary_entry,
        "hist_b":         hist,
        "etcnn_b_prices": prices_npz["etcnn_b_prices"],
        "bt_prices":      prices_npz["bt_prices"],
        "s_eval_arr":     prices_npz["s_eval_arr"],
        "metrics":        metrics,
    }


# ---------------------------------------------------------------------------
# Per-variant diagnostic plots
# ---------------------------------------------------------------------------

def _plot_variant(res: dict, vdir: Path) -> None:
    out   = vdir / "training_metrics"
    out.mkdir(exist_ok=True)
    hist  = res["hist_b"]
    label = res.get("label", res.get("name", "variant"))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].semilogy(hist["iter"], hist["loss"], color="tab:blue")
    axes[0, 0].set_title(r"Total loss $\mathcal{L}$")
    axes[0, 0].set_xlabel("Iteration (Stage B)")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].semilogy(hist["iter"], hist["loss_f"],
                        color="tab:orange", label=r"$\mathcal{L}_f$ (PDE)")
    axes[0, 1].semilogy(hist["iter"], hist["loss_tc"],
                        color="tab:red",    label=r"$\mathcal{L}_{tc}$ (TC penalty)",
                        linestyle="--")
    axes[0, 1].set_title("PDE residual vs TC penalty")
    axes[0, 1].set_xlabel("Iteration (Stage B)")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].semilogy(hist["iter"], hist["grad_norm"], color="tab:purple")
    axes[1, 0].set_title(r"Gradient norm $\|\nabla_\theta\mathcal{L}\|_2$")
    axes[1, 0].set_xlabel("Iteration (Stage B)")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].semilogy(hist["iter"], hist["lr"], color="tab:gray")
    axes[1, 1].set_title("Learning rate schedule")
    axes[1, 1].set_xlabel("Iteration (Stage B)")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"{label}\n{_SUPTITLE_PARAMS}", fontsize=10)
    fig.tight_layout(rect=[0, 0.20, 1, 1])
    _add_formula_box(
        fig, _FORMULA_LF_B + "\n" + _FORMULA_TC + "\n" + _FORMULA_GRAD,
        bottom_margin=0.22,
    )
    fig.savefig(out / "training_curves.png", dpi=150)
    plt.close(fig)

    metrics = res.get("metrics")
    if metrics is not None and "pde_residual_t" in metrics:
        pde = metrics["pde_residual_t"]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(pde["t"], pde["residual"],
                    color=res.get("color", "tab:blue"),
                    linestyle=res.get("linestyle", "-"),
                    linewidth=res.get("linewidth", 2.0),
                    marker="o", ms=4)
        ax.axvline(p3.t1, color="tab:red", linestyle="--", linewidth=1.0,
                   label=rf"$t_1={p3.t1}$")
        ax.set_xlabel(r"$t$  (Stage B time)")
        ax.set_ylabel(r"$\mathbb{E}_{S=K}[|\mathcal{F}[V_\theta]|]$")
        ax.set_title(r"Mean PDE residual along $S=K$")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.suptitle(f"{label}\n{_SUPTITLE_PARAMS}", fontsize=10)
        fig.tight_layout(rect=[0, 0.18, 1, 1])
        _add_formula_box(fig, _FORMULA_PDE_PROFILE, bottom_margin=0.20)
        fig.savefig(out / "pde_residual_by_t.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Comparison plots
# ---------------------------------------------------------------------------

def _plot_comparison(results: list[dict], ablation_dir: Path, iters_b: int,
                     *, output_subdir: str = "comparison",
                     s_star: float | None = None) -> None:
    """Emit every comparison figure under ``ablation_dir / output_subdir``.

    Default ``output_subdir="comparison"`` writes to the canonical folder
    populated by every full ablation run.  Pass a different name to
    produce an alternative figure set without overwriting the canonical
    one — used by the ``--replot --exclude-variant`` workflow to keep
    only the regularisation winner and drop the noisier soft-penalty
    siblings, for instance.

    Args:
        results:        Per-variant result dicts (curves, histories, metrics).
        ablation_dir:   Run directory.  Plots are written under
                        ``ablation_dir / output_subdir / *.png``.
        iters_b:        Stage B iteration budget (for figure titles).
        output_subdir:  Sub-folder name (default ``"comparison"``).
        s_star:         Exercise-boundary asset price at $t_1$.  When provided,
                        each $S$-axis figure (prices, error vs BT, Greeks) gets
                        a vertical reference line drawn by
                        :func:`_add_s_star_line`.  Pass ``None`` to suppress
                        (e.g. older folders pre-dating the metadata field).
    """
    comp_dir = ablation_dir / output_subdir
    comp_dir.mkdir(exist_ok=True)

    colors     = [r.get("color",     "tab:blue") for r in results]
    linestyles = [r.get("linestyle", "-")        for r in results]
    linewidths = [r.get("linewidth", 2.0)        for r in results]
    labels     = [r.get("label", r.get("name", f"v{i}")) for i, r in enumerate(results)]
    has_metrics = [r.get("metrics") is not None  for r in results]

    def _semilogy_all(ax, key_hist: str, xlabel: str, ylabel: str, title: str):
        for i, res in enumerate(results):
            hist = res.get("hist_b")
            if hist is None or key_hist not in hist:
                continue
            ax.semilogy(hist["iter"], hist[key_hist],
                        label=labels[i], color=colors[i],
                        linestyle=linestyles[i], linewidth=linewidths[i])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Plot 1 — PDE residual loss
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    _semilogy_all(ax, "loss_f",
                  "Iteration (Stage B)", r"$\mathcal{L}_f$",
                  r"PDE residual loss $\mathcal{L}_f$" + f"  ({iters_b} iters)")
    fig.suptitle(f"Formulation ablation — PDE residual\n{_SUPTITLE_PARAMS}", fontsize=10)
    fig.tight_layout(rect=[0, 0.18, 1, 1])
    _add_formula_box(fig, _FORMULA_LF_B, bottom_margin=0.20)
    fig.savefig(comp_dir / "form_loss_pde.png", dpi=150)
    plt.close(fig)
    logger.info("[OK] form_loss_pde.png")

    # ------------------------------------------------------------------
    # Plot 2 — TC loss (meaningful only for soft variants; hard ≈ 0)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, res in enumerate(results):
        hist = res.get("hist_b")
        if hist is None:
            continue
        tc_vals = hist["loss_tc"]
        if max(tc_vals) < 1e-12:
            label_tc = labels[i] + "  (hard-enforced: 0)"
            ax.axhline(0, color=colors[i], linestyle=linestyles[i],
                       linewidth=linewidths[i], label=label_tc, alpha=0.6)
        else:
            ax.semilogy(hist["iter"], tc_vals, label=labels[i],
                        color=colors[i], linestyle=linestyles[i],
                        linewidth=linewidths[i])
    ax.set_xlabel("Iteration (Stage B)")
    ax.set_ylabel(r"$\mathcal{L}_{tc}$")
    ax.set_title(r"TC penalty loss $\mathcal{L}_{tc}$  (hard BC: identically $0$)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.suptitle(f"Formulation ablation — TC loss\n{_SUPTITLE_PARAMS}", fontsize=10)
    fig.tight_layout(rect=[0, 0.20, 1, 1])
    _add_formula_box(fig, _FORMULA_TC, bottom_margin=0.22)
    fig.savefig(comp_dir / "form_loss_tc.png", dpi=150)
    plt.close(fig)
    logger.info("[OK] form_loss_tc.png")

    # ------------------------------------------------------------------
    # Plot 3 — Gradient norm
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    _semilogy_all(ax, "grad_norm",
                  "Iteration (Stage B)",
                  r"$\|\nabla_\theta\mathcal{L}\|_2$",
                  "Gradient norm — instability signature")
    fig.suptitle(f"Formulation ablation — Gradient norm\n{_SUPTITLE_PARAMS}", fontsize=10)
    fig.tight_layout(rect=[0, 0.14, 1, 1])
    _add_formula_box(fig, _FORMULA_GRAD, bottom_margin=0.16)
    fig.savefig(comp_dir / "form_grad_norm.png", dpi=150)
    plt.close(fig)
    logger.info("[OK] form_grad_norm.png")

    # ------------------------------------------------------------------
    # Plot 4 — Pricing curves at t=0
    # ------------------------------------------------------------------
    bt_prices  = results[0].get("bt_prices")
    s_eval_arr = results[0].get("s_eval_arr")
    fig, ax = plt.subplots(figsize=(10, 6))
    if bt_prices is not None and s_eval_arr is not None:
        ax.plot(s_eval_arr, bt_prices,
                label=r"$V^{\mathrm{BT}}(S,0)$  (binomial tree, $N=2000$)",
                color="black", linewidth=2.5, zorder=10)
    for i, res in enumerate(results):
        prices = res.get("etcnn_b_prices")
        s_arr  = res.get("s_eval_arr")
        if prices is None or s_arr is None:
            continue
        ax.plot(s_arr, prices, label=r"$V_\theta(S,0)$ — " + labels[i],
                color=colors[i], linestyle=linestyles[i], linewidth=linewidths[i])
    _add_s_star_line(ax, s_star)
    ax.set_xlabel("Asset price $S$")
    ax.set_ylabel("Price at $t=0$")
    ax.set_title(r"$V_\theta(S,0)$ vs $V^{\mathrm{BT}}(S,0)$  —  all variants")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.suptitle(f"Formulation ablation — Pricing comparison\n{_SUPTITLE_PARAMS}", fontsize=10)
    fig.tight_layout(rect=[0, 0.18, 1, 1])
    _add_formula_box(fig, _FORMULA_TC, bottom_margin=0.20)
    fig.savefig(comp_dir / "form_prices.png", dpi=150)
    plt.close(fig)
    logger.info("[OK] form_prices.png")

    # ------------------------------------------------------------------
    # Plot 5 — Pointwise absolute error vs BT at t=0
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, res in enumerate(results):
        prices = res.get("etcnn_b_prices")
        bt     = res.get("bt_prices")
        s_arr  = res.get("s_eval_arr")
        if prices is None or bt is None or s_arr is None:
            continue
        err = np.abs(prices - bt)
        mae = float(res.get("mae_bt", np.mean(err)))
        ax.plot(s_arr, err,
                label=rf"{labels[i]}  ($\mathrm{{MAE}}={mae:.2e}$)",
                color=colors[i], linestyle=linestyles[i], linewidth=linewidths[i])
    _add_s_star_line(ax, s_star)
    ax.set_xlabel("Asset price $S$")
    ax.set_ylabel(r"$|V_\theta(S,0) - V^{\mathrm{BT}}(S,0)|$")
    ax.set_title(r"Pointwise error vs $V^{\mathrm{BT}}$ at $t=0$")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.suptitle(f"Formulation ablation — Error vs BT\n{_SUPTITLE_PARAMS}", fontsize=10)
    fig.tight_layout(rect=[0, 0.14, 1, 1])
    _add_formula_box(fig, _FORMULA_TC, bottom_margin=0.16)
    fig.savefig(comp_dir / "form_error_vs_bt.png", dpi=150)
    plt.close(fig)
    logger.info("[OK] form_error_vs_bt.png")

    # ------------------------------------------------------------------
    # Plot 6 — PDE residual profile along S=K
    # ------------------------------------------------------------------
    if any(has_metrics):
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, res in enumerate(results):
            metrics = res.get("metrics")
            if metrics is None or "pde_residual_t" not in metrics:
                continue
            pde = metrics["pde_residual_t"]
            ax.semilogy(pde["t"], pde["residual"],
                        label=labels[i], color=colors[i],
                        linestyle=linestyles[i], linewidth=linewidths[i],
                        marker="o", ms=3)
        ax.axvline(p3.t1, color="k", linestyle=":", linewidth=0.8,
                   label=rf"$t_1={p3.t1}$")
        ax.set_xlabel(r"$t$  (Stage B time)")
        ax.set_ylabel(r"$\mathbb{E}_{S=K}[|\mathcal{F}[V_\theta]|]$")
        ax.set_title(r"Mean PDE residual along $S=K$  (Stage B interval)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.suptitle(f"Formulation ablation — PDE profile\n{_SUPTITLE_PARAMS}", fontsize=10)
        fig.tight_layout(rect=[0, 0.18, 1, 1])
        _add_formula_box(fig, _FORMULA_PDE_PROFILE, bottom_margin=0.20)
        fig.savefig(comp_dir / "form_pde_residual_by_t.png", dpi=150)
        plt.close(fig)
        logger.info("[OK] form_pde_residual_by_t.png")

    # ------------------------------------------------------------------
    # Plot 7 — TC error at t=t1 (illustrates how well soft BC is satisfied)
    # ------------------------------------------------------------------
    if any(has_metrics):
        tc_mae_vals   = [
            res.get("metrics", {}).get("tc_mae", float("nan")) if res.get("metrics") else float("nan")
            for res in results
        ]
        if any(not np.isnan(v) for v in tc_mae_vals):
            fig, ax = plt.subplots(figsize=(8, 5))
            x = np.arange(len(results))
            bars = ax.bar(x, tc_mae_vals, color=colors, edgecolor="black", linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([r.get("name", f"v{i}") for i, r in enumerate(results)],
                               rotation=20, ha="right", fontsize=9)
            ax.set_ylabel(
                r"$\frac{1}{N}\sum|V_\theta(S_j,t_1) - V_{\mathrm{target}}(S_j)|$"
            )
            ax.set_title(r"TC mean absolute error at $t=t_1$ (hard BC: $0$ by construction)")
            ax.grid(axis="y", alpha=0.3)
            for bar_rect, val in zip(bars, tc_mae_vals):
                if not np.isnan(val):
                    ax.text(bar_rect.get_x() + bar_rect.get_width() / 2,
                            bar_rect.get_height() * 1.02,
                            f"{val:.2e}", ha="center", va="bottom", fontsize=8)
            fig.suptitle(f"Formulation ablation — TC error\n{_SUPTITLE_PARAMS}", fontsize=10)
            fig.tight_layout(rect=[0, 0.18, 1, 1])
            _add_formula_box(fig, _FORMULA_TC_ERROR, bottom_margin=0.20)
            fig.savefig(comp_dir / "form_tc_error_at_t1.png", dpi=150)
            plt.close(fig)
            logger.info("[OK] form_tc_error_at_t1.png")

    # ------------------------------------------------------------------
    # Plot 8 — Greeks comparison (Δ and Γ vs S, all variants overlaid)
    # ------------------------------------------------------------------
    if any(has_metrics):
        greek_results = [
            res for res in results
            if res.get("metrics") is not None
            and res["metrics"].get("greeks") is not None
        ]
        if greek_results:
            fig, (ax_d, ax_g) = plt.subplots(1, 2, figsize=(14, 6))
            # BT reference from the first variant that has it (same for all).
            ref = greek_results[0]["metrics"]["greeks"]
            ax_d.plot(ref["s"], ref["bt_delta"], "k--", linewidth=1.6,
                      label=r"$\Delta^{\mathrm{BT}}$", zorder=10)
            ax_g.plot(ref["s"], ref["bt_gamma"], "k--", linewidth=1.6,
                      label=r"$\Gamma^{\mathrm{BT}}$", zorder=10)
            for res in greek_results:
                g       = res["metrics"]["greeks"]
                lbl     = res.get("label", res.get("name", ""))
                col     = res.get("color")
                ls      = res.get("linestyle", "-")
                lw      = res.get("linewidth", 1.8)
                eps_d   = float(res["metrics"].get("rel_l2_delta", float("nan")))
                eps_g   = float(res["metrics"].get("rel_l2_gamma", float("nan")))
                ax_d.plot(g["s"], g["nn_delta"], color=col, linestyle=ls, linewidth=lw,
                          label=rf"{lbl}  ($\varepsilon_\Delta={eps_d:.2e}$)")
                ax_g.plot(g["s"], g["nn_gamma"], color=col, linestyle=ls, linewidth=lw,
                          label=rf"{lbl}  ($\varepsilon_\Gamma={eps_g:.2e}$)")
            ax_d.axvline(p3.K, color="gray", linestyle=":", linewidth=0.8)
            ax_g.axvline(p3.K, color="gray", linestyle=":", linewidth=0.8)
            _add_s_star_line(ax_d, s_star)
            _add_s_star_line(ax_g, s_star, with_label=False)
            ax_d.set_xlabel(r"Asset price $S$")
            ax_g.set_xlabel(r"Asset price $S$")
            ax_d.set_ylabel(r"$\Delta = \partial V/\partial S$")
            ax_g.set_ylabel(r"$\Gamma = \partial^2 V/\partial S^2$")
            ax_d.set_title(r"Delta at $t=0$  (vs BT reference)")
            ax_g.set_title(r"Gamma at $t=0$  (vs BT reference)")
            ax_d.legend(fontsize=8); ax_d.grid(True, alpha=0.3)
            ax_g.legend(fontsize=8); ax_g.grid(True, alpha=0.3)
            fig.suptitle(f"Formulation ablation — Greeks vs BT\n{_SUPTITLE_PARAMS}", fontsize=10)
            fig.tight_layout(rect=[0, 0.08, 1, 1])
            fig.text(0.5, 0.01,
                     r"$\Delta$ via autograd: $\partial V_\theta/\partial S$. "
                     r"$\Gamma$ via autograd: $\partial^2 V_\theta/\partial S^2$. "
                     r"BT references via finite differences on $V^{\mathrm{BT}}(S,0)$.",
                     ha="center", va="bottom", fontsize=8, bbox=_BOX_STYLE)
            fig.savefig(comp_dir / "form_greeks.png", dpi=150)
            plt.close(fig)
            logger.info("[OK] form_greeks.png")

    # ------------------------------------------------------------------
    # Plot 8b — Price by time slices (multi-time V(S, t_k))
    # ------------------------------------------------------------------
    # Ported from ablation_singularity_logS.py's price_slices.png:  one
    # column per probed time, all variants overlaid.  BT-tree reference
    # at t=0 (left column) and analytical V_target at t=t1 (right column);
    # no analytical reference at intermediate times so we just show the
    # predicted curves there.  Conceptual companion to form_prices.png
    # (which is the t=0 column only) — useful to see the kink at s*
    # appear as t -> t1.
    slice_results = [
        res for res in results
        if res.get("metrics") is not None
        and res["metrics"].get("slices") is not None
    ]
    if slice_results:
        ref0 = slice_results[0]["metrics"]["slices"]
        t_slices = ref0["t_slices"]
        n_t = len(t_slices)
        s_ax = ref0.get("ref_price_t0").shape and (
            slice_results[0]["s_eval_arr"]
        )
        # ``ref_price_t1`` is the same V_target across every variant — pull
        # it from the first variant that has it (None if none was saved).
        ref_t1 = next(
            (r["metrics"]["slices"].get("ref_price_t1")
             for r in slice_results
             if r["metrics"]["slices"].get("ref_price_t1") is not None),
            None,
        )
        ref_delta_t1 = next(
            (r["metrics"]["slices"].get("ref_delta_t1")
             for r in slice_results
             if r["metrics"]["slices"].get("ref_delta_t1") is not None),
            None,
        )
        ref_gamma_t1 = next(
            (r["metrics"]["slices"].get("ref_gamma_t1")
             for r in slice_results
             if r["metrics"]["slices"].get("ref_gamma_t1") is not None),
            None,
        )
        fig, axes = plt.subplots(1, n_t, figsize=(5 * n_t, 5), sharey=True)
        if n_t == 1:
            axes = [axes]
        for k, t_val in enumerate(t_slices):
            ax = axes[k]
            # Reference overlays where they exist.
            if k == 0:
                ax.plot(s_ax, ref0["ref_price_t0"], "k--", linewidth=1.8,
                        label=r"$V^{\mathrm{BT}}(S,0)$", zorder=10)
            elif np.isclose(t_val, p3.t1) and ref_t1 is not None:
                ax.plot(s_ax, ref_t1, "k--", linewidth=1.8,
                        label=r"$V_{\mathrm{target}}(S)$  (analytical)",
                        zorder=10)
            # Predicted curves, all variants.
            for res in slice_results:
                sl = res["metrics"]["slices"]
                lbl = res.get("label", res.get("name", ""))
                ax.plot(s_ax, sl["price_slices"][k],
                        color=res.get("color"),
                        linestyle=res.get("linestyle", "-"),
                        linewidth=res.get("linewidth", 1.8),
                        label=lbl)
            ax.axvline(p3.K, color="gray", linestyle=":", linewidth=0.8)
            _add_s_star_line(ax, s_star, with_label=(k == 0))
            ax.set_xlabel(r"Asset price $S$")
            if k == 0:
                ax.set_ylabel(r"$V_\theta(S, t)$")
            ax.set_title(rf"$t = {t_val:.3f}$")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
        fig.suptitle(
            f"Formulation ablation — Price by time slices\n{_SUPTITLE_PARAMS}",
            fontsize=10,
        )
        fig.tight_layout(rect=[0, 0.08, 1, 1])
        fig.text(
            0.5, 0.01,
            r"References: BT (binomial tree, $N=2000$) at $t=0$;  "
            r"$V_{\mathrm{target}}(S)=\max(\Phi(S), V^{\mathrm{eur}}(S, t_1))$ "
            r"at $t=t_1$ (analytical Stage A only).  No analytical reference "
            r"at intermediate $t$.",
            ha="center", va="bottom", fontsize=8, bbox=_BOX_STYLE,
        )
        fig.savefig(comp_dir / "form_prices_by_t.png", dpi=150)
        plt.close(fig)
        logger.info("[OK] form_prices_by_t.png")

    # ------------------------------------------------------------------
    # Plot 8c — Greeks comparison by time slices (multi-time Delta, Gamma)
    # ------------------------------------------------------------------
    # Ported from ablation_singularity_logS.py's greeks_comparison.png
    # (the multi-tau version produced by _plot_gt_comparison).  Rows:
    # Delta on top, Gamma on bottom.  Columns: probed times.  BT-FD refs
    # at t=0, V_target-FD refs at t=t1; intermediate t shows predicted
    # curves only.
    if slice_results:
        fig, axes = plt.subplots(2, n_t, figsize=(5 * n_t, 9), sharex=True)
        if n_t == 1:
            axes = axes.reshape(2, 1)
        for k, t_val in enumerate(t_slices):
            ax_d, ax_g = axes[0, k], axes[1, k]
            if k == 0:
                ax_d.plot(s_ax, ref0["ref_delta_t0"], "k--", linewidth=1.6,
                          label=r"$\Delta^{\mathrm{BT}}$", zorder=10)
                ax_g.plot(s_ax, ref0["ref_gamma_t0"], "k--", linewidth=1.6,
                          label=r"$\Gamma^{\mathrm{BT}}$", zorder=10)
            elif np.isclose(t_val, p3.t1) and ref_delta_t1 is not None:
                ax_d.plot(s_ax, ref_delta_t1, "k--", linewidth=1.6,
                          label=r"$\Delta^{\mathrm{tgt}}$", zorder=10)
                ax_g.plot(s_ax, ref_gamma_t1, "k--", linewidth=1.6,
                          label=r"$\Gamma^{\mathrm{tgt}}$", zorder=10)
            for res in slice_results:
                sl = res["metrics"]["slices"]
                lbl = res.get("label", res.get("name", ""))
                ax_d.plot(s_ax, sl["delta_slices"][k],
                          color=res.get("color"),
                          linestyle=res.get("linestyle", "-"),
                          linewidth=res.get("linewidth", 1.5),
                          label=lbl)
                ax_g.plot(s_ax, sl["gamma_slices"][k],
                          color=res.get("color"),
                          linestyle=res.get("linestyle", "-"),
                          linewidth=res.get("linewidth", 1.5),
                          label=lbl)
            ax_d.axvline(p3.K, color="gray", linestyle=":", linewidth=0.8)
            ax_g.axvline(p3.K, color="gray", linestyle=":", linewidth=0.8)
            _add_s_star_line(ax_d, s_star, with_label=(k == 0))
            _add_s_star_line(ax_g, s_star, with_label=False)
            ax_d.set_title(rf"$t = {t_val:.3f}$")
            ax_d.grid(True, alpha=0.3)
            ax_g.set_xlabel(r"Asset price $S$")
            ax_g.grid(True, alpha=0.3)
            if k == 0:
                ax_d.set_ylabel(r"$\Delta = \partial V/\partial S$")
                ax_g.set_ylabel(r"$\Gamma = \partial^2 V/\partial S^2$")
            ax_d.legend(fontsize=7)
            ax_g.legend(fontsize=7)
        fig.suptitle(
            f"Formulation ablation — Greeks by time slices\n{_SUPTITLE_PARAMS}",
            fontsize=10,
        )
        fig.tight_layout(rect=[0, 0.06, 1, 1])
        fig.text(
            0.5, 0.01,
            r"$\Delta, \Gamma$ via autograd on $V_\theta$.  References at "
            r"$t=0$: finite differences on $V^{\mathrm{BT}}$.  References at "
            r"$t=t_1$: finite differences on $V_{\mathrm{target}}$ "
            r"(analytical Stage A only).",
            ha="center", va="bottom", fontsize=8, bbox=_BOX_STYLE,
        )
        fig.savefig(comp_dir / "form_greeks_by_t.png", dpi=150)
        plt.close(fig)
        logger.info("[OK] form_greeks_by_t.png")

    # ------------------------------------------------------------------
    # Plot 9 — Rich summary metrics bar chart
    # ------------------------------------------------------------------
    n_variants = len(results)
    x          = np.arange(n_variants)
    vnames     = [r.get("name", f"v{i}") for i, r in enumerate(results)]

    if any(has_metrics):
        metric_keys   = ["mae_bt", "rel_l2_bt", "rel_l2_atm", "rel_l2_delta", "rel_l2_gamma", "gei"]
        metric_labels = [
            r"MAE vs $V^{\mathrm{BT}}$",
            r"$\varepsilon_{L^2}$ (global)",
            r"$\varepsilon_{L^2}^{\mathrm{ATM}}$",
            r"$\varepsilon_{\Delta}$",
            r"$\varepsilon_{\Gamma}$",
            r"GEI",
        ]

        def _get_metric(res: dict, key: str) -> float:
            if key == "mae_bt":
                return float(res.get("mae_bt", float("nan")))
            m = res.get("metrics")
            return m.get(key, float("nan")) if m is not None else float("nan")

        n_m  = len(metric_keys)
        fig, axes = plt.subplots(1, n_m, figsize=(4.2 * n_m, 6))
        for j, (mk, mn) in enumerate(zip(metric_keys, metric_labels)):
            vals = [_get_metric(res, mk) for res in results]
            bars = axes[j].bar(x, vals, color=colors, edgecolor="black", linewidth=0.7)
            axes[j].set_xticks(x)
            axes[j].set_xticklabels(vnames, rotation=30, ha="right", fontsize=8)
            axes[j].set_title(mn, fontsize=9)
            axes[j].set_yscale("log")
            axes[j].grid(axis="y", alpha=0.3)
            for bar_rect, val in zip(bars, vals):
                if not np.isnan(val):
                    axes[j].text(
                        bar_rect.get_x() + bar_rect.get_width() / 2,
                        val * 1.05, f"{val:.2e}", ha="center", va="bottom", fontsize=7,
                    )
        fig.suptitle(
            f"Formulation ablation — Rich summary metrics\n{_SUPTITLE_PARAMS}, {iters_b} iters",
            fontsize=10,
        )
        fig.subplots_adjust(bottom=0.35, top=0.88, wspace=0.35)
        fig.text(0.5, 0.01, _FORMULA_METRICS,
                 ha="center", va="bottom", fontsize=7.5, bbox=_BOX_STYLE)
        fig.savefig(comp_dir / "form_summary_metrics.png", dpi=150)
        plt.close(fig)
        logger.info("[OK] form_summary_metrics.png")


# ---------------------------------------------------------------------------
# Replot mode
# ---------------------------------------------------------------------------

def _replot(ablation_dir: Path, *, extra_exclude: list[str] | None = None) -> None:
    """Regenerate all plots from a saved run directory (no retraining).

    Handles the sbatch-array layout: when the run was produced by parallel
    ``--variant NAME`` tasks, each task writes its own ``summary_<name>.yaml``
    rather than a shared ``summary.yaml`` (to avoid race-overwrites).  This
    helper assembles a combined ``summary.yaml`` on the fly from those files
    and writes it back so subsequent replots run faster.

    When ``extra_exclude`` is non-empty, the function switches to an
    inspection-only mode: the listed variant names are dropped, per-variant
    figures are left untouched, and the comparison figures are written to
    ``comparison_excl_<sorted-names-joined-by-_>/`` so the canonical folder
    is preserved.  Useful when one regularisation strength clearly wins and
    we want a clean two-way comparison without losing the full figure set.
    """
    extra_exclude = list(extra_exclude or [])
    metadata_path = ablation_dir / "metadata.yaml"
    summary_path  = ablation_dir / "summary.yaml"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.yaml not found in {ablation_dir}")
    with open(metadata_path) as f:
        metadata = yaml.safe_load(f)
    if not summary_path.exists():
        # Assemble from per-variant ``summary_<name>.yaml`` files written by
        # parallel sbatch-array tasks.
        per_variant_files = sorted(ablation_dir.glob("summary_*.yaml"))
        if not per_variant_files:
            raise FileNotFoundError(
                f"Neither summary.yaml nor any summary_*.yaml found in "
                f"{ablation_dir}.  Has every array task completed?"
            )
        merged: dict = {}
        for pv in per_variant_files:
            with open(pv) as f:
                merged.update(yaml.safe_load(f) or {})
        with open(summary_path, "w") as f:
            yaml.dump(merged, f, default_flow_style=False, sort_keys=False,
                      width=float("inf"))
        logger.info(
            f"Assembled summary.yaml from {len(per_variant_files)} per-variant files "
            f"({[p.name for p in per_variant_files]})."
        )
    with open(summary_path) as f:
        summary = yaml.safe_load(f)

    iters_b       = metadata.get("iters_b", 0)
    variants_meta = metadata.get("variants", [])

    # Recover the exercise boundary $s^\star$ at $t_1$.  Newer runs record it
    # in ``metadata.yaml`` at training time; older folders predate the field
    # and we recompute it on the fly when the run used analytical Stage A
    # (the only mode that does not require a saved Stage A network on disk).
    # Trained-A runs without a recorded ``s_star`` are left without a line
    # rather than reloading the network here.
    s_star: float | None = metadata.get("s_star")
    if s_star is None and bool(metadata.get("analytical_stage_a", False)):
        s_star = _compute_s_star_bermudan_analytical()
        logger.info(
            f"metadata.yaml had no s_star; recomputed analytically: "
            f"s_star={s_star:.4f}"
        )
    elif s_star is None:
        # Heuristic for very old analytical runs that did not record the flag
        # explicitly: the folder name carries ``analyticalA`` in those cases.
        if "analyticalA" in ablation_dir.name:
            s_star = _compute_s_star_bermudan_analytical()
            logger.info(
                f"metadata.yaml had no s_star and no analytical_stage_a flag, "
                f"but the run directory name ({ablation_dir.name!r}) carries "
                f"the analyticalA tag; recomputed analytically: "
                f"s_star={s_star:.4f}"
            )
        else:
            logger.warning(
                "metadata.yaml has no s_star and the run does not appear to "
                "be analytical-Stage-A; the exercise-boundary vertical line "
                "will be omitted from S-axis plots."
            )

    excluded = set(extra_exclude)
    results = []
    for idx, v_meta in enumerate(variants_meta):
        vname = v_meta["name"]
        if vname in excluded:
            continue
        vdir  = ablation_dir / f"variant_{vname}"
        for path in (vdir / "hist_b.npz", vdir / "prices.npz"):
            if not path.exists():
                raise FileNotFoundError(f"Missing {path.name} in {vdir}")
        known_style = _STYLE_BY_NAME.get(vname)
        style = known_style if known_style is not None else {
            "name":      vname,
            "label":     v_meta.get("label", vname),
            "color":     _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)],
            "linestyle": ["-", "--", "-.", ":"][idx % 4],
            "linewidth": 2.0,
        }
        results.append(_load_variant_results(vdir, summary.get(vname, {}), style))

    logger.info(f"Loaded {len(results)} variants from {ablation_dir}")

    if not extra_exclude:
        output_subdir = "comparison"
    else:
        output_subdir = "comparison_excl_" + "_".join(sorted(extra_exclude))
        logger.info(
            f"Filtered replot: excluding {sorted(extra_exclude)}; writing "
            f"comparison figures to {output_subdir}/.  Per-variant figures "
            f"left untouched."
        )

    _plot_comparison(results, ablation_dir, iters_b,
                     output_subdir=output_subdir, s_star=s_star)
    if not extra_exclude:
        for res in results:
            _plot_variant(res, ablation_dir / f"variant_{res['name']}")
    logger.info(f"All plots written to {ablation_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation — Bermuda put TC enforcement: hard ETCNN vs soft PINN penalty"
    )
    parser.add_argument("--iters-a", type=int, default=50,
                        help="Stage A iterations for hard_etcnn variant (default 50 — smoke test)")
    parser.add_argument("--iters-b", type=int, default=50,
                        help="Stage B iterations per variant (default 50 — smoke test)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--n-tc",  type=int, default=None, help="Override N_TC")
    parser.add_argument("--n-f",   type=int, default=None, help="Override N_F")
    parser.add_argument(
        "--load-stage-a", type=str, default=None, metavar="PATH",
        help="Path to a pre-trained etcnn_a.pt to skip Stage A training.",
    )
    parser.add_argument(
        "--analytical-stage-a", action="store_true", default=False,
        help="Use the exact Black-Scholes European put as Stage A (no NN trained "
             "for Stage A).  V_target(s) = max(payoff(s), BS_eur_put(s, K, r, σ, T-t1)) "
             "is then built from the analytical formula and shared by every variant — "
             "matches the setup of data/ablation_bermudan/20260511_223436_analyticalA_itersB1000/, "
             "which is the correct reference for an IC-impact control study.  "
             "Mutually exclusive with --load-stage-a.",
    )
    parser.add_argument(
        "--exclude-variant", dest="exclude_variant",
        action="append", default=[], metavar="NAME",
        help=("Variant name to exclude from the comparison figures. "
              "Repeatable.  Only meaningful with --replot: filtered "
              "figures are written to <DIR>/comparison_excl_<sorted-"
              "names-joined-by-_>/, leaving the canonical <DIR>/"
              "comparison/ untouched.  Typical use case: drop the "
              "underperforming regularisation strengths of the soft-"
              "penalty sweep to keep a clean hard-vs-best-soft "
              "comparison."),
    )
    parser.add_argument(
        "--replot", type=str, default=None, metavar="DIR",
        help="Regenerate all plots from an existing run directory (no retraining).",
    )
    parser.add_argument(
        "--variant", type=str, default=None, metavar="NAME",
        help="Run a single variant by NAME (one entry of the VARIANTS "
             "catalogue).  Required for sbatch job arrays — every array task "
             "picks its own NAME from $SLURM_ARRAY_TASK_ID.  In "
             "--analytical-stage-a mode the variants are independent, so the "
             "array tasks have no inter-task dependency.  Cross-variant "
             "comparison plots are SKIPPED when --variant is set; run "
             "``--replot <shared-run-dir>`` once every task has populated "
             "its variant_<name>/ subfolder to assemble the comparison.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Mark this run as a test/smoke run.  The timestamped ablation "
             "directory is prefixed with '_debug_' so test runs are visually "
             "separated from real ones in `ls` (leading underscore sorts "
             "after digits in C/UTF-8 locale, so debug runs land at the bottom) "
             "and can be wiped in bulk with: "
             "find data -type d -name '_debug_*' -prune -exec rm -rf {} +",
    )
    parser.add_argument(
        "--ablation-dir", type=str, default=None, metavar="DIR",
        help="Override the auto-generated timestamped output directory and "
             "write into DIR instead.  Required when every task of an sbatch "
             "job array must land under the SAME parent (otherwise each task "
             "picks a different ``datetime.now()`` second and the parent "
             "folders fragment).  Typical pattern: the launcher creates a "
             "shared dir once on the login node, then passes that path to "
             "each --variant array task via this flag.",
    )
    parser.add_argument(
        "--init-only", action="store_true",
        help="Create the timestamped ablation directory + metadata.yaml + "
             "configs/<variant>.yaml per variant, then exit *without* "
             "training.  Designed for SLURM-array workflows orchestrated by "
             "bash_scripts/cluster/jeanzay/python/experiment_array_launcher.sh"
             ": the launcher captures the absolute directory path printed on "
             "the last stdout line and submits one array task per YAML under "
             "configs/.  Requires --analytical-stage-a because the soft "
             "variants would otherwise depend on the etcnn_a.pt produced by "
             "the hard_etcnn variant — a dependency that defeats parallel "
             "submission.",
    )
    parser.add_argument(
        "--config-dir", type=str, default=None, metavar="DIR",
        help="Folder containing per-variant YAML configs (one file per array "
             "task).  Must be used together with --config-name.  Matches the "
             "convention expected by the generic SLURM-array worker at "
             "bash_scripts/cluster/jeanzay/python/slurm_job_array/"
             "job_array_batch_xp.slurm so this script can be plugged into "
             "the existing Jean Zay launcher without modification.",
    )
    parser.add_argument(
        "--config-name", type=str, default=None, metavar="NAME",
        help="Basename (without .yaml) of the config file inside --config-dir "
             "to load.  The YAML must contain at least 'variant_name' and "
             "'ablation_dir'; effect is equivalent to --variant <variant_name>"
             " --ablation-dir <ablation_dir> plus any iter / stage-A "
             "overrides recorded by the init step.",
    )
    parser.add_argument(
        "--mode", type=str, default="tc-enforcement",
        choices=("tc-enforcement", "mollifiers"),
        help="Selects which subset of the VARIANTS catalogue is iterated:\n"
             "  tc-enforcement: original hard ETCNN vs soft PINN penalty\n"
             "                  ablation (4 variants).\n"
             "  mollifiers:     bermudan analogue of the European-call hard-IC\n"
             "                  ansatz study — 5 variants applying different\n"
             "                  mollifiers to V_target(s) = max((K-s)+, V^E),\n"
             "                  with Stage A = exact Black-Scholes (no NN).\n"
             "                  Implicitly forces --analytical-stage-a.",
    )
    args = parser.parse_args()

    # ── Config-driven entry point (YAML) ────────────────────────────────────
    # Mirrors what the Jean Zay job-array worker passes:
    #     python script.py --config-dir DIR --config-name NAME
    # Translate the YAML into the equivalent --variant / --ablation-dir CLI so
    # the rest of main() stays untouched.  Done first because it can override
    # --variant, which the next block validates.
    if args.config_dir is not None or args.config_name is not None:
        if args.config_dir is None or args.config_name is None:
            parser.error("--config-dir and --config-name must be provided together.")
        config_path = Path(args.config_dir) / f"{args.config_name}.yaml"
        if not config_path.exists():
            parser.error(f"Config file does not exist: {config_path}")
        with open(config_path, encoding="utf-8") as f:
            cfg_yaml = yaml.safe_load(f) or {}
        required = ("variant_name", "ablation_dir")
        missing = [k for k in required if k not in cfg_yaml]
        if missing:
            parser.error(f"Config {config_path} is missing required keys: {missing}")
        args.variant      = cfg_yaml["variant_name"]
        args.ablation_dir = cfg_yaml["ablation_dir"]
        # Remaining keys are optional overrides.  Only apply them when the
        # YAML explicitly carries the field; otherwise keep the argparse
        # default so command-line overrides on the worker still win.
        if cfg_yaml.get("iters_a") is not None:
            args.iters_a = int(cfg_yaml["iters_a"])
        if cfg_yaml.get("iters_b") is not None:
            args.iters_b = int(cfg_yaml["iters_b"])
        if cfg_yaml.get("analytical_stage_a", False):
            args.analytical_stage_a = True
        if cfg_yaml.get("load_stage_a") is not None:
            args.load_stage_a = str(cfg_yaml["load_stage_a"])
        if cfg_yaml.get("weight_decay") is not None:
            args.weight_decay = float(cfg_yaml["weight_decay"])
        if cfg_yaml.get("n_tc") is not None:
            args.n_tc = int(cfg_yaml["n_tc"])
        if cfg_yaml.get("n_f") is not None:
            args.n_f = int(cfg_yaml["n_f"])
        if "device" in cfg_yaml and args.device == "auto":
            args.device = cfg_yaml["device"]
        if "mode" in cfg_yaml and cfg_yaml["mode"] is not None:
            args.mode = cfg_yaml["mode"]

    # ── Filter the catalogue to the requested mode ──────────────────────────
    # Every site below that iterated over the full VARIANTS list now uses
    # ``variants_in_mode`` so that "tc-enforcement" and "mollifiers" runs see
    # only their own catalogue rows.  The original module-level VARIANTS is
    # the union of both modes and stays intact for catalogue lookups.
    variants_in_mode = [v for v in VARIANTS if v.get("mode") == args.mode]
    if not variants_in_mode:
        parser.error(f"--mode {args.mode!r} produced an empty variant list — check the catalogue.")

    # Implicit-flag: ``mollifiers`` always uses analytical Stage A (no NN
    # Stage A) and produces no shared etcnn_a.pt, so the variants are
    # independent and the implicit ``--analytical-stage-a`` saves the user
    # from having to type it.  The existing tc-enforcement default is
    # unchanged (user supplies --analytical-stage-a explicitly).
    if args.mode == "mollifiers" and not args.analytical_stage_a:
        args.analytical_stage_a = True
        logger.info("--mode mollifiers implicitly forces --analytical-stage-a.")

    if args.variant is not None:
        names = [v["name"] for v in variants_in_mode]
        if args.variant not in names:
            parser.error(f"--variant {args.variant!r} not in mode {args.mode!r} catalogue. "
                         f"Available: {names}")

    # Smoke-test guard: any run with an iteration budget far below the real-run
    # value (typical real-run is --iters-b 2000) MUST be tagged with --debug so
    # the output folder lands under `_debug_…/` and is swept by
    # `find data -type d -name '_debug_*' -prune -exec rm -rf {} +`.  Threshold
    # is ~10% of the real-run budget; raise --iters-b above it for real runs,
    # or pass --debug for smoke tests / code-path checks.  Skipped for --replot
    # since no training happens in that mode.
    SMOKE_TEST_ITERS_B_THRESHOLD = 200
    if (args.replot is None
            and args.iters_b < SMOKE_TEST_ITERS_B_THRESHOLD
            and not args.debug):
        parser.error(
            f"--iters-b {args.iters_b} is below the smoke-test threshold "
            f"({SMOKE_TEST_ITERS_B_THRESHOLD}).  Pass --debug to flag this as "
            f"a smoke run (output folder gets the `_debug_` prefix), or raise "
            f"--iters-b above the threshold for a real run."
        )

    if args.replot is not None:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
            handlers=[logging.StreamHandler()],
        )
        logging.getLogger("matplotlib.mathtext").setLevel(logging.WARNING)
        _replot(Path(args.replot), extra_exclude=args.exclude_variant)
        return

    # ── Init-only short-circuit ─────────────────────────────────────────────
    # Creates the shared ablation directory + metadata.yaml + one
    # configs/<variant>.yaml per variant (consumed downstream by the worker
    # via --config-dir / --config-name), prints the absolute path on the
    # last stdout line, and exits *without* touching any torch tensor.
    # Compatible with experiment_array_launcher.sh's three-phase contract.
    if args.init_only:
        if args.variant is not None:
            parser.error(
                "--init-only is incompatible with --variant: one ablation "
                "directory covers every variant, and each array task picks "
                "its own --variant via configs/<name>.yaml."
            )
        if not args.analytical_stage_a:
            parser.error(
                "--init-only requires --analytical-stage-a: the soft_pinn_* "
                "variants otherwise depend on the etcnn_a.pt produced by the "
                "hard_etcnn variant, which serialises the array submission "
                "and defeats the point of parallel job-array execution."
            )
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        stage_a_tag = "analyticalA"
        debug_prefix = "_debug_" if args.debug else ""
        if args.ablation_dir is not None:
            ablation_dir = Path(args.ablation_dir)
        else:
            ablation_dir = (
                script_data_dir(__file__)
                / f"{debug_prefix}{timestamp}_{stage_a_tag}_itersB{args.iters_b}"
            )
        ablation_dir.mkdir(parents=True, exist_ok=True)
        (ablation_dir / "comparison").mkdir(exist_ok=True)
        for v in variants_in_mode:
            for sub in ("training_metrics", "pricing", "greeks", "diagnostics", "models"):
                (ablation_dir / f"variant_{v['name']}" / sub).mkdir(parents=True, exist_ok=True)

        n_tc_resolved = args.n_tc if args.n_tc is not None else p3.N_TC
        n_f_resolved  = args.n_f  if args.n_f  is not None else p3.N_F

        # Exercise boundary at $t_1$.  --init-only is gated on
        # --analytical-stage-a (see the parser.error above), so the analytical
        # bisection is always applicable here and we can record s_star in
        # metadata.yaml up-front for every array task to pick up.
        s_star_init = _compute_s_star_bermudan_analytical()

        with open(ablation_dir / "metadata.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump({
                "command":            " ".join(sys.argv),
                "timestamp":          datetime.now(timezone.utc).isoformat(),
                "fixed": {
                    "g2_type":    "bs",
                    "put_ansatz": False,
                    "LAMBDA_F":   p3.LAMBDA_F,
                },
                "ablation_axes":      ["tc_enforcement_method", "lambda_tc_soft"],
                "variants": [
                    {k: vv for k, vv in var.items()
                     if k not in ("color", "linestyle", "linewidth")}
                    for var in variants_in_mode
                ],
                "mode":               args.mode,
                "iters_a":            args.iters_a,
                "iters_b":            args.iters_b,
                "weight_decay":       args.weight_decay,
                "analytical_stage_a": args.analytical_stage_a,
                "load_stage_a":       args.load_stage_a,
                "N_TC":               n_tc_resolved,
                "N_F":                n_f_resolved,
                "LAMBDA_F":           p3.LAMBDA_F,
                "SEED":               p3.SEED,
                "s_star":             float(s_star_init),
            }, f, default_flow_style=False, sort_keys=False)

        # configs/<variant>.yaml — one per array task.  Snapshots every CLI
        # flag that affects the per-task run so the workers reproduce the
        # user's intent without re-parsing the launcher's arguments.
        configs_dir = ablation_dir / "configs"
        configs_dir.mkdir(exist_ok=True)
        for v in variants_in_mode:
            cfg = {
                "variant_name":       v["name"],
                "ablation_dir":       str(ablation_dir.resolve()),
                "mode":               args.mode,
                "iters_a":            args.iters_a,
                "iters_b":            args.iters_b,
                "weight_decay":       args.weight_decay,
                "analytical_stage_a": args.analytical_stage_a,
                "load_stage_a":       args.load_stage_a,
                "n_tc":               args.n_tc,
                "n_f":                args.n_f,
                "device":             args.device,
            }
            with open(configs_dir / f"{v['name']}.yaml", "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)

        # Absolute path on the *last* stdout line — bash launchers capture it
        # via:  EXPDIR=$(python ... --init-only --analytical-stage-a | tail -n1)
        print(str(ablation_dir.resolve()))
        return

    p3._apply_device_arg(args.device)
    if args.n_tc is not None:
        p3.N_TC = args.n_tc
    if args.n_f is not None:
        p3.N_F = args.n_f

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    if args.analytical_stage_a:
        stage_a_tag = "analyticalA"
    elif args.load_stage_a is not None:
        stage_a_tag = "loadedA"
    else:
        stage_a_tag = f"itersA{args.iters_a}"
    debug_prefix = "_debug_" if args.debug else ""
    if args.ablation_dir is not None:
        ablation_dir = Path(args.ablation_dir)
    else:
        variant_suffix = f"_variant_{args.variant}" if args.variant is not None else ""
        ablation_dir = (
            script_data_dir(__file__)
            / f"{debug_prefix}{timestamp}_{stage_a_tag}_itersB{args.iters_b}{variant_suffix}"
        )
    ablation_dir.mkdir(parents=True, exist_ok=True)
    (ablation_dir / "comparison").mkdir(exist_ok=True)
    # When --variant NAME is set, only create the subfolders for the named
    # variant — other tasks of the array own the other variants' folders.
    variants_to_create = (
        [v for v in variants_in_mode if v["name"] == args.variant]
        if args.variant is not None
        else variants_in_mode
    )
    for v in variants_to_create:
        vdir = ablation_dir / f"variant_{v['name']}"
        # Same subfolder set as ablation_bermudan.py so ``bermudan_problem``
        # can save its plots into the expected locations (pricing/, greeks/,
        # diagnostics/) without throwing FileNotFoundError.
        for sub in ("training_metrics", "pricing", "greeks", "diagnostics", "models"):
            (vdir / sub).mkdir(parents=True, exist_ok=True)

    log_path = ablation_dir / "ablation.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )
    logging.getLogger("matplotlib.mathtext").setLevel(logging.WARNING)

    logger.info("=" * 70)
    logger.info("ABLATION STUDY — Bermuda put TC enforcement (hard ETCNN vs soft PINN)")
    logger.info("=" * 70)
    logger.info(f"  Command: {' '.join(sys.argv)}")
    logger.info(f"  Python {sys.version}")
    import torch as _torch
    logger.info(f"  PyTorch {_torch.__version__}")
    logger.info(f"  CUDA available: {_torch.cuda.is_available()}")
    if _torch.cuda.is_available():
        logger.info(f"  GPU: {_torch.cuda.get_device_name(0)}")
    logger.info(f"  Device: {p3.DEVICE}")
    logger.info(f"  iters_a={args.iters_a}  iters_b={args.iters_b}")
    logger.info(f"  N_TC={p3.N_TC}  N_F={p3.N_F}  LAMBDA_F={p3.LAMBDA_F}  SEED={p3.SEED}")
    logger.info(f"  variants: {[v['name'] for v in variants_in_mode]}")
    logger.info(f"  output:   {ablation_dir}")
    logger.info(f"  log:      {log_path}")

    results:         list[dict] = []
    load_etcnn_a_path: Path | None = None
    v_target_fn = None

    if args.load_stage_a is not None and args.analytical_stage_a:
        parser.error("--load-stage-a and --analytical-stage-a are mutually exclusive.")

    if args.analytical_stage_a:
        logger.info("Stage A: analytical Black-Scholes formula (no Stage A training).")
        v_target_fn, _ = _build_analytical_vtarget()
    elif args.load_stage_a is not None:
        requested = Path(args.load_stage_a)
        if requested.is_dir():
            candidate = requested / "models" / "etcnn_a.pt"
            load_etcnn_a_path = candidate if candidate.exists() else requested / "etcnn_a.pt"
        else:
            load_etcnn_a_path = requested
        if not load_etcnn_a_path.exists():
            raise FileNotFoundError(f"--load-stage-a: not found: {load_etcnn_a_path}")
        logger.info(f"Stage A: reusing pre-trained model from {load_etcnn_a_path}")
        v_target_fn, _ = _load_etcnn_a_and_build_vtarget(load_etcnn_a_path)

    # Exercise boundary $s^\star$ at $t_1$.  Computed once Stage A is set up so
    # every variant's plots can carry the same vertical reference line.
    # - analytical Stage A: closed-form bisection on $\Phi - V^{\mathrm{eur}}_{\mathrm{BS}}$.
    # - loaded Stage A: use the network's hold value via
    #   :func:`learning_option_pricing.pricing.singularity.find_exercise_boundary`.
    # - first-time-trained Stage A (the hard_etcnn variant trains its own A
    #   from scratch and there is no model on disk yet):  s_star is recomputed
    #   from that model right after it is saved, see the variant loop below.
    s_star_value: float | None = None
    if args.analytical_stage_a:
        s_star_value = _compute_s_star_bermudan_analytical()
        logger.info(f"Stage A exercise boundary: s_star = {s_star_value:.4f} "
                    f"(analytical bisection)")
    elif load_etcnn_a_path is not None:
        from learning_option_pricing.pricing.singularity import find_exercise_boundary
        from learning_option_pricing.models.etcnn import AmericanPutETCNN
        _etcnn_a = AmericanPutETCNN(
            K=p3.K, r=p3.r, sigma=p3.sigma, T=p3.T,
            normalize_input=True, g2_type="bs",
        )
        _etcnn_a.load_state_dict(torch.load(load_etcnn_a_path, map_location=p3.DEVICE))
        _etcnn_a.eval().to(p3.DEVICE)
        s_star_value = find_exercise_boundary(
            _etcnn_a, K=p3.K, t1=p3.t1,
            s_lo=float(max(1e-3, p3.S_TRAIN_LO)), s_hi=float(p3.K),
            device=p3.DEVICE,
        )
        logger.info(f"Stage A exercise boundary: s_star = {s_star_value:.4f} "
                    f"(loaded ETCNN_A network)")

    # metadata.yaml is written here (post Stage A setup) so it can record the
    # resolved s_star.  Any crash during Stage A is still recoverable from the
    # ablation log file, so the slight delay relative to the previous early
    # write is acceptable.
    with open(ablation_dir / "metadata.yaml", "w") as f:
        yaml.dump({
            "command":   " ".join(sys.argv),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fixed": {
                "g2_type":    "bs",
                "put_ansatz": False,
                "LAMBDA_F":   p3.LAMBDA_F,
            },
            "ablation_axes": ["tc_enforcement_method", "lambda_tc_soft"],
            "variants": [
                {k: v for k, v in var.items()
                 if k not in ("color", "linestyle", "linewidth")}
                for var in variants_in_mode
            ],
            "iters_a":            args.iters_a,
            "iters_b":            args.iters_b,
            "weight_decay":       args.weight_decay,
            "analytical_stage_a": args.analytical_stage_a,
            "load_stage_a":       args.load_stage_a,
            "N_TC":               p3.N_TC,
            "N_F":                p3.N_F,
            "LAMBDA_F":           p3.LAMBDA_F,
            "SEED":               p3.SEED,
            "s_star": (float(s_star_value) if s_star_value is not None else None),
        }, f, default_flow_style=False, sort_keys=False, width=float("inf"))

    t_ablation_start = time.time()

    # In --variant NAME mode, restrict the loop to the named variant only.
    # The catalogue is the source of truth for the variant config so multiple
    # array tasks pulling --variant from $SLURM_ARRAY_TASK_ID get identical
    # hyperparameters to a single-process run.
    variants_to_run = (
        [v for v in variants_in_mode if v["name"] == args.variant]
        if args.variant is not None
        else variants_in_mode
    )

    for idx, variant in enumerate(variants_to_run):
        vname    = variant["name"]
        vdir     = ablation_dir / f"variant_{vname}"
        tc_type  = variant["tc_type"]

        logger.info("")
        logger.info("=" * 70)
        logger.info(f"VARIANT {idx + 1}/{len(variants_in_mode)}: {vname}  (tc_type={tc_type})")
        logger.info("=" * 70)

        t_start = time.time()

        if tc_type == "hard":
            # -----------------------------------------------------------
            # Hard BC variant — use bermudan_problem (no put_ansatz)
            # -----------------------------------------------------------
            res_p3 = bermudan_problem(
                out_dir=vdir,
                total_iters=[args.iters_a, args.iters_b],
                interp_method="pchip",
                put_ansatz=False,
                weight_decay=args.weight_decay,
                load_etcnn_a=load_etcnn_a_path,
                analytic_a=args.analytical_stage_a,
                g2_type="bs",
                bypass_v=False,
                use_spatial_weight=False,
            )
            model_b    = res_p3["etcnn_b"]
            hist_b     = res_p3["hist_b"]
            s_eval_arr = res_p3["s_eval_arr"]
            bt_prices  = res_p3["bt_prices"]
            nn_prices  = res_p3["etcnn_b_prices"]
            mae_bt     = res_p3["mae_bt"]
            rel_l2_bt  = res_p3["rel_l2_bt"]

            # Save model for reference
            torch.save(model_b.state_dict(), vdir / "models" / "stage_b_model.pt")

            # Extract etcnn_a path for subsequent soft variants — only when
            # Stage A was actually trained as a NN.  In ``--analytical-stage-a``
            # mode there is no etcnn_a.pt to load (it is the closed-form BS
            # formula); the ``v_target_fn`` was built up-front from the
            # analytical formula and is shared by all variants directly.
            if (
                not args.analytical_stage_a
                and load_etcnn_a_path is None
            ):
                load_etcnn_a_path = vdir / "models" / "etcnn_a.pt"
                logger.info(f"  Stage A saved — will be shared with soft variants: {load_etcnn_a_path}")
                v_target_fn, _ = _load_etcnn_a_and_build_vtarget(load_etcnn_a_path)

                # Stage A was trained from scratch by this variant; locate the
                # exercise boundary now and back-fill ``s_star`` in
                # metadata.yaml so the comparison plots (and any later
                # --replot) draw the vertical reference line.
                if s_star_value is None:
                    from learning_option_pricing.pricing.singularity import (
                        find_exercise_boundary,
                    )
                    from learning_option_pricing.models.etcnn import AmericanPutETCNN
                    _etcnn_a_reload = AmericanPutETCNN(
                        K=p3.K, r=p3.r, sigma=p3.sigma, T=p3.T,
                        normalize_input=True, g2_type="bs",
                    )
                    _etcnn_a_reload.load_state_dict(
                        torch.load(load_etcnn_a_path, map_location=p3.DEVICE)
                    )
                    _etcnn_a_reload.eval().to(p3.DEVICE)
                    s_star_value = find_exercise_boundary(
                        _etcnn_a_reload, K=p3.K, t1=p3.t1,
                        s_lo=float(max(1e-3, p3.S_TRAIN_LO)), s_hi=float(p3.K),
                        device=p3.DEVICE,
                    )
                    logger.info(f"  Stage A exercise boundary: s_star = "
                                f"{s_star_value:.4f}  (from freshly trained ETCNN_A)")
                    meta_path = ablation_dir / "metadata.yaml"
                    with open(meta_path) as _f:
                        _meta = yaml.safe_load(_f) or {}
                    _meta["s_star"] = float(s_star_value)
                    with open(meta_path, "w") as _f:
                        yaml.dump(_meta, _f, default_flow_style=False,
                                  sort_keys=False, width=float("inf"))

        elif tc_type.startswith("mollifier_"):
            # -----------------------------------------------------------
            # Mollifier-mode variant — analytical Stage A + ETCNN ansatz
            # with a smoothed-max g_2^B(s, t).  No soft TC penalty
            # (hard IC via construction).  See train_stage_b_mollifier_ansatz
            # for the per-variant g_2^B definition.
            # -----------------------------------------------------------
            torch.manual_seed(p3.SEED)
            model_b, hist_b = train_stage_b_mollifier_ansatz(
                variant=variant,
                total_iters=args.iters_b,
            )
            torch.save(model_b.state_dict(), vdir / "models" / "stage_b_model.pt")
            logger.info(f"  Model saved to {vdir / 'models' / 'stage_b_model.pt'}")

            s_eval_arr, bt_prices, nn_prices, mae_bt, rel_l2_bt = (
                _evaluate_vs_binomial_tree(model_b)
            )

        else:
            # -----------------------------------------------------------
            # Soft BC variant — plain PINN + penalty
            # -----------------------------------------------------------
            if v_target_fn is None:
                raise RuntimeError(
                    "v_target_fn is not set — the hard_etcnn variant must run first "
                    "(or provide --load-stage-a) so that etcnn_a.pt exists."
                )

            lambda_tc_soft = float(variant["lambda_tc_soft"])
            logger.info(f"  lambda_tc_soft={lambda_tc_soft}  LAMBDA_F={p3.LAMBDA_F}")

            torch.manual_seed(p3.SEED)
            model_b = _build_soft_pinn()
            hist_b  = train_stage_b_soft_pinn(
                model=model_b,
                total_iters=args.iters_b,
                v_target_fn=v_target_fn,
                lambda_tc_soft=lambda_tc_soft,
                label=vname,
            )
            torch.save(model_b.state_dict(), vdir / "models" / "stage_b_model.pt")
            logger.info(f"  Model saved to {vdir / 'models' / 'stage_b_model.pt'}")

            s_eval_arr, bt_prices, nn_prices, mae_bt, rel_l2_bt = (
                _evaluate_vs_binomial_tree(model_b)
            )

        elapsed = time.time() - t_start
        logger.info(f"  [{vname}] variant done in {elapsed:.1f}s")

        # --- Rich metrics ---------------------------------------------------
        logger.info(f"  [{vname}] computing rich metrics ...")
        metrics = compute_metrics_stage_b(
            model=model_b,
            hist_b=hist_b,
            bt_prices=bt_prices,
            s_eval_arr=s_eval_arr,
            v_target_fn=v_target_fn if tc_type == "soft" else None,
        )
        logger.info(
            f"  [{vname}] metrics: "
            f"rel_L2={metrics['rel_l2_bt']:.4e}  "
            f"rel_L2_ATM={metrics['rel_l2_atm']:.4e}  "
            f"rel_L2_Delta={metrics['rel_l2_delta']:.4e}  "
            f"rel_L2_Gamma={metrics.get('rel_l2_gamma', float('nan')):.4e}  "
            f"GEI={metrics['gei']:.2f}  "
            f"TC_MAE={metrics['tc_mae']:.4e}"
        )

        res = {
            **variant,
            "hist_b":         hist_b,
            "etcnn_b_prices": nn_prices,
            "bt_prices":      bt_prices,
            "s_eval_arr":     s_eval_arr,
            "mae_bt":         mae_bt,
            "rel_l2_bt":      rel_l2_bt,
            "metrics":        metrics,
        }
        _save_variant_results(res, vdir)
        logger.info(f"  [{vname}] data saved to {vdir}/")

        _plot_variant(res, vdir)
        logger.info(f"  [{vname}] per-variant plots written.")
        results.append(res)

    total_elapsed = time.time() - t_ablation_start

    # ------------------------------------------------------------------
    # Save summary
    # ------------------------------------------------------------------
    # Pair results with the *trained* variants, not the full mode catalogue.
    # In single-variant mode (--variant NAME, every array task) results has
    # length 1 and pairing with variants_in_mode silently truncated to the
    # first catalogue entry — so every per-variant summary_<NAME>.yaml ended
    # up keyed under whichever variant was first in the catalogue, regardless
    # of which variant was actually trained.  The merged summary.yaml then
    # collapsed to a single key on the assembler pass.
    summary: dict = {}
    for v, res in zip(variants_to_run, results):
        m = res.get("metrics", {}) or {}
        summary[v["name"]] = {
            "mae_bt":       float(res["mae_bt"]),
            "rel_l2_bt":    float(res["rel_l2_bt"]),
            "rel_l2_atm":   float(m.get("rel_l2_atm",   float("nan"))),
            "rel_l2_delta": float(m.get("rel_l2_delta", float("nan"))),
            "rel_l2_gamma": float(m.get("rel_l2_gamma", float("nan"))),
            "gei":          float(m.get("gei",          float("nan"))),
            "tc_mae":       float(m.get("tc_mae",       float("nan"))),
        }
    # In --variant NAME mode write a per-variant summary file so 4 parallel
    # array tasks don't race-overwrite the shared summary.yaml.  A follow-up
    # ``--replot <shared-dir>`` step assembles the combined summary.yaml from
    # the metrics.npz of every variant subfolder.
    summary_path = (
        ablation_dir / f"summary_{args.variant}.yaml"
        if args.variant is not None
        else ablation_dir / "summary.yaml"
    )
    with open(summary_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False,
                  width=float("inf"))

    # ------------------------------------------------------------------
    # Comparison plots
    # ------------------------------------------------------------------
    # Cross-variant comparison only makes sense once every variant has run.
    # In --variant NAME mode (one array task per variant) we skip this and
    # leave the comparison to a follow-up ``--replot <run>`` pass once every
    # task has populated its variant_<name>/ subfolder under <run>.
    if args.variant is None:
        logger.info("")
        logger.info("Generating comparison plots ...")
        _plot_comparison(results, ablation_dir, args.iters_b,
                         s_star=s_star_value)
    else:
        logger.info(
            f"  [single-variant mode] skipping cross-variant comparison plots. "
            f"Once every array task has finished, run "
            f"`--replot {ablation_dir}` from the SHARED run directory to assemble them."
        )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 70)
    logger.info("ABLATION SUMMARY — TC enforcement method")
    logger.info("=" * 70)
    logger.info(
        f"  Total wall-clock time: {total_elapsed:.1f}s  "
        f"({total_elapsed/len(variants_in_mode):.1f}s per variant)"
    )
    logger.info(f"  {'Variant':<25} {'MAE':>12} {'rel_L2':>12} {'TC_MAE':>12} {'GEI':>8}")
    logger.info("  " + "-" * 72)
    for v, res in zip(variants_to_run, results):
        m      = res.get("metrics") or {}
        tc_mae = m.get("tc_mae", float("nan"))
        gei    = m.get("gei",    float("nan"))
        logger.info(
            f"  {v['name']:<25} {res['mae_bt']:>12.4e}"
            f" {res['rel_l2_bt']:>12.4e} {tc_mae:>12.4e} {gei:>8.2f}"
        )
    logger.info("  " + "=" * 72)
    logger.info(f"  All outputs saved to: {ablation_dir}")
    logger.info(f"  Comparison plots:     {ablation_dir / 'comparison'}/")
    logger.info("")
    logger.info(f"  To follow progress in real time:")
    logger.info(f"    tail -f {log_path.resolve()}")


if __name__ == "__main__":
    main()
