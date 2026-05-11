"""Ablation study — Bermuda put option, ansatz design choices.

Fixed settings for all variants:
    g2_type      = "bs"   (exact Black-Scholes European put as Stage A anchor)
    tc_enforced  = True   (hard-enforced terminal condition via ETCNN ansatz)

Ablation axes (Stage B only, on [0, t1]):
    extraction         singularity extraction ansatz (U_B = v + ũ_θ)
    bypass_v           operator bypass: drop fictitious put v from PDE loss
    use_spatial_weight inverted-Gaussian weighting of PDE loss near s*

Five variants:
    baseline       extraction=False, bypass_v=False, spatial_weight=False
    +extraction    extraction=True,  bypass_v=False, spatial_weight=False
    +bypass        extraction=True,  bypass_v=True,  spatial_weight=False
    +spatial_wt    extraction=True,  bypass_v=False, spatial_weight=True
    full           extraction=True,  bypass_v=True,  spatial_weight=True

Stage A (ETCNN on [t1, T]) is trained once and shared across all Stage B variants
so that only Stage B design choices are compared.

Metrics computed per variant after training (requires model checkpoint):
    rel_l2_bt          relative L2 error vs binomial tree at t=0
    rel_l2_atm         same restricted to S in [0.9K, 1.1K]
    rel_l2_delta       relative L2 of Delta vs BT finite-difference Delta at t=0
    gei                Gradient Explosion Index = max(|g|) / median(|g|) (first 2/3 iters)
    pde_residual_t     mean |F[V](K, t)| along S=K slice in Stage B interval

Usage (from repo root):
    # Smoke test — 50 iters each stage:
    python3 experiments/python_scripts/exp1/ablation_bermudan.py

    # Full run — 2000 iters each stage (recommended):
    python3 experiments/python_scripts/exp1/ablation_bermudan.py --iters-a 2000 --iters-b 2000

    # Full run on GPU:
    python3 experiments/python_scripts/exp1/ablation_bermudan.py --iters-a 2000 --iters-b 2000 --device cuda

    # Reuse a pre-trained Stage A to save time:
    python3 experiments/python_scripts/exp1/ablation_bermudan.py \\
        --iters-b 2000 --load-stage-a data/ablation_bermudan/<run>/variant_baseline/models/etcnn_a.pt

    # Regenerate all comparison plots from a saved run (no retraining):
    python3 experiments/python_scripts/exp1/ablation_bermudan.py \\
        --replot data/ablation_bermudan/20260422_190033_itersA500_itersB500
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# Make learning_option_pricing importable
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
# Make phase3_training importable as a module
sys.path.insert(0, str(Path(__file__).resolve().parent))

import phase3_training as p3
from phase3_training import bermudan_problem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ablation variant definitions
# ---------------------------------------------------------------------------
VARIANTS: list[dict] = [
    {
        "name": "baseline",
        "label": "Baseline (no extraction)",
        "put_ansatz": False,
        "bypass_v": False,
        "use_spatial_weight": False,
        "interp": "pchip",
        "color": "tab:blue",
        "linestyle": "-",
        "linewidth": 2.0,
    },
    {
        "name": "extraction",
        "label": r"$+$extraction",
        "put_ansatz": True,
        "bypass_v": False,
        "use_spatial_weight": False,
        "interp": "pchip",
        "color": "tab:orange",
        "linestyle": "--",
        "linewidth": 2.0,
    },
    {
        "name": "bypass",
        "label": r"$+$extraction $+$bypass$_v$",
        "put_ansatz": True,
        "bypass_v": True,
        "use_spatial_weight": False,
        "interp": "pchip",
        "color": "tab:green",
        "linestyle": "-.",
        "linewidth": 2.0,
    },
    {
        "name": "spatial_wt",
        "label": r"$+$extraction $+$spatial weight",
        "put_ansatz": True,
        "bypass_v": False,
        "use_spatial_weight": True,
        "interp": "pchip",
        "color": "tab:red",
        "linestyle": ":",
        "linewidth": 2.0,
    },
    {
        "name": "full",
        "label": r"Full (ext $+$ bypass$_v$ $+$ sw)",
        "put_ansatz": True,
        "bypass_v": True,
        "use_spatial_weight": True,
        "interp": "pchip",
        "color": "tab:purple",
        "linestyle": "-",
        "linewidth": 2.5,
    },
]

# Map variant name → visual style (for --replot of runs with different variant lists)
_STYLE_BY_NAME: dict[str, dict] = {v["name"]: v for v in VARIANTS}
_FALLBACK_COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:red",
    "tab:purple", "tab:brown", "tab:pink", "tab:gray",
]

_SUPTITLE_PARAMS = (
    r"Bermudan put — $g_2^{(A)}=V^{\mathrm{BS}}$, "
    r"$K=100$, $r=0.02$, $\sigma=0.25$, $T=1$, $t_1=0.5$, "
    r"$\lambda_f=20$, $\lambda_{tc}=1$ (hard-enforced)"
)


# ---------------------------------------------------------------------------
# Formula annotations (matching exp_singularity style)
# ---------------------------------------------------------------------------
_BOX_STYLE = dict(
    boxstyle="round,pad=0.6", facecolor="lightyellow",
    edgecolor="gray", alpha=0.9,
)

_FORMULA_LF_B = "\n".join([
    r"$\mathcal{L}_f^{(B)} = \frac{1}{N_f}\sum_{i} \mathcal{F}[\tilde{u}^{(B)}_\theta](S_i,\,t_i)^2$",
    r"BSM operator:  $\mathcal{F}[V] = \partial_t V + \frac{\sigma^2}{2}S^2\,\partial_{SS}V + rS\,\partial_S V - rV$",
    r"Collocation: $(S_i, t_i) \sim \mathrm{Uniform}([S_{\min}, S_{\max}] \times [0, t_1])$",
])
_FORMULA_LTC_B = "\n".join([
    r"$\mathcal{L}_{tc}^{(B)} \approx 0$  (terminal condition hard-enforced by ETCNN ansatz)",
    r"$\tilde{u}^{(B)}_\theta(S, t_1) \equiv V^{\mathrm{Berm}}_{\bar{\theta}}(S, t_1)$ exactly",
])
_FORMULA_GRAD = "\n".join([
    r"$\|\nabla_\theta\mathcal{L}\|_2 = \sqrt{\sum_l \|\nabla_{\theta_l}\mathcal{L}\|_2^2}$",
    r"Total loss:  $\mathcal{L}^{(B)} = \lambda_f\,\mathcal{L}_f^{(B)} + \lambda_{tc}\,\mathcal{L}_{tc}^{(B)}"
    rf"$   with $\lambda_f={p3.LAMBDA_F}$,  $\lambda_{{tc}}={p3.LAMBDA_TC}$ (hard $\Rightarrow$ $\mathcal{{L}}_{{tc}}\approx 0$)",
])
_FORMULA_ANSATZ = "\n".join([
    r"Baseline:      $\tilde{u}^{(B)}_\theta(S, t) = (t_1-t)\,u_\theta(S,t) + V^{\mathrm{Berm}}_{\bar{\theta}}(S,t_1)$",
    r"$+$extraction: $\tilde{u}^{(B)}_\theta(S, t) = v(S,t) + (t_1-t)\,u_\theta(S,t) + g_2(S)$",
    r"where $v(S,t)$ is the fictitious European put and $g_2(S)=V^{\mathrm{Berm}}_{\bar{\theta}}(S,t_1)-v(S,t_1)$ is the $C^1$ residual",
    r"$+$bypass$_v$: $v$ is dropped from the PDE residual to prevent derivative cancellation near $s^*$",
    r"$+$spatial weight: $w(S)=1-(1-\varepsilon_w)\exp(-(S-s^*)^2/(2\sigma_w^2))$ applied to PDE loss",
])
_FORMULA_METRICS = "\n".join([
    r"$\varepsilon_{L^2} = \|\tilde{u}^{(B)}_\theta(\cdot,0) - V^{\mathrm{BT}}(\cdot,0)\|_2\,/\,\|V^{\mathrm{BT}}(\cdot,0)\|_2$  (grid $S\in[60,120]$)",
    r"$\varepsilon_{L^2}^{\mathrm{ATM}}$: same restricted to $S\in[0.9K,\,1.1K]$",
    r"$\varepsilon_\Delta$: rel.\ $L^2$ of $\partial_S\tilde{u}^{(B)}_\theta(\cdot,0)$ vs $\Delta^{\mathrm{BT}}$ (finite difference of BT prices)",
    r"$\mathrm{GEI} = \max\|\nabla_\theta\mathcal{L}\| / \mathrm{median}\|\nabla_\theta\mathcal{L}\|$   (first 2/3 of Stage B training)",
])
_FORMULA_PDE_PROFILE = "\n".join([
    r"$\bar{F}(t) = \frac{1}{N}\sum_{i=1}^{N}|\mathcal{F}[\tilde{u}^{(B)}_\theta](K,\,t)|$   ($N=50$ points at $S=K$ per slice)",
    r"BSM operator:  $\mathcal{F}[V] = \partial_t V + \frac{\sigma^2}{2}S^2\,\partial_{SS}V + rS\,\partial_S V - rV$",
    r"Profile computed along the $S=K$ slice for $t\in[0,\,t_1]$ (Stage B interval only)",
])


def _add_formula_box(fig: plt.Figure, text: str, bottom_margin: float = 0.18) -> None:
    """Attach a LaTeX formula annotation box at the bottom of a figure."""
    fig.text(0.5, 0.01, text, ha="center", va="bottom", fontsize=8,
             bbox=_BOX_STYLE, linespacing=1.6)
    fig.subplots_adjust(bottom=bottom_margin)


# ---------------------------------------------------------------------------
# Rich metric computation (requires the trained model)
# ---------------------------------------------------------------------------

def compute_metrics_stage_b(
    model: torch.nn.Module,
    hist_b: dict,
    bt_prices: np.ndarray,
    s_eval_arr: np.ndarray,
) -> dict:
    """Compute evaluation metrics for Stage B.

    Args:
        model:       Trained Stage B model (BermudaETCNN or compatible).
        hist_b:      Training history dict (must contain "grad_norm" key).
        bt_prices:   Binomial-tree reference prices at the evaluation grid points.
        s_eval_arr:  1-D array of asset-price evaluation grid points.

    Returns:
        Dict with keys:
            rel_l2_bt       Relative L2 error vs BT at t=0 (full grid).
            rel_l2_atm      Same restricted to the ATM band |S-K| <= 0.1K.
            rel_l2_delta    Relative L2 of model Delta vs BT finite-diff Delta at t=0.
            gei             Gradient Explosion Index from Stage B training history.
            pde_residual_t  Dict with keys "t" and "residual" — profile along S=K.
    """
    device = p3.DEVICE
    model.eval()

    # --- Price comparison at t=0 -------------------------------------------
    s_tensor = torch.tensor(s_eval_arr, dtype=torch.get_default_dtype(), device=device)
    t_zero   = torch.zeros_like(s_tensor)
    x_eval   = torch.stack([s_tensor, t_zero], dim=1)
    with torch.no_grad():
        nn_prices = model(x_eval).squeeze().cpu().numpy()

    err = nn_prices - bt_prices
    rel_l2_bt = float(np.linalg.norm(err) / (np.linalg.norm(bt_prices) + 1e-10))

    atm_mask   = np.abs(s_eval_arr - p3.K) <= 0.1 * p3.K
    rel_l2_atm = float(
        np.linalg.norm(err[atm_mask]) / (np.linalg.norm(bt_prices[atm_mask]) + 1e-10)
    )

    # --- Delta comparison via autograd vs BT finite differences at t=0 ------
    try:
        s_d = torch.tensor(
            s_eval_arr, dtype=torch.get_default_dtype(), device=device
        ).requires_grad_(True)
        t_d = torch.zeros(len(s_eval_arr), device=device, requires_grad=True)
        x_d = torch.stack([s_d, t_d], dim=1)
        V_d = model(x_d).squeeze()
        (nn_delta,) = torch.autograd.grad(V_d.sum(), s_d, create_graph=False)
        nn_delta_np = nn_delta.detach().cpu().numpy()

        bt_delta_np = np.gradient(bt_prices, s_eval_arr)
        delta_err   = nn_delta_np - bt_delta_np
        rel_l2_delta = float(
            np.linalg.norm(delta_err) / (np.linalg.norm(bt_delta_np) + 1e-10)
        )
    except Exception as exc:
        logger.warning(f"compute_metrics_stage_b: Delta computation failed ({exc}) — skipping.")
        rel_l2_delta = float("nan")

    # --- Gradient Explosion Index from Stage B training history ---------------
    norms = np.array(hist_b.get("grad_norm", []))
    if len(norms) > 0:
        cutoff      = max(1, int(len(norms) * 2 / 3))
        norms_early = norms[:cutoff]
        gei = float(norms_early.max() / (np.median(norms_early) + 1e-10))
    else:
        gei = float("nan")

    # --- PDE residual profile along S=K for t in [0, t1] --------------------
    n_profile = 25
    t_profile_tensor = torch.linspace(
        1e-3, p3.t1 - 1e-3, n_profile, device=device
    )
    pde_residuals = []
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
        logger.warning(f"compute_metrics_stage_b: PDE profile computation failed ({exc}).")
        pde_residuals = [float("nan")] * n_profile

    return {
        "rel_l2_bt":     rel_l2_bt,
        "rel_l2_atm":    rel_l2_atm,
        "rel_l2_delta":  rel_l2_delta,
        "gei":           gei,
        "pde_residual_t": {
            "t":        t_profile_tensor.cpu().tolist(),
            "residual": pde_residuals,
        },
    }


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _save_variant_results(res: dict, vdir: Path) -> None:
    """Persist all numeric data needed for --replot and future analysis."""
    hist = res["hist_b"]
    np.savez_compressed(
        vdir / "hist_b.npz",
        iter=np.array(hist["iter"]),
        loss=np.array(hist["loss"]),
        loss_f=np.array(hist["loss_f"]),
        loss_tc=np.array(hist["loss_tc"]),
        grad_norm=np.array(hist["grad_norm"]),
        lr=np.array(hist["lr"]),
        tc_enforced=np.array([hist.get("tc_enforced", True)]),
    )
    np.savez_compressed(
        vdir / "prices.npz",
        etcnn_b_prices=np.array(res["etcnn_b_prices"]),
        bt_prices=np.array(res["bt_prices"]),
        s_eval_arr=np.array(res["s_eval_arr"]),
    )
    metrics = res.get("metrics")
    if metrics is not None:
        pde_t   = np.array(metrics["pde_residual_t"]["t"])
        pde_res = np.array(metrics["pde_residual_t"]["residual"])
        np.savez_compressed(
            vdir / "metrics.npz",
            rel_l2_bt=np.array([metrics["rel_l2_bt"]]),
            rel_l2_atm=np.array([metrics["rel_l2_atm"]]),
            rel_l2_delta=np.array([metrics["rel_l2_delta"]]),
            gei=np.array([metrics["gei"]]),
            pde_t=pde_t,
            pde_residual=pde_res,
        )


def _load_variant_results(vdir: Path, summary_entry: dict, style: dict) -> dict:
    """Reconstruct a full results dict from .npz files, summary entry, and style."""
    hist_npz   = np.load(vdir / "hist_b.npz")
    prices_npz = np.load(vdir / "prices.npz")
    hist = {
        "iter":        hist_npz["iter"].tolist(),
        "loss":        hist_npz["loss"].tolist(),
        "loss_f":      hist_npz["loss_f"].tolist(),
        "loss_tc":     hist_npz["loss_tc"].tolist(),
        "grad_norm":   hist_npz["grad_norm"].tolist(),
        "lr":          hist_npz["lr"].tolist(),
        "tc_enforced": bool(hist_npz["tc_enforced"][0]) if "tc_enforced" in hist_npz else True,
    }
    # Load rich metrics if available
    metrics_path = vdir / "metrics.npz"
    if metrics_path.exists():
        m = np.load(metrics_path)
        metrics = {
            "rel_l2_bt":     float(m["rel_l2_bt"][0]),
            "rel_l2_atm":    float(m["rel_l2_atm"][0]),
            "rel_l2_delta":  float(m["rel_l2_delta"][0]),
            "gei":           float(m["gei"][0]),
            "pde_residual_t": {
                "t":        m["pde_t"].tolist(),
                "residual": m["pde_residual"].tolist(),
            },
        }
    else:
        metrics = None
    return {
        **style,
        **summary_entry,
        "hist_b":          hist,
        "etcnn_b_prices":  prices_npz["etcnn_b_prices"],
        "bt_prices":       prices_npz["bt_prices"],
        "s_eval_arr":      prices_npz["s_eval_arr"],
        "metrics":         metrics,
    }


# ---------------------------------------------------------------------------
# Per-variant diagnostic plots
# ---------------------------------------------------------------------------

def _plot_variant(res: dict, vdir: Path) -> None:
    """Save per-variant training diagnostic plots."""
    out   = vdir / "training_metrics"
    out.mkdir(exist_ok=True)
    hist  = res["hist_b"]
    label = res.get("label", res.get("name", "variant"))

    # ── Training curves (loss total, loss PDE, TC loss, grad norm) -----------
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].semilogy(hist["iter"], hist["loss"], color="tab:blue")
    axes[0, 0].set_title(r"Total loss $\mathcal{L}^{(B)}$")
    axes[0, 0].set_xlabel("Iteration (Stage B)")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].semilogy(hist["iter"], hist["loss_f"],
                        color="tab:orange", label=r"$\mathcal{L}_f^{(B)}$")
    axes[0, 1].semilogy(hist["iter"], hist["loss_tc"],
                        color="tab:red",    label=r"$\mathcal{L}_{tc}^{(B)}$ (hard $\approx 0$)",
                        linestyle="--", alpha=0.5)
    axes[0, 1].set_title("PDE residual vs TC loss")
    axes[0, 1].set_xlabel("Iteration (Stage B)")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].semilogy(hist["iter"], hist["grad_norm"], color="tab:purple")
    axes[1, 0].set_title(r"Gradient norm $\|\nabla_\theta\mathcal{L}^{(B)}\|_2$")
    axes[1, 0].set_xlabel("Iteration (Stage B)")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].semilogy(hist["iter"], hist["lr"], color="tab:gray")
    axes[1, 1].set_title("Learning rate schedule")
    axes[1, 1].set_xlabel("Iteration (Stage B)")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"{label}\n{_SUPTITLE_PARAMS}", fontsize=10)
    fig.tight_layout(rect=[0, 0.20, 1, 1])
    _add_formula_box(
        fig,
        _FORMULA_LF_B + "\n" + _FORMULA_LTC_B + "\n" + _FORMULA_GRAD,
        bottom_margin=0.22,
    )
    fig.savefig(out / "training_curves.png", dpi=150)
    plt.close(fig)

    # ── PDE residual profile along S=K (requires metrics.npz) ----------------
    metrics = res.get("metrics")
    if metrics is not None and "pde_residual_t" in metrics:
        pde = metrics["pde_residual_t"]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(
            pde["t"], pde["residual"],
            color=res.get("color", "tab:blue"),
            linestyle=res.get("linestyle", "-"),
            linewidth=res.get("linewidth", 2.0),
            marker="o", ms=4,
        )
        ax.axvline(p3.t1, color="tab:red", linestyle="--", linewidth=1.0,
                   label=rf"$t_1={p3.t1}$")
        ax.set_xlabel(r"$t$  (Stage B time)")
        ax.set_ylabel(r"$\mathbb{E}_{S=K}[|\mathcal{F}[\tilde{u}^{(B)}_\theta]|]$")
        ax.set_title(r"Mean PDE residual along $S=K$  (Stage B interval)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.suptitle(f"{label}\n{_SUPTITLE_PARAMS}", fontsize=10)
        fig.tight_layout(rect=[0, 0.18, 1, 1])
        _add_formula_box(fig, _FORMULA_PDE_PROFILE, bottom_margin=0.20)
        fig.savefig(out / "pde_residual_by_t.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Aggregation comparison plots
# ---------------------------------------------------------------------------

def _plot_comparison(results: list[dict], ablation_dir: Path, iters_b: int) -> None:
    """Generate cross-variant comparison plots."""
    comp_dir = ablation_dir / "comparison"
    comp_dir.mkdir(exist_ok=True)

    colors     = [r.get("color",     "tab:blue")  for r in results]
    linestyles = [r.get("linestyle", "-")          for r in results]
    linewidths = [r.get("linewidth", 2.0)          for r in results]
    labels     = [r.get("label",     r.get("name", f"v{i}")) for i, r in enumerate(results)]
    has_metrics = [r.get("metrics") is not None    for r in results]

    # ------------------------------------------------------------------
    # Plot 1 — Stage B PDE residual loss
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, res in enumerate(results):
        hist = res.get("hist_b")
        if hist is None:
            continue
        ax.semilogy(hist["iter"], hist["loss_f"],
                    label=labels[i], color=colors[i],
                    linestyle=linestyles[i], linewidth=linewidths[i])
    ax.set_xlabel("Iteration (Stage B)")
    ax.set_ylabel(r"$\mathcal{L}_f^{(B)}$")
    ax.set_title(
        r"PDE residual loss $\mathcal{L}_f^{(B)}$"
        f"  ({iters_b} iters, $N_f={p3.N_F}$)"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.suptitle(f"Ablation — Stage B PDE residual\n{_SUPTITLE_PARAMS}", fontsize=10)
    fig.tight_layout(rect=[0, 0.18, 1, 1])
    _add_formula_box(fig, _FORMULA_LF_B, bottom_margin=0.20)
    fig.savefig(comp_dir / "abl_loss_pde.png", dpi=150)
    plt.close(fig)
    logger.info("[OK] abl_loss_pde.png")

    # ------------------------------------------------------------------
    # Plot 2 — Stage B TC loss (should be ~0 for hard-enforced)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, res in enumerate(results):
        hist = res.get("hist_b")
        if hist is None:
            continue
        ax.semilogy(hist["iter"], hist["loss_tc"],
                    label=labels[i], color=colors[i],
                    linestyle=linestyles[i], linewidth=linewidths[i])
    ax.set_xlabel("Iteration (Stage B)")
    ax.set_ylabel(r"$\mathcal{L}_{tc}^{(B)}$")
    ax.set_title(
        r"Terminal-condition loss $\mathcal{L}_{tc}^{(B)}$  (hard-enforced: expected $\approx 0$)"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.suptitle(f"Ablation — Stage B TC loss\n{_SUPTITLE_PARAMS}", fontsize=10)
    fig.tight_layout(rect=[0, 0.16, 1, 1])
    _add_formula_box(fig, _FORMULA_LTC_B, bottom_margin=0.18)
    fig.savefig(comp_dir / "abl_loss_tc.png", dpi=150)
    plt.close(fig)
    logger.info("[OK] abl_loss_tc.png")

    # ------------------------------------------------------------------
    # Plot 3 — Gradient norm (signature of training instability)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, res in enumerate(results):
        hist = res.get("hist_b")
        if hist is None:
            continue
        ax.semilogy(hist["iter"], hist["grad_norm"],
                    label=labels[i], color=colors[i],
                    linestyle=linestyles[i], linewidth=linewidths[i], alpha=0.85)
    ax.set_xlabel("Iteration (Stage B)")
    ax.set_ylabel(r"$\|\nabla_\theta\mathcal{L}^{(B)}\|_2$")
    ax.set_title("Gradient norm — signature of singularity-induced instability")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.suptitle(f"Ablation — Gradient norm\n{_SUPTITLE_PARAMS}", fontsize=10)
    fig.tight_layout(rect=[0, 0.14, 1, 1])
    _add_formula_box(fig, _FORMULA_GRAD, bottom_margin=0.16)
    fig.savefig(comp_dir / "abl_grad_norm.png", dpi=150)
    plt.close(fig)
    logger.info("[OK] abl_grad_norm.png")

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
        ax.plot(s_arr, prices,
                label=r"$\tilde{u}^{(B)}_\theta(S,0)$ — " + labels[i],
                color=colors[i], linestyle=linestyles[i], linewidth=linewidths[i])
    ax.set_xlabel("Asset price $S$")
    ax.set_ylabel(r"Price at $t=0$")
    ax.set_title(
        r"$\tilde{u}^{(B)}_\theta(S,0)$ vs $V^{\mathrm{BT}}(S,0)$  —  all variants"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.suptitle(f"Ablation — Pricing comparison at $t=0$\n{_SUPTITLE_PARAMS}", fontsize=10)
    fig.tight_layout(rect=[0, 0.20, 1, 1])
    _add_formula_box(fig, _FORMULA_ANSATZ, bottom_margin=0.22)
    fig.savefig(comp_dir / "abl_prices.png", dpi=150)
    plt.close(fig)
    logger.info("[OK] abl_prices.png")

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
    ax.set_xlabel("Asset price $S$")
    ax.set_ylabel(r"$|\tilde{u}^{(B)}_\theta(S,0) - V^{\mathrm{BT}}(S,0)|$")
    ax.set_title(
        r"Pointwise error vs $V^{\mathrm{BT}}$ at $t=0$"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"Ablation — Pointwise error vs $V^{{\\mathrm{{BT}}}}$ at $t=0$\n{_SUPTITLE_PARAMS}",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.20, 1, 1])
    _add_formula_box(fig, _FORMULA_ANSATZ, bottom_margin=0.22)
    fig.savefig(comp_dir / "abl_error_vs_bt.png", dpi=150)
    plt.close(fig)
    logger.info("[OK] abl_error_vs_bt.png")

    # ------------------------------------------------------------------
    # Plot 6 — PDE residual profile along S=K (requires metrics)
    # ------------------------------------------------------------------
    if any(has_metrics):
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, res in enumerate(results):
            metrics = res.get("metrics")
            if metrics is None or "pde_residual_t" not in metrics:
                continue
            pde = metrics["pde_residual_t"]
            ax.semilogy(
                pde["t"], pde["residual"],
                label=labels[i], color=colors[i],
                linestyle=linestyles[i], linewidth=linewidths[i],
                marker="o", ms=3,
            )
        ax.axvline(p3.t1, color="k", linestyle=":", linewidth=0.8,
                   label=rf"$t_1={p3.t1}$ (Stage B terminal)")
        ax.set_xlabel(r"$t$  (Stage B time)")
        ax.set_ylabel(r"$\mathbb{E}_{S=K}[|\mathcal{F}[\tilde{u}^{(B)}_\theta]|]$")
        ax.set_title(r"Mean PDE residual along $S=K$ as a function of $t$")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.suptitle(
            f"Ablation — PDE residual profile (Stage B, $S=K$ slice)\n{_SUPTITLE_PARAMS}",
            fontsize=10,
        )
        fig.tight_layout(rect=[0, 0.18, 1, 1])
        _add_formula_box(fig, _FORMULA_PDE_PROFILE, bottom_margin=0.20)
        fig.savefig(comp_dir / "abl_pde_residual_by_t.png", dpi=150)
        plt.close(fig)
        logger.info("[OK] abl_pde_residual_by_t.png")

    # ------------------------------------------------------------------
    # Plot 7 — Summary metrics bar chart
    # ------------------------------------------------------------------
    n_variants  = len(results)
    variant_names = [r.get("name", f"v{i}") for i, r in enumerate(results)]
    bar_colors  = colors
    x           = np.arange(n_variants)

    if any(has_metrics):
        # Rich bar chart: 5 panels
        metric_keys  = ["mae_bt", "rel_l2_bt", "rel_l2_atm", "rel_l2_delta", "gei"]
        metric_labels = [
            r"MAE  $= \frac{1}{N}\sum|\tilde{u}-V^{\mathrm{BT}}|$",
            r"$\varepsilon_{L^2}$ (global)",
            r"$\varepsilon_{L^2}^{\mathrm{ATM}}$",
            r"$\varepsilon_{\Delta}$",
            r"GEI",
        ]

        def _get_metric(res: dict, key: str) -> float:
            if key in ("rel_l2_bt", "rel_l2_atm", "rel_l2_delta", "gei"):
                m = res.get("metrics")
                if m is not None:
                    return m.get(key, float("nan"))
            if key == "mae_bt":
                return float(res.get("mae_bt", float("nan")))
            return float("nan")

        n_m = len(metric_keys)
        fig, axes = plt.subplots(1, n_m, figsize=(4.2 * n_m, 6))
        for j, (mk, mn) in enumerate(zip(metric_keys, metric_labels)):
            vals = [_get_metric(res, mk) for res in results]
            bars = axes[j].bar(x, vals, color=bar_colors, edgecolor="black", linewidth=0.7)
            axes[j].set_xticks(x)
            axes[j].set_xticklabels(variant_names, rotation=30, ha="right", fontsize=8)
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
            f"Ablation — Rich summary metrics\n{_SUPTITLE_PARAMS}, {iters_b} iters",
            fontsize=10,
        )
        fig.subplots_adjust(bottom=0.35, top=0.88, wspace=0.35)
        fig.text(0.5, 0.01, _FORMULA_METRICS,
                 ha="center", va="bottom", fontsize=7.5, bbox=_BOX_STYLE)
        fig.savefig(comp_dir / "abl_summary_metrics_rich.png", dpi=150)
        plt.close(fig)
        logger.info("[OK] abl_summary_metrics_rich.png")

    else:
        # Fallback: 2-panel bar chart (MAE + rel_L2) from saved summary
        maes    = [float(r.get("mae_bt",    float("nan"))) for r in results]
        rel_l2s = [float(r.get("rel_l2_bt", float("nan"))) for r in results]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        bars0 = axes[0].bar(x, maes, color=bar_colors, edgecolor="black", linewidth=0.8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(variant_names, rotation=20, ha="right", fontsize=9)
        axes[0].set_ylabel(
            r"$\mathrm{MAE} = \frac{1}{N}\sum_i|\tilde{u}^{(B)}_\theta(S_i,0)-V^{\mathrm{BT}}(S_i,0)|$"
        )
        axes[0].set_title("Mean Absolute Error vs $V^{\\mathrm{BT}}$")
        axes[0].grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars0, maes):
            if not np.isnan(val):
                axes[0].text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.2e}", ha="center", va="bottom", fontsize=8,
                )
        bars1 = axes[1].bar(x, rel_l2s, color=bar_colors, edgecolor="black", linewidth=0.8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(variant_names, rotation=20, ha="right", fontsize=9)
        axes[1].set_ylabel(
            r"$\|\tilde{u}^{(B)}_\theta(\cdot,0)-V^{\mathrm{BT}}(\cdot,0)\|_2 / \|V^{\mathrm{BT}}(\cdot,0)\|_2$"
        )
        axes[1].set_title("Relative $L^2$ error vs $V^{\\mathrm{BT}}$")
        axes[1].grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars1, rel_l2s):
            if not np.isnan(val):
                axes[1].text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.2e}", ha="center", va="bottom", fontsize=8,
                )
        fig.suptitle(f"Ablation — Summary metrics\n{_SUPTITLE_PARAMS}", fontsize=10)
        fig.tight_layout()
        fig.savefig(comp_dir / "abl_summary_metrics.png", dpi=150)
        plt.close(fig)
        logger.info("[OK] abl_summary_metrics.png")


# ---------------------------------------------------------------------------
# Replot mode — regenerate all plots from a saved run directory
# ---------------------------------------------------------------------------

def _replot(ablation_dir: Path) -> None:
    """Load saved variant data and regenerate all comparison and per-variant plots.

    Works with both old runs (no metrics.npz, variant names from metadata) and
    new runs (with metrics.npz).  Variant names and order are read from
    metadata.yaml rather than the hardcoded VARIANTS list, so runs with
    different variant configurations are handled correctly.
    """
    summary_path  = ablation_dir / "summary.yaml"
    metadata_path = ablation_dir / "metadata.yaml"

    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.yaml not found in {ablation_dir}")

    with open(metadata_path) as f:
        metadata = yaml.safe_load(f)
    with open(summary_path) as f:
        summary = yaml.safe_load(f)

    iters_b = metadata.get("iters_b", 0)
    variants_meta = metadata.get("variants", [])

    results = []
    for idx, v_meta in enumerate(variants_meta):
        vname = v_meta["name"]
        vdir  = ablation_dir / f"variant_{vname}"

        hist_path   = vdir / "hist_b.npz"
        prices_path = vdir / "prices.npz"
        if not hist_path.exists() or not prices_path.exists():
            raise FileNotFoundError(
                f"Missing hist_b.npz or prices.npz in {vdir}.\n"
                f"This run predates --replot support. Re-run the ablation to regenerate."
            )

        # Use known style if name is in our VARIANTS, otherwise assign a fallback style
        known_style = _STYLE_BY_NAME.get(vname)
        style = known_style if known_style is not None else {
            "name":      vname,
            "label":     v_meta.get("label", vname),
            "color":     _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)],
            "linestyle": ["-", "--", "-.", ":"][idx % 4],
            "linewidth": 2.0,
        }

        summary_entry = summary.get(vname, {})
        results.append(_load_variant_results(vdir, summary_entry, style))

    logger.info(f"Loaded {len(results)} variants from {ablation_dir}")
    _plot_comparison(results, ablation_dir, iters_b)

    # Regenerate per-variant plots
    for res in results:
        vdir = ablation_dir / f"variant_{res['name']}"
        _plot_variant(res, vdir)

    logger.info(f"All plots written to {ablation_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation study — Bermuda put ansatz design choices (extraction, bypass_v, spatial_weight)"
    )
    parser.add_argument(
        "--iters-a", type=int, default=50,
        help="Stage A iterations (default 50 — smoke test; use 2000+ for production)",
    )
    parser.add_argument(
        "--iters-b", type=int, default=50,
        help="Stage B iterations per variant (default 50 — smoke test; use 2000+ for production)",
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--sigma-w", type=float, default=1.0,
        help="Bandwidth for inverted-Gaussian spatial weight (default 1.0)",
    )
    parser.add_argument(
        "--eps-w", type=float, default=1e-3,
        help="Floor of spatial weight at s* (default 1e-3)",
    )
    parser.add_argument("--n-tc", type=int, default=None, help="Override N_TC")
    parser.add_argument("--n-f",  type=int, default=None, help="Override N_F")
    parser.add_argument(
        "--load-stage-a", type=str, default=None, metavar="PATH",
        help="Path to a pre-trained etcnn_a.pt (or a run directory containing "
             "models/etcnn_a.pt) to skip Stage A training for all variants.",
    )
    parser.add_argument(
        "--replot", type=str, default=None, metavar="DIR",
        help="Regenerate all plots from an existing ablation directory "
             "(no retraining). Reads variant list from metadata.yaml.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Replot mode: just regenerate plots, no training
    # ------------------------------------------------------------------
    if args.replot is not None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(message)s",
            datefmt="%H:%M:%S",
            handlers=[logging.StreamHandler()],
        )
        logging.getLogger("matplotlib.mathtext").setLevel(logging.WARNING)
        _replot(Path(args.replot))
        return

    # ------------------------------------------------------------------
    # Apply settings to phase3_training globals
    # ------------------------------------------------------------------
    p3._apply_device_arg(args.device)
    if args.n_tc is not None:
        p3.N_TC = args.n_tc
    if args.n_f is not None:
        p3.N_F = args.n_f

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    timestamp    = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    ablation_dir = (
        Path("data/ablation_bermudan")
        / f"{timestamp}_itersA{args.iters_a}_itersB{args.iters_b}"
    )
    ablation_dir.mkdir(parents=True, exist_ok=True)
    (ablation_dir / "comparison").mkdir(exist_ok=True)

    for v in VARIANTS:
        vdir = ablation_dir / f"variant_{v['name']}"
        for sub in ("training_metrics", "pricing", "greeks", "diagnostics", "models"):
            (vdir / sub).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Logging — file + console, self-contained as required by CLAUDE.md
    # ------------------------------------------------------------------
    log_path = ablation_dir / "ablation.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path),
        ],
    )
    logging.getLogger("matplotlib.mathtext").setLevel(logging.WARNING)

    logger.info("=" * 70)
    logger.info("ABLATION STUDY — Bermuda put ansatz (extraction / bypass_v / spatial_weight)")
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
    logger.info(f"  N_TC={p3.N_TC}  N_F={p3.N_F}")
    logger.info(f"  LAMBDA_F={p3.LAMBDA_F}  LAMBDA_TC={p3.LAMBDA_TC}")
    logger.info(f"  SEED={p3.SEED}  weight_decay={args.weight_decay}")
    logger.info(f"  variants: {[v['name'] for v in VARIANTS]}")
    logger.info(f"  output:   {ablation_dir}")
    logger.info(f"  log:      {log_path}")

    # ------------------------------------------------------------------
    # Save metadata
    # ------------------------------------------------------------------
    metadata = {
        "command":   " ".join(sys.argv),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fixed": {"g2_type": "bs", "tc_enforced": True},
        "ablation_axes": ["extraction", "bypass_v", "use_spatial_weight"],
        "variants": [
            {k: v for k, v in var.items()
             if k not in ("color", "linestyle", "linewidth")}
            for var in VARIANTS
        ],
        "iters_a":      args.iters_a,
        "iters_b":      args.iters_b,
        "sigma_w":      args.sigma_w,
        "eps_w":        args.eps_w,
        "weight_decay": args.weight_decay,
        "N_TC":         p3.N_TC,
        "N_F":          p3.N_F,
        "LAMBDA_F":     p3.LAMBDA_F,
        "LAMBDA_TC":    p3.LAMBDA_TC,
        "SEED":         p3.SEED,
    }
    with open(ablation_dir / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False,
                  width=float("inf"))

    # ------------------------------------------------------------------
    # Run variants
    # ------------------------------------------------------------------
    results: list[dict] = []

    load_etcnn_a_path: Path | None = None
    if args.load_stage_a is not None:
        requested = Path(args.load_stage_a)
        if requested.is_dir():
            candidate = requested / "models" / "etcnn_a.pt"
            load_etcnn_a_path = candidate if candidate.exists() else requested / "etcnn_a.pt"
        else:
            load_etcnn_a_path = requested
        if not load_etcnn_a_path.exists():
            raise FileNotFoundError(f"--load-stage-a: file not found: {load_etcnn_a_path}")
        logger.info(f"Stage A: reusing pre-trained model from {load_etcnn_a_path}")

    t_ablation_start = time.time()

    for idx, variant in enumerate(VARIANTS):
        vname = variant["name"]
        vdir  = ablation_dir / f"variant_{vname}"

        logger.info("")
        logger.info("=" * 70)
        logger.info(f"VARIANT {idx + 1}/{len(VARIANTS)}: {vname}")
        logger.info(f"  put_ansatz={variant['put_ansatz']}"
                    f"  bypass_v={variant['bypass_v']}"
                    f"  use_spatial_weight={variant['use_spatial_weight']}")
        logger.info("=" * 70)

        t_variant_start = time.time()
        res = bermudan_problem(
            out_dir=vdir,
            total_iters=[args.iters_a, args.iters_b],
            interp_method=variant["interp"],
            put_ansatz=variant["put_ansatz"],
            weight_decay=args.weight_decay,
            load_etcnn_a=load_etcnn_a_path,
            g2_type="bs",
            bypass_v=variant["bypass_v"],
            sigma_w=args.sigma_w,
            eps_w=args.eps_w,
            use_spatial_weight=variant["use_spatial_weight"],
        )
        t_variant_elapsed = time.time() - t_variant_start
        logger.info(f"  [{vname}] training done in {t_variant_elapsed:.1f}s")

        # Compute rich metrics using the returned model
        etcnn_b = res.get("etcnn_b")
        if etcnn_b is not None:
            logger.info(f"  [{vname}] computing rich metrics ...")
            metrics = compute_metrics_stage_b(
                etcnn_b, res["hist_b"], res["bt_prices"], res["s_eval_arr"]
            )
            res["metrics"] = metrics
            logger.info(
                f"  [{vname}] metrics: "
                f"rel_L2={metrics['rel_l2_bt']:.4e}  "
                f"rel_L2_ATM={metrics['rel_l2_atm']:.4e}  "
                f"rel_L2_Delta={metrics['rel_l2_delta']:.4e}  "
                f"GEI={metrics['gei']:.2f}"
            )
        else:
            res["metrics"] = None

        # Add visual style to result
        res.update({k: variant[k] for k in ("color", "linestyle", "linewidth", "label", "name")})

        # Persist data for --replot
        _save_variant_results(res, vdir)
        logger.info(f"  [{vname}] data saved to {vdir}/")

        # Per-variant diagnostic plots
        _plot_variant(res, vdir)
        logger.info(f"  [{vname}] per-variant plots written to {vdir / 'training_metrics'}/")

        logger.info(
            f"  [{vname}] summary — "
            f"MAE={res['mae_bt']:.4e}  rel_L2={res['rel_l2_bt']:.4e}"
            f"  jump@t1={res['jump_at_t1']:.4e}"
        )
        results.append(res)

        # Share Stage A across all subsequent variants
        if load_etcnn_a_path is None:
            load_etcnn_a_path = vdir / "models" / "etcnn_a.pt"
            logger.info(f"  Stage A checkpoint for subsequent variants: {load_etcnn_a_path}")

    total_elapsed = time.time() - t_ablation_start

    # ------------------------------------------------------------------
    # Save aggregated summary
    # ------------------------------------------------------------------
    summary: dict = {}
    for v, res in zip(VARIANTS, results):
        entry = {
            "mae_bt":       float(res["mae_bt"]),
            "rel_l2_bt":    float(res["rel_l2_bt"]),
            "jump_at_t1":   float(res["jump_at_t1"]),
            "etcnn_b_at_K": float(res["etcnn_b_at_K"]),
            "s_star": (
                float(res["s_star"])
                if not (isinstance(res["s_star"], float) and np.isnan(res["s_star"]))
                else "nan"
            ),
        }
        metrics = res.get("metrics")
        if metrics is not None:
            entry.update({
                "rel_l2_atm":    float(metrics.get("rel_l2_atm",   float("nan"))),
                "rel_l2_delta":  float(metrics.get("rel_l2_delta", float("nan"))),
                "gei":           float(metrics.get("gei",          float("nan"))),
            })
        summary[v["name"]] = entry

    with open(ablation_dir / "summary.yaml", "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False,
                  width=float("inf"))

    # ------------------------------------------------------------------
    # Aggregation comparison plots
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("Generating comparison plots ...")
    _plot_comparison(results, ablation_dir, args.iters_b)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 70)
    logger.info("ABLATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Total wall-clock time: {total_elapsed:.1f}s  ({total_elapsed/len(VARIANTS):.1f}s per variant)")
    logger.info(f"  {'Variant':<22} {'MAE':>12} {'rel_L2':>12} {'jump@t1':>12} {'GEI':>8}")
    logger.info("  " + "-" * 68)
    for v, res in zip(VARIANTS, results):
        gei_str = f"{res['metrics']['gei']:.2f}" if res.get("metrics") else "n/a"
        logger.info(
            f"  {v['name']:<22} {res['mae_bt']:>12.4e}"
            f" {res['rel_l2_bt']:>12.4e} {res['jump_at_t1']:>12.4e} {gei_str:>8}"
        )
    logger.info("  " + "=" * 68)
    logger.info(f"  All outputs saved to: {ablation_dir}")
    logger.info(f"  Comparison plots:     {ablation_dir / 'comparison'}/")
    logger.info("")
    logger.info(f"  To follow progress in real time:")
    logger.info(f"    tail -f {log_path.resolve()}")


if __name__ == "__main__":
    main()
