"""Ablation study — Bermuda put option, ansatz design choices.

Fixed settings for all variants:
    g2_type      = "bs"   (exact Black-Scholes European put as Stage A anchor)
    tc_enforced  = True   (hard-enforced terminal condition via ETCNN ansatz)

Ablation axes (Stage B only, on [0, t1]):
    put_ansatz         singularity extraction ansatz (U_B = v + ũ_θ)
    bypass_v           operator bypass: drop fictitious put v from PDE loss
    use_spatial_weight inverted-Gaussian weighting of PDE loss near s*

Five variants:
    baseline       put_ansatz=False, bypass_v=False, spatial_weight=False
    +put-ansatz    put_ansatz=True,  bypass_v=False, spatial_weight=False
    +bypass        put_ansatz=True,  bypass_v=True,  spatial_weight=False
    +spatial_wt    put_ansatz=True,  bypass_v=False, spatial_weight=True
    full           put_ansatz=True,  bypass_v=True,  spatial_weight=True

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

    # Reuse a pre-trained Stage A checkpoint to save time:
    python3 experiments/python_scripts/exp1/ablation_bermudan.py \\
        --iters-b 2000 --load-stage-a data/ablation_bermudan/<run>/variant_baseline/models/etcnn_a.pt

    # Use the exact Black-Scholes formula for Stage A (no Stage A training at all):
    python3 experiments/python_scripts/exp1/ablation_bermudan.py \\
        --iters-b 2000 --analytical-stage-a --device cuda

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
from learning_option_pricing.solvers.binomial_tree import bermuda_put_binomial_tree
from learning_option_pricing.utils.run_context import script_data_dir


# Time slices used for Greeks-vs-BT comparison.  Mirrors the
# ``_GT_TAU_SLICES`` constant in ablation_singularity_logS but expressed as
# absolute Stage B times in [0, t1] rather than time-to-maturity.  The final
# slice deliberately sits very close to t1 to expose how each variant
# behaves near the C0 kink in the intermediate terminal condition.
# Resolved at module load via lambda to defer access to ``p3.t1`` (which
# is the canonical t1 value from phase3_training).
def _build_greeks_t_slices() -> list[float]:
    t1 = float(p3.t1)
    return [
        0.0,
        0.25 * t1,
        0.50 * t1,
        0.75 * t1,
        t1 * (1.0 - 0.02),   # near-singularity: t ≈ t1 from below
    ]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ablation variant definitions
# ---------------------------------------------------------------------------
VARIANTS: list[dict] = [
    {
        "name": "baseline",
        "label": "Baseline (no put-ansatz)",
        "put_ansatz": False,
        "bypass_v": False,
        "use_spatial_weight": False,
        "interp": "pchip",
        "color": "tab:blue",
        "linestyle": "-",
        "linewidth": 2.0,
    },
    {
        "name": "put-ansatz",
        "label": r"$+$put-ansatz",
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
        "label": r"$+$put-ansatz $+$bypass$_v$",
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
        "label": r"$+$put-ansatz $+$spatial weight",
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
_FORMULA_NN = "\n".join([
    r"Backbone $u_\theta:\mathbb{R}^2\to\mathbb{R}$:  $u_\theta(S,t) = W_{\text{out}}\,(\mathrm{Block}_M\circ\cdots\circ\mathrm{Block}_1)(\tanh(W_{\text{in}}\,[S/K,\,t]+b_{\text{in}})) + b_{\text{out}}$",
    r"Residual block: $\mathrm{Block}_m(h) = h + \tanh(W^{(m)}_2\,\tanh(W^{(m)}_1\,h + b^{(m)}_1) + b^{(m)}_2)$"
    r"   with $M=4$ blocks of $L=2$ layers, width $n=50$, input normalisation $S\mapsto S/K$",
])
_FORMULA_ANSATZ = "\n".join([
    r"Baseline:      $\tilde{u}^{(B)}_\theta(S, t) = (t_1-t)\,u_\theta(S,t) + V^{\mathrm{Berm}}_{\bar{\theta}}(S,t_1)$",
    r"$+$put-ansatz: $\tilde{u}^{(B)}_\theta(S, t) = v(S,t) + (t_1-t)\,u_\theta(S,t) + g_2(S)$",
    r"where $v(S,t)$ is the fictitious European put and $g_2(S)=V^{\mathrm{Berm}}_{\bar{\theta}}(S,t_1)-v(S,t_1)$ is the $C^1$ residual",
    r"$+$bypass$_v$: $v$ is dropped from the PDE residual to prevent derivative cancellation near $s^*$",
    r"$+$spatial weight: $w(S)=1-(1-\varepsilon_w)\exp(-(S-s^*)^2/(2\sigma_w^2))$ applied to PDE loss",
    _FORMULA_NN,
])
_FORMULA_METRICS = "\n".join([
    r"$\varepsilon_{L^2} = \|\tilde{u}^{(B)}_\theta(\cdot,0) - V^{\mathrm{BT}}(\cdot,0)\|_2\,/\,\|V^{\mathrm{BT}}(\cdot,0)\|_2$  (grid $S\in[60,120]$)",
    r"$\varepsilon_{L^2}^{\mathrm{ATM}}$: same restricted to $S\in[0.9K,\,1.1K]$",
    r"$\varepsilon_\Delta$: rel.\ $L^2$ of $\partial_S\tilde{u}^{(B)}_\theta(\cdot,0)$ vs $\Delta^{\mathrm{BT}}$ (first finite difference of BT prices)",
    r"$\varepsilon_\Gamma$: rel.\ $L^2$ of $\partial_{SS}\tilde{u}^{(B)}_\theta(\cdot,0)$ vs $\Gamma^{\mathrm{BT}}$ (second finite difference of BT prices, noisy near $S=K$)",
    r"$\mathrm{GEI} = \max\|\nabla_\theta\mathcal{L}\| / \mathrm{median}\|\nabla_\theta\mathcal{L}\|$   (first 2/3 of Stage B training)",
])
_FORMULA_GREEKS = "\n".join([
    r"$\Delta(S) = \partial_S V(S, 0)$,  $\Gamma(S) = \partial_{SS} V(S, 0)$  (via autograd)",
    r"Reference: $\Delta^{\mathrm{BT}} = \nabla_S V^{\mathrm{BT}}$,  $\Gamma^{\mathrm{BT}} = \nabla_S^2 V^{\mathrm{BT}}$  (centred finite differences on the binomial-tree price curve)",
    r"BT Gamma is noisy near $S=K$ because the early-exercise kink is sharper than the BT grid; treat as a qualitative reference only",
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


def _add_s_star_line(ax, s_star) -> None:
    """Draw the exercise-boundary vertical line at ``S = s*`` on an S-axis plot.

    s* is the corner of the intermediate terminal payoff at ``t1``:

        V(s, t1) = max(Phi(s), V^e(s, t1))

    where the two branches of the max meet — the only kink of the Bermudan
    value function on the Stage-B time interval, and a useful visual cue when
    comparing prices / errors / Greeks at ``t=0``.  No-op when ``s_star`` is
    missing, NaN, or the literal string ``"nan"`` (which is how
    ``summary.yaml`` records a Stage A that failed to locate a boundary).
    """
    if s_star is None or s_star == "nan":
        return
    try:
        s_star_f = float(s_star)
    except (TypeError, ValueError):
        return
    if not np.isfinite(s_star_f):
        return
    ax.axvline(s_star_f, color="tab:green", linestyle="-.", linewidth=1.0,
               label=rf"$s^\star \approx {s_star_f:.2f}$ (exercise boundary at $t_1$)")


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
            rel_l2_gamma    Relative L2 of model Gamma vs BT finite-diff Gamma at t=0.
            gei             Gradient Explosion Index from Stage B training history.
            pde_residual_t  Dict with keys "t" and "residual" — profile along S=K.
            greeks          Dict with the per-point Delta/Gamma curves used to
                            compute the rel_L2 metrics, for diagnostic plotting:
                            {"s", "nn_delta", "bt_delta", "nn_gamma", "bt_gamma"}.
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

    # --- Delta and Gamma comparison via autograd vs BT finite differences at t=0
    # Delta = dV/dS  (first autograd pass)
    # Gamma = d^2V/dS^2  (second autograd pass on the same graph)
    # References come from one-sided / centred finite differences on the
    # binomial-tree price curve (np.gradient applied once for Delta, twice
    # for Gamma).  The Gamma reference is noisy at the kink at S = K, so
    # we also report it for diagnostic purposes only.
    try:
        s_d = torch.tensor(
            s_eval_arr, dtype=torch.get_default_dtype(), device=device
        ).requires_grad_(True)
        t_d = torch.zeros(len(s_eval_arr), device=device, requires_grad=True)
        x_d = torch.stack([s_d, t_d], dim=1)
        V_d = model(x_d).squeeze()
        (nn_delta,) = torch.autograd.grad(V_d.sum(), s_d, create_graph=True)
        (nn_gamma,) = torch.autograd.grad(nn_delta.sum(), s_d, create_graph=False)
        nn_delta_np = nn_delta.detach().cpu().numpy()
        nn_gamma_np = nn_gamma.detach().cpu().numpy()

        bt_delta_np = np.gradient(bt_prices, s_eval_arr)
        bt_gamma_np = np.gradient(bt_delta_np, s_eval_arr)
        delta_err   = nn_delta_np - bt_delta_np
        gamma_err   = nn_gamma_np - bt_gamma_np
        rel_l2_delta = float(
            np.linalg.norm(delta_err) / (np.linalg.norm(bt_delta_np) + 1e-10)
        )
        rel_l2_gamma = float(
            np.linalg.norm(gamma_err) / (np.linalg.norm(bt_gamma_np) + 1e-10)
        )
        greeks_curves = {
            "s":        np.asarray(s_eval_arr),
            "nn_delta": nn_delta_np,
            "bt_delta": bt_delta_np,
            "nn_gamma": nn_gamma_np,
            "bt_gamma": bt_gamma_np,
        }
    except Exception as exc:
        logger.warning(f"compute_metrics_stage_b: Greeks computation failed ({exc}) — skipping.")
        rel_l2_delta = float("nan")
        rel_l2_gamma = float("nan")
        greeks_curves = None

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

    # --- Multi-slice Greeks (mirrors ablation_singularity_logS pattern) -------
    # Compute Delta and Gamma at several t-slices in [0, t1] and compare to the
    # binomial-tree reference at each slice.  This exposes the near-singularity
    # behaviour as t → t1 (where V(·, t) has a sharper kink at s*).
    # We reuse the same S evaluation grid as for the t=0 metrics.
    greeks_slices: dict | None = None
    try:
        t_slices = _build_greeks_t_slices()
        n_t = len(t_slices)
        n_s = len(s_eval_arr)
        nn_d  = np.full((n_t, n_s), np.nan)
        nn_g  = np.full((n_t, n_s), np.nan)
        bt_d  = np.full((n_t, n_s), np.nan)
        bt_g  = np.full((n_t, n_s), np.nan)
        for i, t_val in enumerate(t_slices):
            s_d = torch.tensor(
                s_eval_arr, dtype=torch.get_default_dtype(), device=device
            ).requires_grad_(True)
            t_d = torch.full((n_s,), float(t_val), device=device, requires_grad=True)
            V_d = model(torch.stack([s_d, t_d], dim=1)).squeeze()
            (d_nn,)  = torch.autograd.grad(V_d.sum(), s_d, create_graph=True)
            (g_nn,)  = torch.autograd.grad(d_nn.sum(), s_d, create_graph=False)
            nn_d[i, :] = d_nn.detach().cpu().numpy()
            nn_g[i, :] = g_nn.detach().cpu().numpy()

            # BT reference at t > 0: run a Bermudan BT with remaining maturity
            # T-t and a single exercise date at t1-t (or no exercise dates if
            # t >= t1, which puts us past the kink in pure-European territory).
            t_rem  = float(p3.T) - float(t_val)
            t1_rem = float(p3.t1) - float(t_val)
            exer_dates = [t1_rem] if t1_rem > 0.0 else []
            bt_prices_t = np.array([
                bermuda_put_binomial_tree(
                    float(s), float(p3.K), float(p3.r), float(p3.sigma),
                    t_rem, exer_dates, N=2000,
                )
                for s in s_eval_arr
            ])
            bt_d[i, :] = np.gradient(bt_prices_t, s_eval_arr)
            bt_g[i, :] = np.gradient(bt_d[i, :], s_eval_arr)
        greeks_slices = {
            "t":        np.asarray(t_slices, dtype=float),
            "s":        np.asarray(s_eval_arr, dtype=float),
            "nn_delta": nn_d,
            "nn_gamma": nn_g,
            "bt_delta": bt_d,
            "bt_gamma": bt_g,
        }
    except Exception as exc:
        logger.warning(
            f"compute_metrics_stage_b: multi-slice greeks computation failed ({exc})."
        )
        greeks_slices = None

    return {
        "rel_l2_bt":     rel_l2_bt,
        "rel_l2_atm":    rel_l2_atm,
        "rel_l2_delta":  rel_l2_delta,
        "rel_l2_gamma":  rel_l2_gamma,
        "gei":           gei,
        "pde_residual_t": {
            "t":        t_profile_tensor.cpu().tolist(),
            "residual": pde_residuals,
        },
        "greeks":         greeks_curves,
        "greeks_slices":  greeks_slices,
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
        greeks  = metrics.get("greeks")
        npz_payload: dict[str, np.ndarray] = {
            "rel_l2_bt":    np.array([metrics["rel_l2_bt"]]),
            "rel_l2_atm":   np.array([metrics["rel_l2_atm"]]),
            "rel_l2_delta": np.array([metrics["rel_l2_delta"]]),
            "rel_l2_gamma": np.array([metrics.get("rel_l2_gamma", float("nan"))]),
            "gei":          np.array([metrics["gei"]]),
            "pde_t":        pde_t,
            "pde_residual": pde_res,
        }
        if greeks is not None:
            npz_payload.update({
                "greeks_s":        np.asarray(greeks["s"]),
                "greeks_nn_delta": np.asarray(greeks["nn_delta"]),
                "greeks_bt_delta": np.asarray(greeks["bt_delta"]),
                "greeks_nn_gamma": np.asarray(greeks["nn_gamma"]),
                "greeks_bt_gamma": np.asarray(greeks["bt_gamma"]),
            })
        slices = metrics.get("greeks_slices")
        if slices is not None:
            npz_payload.update({
                "greeks_slices_t":        np.asarray(slices["t"]),
                "greeks_slices_s":        np.asarray(slices["s"]),
                "greeks_slices_nn_delta": np.asarray(slices["nn_delta"]),
                "greeks_slices_nn_gamma": np.asarray(slices["nn_gamma"]),
                "greeks_slices_bt_delta": np.asarray(slices["bt_delta"]),
                "greeks_slices_bt_gamma": np.asarray(slices["bt_gamma"]),
            })
        np.savez_compressed(vdir / "metrics.npz", **npz_payload)


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
            "rel_l2_gamma":  float(m["rel_l2_gamma"][0]) if "rel_l2_gamma" in m.files else float("nan"),
            "gei":           float(m["gei"][0]),
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
        if "greeks_slices_t" in m.files:
            metrics["greeks_slices"] = {
                "t":        m["greeks_slices_t"],
                "s":        m["greeks_slices_s"],
                "nn_delta": m["greeks_slices_nn_delta"],
                "nn_gamma": m["greeks_slices_nn_gamma"],
                "bt_delta": m["greeks_slices_bt_delta"],
                "bt_gamma": m["greeks_slices_bt_gamma"],
            }
        else:
            metrics["greeks_slices"] = None
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

    # ── Greeks at t=0: Delta and Gamma (autograd) vs BT finite differences ----
    greeks = metrics.get("greeks") if metrics is not None else None
    if greeks is not None:
        greeks_dir = vdir / "greeks"
        greeks_dir.mkdir(exist_ok=True)
        col = res.get("color", "tab:blue")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
        axes[0].plot(greeks["s"], greeks["bt_delta"],
                     color="black", linestyle="-", linewidth=1.5,
                     label=r"$\Delta^{\mathrm{BT}}$ (reference)")
        axes[0].plot(greeks["s"], greeks["nn_delta"],
                     color=col, linestyle="--", linewidth=2.0,
                     label=r"$\Delta_\theta=\partial_S\tilde{u}^{(B)}_\theta$")
        axes[0].set_title(r"Delta $\Delta(S, 0)$")
        axes[0].set_xlabel("Asset price $S$")
        axes[0].set_ylabel(r"$\Delta$")
        axes[0].axvline(p3.K, color="grey", linestyle=":", linewidth=0.8,
                        label=rf"$S=K={p3.K:g}$")
        _add_s_star_line(axes[0], res.get("s_star"))
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=9)

        axes[1].plot(greeks["s"], greeks["bt_gamma"],
                     color="black", linestyle="-", linewidth=1.5,
                     label=r"$\Gamma^{\mathrm{BT}}$ (reference, noisy)")
        axes[1].plot(greeks["s"], greeks["nn_gamma"],
                     color=col, linestyle="--", linewidth=2.0,
                     label=r"$\Gamma_\theta=\partial_{SS}\tilde{u}^{(B)}_\theta$")
        axes[1].set_title(r"Gamma $\Gamma(S, 0)$")
        axes[1].set_xlabel("Asset price $S$")
        axes[1].set_ylabel(r"$\Gamma$")
        axes[1].axvline(p3.K, color="grey", linestyle=":", linewidth=0.8,
                        label=rf"$S=K={p3.K:g}$")
        _add_s_star_line(axes[1], res.get("s_star"))
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=9)

        fig.suptitle(f"{label}\n{_SUPTITLE_PARAMS}", fontsize=10)
        fig.tight_layout(rect=[0, 0.22, 1, 1])
        _add_formula_box(fig, _FORMULA_GREEKS, bottom_margin=0.24)
        fig.savefig(greeks_dir / "delta_gamma_at_t0.png", dpi=150)
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

    # All variants share the same Stage A, hence the same exercise boundary
    # ``s_star`` at ``t1``.  Pick the first non-NaN entry as the canonical
    # reference to mark on cross-variant S-axis plots.
    s_star_ref = next(
        (r.get("s_star") for r in results
         if r.get("s_star") not in (None, "nan")
         and not (isinstance(r.get("s_star"), float)
                  and not np.isfinite(r["s_star"]))),
        None,
    )

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
    _add_s_star_line(ax, s_star_ref)
    ax.set_xlabel("Asset price $S$")
    ax.set_ylabel(r"Price at $t=0$")
    ax.set_title(
        r"$\tilde{u}^{(B)}_\theta(S,0)$ vs $V^{\mathrm{BT}}(S,0)$  —  all variants"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.suptitle(f"Ablation — Pricing comparison at $t=0$\n{_SUPTITLE_PARAMS}", fontsize=10)
    fig.tight_layout(rect=[0, 0.32, 1, 1])
    _add_formula_box(fig, _FORMULA_ANSATZ, bottom_margin=0.34)
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
    _add_s_star_line(ax, s_star_ref)
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
    fig.tight_layout(rect=[0, 0.32, 1, 1])
    _add_formula_box(fig, _FORMULA_ANSATZ, bottom_margin=0.34)
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
    # Plot 6b — Delta and Gamma curves at t=0 (requires metrics + greeks)
    # ------------------------------------------------------------------
    has_greeks = [
        (res.get("metrics") is not None
         and res["metrics"].get("greeks") is not None)
        for res in results
    ]
    if any(has_greeks):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
        # Use the first variant with greeks data to draw the BT reference once
        ref_idx = next(i for i, ok in enumerate(has_greeks) if ok)
        g_ref   = results[ref_idx]["metrics"]["greeks"]
        s_ref   = g_ref["s"]
        axes[0].plot(s_ref, g_ref["bt_delta"], color="black",
                     linestyle="-", linewidth=1.5, label=r"$\Delta^{\mathrm{BT}}$ (reference)")
        axes[1].plot(s_ref, g_ref["bt_gamma"], color="black",
                     linestyle="-", linewidth=1.5, label=r"$\Gamma^{\mathrm{BT}}$ (reference, noisy)")
        for i, (res, ok) in enumerate(zip(results, has_greeks)):
            if not ok:
                continue
            g = res["metrics"]["greeks"]
            axes[0].plot(g["s"], g["nn_delta"],
                         label=labels[i], color=colors[i],
                         linestyle=linestyles[i], linewidth=linewidths[i])
            axes[1].plot(g["s"], g["nn_gamma"],
                         label=labels[i], color=colors[i],
                         linestyle=linestyles[i], linewidth=linewidths[i])
        for ax, name, sym in zip(axes,
                                 (r"Delta $\Delta(S,0)$", r"Gamma $\Gamma(S,0)$"),
                                 (r"$\Delta$", r"$\Gamma$")):
            ax.set_xlabel("Asset price $S$")
            ax.set_ylabel(sym)
            ax.set_title(name)
            ax.axvline(p3.K, color="grey", linestyle=":", linewidth=0.8,
                       label=rf"$S=K={p3.K:g}$")
            _add_s_star_line(ax, s_star_ref)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc="best")
        fig.suptitle(
            f"Ablation — Greeks at $t=0$  (autograd vs BT finite differences)\n{_SUPTITLE_PARAMS}",
            fontsize=10,
        )
        fig.tight_layout(rect=[0, 0.22, 1, 1])
        _add_formula_box(fig, _FORMULA_GREEKS, bottom_margin=0.24)
        fig.savefig(comp_dir / "abl_greeks.png", dpi=150)
        plt.close(fig)
        logger.info("[OK] abl_greeks.png")

    # ------------------------------------------------------------------
    # Plot 6c — Greeks at multiple t-slices in [0, t1]
    # 2 rows (Delta, Gamma) × n_t columns (one per slice).  Mirrors the
    # `greeks_comparison.png` produced by ablation_singularity_logS so
    # readers can compare both ablations on an equal footing.
    # ------------------------------------------------------------------
    has_greek_slices = [
        (res.get("metrics") is not None
         and res["metrics"].get("greeks_slices") is not None)
        for res in results
    ]
    if any(has_greek_slices):
        ref_idx = next(i for i, ok in enumerate(has_greek_slices) if ok)
        gs_ref  = results[ref_idx]["metrics"]["greeks_slices"]
        t_arr   = np.asarray(gs_ref["t"])
        s_arr   = np.asarray(gs_ref["s"])
        n_t     = len(t_arr)
        fig, axes = plt.subplots(2, n_t, figsize=(4.8 * n_t, 9), sharex=True)
        if n_t == 1:
            axes = axes.reshape(2, 1)
        for j, t_val in enumerate(t_arr):
            ax_d, ax_g = axes[0, j], axes[1, j]
            ax_d.plot(s_arr, gs_ref["bt_delta"][j], "k--", linewidth=1.5,
                      label=r"$\Delta^{\mathrm{BT}}$ (reference)", zorder=10)
            ax_g.plot(s_arr, gs_ref["bt_gamma"][j], "k--", linewidth=1.5,
                      label=r"$\Gamma^{\mathrm{BT}}$ (reference, noisy)", zorder=10)
            for i, (res, ok) in enumerate(zip(results, has_greek_slices)):
                if not ok:
                    continue
                gs = res["metrics"]["greeks_slices"]
                ax_d.plot(np.asarray(gs["s"]), np.asarray(gs["nn_delta"])[j],
                          color=colors[i], linestyle=linestyles[i], linewidth=linewidths[i],
                          label=labels[i])
                ax_g.plot(np.asarray(gs["s"]), np.asarray(gs["nn_gamma"])[j],
                          color=colors[i], linestyle=linestyles[i], linewidth=linewidths[i],
                          label=labels[i])
            ax_d.axvline(p3.K, color="grey", linestyle=":", linewidth=0.8)
            ax_g.axvline(p3.K, color="grey", linestyle=":", linewidth=0.8)
            ax_d.set_title(rf"$t = {float(t_val):.3f}$  (Stage B)")
            ax_d.set_ylabel(r"$\Delta$")
            ax_g.set_xlabel("Asset price $S$")
            ax_g.set_ylabel(r"$\Gamma$")
            ax_d.grid(True, alpha=0.3)
            ax_g.grid(True, alpha=0.3)
            if j == n_t - 1:
                ax_d.legend(fontsize=7, loc="best")
                ax_g.legend(fontsize=7, loc="best")
        fig.suptitle(
            f"Ablation — Greeks at multiple $t$-slices in $[0,\\,t_1]$\n"
            f"{_SUPTITLE_PARAMS}  |  last slice $t\\to t_1^-$ probes the kink",
            fontsize=10,
        )
        fig.tight_layout(rect=[0, 0.10, 1, 1])
        _add_formula_box(fig, _FORMULA_GREEKS, bottom_margin=0.12)
        fig.savefig(comp_dir / "abl_greeks_slices.png", dpi=150)
        plt.close(fig)
        logger.info("[OK] abl_greeks_slices.png")

    # ------------------------------------------------------------------
    # Plot 7 — Summary metrics bar chart
    # ------------------------------------------------------------------
    n_variants  = len(results)
    variant_names = [r.get("name", f"v{i}") for i, r in enumerate(results)]
    bar_colors  = colors
    x           = np.arange(n_variants)

    if any(has_metrics):
        # Rich bar chart: 6 panels (MAE, L2 global, L2 ATM, Delta, Gamma, GEI)
        metric_keys  = ["mae_bt", "rel_l2_bt", "rel_l2_atm", "rel_l2_delta", "rel_l2_gamma", "gei"]
        metric_labels = [
            r"MAE  $= \frac{1}{N}\sum|\tilde{u}-V^{\mathrm{BT}}|$",
            r"$\varepsilon_{L^2}$ (global)",
            r"$\varepsilon_{L^2}^{\mathrm{ATM}}$",
            r"$\varepsilon_{\Delta}$",
            r"$\varepsilon_{\Gamma}$",
            r"GEI",
        ]

        def _get_metric(res: dict, key: str) -> float:
            if key in ("rel_l2_bt", "rel_l2_atm", "rel_l2_delta", "rel_l2_gamma", "gei"):
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
# Add-variant mode — retrain or rescore a single variant inside an existing run
# ---------------------------------------------------------------------------

def _resolve_stage_a_for_add_variant(metadata: dict, ablation_dir: Path
                                     ) -> tuple[bool, Path | None]:
    """Inspect ``metadata.yaml`` and the on-disk run to determine how Stage A
    was produced, so that ``--add-variant`` can reproduce the exact same
    intermediate terminal condition as the original variants.

    Returns:
        ``(analytic_a, load_etcnn_a_path)`` — exactly one of the two will be
        meaningful: ``analytic_a=True`` when the original run used the BS
        formula, otherwise ``load_etcnn_a_path`` points at the etcnn_a.pt
        checkpoint (either the one originally loaded, or — when the original
        run trained Stage A from scratch — the checkpoint saved by the first
        variant that was trained).
    """
    if metadata.get("analytical_stage_a"):
        return True, None

    loaded = metadata.get("loaded_stage_a")
    if loaded:
        return False, Path(loaded)

    # Stage A was trained from scratch in the original run.  Find any saved
    # etcnn_a.pt checkpoint among the variants and reuse it (the script
    # already does this for variants 2..N during a fresh ablation run).
    for v in metadata.get("variants", []):
        candidate = ablation_dir / f"variant_{v['name']}" / "models" / "etcnn_a.pt"
        if candidate.exists():
            return False, candidate
    raise FileNotFoundError(
        f"Could not locate etcnn_a.pt under any variant_*/models/ in "
        f"{ablation_dir}.  Cannot reproduce Stage A for --add-variant."
    )


def _run_add_variant(args: argparse.Namespace) -> None:
    """Implementation of the ``--add-variant NAME:DIR`` CLI mode."""
    if ":" not in args.add_variant:
        raise SystemExit("--add-variant expects NAME:DIR  (e.g. baseline:data/ablation_bermudan/<run>)")
    variant_name, ablation_dir_str = args.add_variant.split(":", 1)
    ablation_dir = Path(ablation_dir_str)
    metadata_path = ablation_dir / "metadata.yaml"
    summary_path  = ablation_dir / "summary.yaml"
    if not metadata_path.exists():
        raise SystemExit(f"metadata.yaml not found in {ablation_dir}")

    with open(metadata_path) as f:
        metadata = yaml.safe_load(f)

    # Look up the variant config from this script's VARIANTS list.  This
    # guarantees consistency with what _replot expects, and with the styles
    # used in the comparison plots.
    matching = [v for v in VARIANTS if v["name"] == variant_name]
    if not matching:
        available = [v["name"] for v in VARIANTS]
        raise SystemExit(
            f"Variant {variant_name!r} not found in VARIANTS. Available: {available}"
        )
    variant = matching[0]

    # Apply phase3 globals from metadata so that this rerun matches the original
    p3._apply_device_arg(args.device)
    if metadata.get("N_TC") is not None:
        p3.N_TC = int(metadata["N_TC"])
    if metadata.get("N_F") is not None:
        p3.N_F = int(metadata["N_F"])

    vdir = ablation_dir / f"variant_{variant_name}"
    for sub in ("training_metrics", "pricing", "greeks", "diagnostics", "models"):
        (vdir / sub).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(ablation_dir / "ablation.log", mode="a"),
        ],
    )
    logging.getLogger("matplotlib.mathtext").setLevel(logging.WARNING)

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"ADD-VARIANT  {variant_name}  ->  {ablation_dir}")
    logger.info(f"  command: {' '.join(sys.argv)}")
    logger.info(f"  put_ansatz={variant['put_ansatz']}"
                f"  bypass_v={variant['bypass_v']}"
                f"  use_spatial_weight={variant['use_spatial_weight']}")
    logger.info(f"  reuse_checkpoint={args.reuse_checkpoint}")
    logger.info("=" * 70)

    analytic_a, load_etcnn_a_path = _resolve_stage_a_for_add_variant(metadata, ablation_dir)
    if analytic_a:
        logger.info("  Stage A: reproduced via analytical Black-Scholes formula (no model file).")
    else:
        logger.info(f"  Stage A: reusing checkpoint {load_etcnn_a_path}")

    # Determine Stage B mode.  Three options are mutually exclusive:
    #   * --reuse-checkpoint : load etcnn_b.pt + iters_b=0 (no training, just rescore)
    #   * --resume           : load stage_b_checkpoint.pt + iters_b=<metadata>
    #   * neither            : full retrain from scratch
    load_etcnn_b_path: Path | None = None
    resume_from_path: Path | None = None
    iters_b = int(metadata.get("iters_b", 0))
    if args.reuse_checkpoint and args.resume:
        raise SystemExit("--reuse-checkpoint and --resume are mutually exclusive.")
    if args.reuse_checkpoint:
        candidate = vdir / "models" / "etcnn_b.pt"
        if not candidate.exists():
            raise FileNotFoundError(
                f"--reuse-checkpoint requested but {candidate} does not exist."
            )
        load_etcnn_b_path = candidate
        iters_b = 0
        logger.info(f"  Stage B: reusing weights {candidate} (iters_b forced to 0)")
    elif args.resume:
        candidate = vdir / "models" / "stage_b_checkpoint.pt"
        if not candidate.exists():
            raise FileNotFoundError(
                f"--resume requested but {candidate} does not exist."
            )
        resume_from_path = candidate
        logger.info(f"  Stage B: resuming from {candidate}, target iters_b={iters_b}")
    else:
        logger.info(f"  Stage B: training from scratch for {iters_b} iterations")

    # g2_gamma source of truth: metadata.yaml is authoritative for an existing
    # run (the saved weights were trained with that gamma).  Allow a CLI
    # override only when no checkpoint is being reused, so the truncation
    # profile that gets plotted always matches the weights on disk.
    metadata_g2_gamma = metadata.get("g2_gamma")
    if args.reuse_checkpoint or args.resume:
        if args.g2_gamma is not None and args.g2_gamma != metadata_g2_gamma:
            logger.warning(
                f"  --g2-gamma={args.g2_gamma} ignored: metadata.yaml records "
                f"g2_gamma={metadata_g2_gamma!r} and a checkpoint is being "
                f"reused — using the metadata value to keep weights/plots consistent."
            )
        effective_g2_gamma = metadata_g2_gamma
    else:
        effective_g2_gamma = (
            args.g2_gamma if args.g2_gamma is not None else metadata_g2_gamma
        )
    if effective_g2_gamma is not None:
        logger.info(f"  Stage B temporal truncation: g2_gamma={effective_g2_gamma}")

    t0 = time.time()
    res = bermudan_problem(
        out_dir=vdir,
        total_iters=[0, iters_b],
        interp_method=variant["interp"],
        put_ansatz=variant["put_ansatz"],
        weight_decay=float(metadata.get("weight_decay", 0.0)),
        load_etcnn_a=load_etcnn_a_path,
        analytic_a=analytic_a,
        g2_type=str(metadata.get("fixed", {}).get("g2_type", "bs")),
        bypass_v=variant["bypass_v"],
        sigma_w=float(metadata.get("sigma_w", 1.0)),
        eps_w=float(metadata.get("eps_w", 1e-3)),
        use_spatial_weight=variant["use_spatial_weight"],
        g2_gamma=effective_g2_gamma,
        load_etcnn_b=load_etcnn_b_path,
        stage_b_checkpoint_every=int(args.checkpoint_every),
        stage_b_resume_from=resume_from_path,
    )
    elapsed = time.time() - t0
    logger.info(f"  [{variant_name}] bermudan_problem returned in {elapsed:.1f}s")

    # In --reuse-checkpoint mode, train_model never ran, so hist_b is empty.
    # Restore the original training history from disk so that GEI (which is
    # derived from the gradient-norm trace) stays meaningful and so that
    # _save_variant_results does not overwrite the saved hist_b.npz with the
    # empty placeholder.
    if args.reuse_checkpoint:
        hist_path = vdir / "hist_b.npz"
        if hist_path.exists():
            hist_npz = np.load(hist_path)
            res["hist_b"] = {
                "iter":        hist_npz["iter"].tolist(),
                "loss":        hist_npz["loss"].tolist(),
                "loss_f":      hist_npz["loss_f"].tolist(),
                "loss_tc":     hist_npz["loss_tc"].tolist(),
                "grad_norm":   hist_npz["grad_norm"].tolist(),
                "lr":          hist_npz["lr"].tolist(),
                "tc_enforced": bool(hist_npz["tc_enforced"][0])
                                if "tc_enforced" in hist_npz else True,
            }
            logger.info(f"  [{variant_name}] hist_b restored from {hist_path} "
                        f"({len(res['hist_b']['iter'])} steps)")

    etcnn_b = res.get("etcnn_b")
    if etcnn_b is not None:
        metrics = compute_metrics_stage_b(
            etcnn_b, res["hist_b"], res["bt_prices"], res["s_eval_arr"]
        )
        res["metrics"] = metrics
        logger.info(
            f"  [{variant_name}] metrics: "
            f"rel_L2={metrics['rel_l2_bt']:.4e}  "
            f"rel_L2_ATM={metrics['rel_l2_atm']:.4e}  "
            f"rel_L2_Delta={metrics['rel_l2_delta']:.4e}  "
            f"rel_L2_Gamma={metrics['rel_l2_gamma']:.4e}  "
            f"GEI={metrics['gei']:.2f}"
        )
    else:
        res["metrics"] = None

    res.update({k: variant[k] for k in ("color", "linestyle", "linewidth", "label", "name")})
    _save_variant_results(res, vdir)
    _plot_variant(res, vdir)

    # Update summary.yaml: replace the entry for this variant if it exists,
    # otherwise insert it at the position matching VARIANTS so subsequent plots
    # keep the canonical ordering.
    summary: dict
    if summary_path.exists():
        with open(summary_path) as f:
            summary = yaml.safe_load(f) or {}
    else:
        summary = {}

    new_entry = {
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
        new_entry.update({
            "rel_l2_atm":    float(metrics.get("rel_l2_atm",   float("nan"))),
            "rel_l2_delta":  float(metrics.get("rel_l2_delta", float("nan"))),
            "rel_l2_gamma":  float(metrics.get("rel_l2_gamma", float("nan"))),
            "gei":           float(metrics.get("gei",          float("nan"))),
        })

    canonical_order = [v["name"] for v in VARIANTS]
    summary[variant_name] = new_entry
    summary = {name: summary[name] for name in canonical_order if name in summary}

    with open(summary_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False, width=float("inf"))
    logger.info(f"  [{variant_name}] summary.yaml updated.")

    logger.info(f"  [{variant_name}] regenerating comparison plots ...")
    _replot(ablation_dir)
    logger.info(f"\nAdd-variant {variant_name!r} done — results in {ablation_dir}")


# ---------------------------------------------------------------------------
# Single-variant mode — train a single variant in a fresh ablation directory
# ---------------------------------------------------------------------------

def _run_single_variant(args: argparse.Namespace) -> None:
    """Implementation of the ``--variant NAME`` CLI mode.

    Creates a fresh timestamped output directory (with a ``_variant_<NAME>``
    suffix) holding only the requested variant — no cross-variant comparison
    plots are generated.  Intended for smoke tests and quick iteration on a
    single architecture without paying for the full 5-variant sweep.
    """
    matching = [v for v in VARIANTS if v["name"] == args.variant]
    if not matching:
        available = [v["name"] for v in VARIANTS]
        raise SystemExit(
            f"--variant {args.variant!r} not found in VARIANTS. Available: {available}"
        )
    variant = matching[0]

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
    ablation_dir = (
        script_data_dir(__file__)
        / f"{timestamp}_{stage_a_tag}_itersB{args.iters_b}_variant_{args.variant}"
    )
    vdir = ablation_dir / f"variant_{args.variant}"
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
    logger.info(f"SINGLE-VARIANT MODE: {args.variant}")
    logger.info("=" * 70)
    logger.info(f"  Command: {' '.join(sys.argv)}")
    logger.info(f"  output:  {ablation_dir}")
    logger.info(f"  log:     {log_path}")

    # Build the same Stage A wiring as the full ablation main path
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

    resume_from_path: Path | None = None
    if args.resume:
        candidate = vdir / "models" / "stage_b_checkpoint.pt"
        if not candidate.exists():
            raise FileNotFoundError(
                f"--resume requested but {candidate} does not exist."
            )
        resume_from_path = candidate
        logger.info(f"  Stage B: resuming from {candidate}")

    # Save a minimal metadata.yaml so that --replot / --add-variant can later
    # operate on this directory just like on a full-ablation directory.
    metadata = {
        "command":           " ".join(sys.argv),
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "mode":              "single-variant",
        "fixed":             {"g2_type": "bs", "tc_enforced": True},
        "ablation_axes":     ["put_ansatz", "bypass_v", "use_spatial_weight"],
        "variants": [
            {k: v for k, v in variant.items()
             if k not in ("color", "linestyle", "linewidth")},
        ],
        "iters_a":            None if (args.load_stage_a is not None or args.analytical_stage_a) else args.iters_a,
        "loaded_stage_a":     args.load_stage_a,
        "analytical_stage_a": args.analytical_stage_a,
        "iters_b":            args.iters_b,
        "sigma_w":            args.sigma_w,
        "eps_w":              args.eps_w,
        "g2_gamma":           args.g2_gamma,
        "weight_decay":       args.weight_decay,
        "N_TC":               p3.N_TC,
        "N_F":                p3.N_F,
        "LAMBDA_F":           p3.LAMBDA_F,
        "LAMBDA_TC":          p3.LAMBDA_TC,
        "SEED":               p3.SEED,
        "checkpoint_every":   args.checkpoint_every,
    }
    with open(ablation_dir / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False, width=float("inf"))

    t0 = time.time()
    res = bermudan_problem(
        out_dir=vdir,
        total_iters=[args.iters_a, args.iters_b],
        interp_method=variant["interp"],
        put_ansatz=variant["put_ansatz"],
        weight_decay=args.weight_decay,
        load_etcnn_a=load_etcnn_a_path,
        analytic_a=args.analytical_stage_a,
        g2_type="bs",
        bypass_v=variant["bypass_v"],
        sigma_w=args.sigma_w,
        eps_w=args.eps_w,
        use_spatial_weight=variant["use_spatial_weight"],
        g2_gamma=args.g2_gamma,
        stage_b_checkpoint_every=int(args.checkpoint_every),
        stage_b_resume_from=resume_from_path,
    )
    logger.info(f"  [{args.variant}] training done in {time.time() - t0:.1f}s")

    etcnn_b = res.get("etcnn_b")
    if etcnn_b is not None:
        metrics = compute_metrics_stage_b(
            etcnn_b, res["hist_b"], res["bt_prices"], res["s_eval_arr"]
        )
        res["metrics"] = metrics
        logger.info(
            f"  [{args.variant}] metrics: "
            f"rel_L2={metrics['rel_l2_bt']:.4e}  "
            f"rel_L2_Delta={metrics['rel_l2_delta']:.4e}  "
            f"rel_L2_Gamma={metrics['rel_l2_gamma']:.4e}  "
            f"GEI={metrics['gei']:.2f}"
        )
    else:
        res["metrics"] = None

    res.update({k: variant[k] for k in ("color", "linestyle", "linewidth", "label", "name")})
    _save_variant_results(res, vdir)
    _plot_variant(res, vdir)

    summary = {
        args.variant: {
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
    }
    m = res.get("metrics")
    if m is not None:
        summary[args.variant].update({
            "rel_l2_atm":   float(m.get("rel_l2_atm",   float("nan"))),
            "rel_l2_delta": float(m.get("rel_l2_delta", float("nan"))),
            "rel_l2_gamma": float(m.get("rel_l2_gamma", float("nan"))),
            "gei":          float(m.get("gei",          float("nan"))),
        })
    with open(ablation_dir / "summary.yaml", "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False, width=float("inf"))
    logger.info(f"Single-variant {args.variant!r} done — results in {ablation_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation study — Bermuda put ansatz design choices (put-ansatz, bypass_v, spatial_weight)"
    )
    parser.add_argument(
        "--iters-a", type=int, default=None,
        help="Stage A iterations (default 50 — smoke test; use 2000+ for production). "
             "Mutually exclusive with --load-stage-a.",
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
    parser.add_argument(
        "--g2-gamma", type=float, default=None, metavar="GAMMA",
        help="Curvature gamma >= 0 of the Stage B temporal truncation profile "
             "h(t) = exp(-gamma * (t1 - t)^2) applied to the g2 field. "
             "None (default) disables the truncation (h ≡ 1, standard ETCNN). "
             "When set, an additional diagnostics plot "
             "(plotBh_temporal_truncation.png) is produced by bermudan_problem.",
    )
    parser.add_argument("--n-tc", type=int, default=None, help="Override N_TC")
    parser.add_argument("--n-f",  type=int, default=None, help="Override N_F")
    parser.add_argument(
        "--load-stage-a", type=str, default=None, metavar="PATH",
        help="Path to a pre-trained etcnn_a.pt (or a run directory containing "
             "models/etcnn_a.pt) to skip Stage A training for all variants. "
             "Mutually exclusive with --iters-a and --analytical-stage-a.",
    )
    parser.add_argument(
        "--analytical-stage-a", action="store_true", default=False,
        help="Replace Stage A with the exact Black-Scholes European put formula "
             "(no neural network trained for Stage A). The terminal condition at "
             "t1 becomes max(payoff(s), BS_put(s, K, r, sigma, T-t1)) exactly, "
             "isolating Stage B from any Stage A approximation error. "
             "Mutually exclusive with --iters-a and --load-stage-a.",
    )
    parser.add_argument(
        "--replot", type=str, default=None, metavar="DIR",
        help="Regenerate all plots from an existing ablation directory "
             "(no retraining). Reads variant list from metadata.yaml.",
    )
    parser.add_argument(
        "--add-variant", type=str, default=None, metavar="NAME:DIR",
        help="Train (or rescore) a single variant inside an existing ablation "
             "directory and update its entry in summary.yaml. NAME must match "
             "one of the names in the VARIANTS list of this script. The Stage A "
             "mode (analytical / loaded / from-scratch) and all other run "
             "hyperparameters are read from metadata.yaml — they cannot be "
             "overridden so that the resulting variant remains comparable to "
             "the existing ones.",
    )
    parser.add_argument(
        "--reuse-checkpoint", action="store_true", default=False,
        help="When used with --add-variant: load the existing "
             "<vdir>/models/etcnn_b.pt and skip Stage B training (iters_b=0). "
             "Only the post-training evaluation runs, which is the right mode "
             "for backfilling new metrics (e.g. Gamma) on previously trained "
             "variants without retraining them.",
    )
    parser.add_argument(
        "--variant", type=str, default=None, metavar="NAME",
        help="Run a single variant NAME in a fresh ablation directory "
             "(skips comparison plots that span variants). Useful for smoke "
             "tests and quick iteration on a single architecture. NAME must "
             "match an entry in this script's VARIANTS list.",
    )
    parser.add_argument(
        "--resume", action="store_true", default=False,
        help="When used with --add-variant or --variant: resume Stage B "
             "training from <vdir>/models/stage_b_checkpoint.pt instead of "
             "starting from scratch. The checkpoint includes model, optimizer, "
             "scheduler, full RNG state and accumulated training history, so "
             "the resumed run produces results indistinguishable from an "
             "uninterrupted one (provided --iters-b matches the target).",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=0, metavar="K",
        help="Save a Stage B training checkpoint every K iterations (default "
             "0 = disabled, only one final checkpoint at the end). Use a "
             "positive value to cap the cost of an interruption — e.g. "
             "--checkpoint-every 100 means losing at most ~100 iterations "
             "of work if the job is killed mid-training.",
    )
    args = parser.parse_args()

    # Exactly one of {train from scratch, load checkpoint, analytical formula}
    # must be chosen for Stage A.  --iters-a belongs only to the train-from-scratch
    # path, so it is mutually exclusive with the other two options.
    stage_a_modes_active = sum([
        args.load_stage_a is not None,
        args.analytical_stage_a,
    ])
    if stage_a_modes_active > 1:
        parser.error("--load-stage-a and --analytical-stage-a are mutually exclusive: "
                     "choose at most one Stage A override.")
    if args.iters_a is not None and stage_a_modes_active > 0:
        parser.error("--iters-a is only valid when training Stage A from scratch. "
                     "Drop it when using --load-stage-a or --analytical-stage-a.")
    if args.iters_a is None:
        args.iters_a = 50  # default for training-from-scratch runs

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
    # Add-variant mode: train (or rescore) a single variant inside an
    # existing ablation directory and update its summary.yaml entry.
    # ------------------------------------------------------------------
    if args.add_variant is not None:
        _run_add_variant(args)
        return

    # ------------------------------------------------------------------
    # Single-variant mode: train one variant in a fresh ablation directory.
    # Useful for smoke tests and quick iteration on a single architecture.
    # ------------------------------------------------------------------
    if args.variant is not None:
        _run_single_variant(args)
        return

    if args.reuse_checkpoint and args.add_variant is None:
        parser.error("--reuse-checkpoint only makes sense with --add-variant.")
    if args.resume and args.add_variant is None and args.variant is None:
        parser.error("--resume requires --add-variant or --variant.")

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
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    if args.analytical_stage_a:
        stage_a_tag = "analyticalA"
    elif args.load_stage_a is not None:
        stage_a_tag = "loadedA"
    else:
        stage_a_tag = f"itersA{args.iters_a}"
    ablation_dir = (
        script_data_dir(__file__)
        / f"{timestamp}_{stage_a_tag}_itersB{args.iters_b}"
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
    logger.info("ABLATION STUDY — Bermuda put ansatz (put-ansatz / bypass_v / spatial_weight)")
    logger.info("=" * 70)
    logger.info(f"  Command: {' '.join(sys.argv)}")
    logger.info(f"  Python {sys.version}")
    import torch as _torch
    logger.info(f"  PyTorch {_torch.__version__}")
    logger.info(f"  CUDA available: {_torch.cuda.is_available()}")
    if _torch.cuda.is_available():
        logger.info(f"  GPU: {_torch.cuda.get_device_name(0)}")
    logger.info(f"  Device: {p3.DEVICE}")
    if args.analytical_stage_a:
        logger.info(f"  iters_a=N/A (Stage A = exact Black-Scholes formula)  iters_b={args.iters_b}")
    elif args.load_stage_a is not None:
        logger.info(f"  iters_a=N/A (Stage A loaded from checkpoint)  iters_b={args.iters_b}")
    else:
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
        "ablation_axes": ["put_ansatz", "bypass_v", "use_spatial_weight"],
        "variants": [
            {k: v for k, v in var.items()
             if k not in ("color", "linestyle", "linewidth")}
            for var in VARIANTS
        ],
        "iters_a":           None if (args.load_stage_a is not None or args.analytical_stage_a) else args.iters_a,
        "loaded_stage_a":    args.load_stage_a,
        "analytical_stage_a": args.analytical_stage_a,
        "iters_b":           args.iters_b,
        "sigma_w":      args.sigma_w,
        "eps_w":        args.eps_w,
        "g2_gamma":     args.g2_gamma,
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
            analytic_a=args.analytical_stage_a,
            g2_type="bs",
            bypass_v=variant["bypass_v"],
            sigma_w=args.sigma_w,
            eps_w=args.eps_w,
            use_spatial_weight=variant["use_spatial_weight"],
            g2_gamma=args.g2_gamma,
            stage_b_checkpoint_every=int(args.checkpoint_every),
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
                f"rel_L2_Gamma={metrics['rel_l2_gamma']:.4e}  "
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

        # Share Stage A across all subsequent variants.
        # In analytical mode, every variant uses the closed-form BS formula
        # independently — no checkpoint to share.
        if not args.analytical_stage_a and load_etcnn_a_path is None:
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
                "rel_l2_gamma":  float(metrics.get("rel_l2_gamma", float("nan"))),
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
