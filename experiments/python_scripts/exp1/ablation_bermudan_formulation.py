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
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

VARIANTS: list[dict] = [
    {
        "name":           "hard_etcnn",
        "label":          "Hard BC (ETCNN ansatz)",
        "tc_type":        "hard",
        "lambda_tc_soft": None,
        "color":          "tab:blue",
        "linestyle":      "-",
        "linewidth":      2.5,
    },
    {
        "name":           "soft_pinn_lam10",
        "label":          r"Soft BC (PINN, $\lambda_{tc}=10$)",
        "tc_type":        "soft",
        "lambda_tc_soft": 10.0,
        "color":          "tab:orange",
        "linestyle":      "--",
        "linewidth":      2.0,
    },
    {
        "name":           "soft_pinn_lam100",
        "label":          r"Soft BC (PINN, $\lambda_{tc}=100$)",
        "tc_type":        "soft",
        "lambda_tc_soft": 100.0,
        "color":          "tab:green",
        "linestyle":      "-.",
        "linewidth":      2.0,
    },
    {
        "name":           "soft_pinn_lam1000",
        "label":          r"Soft BC (PINN, $\lambda_{tc}=1000$)",
        "tc_type":        "soft",
        "lambda_tc_soft": 1000.0,
        "color":          "tab:red",
        "linestyle":      ":",
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
# Load Stage A and compute Bermudean TC target at t=t1
# ---------------------------------------------------------------------------

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

    # --- Delta comparison ---------------------------------------------------
    try:
        s_d = torch.tensor(
            s_eval_arr, dtype=torch.get_default_dtype(), device=device
        ).requires_grad_(True)
        t_d = torch.zeros(len(s_eval_arr), device=device, requires_grad=True)
        x_d = torch.stack([s_d, t_d], dim=1)
        V_d = model(x_d).squeeze()
        (nn_delta,) = torch.autograd.grad(V_d.sum(), s_d, create_graph=False)
        nn_delta_np  = nn_delta.detach().cpu().numpy()
        bt_delta_np  = np.gradient(bt_prices, s_eval_arr)
        delta_err    = nn_delta_np - bt_delta_np
        rel_l2_delta = float(
            np.linalg.norm(delta_err) / (np.linalg.norm(bt_delta_np) + 1e-10)
        )
    except Exception as exc:
        logger.warning(f"compute_metrics_stage_b: Delta failed ({exc})")
        rel_l2_delta = float("nan")

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

    return {
        "rel_l2_bt":     rel_l2_bt,
        "rel_l2_atm":    rel_l2_atm,
        "rel_l2_delta":  rel_l2_delta,
        "gei":           gei,
        "tc_mae":        tc_mae,
        "pde_residual_t": {
            "t":        t_profile_tensor.cpu().tolist(),
            "residual": pde_residuals,
        },
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
        np.savez_compressed(
            vdir / "metrics.npz",
            rel_l2_bt=np.array([metrics["rel_l2_bt"]]),
            rel_l2_atm=np.array([metrics["rel_l2_atm"]]),
            rel_l2_delta=np.array([metrics["rel_l2_delta"]]),
            gei=np.array([metrics["gei"]]),
            tc_mae=np.array([metrics["tc_mae"]]),
            pde_t=np.array(metrics["pde_residual_t"]["t"]),
            pde_residual=np.array(metrics["pde_residual_t"]["residual"]),
        )


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
            "gei":           float(m["gei"][0]),
            "tc_mae":        float(m["tc_mae"][0]) if "tc_mae" in m else float("nan"),
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

def _plot_comparison(results: list[dict], ablation_dir: Path, iters_b: int) -> None:
    comp_dir = ablation_dir / "comparison"
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
    # Plot 8 — Rich summary metrics bar chart
    # ------------------------------------------------------------------
    n_variants = len(results)
    x          = np.arange(n_variants)
    vnames     = [r.get("name", f"v{i}") for i, r in enumerate(results)]

    if any(has_metrics):
        metric_keys   = ["mae_bt", "rel_l2_bt", "rel_l2_atm", "rel_l2_delta", "gei"]
        metric_labels = [
            r"MAE vs $V^{\mathrm{BT}}$",
            r"$\varepsilon_{L^2}$ (global)",
            r"$\varepsilon_{L^2}^{\mathrm{ATM}}$",
            r"$\varepsilon_{\Delta}$",
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

def _replot(ablation_dir: Path) -> None:
    """Regenerate all plots from a saved run directory (no retraining)."""
    metadata_path = ablation_dir / "metadata.yaml"
    summary_path  = ablation_dir / "summary.yaml"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.yaml not found in {ablation_dir}")
    with open(metadata_path) as f:
        metadata = yaml.safe_load(f)
    with open(summary_path) as f:
        summary = yaml.safe_load(f)

    iters_b       = metadata.get("iters_b", 0)
    variants_meta = metadata.get("variants", [])

    results = []
    for idx, v_meta in enumerate(variants_meta):
        vname = v_meta["name"]
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
    _plot_comparison(results, ablation_dir, iters_b)
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
        "--replot", type=str, default=None, metavar="DIR",
        help="Regenerate all plots from an existing run directory (no retraining).",
    )
    args = parser.parse_args()

    if args.replot is not None:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
            handlers=[logging.StreamHandler()],
        )
        logging.getLogger("matplotlib.mathtext").setLevel(logging.WARNING)
        _replot(Path(args.replot))
        return

    p3._apply_device_arg(args.device)
    if args.n_tc is not None:
        p3.N_TC = args.n_tc
    if args.n_f is not None:
        p3.N_F = args.n_f

    timestamp    = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    ablation_dir = (
        Path("data/ablation_bermudan_formulation")
        / f"{timestamp}_itersA{args.iters_a}_itersB{args.iters_b}"
    )
    ablation_dir.mkdir(parents=True, exist_ok=True)
    (ablation_dir / "comparison").mkdir(exist_ok=True)
    for v in VARIANTS:
        vdir = ablation_dir / f"variant_{v['name']}"
        for sub in ("training_metrics", "models"):
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
    logger.info(f"  variants: {[v['name'] for v in VARIANTS]}")
    logger.info(f"  output:   {ablation_dir}")
    logger.info(f"  log:      {log_path}")

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
                for var in VARIANTS
            ],
            "iters_a":      args.iters_a,
            "iters_b":      args.iters_b,
            "weight_decay": args.weight_decay,
            "N_TC":         p3.N_TC,
            "N_F":          p3.N_F,
            "LAMBDA_F":     p3.LAMBDA_F,
            "SEED":         p3.SEED,
        }, f, default_flow_style=False, sort_keys=False, width=float("inf"))

    results:         list[dict] = []
    load_etcnn_a_path: Path | None = None
    v_target_fn = None

    if args.load_stage_a is not None:
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

    t_ablation_start = time.time()

    for idx, variant in enumerate(VARIANTS):
        vname    = variant["name"]
        vdir     = ablation_dir / f"variant_{vname}"
        tc_type  = variant["tc_type"]

        logger.info("")
        logger.info("=" * 70)
        logger.info(f"VARIANT {idx + 1}/{len(VARIANTS)}: {vname}  (tc_type={tc_type})")
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

            # Extract etcnn_a path for subsequent soft variants
            if load_etcnn_a_path is None:
                load_etcnn_a_path = vdir / "models" / "etcnn_a.pt"
                logger.info(f"  Stage A saved — will be shared with soft variants: {load_etcnn_a_path}")
                v_target_fn, _ = _load_etcnn_a_and_build_vtarget(load_etcnn_a_path)

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
    summary: dict = {}
    for v, res in zip(VARIANTS, results):
        m = res.get("metrics", {}) or {}
        summary[v["name"]] = {
            "mae_bt":       float(res["mae_bt"]),
            "rel_l2_bt":    float(res["rel_l2_bt"]),
            "rel_l2_atm":   float(m.get("rel_l2_atm",   float("nan"))),
            "rel_l2_delta": float(m.get("rel_l2_delta", float("nan"))),
            "gei":          float(m.get("gei",          float("nan"))),
            "tc_mae":       float(m.get("tc_mae",       float("nan"))),
        }
    with open(ablation_dir / "summary.yaml", "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False,
                  width=float("inf"))

    # ------------------------------------------------------------------
    # Comparison plots
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("Generating comparison plots ...")
    _plot_comparison(results, ablation_dir, args.iters_b)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 70)
    logger.info("ABLATION SUMMARY — TC enforcement method")
    logger.info("=" * 70)
    logger.info(
        f"  Total wall-clock time: {total_elapsed:.1f}s  "
        f"({total_elapsed/len(VARIANTS):.1f}s per variant)"
    )
    logger.info(f"  {'Variant':<25} {'MAE':>12} {'rel_L2':>12} {'TC_MAE':>12} {'GEI':>8}")
    logger.info("  " + "-" * 72)
    for v, res in zip(VARIANTS, results):
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
