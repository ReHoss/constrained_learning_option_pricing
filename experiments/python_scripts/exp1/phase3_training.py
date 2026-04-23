"""Phase 3 — ETCNN training and validation on European & Bermudan put options.

Problem 1: European put (analytical reference).
Problem 2: Bermudan put with one intermediate exercise date t1=0.5
           (binomial tree reference, two-stage backward solving).

Parameters: K=100, r=0.02, sigma=0.25, T=1, q=0 (Section 4.1.2).

Usage:
    python3 experiments/python_scripts/exp1/phase3_training.py [--iters 50000] [--device auto|cuda|cpu]

    Default device is ``auto``: use CUDA when ``torch.cuda.is_available()``, else CPU.
    For GPU training, install PyTorch with CUDA (see https://pytorch.org/get-started/locally/),
    then run with ``--device cuda`` or rely on ``auto``.
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
import yaml
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from learning_option_pricing.models.etcnn import (
    AnalyticalEuropeanPut,
    BermudaETCNN,
    ETCNN,
    AmericanPutETCNN,
    InputNormalization,
    PINN,
)
from learning_option_pricing.models.resnet import ResNet
from learning_option_pricing.pricing.interpolation import (
    CubicSplineInterpolator,
    PchipInterpolator,
    PiecewiseLinearInterpolator,
)
from learning_option_pricing.pricing.singularity import (
    build_singularity_extraction,
)
from learning_option_pricing.pricing.terminal import (
    black_scholes_put,
    bsm_operator,
    g1_linear,
    g2_american_put,
    payoff_put,
)
from learning_option_pricing.solvers.binomial_tree import (
    bermuda_put_binomial_tree,
    european_put_binomial_tree,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameters (Section 4.1.2)
# ---------------------------------------------------------------------------
K = 100.0
r = 0.02
sigma = 0.25
T = 1.0
q = 0.0

# Training domain
S_TRAIN_LO, S_TRAIN_HI = 20.0, 160.0
# Evaluation domain
S_EVAL_LO, S_EVAL_HI = 60.0, 120.0

# Hyperparameters
M, L_BLOCK, n = 4, 2, 50
N_TC = 1024
N_F = 4 * N_TC
LAMBDA_F = 20.0
LAMBDA_TC = 1.0
SEED = 42

# Bermudan
t1 = 0.5  # intermediate exercise date

# Device (overridden in main() via --device)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _apply_device_arg(device_arg: str) -> None:
    """Set global DEVICE from CLI: auto, cuda, or cpu."""
    global DEVICE
    if device_arg == "auto":
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_arg == "cuda":
        if not torch.cuda.is_available():
            print(
                "ERROR: --device cuda but torch.cuda.is_available() is False.\n"
                "Install a CUDA-enabled PyTorch wheel, e.g.:\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cu126\n"
                "(Pick the CUDA version that matches your driver; see https://pytorch.org/get-started/locally/)",
                file=sys.stderr,
            )
            sys.exit(2)
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")


def to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Greeks Computation
# ---------------------------------------------------------------------------
def compute_greeks_nn(model: torch.nn.Module, s: torch.Tensor, t: torch.Tensor):
    """Compute Delta, Gamma, and Theta for a given model using autograd."""
    s = s.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    x = torch.stack([s, t], dim=1)
    
    # Forward pass
    V = model(x).squeeze()
    
    # Delta = dV/ds
    delta = torch.autograd.grad(V, s, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    
    # Gamma = d(Delta)/ds — retain_graph so V→t path survives for theta
    gamma = torch.autograd.grad(delta, s, grad_outputs=torch.ones_like(delta),
                                create_graph=False, retain_graph=True)[0]
    
    # Theta = dV/dt
    theta = torch.autograd.grad(V, t, grad_outputs=torch.ones_like(V), create_graph=False)[0]
    
    return delta.detach(), gamma.detach(), theta.detach()


def compute_greeks_analytical(s: torch.Tensor, t: torch.Tensor, K: float, r: float, sigma: float, T: float):
    """Compute Delta, Gamma, and Theta for analytical Black-Scholes using autograd."""
    s = s.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    tau = T - t
    
    V = black_scholes_put(s, K, r, sigma, tau)
    
    delta = torch.autograd.grad(V, s, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    gamma = torch.autograd.grad(delta, s, grad_outputs=torch.ones_like(delta),
                                create_graph=False, retain_graph=True)[0]
    theta = torch.autograd.grad(V, t, grad_outputs=torch.ones_like(V), create_graph=False)[0]
    
    return delta.detach(), gamma.detach(), theta.detach()


# ---------------------------------------------------------------------------
# LR scheduler matching the paper (Section 3.4)
# ---------------------------------------------------------------------------
def build_lr_lambda(total_iters: int):
    """Two-stage exponential decay: gamma=0.85 every 2000 steps for first 10k,
    then every 5000 steps thereafter."""
    gamma = 0.85

    def lr_lambda(step: int) -> float:
        decays = 0
        if step <= 10_000:
            decays = step // 2000
        else:
            decays = 10_000 // 2000  # = 5 decays in first 10k
            decays += (step - 10_000) // 5000
        return gamma ** decays

    return lr_lambda


# ---------------------------------------------------------------------------
# Collocation sampling
# ---------------------------------------------------------------------------
def sample_collocation(n_f: int, n_tc: int, s_lo: float, s_hi: float,
                       t_lo: float, t_hi: float):
    """Sample interior + terminal collocation points with requires_grad."""
    # Interior points
    s_f = (torch.rand(n_f) * (s_hi - s_lo) + s_lo).to(DEVICE)
    t_f = (torch.rand(n_f) * (t_hi - t_lo) + t_lo).to(DEVICE)
    s_f.requires_grad_(True)
    t_f.requires_grad_(True)

    # Terminal points
    s_tc = (torch.rand(n_tc) * (s_hi - s_lo) + s_lo).to(DEVICE)
    t_tc = torch.full((n_tc,), t_hi, device=DEVICE)

    return s_f, t_f, s_tc, t_tc


# ---------------------------------------------------------------------------
# Compute PDE + terminal losses
# ---------------------------------------------------------------------------
def compute_losses(
    model, s_f, t_f, s_tc, t_tc, payoff_fn, lam_f, lam_tc,
    s_star: float | None = None,
    sigma_w: float = 1.0,
    eps_w: float = 1e-3,
    use_spatial_weight: bool = False,
):
    """Evaluate L_f (PDE residual) and L_tc (terminal MSE).

    Args:
        s_star:             Exercise boundary coordinate.
        sigma_w:            Bandwidth of the inverted-Gaussian suppression window.
        eps_w:              Lower bound of the weight at s* (prevents nullification).
        use_spatial_weight: If True *and* s_star is finite, apply the inverted-Gaussian
                            spatial weighting to suppress the PCHIP knot spike:

                                w(s) = 1 - (1 - eps_w) * exp(-(s - s*)^2 / (2 sigma_w^2))

                            W is detached from the graph and acts as a static scaler.
                            Default False — plain mean(F_u^2) is used.
    """
    # PDE residual at interior points
    x_f = torch.stack([s_f, t_f], dim=1)

    # Operator Bypass: if the model has a specific forward pass for the PDE
    # (e.g. BermudaETCNN bypassing v(s,t) to avoid catastrophic cancellation), use it.
    if hasattr(model, "forward_pde"):
        u_f = model.forward_pde(x_f).squeeze()
    else:
        u_f = model(x_f).squeeze()

    F_u = bsm_operator(u_f, s_f, t_f, r, q, sigma)

    if use_spatial_weight and s_star is not None and not (isinstance(s_star, float) and s_star != s_star):
        # Inverted Gaussian: dips to eps_w at s*, approaches 1 far away.
        W = 1.0 - (1.0 - eps_w) * torch.exp(
            -((s_f - s_star) ** 2) / (2.0 * sigma_w ** 2)
        )
        W = W.detach()
        loss_f = torch.mean(W * F_u ** 2)
    else:
        loss_f = torch.mean(F_u ** 2)

    # Terminal condition
    x_tc = torch.stack([s_tc, t_tc], dim=1)
    with torch.no_grad():
        phi_tc = payoff_fn(s_tc)
    u_tc = model(x_tc).squeeze()
    loss_tc = torch.mean((u_tc - phi_tc) ** 2)

    total = lam_f * loss_f + lam_tc * loss_tc
    return total, loss_f.item(), loss_tc.item()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def _adaptive_log_every(total_iters: int, n_target: int = 50) -> int:
    """Return a round log period targeting ~n_target log points.

    Uses a 1-2-5 scale (like axis tick locators) so the period is always a
    human-readable number: 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, …
    """
    raw = max(1, total_iters / n_target)
    mag = 10 ** math.floor(math.log10(raw))
    for factor in (1, 2, 5, 10):
        candidate = int(factor * mag)
        if candidate >= raw:
            return candidate
    return int(10 * mag)  # fallback (unreachable in practice)


def train_model(
    model: torch.nn.Module,
    total_iters: int,
    s_lo: float,
    s_hi: float,
    t_lo: float,
    t_hi: float,
    payoff_fn,
    label: str = "model",
    log_every: int | None = None,
    weight_decay: float = 0.0,
    tc_enforced: bool = False,
    s_star: float | None = None,
    sigma_w: float = 1.0,
    eps_w: float = 1e-3,
    use_spatial_weight: bool = False,
):
    """Train a model with Adam + two-stage LR schedule.

    Args:
        tc_enforced: Set to True when the terminal condition is hard-enforced
            by the model ansatz (e.g. ETCNN with g1(s,T)=0). L_tc will then
            be labelled ``<enforced>`` in the log rather than a numeric value,
            to avoid the misleading impression that the optimiser drove it to zero.
    """
    if log_every is None:
        log_every = _adaptive_log_every(total_iters)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, build_lr_lambda(total_iters))

    history = {"loss": [], "loss_f": [], "loss_tc": [], "iter": [], "grad_norm": [], "lr": [], "tc_enforced": tc_enforced}
    model.train()

    t0 = time.time()
    for it in range(1, total_iters + 1):
        optimizer.zero_grad()
        s_f, t_f, s_tc, t_tc = sample_collocation(
            N_F, N_TC, s_lo, s_hi, t_lo, t_hi,
        )
        loss, lf, ltc = compute_losses(
            model, s_f, t_f, s_tc, t_tc, payoff_fn, LAMBDA_F, LAMBDA_TC,
            s_star=s_star, sigma_w=sigma_w, eps_w=eps_w,
            use_spatial_weight=use_spatial_weight,
        )
        loss.backward()
        
        # Calculate gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
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
            
            elapsed = time.time() - t0
            ltc_str = f"<enforced>({ltc:.6e})" if tc_enforced else f"{ltc:.6e}"
            logger.info(
                f"[{label}] iter {it:>6d}/{total_iters}  "
                f"loss={loss.item():.6e}  L_f={lf:.6e}  L_tc={ltc_str}  "
                f"|grad|={total_norm:.2e}  lr={lr_now:.6f}  ({elapsed:.1f}s)"
            )

    model.eval()
    elapsed = time.time() - t0
    logger.info(f"[{label}] Training done in {elapsed:.1f}s")
    return history


def plot_training_metrics(hist: dict, label: str, out_dir: Path):
    """Plot comprehensive training metrics including gradient norm and LR."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss curves
    tc_enforced = hist.get("tc_enforced", False)
    axes[0, 0].semilogy(hist["iter"], hist["loss_f"], label="$L_f$")
    if tc_enforced:
        # Ltc is identically zero (hard-enforced by the ansatz); plotting it on a
        # log scale would collapse the y-axis or hide Lf entirely.  Show a note
        # in the title instead.
        axes[0, 0].set_title("Loss Components\n($L_{tc}$ not shown: exact BC enforced by ansatz)")
    else:
        axes[0, 0].semilogy(hist["iter"], hist["loss_tc"], label="$L_{tc}$", color="tab:orange")
        axes[0, 0].set_title("Loss Components")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Total loss
    axes[0, 1].semilogy(hist["iter"], hist["loss"], label="Total Loss", color="tab:green")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Total Loss")
    axes[0, 1].set_title("Total Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gradient norm
    axes[1, 0].semilogy(hist["iter"], hist["grad_norm"], label="Gradient Norm", color="tab:red")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("L2 Norm")
    axes[1, 0].set_title("Gradient Magnitude")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1, 1].plot(hist["iter"], hist["lr"], label="Learning Rate", color="tab:purple")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("LR")
    axes[1, 1].set_title("Learning Rate Schedule")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    fig.suptitle(f"Training Metrics: {label}")
    fig.tight_layout()
    fig.savefig(out_dir / "training_metrics" / f"metrics_{label}.png", dpi=150)
    plt.close(fig)


# ===================================================================
#  EUROPEAN PROBLEM
# ===================================================================
def european_problem(
    out_dir: Path,
    total_iters: int,
    weight_decay: float = 0.0,
    g2_type: str = "taylor",
):
    """Train ETCNN and PINN on the European put, produce plots E1–E5.

    Args:
        out_dir: Output directory for plots.
        total_iters: Number of training iterations.
        weight_decay: L2 regularization penalty for Adam.
        g2_type: Terminal function type for ETCNN — ``"taylor"`` or ``"bs"``.
    """
    logger.info("=" * 70)
    logger.info("EUROPEAN PUT PROBLEM")
    logger.info("=" * 70)
    logger.info(f"  ETCNN g2 type: {g2_type}")

    # Config summary string for plot titles
    cfg_str = (
        f"$K={K},\\ r={r},\\ \\sigma={sigma},\\ T={T},\\ q={q}$"
        f"  |  $g_2$={g2_type}"
        f"  |  iters={total_iters}, $N_f$={N_F}, $N_{{tc}}$={N_TC}"
        f", wd={weight_decay}, seed={SEED}"
    )

    # --- Build models ---
    torch.manual_seed(SEED)
    etcnn = AmericanPutETCNN(K=K, r=r, sigma=sigma, T=T, normalize_input=True, g2_type=g2_type)

    torch.manual_seed(SEED)
    pinn = PINN(
        resnet=ResNet(d_in=2, d_out=1, n=n, M=M, L=L_BLOCK),
        normalizer=InputNormalization(K),
    )

    payoff_fn = lambda s: payoff_put(s, K)

    # --- Train ETCNN ---
    logger.info("Training ETCNN ...")
    hist_etcnn = train_model(
        etcnn, total_iters, S_TRAIN_LO, S_TRAIN_HI, 0.0, T,
        payoff_fn, label="ETCNN-Eur", weight_decay=weight_decay
    )

    # --- Train PINN ---
    logger.info("Training PINN ...")
    hist_pinn = train_model(
        pinn, total_iters, S_TRAIN_LO, S_TRAIN_HI, 0.0, T,
        payoff_fn, label="PINN-Eur", weight_decay=weight_decay
    )

    # --- Evaluation grids ---
    ns, nt = 200, 200
    s_vals = torch.linspace(S_EVAL_LO, S_EVAL_HI, ns)
    t_vals = torch.linspace(0.0, T, nt)
    S_grid, T_grid = torch.meshgrid(s_vals, t_vals, indexing="ij")
    tau_grid = T - T_grid
    Ve_grid = black_scholes_put(S_grid, K, r, sigma, tau_grid)

    x_eval = torch.stack([S_grid.reshape(-1), T_grid.reshape(-1)], dim=1).to(DEVICE)
    with torch.no_grad():
        u_etcnn = etcnn(x_eval).cpu().reshape(ns, nt)
        u_pinn = pinn(x_eval).cpu().reshape(ns, nt)

    err_etcnn = torch.abs(u_etcnn - Ve_grid)
    err_pinn = torch.abs(u_pinn - Ve_grid)

    # --- Metrics ---
    l2_etcnn = float(torch.sqrt(torch.mean((u_etcnn - Ve_grid) ** 2)))
    l2_pinn = float(torch.sqrt(torch.mean((u_pinn - Ve_grid) ** 2)))
    l2_ref = float(torch.sqrt(torch.mean(Ve_grid ** 2)))
    rel_l2_etcnn = l2_etcnn / l2_ref
    rel_l2_pinn = l2_pinn / l2_ref
    mae_etcnn = float(torch.mean(err_etcnn))
    mae_pinn = float(torch.mean(err_pinn))

    # === Plot E1 — Loss curves ===
    plot_training_metrics(hist_etcnn, "ETCNN-Eur", out_dir)
    plot_training_metrics(hist_pinn, "PINN-Eur", out_dir)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].semilogy(hist_etcnn["iter"], hist_etcnn["loss_f"], label="ETCNN $L_f$")
    axes[0].semilogy(hist_pinn["iter"], hist_pinn["loss_f"], label="PINN $L_f$", linestyle="--")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("$L_f$ (PDE residual)")
    axes[0].set_title("PDE residual loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(hist_etcnn["iter"], hist_etcnn["loss_tc"], label="ETCNN $L_{tc}$")
    axes[1].semilogy(hist_pinn["iter"], hist_pinn["loss_tc"], label="PINN $L_{tc}$", linestyle="--")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("$L_{tc}$ (terminal)")
    axes[1].set_title("Terminal condition loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Plot E1 — Loss curves (European Put, ETCNN vs PINN)\n{cfg_str}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "training_metrics" / "plotE1_loss_curves.png", dpi=150)
    plt.close(fig)
    logger.info("[OK] Plot E1 — Loss curves")

    # === Plot E2 — Predicted vs analytical surface ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    vmin = float(Ve_grid.min())
    vmax = float(Ve_grid.max())
    im0 = axes[0].pcolormesh(to_np(t_vals), to_np(s_vals), to_np(u_etcnn),
                              shading="auto", cmap="Blues", vmin=vmin, vmax=vmax)
    fig.colorbar(im0, ax=axes[0])
    axes[0].set_xlabel("t"); axes[0].set_ylabel("s")
    axes[0].set_title(r"ETCNN $\tilde{u}_{NN}(s,t)$")

    im1 = axes[1].pcolormesh(to_np(t_vals), to_np(s_vals), to_np(Ve_grid),
                              shading="auto", cmap="Blues", vmin=vmin, vmax=vmax)
    fig.colorbar(im1, ax=axes[1])
    axes[1].set_xlabel("t"); axes[1].set_ylabel("s")
    axes[1].set_title(r"Analytical $V^e(s,t)$")

    fig.suptitle(f"Plot E2 — ETCNN price surface vs analytical $V^e(s,t)$\n{cfg_str}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "pricing" / "plotE2_surface_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("[OK] Plot E2 — Surface comparison")

    # === Plot E3 — Pointwise errors: ETCNN vs PINN (2-panel comparison) ===
    err_max = max(float(err_etcnn.max()), float(err_pinn.max()))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im0 = axes[0].pcolormesh(to_np(t_vals), to_np(s_vals), to_np(err_etcnn),
                              shading="auto", cmap="hot_r", vmin=0, vmax=err_max)
    fig.colorbar(im0, ax=axes[0], label=r"$|\tilde{u}_\theta(s,t) - V^e(s,t)|$")
    axes[0].set_xlabel("t"); axes[0].set_ylabel("s")
    axes[0].set_title(f"ETCNN  (max={float(err_etcnn.max()):.2e})")

    im1 = axes[1].pcolormesh(to_np(t_vals), to_np(s_vals), to_np(err_pinn),
                              shading="auto", cmap="hot_r", vmin=0, vmax=err_max)
    fig.colorbar(im1, ax=axes[1], label=r"$|u_\theta(s,t) - V^e(s,t)|$")
    axes[1].set_xlabel("t"); axes[1].set_ylabel("s")
    axes[1].set_title(f"PINN  (max={float(err_pinn.max()):.2e})")

    fig.suptitle(
        f"Plot E3 — Absolute price error vs $V^e$  (ETCNN max={float(err_etcnn.max()):.2e}, PINN max={float(err_pinn.max()):.2e})\n{cfg_str}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "diagnostics" / "plotE3_errors.png", dpi=150)
    plt.close(fig)
    logger.info(f"[OK] Plot E3 — errors: ETCNN max={float(err_etcnn.max()):.2e}, PINN max={float(err_pinn.max()):.2e}")

    # === Plot E5 — Slice comparison at fixed t ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, t_fix in enumerate([0.25, 0.5, 0.75]):
        s_slice = torch.linspace(S_EVAL_LO, S_EVAL_HI, 300)
        t_slice = torch.full_like(s_slice, t_fix)
        x_slice = torch.stack([s_slice, t_slice], dim=1).to(DEVICE)
        tau_slice = T - t_slice

        with torch.no_grad():
            u_et = etcnn(x_slice).cpu().squeeze()
            u_pi = pinn(x_slice).cpu().squeeze()
        ve_slice = black_scholes_put(s_slice, K, r, sigma, tau_slice)

        axes[idx].plot(to_np(s_slice), to_np(ve_slice), label=r"$V^e$ (analytical)", linewidth=2)
        axes[idx].plot(to_np(s_slice), to_np(u_et), label="ETCNN", linewidth=2, linestyle="--")
        axes[idx].plot(to_np(s_slice), to_np(u_pi), label="PINN", linewidth=2, linestyle=":")
        axes[idx].set_xlabel("s")
        axes[idx].set_ylabel("Option value")
        axes[idx].set_title(f"$t = {t_fix}$")
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    fig.suptitle(f"Plot E5 — Slice comparison at fixed $t$  ($t \\in \\{{0.25, 0.5, 0.75\\}}$)\n{cfg_str}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "pricing" / "plotE5_slices.png", dpi=150)
    plt.close(fig)
    logger.info("[OK] Plot E5 — Slice comparison")

    # === Plot E6 — Greeks at t=0 ===
    s_greeks = torch.linspace(S_EVAL_LO, S_EVAL_HI, 300).to(DEVICE)
    t_greeks = torch.zeros_like(s_greeks).to(DEVICE)
    
    delta_etcnn, gamma_etcnn, theta_etcnn = compute_greeks_nn(etcnn, s_greeks, t_greeks)
    delta_pinn, gamma_pinn, theta_pinn = compute_greeks_nn(pinn, s_greeks, t_greeks)
    delta_true, gamma_true, theta_true = compute_greeks_analytical(s_greeks, t_greeks, K, r, sigma, T)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Delta
    axes[0].plot(to_np(s_greeks), to_np(delta_true), label="Analytical", linewidth=2)
    axes[0].plot(to_np(s_greeks), to_np(delta_etcnn), label="ETCNN", linewidth=2, linestyle="--")
    axes[0].plot(to_np(s_greeks), to_np(delta_pinn), label="PINN", linewidth=2, linestyle=":")
    axes[0].set_xlabel("s")
    axes[0].set_ylabel(r"$\Delta = \partial V / \partial s$")
    axes[0].set_title("Delta")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Gamma
    axes[1].plot(to_np(s_greeks), to_np(gamma_true), label="Analytical", linewidth=2)
    axes[1].plot(to_np(s_greeks), to_np(gamma_etcnn), label="ETCNN", linewidth=2, linestyle="--")
    axes[1].plot(to_np(s_greeks), to_np(gamma_pinn), label="PINN", linewidth=2, linestyle=":")
    axes[1].set_xlabel("s")
    axes[1].set_ylabel(r"$\Gamma = \partial^2 V / \partial s^2$")
    axes[1].set_title("Gamma")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Theta
    axes[2].plot(to_np(s_greeks), to_np(theta_true), label="Analytical", linewidth=2)
    axes[2].plot(to_np(s_greeks), to_np(theta_etcnn), label="ETCNN", linewidth=2, linestyle="--")
    axes[2].plot(to_np(s_greeks), to_np(theta_pinn), label="PINN", linewidth=2, linestyle=":")
    axes[2].set_xlabel("s")
    axes[2].set_ylabel(r"$\Theta = \partial V / \partial t$")
    axes[2].set_title("Theta")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    fig.suptitle(
        f"Plot E6 — Greeks $\\Delta, \\Gamma, \\Theta$ at $t = 0$  (ETCNN & PINN vs analytical)\n{cfg_str}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "greeks" / "plotE6_greeks.png", dpi=150)
    plt.close(fig)
    logger.info("[OK] Plot E6 — Greeks at t=0")

    # === Print E — Summary ===
    logger.info("")
    logger.info("=" * 60)
    logger.info("EUROPEAN PUT — SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  ETCNN relative L2 error: {rel_l2_etcnn:.6e}")
    logger.info(f"  ETCNN MAE:               {mae_etcnn:.6e}")
    logger.info(f"  PINN  relative L2 error: {rel_l2_pinn:.6e}")
    logger.info(f"  PINN  MAE:               {mae_pinn:.6e}")
    logger.info(f"  ETCNN/PINN L2 ratio:     {rel_l2_pinn / max(rel_l2_etcnn, 1e-15):.1f}x")
    logger.info("=" * 60)

    model_dir = out_dir / "models"
    torch.save(etcnn.state_dict(), model_dir / "etcnn_eur.pt")
    torch.save(pinn.state_dict(), model_dir / "pinn_eur.pt")
    logger.info("  Saved models: models/etcnn_eur.pt, models/pinn_eur.pt")

    return {
        "etcnn": etcnn,
        "pinn": pinn,
        "rel_l2_etcnn": rel_l2_etcnn,
        "rel_l2_pinn": rel_l2_pinn,
        "mae_etcnn": mae_etcnn,
        "mae_pinn": mae_pinn,
    }


# ===================================================================
#  Interpolation diagnostic (Plot B1b)
# ===================================================================

def _plot_interp_diagnostic(
    interp_cubic: CubicSplineInterpolator,
    interp_linear: PiecewiseLinearInterpolator,
    s_star: float,
    K: float,
    r: float,
    sigma: float,
    out_dir: Path,
    cfg_str: str = "",
) -> None:
    """Plot B1b — Compare C² (cubic) vs C⁰ (linear) interpolation of V(s, t1).

    Produces a 2×2 figure:
      (a) g2(s) for both interpolants (visually identical),
      (b) dg2/ds via autograd,
      (c) d²g2/ds² via autograd (key diagnostic — zero for linear),
      (d) F(g2) = r·s·dg2/ds − r·g2 + ½σ²s²·d²g2/ds² applied to g2 alone.
    """
    s_plot = torch.linspace(60.0, 140.0, 800)

    def _derivatives(interp_fn, s):
        s_ad = s.clone().requires_grad_(True)
        v = interp_fn(s_ad)
        dv = torch.autograd.grad(
            v, s_ad, grad_outputs=torch.ones_like(v), create_graph=True,
        )[0]
        if dv.grad_fn is not None:
            d2v = torch.autograd.grad(
                dv, s_ad, grad_outputs=torch.ones_like(dv),
            )[0]
        else:
            # C⁰ interpolant: first derivative is piecewise-constant with no
            # computational graph, so d²v/ds² is zero everywhere.
            d2v = torch.zeros_like(dv)
        return v.detach(), dv.detach(), d2v.detach()

    v_c, dv_c, d2v_c = _derivatives(interp_cubic, s_plot)
    v_l, dv_l, d2v_l = _derivatives(interp_linear, s_plot)

    s_np = to_np(s_plot)

    # F(g2) = ∂g2/∂t + ½σ²s²·∂²g2/∂s² + r·s·∂g2/∂s − r·g2
    # Since g2 is constant in t, ∂g2/∂t = 0.
    def _bsm_on_g2(v, dv, d2v, s):
        return 0.5 * sigma**2 * s**2 * d2v + r * s * dv - r * v

    F_c = to_np(_bsm_on_g2(v_c, dv_c, d2v_c, s_plot))
    F_l = to_np(_bsm_on_g2(v_l, dv_l, d2v_l, s_plot))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) g2 values
    ax = axes[0, 0]
    ax.plot(s_np, to_np(v_c), label="Cubic ($C^2$)", linewidth=2)
    ax.plot(s_np, to_np(v_l), label="Linear ($C^0$)", linewidth=2, linestyle="--")
    if not np.isnan(s_star):
        ax.axvline(s_star, color="red", ls=":", alpha=0.5)
    ax.set_xlabel("$s$")
    ax.set_ylabel("$V^{\\mathrm{Berm}}_\\theta(s, t_1)$")
    ax.set_title(r"(a)  $V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_1)$  [cubic vs linear]")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) first derivative
    ax = axes[0, 1]
    ax.plot(s_np, to_np(dv_c), label="Cubic ($C^2$)", linewidth=2)
    ax.plot(s_np, to_np(dv_l), label="Linear ($C^0$)", linewidth=2, linestyle="--")
    if not np.isnan(s_star):
        ax.axvline(s_star, color="red", ls=":", alpha=0.5)
    ax.set_xlabel("$s$")
    ax.set_ylabel("$\\partial V^{\\mathrm{Berm}}_\\theta(s, t_1) / \\partial s$")
    ax.set_title("(b)  First derivative $\\partial V^{\\mathrm{Berm}}_\\theta(s, t_1) / \\partial s$")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (c) second derivative — the key diagnostic
    ax = axes[1, 0]
    ax.plot(s_np, to_np(d2v_c), label="Cubic ($C^2$)", linewidth=2)
    ax.plot(s_np, to_np(d2v_l), label="Linear ($C^0$): $\\equiv 0$ a.e.", linewidth=2, linestyle="--")
    if not np.isnan(s_star):
        ax.axvline(s_star, color="red", ls=":", alpha=0.5)
    ax.set_xlabel("$s$")
    ax.set_ylabel("$\\partial^2 V^{\\mathrm{Berm}}_\\theta(s, t_1) / \\partial s^2$")
    ax.set_title("(c)  Second derivative $\\partial^2 V^{\\mathrm{Berm}}_\\theta(s, t_1) / \\partial s^2$  [KEY]")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (d) F(g2) — PDE operator on g2 alone
    ax = axes[1, 1]
    ax.plot(s_np, F_c, label="Cubic ($C^2$)", linewidth=2)
    ax.plot(s_np, F_l, label="Linear ($C^0$)", linewidth=2, linestyle="--")
    if not np.isnan(s_star):
        ax.axvline(s_star, color="red", ls=":", alpha=0.5)
    ax.set_xlabel("$s$")
    ax.set_ylabel("$\\mathcal{F}[V^{\\mathrm{Berm}}_\\theta(\\cdot, t_1)]$")
    ax.set_title(
        "(d)  $\\mathcal{F}[V^{\\mathrm{Berm}}_\\theta(\\cdot, t_1)]"
        " = rs\\,\\partial_s V^{\\mathrm{Berm}}_\\theta - r V^{\\mathrm{Berm}}_\\theta"
        " + \\frac{1}{2}\\sigma^2 s^2\\,\\partial_s^2 V^{\\mathrm{Berm}}_\\theta$"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    cfg_line = f"\n{cfg_str}" if cfg_str else ""
    fig.suptitle(
        "Plot B1b — Interpolation diagnostic: cubic ($C^2$) vs linear ($C^0$) vs PCHIP ($C^1$)\n"
        f"Bermudan Put, K={K}, r={r}, $\\sigma$={sigma}{cfg_line}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "diagnostics" / "plotB1b_interp_diagnostic.png", dpi=150)
    plt.close(fig)
    logger.info("[OK] Plot B1b — Interpolation diagnostic (C² vs C⁰)")


# ===================================================================
#  BERMUDAN PROBLEM (two-stage backward solving)
# ===================================================================
def bermudan_problem(
    out_dir: Path,
    total_iters: list[int],
    interp_method: str = "cubic",
    put_ansatz: bool = False,
    weight_decay: float = 0.0,
    load_etcnn_a: Path | None = None,
    g2_type: str = "taylor",
    bypass_v: bool = False,
    sigma_w: float = 1.0,
    eps_w: float = 1e-3,
    use_spatial_weight: bool = False,
    g2_gamma: float | None = None,
    load_etcnn_b: Path | None = None,
    analytic_a: bool = False,
):
    """Two-stage Bermudan put with exercise date t1=0.5.

    Args:
        out_dir: Output directory for plots and logs.
        total_iters: List of training iterations per stage [Stage A, Stage B].
        interp_method: Interpolation for V(s, t1) when *put_ansatz* is False.
            ``"cubic"`` (default) uses a C^2 natural cubic spline;
            ``"pchip"`` uses a C^1 PCHIP interpolant;
            ``"linear"`` uses the original C^0 piecewise-linear interpolant.
        put_ansatz: If True, use the singularity extraction ansatz to
            decompose U_B = v + u_tilde, removing the C^0 kink at s*.
            Default False (standard interpolation approach).
        weight_decay: L2 regularization penalty for Adam.
        load_etcnn_a: Path to pre-trained ETCNN_A model to skip Stage A training.
        g2_type: Terminal function type for Stage A — ``"taylor"`` or ``"bs"``.
        bypass_v: If True, drops only the fictitious put v(s,t) from the PDE
            loss to prevent catastrophic cancellation of its diverging derivatives
            near t1.  g2 remains in the graph so the network corrects L(g2).
        sigma_w:            Bandwidth of the inverted-Gaussian spatial weight (default 1.0).
        eps_w:              Lower bound of the spatial weight at s* (default 1e-3).
        use_spatial_weight: If True, activate the inverted-Gaussian weighting in
                            Stage B.  Default False (plain MSE).
        g2_gamma: γ ≥ 0 for the temporal truncation h(t) = exp(-γ(t1-t)²)
            applied to the Stage B g2 field.  ``None`` (default) disables
            truncation (h ≡ 1, standard ETCNN behaviour).
        load_etcnn_b: Path to a pre-trained ``etcnn_b.pt`` (or a run directory
            containing ``models/etcnn_b.pt``) whose weights are loaded into the
            freshly-constructed Stage B model before training begins.  The model
            architecture is always rebuilt from the current flags so that
            ``g2_gamma``, ``bypass_v``, etc. are honoured; only the ResNet
            weights are warm-started.  ``iters_b`` additional gradient steps are
            then taken on top of the loaded checkpoint.  Default ``None``
            (random initialisation).
        analytic_a: If True, replace Stage A with the exact Black-Scholes
            European put formula (no network trained).  The intermediate
            terminal condition at t1 becomes max(Φ(s), V^e(s, t1)) exactly.
            Useful to isolate Stage B from Stage A approximation errors.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("BERMUDAN PUT PROBLEM (t1=0.5)")
    logger.info("=" * 70)

    iters_a = total_iters[0]
    iters_b = total_iters[1] if len(total_iters) > 1 else total_iters[0]

    # Config summary string for plot titles — encodes all run-specific settings
    _flags = []
    if put_ansatz:
        _flags.append("put-ansatz")
    if bypass_v:
        _flags.append("bypass\\_v")
    if g2_gamma is not None:
        _flags.append(f"$h(t),\\gamma={g2_gamma}$")
    _flags_str = (", " + ", ".join(_flags)) if _flags else ""
    cfg_str = (
        f"$K={K},\\ r={r},\\ \\sigma={sigma},\\ T={T},\\ q={q},\\ t_1={t1}$"
        f"  |  $g_2$={g2_type}, interp={interp_method}{_flags_str}"
        f"  |  iters A={iters_a}/B={iters_b}, $N_f$={N_F}, $N_{{tc}}$={N_TC}"
        f", wd={weight_decay}, seed={SEED}"
    )

    payoff_fn = lambda s: payoff_put(s, K)

    # ---------------------------------------------------------------
    # Stage A — train ETCNN on [t1, T]  (or use analytical BS)
    # ---------------------------------------------------------------
    torch.manual_seed(SEED)
    hist_a = None

    if analytic_a:
        logger.info("Stage A: using analytical Black-Scholes European put (no training)")
        etcnn_a = AnalyticalEuropeanPut(K=K, r=r, sigma=sigma, T=T)
        etcnn_a.to(DEVICE)
        etcnn_a.eval()
    else:
        etcnn_a = AmericanPutETCNN(K=K, r=r, sigma=sigma, T=T, normalize_input=True, g2_type=g2_type)
        etcnn_a.to(DEVICE)
        logger.info(f"  Stage A g2 type: {g2_type}")

        if load_etcnn_a is not None and load_etcnn_a.exists():
            logger.info(f"Stage A: Loading pre-trained ETCNN_A from {load_etcnn_a} ...")
            etcnn_a.load_state_dict(torch.load(load_etcnn_a, map_location=DEVICE))
            etcnn_a.eval()
        else:
            logger.info(f"Stage A: training ETCNN_A on [t1, T] for {iters_a} iterations ...")
            hist_a = train_model(
                etcnn_a, iters_a, S_TRAIN_LO, S_TRAIN_HI, t1, T,
                payoff_fn, label="ETCNN_A", weight_decay=weight_decay, tc_enforced=True,
            )
            plot_training_metrics(hist_a, "ETCNN_A", out_dir)

    # ---------------------------------------------------------------
    # Stage B — construct intermediate terminal condition at t1
    # ---------------------------------------------------------------
    logger.info("Stage B: constructing V(s, t1) at t1 ...")
    s_dense = torch.linspace(S_TRAIN_LO - 10, S_TRAIN_HI + 10, 2000)
    t1_dense = torch.full_like(s_dense, t1)
    x_t1 = torch.stack([s_dense, t1_dense], dim=1).to(DEVICE)

    with torch.no_grad():
        hold_val = etcnn_a(x_t1).cpu().squeeze()
    exercise_val = payoff_put(s_dense, K)
    v_t1_vals = torch.maximum(exercise_val, hold_val)

    # Find exercise boundary s*: where Phi(s) = hold(s)
    diff_ex = exercise_val - hold_val
    sign_changes = torch.where(diff_ex[:-1] * diff_ex[1:] < 0)[0]
    if len(sign_changes) > 0:
        idx = sign_changes[0].item()
        s_star = float(s_dense[idx])
    else:
        s_star = float("nan")
    logger.info(f"  Exercise boundary at t1: s* ≈ {s_star:.2f}")

    s_nodes_cpu = s_dense.detach().cpu()
    v_nodes_cpu = v_t1_vals.detach().cpu()

    # Build interpolants of V_target (for diagnostics and non-extraction path)
    interp_cubic = CubicSplineInterpolator(s_nodes_cpu, v_nodes_cpu)
    interp_linear = PiecewiseLinearInterpolator(s_nodes_cpu, v_nodes_cpu)
    interp_pchip = PchipInterpolator(s_nodes_cpu, v_nodes_cpu)

    c_scale = float("nan")  # only set when extraction=True

    if put_ansatz:
        # -----------------------------------------------------------
        # Singularity extraction ansatz
        # -----------------------------------------------------------
        logger.info("  Mode: singularity extraction ansatz")
        s_star_ext, c_scale, fict_put, s_nodes_ext, v_target_ext, residual_cpu = \
            build_singularity_extraction(
                etcnn_a, K, r, sigma, t1,
                s_lo=S_TRAIN_LO - 10, s_hi=S_TRAIN_HI + 10,
                device=DEVICE, n_grid=2000,
            )
        s_star = s_star_ext  # refined via bisection

        # PCHIP for V_target (TC loss) and C^1 residual (g2)
        interp_vtarget = PchipInterpolator(s_nodes_ext, v_target_ext)
        residual_interp = PchipInterpolator(s_nodes_ext, residual_cpu)
        logger.info("  Using C^1 PCHIP interpolation for the extracted residual g2")
    else:
        # -----------------------------------------------------------
        # Classic interpolation of V_target (no extraction)
        # -----------------------------------------------------------
        logger.info(f"  Mode: direct interpolation ({interp_method})")
        if interp_method == "cubic":
            v_interp_t1 = interp_cubic
            logger.info("  Using C^2 cubic spline interpolation for V(s, t1)")
        elif interp_method == "pchip":
            v_interp_t1 = interp_pchip
            logger.info("  Using C^1 PCHIP (shape-preserving) interpolation for V(s, t1)")
        elif interp_method == "linear":
            v_interp_t1 = interp_linear
            logger.info("  Using C^0 piecewise-linear interpolation for V(s, t1)")
        else:
            raise ValueError(f"Unknown interp_method: {interp_method!r}")

    # === Plot B1b — Interpolation diagnostic: derivatives of g2 ===
    _plot_interp_diagnostic(
        interp_cubic, interp_linear, s_star, K, r, sigma, out_dir, cfg_str,
    )

    # === Plot B1 — Intermediate terminal condition ===
    s_plot = torch.linspace(60.0, 140.0, 500)
    t1_plot = torch.full_like(s_plot, t1)
    x_plot = torch.stack([s_plot, t1_plot], dim=1).to(DEVICE)
    with torch.no_grad():
        hold_plot = etcnn_a(x_plot).cpu().squeeze()
    phi_plot = payoff_put(s_plot, K)
    v_t1_plot = torch.maximum(phi_plot, hold_plot)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(to_np(s_plot), to_np(hold_plot), label=r"Hold: ETCNN$_A(s, t_1)$", linewidth=2)
    ax.plot(to_np(s_plot), to_np(phi_plot), label=r"Exercise: $\Phi(s) = (K-s)^+$", linewidth=2, linestyle="--")
    ax.plot(to_np(s_plot), to_np(v_t1_plot), label=r"$V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_1) = \max(\Phi(s),\, \tilde{u}^{(A)}_{\bar{\theta}}(s, t_1))$",
            linewidth=2.5, linestyle="-.", color="black")
    if not np.isnan(s_star):
        ax.axvline(s_star, color="red", linestyle=":", alpha=0.7, label=f"$s^* \\approx {s_star:.1f}$")
        s_star_t = torch.tensor([s_star])
        v_at_star = float(payoff_put(s_star_t, K))
        ax.plot(s_star, v_at_star, "ro", markersize=8, zorder=5)
        ax.annotate(f"$s^* = {s_star:.1f}$", (s_star, v_at_star),
                    textcoords="offset points", xytext=(15, 10), fontsize=11,
                    arrowprops=dict(arrowstyle="->", color="red"))
    ax.set_xlabel("s")
    ax.set_ylabel("Value")
    ax.set_title(f"Hold vs exercise vs Bermudan value at $t_1 = {t1}$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.suptitle(f"Plot B1 — Intermediate terminal condition\n{cfg_str}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "pricing" / "plotB1a_intermediate.png", dpi=150)
    plt.close(fig)
    logger.info("[OK] Plot B1a — Intermediate terminal condition (hold, exercise, Bermudan value, s*)")

    # === Plot B1c / B1d — mode-specific diagnostics ===
    if put_ansatz:
        # B1c — Extracted residual g2 at t1
        s_plot_1d = torch.linspace(60.0, 140.0, 500)
        with torch.no_grad():
            g2_residual_plot = residual_interp(s_plot_1d)
            v_fict_plot = fict_put.at_maturity(s_plot_1d)
            vtarget_plot = interp_vtarget(s_plot_1d)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(to_np(s_plot_1d), to_np(vtarget_plot),
                label=r"$V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_1)$", color="black", linewidth=2.5, linestyle="-.")
        ax.plot(to_np(s_plot_1d), to_np(v_fict_plot),
                label=f"$v(s, t_1) = {c_scale:.3f} \\cdot (s^* - s)^+$", color="green", linewidth=2, linestyle="--")
        ax.plot(to_np(s_plot_1d), to_np(g2_residual_plot),
                label=r"Residual $g_2(s, t_1) = V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_1) - v(s, t_1)$", color="blue", linewidth=2)
        ax.axvline(s_star, color="red", linestyle=":", alpha=0.7, label=f"$s^* \\approx {s_star:.1f}$")
        ax.set_title(
            f"Decomposition at $t_1 = {t1}$: "
            r"$V^{\mathrm{Berm}}_{\bar{\theta}}(s,t_1) = v(s,t_1) + g_2(s,t_1)$"
            f",  $c = {c_scale:.4f}$,  $s^* = {s_star:.2f}$"
        )
        ax.set_xlabel("Asset Price $s$")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.suptitle(f"Plot B1c — Singularity extraction at $t_1$\n{cfg_str}", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / "pricing" / "plotB1c_extraction.png", dpi=150)
        plt.close(fig)
        logger.info("[OK] Plot B1c — Singularity extraction at t1")

        # B1d — Curvature comparison: V_target vs residual
        s_wide = torch.linspace(s_star - 20.0, s_star + 20.0, 2000)
        s_fine = torch.linspace(s_star - 3.0, s_star + 3.0, 1000)
        h_fd = 1e-3

        def raw_target(s_tensor):
            t_tensor = torch.full_like(s_tensor, t1).to(DEVICE)
            x_tensor = torch.stack([s_tensor.to(DEVICE), t_tensor], dim=1)
            hold = etcnn_a(x_tensor).cpu().squeeze()
            phi = payoff_put(s_tensor, K)
            return torch.maximum(phi, hold)

        with torch.no_grad():
            vtarget_wide = raw_target(s_wide)
            vtarget_fine = raw_target(s_fine)
            v_fine = fict_put.at_maturity(s_fine)
            u_tilde_fine = residual_interp(s_fine)
            gamma_raw_wide = (raw_target(s_wide + h_fd) - 2 * raw_target(s_wide) + raw_target(s_wide - h_fd)) / h_fd**2
            gamma_raw_fine = (raw_target(s_fine + h_fd) - 2 * raw_target(s_fine) + raw_target(s_fine - h_fd)) / h_fd**2
            gamma_res_wide = (residual_interp(s_wide + h_fd) - 2 * residual_interp(s_wide) + residual_interp(s_wide - h_fd)) / h_fd**2
            gamma_res_fine = (residual_interp(s_fine + h_fd) - 2 * residual_interp(s_fine) + residual_interp(s_fine - h_fd)) / h_fd**2
            deriv_raw_fine = (raw_target(s_fine + h_fd) - raw_target(s_fine - h_fd)) / (2 * h_fd)
            deriv_res_fine = (residual_interp(s_fine + h_fd) - residual_interp(s_fine - h_fd)) / (2 * h_fd)

        fig, axes = plt.subplots(4, 1, figsize=(10, 17))

        axes[0].plot(to_np(s_fine), to_np(vtarget_fine),
                     label=r"$V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_1)$", color="black", linewidth=2)
        axes[0].plot(to_np(s_fine), to_np(v_fine),
                     label=r"$v(s, t_1) = c\,(s^* - s)^+$", color="green", linewidth=2, linestyle="--")
        axes[0].plot(to_np(s_fine), to_np(u_tilde_fine),
                     label=r"$g_2(s, t_1) = V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_1) - v(s, t_1)$ (PCHIP residual)", color="blue", linewidth=2)
        axes[0].axvline(s_star, color="grey", linestyle=":", alpha=0.5)
        axes[0].set_title(r"Function decomposition at $t_1$: $V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_1) = v(s, t_1) + g_2(s, t_1)$")
        axes[0].set_xlabel("$s$"); axes[0].set_ylabel("Value")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(to_np(s_fine), to_np(deriv_raw_fine),
                     label=r"$\partial V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_1) / \partial s$ (has jump at $s^*$)", color="red", linewidth=2)
        axes[1].plot(to_np(s_fine), to_np(deriv_res_fine),
                     label=r"$\partial g_2(s, t_1) / \partial s$ ($C^1$ smooth)", color="blue", linewidth=2, linestyle="--")
        axes[1].axvline(s_star, color="grey", linestyle=":", alpha=0.5)
        axes[1].set_title(r"First derivative: jump in $\partial_s V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_1)$ removed in $\partial_s g_2(s, t_1)$")
        axes[1].set_xlabel("$s$"); axes[1].set_ylabel("$\\partial / \\partial s$")
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

        axes[2].plot(to_np(s_wide), to_np(gamma_raw_wide),
                     label=r"$\partial^2 V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_1) / \partial s^2$ (Dirac spike at $s^*$)", color="red", linewidth=2)
        axes[2].plot(to_np(s_wide), to_np(gamma_res_wide),
                     label=r"$\partial^2 g_2(s, t_1) / \partial s^2$ (PCHIP, finite)", color="blue", linewidth=2, linestyle="--")
        axes[2].axvline(s_star, color="grey", linestyle=":", alpha=0.5)
        axes[2].set_title(r"Second derivative: Dirac $\delta$ singularity removed (wide view)")
        axes[2].set_xlabel("$s$"); axes[2].set_ylabel("$\\partial^2 / \\partial s^2$")
        axes[2].set_ylim(-20, 20); axes[2].legend(); axes[2].grid(True, alpha=0.3)

        axes[3].plot(to_np(s_fine), to_np(gamma_raw_fine),
                     label=r"$\partial^2 V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_1) / \partial s^2$", color="red", linewidth=2)
        axes[3].plot(to_np(s_fine), to_np(gamma_res_fine),
                     label=r"$\partial^2 g_2(s, t_1) / \partial s^2$ (PCHIP)", color="blue", linewidth=2, linestyle="--")
        axes[3].axvline(s_star, color="grey", linestyle=":", alpha=0.5)
        axes[3].set_title(r"Curvature near $s^*$ (zoomed)")
        axes[3].set_xlabel("$s$"); axes[3].set_ylabel("$\\partial^2 / \\partial s^2$")
        axes[3].legend(); axes[3].grid(True, alpha=0.3)

        fig.suptitle(
            f"Plot B1d — Singularity extraction curvature diagnostic"
            f":  $c = {c_scale:.4f}$,  $s^* = {s_star:.2f}$\n{cfg_str}",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(out_dir / "diagnostics" / "plotB1d_extraction_curvature.png", dpi=150)
        plt.close(fig)
        logger.info("[OK] Plot B1d — Singularity extraction curvature diagnostic")
    else:
        # B1c — Interpolated function at t1
        s_plot_1d = torch.linspace(60.0, 140.0, 500)
        with torch.no_grad():
            v_interp_plot = v_interp_t1(s_plot_1d)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(to_np(s_plot_1d), to_np(v_interp_plot), label=r"$V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_1)$ (interpolated)", color="blue", linewidth=2)
        ax.plot(to_np(s_plot_1d), to_np(payoff_put(s_plot_1d, K)), label="$\\Phi(s) = (K - s)^+$", color="red", linestyle="--")
        ax.axvline(x=K, color="grey", linestyle=":", label="Strike K")
        if not np.isnan(s_star):
            ax.axvline(s_star, color="red", linestyle=":", alpha=0.7, label=f"$s^* \\approx {s_star:.1f}$")
        ax.set_title(
            f"Interpolated $V^{{\\mathrm{{Berm}}}}_{{\\bar{{\\theta}}}}(s, t_1)$"
            f" at $t_1 = {t1}$,  $s^* = {s_star:.2f}$"
        )
        ax.set_xlabel("Asset Price $s$"); ax.set_ylabel("Option Value")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.suptitle(f"Plot B1c — Interpolated terminal condition at $t_1$\n{cfg_str}", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / "pricing" / "plotB1c_interpolated_t1.png", dpi=150)
        plt.close(fig)
        logger.info("[OK] Plot B1c — Interpolated function at t1")

        # B1d — Interpolant Curvature (Gamma)
        if not np.isnan(s_star):
            s_fine = torch.linspace(s_star - 1.0, 82.0, 1000)
            s_wide = torch.linspace(s_star - 20.0, s_star + 20.0, 2000)
        else:
            s_fine = torch.linspace(75.0, 85.0, 1000)
            s_wide = torch.linspace(60.0, 100.0, 2000)

        h_fd = 1e-3

        def raw_target(s_tensor):
            t_tensor = torch.full_like(s_tensor, t1).to(DEVICE)
            x_tensor = torch.stack([s_tensor.to(DEVICE), t_tensor], dim=1)
            hold = etcnn_a(x_tensor).cpu().squeeze()
            phi = payoff_put(s_tensor, K)
            return torch.maximum(phi, hold)

        with torch.no_grad():
            v_plus = v_interp_t1(s_fine + h_fd)
            v_center = v_interp_t1(s_fine)
            v_minus = v_interp_t1(s_fine - h_fd)
            gamma_spline = (v_plus - 2 * v_center + v_minus) / (h_fd ** 2)
            v_raw_plus = raw_target(s_fine + h_fd)
            v_raw_center = raw_target(s_fine)
            v_raw_minus = raw_target(s_fine - h_fd)
            gamma_raw = (v_raw_plus - 2 * v_raw_center + v_raw_minus) / (h_fd ** 2)
            v_wide_plus = v_interp_t1(s_wide + h_fd)
            v_wide_center = v_interp_t1(s_wide)
            v_wide_minus = v_interp_t1(s_wide - h_fd)
            gamma_spline_wide = (v_wide_plus - 2 * v_wide_center + v_wide_minus) / (h_fd ** 2)
            v_raw_wide_plus = raw_target(s_wide + h_fd)
            v_raw_wide_center = raw_target(s_wide)
            v_raw_wide_minus = raw_target(s_wide - h_fd)
            gamma_raw_wide = (v_raw_wide_plus - 2 * v_raw_wide_center + v_raw_wide_minus) / (h_fd ** 2)

        interp_name = "PCHIP" if interp_method == "pchip" else ("Spline" if interp_method == "cubic" else "Linear")
        fig, axes = plt.subplots(3, 1, figsize=(9, 12))
        axes[0].plot(to_np(s_wide), to_np(gamma_spline_wide), label=f"$\\partial^2 g_2(s, t_1) / \\partial s^2$ ({interp_name})", color="purple", linewidth=2)
        axes[0].plot(to_np(s_wide), to_np(gamma_raw_wide), label="$\\partial^2 V^{\\mathrm{Berm}}_\\theta(s, t_1) / \\partial s^2$", color="green", linewidth=2, linestyle="--")
        if not np.isnan(s_star):
            axes[0].axvline(s_star, color="red", linestyle=":", alpha=0.7, label=f"$s^* \\approx {s_star:.1f}$")
        axes[0].set_title(f"$\\partial^2 g_2(s, t_1) / \\partial s^2$ vs $\\partial^2 V^{{\\mathrm{{Berm}}}}_\\theta(s, t_1) / \\partial s^2$ — {interp_name} (wide)")
        axes[0].set_xlabel("$s$"); axes[0].set_ylabel("$\\partial^2 / \\partial s^2$")
        axes[0].set_ylim(-20, 20); axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(to_np(s_fine), to_np(gamma_spline), label=f"$\\partial^2 g_2(s, t_1) / \\partial s^2$ ({interp_name})", color="purple", linewidth=2)
        axes[1].plot(to_np(s_fine), to_np(gamma_raw), label="$\\partial^2 V^{\\mathrm{Berm}}_\\theta(s, t_1) / \\partial s^2$", color="green", linewidth=2, linestyle="--")
        if not np.isnan(s_star):
            axes[1].axvline(s_star, color="red", linestyle=":", alpha=0.7, label=f"$s^* \\approx {s_star:.1f}$")
        axes[1].set_title(f"$\\partial^2 g_2(s, t_1) / \\partial s^2$ vs $\\partial^2 V^{{\\mathrm{{Berm}}}}_\\theta(s, t_1) / \\partial s^2$ — {interp_name} (zoomed)")
        axes[1].set_xlabel("$s$"); axes[1].set_ylabel("$\\partial^2 / \\partial s^2$")
        axes[1].set_ylim(-20, 20); axes[1].legend(); axes[1].grid(True, alpha=0.3)

        axes[2].plot(to_np(s_fine), to_np(v_center), label=f"$V^{{\\mathrm{{Berm}}}}_\\theta(s, t_1)$ ({interp_name})", color="blue", linewidth=2)
        axes[2].plot(to_np(s_fine), to_np(v_raw_center), label="$V^{\\mathrm{Berm}}_\\theta(s, t_1) = \\max(\\Phi(s),\\, \\tilde{u}^{(A)}_\\theta(s, t_1))$", color="orange", linewidth=2, linestyle="--")
        if not np.isnan(s_star):
            axes[2].axvline(s_star, color="red", linestyle=":", alpha=0.7, label=f"$s^* \\approx {s_star:.1f}$")
        axes[2].set_title(f"$V^{{\\mathrm{{Berm}}}}_\\theta(s, t_1)$: interpolant vs raw target around $s^*$ (zoomed)")
        axes[2].set_xlabel("$s$"); axes[2].set_ylabel("Option Value")
        axes[2].legend(); axes[2].grid(True, alpha=0.3)
        fig.suptitle(
            f"Plot B1d — Curvature diagnostic ({interp_name}),  $s^* = {s_star:.2f}$\n{cfg_str}",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(out_dir / "diagnostics" / "plotB1d_interpolant_gamma.png", dpi=150)
        plt.close(fig)
        logger.info(f"[OK] Plot B1d — {interp_name} Curvature")

    # === Plot B3 — Stage A loss curves ===
    if hist_a is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].semilogy(hist_a["iter"], hist_a["loss_f"], label="$\\mathcal{L}_f$")
        axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("$\\mathcal{L}_f$")
        axes[0].set_title("Stage A: PDE residual"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[1].semilogy(hist_a["iter"], hist_a["loss_tc"], label="$\\mathcal{L}_{tc}$", color="tab:orange")
        axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("$\\mathcal{L}_{tc}$")
        axes[1].set_title("Stage A: Terminal loss"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
        fig.suptitle(
            f"Plot B3 — Stage A loss curves  (ETCNN$^{{(A)}}$ on $[t_1, T]$,  {iters_a} iters)\n{cfg_str}",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(out_dir / "training_metrics" / "plotB3_stageA_loss.png", dpi=150)
        plt.close(fig)
        logger.info("[OK] Plot B3 — Stage A loss curves")
    else:
        logger.info("[SKIP] Plot B3 — Stage A loss curves (model loaded from disk)")

    # ---------------------------------------------------------------
    # Stage C+D — train ETCNN_B on [0, t1]
    # ---------------------------------------------------------------

    def g1_b(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return t1 - t

    torch.manual_seed(SEED + 1)
    resnet_b = ResNet(d_in=2, d_out=1, n=n, M=M, L=L_BLOCK)
    normalizer_b = InputNormalization(K)

    if g2_gamma is not None:
        logger.info(
            f"Stage B temporal truncation: ACTIVE  h(t) = exp(-{g2_gamma}·(t1-t)²)"
            f"  h(0) = {math.exp(-g2_gamma * t1**2):.4f}"
        )
    else:
        logger.info("Stage B temporal truncation: OFF (h ≡ 1)")

    if put_ansatz:
        # Singularity extraction: U_B = v + ũ_θ
        logger.info(f"Stage C+D: training BermudaETCNN on [0, t1] for {iters_b} iterations ...")
        logger.info(f"  Ansatz: U_B(s,t) = v(s,t) + g1(s,t)·u_NN(s,t) + h(t)·g2(s)")
        logger.info(f"  v(s,t) = {c_scale:.4f} · P^BS(s, {s_star:.2f}, r, σ, t1-t)")

        def g2_b(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return residual_interp(s)

        etcnn_residual = ETCNN(
            resnet=resnet_b, g1=g1_b, g2=g2_b, normalizer=normalizer_b,
            g2_temporal_gamma=g2_gamma, t_terminal=t1,
        )
        fict_put.to(DEVICE)
        etcnn_b = BermudaETCNN(etcnn=etcnn_residual, fictitious_put=fict_put, bypass_v=bypass_v)
        payoff_t1 = lambda s: interp_vtarget(s)
    else:
        # Classic: g2 = interpolated V_target
        logger.info(f"Stage C+D: training ETCNN_B on [0, t1] for {iters_b} iterations ...")

        def g2_b(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return v_interp_t1(s)

        etcnn_b = ETCNN(
            resnet=resnet_b, g1=g1_b, g2=g2_b, normalizer=normalizer_b,
            g2_temporal_gamma=g2_gamma, t_terminal=t1,
        )
        payoff_t1 = lambda s: v_interp_t1(s)

    # === Plot B_h — temporal truncation profile h(t) ===
    if g2_gamma is not None:
        t_plot_h = torch.linspace(0.0, t1, 500)
        tau_plot = t1 - t_plot_h
        h_plot = torch.exp(-g2_gamma * tau_plot ** 2)

        fig_h, ax_h = plt.subplots(figsize=(7, 4))
        ax_h.plot(to_np(t_plot_h), to_np(h_plot), color="tab:blue", linewidth=2.5,
                  label=rf"$h(t) = \exp(-{g2_gamma}\,(t_1 - t)^2)$")
        ax_h.axvline(t1, color="tab:red", linestyle="--", linewidth=1.4, label=f"$t_1 = {t1}$")
        ax_h.axvline(0.0, color="grey", linestyle=":", linewidth=1.0)
        ax_h.annotate(
            f"$h(0) = {float(h_plot[0]):.4f}$",
            xy=(0.0, float(h_plot[0])),
            xytext=(t1 * 0.12, float(h_plot[0]) - 0.06),
            fontsize=11,
            arrowprops=dict(arrowstyle="->", color="black"),
        )
        ax_h.annotate(
            "$h(t_1) = 1$",
            xy=(t1, 1.0),
            xytext=(t1 - t1 * 0.28, 1.04),
            fontsize=11,
            arrowprops=dict(arrowstyle="->", color="black"),
        )
        ax_h.set_xlabel("$t$", fontsize=13)
        ax_h.set_ylabel("$h(t)$", fontsize=13)
        ax_h.set_title(
            rf"Temporal truncation profile $h(t) = \exp(-\gamma\,(t_1 - t)^2)$,"
            rf"  $\gamma = {g2_gamma}$",
            fontsize=12,
        )
        ax_h.set_xlim(-0.01, t1 + 0.02)
        ax_h.set_ylim(-0.05, 1.12)
        ax_h.legend(fontsize=11)
        ax_h.grid(True, alpha=0.3)
        fig_h.suptitle(
            f"Plot B_h — $h(t)$ profile  ($\\gamma = {g2_gamma}$)\n{cfg_str}",
            fontsize=10,
        )
        fig_h.tight_layout()
        fig_h.savefig(out_dir / "diagnostics" / "plotBh_temporal_truncation.png", dpi=150)
        plt.close(fig_h)
        logger.info(f"[OK] Plot B_h — temporal truncation h(t)  (gamma={g2_gamma})")

    # Evaluate ETCNN_B before training to enable before/after error comparison in Plot B7
    etcnn_b.to(DEVICE)

    if load_etcnn_b is not None:
        _b_path = load_etcnn_b
        if _b_path.is_dir():
            _b_path = _b_path / "models" / "etcnn_b.pt"
        if _b_path.exists():
            logger.info(f"Stage B: loading pre-trained weights from {_b_path} ...")
            etcnn_b.load_state_dict(torch.load(_b_path, map_location=DEVICE))
            logger.info(f"  Weights loaded — will train for {iters_b} additional iterations.")
        else:
            logger.warning(f"Stage B: --load-etcnn-b path not found: {_b_path}  (starting from random init)")

    _s_eval_init = torch.tensor(np.linspace(60.0, 140.0, 81), dtype=torch.get_default_dtype()).to(DEVICE)
    _x_eval_init = torch.stack([_s_eval_init, torch.zeros_like(_s_eval_init)], dim=1)
    with torch.no_grad():
        etcnn_b_prices_init = to_np(etcnn_b(_x_eval_init).squeeze())

    _s_star_b = s_star if (not isinstance(s_star, float) or s_star == s_star) else None
    if use_spatial_weight:
        logger.info(
            f"Stage B spatial weighting: ACTIVE  s*={_s_star_b}, sigma_w={sigma_w}, eps_w={eps_w}"
        )
    else:
        logger.info("Stage B spatial weighting: OFF (plain MSE)")
    hist_b = train_model(
        etcnn_b, iters_b, S_TRAIN_LO, S_TRAIN_HI, 0.0, t1,
        payoff_t1, label="ETCNN_B", weight_decay=weight_decay, tc_enforced=True,
        s_star=_s_star_b, sigma_w=sigma_w, eps_w=eps_w,
        use_spatial_weight=use_spatial_weight,
    )
    plot_training_metrics(hist_b, "ETCNN_B", out_dir)

    # === Plot B4 — Stage B loss curves ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].semilogy(hist_b["iter"], hist_b["loss_f"], label="$L_f$")
    axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("$L_f$")
    axes[0].set_title("Stage B: PDE residual"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].semilogy(hist_b["iter"], hist_b["loss_tc"], label="$L_{tc}$", color="tab:orange")
    axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("$L_{tc}$")
    axes[1].set_title("Stage B: Terminal loss"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    fig.suptitle(
        f"Plot B4 — Stage B loss curves  (ETCNN$_B$ on $[0, t_1]$,  {iters_b} iters)\n{cfg_str}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "training_metrics" / "plotB4_stageD_loss.png", dpi=150)
    plt.close(fig)
    logger.info("[OK] Plot B4 — Stage D loss curves")

    # ---------------------------------------------------------------
    # Evaluation vs binomial tree
    # ---------------------------------------------------------------
    logger.info("Computing binomial tree reference prices ...")
    s_eval_arr = np.linspace(60.0, 140.0, 81)
    bt_prices = np.array([
        bermuda_put_binomial_tree(float(s), K, r, sigma, T, [t1], N=2000)
        for s in s_eval_arr
    ])
    euro_bt = np.array([
        european_put_binomial_tree(float(s), K, r, sigma, T, N=2000)
        for s in s_eval_arr
    ])
    logger.info("  BT prices computed.")

    # ETCNN_B(s, 0)
    s_eval_t = torch.tensor(s_eval_arr, dtype=torch.get_default_dtype())
    t_zero = torch.zeros_like(s_eval_t)
    x_eval_0 = torch.stack([s_eval_t, t_zero], dim=1).to(DEVICE)
    with torch.no_grad():
        etcnn_b_prices = to_np(etcnn_b(x_eval_0).squeeze())

    # Ve(s, 0) analytical
    tau_full = torch.full_like(s_eval_t, T)
    ve_prices = to_np(black_scholes_put(s_eval_t, K, r, sigma, tau_full))

    # === Plot B5 — Final price comparison ===
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(s_eval_arr, bt_prices, label="Bermudan BT (N=2000)", linewidth=2.5)
    ax.plot(s_eval_arr, etcnn_b_prices, label=r"ETCNN$_B(s, 0)$", linewidth=2, linestyle="--")
    ax.plot(s_eval_arr, ve_prices, label=r"European $V^e(s, 0)$", linewidth=2, linestyle=":")
    ax.set_xlabel("s")
    ax.set_ylabel("Option price at $t=0$")
    ax.set_title("Price at $t = 0$:  ETCNN$_B$ vs Bermudan BT (N=2000) vs European $V^e$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.suptitle(f"Plot B5 — Price comparison at $t = 0$\n{cfg_str}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "pricing" / "plotB5_price_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("[OK] Plot B5 — Price comparison")

    # Check price ordering: Bermudan >= European (early exercise premium)
    bermuda_at_K = float(np.interp(K, s_eval_arr, bt_prices))
    euro_at_K = float(np.interp(K, s_eval_arr, ve_prices))
    etcnn_b_at_K = float(np.interp(K, s_eval_arr, etcnn_b_prices))
    logger.info(f"  At s=K=100: Bermudan BT = {bermuda_at_K:.4f}, "
                f"ETCNN_B = {etcnn_b_at_K:.4f}, European = {euro_at_K:.4f}")
    if bermuda_at_K >= euro_at_K - 1e-4:
        logger.info("  Price ordering: Bermudan >= European — financially consistent.")
    else:
        logger.warning("  Price ordering VIOLATED: Bermudan < European!")

    # === Plot B6 — Full piecewise price surface ===
    ns, nt = 200, 200
    s_surf = torch.linspace(60.0, 140.0, ns)
    t_surf = torch.linspace(0.0, T, nt)
    S_g, T_g = torch.meshgrid(s_surf, t_surf, indexing="ij")

    # For t >= t1 use ETCNN_A, for t < t1 use ETCNN_B
    mask_a = T_g >= t1
    mask_b = T_g < t1

    x_all = torch.stack([S_g.reshape(-1), T_g.reshape(-1)], dim=1).to(DEVICE)
    V_surface = torch.zeros(ns, nt)

    # ETCNN_A region
    idx_a = mask_a.reshape(-1)
    if idx_a.any():
        with torch.no_grad():
            V_surface.reshape(-1)[idx_a] = etcnn_a(x_all[idx_a]).cpu().squeeze()

    # ETCNN_B region
    idx_b = mask_b.reshape(-1)
    if idx_b.any():
        with torch.no_grad():
            V_surface.reshape(-1)[idx_b] = etcnn_b(x_all[idx_b]).cpu().squeeze()

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.pcolormesh(to_np(t_surf), to_np(s_surf), to_np(V_surface),
                       shading="auto", cmap="Blues")
    fig.colorbar(im, ax=ax, label="Bermudan put price")
    ax.axvline(t1, color="red", linestyle="--", linewidth=1.5, label=f"$t_1 = {t1}$")
    ax.set_xlabel("t"); ax.set_ylabel("s")
    ax.set_title(
        "Full piecewise price surface  "
        r"($\tilde{u}^{(A)}$ for $t \geq t_1$,  $\tilde{u}^{(B)}$ for $t < t_1$)"
    )
    ax.legend()
    fig.suptitle(f"Plot B6 — Bermudan put price surface over $[0, T]$\n{cfg_str}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "pricing" / "plotB6_bermudan_surface.png", dpi=150)
    plt.close(fig)

    # Check continuity at t1 by directly comparing model outputs at the boundary
    # (not adjacent time slices, which may differ due to different training objectives)
    s_stitch = torch.linspace(60.0, 140.0, 100).to(DEVICE)
    t_stitch = torch.full_like(s_stitch, t1)
    x_stitch = torch.stack([s_stitch, t_stitch], dim=1)
    
    with torch.no_grad():
        u_a_stitch = etcnn_a(x_stitch).cpu().squeeze()  # Stage A at t1
        u_b_stitch = etcnn_b(x_stitch).cpu().squeeze()  # Stage B at t1
    
    jump = float(torch.max(torch.abs(u_a_stitch - u_b_stitch)))
    logger.info(f"  Max jump at t1 intermediate boundary: {jump:.4e}")
    logger.info("[OK] Plot B6 — Bermudan surface")

    # === Plot B7 — Error vs BT at t=0 (init vs trained) ===
    err_vs_bt_init = np.abs(etcnn_b_prices_init - bt_prices)
    err_vs_bt = np.abs(etcnn_b_prices - bt_prices)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(s_eval_arr, err_vs_bt_init, linewidth=2, linestyle="--", label=f"Init  (max={err_vs_bt_init.max():.2e})")
    ax.plot(s_eval_arr, err_vs_bt, linewidth=2, label=f"Trained  (max={err_vs_bt.max():.2e})")
    ax.set_xlabel("$s$")
    ax.set_ylabel("$|$ETCNN$_B(s,0) - V^{BT}(s,0)|$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.suptitle(f"Plot B7 — Pointwise error vs binomial tree at $t = 0$\n{cfg_str}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "diagnostics" / "plotB7_error_vs_bt.png", dpi=150)
    plt.close(fig)
    logger.info(f"[OK] Plot B7 — Error vs BT: init max={err_vs_bt_init.max():.2e}, trained max={err_vs_bt.max():.2e}")

    # === Plot B8 — Greeks at t=0 ===
    s_greeks = torch.linspace(60.0, 140.0, 300).to(DEVICE)
    t_greeks = torch.zeros_like(s_greeks).to(DEVICE)
    
    delta_berm, gamma_berm, theta_berm = compute_greeks_nn(etcnn_b, s_greeks, t_greeks)
    delta_euro, gamma_euro, theta_euro = compute_greeks_analytical(s_greeks, t_greeks, K, r, sigma, T)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Delta
    axes[0].plot(to_np(s_greeks), to_np(delta_berm), label="ETCNN_B (Bermudan)", linewidth=2)
    axes[0].plot(to_np(s_greeks), to_np(delta_euro), label="Analytical (European)", linewidth=2, linestyle=":")
    axes[0].set_xlabel("s")
    axes[0].set_ylabel(r"$\Delta = \partial V / \partial s$")
    axes[0].set_title("Delta")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Gamma
    axes[1].plot(to_np(s_greeks), to_np(gamma_berm), label="ETCNN_B (Bermudan)", linewidth=2)
    axes[1].plot(to_np(s_greeks), to_np(gamma_euro), label="Analytical (European)", linewidth=2, linestyle=":")
    axes[1].set_xlabel("s")
    axes[1].set_ylabel(r"$\Gamma = \partial^2 V / \partial s^2$")
    axes[1].set_title("Gamma")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Theta
    axes[2].plot(to_np(s_greeks), to_np(theta_berm), label="ETCNN_B (Bermudan)", linewidth=2)
    axes[2].plot(to_np(s_greeks), to_np(theta_euro), label="Analytical (European)", linewidth=2, linestyle=":")
    axes[2].set_xlabel("s")
    axes[2].set_ylabel(r"$\Theta = \partial V / \partial t$")
    axes[2].set_title("Theta")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    fig.suptitle(
        f"Plot B8 — Greeks $\\Delta, \\Gamma, \\Theta$ at $t = 0$  (ETCNN$_B$ Bermudan vs analytical European)\n{cfg_str}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "greeks" / "plotB8_greeks.png", dpi=150)
    plt.close(fig)
    logger.info("[OK] Plot B8 — Greeks at t=0")

    # === Plot B9 — Test II: Spatial Distribution of PDE Residual ===
    logger.info("Running Test II: Spatial Distribution of PDE Residual at t1- ...")
    s_res = torch.linspace(S_EVAL_LO, S_EVAL_HI, 1000).to(DEVICE)
    s_res.requires_grad_(True)
    t_res = torch.full_like(s_res, t1 - 1e-4).to(DEVICE)
    t_res.requires_grad_(True)

    x_res = torch.stack([s_res, t_res], dim=1)

    # Operator Bypass for diagnostic plot too
    if hasattr(etcnn_b, "forward_pde"):
        u_b_res = etcnn_b.forward_pde(x_res).squeeze()
    else:
        u_b_res = etcnn_b(x_res).squeeze()

    residual_t1 = bsm_operator(u_b_res, s_res, t_res, r, q, sigma)
    R_s = residual_t1 ** 2

    inner_etcnn = etcnn_b.etcnn if hasattr(etcnn_b, "etcnn") else etcnn_b

    # PDE residual of g1 * u_theta (ultimate bypass: no g2, no v)
    s_res_g1 = torch.linspace(S_EVAL_LO, S_EVAL_HI, 1000).to(DEVICE)
    s_res_g1.requires_grad_(True)
    t_res_g1 = torch.full_like(s_res_g1, t1 - 1e-4).to(DEVICE)
    t_res_g1.requires_grad_(True)
    x_res_g1 = torch.stack([s_res_g1, t_res_g1], dim=1)
    net_input_g1 = inner_etcnn.normalizer(x_res_g1) if inner_etcnn.normalizer is not None else x_res_g1
    u_theta_g1 = inner_etcnn.resnet(net_input_g1).squeeze()
    g1_val = inner_etcnn._g1(s_res_g1.unsqueeze(1), t_res_g1.unsqueeze(1)).squeeze()
    g1_u_theta = g1_val * u_theta_g1
    residual_g1 = bsm_operator(g1_u_theta, s_res_g1, t_res_g1, r, q, sigma)
    R_s_g1 = residual_g1 ** 2

    # PDE residual of raw u_theta (resnet output, no ansatz)
    s_res_nn = torch.linspace(S_EVAL_LO, S_EVAL_HI, 1000).to(DEVICE)
    s_res_nn.requires_grad_(True)
    t_res_nn = torch.full_like(s_res_nn, t1 - 1e-4).to(DEVICE)
    t_res_nn.requires_grad_(True)
    x_res_nn = torch.stack([s_res_nn, t_res_nn], dim=1)
    net_input_nn = inner_etcnn.normalizer(x_res_nn) if inner_etcnn.normalizer is not None else x_res_nn
    u_theta_nn = inner_etcnn.resnet(net_input_nn).squeeze()
    residual_nn = bsm_operator(u_theta_nn, s_res_nn, t_res_nn, r, q, sigma)
    R_s_nn = residual_nn ** 2

    fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 13))

    _bypass_v = hasattr(etcnn_b, "bypass_v") and etcnn_b.bypass_v
    if _bypass_v:
        title_a = (
            "(a)  forward\\_pde — bypass\\_v active ($v$ dropped, $g_2$ kept):\n"
            r"     $R(s) = |\mathcal{F}(g_1 u_\theta + g_2)(s, t_1^-)|^2$"
        )
        ylabel_a = "$R(s) = |\\mathcal{F}(g_1 u_\\theta + g_2)(s, t_1^-)|^2$"
    else:
        title_a = (
            "(a)  forward\\_pde — full ansatz $v + g_1 u_\\theta + g_2$:\n"
            r"     $R(s) = |\mathcal{F}[\tilde{u}_{NN}^{(B)}](s, t_1^-)|^2$"
        )
        ylabel_a = "$R(s) = |\\mathcal{F}(\\tilde{u}_{NN}^{(B)})(s, t_1^-)|^2$"

    ax.plot(to_np(s_res), to_np(R_s), label="PDE Residual $R(s)$", color="darkorange", linewidth=2)
    if not np.isnan(s_star):
        ax.axvline(s_star, color="red", linestyle=":", alpha=0.7, label=f"Exercise boundary $s^* \\approx {s_star:.1f}$")
    ax.set_title(title_a)
    ax.set_xlabel("Asset Price $s$")
    ax.set_ylabel(ylabel_a)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2.plot(to_np(s_res_g1), to_np(R_s_g1), label="PDE Residual $R(s)$", color="forestgreen", linewidth=2)
    if not np.isnan(s_star):
        ax2.axvline(s_star, color="red", linestyle=":", alpha=0.7, label=f"Exercise boundary $s^* \\approx {s_star:.1f}$")
    ax2.set_title(
        r"(b)  Direct $g_1 u_\theta$ via forward\_neural\_manifold"
        "\n"
        r"     $R(s) = |\mathcal{F}(g_1 u_\theta)(s, t_1^-)|^2$"
    )
    ax2.set_xlabel("Asset Price $s$")
    ax2.set_ylabel("$R(s) = |\\mathcal{F}(g_1 u_\\theta)(s, t_1^-)|^2$")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.plot(to_np(s_res_nn), to_np(R_s_nn), label="PDE Residual $R(s)$", color="steelblue", linewidth=2)
    if not np.isnan(s_star):
        ax3.axvline(s_star, color="red", linestyle=":", alpha=0.7, label=f"Exercise boundary $s^* \\approx {s_star:.1f}$")
    ax3.set_title("(c)  ResNet only (no ansatz):  $R(s) = |\\mathcal{F}(u_\\theta)(s, t_1^-)|^2$")
    ax3.set_xlabel("Asset Price $s$")
    ax3.set_ylabel("$R(s) = |\\mathcal{F}(u_\\theta)(s, t_1^-)|^2$")
    ax3.set_yscale("log")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    fig.suptitle(
        f"Plot B9 — Spatial distribution of PDE residual at $t_1^-$  ($s^* \\approx {s_star:.1f}$)\n{cfg_str}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "diagnostics" / "plotB9_test2_pde_residual.png", dpi=150)
    plt.close(fig)
    logger.info("[OK] Plot B9 — Test II: PDE Residual at t1-")

    # === Plot B9b — Spatio-temporal PDE residual heatmap ===
    # This plot shows |F[ũ](s,t)| as a 2D heatmap over the full (s,t) domain,
    # split at t1 between ETCNN_B (left panel) and ETCNN_A (right panel).
    #
    # Validity checks before generating:
    #
    #   Check 1 — Structural: requires two sub-intervals (t1 > 0 and T > t1).
    #   Check 2 — Spatial dimension: the BSM PDE is 1D in s (single-asset), so
    #     a (s, t) heatmap is well-defined. For a multi-factor model one would
    #     need to fix the extra dimensions — not applicable here.
    #   Check 3 — Physical: for a put option the exercise boundary satisfies
    #     s* < K when r > 0. If r <= 0 it can be optimal to never exercise early,
    #     making s* = 0 and the PDE residual uniform (no kink to reveal).
    #   Check 4 — Domain coverage: s* must lie inside [S_EVAL_LO, S_EVAL_HI] for
    #     the heatmap to capture the kink region.
    logger.info("Running Plot B9b — Spatio-temporal PDE residual heatmap ...")

    # Check 1
    _b9b_ok = True
    if not (t1 > 0 and T > t1):
        logger.warning(
            "[SKIP] Plot B9b — requires two sub-intervals (t1 > 0 and T > t1); "
            f"got t1={t1}, T={T}."
        )
        _b9b_ok = False

    # Check 3
    if _b9b_ok and r <= 0.0:
        logger.warning(
            "[WARN] Plot B9b — r <= 0: early exercise may never be optimal for "
            "a put (s* → 0), so the kink region may not appear in the heatmap. "
            f"r = {r}. Proceeding anyway."
        )

    # Check 4
    if _b9b_ok and not np.isnan(s_star):
        if not (S_EVAL_LO <= s_star <= S_EVAL_HI):
            logger.warning(
                f"[WARN] Plot B9b — exercise boundary s* = {s_star:.2f} lies "
                f"outside evaluation domain [{S_EVAL_LO}, {S_EVAL_HI}]; "
                "kink may not be visible in the heatmap."
            )
        elif s_star >= K:
            logger.warning(
                f"[WARN] Plot B9b — s* = {s_star:.2f} >= K = {K}; "
                "for a standard put with r > 0 we expect s* < K."
            )
        else:
            logger.info(
                f"  [OK] s* = {s_star:.2f} < K = {K}, inside eval domain — "
                "kink region will be visible."
            )

    if _b9b_ok:
        # Resolution: 120 spatial × 70 temporal per panel.
        # Keep moderate to avoid long autograd computation on CPU.
        ns_heat, nt_heat = 120, 70

        s_heat = torch.linspace(S_EVAL_LO, S_EVAL_HI, ns_heat).to(DEVICE)
        s_np_heat = to_np(s_heat)

        def _residual_grid(model, t_lo: float, t_hi: float, use_forward_pde: bool) -> np.ndarray:
            """Compute |F[model](s, t)|² on a (ns_heat × nt_heat) grid.

            Returns a (ns_heat, nt_heat) numpy array.
            t_lo / t_hi define the temporal extent of the panel.
            """
            t_vals = torch.linspace(t_lo, t_hi, nt_heat).to(DEVICE)

            # Flattened grid — s varies along axis 0, t along axis 1
            S_g, T_g = torch.meshgrid(s_heat, t_vals, indexing="ij")  # (ns, nt)
            s_flat = S_g.reshape(-1).clone().requires_grad_(True)
            t_flat = T_g.reshape(-1).clone().requires_grad_(True)
            x_flat = torch.stack([s_flat, t_flat], dim=1)

            if use_forward_pde and hasattr(model, "forward_pde"):
                u_flat = model.forward_pde(x_flat).squeeze()
            else:
                u_flat = model(x_flat).squeeze()

            res_flat = bsm_operator(u_flat, s_flat, t_flat, r, q, sigma)
            return res_flat.detach().abs().reshape(ns_heat, nt_heat).cpu().numpy(), to_np(t_vals)

        # Panel 1: ETCNN_B on [0, t1^-]   include t1^- = t1 - 1e-4
        R_b, t_b_np = _residual_grid(etcnn_b, 0.0, t1 - 1e-4, use_forward_pde=True)
        # Panel 2: ETCNN_A on [t1, T]
        R_a, t_a_np = _residual_grid(etcnn_a, t1, T, use_forward_pde=False)

        # Shared log-scale colour limits (1st–99th percentile to avoid outlier saturation)
        all_vals = np.concatenate([R_b.ravel(), R_a.ravel()])
        all_pos = all_vals[all_vals > 0]
        if all_pos.size == 0:
            logger.warning("[WARN] Plot B9b — all residuals are zero; skipping heatmap.")
            _b9b_ok = False
        else:
            vmin_heat = float(np.percentile(all_pos, 1))
            vmax_heat = float(np.percentile(all_pos, 99))
            if vmin_heat >= vmax_heat:
                vmin_heat = vmax_heat * 1e-6

    if _b9b_ok:
        norm_heat = mcolors.LogNorm(vmin=vmin_heat, vmax=vmax_heat)
        # YlOrRd: low residual → yellow (light), high residual → deep red.
        # This avoids the black-zone artifact produced by hot_r, where high
        # values (including anything clamped above vmax) map to black —
        # visually indistinguishable from missing data.
        cmap_heat = plt.cm.YlOrRd.copy()
        cmap_heat.set_over("darkred")   # values above 99th-pct → darkred, not black
        cmap_heat.set_bad("lightgray")  # NaN / LogNorm-invalid (zero) → gray

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # --- Panel 1: ETCNN_B on [0, t1^-] ---
        im0 = axes[0].pcolormesh(
            t_b_np, s_np_heat, R_b,
            shading="auto", cmap=cmap_heat, norm=norm_heat,
        )
        fig.colorbar(im0, ax=axes[0], extend="both", label=r"$|\mathcal{F}[\tilde{u}^{(B)}](s,t)|$")
        axes[0].axvline(t1 - 1e-4, color="cyan", linestyle="--", linewidth=1.2,
                        label=f"$t_1^- \\approx {t1}$")
        if not np.isnan(s_star):
            axes[0].axhline(s_star, color="white", linestyle=":", linewidth=1.2,
                            alpha=0.9, label=f"$s^* = {s_star:.1f}$")
        axes[0].set_xlabel("$t$")
        axes[0].set_ylabel("$s$")
        axes[0].set_title(
            r"ETCNN$_B$,  $t \in [0,\, t_1^-]$"
            "\n"
            r"$|\mathcal{F}[\tilde{u}^{(B)}](s,t)|$  (continuation region: $s > s^*$)"
        )
        axes[0].legend(fontsize=9, loc="upper right")

        # --- Panel 2: ETCNN_A on [t1, T] ---
        im1 = axes[1].pcolormesh(
            t_a_np, s_np_heat, R_a,
            shading="auto", cmap=cmap_heat, norm=norm_heat,
        )
        fig.colorbar(im1, ax=axes[1], extend="both", label=r"$|\mathcal{F}[\tilde{u}^{(A)}](s,t)|$")
        axes[1].axvline(t1, color="cyan", linestyle="--", linewidth=1.2,
                        label=f"$t_1 = {t1}$")
        axes[1].axvline(T, color="lime", linestyle=":", linewidth=1.2,
                        label=f"$T = {T}$")
        if not np.isnan(s_star):
            axes[1].axhline(s_star, color="white", linestyle=":", linewidth=1.2,
                            alpha=0.9, label=f"$s^* = {s_star:.1f}$")
        axes[1].set_xlabel("$t$")
        axes[1].set_ylabel("$s$")
        axes[1].set_title(
            r"ETCNN$_A$,  $t \in [t_1,\, T]$"
            "\n"
            r"$|\mathcal{F}[\tilde{u}^{(A)}](s,t)|$"
        )
        axes[1].legend(fontsize=9, loc="upper right")

        fig.suptitle(
            f"Plot B9b — Spatio-temporal PDE residual  ($s^* \\approx {s_star:.1f}$,  "
            f"shared log colour scale)\n{cfg_str}",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(out_dir / "diagnostics" / "plotB9b_pde_residual_heatmap.png", dpi=150)
        plt.close(fig)
        logger.info("[OK] Plot B9b — Spatio-temporal PDE residual heatmap")

    # === Plot B10 — Test III: Neuron Weight Magnitudes ===
    if not hasattr(etcnn_a, "resnet"):
        logger.info("[SKIP] Plot B10 — Stage A has no ResNet (analytic mode)")
    else:
        logger.info("Running Test III: Neuron Weight Magnitudes in ETCNN_A ...")
        weights = []
        layer_names = []

        w = etcnn_a.resnet.input_layer[0].weight.detach().cpu().numpy().flatten()
        weights.append(w)
        layer_names.append("Input")

        for i, block in enumerate(etcnn_a.resnet.blocks):
            for j, layer in enumerate(block.layers):
                if isinstance(layer, torch.nn.Linear):
                    w = layer.weight.detach().cpu().numpy().flatten()
                    weights.append(w)
                    layer_names.append(f"B{i} L{j//2}")

        w = etcnn_a.resnet.output_layer.weight.detach().cpu().numpy().flatten()
        weights.append(w)
        layer_names.append("Output")

        fig, ax = plt.subplots(figsize=(12, 6))
        parts = ax.violinplot(weights, showmeans=True, showextrema=True)

        for pc in parts['bodies']:
            pc.set_facecolor('purple')
            pc.set_edgecolor('black')
            pc.set_alpha(0.5)

        ax.set_xticks(np.arange(1, len(layer_names) + 1))
        ax.set_xticklabels(layer_names, rotation=45, ha="right")
        ax.set_ylabel("Weight Value ($w$)")
        ax.set_title(
            "Plot B10 — Neuron weight magnitudes in ETCNN$^{(A)}$  "
            f"(M={M} blocks × L={L_BLOCK} layers, n={n} hidden units)\n"
            f"Large weights ($|w| > 1$) amplify high-frequency noise via $w^2$ in $\\partial^2_s \\tilde{{u}}$\n"
            f"{cfg_str}"
        )
        ax.grid(True, alpha=0.3)
        ax.axhline(1.0, color='red', linestyle=':', alpha=0.5)
        ax.axhline(-1.0, color='red', linestyle=':', alpha=0.5)

        fig.tight_layout()
        fig.savefig(out_dir / "diagnostics" / "plotB10_test3_weight_distribution.png", dpi=150)
        plt.close(fig)
        logger.info("[OK] Plot B10 — Test III: Neuron Weight Magnitudes")

    # Metrics
    l2_bt = np.sqrt(np.mean((etcnn_b_prices - bt_prices) ** 2))
    l2_ref_bt = np.sqrt(np.mean(bt_prices ** 2))
    rel_l2_bt = l2_bt / l2_ref_bt
    mae_bt = np.mean(err_vs_bt)

    # === Print B — Summary ===
    logger.info("")
    logger.info("=" * 60)
    logger.info("BERMUDAN PUT — SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Relative L2 error vs BT:     {rel_l2_bt:.6e}")
    logger.info(f"  MAE vs BT:                   {mae_bt:.6e}")
    logger.info(f"  Exercise boundary s* at t1:  {s_star:.2f}")
    logger.info(f"  Scaling constant c:          {c_scale:.6f}")
    logger.info(f"  Bermudan price at s=100:     {etcnn_b_at_K:.4f} (ETCNN_B)")
    logger.info(f"  Bermudan price at s=100:     {bermuda_at_K:.4f} (BT ref)")
    logger.info(f"  European price at s=100:     {euro_at_K:.4f}")
    logger.info(f"  Bermudan >= European?        {etcnn_b_at_K >= euro_at_K - 1e-4}")
    logger.info(f"  Max jump at t1:              {jump:.4e}")
    logger.info("=" * 60)

    model_dir = out_dir / "models"
    if not analytic_a:
        torch.save(etcnn_a.state_dict(), model_dir / "etcnn_a.pt")
    torch.save(etcnn_b.state_dict(), model_dir / "etcnn_b.pt")
    logger.info(
        "  Saved models: "
        + ("models/etcnn_a.pt, " if not analytic_a else "(no etcnn_a — analytic mode), ")
        + "models/etcnn_b.pt"
    )

    return {
        "rel_l2_bt": rel_l2_bt,
        "mae_bt": mae_bt,
        "s_star": s_star,
        "c_scale": c_scale,
        "bermuda_at_K": bermuda_at_K,
        "etcnn_b_at_K": etcnn_b_at_K,
        "euro_at_K": euro_at_K,
        "jump_at_t1": jump,
        # Extra fields for ablation aggregation
        "hist_b": hist_b,
        "etcnn_b_prices": etcnn_b_prices,
        "bt_prices": bt_prices,
        "s_eval_arr": s_eval_arr,
    }


# ===================================================================
#  MAIN
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="Phase 3 — ETCNN training")
    parser.add_argument("--iters", nargs='+', type=int, default=[50_000], help="Training iterations (can provide multiple values for different stages, e.g., 20000 1000)")
    parser.add_argument("--log-every", type=int, default=1000, help="Log interval")
    parser.add_argument(
        "--interp", type=str, default="cubic", choices=["cubic", "pchip", "linear"],
        help="Interpolation for Bermudan V(s,t1) when --put-ansatz is NOT set: "
             "'cubic' (C^2, default), 'pchip' (C^1, shape-preserving), or 'linear' (C^0)",
    )
    parser.add_argument(
        "--put-ansatz", action="store_true",
        help="Enable singularity extraction ansatz for Stage B (default: off). "
             "Decomposes U_B = v + u_tilde, removing the C^0 kink at s*.",
    )
    parser.add_argument(
        "--bypass-v", action="store_true",
        help="Operator Bypass: drop the fictitious put v(s,t) from the PDE loss "
             "to prevent catastrophic cancellation of its diverging derivatives "
             "near the exercise boundary.  g2 remains coupled so the network "
             "correctly compensates for L(g2).",
    )
    parser.add_argument(
        "--g2", type=str, default="taylor", choices=["taylor", "bs", "bs2002"],
        help="Terminal function g2 for ETCNN: 'taylor' (V1^e + V2^e, default), "
             "'bs' (exact Black-Scholes European put), or "
             "'bs2002' (Bjerksund-Stensland 2002 American put approximation)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device: auto (CUDA if available), cuda (fail if unavailable), or cpu",
    )
    parser.add_argument("--bermudan-only", action="store_true", help="Skip European problem and only run Bermudan")
    parser.add_argument("--european-only", action="store_true", help="Skip Bermudan problem and only run European")
    parser.add_argument(
        "--analytic-a", action="store_true",
        help="Replace Stage A with the exact Black-Scholes European put formula. "
             "No Stage A network is trained; the intermediate terminal condition "
             "at t1 becomes max(Φ(s), V^e(s,t1)) exactly.  Useful to isolate "
             "Stage B from Stage A approximation errors.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0, help="L2 regularization penalty for Adam optimizer")
    parser.add_argument(
        "--spatial-weight", action="store_true",
        help="Enable inverted-Gaussian spatial weighting of the Stage B PDE loss "
             "to suppress the gradient spike from the C^1 PCHIP knot at s*. "
             "Off by default (plain MSE). Requires --put-ansatz.",
    )
    parser.add_argument(
        "--sigma-w", type=float, default=1.0,
        help="Bandwidth of the inverted-Gaussian suppression window around s* "
             "(default 1.0). Only used when --spatial-weight is set.",
    )
    parser.add_argument(
        "--eps-w", type=float, default=1e-3,
        help="Lower bound of the spatial weight at s* (default 1e-3). "
             "Prevents complete nullification of the loss at the exercise boundary. "
             "Only used when --spatial-weight is set.",
    )
    parser.add_argument(
        "--g2-gamma",
        type=float,
        default=None,
        help=(
            "γ ≥ 0 for the temporal truncation h(t) = exp(-γ(t1-t)²) applied "
            "to the Stage B g2 field.  Disabled by default (h ≡ 1).  "
            "Typical values: 1–20."
        ),
    )
    parser.add_argument(
        "--load-etcnn-a",
        type=str,
        default=None,
        help=(
            "Path to pre-trained etcnn_a.pt to skip Stage A training. "
            "Can be either a .pt file or a run directory containing models/etcnn_a.pt"
        ),
    )
    parser.add_argument(
        "--load-etcnn-b",
        type=str,
        default=None,
        help=(
            "Path to a pre-trained etcnn_b.pt (or a run directory containing "
            "models/etcnn_b.pt) to warm-start Stage B.  The architecture is "
            "always rebuilt from the current flags; only the ResNet weights are "
            "loaded.  Training then continues for the number of iterations given "
            "by --iters."
        ),
    )
    parser.add_argument("--n-tc", type=int, default=1024, help="Number of terminal condition points")
    parser.add_argument("--n-f", type=int, default=4096, help="Number of interior PDE collocation points")
    parser.add_argument("--lam-f", type=float, default=20.0, help="Weight for PDE loss (lambda_f)")
    parser.add_argument("--lam-tc", type=float, default=1.0, help="Weight for terminal condition loss (lambda_tc)")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"], help="PyTorch default dtype")
    args = parser.parse_args()
    
    if args.dtype == "float64":
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)
        
    _apply_device_arg(args.device)

    # Update global hyperparameters if provided
    global N_TC, N_F, LAMBDA_F, LAMBDA_TC
    if args.n_tc is not None:
        N_TC = args.n_tc
    if args.n_f is not None:
        N_F = args.n_f
    if getattr(args, 'lam_f', None) is not None:
        LAMBDA_F = args.lam_f
    if getattr(args, 'lam_tc', None) is not None:
        LAMBDA_TC = args.lam_tc

    # Output directory
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    iters_str = "_".join(map(str, args.iters))
    mode_tag = "put-ansatz" if args.put_ansatz else f"interp-{args.interp}"
    gamma_tag = f"_h{args.g2_gamma}" if args.g2_gamma is not None else ""
    out_dir = Path("data/phase3_training") / (
        f"{timestamp}_iters{iters_str}_K{K:.0f}_{mode_tag}_g2-{args.g2}{gamma_tag}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for grouped plots
    (out_dir / "training_metrics").mkdir(exist_ok=True)
    (out_dir / "pricing").mkdir(exist_ok=True)
    (out_dir / "greeks").mkdir(exist_ok=True)
    (out_dir / "diagnostics").mkdir(exist_ok=True)
    (out_dir / "models").mkdir(exist_ok=True)

    # Save metadata
    metadata = {
        "command": " ".join(sys.argv),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "K": K,
            "r": r,
            "sigma": sigma,
            "T": T,
            "q": q,
            "t1": t1,
        },
        "hyperparameters": {
            "M": M,
            "L_BLOCK": L_BLOCK,
            "n": n,
            "N_TC": N_TC,
            "N_F": N_F,
            "LAMBDA_F": LAMBDA_F,
            "LAMBDA_TC": LAMBDA_TC,
            "SEED": SEED,
            "iters": args.iters,
            "log_every": args.log_every,
            "put_ansatz": args.put_ansatz,
            "bypass_v": args.bypass_v,
            "spatial_weight": args.spatial_weight,
            "sigma_w": args.sigma_w,
            "eps_w": args.eps_w,
            "g2_gamma": args.g2_gamma,
            "load_etcnn_b": args.load_etcnn_b,
            "analytic_a": args.analytic_a,
            "interp_method": args.interp,
            "g2_type": args.g2,
            "device": args.device,
            "european_only": args.european_only,
            "bermudan_only": args.bermudan_only,
        },
        "domain": {
            "S_TRAIN_LO": S_TRAIN_LO,
            "S_TRAIN_HI": S_TRAIN_HI,
            "S_EVAL_LO": S_EVAL_LO,
            "S_EVAL_HI": S_EVAL_HI,
        }
    }
    with open(out_dir / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False, width=float("inf"))

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out_dir / "training.log"),
        ],
    )
    # Suppress matplotlib's INFO-level font substitution messages (e.g.
    # "Substituting symbol F from STIXNonUnicode") — these are harmless
    # fallbacks when rendering \mathcal glyphs and do not affect output.
    logging.getLogger("matplotlib.mathtext").setLevel(logging.WARNING)
    logger.info(f"Phase 3 — ETCNN Training")
    logger.info(f"  Output: {out_dir}")
    logger.info(f"  Command: {' '.join(sys.argv)}")
    logger.info(f"  Iters: {args.iters}")
    if args.put_ansatz:
        logger.info(f"  Stage B method: singularity extraction ansatz (PCHIP residual)")
    else:
        logger.info(f"  Stage B method: {args.interp} interpolation of V(s, t1)")
    logger.info(f"  g2 type (ETCNN terminal): {args.g2}")
    if args.g2_gamma is not None:
        logger.info(f"  Stage B g2 temporal truncation: gamma={args.g2_gamma}")
    logger.info(f"  Device (requested): {args.device}")
    logger.info(f"  Parameters: K={K}, r={r}, sigma={sigma}, T={T}, q={q}, t1={t1}")
    logger.info(f"  Hyperparameters: M={M}, L_BLOCK={L_BLOCK}, n={n}, N_TC={N_TC}, N_F={N_F}, LAMBDA_F={LAMBDA_F}, LAMBDA_TC={LAMBDA_TC}, SEED={SEED}")
    logger.info(f"  PyTorch: {torch.__version__}")
    logger.info(f"  Device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))

    if args.bermudan_only and args.european_only:
        print("ERROR: --bermudan-only and --european-only are mutually exclusive.", file=sys.stderr)
        sys.exit(2)

    # Run problems
    load_path = None
    if args.load_etcnn_a:
        requested = Path(args.load_etcnn_a)
        if requested.is_dir():
            candidate_models = requested / "models" / "etcnn_a.pt"
            candidate_root = requested / "etcnn_a.pt"
            if candidate_models.exists():
                load_path = candidate_models
            else:
                load_path = candidate_root
        else:
            load_path = requested

    load_b_path = Path(args.load_etcnn_b) if args.load_etcnn_b else None
    if load_b_path is not None:
        logger.info(f"  Stage B warm-start: {load_b_path}")

    run_european = not args.bermudan_only
    run_bermudan = not args.european_only

    if run_european:
        eur_results = european_problem(
            out_dir,
            args.iters[0],
            weight_decay=args.weight_decay,
            g2_type=args.g2,
        )
    else:
        eur_results = None

    if run_bermudan:
        ber_results = bermudan_problem(
            out_dir,
            args.iters,
            interp_method=args.interp,
            put_ansatz=args.put_ansatz,
            weight_decay=args.weight_decay,
            load_etcnn_a=load_path,
            g2_type=args.g2,
            bypass_v=args.bypass_v,
            sigma_w=args.sigma_w,
            eps_w=args.eps_w,
            use_spatial_weight=args.spatial_weight,
            g2_gamma=args.g2_gamma,
            load_etcnn_b=load_b_path,
            analytic_a=args.analytic_a,
        )
    else:
        ber_results = None

    # === Joint summary ===
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 3 — JOINT SUMMARY")
    logger.info("=" * 70)
    if eur_results:
        logger.info(f"  European ETCNN rel L2:     {eur_results['rel_l2_etcnn']:.6e}")
        logger.info(f"  European PINN  rel L2:     {eur_results['rel_l2_pinn']:.6e}")
    if ber_results:
        logger.info(f"  Bermudan ETCNN rel L2 (vs BT): {ber_results['rel_l2_bt']:.6e}")
        logger.info(f"  Bermudan at s=100:         {ber_results['etcnn_b_at_K']:.4f}")
        logger.info(f"  European at s=100:         {ber_results['euro_at_K']:.4f}")
        logger.info(f"  Bermudan >= European?      {ber_results['etcnn_b_at_K'] >= ber_results['euro_at_K'] - 1e-4}")
        logger.info(f"  Max jump at t1:            {ber_results['jump_at_t1']:.4e}")
    logger.info("=" * 70)

    # Pass/fail
    eur_pass = eur_results["rel_l2_etcnn"] < 5e-4 if eur_results else True
    if ber_results:
        ber_ordering = ber_results["etcnn_b_at_K"] >= ber_results["euro_at_K"] - 1e-4
        # The jump measures max(0, Phi(s) - etcnn_a(s,t1)) — non-zero only where Stage A
        # violates the early-exercise constraint (prices below intrinsic value).
        # A large jump indicates undertraining of Stage A, not a stitching artefact.
        ber_continuity = ber_results["jump_at_t1"] < 1e-2
    else:
        ber_ordering = True
        ber_continuity = True

    if eur_pass and ber_ordering and ber_continuity:
        logger.info("\n  All Phase 3 checks PASSED. Ready for Phase 4.")
    else:
        logger.info("\n  Some checks failed:")
        if not eur_pass and eur_results:
            logger.info(f"    - European L2 error {eur_results['rel_l2_etcnn']:.2e} > 5e-4")
        if not ber_ordering:
            logger.info("    - Bermudan < European (price ordering violated)")
        if not ber_continuity:
            logger.info(f"    - Intermediate jump {ber_results['jump_at_t1']:.2e} > 1e-2 (Stage A violates early-exercise constraint)")

    logger.info(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
