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
import sys
import yaml
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from learning_option_pricing.models.etcnn import (
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
def compute_losses(model, s_f, t_f, s_tc, t_tc, payoff_fn, lam_f, lam_tc):
    """Evaluate L_f (PDE residual) and L_tc (terminal MSE)."""
    # PDE residual at interior points
    x_f = torch.stack([s_f, t_f], dim=1)
    u_f = model(x_f).squeeze()
    F_u = bsm_operator(u_f, s_f, t_f, r, q, sigma)
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
def train_model(
    model: torch.nn.Module,
    total_iters: int,
    s_lo: float,
    s_hi: float,
    t_lo: float,
    t_hi: float,
    payoff_fn,
    label: str = "model",
    log_every: int = 1000,
    weight_decay: float = 0.0,
):
    """Train a model with Adam + two-stage LR schedule."""
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, build_lr_lambda(total_iters))

    history = {"loss": [], "loss_f": [], "loss_tc": [], "iter": [], "grad_norm": [], "lr": []}
    model.train()

    t0 = time.time()
    for it in range(1, total_iters + 1):
        optimizer.zero_grad()
        s_f, t_f, s_tc, t_tc = sample_collocation(
            N_F, N_TC, s_lo, s_hi, t_lo, t_hi,
        )
        loss, lf, ltc = compute_losses(
            model, s_f, t_f, s_tc, t_tc, payoff_fn, LAMBDA_F, LAMBDA_TC,
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
            logger.info(
                f"[{label}] iter {it:>6d}/{total_iters}  "
                f"loss={loss.item():.6e}  L_f={lf:.6e}  L_tc={ltc:.6e}  "
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
    axes[0, 0].semilogy(hist["iter"], hist["loss_f"], label="$L_f$")
    axes[0, 0].semilogy(hist["iter"], hist["loss_tc"], label="$L_{tc}$", color="tab:orange")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Loss Components")
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

    fig.suptitle(f"Plot E1 — Loss curves (European Put Option, K={K}, r={r}, $\\sigma$={sigma}, T={T})")
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

    fig.suptitle(f"Plot E2 — ETCNN vs analytical (European Put Option, K={K}, r={r}, $\\sigma$={sigma}, T={T})")
    fig.tight_layout()
    fig.savefig(out_dir / "pricing" / "plotE2_surface_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("[OK] Plot E2 — Surface comparison")

    # === Plot E3 — Pointwise error (ETCNN) ===
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.pcolormesh(to_np(t_vals), to_np(s_vals), to_np(err_etcnn),
                        shading="auto", cmap="hot_r")
    fig.colorbar(im, ax=ax, label="Absolute error")
    ax.set_xlabel("t"); ax.set_ylabel("s")
    ax.set_title(f"Plot E3 — ETCNN pointwise error (European Put Option, K={K}, r={r}, $\\sigma$={sigma}, T={T}, max={float(err_etcnn.max()):.2e})")
    fig.tight_layout()
    fig.savefig(out_dir / "diagnostics" / "plotE3_etcnn_error.png", dpi=150)
    plt.close(fig)
    logger.info(f"[OK] Plot E3 — ETCNN error: max={float(err_etcnn.max()):.2e}")

    # === Plot E4 — PINN pointwise error ===
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.pcolormesh(to_np(t_vals), to_np(s_vals), to_np(err_pinn),
                        shading="auto", cmap="hot_r")
    fig.colorbar(im, ax=ax, label="Absolute error")
    ax.set_xlabel("t"); ax.set_ylabel("s")
    ax.set_title(f"Plot E4 — PINN pointwise error (European Put Option, K={K}, r={r}, $\\sigma$={sigma}, T={T}, max={float(err_pinn.max()):.2e})")
    fig.tight_layout()
    fig.savefig(out_dir / "diagnostics" / "plotE4_pinn_error.png", dpi=150)
    plt.close(fig)
    logger.info(f"[OK] Plot E4 — PINN error: max={float(err_pinn.max()):.2e}")

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

    fig.suptitle(f"Plot E5 — Slice comparison at fixed $t$ (European Put Option, K={K}, r={r}, $\\sigma$={sigma}, T={T})")
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
    
    fig.suptitle(f"Plot E6 — Greeks at $t=0$ (European Put Option, K={K}, r={r}, $\\sigma$={sigma}, T={T})")
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

    torch.save(etcnn.state_dict(), out_dir / "etcnn_eur.pt")
    torch.save(pinn.state_dict(), out_dir / "pinn_eur.pt")
    logger.info("  Saved models: etcnn_eur.pt, pinn_eur.pt")

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
    ax.set_ylabel("$g_2(s)$")
    ax.set_title("(a)  $g_2(s) = V(s, t_1)$")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) first derivative
    ax = axes[0, 1]
    ax.plot(s_np, to_np(dv_c), label="Cubic ($C^2$)", linewidth=2)
    ax.plot(s_np, to_np(dv_l), label="Linear ($C^0$)", linewidth=2, linestyle="--")
    if not np.isnan(s_star):
        ax.axvline(s_star, color="red", ls=":", alpha=0.5)
    ax.set_xlabel("$s$")
    ax.set_ylabel("$\\partial g_2 / \\partial s$")
    ax.set_title("(b)  First derivative $\\partial g_2 / \\partial s$")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (c) second derivative — the key diagnostic
    ax = axes[1, 0]
    ax.plot(s_np, to_np(d2v_c), label="Cubic ($C^2$)", linewidth=2)
    ax.plot(s_np, to_np(d2v_l), label="Linear ($C^0$): $\\equiv 0$ a.e.", linewidth=2, linestyle="--")
    if not np.isnan(s_star):
        ax.axvline(s_star, color="red", ls=":", alpha=0.5)
    ax.set_xlabel("$s$")
    ax.set_ylabel("$\\partial^2 g_2 / \\partial s^2$")
    ax.set_title("(c)  Second derivative $\\partial^2 g_2 / \\partial s^2$  [KEY]")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (d) F(g2) — PDE operator on g2 alone
    ax = axes[1, 1]
    ax.plot(s_np, F_c, label="Cubic ($C^2$)", linewidth=2)
    ax.plot(s_np, F_l, label="Linear ($C^0$)", linewidth=2, linestyle="--")
    if not np.isnan(s_star):
        ax.axvline(s_star, color="red", ls=":", alpha=0.5)
    ax.set_xlabel("$s$")
    ax.set_ylabel("$\\mathcal{F}(g_2)$")
    ax.set_title(
        "(d)  $\\mathcal{F}(g_2) = rs \\, g_2' - r g_2"
        " + \\frac{1}{2}\\sigma^2 s^2 \\, g_2''$"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Interpolation diagnostic: cubic ($C^2$) vs linear ($C^0$)\n"
        f"(Bermudan Put, K={K}, r={r}, $\\sigma$={sigma})",
        fontsize=13,
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
    extraction: bool = False,
    weight_decay: float = 0.0,
    load_etcnn_a: Path | None = None,
    g2_type: str = "taylor",
):
    """Two-stage Bermudan put with exercise date t1=0.5.

    Args:
        out_dir: Output directory for plots and logs.
        total_iters: List of training iterations per stage [Stage A, Stage B].
        interp_method: Interpolation for V(s, t1) when *extraction* is False.
            ``"cubic"`` (default) uses a C^2 natural cubic spline;
            ``"pchip"`` uses a C^1 PCHIP interpolant;
            ``"linear"`` uses the original C^0 piecewise-linear interpolant.
        extraction: If True, use the singularity extraction ansatz to
            decompose U_B = v + u_tilde, removing the C^0 kink at s*.
            Default False (standard interpolation approach).
        weight_decay: L2 regularization penalty for Adam.
        load_etcnn_a: Path to pre-trained ETCNN_A model to skip Stage A training.
        g2_type: Terminal function type for Stage A — ``"taylor"`` or ``"bs"``.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("BERMUDAN PUT PROBLEM (t1=0.5)")
    logger.info("=" * 70)

    iters_a = total_iters[0]
    iters_b = total_iters[1] if len(total_iters) > 1 else total_iters[0]

    payoff_fn = lambda s: payoff_put(s, K)

    # ---------------------------------------------------------------
    # Stage A — train ETCNN on [t1, T]
    # ---------------------------------------------------------------
    torch.manual_seed(SEED)
    etcnn_a = AmericanPutETCNN(K=K, r=r, sigma=sigma, T=T, normalize_input=True, g2_type=g2_type)
    etcnn_a.to(DEVICE)
    logger.info(f"  Stage A g2 type: {g2_type}")

    if load_etcnn_a is not None and load_etcnn_a.exists():
        logger.info(f"Stage A: Loading pre-trained ETCNN_A from {load_etcnn_a} ...")
        etcnn_a.load_state_dict(torch.load(load_etcnn_a, map_location=DEVICE))
        etcnn_a.eval()
        hist_a = None
    else:
        logger.info(f"Stage A: training ETCNN_A on [t1, T] for {iters_a} iterations ...")
        hist_a = train_model(
            etcnn_a, iters_a, S_TRAIN_LO, S_TRAIN_HI, t1, T,
            payoff_fn, label="ETCNN_A", weight_decay=weight_decay
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

    if extraction:
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
        interp_cubic, interp_linear, s_star, K, r, sigma, out_dir,
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
    ax.plot(to_np(s_plot), to_np(v_t1_plot), label=r"$V(s, t_1) = \max(\Phi, \mathrm{hold})$",
            linewidth=2.5, linestyle="-.", color="black")
    if not np.isnan(s_star):
        ax.axvline(s_star, color="red", linestyle=":", alpha=0.7, label=f"$s^* \\approx {s_star:.1f}$")
    ax.set_xlabel("s")
    ax.set_ylabel("Value")
    ax.set_title(f"Intermediate terminal condition at $t_1 = {t1}$ (Bermudan Put Option, K={K}, r={r}, $\\sigma$={sigma}, T={T})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pricing" / "plotB1_intermediate.png", dpi=150)
    plt.close(fig)
    logger.info("[OK] Plot B1 — Intermediate terminal condition")

    # === Plot B1c / B1d — mode-specific diagnostics ===
    if extraction:
        # B1c — Extracted residual g2 at t1
        s_plot_1d = torch.linspace(60.0, 140.0, 500)
        with torch.no_grad():
            g2_residual_plot = residual_interp(s_plot_1d)
            v_fict_plot = fict_put.at_maturity(s_plot_1d)
            vtarget_plot = interp_vtarget(s_plot_1d)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(to_np(s_plot_1d), to_np(vtarget_plot),
                label=r"$V_{\mathrm{target}}(s, t_1)$", color="black", linewidth=2.5, linestyle="-.")
        ax.plot(to_np(s_plot_1d), to_np(v_fict_plot),
                label=f"$v(s, t_1) = {c_scale:.3f} \\cdot (s^* - s)^+$", color="green", linewidth=2, linestyle="--")
        ax.plot(to_np(s_plot_1d), to_np(g2_residual_plot),
                label=r"Residual $g_2(s) = V_{\mathrm{target}} - v$", color="blue", linewidth=2)
        ax.axvline(s_star, color="red", linestyle=":", alpha=0.7, label=f"$s^* \\approx {s_star:.1f}$")
        ax.set_title(f"Singularity extraction at $t_1 = {t1}$: $c = {c_scale:.4f}$")
        ax.set_xlabel("Asset Price $s$")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
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
            gamma_raw_wide = (raw_target(s_wide + h_fd) - 2 * raw_target(s_wide) + raw_target(s_wide - h_fd)) / h_fd**2
            gamma_raw_fine = (raw_target(s_fine + h_fd) - 2 * raw_target(s_fine) + raw_target(s_fine - h_fd)) / h_fd**2
            gamma_res_wide = (residual_interp(s_wide + h_fd) - 2 * residual_interp(s_wide) + residual_interp(s_wide - h_fd)) / h_fd**2
            gamma_res_fine = (residual_interp(s_fine + h_fd) - 2 * residual_interp(s_fine) + residual_interp(s_fine - h_fd)) / h_fd**2
            deriv_raw_fine = (raw_target(s_fine + h_fd) - raw_target(s_fine - h_fd)) / (2 * h_fd)
            deriv_res_fine = (residual_interp(s_fine + h_fd) - residual_interp(s_fine - h_fd)) / (2 * h_fd)

        fig, axes = plt.subplots(3, 1, figsize=(10, 14))
        axes[0].plot(to_np(s_fine), to_np(deriv_raw_fine),
                     label=r"$\partial V_{\mathrm{target}} / \partial s$ (has jump)", color="red", linewidth=2)
        axes[0].plot(to_np(s_fine), to_np(deriv_res_fine),
                     label=r"$\partial g_2 / \partial s$ ($C^1$ smooth)", color="blue", linewidth=2, linestyle="--")
        axes[0].axvline(s_star, color="grey", linestyle=":", alpha=0.5)
        axes[0].set_title(r"First derivative: jump in $\partial V_{\mathrm{target}}/\partial s$ removed by extraction")
        axes[0].set_xlabel("$s$"); axes[0].set_ylabel("$\\partial / \\partial s$")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(to_np(s_wide), to_np(gamma_raw_wide),
                     label=r"$\partial^2 V_{\mathrm{target}} / \partial s^2$ (Dirac spike)", color="red", linewidth=2)
        axes[1].plot(to_np(s_wide), to_np(gamma_res_wide),
                     label=r"$\partial^2 g_2 / \partial s^2$ (finite)", color="blue", linewidth=2, linestyle="--")
        axes[1].axvline(s_star, color="grey", linestyle=":", alpha=0.5)
        axes[1].set_title(r"Second derivative: Dirac $\delta$ singularity removed")
        axes[1].set_xlabel("$s$"); axes[1].set_ylabel("$\\partial^2 / \\partial s^2$")
        axes[1].set_ylim(-20, 20); axes[1].legend(); axes[1].grid(True, alpha=0.3)

        axes[2].plot(to_np(s_fine), to_np(gamma_raw_fine),
                     label=r"$\partial^2 V_{\mathrm{target}} / \partial s^2$", color="red", linewidth=2)
        axes[2].plot(to_np(s_fine), to_np(gamma_res_fine),
                     label=r"$\partial^2 g_2 / \partial s^2$", color="blue", linewidth=2, linestyle="--")
        axes[2].axvline(s_star, color="grey", linestyle=":", alpha=0.5)
        axes[2].set_title(r"Curvature near $s^*$ (zoomed)")
        axes[2].set_xlabel("$s$"); axes[2].set_ylabel("$\\partial^2 / \\partial s^2$")
        axes[2].legend(); axes[2].grid(True, alpha=0.3)

        fig.suptitle(
            f"Singularity extraction diagnostic: $c = {c_scale:.4f}$, $s^* = {s_star:.2f}$\n"
            f"(Bermudan Put, K={K}, r={r}, $\\sigma$={sigma})", fontsize=13,
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
        ax.plot(to_np(s_plot_1d), to_np(v_interp_plot), label="Interpolated $V(s, t_1)$", color="blue", linewidth=2)
        ax.plot(to_np(s_plot_1d), to_np(payoff_put(s_plot_1d, K)), label="Payoff $\\Phi(s)$", color="red", linestyle="--")
        ax.axvline(x=K, color="grey", linestyle=":", label="Strike K")
        if not np.isnan(s_star):
            ax.axvline(s_star, color="red", linestyle=":", alpha=0.7, label=f"$s^* \\approx {s_star:.1f}$")
        ax.set_title(f"Interpolated Function at $t_1 = {t1}$")
        ax.set_xlabel("Asset Price $s$"); ax.set_ylabel("Option Value")
        ax.legend(); ax.grid(True, alpha=0.3)
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
        axes[0].plot(to_np(s_wide), to_np(gamma_spline_wide), label=f"Curvature of $\\mathcal{{I}}(s)$ ({interp_name})", color="purple", linewidth=2)
        axes[0].plot(to_np(s_wide), to_np(gamma_raw_wide), label="Curvature of Raw Target $\\max(\\Phi, \\tilde{u}_{NN}^{(A)})$", color="green", linewidth=2, linestyle="--")
        if not np.isnan(s_star):
            axes[0].axvline(s_star, color="red", linestyle=":", alpha=0.7, label=f"Exercise boundary $s^* \\approx {s_star:.1f}$")
        axes[0].set_title(f"Test 1: {interp_name} Curvature around $s^*$ (Zoomed Out)")
        axes[0].set_xlabel("Asset Price $s$"); axes[0].set_ylabel("$[V(s+h) - 2V(s) + V(s-h)] / h^2$")
        axes[0].set_ylim(-20, 20); axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(to_np(s_fine), to_np(gamma_spline), label=f"Curvature of $\\mathcal{{I}}(s)$ ({interp_name})", color="purple", linewidth=2)
        axes[1].plot(to_np(s_fine), to_np(gamma_raw), label="Curvature of Raw Target $\\max(\\Phi, \\tilde{u}_{NN}^{(A)})$", color="green", linewidth=2, linestyle="--")
        if not np.isnan(s_star):
            axes[1].axvline(s_star, color="red", linestyle=":", alpha=0.7, label=f"Exercise boundary $s^* \\approx {s_star:.1f}$")
        axes[1].set_title(f"Test 1: {interp_name} Curvature around $s^*$ (Zoomed In)")
        axes[1].set_xlabel("Asset Price $s$"); axes[1].set_ylabel("$[V(s+h) - 2V(s) + V(s-h)] / h^2$")
        axes[1].set_ylim(-20, 20); axes[1].legend(); axes[1].grid(True, alpha=0.3)

        axes[2].plot(to_np(s_fine), to_np(v_center), label=f"$\\mathcal{{I}}(s)$ ({interp_name})", color="blue", linewidth=2)
        axes[2].plot(to_np(s_fine), to_np(v_raw_center), label="Raw Target $\\max(\\Phi, \\tilde{u}_{NN}^{(A)})$", color="orange", linewidth=2, linestyle="--")
        if not np.isnan(s_star):
            axes[2].axvline(s_star, color="red", linestyle=":", alpha=0.7, label=f"Exercise boundary $s^* \\approx {s_star:.1f}$")
        axes[2].set_title(f"Test 1: Function values around $s^*$ (Zoomed In)")
        axes[2].set_xlabel("Asset Price $s$"); axes[2].set_ylabel("Option Value")
        axes[2].legend(); axes[2].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "diagnostics" / "plotB1d_interpolant_gamma.png", dpi=150)
        plt.close(fig)
        logger.info(f"[OK] Plot B1d — {interp_name} Curvature")

    # === Plot B2 — Exercise boundary ===
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(to_np(s_plot), to_np(hold_plot), label=r"Hold: $\tilde{u}_{NN}^{(A)}(s, t_1)$", linewidth=2)
    ax.plot(to_np(s_plot), to_np(phi_plot), label=r"Exercise: $\Phi(s)$", linewidth=2, linestyle="--")
    if not np.isnan(s_star):
        ax.axvline(s_star, color="red", linewidth=2, linestyle="--",
                   label=f"Exercise boundary $s^* \\approx {s_star:.1f}$")
        s_star_t = torch.tensor([s_star])
        v_at_star = float(payoff_put(s_star_t, K))
        ax.plot(s_star, v_at_star, "ro", markersize=10, zorder=5)
        ax.annotate(f"$s^* = {s_star:.1f}$", (s_star, v_at_star),
                    textcoords="offset points", xytext=(15, 10), fontsize=11,
                    arrowprops=dict(arrowstyle="->", color="red"))
    ax.set_xlabel("s")
    ax.set_ylabel("Value")
    ax.set_title(f"Exercise boundary at $t_1 = {t1}$ (Bermudan Put Option, K={K}, r={r}, $\\sigma$={sigma}, T={T})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pricing" / "plotB2_exercise_boundary.png", dpi=150)
    plt.close(fig)
    logger.info(f"[OK] Plot B2 — Exercise boundary: s* ≈ {s_star:.2f}")

    # === Plot B3 — Stage A loss curves ===
    if hist_a is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].semilogy(hist_a["iter"], hist_a["loss_f"], label="$\\mathcal{L}_f$")
        axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("$\\mathcal{L}_f$")
        axes[0].set_title("Stage A: PDE residual"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[1].semilogy(hist_a["iter"], hist_a["loss_tc"], label="$\\mathcal{L}_{tc}$", color="tab:orange")
        axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("$\\mathcal{L}_{tc}$")
        axes[1].set_title("Stage A: Terminal loss"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
        fig.suptitle(f"Stage A loss curves (ETCNN$^{{(A)}}$ on $[t_1, T]$, Bermudan Put Option, K={K}, r={r}, $\\sigma$={sigma}, T={T})")
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

    if extraction:
        # Singularity extraction: U_B = v + ũ_θ
        logger.info(f"Stage C+D: training BermudaETCNN on [0, t1] for {iters_b} iterations ...")
        logger.info(f"  Ansatz: U_B(s,t) = v(s,t) + g1(s,t)·u_NN(s,t) + g2(s)")
        logger.info(f"  v(s,t) = {c_scale:.4f} · P^BS(s, {s_star:.2f}, r, σ, t1-t)")

        def g2_b(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return residual_interp(s)

        etcnn_residual = ETCNN(resnet=resnet_b, g1=g1_b, g2=g2_b, normalizer=normalizer_b)
        fict_put.to(DEVICE)
        etcnn_b = BermudaETCNN(etcnn=etcnn_residual, fictitious_put=fict_put)
        payoff_t1 = lambda s: interp_vtarget(s)
    else:
        # Classic: g2 = interpolated V_target
        logger.info(f"Stage C+D: training ETCNN_B on [0, t1] for {iters_b} iterations ...")

        def g2_b(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return v_interp_t1(s)

        etcnn_b = ETCNN(resnet=resnet_b, g1=g1_b, g2=g2_b, normalizer=normalizer_b)
        payoff_t1 = lambda s: v_interp_t1(s)

    hist_b = train_model(
        etcnn_b, iters_b, S_TRAIN_LO, S_TRAIN_HI, 0.0, t1,
        payoff_t1, label="ETCNN_B", weight_decay=weight_decay
    )
    plot_training_metrics(hist_b, "ETCNN_B", out_dir)

    # === Plot B4 — Stage B loss curves ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].semilogy(hist_b["iter"], hist_b["loss_f"], label="$L_f$")
    axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("$L_f$")
    axes[0].set_title("Stage D: PDE residual"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].semilogy(hist_b["iter"], hist_b["loss_tc"], label="$L_{tc}$", color="tab:orange")
    axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("$L_{tc}$")
    axes[1].set_title("Stage D: Terminal loss"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    fig.suptitle(f"Stage D loss curves (ETCNN$_B$ on $[0, t_1]$, Bermudan Put Option, K={K}, r={r}, $\\sigma$={sigma}, T={T})")
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
    s_eval_t = torch.tensor(s_eval_arr, dtype=torch.float32)
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
    ax.set_title(f"Bermudan Put vs European Put vs BT at $t=0$ (K={K}, r={r}, $\\sigma$={sigma}, T={T})")
    ax.legend()
    ax.grid(True, alpha=0.3)
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
    ax.set_title(f"Full Bermudan Put Option price surface (piecewise, K={K}, r={r}, $\\sigma$={sigma}, T={T})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pricing" / "plotB6_bermudan_surface.png", dpi=150)
    plt.close(fig)

    # Check continuity at t1
    t1_idx = int(nt * t1 / T)
    if t1_idx > 0 and t1_idx < nt:
        jump = float(torch.max(torch.abs(V_surface[:, t1_idx] - V_surface[:, t1_idx - 1])))
        logger.info(f"  Max jump at t1 intermediate boundary: {jump:.4e}")
    else:
        jump = 0.0
    logger.info("[OK] Plot B6 — Bermudan surface")

    # === Plot B7 — Error vs BT at t=0 ===
    err_vs_bt = np.abs(etcnn_b_prices - bt_prices)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(s_eval_arr, err_vs_bt, linewidth=2)
    ax.set_xlabel("s")
    ax.set_ylabel("$|$ETCNN$_B(s,0) - $BT$(s,0)|$")
    ax.set_title(f"Error vs binomial tree at $t=0$ (Bermudan Put Option, K={K}, r={r}, $\\sigma$={sigma}, T={T}, max={err_vs_bt.max():.2e})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "diagnostics" / "plotB7_error_vs_bt.png", dpi=150)
    plt.close(fig)
    logger.info(f"[OK] Plot B7 — Error vs BT: max={err_vs_bt.max():.2e}")

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
    
    fig.suptitle(f"Greeks at $t=0$ (Bermudan vs European Put Option, K={K}, r={r}, $\\sigma$={sigma}, T={T})")
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
    u_b_res = etcnn_b(x_res).squeeze()
    
    residual_t1 = bsm_operator(u_b_res, s_res, t_res, r, q, sigma)
    R_s = residual_t1 ** 2

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(to_np(s_res), to_np(R_s), label="PDE Residual $R(s)$", color="darkorange", linewidth=2)
    if not np.isnan(s_star):
        ax.axvline(s_star, color="red", linestyle=":", alpha=0.7, label=f"Exercise boundary $s^* \\approx {s_star:.1f}$")
    ax.set_title(f"Test II: Spatial Distribution of PDE Residual at $t_1^-$")
    ax.set_xlabel("Asset Price $s$")
    ax.set_ylabel("$R(s) = |\\mathcal{F}(\\tilde{u}_{NN}^{(B)})(s, t_1^-)|^2$")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "diagnostics" / "plotB9_test2_pde_residual.png", dpi=150)
    plt.close(fig)
    logger.info("[OK] Plot B9 — Test II: PDE Residual at t1-")

    # === Plot B10 — Test III: Neuron Weight Magnitudes ===
    logger.info("Running Test III: Neuron Weight Magnitudes in ETCNN_A ...")
    weights = []
    layer_names = []

    # Input layer (Sequential: Linear -> Tanh)
    w = etcnn_a.resnet.input_layer[0].weight.detach().cpu().numpy().flatten()
    weights.append(w)
    layer_names.append("Input")

    # Blocks
    for i, block in enumerate(etcnn_a.resnet.blocks):
        for j, layer in enumerate(block.layers):
            if isinstance(layer, torch.nn.Linear):
                w = layer.weight.detach().cpu().numpy().flatten()
                weights.append(w)
                layer_names.append(f"B{i} L{j//2}")

    # Output layer
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
    ax.set_title("Test III: Neuron Weight Magnitudes in ETCNN$^{(A)}$\nLarge weights ($|w| > 1$) amplify high-frequency noise via $w^2$ in the second derivative")
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

    torch.save(etcnn_a.state_dict(), out_dir / "etcnn_a.pt")
    torch.save(etcnn_b.state_dict(), out_dir / "etcnn_b.pt")
    logger.info("  Saved models: etcnn_a.pt, etcnn_b.pt")

    return {
        "rel_l2_bt": rel_l2_bt,
        "mae_bt": mae_bt,
        "s_star": s_star,
        "c_scale": c_scale,
        "bermuda_at_K": bermuda_at_K,
        "etcnn_b_at_K": etcnn_b_at_K,
        "euro_at_K": euro_at_K,
        "jump_at_t1": jump,
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
        help="Interpolation for Bermudan V(s,t1) when --extraction is NOT set: "
             "'cubic' (C^2, default), 'pchip' (C^1, shape-preserving), or 'linear' (C^0)",
    )
    parser.add_argument(
        "--extraction", action="store_true",
        help="Enable singularity extraction ansatz for Stage B (default: off). "
             "Decomposes U_B = v + u_tilde, removing the C^0 kink at s*.",
    )
    parser.add_argument(
        "--g2", type=str, default="taylor", choices=["taylor", "bs"],
        help="Terminal function g2 for ETCNN: 'taylor' (V1^e + V2^e, default) "
             "or 'bs' (exact Black-Scholes European put)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device: auto (CUDA if available), cuda (fail if unavailable), or cpu",
    )
    parser.add_argument("--bermudan-only", action="store_true", help="Skip European problem and only run Bermudan")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="L2 regularization penalty for Adam optimizer")
    parser.add_argument("--load-etcnn-a", type=str, default=None, help="Path to pre-trained etcnn_a.pt to skip Stage A training")
    parser.add_argument("--n-tc", type=int, default=1024, help="Number of terminal condition points")
    parser.add_argument("--n-f", type=int, default=4096, help="Number of interior PDE collocation points")
    args = parser.parse_args()
    _apply_device_arg(args.device)

    # Update global hyperparameters if provided
    global N_TC, N_F
    if args.n_tc is not None:
        N_TC = args.n_tc
    if args.n_f is not None:
        N_F = args.n_f

    # Output directory
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    iters_str = "_".join(map(str, args.iters))
    mode_tag = "extraction" if args.extraction else f"interp-{args.interp}"
    out_dir = Path("data/phase3_training") / (
        f"{timestamp}_iters{iters_str}_K{K:.0f}_{mode_tag}_g2-{args.g2}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for grouped plots
    (out_dir / "training_metrics").mkdir(exist_ok=True)
    (out_dir / "pricing").mkdir(exist_ok=True)
    (out_dir / "greeks").mkdir(exist_ok=True)
    (out_dir / "diagnostics").mkdir(exist_ok=True)

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
            "extraction": args.extraction,
            "interp_method": args.interp,
            "g2_type": args.g2,
            "device": args.device,
        },
        "domain": {
            "S_TRAIN_LO": S_TRAIN_LO,
            "S_TRAIN_HI": S_TRAIN_HI,
            "S_EVAL_LO": S_EVAL_LO,
            "S_EVAL_HI": S_EVAL_HI,
        }
    }
    with open(out_dir / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

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
    logger.info(f"Phase 3 — ETCNN Training")
    logger.info(f"  Output: {out_dir}")
    logger.info(f"  Command: {' '.join(sys.argv)}")
    logger.info(f"  Iters: {args.iters}")
    if args.extraction:
        logger.info(f"  Stage B method: singularity extraction ansatz (PCHIP residual)")
    else:
        logger.info(f"  Stage B method: {args.interp} interpolation of V(s, t1)")
    logger.info(f"  g2 type (ETCNN terminal): {args.g2}")
    logger.info(f"  Device (requested): {args.device}")
    logger.info(f"  Parameters: K={K}, r={r}, sigma={sigma}, T={T}, q={q}, t1={t1}")
    logger.info(f"  Hyperparameters: M={M}, L_BLOCK={L_BLOCK}, n={n}, N_TC={N_TC}, N_F={N_F}, LAMBDA_F={LAMBDA_F}, LAMBDA_TC={LAMBDA_TC}, SEED={SEED}")
    logger.info(f"  PyTorch: {torch.__version__}")
    logger.info(f"  Device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))

    # Run problems
    load_path = Path(args.load_etcnn_a) if args.load_etcnn_a else None
    
    if not args.bermudan_only:
        eur_results = european_problem(
            out_dir,
            args.iters[0],
            weight_decay=args.weight_decay,
            g2_type=args.g2,
        )
    else:
        eur_results = None

    ber_results = bermudan_problem(
        out_dir,
        args.iters,
        interp_method=args.interp,
        extraction=args.extraction,
        weight_decay=args.weight_decay,
        load_etcnn_a=load_path,
        g2_type=args.g2,
    )

    # === Joint summary ===
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 3 — JOINT SUMMARY")
    logger.info("=" * 70)
    if eur_results:
        logger.info(f"  European ETCNN rel L2:     {eur_results['rel_l2_etcnn']:.6e}")
        logger.info(f"  European PINN  rel L2:     {eur_results['rel_l2_pinn']:.6e}")
    logger.info(f"  Bermudan ETCNN rel L2 (vs BT): {ber_results['rel_l2_bt']:.6e}")
    logger.info(f"  Bermudan at s=100:         {ber_results['etcnn_b_at_K']:.4f}")
    logger.info(f"  European at s=100:         {ber_results['euro_at_K']:.4f}")
    logger.info(f"  Bermudan >= European?      {ber_results['etcnn_b_at_K'] >= ber_results['euro_at_K'] - 1e-4}")
    logger.info(f"  Max jump at t1:            {ber_results['jump_at_t1']:.4e}")
    logger.info("=" * 70)

    # Pass/fail
    eur_pass = eur_results["rel_l2_etcnn"] < 5e-4 if eur_results else True
    ber_ordering = ber_results["etcnn_b_at_K"] >= ber_results["euro_at_K"] - 1e-4
    ber_continuity = ber_results["jump_at_t1"] < 1e-2

    if eur_pass and ber_ordering and ber_continuity:
        logger.info("\n  All Phase 3 checks PASSED. Ready for Phase 4.")
    else:
        logger.info("\n  Some checks failed:")
        if not eur_pass and eur_results:
            logger.info(f"    - European L2 error {eur_results['rel_l2_etcnn']:.2e} > 5e-4")
        if not ber_ordering:
            logger.info("    - Bermudan < European (price ordering violated)")
        if not ber_continuity:
            logger.info(f"    - Intermediate jump {ber_results['jump_at_t1']:.2e} > 1e-2")

    logger.info(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
