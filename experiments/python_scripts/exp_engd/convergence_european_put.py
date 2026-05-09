"""Convergence study : ENGD vs Adam on the European put.

The European put has a closed-form Black-Scholes solution; we use it as
ground truth to compute an L2 error on a fixed validation grid:

    err(theta) = || u_theta - V_BS ||_{L^2(Omega_eval)}

We follow the JAX paper's deterministic-collocation convention: a fixed
tensor-product grid in (s, t) is used for the loss, and the Gram matrix
points are a fixed sub-grid (NOT resampled at each step). This is critical
for ENGD stability — stochastic resampling injects too much noise into the
preconditioner.

Outputs (in ``data/exp_engd/<timestamp>_european_put/``):

* ``history_engd.yaml``,  ``history_adam.yaml``  — per-iteration metrics
* ``convergence_iters.png``                      — L2 error vs iterations
* ``convergence_time.png``                       — L2 error vs wall time
* ``loss_curves.png``                            — train loss curves
* ``run_info.yaml``                              — config and final metrics

Usage::

    python experiments/python_scripts/exp_engd/convergence_european_put.py \\
        --iters-engd 100 --iters-adam 5000 --hidden 16

The default budget is light enough to finish in a few minutes on CPU.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml

# package import
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from learning_option_pricing.models.etcnn import InputNormalization
from learning_option_pricing.models.resnet import ResNet
from learning_option_pricing.optimizers import ENGDOptimizer, flat_grad
from learning_option_pricing.pricing.terminal import (
    black_scholes_put,
    bsm_operator,
    payoff_put,
)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("convergence_european_put")


# =========================================================================
# Problem definition (Section 4.1.2 of the paper)
# =========================================================================

CFG = dict(
    K=100.0,
    r=0.02,
    sigma=0.25,
    T=1.0,
    q=0.0,
    S_TRAIN_LO=20.0,
    S_TRAIN_HI=160.0,
    S_EVAL_LO=60.0,
    S_EVAL_HI=140.0,
)


# =========================================================================
# Helpers
# =========================================================================


class NormalizedPINN(torch.nn.Module):
    """ResNet with input normalisation s -> s/K  (t left untouched).

    This avoids tanh saturation when the asset price is fed in raw units.
    The PINN baseline in ``phase3_training.py`` follows the same convention.
    """

    def __init__(self, K: float, hidden: int, blocks: int, layers_per_block: int):
        super().__init__()
        self.norm = InputNormalization(K=K)
        self.net = ResNet(d_in=2, d_out=1, n=hidden, M=blocks, L=layers_per_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x))


def make_model(hidden: int, blocks: int, layers_per_block: int, seed: int = 7):
    """Build a ResNet with default PyTorch init (Kaiming-uniform) and s/K
    input normalisation."""
    torch.manual_seed(seed)
    return NormalizedPINN(CFG["K"], hidden, blocks, layers_per_block)


def build_fixed_grid(n_s: int, n_t: int, T: float):
    """Tensor-product grid in (s, t) for the *training* collocation."""
    s_lin = torch.linspace(CFG["S_TRAIN_LO"], CFG["S_TRAIN_HI"], n_s)
    t_lin = torch.linspace(0.01, T - 0.01, n_t)  # avoid t=0, t=T endpoints
    SS, TT = torch.meshgrid(s_lin, t_lin, indexing="ij")
    return SS.flatten().clone(), TT.flatten().clone()


def build_terminal_points(n_tc: int, T: float):
    s_tc = torch.linspace(CFG["S_TRAIN_LO"], CFG["S_TRAIN_HI"], n_tc)
    t_tc = torch.full((n_tc,), T)
    return s_tc, t_tc


def build_eval_grid(n_s: int = 200, n_t: int = 50, T: float = 1.0):
    """Validation grid — never used for training."""
    s_lin = torch.linspace(CFG["S_EVAL_LO"], CFG["S_EVAL_HI"], n_s)
    t_lin = torch.linspace(0.05, T - 0.05, n_t)
    SS, TT = torch.meshgrid(s_lin, t_lin, indexing="ij")
    return SS.flatten().clone(), TT.flatten().clone()


def reference_solution(s: torch.Tensor, t: torch.Tensor, K: float, r: float,
                       sigma: float, T: float):
    return black_scholes_put(s, K, r, sigma, T - t)


def make_loss_fn(s_f, t_f, s_tc, t_tc, K, r, q, sigma, lam_f, lam_tc):
    """Closure capturing fixed collocation tensors."""
    phi = payoff_put(s_tc, K)

    def loss(model: torch.nn.Module) -> tuple[torch.Tensor, float, float]:
        s = s_f.detach().clone().requires_grad_(True)
        t = t_f.detach().clone().requires_grad_(True)
        V = model(torch.stack([s, t], dim=1)).squeeze()
        F = bsm_operator(V, s, t, r, q, sigma)
        V_tc = model(torch.stack([s_tc, t_tc], dim=1)).squeeze()
        l_f = F.pow(2).mean()
        l_tc = (V_tc - phi).pow(2).mean()
        total = lam_f * l_f + lam_tc * l_tc
        return total, l_f.item(), l_tc.item()

    return loss


def l2_error(model: torch.nn.Module, s_eval, t_eval, v_ref):
    with torch.no_grad():
        x = torch.stack([s_eval, t_eval], dim=1)
        v = model(x).squeeze()
        rel = ((v - v_ref).pow(2).mean().sqrt() / v_ref.pow(2).mean().sqrt()).item()
        abs_ = (v - v_ref).pow(2).mean().sqrt().item()
    return abs_, rel


# =========================================================================
# Trainers
# =========================================================================


def train_adam(
    model, loss_fn, n_iters, lr, eval_grid, v_ref, log_every, out_log
):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"iter": [], "wall_time": [], "loss": [], "l2_abs": [], "l2_rel": []}
    t0 = time.time()
    s_eval, t_eval = eval_grid
    abs0, rel0 = l2_error(model, s_eval, t_eval, v_ref)
    history["iter"].append(0)
    history["wall_time"].append(0.0)
    history["loss"].append(loss_fn(model)[0].item())
    history["l2_abs"].append(abs0)
    history["l2_rel"].append(rel0)
    out_log.info(f"[Adam] iter=0  L2_abs={abs0:.3e}  L2_rel={rel0:.3e}")

    for it in range(1, n_iters + 1):
        opt.zero_grad()
        loss, _, _ = loss_fn(model)
        loss.backward()
        opt.step()

        if it % log_every == 0 or it == n_iters:
            ab, re = l2_error(model, s_eval, t_eval, v_ref)
            history["iter"].append(it)
            history["wall_time"].append(time.time() - t0)
            history["loss"].append(loss.item())
            history["l2_abs"].append(ab)
            history["l2_rel"].append(re)
            out_log.info(
                f"[Adam] iter={it}  loss={loss.item():.3e}  "
                f"L2_abs={ab:.3e}  L2_rel={re:.3e}  "
                f"({history['wall_time'][-1]:.1f}s)"
            )

    return history


def train_engd(
    model, loss_fn_full, s_gram, t_gram, s_tc_gram, t_tc_gram,
    n_iters, lam_f, lam_tc, K_, r_, q_, sigma_,
    reg, cg_iters, ls_steps, ls_step_max,
    eval_grid, v_ref, log_every, out_log,
):
    engd = ENGDOptimizer(
        model, r=r_, q=q_, sigma=sigma_, lam_f=lam_f, lam_tc=lam_tc,
        reg=reg, cg_iters=cg_iters, ls_steps=ls_steps, ls_step_max=ls_step_max,
    )
    history = {
        "iter": [], "wall_time": [], "loss": [], "l2_abs": [], "l2_rel": [],
        "step_size": [], "cg_residual_norm": [], "J_F_norm": [],
    }
    t0 = time.time()
    s_eval, t_eval = eval_grid
    ab0, re0 = l2_error(model, s_eval, t_eval, v_ref)
    history["iter"].append(0)
    history["wall_time"].append(0.0)
    history["loss"].append(loss_fn_full(model)[0].item())
    history["l2_abs"].append(ab0)
    history["l2_rel"].append(re0)
    history["step_size"].append(float("nan"))
    history["cg_residual_norm"].append(float("nan"))
    history["J_F_norm"].append(float("nan"))
    out_log.info(f"[ENGD] iter=0  L2_abs={ab0:.3e}  L2_rel={re0:.3e}")

    for it in range(1, n_iters + 1):
        model.zero_grad()
        loss, _, _ = loss_fn_full(model)
        loss.backward()
        g = flat_grad(model)

        info = engd.step(
            g, s_gram, t_gram, s_tc_gram, t_tc_gram,
            lambda: loss_fn_full(model)[0],
        )

        if it % log_every == 0 or it == 1 or it == n_iters:
            ab, re = l2_error(model, s_eval, t_eval, v_ref)
            history["iter"].append(it)
            history["wall_time"].append(time.time() - t0)
            history["loss"].append(loss.item())
            history["l2_abs"].append(ab)
            history["l2_rel"].append(re)
            history["step_size"].append(info["step_size"])
            history["cg_residual_norm"].append(info["cg_residual_norm"])
            history["J_F_norm"].append(info["J_F_norm"])
            out_log.info(
                f"[ENGD] iter={it}  loss={loss.item():.3e}  "
                f"L2_abs={ab:.3e}  L2_rel={re:.3e}  "
                f"alpha={info['step_size']:.2e}  CG_res={info['cg_residual_norm']:.2e}  "
                f"({history['wall_time'][-1]:.1f}s)"
            )

    return history


# =========================================================================
# Plots
# =========================================================================


def plot_convergence(h_engd, h_adam, out_dir: Path, hyp_box: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].semilogy(h_engd["iter"], h_engd["l2_abs"], "o-", label="ENGD", color="C0", lw=2)
    axes[0].semilogy(h_adam["iter"], h_adam["l2_abs"], "-", label="Adam", color="C1", lw=1.5)
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel(r"$\|u_\theta - V^{BS}\|_{L^2}$  (absolute)")
    axes[0].set_title("Convergence vs. iterations")
    axes[0].grid(True, alpha=0.3, which="both")
    axes[0].legend()

    axes[1].semilogy(h_engd["wall_time"], h_engd["l2_abs"], "o-", label="ENGD", color="C0", lw=2)
    axes[1].semilogy(h_adam["wall_time"], h_adam["l2_abs"], "-", label="Adam", color="C1", lw=1.5)
    axes[1].set_xlabel("wall time (s)")
    axes[1].set_ylabel(r"$\|u_\theta - V^{BS}\|_{L^2}$  (absolute)")
    axes[1].set_title("Convergence vs. wall time")
    axes[1].grid(True, alpha=0.3, which="both")
    axes[1].legend()

    fig.suptitle("ENGD vs Adam — European put (deterministic grid)", y=1.02)
    fig.text(0.5, -0.02, hyp_box, ha="center", fontsize=8, family="monospace",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    fig.tight_layout()
    fig.savefig(out_dir / "convergence.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(h_engd, h_adam, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(h_engd["iter"], h_engd["loss"], "o-", label="ENGD")
    ax.semilogy(h_adam["iter"], h_adam["loss"], "-", label="Adam")
    ax.set_xlabel("iteration")
    ax.set_ylabel("training loss  $L = \\lambda_F \\cdot L_F + \\lambda_{TC} \\cdot L_{TC}$")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    ax.set_title("Training loss")
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curves.png", dpi=140)
    plt.close(fig)


def plot_engd_diagnostics(h_engd, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    iters = h_engd["iter"][1:]  # skip the iter=0 NaN
    axes[0].semilogy(iters, h_engd["step_size"][1:], "o-")
    axes[0].set_xlabel("iter"); axes[0].set_ylabel(r"$\alpha$ (line search)")
    axes[0].grid(alpha=0.3, which="both"); axes[0].set_title("Step size")
    axes[1].semilogy(iters, h_engd["cg_residual_norm"][1:], "o-", color="C2")
    axes[1].set_xlabel("iter"); axes[1].set_ylabel(r"$\|G\delta - g\|_2$")
    axes[1].grid(alpha=0.3, which="both"); axes[1].set_title("CG residual")
    axes[2].plot(iters, h_engd["J_F_norm"][1:], "o-", color="C3")
    axes[2].set_xlabel("iter"); axes[2].set_ylabel(r"$\|J_F\|_F$")
    axes[2].grid(alpha=0.3); axes[2].set_title("PDE Jacobian norm")
    fig.tight_layout()
    fig.savefig(out_dir / "engd_diagnostics.png", dpi=140)
    plt.close(fig)


# =========================================================================
# Main
# =========================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters-engd", type=int, default=80)
    parser.add_argument("--iters-adam", type=int, default=4000)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--n-s", type=int, default=20, help="grid size in s")
    parser.add_argument("--n-t", type=int, default=20, help="grid size in t")
    parser.add_argument("--n-tc", type=int, default=64)
    parser.add_argument("--n-gram", type=int, default=64)
    parser.add_argument("--n-tc-gram", type=int, default=32)
    parser.add_argument("--lam-f", type=float, default=1.0)
    parser.add_argument("--lam-tc", type=float, default=10.0)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--cg-iters", type=int, default=50)
    parser.add_argument("--ls-steps", type=int, default=30)
    parser.add_argument("--ls-step-max", type=float, default=1.0)
    parser.add_argument("--lr-adam", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    # Output directory
    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(__file__).resolve().parents[3] / "data" / "exp_engd" / f"{ts}_european_put"
    else:
        out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(out_dir / "run.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    logger.info(f"Output: {out_dir}")
    logger.info(f"Args: {vars(args)}")

    torch.set_default_dtype(torch.float64)

    # ---- Collocation: deterministic grid ----
    s_f, t_f = build_fixed_grid(args.n_s, args.n_t, CFG["T"])
    s_tc, t_tc = build_terminal_points(args.n_tc, CFG["T"])
    logger.info(f"Interior collocation: {s_f.numel()}  Terminal: {s_tc.numel()}")

    # Sub-grid for Gram (fixed throughout training)
    s_gram = s_f[: args.n_gram].clone()
    t_gram = t_f[: args.n_gram].clone()
    s_tc_gram = s_tc[: args.n_tc_gram].clone()
    t_tc_gram = t_tc[: args.n_tc_gram].clone()

    # ---- Eval grid + reference ----
    s_eval, t_eval = build_eval_grid(T=CFG["T"])
    v_ref = reference_solution(
        s_eval, t_eval, CFG["K"], CFG["r"], CFG["sigma"], CFG["T"]
    )

    # ---- Loss factory ----
    loss_fn = make_loss_fn(
        s_f, t_f, s_tc, t_tc, CFG["K"], CFG["r"], CFG["q"], CFG["sigma"],
        args.lam_f, args.lam_tc,
    )

    # ============================ ADAM ===============================
    logger.info("=" * 60)
    logger.info("Training ADAM")
    logger.info("=" * 60)
    model_adam = make_model(args.hidden, args.blocks, args.layers, args.seed)
    n_params = sum(p.numel() for p in model_adam.parameters())
    logger.info(f"n_params = {n_params}")

    h_adam = train_adam(
        model_adam, loss_fn, args.iters_adam, args.lr_adam,
        (s_eval, t_eval), v_ref,
        log_every=max(1, args.iters_adam // 50),
        out_log=logger,
    )

    # ============================ ENGD ===============================
    logger.info("=" * 60)
    logger.info("Training ENGD")
    logger.info("=" * 60)
    model_engd = make_model(args.hidden, args.blocks, args.layers, args.seed)

    h_engd = train_engd(
        model_engd, loss_fn,
        s_gram, t_gram, s_tc_gram, t_tc_gram,
        args.iters_engd,
        args.lam_f, args.lam_tc,
        CFG["K"], CFG["r"], CFG["q"], CFG["sigma"],
        args.reg, args.cg_iters, args.ls_steps, args.ls_step_max,
        (s_eval, t_eval), v_ref,
        log_every=max(1, args.iters_engd // 30),
        out_log=logger,
    )

    # ---- Save ----
    with open(out_dir / "history_adam.yaml", "w") as f:
        yaml.safe_dump(h_adam, f)
    with open(out_dir / "history_engd.yaml", "w") as f:
        yaml.safe_dump(h_engd, f)
    summary = {
        "config": vars(args),
        "n_params": n_params,
        "engd_final": {
            "L2_abs": h_engd["l2_abs"][-1],
            "L2_rel": h_engd["l2_rel"][-1],
            "loss": h_engd["loss"][-1],
            "wall_time": h_engd["wall_time"][-1],
        },
        "adam_final": {
            "L2_abs": h_adam["l2_abs"][-1],
            "L2_rel": h_adam["l2_rel"][-1],
            "loss": h_adam["loss"][-1],
            "wall_time": h_adam["wall_time"][-1],
        },
    }
    with open(out_dir / "run_info.yaml", "w") as f:
        yaml.safe_dump(summary, f)
    logger.info("Final summary:\n" + json.dumps(summary, indent=2))

    # ---- Plots ----
    hyp = (
        f"net=ResNet(M={args.blocks},L={args.layers},n={args.hidden})  n_params={n_params}\n"
        f"grid={args.n_s}x{args.n_t}={args.n_s*args.n_t} interior, {args.n_tc} TC; "
        f"n_gram={args.n_gram}, n_tc_gram={args.n_tc_gram}\n"
        f"loss: lam_F={args.lam_f}  lam_TC={args.lam_tc}\n"
        f"ENGD: reg={args.reg}, cg_iters={args.cg_iters}, "
        f"ls_steps={args.ls_steps}, ls_step_max={args.ls_step_max}\n"
        f"Adam: lr={args.lr_adam}"
    )
    plot_convergence(h_engd, h_adam, out_dir, hyp)
    plot_loss_curves(h_engd, h_adam, out_dir)
    plot_engd_diagnostics(h_engd, out_dir)
    logger.info(f"Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
