"""Convergence test : VPINN-ENGD vs VPINN-Adam.

Sanity-check setup:

* loss = VPINN weak residual loss (no initial / terminal condition).
* The trivial function $u \\equiv 0$ satisfies the PDE on the interior, so
  the loss can be driven to zero by a sufficiently expressive network
  (or by simply collapsing to zero) — we only care about the *speed* of
  convergence, not the final solution quality.

For a more realistic comparison (with initial condition) one would add an
IC penalty or use an ETCNN ansatz; that's out of scope for this script.

Outputs (under ``data/exp_engd/<timestamp>_vpinn/``):

* ``run_info.yaml``       — config + final loss
* ``loss_curves.png``     — VPINN loss vs iterations and vs wall time
* ``run.log``             — per-iteration log

Usage::

    python experiments/python_scripts/exp_engd/convergence_vpinn.py \\
        --iters-engd 60 --iters-adam 5000
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from learning_option_pricing.optimizers import VPINNENGDOptimizer, flat_grad
from learning_option_pricing.vpinn import VPINNLoss


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("convergence_vpinn")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class TanhMLP(torch.nn.Module):
    """Simple fully-connected tanh MLP — used as the underlying $u_\\theta$."""

    def __init__(self, hidden: int, depth: int):
        super().__init__()
        layers: list[torch.nn.Module] = [torch.nn.Linear(2, hidden)]
        for _ in range(depth - 1):
            layers += [torch.nn.Tanh(), torch.nn.Linear(hidden, hidden)]
        layers += [torch.nn.Tanh(), torch.nn.Linear(hidden, 1)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_model(hidden: int, depth: int, seed: int = 7) -> TanhMLP:
    torch.manual_seed(seed)
    return TanhMLP(hidden=hidden, depth=depth)


# ---------------------------------------------------------------------------
# Trainers
# ---------------------------------------------------------------------------


def train_adam(model, vpinn_loss, tau_batch, n_iters, lr, log_every):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"iter": [], "wall_time": [], "loss": []}
    t0 = time.time()
    history["iter"].append(0)
    history["wall_time"].append(0.0)
    history["loss"].append(vpinn_loss(model, tau_batch).item())
    logger.info(f"[Adam] iter=0  loss={history['loss'][0]:.3e}")

    for it in range(1, n_iters + 1):
        opt.zero_grad()
        L = vpinn_loss(model, tau_batch)
        L.backward()
        opt.step()
        if it % log_every == 0 or it == n_iters:
            history["iter"].append(it)
            history["wall_time"].append(time.time() - t0)
            history["loss"].append(L.item())
            logger.info(
                f"[Adam] iter={it}  loss={L.item():.3e}  "
                f"({history['wall_time'][-1]:.1f}s)"
            )
    return history


def train_engd(model, vpinn_loss, tau_batch, tau_gram, n_iters,
               reg, cg_iters, ls_steps, ls_step_max, log_every):
    engd = VPINNENGDOptimizer(
        model, vpinn_loss, reg=reg,
        cg_iters=cg_iters, ls_steps=ls_steps, ls_step_max=ls_step_max,
    )
    history = {
        "iter": [], "wall_time": [], "loss": [],
        "step_size": [], "cg_residual_norm": [], "J_norm": [],
    }
    t0 = time.time()
    history["iter"].append(0)
    history["wall_time"].append(0.0)
    history["loss"].append(vpinn_loss(model, tau_batch).item())
    history["step_size"].append(float("nan"))
    history["cg_residual_norm"].append(float("nan"))
    history["J_norm"].append(float("nan"))
    logger.info(f"[ENGD] iter=0  loss={history['loss'][0]:.3e}")

    for it in range(1, n_iters + 1):
        model.zero_grad()
        L = vpinn_loss(model, tau_batch)
        L.backward()
        g = flat_grad(model)
        info = engd.step(g, tau_gram, lambda: vpinn_loss(model, tau_batch))
        if it % log_every == 0 or it == 1 or it == n_iters:
            new_L = vpinn_loss(model, tau_batch).item()
            history["iter"].append(it)
            history["wall_time"].append(time.time() - t0)
            history["loss"].append(new_L)
            history["step_size"].append(info["step_size"])
            history["cg_residual_norm"].append(info["cg_residual_norm"])
            history["J_norm"].append(info["J_norm"])
            logger.info(
                f"[ENGD] iter={it}  loss={new_L:.3e}  alpha={info['step_size']:.2e}  "
                f"CG_res={info['cg_residual_norm']:.2e}  ({history['wall_time'][-1]:.1f}s)"
            )
    return history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters-engd", type=int, default=60)
    parser.add_argument("--iters-adam", type=int, default=4000)
    parser.add_argument("--hidden", type=int, default=20)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--n-tau", type=int, default=12)
    parser.add_argument("--K-test", type=int, default=8)
    parser.add_argument("--n-quad", type=int, default=24)
    parser.add_argument("--x-max", type=float, default=1.5)
    parser.add_argument("--reg", type=float, default=1e-3)
    parser.add_argument("--cg-iters", type=int, default=120)
    parser.add_argument("--ls-steps", type=int, default=30)
    parser.add_argument("--ls-step-max", type=float, default=1.0)
    parser.add_argument("--lr-adam", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(__file__).resolve().parents[3] / "data" / "exp_engd" / f"{ts}_vpinn"
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

    # VPINN setup
    vpinn_loss = VPINNLoss(
        sigma=0.25, r=0.02, q=0.0,
        x_max=args.x_max, K_test=args.K_test, n_quad=args.n_quad,
        dtype=torch.float64,
    )
    tau_batch = torch.linspace(0.05, 0.95, args.n_tau)
    tau_gram = tau_batch.clone()  # use the same set for the Gram

    # ============================ ENGD ===============================
    logger.info("=" * 60 + "\nTraining VPINN-ENGD\n" + "=" * 60)
    model_engd = make_model(args.hidden, args.depth, args.seed)
    n_params = sum(p.numel() for p in model_engd.parameters())
    logger.info(f"n_params = {n_params}, K*N_tau = {args.K_test * args.n_tau}")

    h_engd = train_engd(
        model_engd, vpinn_loss, tau_batch, tau_gram,
        args.iters_engd, args.reg, args.cg_iters,
        args.ls_steps, args.ls_step_max,
        log_every=max(1, args.iters_engd // 30),
    )

    # ============================ ADAM ===============================
    logger.info("=" * 60 + "\nTraining VPINN-Adam\n" + "=" * 60)
    model_adam = make_model(args.hidden, args.depth, args.seed)
    h_adam = train_adam(
        model_adam, vpinn_loss, tau_batch, args.iters_adam, args.lr_adam,
        log_every=max(1, args.iters_adam // 50),
    )

    # ---- Save ----
    summary = {
        "config": vars(args),
        "n_params": n_params,
        "engd_final": {
            "loss": h_engd["loss"][-1],
            "wall_time": h_engd["wall_time"][-1],
            "iters": h_engd["iter"][-1],
        },
        "adam_final": {
            "loss": h_adam["loss"][-1],
            "wall_time": h_adam["wall_time"][-1],
            "iters": h_adam["iter"][-1],
        },
    }
    with open(out_dir / "run_info.yaml", "w") as f:
        yaml.safe_dump(summary, f)
    with open(out_dir / "history_engd.yaml", "w") as f:
        yaml.safe_dump(h_engd, f)
    with open(out_dir / "history_adam.yaml", "w") as f:
        yaml.safe_dump(h_adam, f)
    logger.info("Summary:\n" + yaml.safe_dump(summary, indent=2))

    # ---- Plots ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].semilogy(h_engd["iter"], h_engd["loss"], "o-", label="VPINN-ENGD", color="C0", lw=2)
    axes[0].semilogy(h_adam["iter"], h_adam["loss"], "-", label="VPINN-Adam", color="C1", lw=1.5)
    axes[0].set_xlabel("iteration"); axes[0].set_ylabel("VPINN loss")
    axes[0].grid(True, alpha=0.3, which="both"); axes[0].legend()
    axes[0].set_title("Convergence vs iterations")
    axes[1].semilogy(h_engd["wall_time"], h_engd["loss"], "o-", label="VPINN-ENGD", color="C0", lw=2)
    axes[1].semilogy(h_adam["wall_time"], h_adam["loss"], "-", label="VPINN-Adam", color="C1", lw=1.5)
    axes[1].set_xlabel("wall time (s)"); axes[1].set_ylabel("VPINN loss")
    axes[1].grid(True, alpha=0.3, which="both"); axes[1].legend()
    axes[1].set_title("Convergence vs wall time")
    fig.suptitle("VPINN-ENGD vs VPINN-Adam (no BC term)")
    fig.text(0.5, -0.02, (
        f"net=TanhMLP(hidden={args.hidden}, depth={args.depth}); n_params={n_params}\n"
        f"VPINN: K_test={args.K_test}, n_quad={args.n_quad}, x_max={args.x_max}, "
        f"N_tau={args.n_tau}\n"
        f"ENGD: reg={args.reg}, cg_iters={args.cg_iters}; Adam: lr={args.lr_adam}"
    ), ha="center", fontsize=8, family="monospace",
       bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curves.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
