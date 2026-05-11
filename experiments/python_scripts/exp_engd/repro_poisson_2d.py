"""Reproduction of the Poisson 2D benchmark from Zeinhofer et al. (ICML 2023).

Original JAX script:
    https://github.com/MariusZeinhofer/Natural-Gradient-PINNs-ICML23/blob/main/engd_poisson_2d.py

PDE
---
$-\\Delta u(x,y) = 2\\pi^2 \\sin(\\pi x)\\sin(\\pi y)$ on $\\Omega=(0,1)^2$
$u = 0$ on $\\partial\\Omega$
Exact solution: $u^*(x,y) = \\sin(\\pi x)\\sin(\\pi y)$.

Setup (matches original verbatim)
---------------------------------
* Network: MLP `[2, 32, 1]` with tanh, **129 parameters**.
* Interior grid: `Square(1.).deterministic_integration_points(30)` →
  28 × 28 = **784 points** (strict interior, excluding boundary).
* Boundary grid: `SquareBoundary(1.).deterministic_integration_points(30)` →
  4 × 29 = **116 points** (no double corners).
* Loss weights: `lam_F=1, lam_TC=4` (the boundary integrator multiplies by
  `|∂Ω|=4` whereas the interior multiplies by `|Ω|=1`).
* Solve via `torch.linalg.lstsq` on the **explicit** Gram matrix
  $G = \\frac{1}{N_F}J_F^\\top J_F + \\frac{4}{N_{TC}}J_{TC}^\\top J_{TC}$
  (no Tikhonov regularisation — same as the original).
* Line search grid: $\\alpha_k = 0.5^k$ for $k \\in \\{0,\\dots,30\\}$.
* Total iterations: 51.

Purpose
-------
Sanity-check that our PyTorch port of the ENGD building blocks
(`measurement_jacobian`, parameter helpers) produces the same convergence
behaviour as the original JAX implementation on a problem where ENGD is
known to work well (M=900 measurements ≫ n=129 parameters).

Expected result (from the paper / repo)
---------------------------------------
By iteration 51 the relative $L^2$ error should drop to **≈1e-5** or better,
which is several orders of magnitude smaller than what Adam achieves in
millions of steps.

Usage
-----
    python experiments/python_scripts/exp_engd/repro_poisson_2d.py
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.func import functional_call, grad as func_grad, vmap
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from learning_option_pricing.optimizers.natural_gradient import (
    flat_grad, flat_params, set_flat_params, measurement_jacobian,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Network: MLP [2, 32, 1] with tanh — matches the original
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Same architecture as the original `mlp(activation)` with `[2, 32, 1]`."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.tanh(self.fc1(x)))


# ---------------------------------------------------------------------------
# Domains: deterministic grids matching ngrad.domains
# ---------------------------------------------------------------------------

def square_interior_grid(N: int = 30) -> torch.Tensor:
    """Strict-interior grid on $[0,1]^2$: $(i/(N-1), j/(N-1))$ for $i,j\\in\\{1,\\dots,N-2\\}$.

    For N=30 this gives 28×28 = 784 points.  Boundary nodes are excluded so
    they don't double-count with the boundary integrator.
    """
    M = max(N, 2)
    coords = torch.tensor([i / (M - 1) for i in range(1, M - 1)], dtype=torch.float64)
    grid = torch.cartesian_prod(coords, coords)  # (N-2)² rows
    return grid


def square_boundary_grid(N: int = 30) -> torch.Tensor:
    """4-side boundary grid on $\\partial[0,1]^2$ with no double corners.

    Each side uses `linspace(a, b, N)[0:N-1]` (N-1 points).  For N=30 this
    yields 4×29 = 116 points.
    """
    M = max(N, 2)
    interval     = torch.linspace(0.0, 1.0, M, dtype=torch.float64)[:M - 1].unsqueeze(1)
    interval_rev = torch.linspace(1.0, 0.0, M, dtype=torch.float64)[:M - 1].unsqueeze(1)
    zeros = torch.zeros((M - 1, 1), dtype=torch.float64)
    ones  = torch.ones((M - 1, 1), dtype=torch.float64)

    side_0 = torch.cat([interval, zeros], dim=1)        # bottom
    side_1 = torch.cat([ones, interval], dim=1)         # right
    side_2 = torch.cat([interval_rev, ones], dim=1)     # top
    side_3 = torch.cat([zeros, interval_rev], dim=1)    # left
    return torch.cat([side_0, side_1, side_2, side_3], dim=0)


# ---------------------------------------------------------------------------
# PDE — exact solution and right-hand side
# ---------------------------------------------------------------------------

def u_star(xy: torch.Tensor) -> torch.Tensor:
    """$u^*(x,y) = \\sin(\\pi x)\\sin(\\pi y)$."""
    return torch.sin(math.pi * xy[..., 0]) * torch.sin(math.pi * xy[..., 1])


def f_rhs(xy: torch.Tensor) -> torch.Tensor:
    """$f = 2\\pi^2 \\sin(\\pi x)\\sin(\\pi y)$ — so that $-\\Delta u^* = f$."""
    return 2.0 * math.pi**2 * u_star(xy)


# ---------------------------------------------------------------------------
# Functional residual & boundary measurements (composable with jacrev)
# ---------------------------------------------------------------------------

def _laplace_residual(
    params_dict: dict,
    model: nn.Module,
    pts: torch.Tensor,
) -> torch.Tensor:
    """Vector of strong-form Poisson residuals $r_i = -\\Delta u_\\theta(x_i) - f(x_i)$.

    Returns shape `(N,)` to be used by `measurement_jacobian` (jacrev over params).
    """
    def u_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        inp = torch.stack([x, y]).unsqueeze(0)
        return functional_call(model, params_dict, inp).squeeze()

    def laplace_at(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        d2u_dx2 = func_grad(func_grad(lambda xv: u_fn(xv, y)))(x)
        d2u_dy2 = func_grad(func_grad(lambda yv: u_fn(x, yv)))(y)
        return d2u_dx2 + d2u_dy2

    lap = vmap(laplace_at)(pts[:, 0], pts[:, 1])
    f_vals = f_rhs(pts)
    return -lap - f_vals  # residual: -Δu - f  (zero when u = u*)


def _boundary_value(
    params_dict: dict,
    model: nn.Module,
    pts: torch.Tensor,
) -> torch.Tensor:
    """$u_\\theta(x_i)$ for boundary points.  Target is 0 (Dirichlet BC)."""
    def u_at(xy: torch.Tensor) -> torch.Tensor:
        return functional_call(model, params_dict, xy.unsqueeze(0)).squeeze()
    return vmap(u_at)(pts)


# ---------------------------------------------------------------------------
# Loss (autograd-friendly) — uses normal autograd, not torch.func
# ---------------------------------------------------------------------------

def poisson_loss(
    model: nn.Module,
    pts_int: torch.Tensor,
    pts_bdry: torch.Tensor,
    lam_f: float = 1.0,
    lam_tc: float = 4.0,
) -> tuple[torch.Tensor, float, float]:
    """Total loss = lam_f · mean(r²) + lam_tc · mean(u_bdry²).

    The lam_tc=4 factor reflects |∂Ω|=4 in the original integrator.
    """
    x = pts_int[:, 0].detach().clone().requires_grad_(True)
    y = pts_int[:, 1].detach().clone().requires_grad_(True)
    u = model(torch.stack([x, y], dim=1)).squeeze()
    (du_dx,) = torch.autograd.grad(u.sum(), x, create_graph=True)
    (du_dy,) = torch.autograd.grad(u.sum(), y, create_graph=True)
    (d2u_dx2,) = torch.autograd.grad(du_dx.sum(), x, create_graph=True)
    (d2u_dy2,) = torch.autograd.grad(du_dy.sum(), y, create_graph=True)
    laplace = d2u_dx2 + d2u_dy2
    res = -laplace - f_rhs(pts_int)

    loss_int = res.pow(2).mean()
    u_bdry   = model(pts_bdry).squeeze()
    loss_bdry = u_bdry.pow(2).mean()
    total = lam_f * loss_int + lam_tc * loss_bdry
    return total, loss_int.item(), loss_bdry.item()


# ---------------------------------------------------------------------------
# L2 / H1 error vs exact solution
# ---------------------------------------------------------------------------

def compute_errors(model: nn.Module, n_eval: int = 200) -> tuple[float, float]:
    """Relative $L^2$ and $H^1$ errors on a fine $n_{eval}\\times n_{eval}$ grid.

    Mirrors the `eval_integrator = DeterministicIntegrator(interior, 200)` in the
    original — uses 198×198 strict-interior points.
    """
    M = max(n_eval, 2)
    coords = torch.tensor(
        [i / (M - 1) for i in range(1, M - 1)],
        dtype=torch.float64, device=next(model.parameters()).device,
    )
    grid = torch.cartesian_prod(coords, coords)
    x = grid[:, 0].clone().requires_grad_(True)
    y = grid[:, 1].clone().requires_grad_(True)
    u_pred = model(torch.stack([x, y], dim=1)).squeeze()
    # u_ref must depend on the *leaf* x, y so its gradient is captured below.
    u_ref = torch.sin(math.pi * x) * torch.sin(math.pi * y)

    err  = u_pred - u_ref
    l2   = err.pow(2).mean().sqrt()
    ref_l2 = u_ref.detach().pow(2).mean().sqrt()
    rel_l2 = (l2 / ref_l2).item()

    de_dx, de_dy = torch.autograd.grad(err.sum(), [x, y], create_graph=False)
    grad_l2 = (de_dx.pow(2) + de_dy.pow(2)).mean().sqrt()
    h1 = (l2 + grad_l2).item()
    return rel_l2, h1


# ---------------------------------------------------------------------------
# Grid line search — same schedule as the original (steps = 0.5^k, k=0..30)
# ---------------------------------------------------------------------------

def grid_line_search_paper(
    model: nn.Module,
    loss_fn,
    direction: torch.Tensor,
) -> tuple[float, float]:
    """Try $\\alpha = 0.5^k$ for $k=0,\\dots,30$, pick the loss-minimising step.

    Returns `(step, best_loss)`.
    """
    flat0 = flat_params(model).clone()
    grid_k = torch.arange(0, 31, dtype=direction.dtype, device=direction.device)
    steps = 0.5 ** grid_k

    best_loss = float("inf")
    best_step = 0.0
    for alpha in steps:
        set_flat_params(model, flat0 - alpha * direction)
        with torch.enable_grad():
            loss_val = loss_fn().item()
        if not math.isfinite(loss_val):
            continue
        if loss_val < best_loss:
            best_loss = loss_val
            best_step = alpha.item()
    set_flat_params(model, flat0)
    return best_step, best_loss


# ---------------------------------------------------------------------------
# ENGD step: explicit Gram + lstsq (matches original exactly)
# ---------------------------------------------------------------------------

def engd_step_lstsq(
    model: nn.Module,
    pts_int: torch.Tensor,
    pts_bdry: torch.Tensor,
    g: torch.Tensor,
    lam_f: float,
    lam_tc: float,
) -> tuple[torch.Tensor, int, int]:
    """Build $G$ explicitly, solve $G\\delta=g$ via `lstsq` (SVD pseudoinverse).

    For 129 parameters and 900 measurements this is fast and avoids the
    rank-deficient-CG issue we hit in the VPINN+ENGD experiments.
    """
    params_dict = {k: v.detach().clone() for k, v in model.named_parameters()}

    J_F = measurement_jacobian(_laplace_residual, params_dict, model, pts_int)
    J_TC = measurement_jacobian(_boundary_value, params_dict, model, pts_bdry)

    # G = (lam_F/N_F) J_F^T J_F + (lam_TC/N_TC) J_TC^T J_TC
    # Original has lam_F=1 (|Ω|=1), lam_TC=4 (|∂Ω|=4).
    G = (lam_f / J_F.shape[0]) * (J_F.T @ J_F) \
      + (lam_tc / J_TC.shape[0]) * (J_TC.T @ J_TC)

    # lstsq with SVD-driver — robust to rank-deficient G (matches jnp.linalg.lstsq)
    sol = torch.linalg.lstsq(G, g.unsqueeze(1), driver="gelsd")
    delta = sol.solution.squeeze(1)
    if not torch.isfinite(delta).all():
        delta = torch.zeros_like(g)
    return delta, J_F.shape[0], J_TC.shape[0]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run(args) -> dict:
    torch.set_default_dtype(torch.float64)  # match jax_enable_x64 in the original
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    model = MLP(hidden=32).to(device).to(torch.float64)

    n_params = sum(p.numel() for p in model.parameters())
    pts_int  = square_interior_grid(args.N).to(device)
    pts_bdry = square_boundary_grid(args.N).to(device)

    logger.info(f"Network params:        {n_params}")
    logger.info(f"Interior points:       {pts_int.shape[0]}")
    logger.info(f"Boundary points:       {pts_bdry.shape[0]}")
    logger.info(f"Total measurements:    {pts_int.shape[0] + pts_bdry.shape[0]}")
    logger.info(f"Iterations:            {args.iters}")
    logger.info(f"Loss weights:          lam_F=1, lam_TC=4 (matches |∂Ω|=4)")
    logger.info(f"Solver:                lstsq (SVD pseudoinverse, no Tikhonov)")

    history = {"iter": [], "loss": [], "loss_int": [], "loss_bdry": [],
               "rel_l2": [], "h1": [], "step": [], "time": []}
    t0 = time.time()

    for it in range(args.iters):
        # 1. standard gradient
        model.zero_grad()
        loss, lf, lb = poisson_loss(model, pts_int, pts_bdry,
                                    lam_f=1.0, lam_tc=4.0)
        loss.backward()
        g = flat_grad(model)

        # 2. natural gradient direction via lstsq
        delta, _, _ = engd_step_lstsq(model, pts_int, pts_bdry, g,
                                       lam_f=1.0, lam_tc=4.0)

        # 3. line search with steps 0.5^k, k=0..30
        def _loss_fn():
            l, _, _ = poisson_loss(model, pts_int, pts_bdry,
                                   lam_f=1.0, lam_tc=4.0)
            return l
        step, best_loss = grid_line_search_paper(model, _loss_fn, delta)

        # 4. apply update
        flat0 = flat_params(model)
        set_flat_params(model, flat0 - step * delta)

        # 5. log every 5 iterations (matching original)
        if it % 5 == 0 or it == args.iters - 1:
            rel_l2, h1 = compute_errors(model, n_eval=200)
            history["iter"].append(it)
            history["loss"].append(best_loss)
            history["loss_int"].append(lf)
            history["loss_bdry"].append(lb)
            history["rel_l2"].append(rel_l2)
            history["h1"].append(h1)
            history["step"].append(step)
            history["time"].append(time.time() - t0)
            logger.info(
                f"iter {it:>3d}  loss={best_loss:.3e}  "
                f"L2_err={rel_l2:.3e}  H1_err={h1:.3e}  "
                f"step={step:.3e}  ({time.time()-t0:.1f}s)"
            )

    return history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=51)
    p.add_argument("--N", type=int, default=30,
                   help="Integrator density (matches Square/SquareBoundary N=30)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    if args.out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = Path(__file__).resolve().parents[3] / "data" / "repro_poisson_2d" / f"{ts}_iters{args.iters}"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    log_file = args.out_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger.info(f"Run directory: {args.out_dir}")
    logger.info(f"Log file:      {log_file}")

    history = run(args)

    # Persist
    with open(args.out_dir / "history.yaml", "w") as f:
        yaml.safe_dump(history, f, sort_keys=False)
    with open(args.out_dir / "config.yaml", "w") as f:
        cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
        yaml.safe_dump(cfg, f, default_flow_style=False)

    logger.info(f"Final L2 error: {history['rel_l2'][-1]:.3e}")
    logger.info(f"Final H1 error: {history['h1'][-1]:.3e}")
    logger.info(f"Saved history → {args.out_dir / 'history.yaml'}")


if __name__ == "__main__":
    main()
