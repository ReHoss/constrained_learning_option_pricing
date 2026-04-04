#!/usr/bin/env python3
"""Pathological beta-case debug run for stationary shock relaxation.

Focused single-case run:
  - Ma = 1.8
  - beta = 1e-6
  - tau = 1.0
  - x in [-15, 15], nx=3000

Convergence controls:
  - steady_tol = 1e-6
  - n_steps_max = 250000
  - cfl = 0.1

Outputs are archived under:
  data/shock_structure_runs/pathological_beta_case/<timestamp>/
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


def _import_from_repo():
    repo_root = Path(__file__).resolve().parents[2]
    import sys

    sys.path.insert(0, str(repo_root))

    import torch  # noqa: F401
    from learning_moments_kinetics.closures.beta_closure_net import (
        compute_s_beta_torch,
        compute_sigma_torch,
    )
    from learning_moments_kinetics.solvers.macroscopic_moment_1d import (
        MacroscopicMomentSolver1D,
        conservative_to_primitive,
        primitive_to_conservative,
    )

    return {
        "torch": __import__("torch"),
        "compute_sigma_torch": compute_sigma_torch,
        "compute_s_beta_torch": compute_s_beta_torch,
        "MacroscopicMomentSolver1D": MacroscopicMomentSolver1D,
        "primitive_to_conservative": primitive_to_conservative,
        "conservative_to_primitive": conservative_to_primitive,
        "repo_root": repo_root,
    }


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_style():
    import matplotlib.pyplot as plt

    for name in ("seaborn-v0_8-paper", "seaborn-paper", "seaborn-v0_8-whitegrid", "classic"):
        try:
            plt.style.use(name)
            return
        except OSError:
            continue


@dataclass(frozen=True)
class RunConfig:
    ma: float = 1.8
    beta: float = 1e-6
    tau: float = 1.0
    x_min: float = -15.0
    x_max: float = 15.0
    n_x: int = 3000
    cfl: float = 0.1
    cfl_max: float = 0.20
    n_steps_max: int = 250000
    check_every: int = 200
    checkpoint_every: int = 1000
    steady_tol: float = 1e-6
    init_width: float = 3.0
    max_nan_restarts: int = 20


def _upstream_equilibrium(ma: float) -> np.ndarray:
    return np.array([1.0, ma * np.sqrt(3.0), 1.0, 0.0, 3.0], dtype=np.float64)


def _downstream_from_fluxes(u0: np.ndarray) -> np.ndarray:
    rho0, v0, theta0, q0, _r0 = u0
    m = rho0 * v0
    mom = rho0 * v0**2 + rho0 * theta0
    energy = rho0 * v0**3 + 3.0 * rho0 * v0 * theta0 + rho0 * theta0**1.5 * q0
    disc = 9.0 * mom**2 - 8.0 * m * energy
    if disc < 0.0:
        raise RuntimeError(f"Invalid upstream state: RH discriminant={disc:.6e} < 0.")
    v1 = 3.0 * mom / (2.0 * m) - v0
    rho1 = m / v1
    theta1 = v1 * (mom / m - v1)
    return np.array([rho1, v1, theta1, 0.0, 3.0], dtype=np.float64)


def _smooth_ic(
    x: np.ndarray, u_left: np.ndarray, u_right: np.ndarray, width: float
) -> np.ndarray:
    xi = x / max(width, 1e-12)
    w = 0.5 * (1.0 + np.tanh(xi))
    return (1.0 - w)[:, None] * u_left[None, :] + w[:, None] * u_right[None, :]


def main() -> None:
    m = _import_from_repo()
    torch = m["torch"]

    cfg = RunConfig()
    repo_root = m["repo_root"]
    out_dir = (
        repo_root
        / "data"
        / "shock_structure_runs"
        / "pathological_beta_case"
        / _timestamp()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    u_left = _upstream_equilibrium(cfg.ma)
    u_right = _downstream_from_fluxes(u_left)

    compute_sigma_torch = m["compute_sigma_torch"]
    compute_s_beta_torch = m["compute_s_beta_torch"]

    def closure_fn(q_hat, r_hat):
        sigma = compute_sigma_torch(q_hat, r_hat, sigma_min=1e-10)
        beta_t = torch.full_like(q_hat, float(cfg.beta))
        return compute_s_beta_torch(q_hat, r_hat, sigma, beta_t)

    solver = m["MacroscopicMomentSolver1D"](
        x_min=cfg.x_min,
        x_max=cfg.x_max,
        n_x=cfg.n_x,
        tau=cfg.tau,
        closure_fn=closure_fn,
        dtype=torch.float64,
        device="cpu",
    )

    primitive_to_conservative = m["primitive_to_conservative"]
    conservative_to_primitive = m["conservative_to_primitive"]

    x = solver.x.detach().cpu().numpy()
    prim0 = _smooth_ic(x, u_left, u_right, cfg.init_width)
    E = primitive_to_conservative(torch.tensor(prim0, dtype=torch.float64))

    bc_left = torch.tensor(u_left, dtype=torch.float64)
    bc_right = torch.tensor(u_right, dtype=torch.float64)

    def _clamp_positivity(E_):
        E_c = E_.clone()
        E_c[..., 0] = E_c[..., 0].clamp(min=1e-12)
        rho = E_c[..., 0]
        v = E_c[..., 1] / rho
        theta = (E_c[..., 2] / rho - v**2).clamp(min=1e-12)
        E_c[..., 2] = rho * (v**2 + theta)
        return E_c

    log_lines: list[str] = []
    cfl_current = float(cfg.cfl)
    E_checkpoint = E.clone()
    step_checkpoint = 0
    nan_restarts = 0
    converged = False
    stable_streak = 0
    final_residual = float("nan")
    stop_reason = "max_steps_reached"
    final_step = cfg.n_steps_max

    for step in range(cfg.n_steps_max):
        dt = solver.compute_dt(E, cfl=cfl_current)
        if not np.isfinite(dt) or dt <= 0.0:
            nan_restarts += 1
            log_lines.append(
                f"step={step+1:7d} dt={dt:.3e} ** bad dt, restart {nan_restarts}"
            )
            if nan_restarts > cfg.max_nan_restarts:
                stop_reason = "max_nan_restarts_exceeded_dt"
                final_step = step + 1
                break
            cfl_current *= 0.5
            E = E_checkpoint.clone()
            stable_streak = 0
            continue

        E_new = solver.step(E, dt, bc_left=bc_left, bc_right=bc_right)
        E_new = _clamp_positivity(E_new)
        if not torch.isfinite(E_new).all():
            nan_restarts += 1
            log_lines.append(
                f"step={step+1:7d} dt={dt:.3e} cfl={cfl_current:.4f} ** NaN restart {nan_restarts}"
            )
            if nan_restarts > cfg.max_nan_restarts:
                stop_reason = "max_nan_restarts_exceeded_state"
                final_step = step + 1
                break
            cfl_current *= 0.5
            E = E_checkpoint.clone()
            stable_streak = 0
            continue

        stable_streak += 1
        if stable_streak % 500 == 0:
            cfl_current = min(cfl_current * 1.05, cfg.cfl_max)

        final_residual = float((E_new - E).abs().max().item())
        if (step + 1) % cfg.check_every == 0 or step == 0:
            line = (
                f"step={step+1:7d} dt={dt:.3e} cfl={cfl_current:.4f} "
                f"max|dE|={final_residual:.3e}"
            )
            print(line)
            log_lines.append(line)

        E = E_new
        final_step = step + 1

        if final_residual < cfg.steady_tol:
            converged = True
            stop_reason = "steady_tol_reached"
            log_lines.append("Converged: steady tolerance reached.")
            break

        if (step + 1) % cfg.checkpoint_every == 0:
            E_checkpoint = E.clone()
            step_checkpoint = step + 1

    if not torch.isfinite(E).all():
        log_lines.append("Final E not finite; fallback to checkpoint.")
        E = E_checkpoint
        final_step = step_checkpoint
        stop_reason += "_fallback_checkpoint"

    prim_final = conservative_to_primitive(E).detach().cpu().numpy()

    np.save(out_dir / "x.npy", x)
    np.save(out_dir / "u.npy", prim_final)
    (out_dir / "relaxation_log.txt").write_text("\n".join(log_lines) + "\n")

    # Requested figure: density and normalized heat flux.
    import matplotlib.pyplot as plt

    _safe_style()
    fig, (ax_rho, ax_q) = plt.subplots(1, 2, figsize=(11.0, 3.8), constrained_layout=True)
    ax_rho.plot(x, prim_final[:, 0], color="black", linewidth=2.0)
    ax_q.plot(x, prim_final[:, 3], color="black", linewidth=2.0)

    ax_rho.set_xlabel(r"$x$")
    ax_rho.set_ylabel(r"$\rho$")
    ax_rho.set_title(rf"Density, $\beta={cfg.beta:.0e}$")
    ax_rho.set_xlim(-8.0, 8.0)
    ax_rho.grid(True, alpha=0.25)

    ax_q.set_xlabel(r"$x$")
    ax_q.set_ylabel(r"$\hat{Q}$")
    ax_q.set_title(rf"Normalized heat flux, $\beta={cfg.beta:.0e}$")
    ax_q.set_xlim(-8.0, 8.0)
    ax_q.grid(True, alpha=0.25)

    fig.savefig(out_dir / "pathological_beta_rho_qhat.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    meta = {
        "config": asdict(cfg),
        "converged": converged,
        "stop_reason": stop_reason,
        "final_step": final_step,
        "final_residual_max_abs_dE": final_residual,
        "nan_restarts": nan_restarts,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # Explicit console diagnostics requested by user.
    print(f"Final step count: {final_step}")
    print(f"Final residual (max|dE|): {final_residual:.6e}")
    print(f"Stop reason: {stop_reason}")
    print(f"Archived run: {out_dir}")


if __name__ == "__main__":
    main()

