#!/usr/bin/env python3
"""Generate a Torrilhon/Schaerer-style 5-moment shock-structure profile.

This script runs the *time-dependent* macroscopic 5-moment solver until it
relaxes to a near-stationary shock layer under fixed equilibrium Dirichlet
boundary conditions.

It then:
  - validates the final profile with `validate_shock_profile`
  - generates the 1×3 visualization figure with `generate_validation_suite`
  - archives all artifacts under `data/shock_structure_runs/<timestamp>/`

Preset (paper-style) upstream equilibrium state:
  U_L = (rho, v, theta, Q_hat, R_hat) = (1, Ma*sqrt(3), 1, 0, 3)
with Ma = 1.8 by default.

Note:
  - This is a practical reproduction harness: the repository does not contain
    a dedicated stationary ODE integrator for the heteroclinic orbit in x.
  - The closure used here is the static β-closure with fixed β (default 1e-3).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


def _import_from_repo():
    """Import from local repo without requiring installation."""
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
    from learning_moments_kinetics.solvers.shock_validation import (
        validate_shock_profile,
    )
    from learning_moments_kinetics.visualization.shock_visualization import (
        generate_validation_suite,
    )

    return {
        "torch": __import__("torch"),
        "compute_sigma_torch": compute_sigma_torch,
        "compute_s_beta_torch": compute_s_beta_torch,
        "MacroscopicMomentSolver1D": MacroscopicMomentSolver1D,
        "primitive_to_conservative": primitive_to_conservative,
        "conservative_to_primitive": conservative_to_primitive,
        "validate_shock_profile": validate_shock_profile,
        "generate_validation_suite": generate_validation_suite,
        "repo_root": repo_root,
    }


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass(frozen=True)
class RunConfig:
    ma: float = 1.8
    beta: float = 1e-1
    tau: float = 1.0
    x_min: float = -15.0
    x_max: float = 15.0
    n_x: int = 2000
    cfl: float = 0.15
    cfl_max: float = 0.30
    n_steps_max: int = 80000
    check_every: int = 200
    checkpoint_every: int = 1000
    steady_tol: float = 5e-11
    validation_tol: float = 1e-5
    init_width: float = 3.0
    max_nan_restarts: int = 5


def _upstream_equilibrium(ma: float) -> np.ndarray:
    """Paper-style upstream equilibrium: (1, Ma*sqrt(3), 1, 0, 3)."""
    return np.array([1.0, ma * np.sqrt(3.0), 1.0, 0.0, 3.0], dtype=np.float64)


def _downstream_from_fluxes(u0: np.ndarray) -> np.ndarray:
    """Downstream equilibrium state by conserving the three 1D macroscopic fluxes."""
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
    """Smooth tanh initial condition connecting left/right equilibrium states."""
    xi = (x - 0.0) / max(width, 1e-12)
    w = 0.5 * (1.0 + np.tanh(xi))
    return (1.0 - w)[:, None] * u_left[None, :] + w[:, None] * u_right[None, :]


def main() -> None:
    m = _import_from_repo()
    torch = m["torch"]

    parser = argparse.ArgumentParser(
        description="Run a paper-style shock-structure relaxation and archive outputs."
    )
    parser.add_argument("--ma", type=float, default=1.8)
    parser.add_argument("--beta", type=float, default=1e-1)
    parser.add_argument("--nx", type=int, default=2000)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--cfl", type=float, default=0.15)
    parser.add_argument("--cfl-max", type=float, default=0.30)
    parser.add_argument("--steps", type=int, default=80000)
    parser.add_argument("--x-min", type=float, default=-15.0)
    parser.add_argument("--x-max", type=float, default=15.0)
    parser.add_argument("--steady-tol", type=float, default=5e-11)
    parser.add_argument("--check-every", type=int, default=200)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--validation-tol", type=float, default=1e-5)
    parser.add_argument("--init-width", type=float, default=3.0)
    parser.add_argument("--max-nan-restarts", type=int, default=5)
    args = parser.parse_args()

    cfg = RunConfig(
        ma=args.ma,
        beta=args.beta,
        tau=args.tau,
        x_min=args.x_min,
        x_max=args.x_max,
        n_x=args.nx,
        cfl=args.cfl,
        cfl_max=args.cfl_max,
        n_steps_max=args.steps,
        check_every=args.check_every,
        checkpoint_every=args.checkpoint_every,
        steady_tol=args.steady_tol,
        validation_tol=args.validation_tol,
        init_width=args.init_width,
        max_nan_restarts=args.max_nan_restarts,
    )

    repo_root = m["repo_root"]
    out_dir = repo_root / "data" / "shock_structure_runs" / _timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)

    u_left = _upstream_equilibrium(cfg.ma)
    u_right = _downstream_from_fluxes(u_left)

    # Build solver.
    compute_sigma_torch = m["compute_sigma_torch"]
    compute_s_beta_torch = m["compute_s_beta_torch"]

    beta_const = float(cfg.beta)

    def closure_fn(q_hat, r_hat):
        sigma = compute_sigma_torch(q_hat, r_hat, sigma_min=1e-10)
        beta = torch.full_like(q_hat, beta_const)
        return compute_s_beta_torch(q_hat, r_hat, sigma, beta)

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

    prim_t = torch.tensor(prim0, dtype=torch.float64)
    E = primitive_to_conservative(prim_t)

    bc_left = torch.tensor(u_left, dtype=torch.float64)
    bc_right = torch.tensor(u_right, dtype=torch.float64)

    # ── Hardened relaxation loop ──────────────────────────────────────────
    log_lines: list[str] = []
    cfl_current = float(cfg.cfl)
    E_checkpoint = E.clone()
    step_checkpoint = 0
    nan_restarts = 0
    converged = False
    stable_streak = 0

    def _clamp_positivity(E_):
        """Enforce rho > 0 and theta > 0 in conservative variables."""
        E_c = E_.clone()
        E_c[..., 0] = E_c[..., 0].clamp(min=1e-12)
        rho = E_c[..., 0]
        v = E_c[..., 1] / rho
        theta = (E_c[..., 2] / rho - v**2).clamp(min=1e-12)
        E_c[..., 2] = rho * (v**2 + theta)
        return E_c

    for step in range(cfg.n_steps_max):
        dt = solver.compute_dt(E, cfl=cfl_current)

        # Guard against NaN/Inf in dt.
        if not np.isfinite(dt) or dt <= 0.0:
            log_lines.append(
                f"step={step+1:6d}  dt={dt:.3e}  ** dt is non-finite, "
                f"reverting to checkpoint (step {step_checkpoint})"
            )
            nan_restarts += 1
            if nan_restarts > cfg.max_nan_restarts:
                log_lines.append("ABORT: max NaN restarts exceeded.")
                break
            cfl_current *= 0.5
            E = E_checkpoint.clone()
            stable_streak = 0
            continue

        E_new = solver.step(E, dt, bc_left=bc_left, bc_right=bc_right)
        E_new = _clamp_positivity(E_new)

        # Check for NaN/Inf in the new state.
        if not torch.isfinite(E_new).all():
            nan_restarts += 1
            log_lines.append(
                f"step={step+1:6d}  dt={dt:.3e}  cfl={cfl_current:.4f}  "
                f"** NaN detected, reverting to checkpoint (step {step_checkpoint}), "
                f"restart {nan_restarts}/{cfg.max_nan_restarts}"
            )
            if nan_restarts > cfg.max_nan_restarts:
                log_lines.append("ABORT: max NaN restarts exceeded.")
                break
            cfl_current *= 0.5
            E = E_checkpoint.clone()
            stable_streak = 0
            continue

        stable_streak += 1

        # Adaptive CFL: gradually restore toward cfl_max after sustained stability.
        if stable_streak > 0 and stable_streak % 500 == 0:
            cfl_current = min(cfl_current * 1.1, cfg.cfl_max)

        # Convergence diagnostics.
        if (step + 1) % cfg.check_every == 0 or step == 0:
            delta = (E_new - E).abs().max().item()
            line = (
                f"step={step+1:6d}  dt={dt:.3e}  cfl={cfl_current:.4f}  "
                f"max|dE|={delta:.3e}"
            )
            log_lines.append(line)
            print(line)
            if delta < cfg.steady_tol:
                E = E_new
                log_lines.append("Converged: steady tolerance reached.")
                print("Converged: steady tolerance reached.")
                converged = True
                break

        # Periodic checkpoints.
        if (step + 1) % cfg.checkpoint_every == 0:
            E_checkpoint = E_new.clone()
            step_checkpoint = step + 1

        E = E_new

    if not converged:
        log_lines.append(
            f"WARNING: did not converge within {cfg.n_steps_max} steps. "
            "Saving last state."
        )
        print(
            f"WARNING: did not converge within {cfg.n_steps_max} steps. "
            "Saving last state."
        )

    # Convert final state.  Fall back to last checkpoint if current E has NaN.
    if not torch.isfinite(E).all():
        log_lines.append("Final E has NaN; falling back to last checkpoint.")
        E = E_checkpoint

    prim_final = conservative_to_primitive(E).detach().cpu().numpy()

    # ── Archive numeric outputs ───────────────────────────────────────────
    np.save(out_dir / "x.npy", x)
    np.save(out_dir / "u.npy", prim_final)
    (out_dir / "relaxation_log.txt").write_text("\n".join(log_lines) + "\n")

    meta = {
        "config": asdict(cfg),
        "upstream_state": u_left.tolist(),
        "downstream_state": u_right.tolist(),
        "converged": converged,
        "nan_restarts": nan_restarts,
        "outputs": {
            "x_npy": "x.npy",
            "u_npy": "u.npy",
            "relaxation_log": "relaxation_log.txt",
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # ── Validate and plot ─────────────────────────────────────────────────
    validate_shock_profile = m["validate_shock_profile"]
    generate_validation_suite = m["generate_validation_suite"]

    cwd = Path.cwd()
    figure_path = None
    try:
        os.chdir(out_dir)
        try:
            validate_shock_profile(
                x, prim_final, cfg.ma, tolerance=cfg.validation_tol,
            )
            print("Validation: PASSED all axiom checks.")
        except Exception as exc:
            print(f"Validation: FAILED — {exc}")
            log_lines.append(f"Validation FAILED: {exc}")

        try:
            figure_path = generate_validation_suite(x, prim_final, cfg.ma)
        except Exception as exc:
            figure_path = None
            print(f"Visualization: FAILED — {exc}")
            log_lines.append(f"Visualization FAILED: {exc}")
    finally:
        os.chdir(cwd)

    # Re-save log with any post-loop messages.
    (out_dir / "relaxation_log.txt").write_text("\n".join(log_lines) + "\n")

    print(f"\nArchived run: {out_dir}")
    print(f"Wrote: {out_dir / 'x.npy'}")
    print(f"Wrote: {out_dir / 'u.npy'}")
    print(f"Wrote: {out_dir / 'meta.json'}")
    print(f"Wrote: {out_dir / 'relaxation_log.txt'}")
    if figure_path is not None:
        print(f"Wrote: {out_dir / Path(figure_path).name}")


if __name__ == "__main__":
    main()

