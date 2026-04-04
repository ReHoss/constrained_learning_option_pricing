#!/usr/bin/env python3
"""Solve the stationary 5-moment shock-structure ODE and validate the result.

Uses the scipy-based ODE integrator in
``learning_moments_kinetics.solvers.stationary_shock_ode`` to integrate the
heteroclinic orbit connecting the upstream and downstream equilibria.

Artifacts are archived under ``data/shock_structure_runs/<timestamp>/``.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np


def _import_from_repo():
    """Import from local repo without requiring installation."""
    repo_root = Path(__file__).resolve().parents[2]
    import sys

    sys.path.insert(0, str(repo_root))

    from learning_moments_kinetics.solvers.stationary_shock_ode import (
        solve_stationary_shock,
    )
    from learning_moments_kinetics.solvers.shock_validation import (
        validate_shock_profile,
    )
    from learning_moments_kinetics.visualization.shock_visualization import (
        generate_validation_suite,
    )

    return {
        "solve_stationary_shock": solve_stationary_shock,
        "validate_shock_profile": validate_shock_profile,
        "generate_validation_suite": generate_validation_suite,
        "repo_root": repo_root,
    }


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve the stationary shock-structure ODE and archive outputs."
    )
    parser.add_argument("--ma", type=float, default=1.8, help="Upstream Mach number.")
    parser.add_argument("--beta", type=float, default=1e-1, help="Beta-closure parameter.")
    parser.add_argument("--tau", type=float, default=1.0, help="BGK relaxation time.")
    parser.add_argument("--x-min", type=float, default=-15.0)
    parser.add_argument("--x-max", type=float, default=15.0)
    parser.add_argument("--n-points", type=int, default=500, help="Output grid points.")
    parser.add_argument("--validation-tol", type=float, default=1e-4)
    args = parser.parse_args()

    m = _import_from_repo()
    repo_root = m["repo_root"]

    out_dir = repo_root / "data" / "shock_structure_runs" / _timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Solving stationary shock ODE for Ma = {args.ma} ...")
    sol = m["solve_stationary_shock"](
        Ma=args.ma,
        tau=args.tau,
        beta=args.beta,
        x_span=(args.x_min, args.x_max),
        n_points=args.n_points,
    )

    x, u = sol.x, sol.u
    print(f"  Grid: {len(x)} points, x in [{x[0]:.2f}, {x[-1]:.2f}]")
    print(f"  Upstream:   rho={u[0,0]:.6f}  v={u[0,1]:.6f}  theta={u[0,2]:.6f}")
    print(f"  Downstream: rho={u[-1,0]:.6f}  v={u[-1,1]:.6f}  theta={u[-1,2]:.6f}")

    np.save(out_dir / "x.npy", x)
    np.save(out_dir / "u.npy", u)

    meta = {
        "solver": "stationary_shock_ode",
        "config": {
            "ma": args.ma,
            "beta": args.beta,
            "tau": args.tau,
            "x_span": [args.x_min, args.x_max],
            "n_points": args.n_points,
        },
        "upstream_state": sol.u_upstream.tolist(),
        "downstream_state": sol.u_downstream.tolist(),
        "outputs": {
            "x_npy": "x.npy",
            "u_npy": "u.npy",
        },
    }

    # Validate.
    print("\nRunning validation suite ...")
    try:
        m["validate_shock_profile"](x, u, args.ma, tolerance=args.validation_tol)
        print("Validation: PASSED all axiom checks.")
        meta["validation"] = "PASSED"
    except Exception as exc:
        print(f"Validation: FAILED — {exc}")
        meta["validation"] = f"FAILED: {exc}"

    # Visualize.
    cwd = Path.cwd()
    try:
        os.chdir(out_dir)
        figure_path = m["generate_validation_suite"](x, u, args.ma)
        meta["outputs"]["png"] = str(Path(figure_path).name)
    except Exception as exc:
        figure_path = None
        print(f"Visualization: FAILED — {exc}")
    finally:
        os.chdir(cwd)

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nArchived run: {out_dir}")
    print(f"Wrote: {out_dir / 'x.npy'}")
    print(f"Wrote: {out_dir / 'u.npy'}")
    print(f"Wrote: {out_dir / 'meta.json'}")
    if figure_path is not None:
        print(f"Wrote: {out_dir / Path(figure_path).name}")


if __name__ == "__main__":
    main()
