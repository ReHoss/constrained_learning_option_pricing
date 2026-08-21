r"""Down-and-out put: error vs. epsilon, per-seed dispersion diagnostic.

Reads the per-seed run directories left by ``pilot_down_and_out_put.py``
(pattern ``<timestamp>_iters<ITERS>_eps<EPSILON>_seed<SEED>``, one directory
per (epsilon, seed) pair) and plots, for each epsilon, the five individual
seed values of the relative L^2 error against the closed form -- both the
global metric and the corner-window metric -- as semi-transparent scatter
points, together with the across-seed median as a solid line. Log-log scale.

Purpose: the median (or a mean +/- standard deviation) alone hides whether
a given epsilon value trains reliably across seeds or occasionally diverges;
plotting every individual seed makes an unstable epsilon visible at a glance.

Does not retrain or re-evaluate anything: reads only the already-saved
``summary_eps<EPSILON>.yaml`` files (never the model checkpoints), so it can
be re-run after any subset of the sweep without touching the trained models.

Usage:
    python3 experiments/python_scripts/exp_barrier_option/diagnostic_scripts/\
plot_error_vs_epsilon_seed_dispersion.py --iters 20000
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from learning_option_pricing.utils.figure_layout import finalize_figure  # noqa: E402
from learning_option_pricing.utils.run_context import script_data_dir  # noqa: E402

FORMULA_TEXT = (
    r"$\mathrm{rel}_{L^2}=\|V_\theta-V_{DO}\|_{L^2(\Omega)}/\|V_{DO}\|_{L^2(\Omega)}$;  "
    r"global: $\Omega=(B,s_\infty)\times(0,T)$;  corner: $\Omega=\{(s,t):|s-B|+(T-t)\leq\varepsilon\}$"
    "\n"
    r"Each $\varepsilon$: 5 seeds $\in\{0,1,2,3,4\}$, shared-seeding policy across $\varepsilon$ within a seed; "
    r"points = individual seeds (semi-transparent), solid line = median."
)


def _collect_runs(base_dir: Path, iters: int) -> dict[float, list[dict]]:
    """Group every ``summary_eps<epsilon>.yaml`` under *base_dir* by epsilon.

    Args:
        base_dir: The ``pilot_down_and_out_put`` data directory, containing
            one subdirectory per (epsilon, seed) run.
        iters:    The ``--iters`` value used for the sweep (part of the run
            directory name), so unrelated runs (different iteration budgets,
            ``_debug_`` smoke tests, harmonised-corner-window combined
            directories) are not picked up.

    Returns:
        Mapping epsilon -> list of {"seed": int, **summary_dict}.
    """
    run_dir_re = re.compile(rf"^\d{{8}}_\d{{6}}_iters{iters}_eps([\d.]+)_seed(\d+)$")
    runs_by_epsilon: dict[float, list[dict]] = {}
    for run_dir in sorted(base_dir.iterdir()):
        match = run_dir_re.match(run_dir.name)
        if not match:
            continue
        epsilon = float(match.group(1))
        seed = int(match.group(2))
        summary_path = run_dir / f"summary_eps{epsilon:g}.yaml"
        with open(summary_path) as f:
            summary = yaml.safe_load(f)
        runs_by_epsilon.setdefault(epsilon, []).append({"seed": seed, **summary})
    return runs_by_epsilon


def plot_seed_dispersion(runs_by_epsilon: dict[float, list[dict]], out_path: Path) -> None:
    """Build and save the error-vs-epsilon seed-dispersion figure."""
    epsilons = sorted(runs_by_epsilon)
    median_global = [statistics.median(r["rel_l2_global"] for r in runs_by_epsilon[eps]) for eps in epsilons]
    median_corner = [statistics.median(r["rel_l2_corner"] for r in runs_by_epsilon[eps]) for eps in epsilons]

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    for i, epsilon in enumerate(epsilons):
        runs = runs_by_epsilon[epsilon]
        ax.scatter(
            [epsilon] * len(runs), [r["rel_l2_global"] for r in runs],
            color="tab:blue", alpha=0.35, s=40, zorder=2,
            label="Individual seeds, global" if i == 0 else None,
        )
        ax.scatter(
            [epsilon] * len(runs), [r["rel_l2_corner"] for r in runs],
            color="tab:red", alpha=0.35, s=40, zorder=2,
            label="Individual seeds, corner" if i == 0 else None,
        )

    ax.plot(epsilons, median_global, marker="o", linestyle="-", color="tab:blue",
            linewidth=2, zorder=3, label="Median, global rel. $L^2$")
    ax.plot(epsilons, median_corner, marker="s", linestyle="-", color="tab:red",
            linewidth=2, zorder=3, label="Median, corner-window rel. $L^2$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Corner-regularisation bandwidth $\varepsilon$")
    ax.set_ylabel(r"Relative $L^2$ error vs. closed form")
    ax.set_title("Down-and-out put: error vs. $\\varepsilon$, 5-seed dispersion", fontsize=11)
    ax.grid(alpha=0.3, which="both")

    legend = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.subplots_adjust(right=0.62, bottom=0.32)
    finalize_figure(fig, out_path, legends=[legend], formula=FORMULA_TEXT, axes=[ax])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot error-vs-epsilon seed dispersion for a pilot_down_and_out_put.py seed sweep.",
    )
    parser.add_argument("--iters", type=int, default=20_000,
                         help="Iterations-per-run value used for the sweep (selects matching run directories).")
    parser.add_argument("--base-dir", type=str, default=None,
                         help="Directory containing the per-(epsilon, seed) run subdirectories "
                              "(default: the pilot_down_and_out_put script's own data directory).")
    parser.add_argument("--out-dir", type=str, default=None,
                         help="Output directory for the figure (default: <base-dir>/<timestamp>_seed_sweep_summary/figures).")
    args = parser.parse_args()

    pilot_script_path = Path(__file__).resolve().parents[1] / "pilot_down_and_out_put.py"
    base_dir = Path(args.base_dir) if args.base_dir is not None else script_data_dir(pilot_script_path)

    runs_by_epsilon = _collect_runs(base_dir, args.iters)
    if not runs_by_epsilon:
        print(f"No runs matching iters={args.iters} found under {base_dir}", file=sys.stderr)
        sys.exit(1)
    for epsilon, runs in sorted(runs_by_epsilon.items()):
        if len(runs) != 5:
            print(f"WARNING: epsilon={epsilon:g} has {len(runs)} seeds (expected 5)", file=sys.stderr)

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        from datetime import datetime
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        out_dir = base_dir / f"{timestamp}_seed_sweep_summary" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "error_vs_epsilon_seed_dispersion.png"
    plot_seed_dispersion(runs_by_epsilon, out_path)
    print(f"Figure saved -> {out_path}")


if __name__ == "__main__":
    main()
