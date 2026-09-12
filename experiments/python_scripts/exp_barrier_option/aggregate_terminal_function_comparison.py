r"""Down-and-out put: aggregate the terminal-function comparison across seeds.

Reads the per-run directories left by ``pilot_down_and_out_put.py`` for one
iteration budget and one corner-layer bandwidth epsilon, groups them by
terminal-function configuration (raw payoff, Chen-Mangasarian smoothed payoff,
Black-Scholes payoff through the ordinary autograd route, Black-Scholes payoff
through the two-term analytic-residual route, split-semigroup profile) and by
master seed, and produces:

- ``summary.yaml``: every per-seed metric value with the run directory it was
  read from, plus the across-seed median, mean, standard deviation, minimum and
  maximum per configuration;
- ``table.md``: the same, as a Markdown table (median [min, max] over seeds);
- ``figures/terminal_function_comparison.png``: one panel per metric, every
  individual seed as a semi-transparent point and the across-seed median as a
  filled marker, configurations on the abscissa.

The comparison metric is ``rel_l2_outside_corner`` (relative L2 error against
the Reiner-Rubinstein closed form on the complement of the ell^1 corner
window, i.e. exactly the region where the PDE residual is enforced when the
corner is excluded from collocation). ``rel_l2_global`` (corner window
included) and ``rel_l2_corner`` (window alone) are plotted as well, labelled as
such: the first is contaminated by the corner discontinuity, the second is a
diagnostic of a region where nothing is enforced.

Does not retrain or re-evaluate anything: reads only the already-saved
``summary_eps<EPSILON>.yaml`` files. Runs trained before
``rel_l2_outside_corner`` existed must first be refreshed with
``pilot_down_and_out_put.py --replot <RUN_DIR>`` (which recomputes the
evaluation from the saved model and writes the metric back).

Usage:
    python3 experiments/python_scripts/exp_barrier_option/\
aggregate_terminal_function_comparison.py --iters 20000 --epsilon 0.1
"""
from __future__ import annotations

import argparse
import logging
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from learning_option_pricing.utils.figure_layout import finalize_figure  # noqa: E402
from learning_option_pricing.utils.run_context import script_data_dir  # noqa: E402

logger = logging.getLogger("aggregate_terminal_function_comparison")

PILOT_SCRIPT_PATH = Path(__file__).resolve().parent / "pilot_down_and_out_put.py"

# Run-directory name written by the pilot:
#   <timestamp>_iters<ITERS>_eps<EPS>_seed<SEED>[<payoff_tag>][_nocorner]
RUN_DIRECTORY_PATTERN = re.compile(
    r"^(?P<timestamp>\d{8}_\d{6})_iters(?P<iters>\d+)_eps(?P<eps>[0-9.]+)_seed(?P<seed>\d+)"
    r"(?P<payoff_tag>(?:_(?!nocorner)[A-Za-z0-9.]+)*)(?P<nocorner>_nocorner)?$"
)

# Ordered so that the figure's abscissa reads from the least to the most
# structured terminal function.
CONFIGURATION_LABELS: dict[str, str] = {
    "raw": "Raw payoff $(K-s)^+$",
    "smoothed": "Chen-Mangasarian smoothed payoff",
    "blackscholes": "Black-Scholes put price\n(ordinary autograd route)",
    "blackscholes_analyticres": "Black-Scholes put price\n(two-term analytic-residual route)",
    "split": "Split-semigroup profile",
}

METRIC_PANELS: list[tuple[str, str, str]] = [
    ("rel_l2_outside_corner", "Relative $L^2$ error outside the corner window\n(comparison metric)", "log"),
    ("rel_l2_global", "Relative $L^2$ error, whole domain\n(corner window INCLUDED: contaminated by the corner)", "log"),
    ("rel_l2_corner", "Relative $L^2$ error inside the corner window\n(diagnostic: residual not enforced there)", "log"),
    ("best_loss", "Best interior residual loss\n(minimum over iterations of a noisy batch loss)", "log"),
]

FORMULA_TEXT = (
    r"$\mathrm{rel}_{L^2}(\Omega)=\|V_\theta-V_{DO}\|_{L^2(\Omega)}/\|V_{DO}\|_{L^2(\Omega)}$ on a "
    r"$300\times100$ grid of $(B,s_\infty)\times(0,T)$;  corner window "
    r"$N=\{(s,t):|s-B|+(T-t)\leq w\}$, $w$ = --corner-window;  "
    r"outside corner: $\Omega\setminus N$;  global: $\Omega$;  corner: $N$."
    "\n"
    r"$V_{DO}$ = Reiner-Rubinstein closed form;  "
    r"best loss $=\min_k \mathrm{mean}_{(s,t)\in\mathrm{batch}_k}\,(\mathcal{L}^{BS}V_\theta)^2$ "
    r"over the $n_f$ fresh collocation points of iteration $k$.  "
    "Points: individual master seeds; filled marker: across-seed median."
)


def configuration_key_from_payoff_tag(payoff_tag: str) -> str:
    """Map the pilot's directory payoff tag to a configuration key."""
    if payoff_tag == "":
        return "raw"
    if payoff_tag.startswith("_smoothed"):
        return "smoothed"
    if payoff_tag == "_blackscholes":
        return "blackscholes"
    if payoff_tag == "_blackscholes_analyticres":
        return "blackscholes_analyticres"
    if payoff_tag.startswith("_split"):
        return "split"
    raise ValueError(f"unrecognised payoff tag {payoff_tag!r}")


def collect_runs(base_dir: Path, iters: int, epsilon: float, require_nocorner: bool) -> dict[str, dict[int, dict]]:
    """Return ``{configuration_key: {seed: summary_with_run_dir}}``.

    When several run directories share a configuration and a seed, the most
    recent timestamp is kept and the others are reported.
    """
    runs: dict[str, dict[int, dict]] = defaultdict(dict)
    for run_dir in sorted(base_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("_debug_"):
            continue
        match = RUN_DIRECTORY_PATTERN.match(run_dir.name)
        if match is None:
            continue
        if int(match["iters"]) != iters or float(match["eps"]) != epsilon:
            continue
        corner_excluded = match["nocorner"] is not None
        if require_nocorner and not corner_excluded:
            logger.info(f"  skipping {run_dir.name}: corner not excluded from collocation")
            continue
        summary_path = run_dir / f"summary_eps{epsilon:g}.yaml"
        if not summary_path.exists():
            logger.warning(f"  skipping {run_dir.name}: no {summary_path.name} (run incomplete?)")
            continue
        configuration = configuration_key_from_payoff_tag(match["payoff_tag"])
        seed = int(match["seed"])
        with open(summary_path) as f:
            summary = yaml.safe_load(f)
        summary["run_dir"] = str(run_dir)
        summary["timestamp"] = match["timestamp"]
        summary["corner_excluded_from_collocation"] = corner_excluded
        if seed in runs[configuration]:
            previous = runs[configuration][seed]
            kept, dropped = ((summary, previous) if summary["timestamp"] > previous["timestamp"]
                             else (previous, summary))
            logger.warning(
                f"  duplicate ({configuration}, seed {seed}): keeping {Path(kept['run_dir']).name}, "
                f"ignoring {Path(dropped['run_dir']).name}"
            )
            runs[configuration][seed] = kept
        else:
            runs[configuration][seed] = summary
    return runs


def aggregate(runs: dict[str, dict[int, dict]], metrics: list[str]) -> dict:
    """Per-configuration, per-metric across-seed statistics."""
    aggregated: dict = {}
    for configuration in CONFIGURATION_LABELS:
        if configuration not in runs:
            continue
        per_seed = runs[configuration]
        entry: dict = {
            "label": CONFIGURATION_LABELS[configuration].replace("\n", " "),
            "seeds": sorted(per_seed),
            "runs": {seed: per_seed[seed]["run_dir"] for seed in sorted(per_seed)},
            "metrics": {},
        }
        for metric in metrics:
            values = {seed: per_seed[seed].get(metric) for seed in sorted(per_seed)}
            present = [v for v in values.values() if v is not None]
            missing = [seed for seed, v in values.items() if v is None]
            if missing:
                logger.warning(
                    f"  {configuration}: metric {metric!r} missing for seeds {missing} "
                    f"(refresh those runs with --replot)"
                )
            stats = {
                "per_seed": values,
                "n": len(present),
                "median": statistics.median(present) if present else None,
                "mean": statistics.fmean(present) if present else None,
                "std": statistics.stdev(present) if len(present) > 1 else None,
                "min": min(present) if present else None,
                "max": max(present) if present else None,
            }
            entry["metrics"][metric] = stats
        aggregated[configuration] = entry
    return aggregated


def write_markdown_table(aggregated: dict, metrics: list[str], path: Path, iters: int, epsilon: float) -> None:
    lines = [
        f"# Terminal-function comparison — {iters} iterations, epsilon = {epsilon:g}",
        "",
        "Median [min, max] over master seeds; n = number of seeds with the metric present.",
        "`rel_l2_outside_corner` is the comparison metric; `rel_l2_global` includes the corner window "
        "(contaminated by the corner discontinuity); `rel_l2_corner` is the window alone (diagnostic).",
        "",
        "| Configuration | n | " + " | ".join(f"`{m}`" for m in metrics) + " |",
        "|---|---|" + "|".join("---" for _ in metrics) + "|",
    ]
    for configuration, entry in aggregated.items():
        cells = []
        for metric in metrics:
            stats = entry["metrics"][metric]
            if stats["n"] == 0:
                cells.append("—")
            elif metric == "best_iter":
                cells.append(f"{stats['median']:.0f} [{stats['min']:.0f}, {stats['max']:.0f}]")
            else:
                cells.append(f"{stats['median']:.3e} [{stats['min']:.3e}, {stats['max']:.3e}]")
        n_seeds = max(entry["metrics"][m]["n"] for m in metrics)
        lines.append(f"| {entry['label']} | {n_seeds} | " + " | ".join(cells) + " |")
    lines += ["", "Run directories:", ""]
    for configuration, entry in aggregated.items():
        for seed, run_dir in entry["runs"].items():
            lines.append(f"- {configuration}, seed {seed}: `{run_dir}`")
    path.write_text("\n".join(lines) + "\n")


def plot_comparison(aggregated: dict, path: Path, iters: int, epsilon: float) -> None:
    configurations = list(aggregated)
    fig, axes = plt.subplots(1, len(METRIC_PANELS), figsize=(4.2 * len(METRIC_PANELS), 4.8))
    positions = range(len(configurations))
    for ax, (metric, title, scale) in zip(axes, METRIC_PANELS):
        for position, configuration in zip(positions, configurations):
            stats = aggregated[configuration]["metrics"][metric]
            values = [v for v in stats["per_seed"].values() if v is not None]
            ax.scatter([position] * len(values), values, s=28, alpha=0.45, color="tab:blue", zorder=2)
            if stats["median"] is not None:
                ax.scatter([position], [stats["median"]], s=70, marker="D", color="tab:blue",
                           edgecolor="black", zorder=3)
        ax.set_yscale(scale)
        ax.set_xticks(list(positions))
        ax.set_xticklabels([CONFIGURATION_LABELS[c] for c in configurations], rotation=30, ha="right", fontsize=7)
        ax.set_title(title, fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("Metric value")
    fig.suptitle(
        f"Down-and-out put — terminal-function comparison, {iters} iterations, "
        f"$\\varepsilon={epsilon:g}$, corner window excluded from collocation",
        fontsize=10,
    )
    fig.tight_layout()
    finalize_figure(fig, path, formula=FORMULA_TEXT, axes=list(axes), formula_fontsize=6.5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iters", type=int, required=True, help="Iteration budget of the runs to aggregate.")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Corner-layer bandwidth of the runs to aggregate.")
    parser.add_argument("--base-dir", type=str, default=None,
                        help="Directory holding the pilot's run directories "
                             "(default: the pilot's own data directory).")
    parser.add_argument("--include-corner-trained-runs", action="store_true",
                        help="Also aggregate runs whose collocation sampler did NOT exclude the corner "
                             "window (default: only _nocorner runs, so all configurations share the "
                             "same training domain).")
    parser.add_argument("--metrics", nargs="+", type=str,
                        default=["rel_l2_outside_corner", "rel_l2_global", "rel_l2_corner",
                                 "max_abs_error_outside_corner", "best_loss", "best_iter"],
                        help="Summary keys to aggregate.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: data/<this script>/<timestamp>_iters<ITERS>_eps<EPS>).")
    args = parser.parse_args()

    base_dir = Path(args.base_dir) if args.base_dir is not None else script_data_dir(PILOT_SCRIPT_PATH)
    corner_tag = "" if args.include_corner_trained_runs else "_nocorner"
    out_dir = (Path(args.out_dir) if args.out_dir is not None else script_data_dir(__file__) / (
        f"{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}_iters{args.iters}_eps{args.epsilon:g}{corner_tag}"
    ))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "aggregate.log")],
    )
    logger.info("Terminal-function comparison — across-seed aggregation")
    logger.info(f"  Command: {' '.join(sys.argv)}")
    logger.info(f"  Python: {sys.version.split()[0]}")
    logger.info(f"  Run directories read from: {base_dir}")
    logger.info(f"  Output directory: {out_dir}")
    logger.info(f"  iters={args.iters}  epsilon={args.epsilon:g}  corner-trained runs included: "
                f"{args.include_corner_trained_runs}  metrics={args.metrics}")

    runs = collect_runs(base_dir, args.iters, args.epsilon, require_nocorner=not args.include_corner_trained_runs)
    if not runs:
        logger.error("No matching run directory found.")
        sys.exit(1)
    for configuration, per_seed in runs.items():
        logger.info(f"  {configuration}: seeds {sorted(per_seed)}")

    aggregated = aggregate(runs, args.metrics)
    with open(out_dir / "summary.yaml", "w") as f:
        yaml.dump({
            "command": " ".join(sys.argv),
            "iters": args.iters, "epsilon": args.epsilon,
            "corner_trained_runs_included": args.include_corner_trained_runs,
            "base_dir": str(base_dir),
            "configurations": aggregated,
        }, f, default_flow_style=False, sort_keys=False)
    logger.info(f"  Summary saved -> {out_dir / 'summary.yaml'}")
    write_markdown_table(aggregated, args.metrics, out_dir / "table.md", args.iters, args.epsilon)
    logger.info(f"  Table saved -> {out_dir / 'table.md'}")
    figure_path = out_dir / "figures" / "terminal_function_comparison.png"
    plot_comparison(aggregated, figure_path, args.iters, args.epsilon)
    logger.info(f"  Figure saved -> {figure_path}")
    for configuration, entry in aggregated.items():
        stats = entry["metrics"].get("rel_l2_outside_corner")
        if stats and stats["n"]:
            logger.info(
                f"  {configuration:<26s} rel_l2_outside_corner median={stats['median']:.4e} "
                f"[{stats['min']:.4e}, {stats['max']:.4e}] (n={stats['n']})"
            )


if __name__ == "__main__":
    main()
