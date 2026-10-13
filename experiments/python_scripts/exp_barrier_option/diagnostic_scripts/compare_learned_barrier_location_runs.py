r"""Learned barrier location of the enrichment profile: comparison of the runs.

Reference: section 19 of ``documents/methodology/barrier_option.md``.

The runs of ``pilot_down_and_out_put.py --corner-treatment learned_barrier_location``
centre the error-function profile of the corner enrichment at a trained
:math:`B' = B e^{\eta}`. This script gathers, from the saved artefacts only (no
retraining), the quantities that answer whether training returns :math:`B'` to
:math:`B`:

- the trajectory of :math:`B'` and of the interior loss over training (read from
  ``history_eps0.yaml``);
- price slices :math:`\Phi_\theta(\cdot,t)` against the closed form
  :math:`V_{DO}(\cdot,t)` at three calendar times, from the saved models;
- one table of the summary metrics.

Every run directory is given on the command line with a label
(``--run LABEL=RUN_DIR``); the control run (analytic enrichment, :math:`B' = B`
fixed) has no ``learned_barrier`` history and appears only in the loss and price
panels.

Outputs (under ``data/compare_learned_barrier_location_runs/<timestamp>/``):

- ``figures/learned_barrier_trajectories.png``;
- ``figures/price_slices.png``;
- ``curves.pt`` (histories and price slices) and ``summary.yaml``;
  ``--replot <OUTPUT_DIR>`` rebuilds both figures from ``curves.pt``.

Usage:
    python experiments/python_scripts/exp_barrier_option/diagnostic_scripts/compare_learned_barrier_location_runs.py \
        --run "Control=data/pilot_down_and_out_put/<control_dir>" \
        --run "B'_0=0.55, lambda_B=0=data/pilot_down_and_out_put/<run_dir>" ...
"""
from __future__ import annotations

import argparse
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from learning_option_pricing.pricing.barrier import reiner_rubinstein_down_and_out_put  # noqa: E402
from learning_option_pricing.utils.figure_layout import finalize_figure  # noqa: E402
from learning_option_pricing.utils.run_context import find_repo_root, get_git_metadata, script_data_dir  # noqa: E402
from pilot_down_and_out_put import DEVICE, load_trained_model, read_run_metadata  # noqa: E402

logger = logging.getLogger("compare_learned_barrier_location_runs")

#: Calendar times of the price slices.
SLICE_TIMES = (0.0, 0.5, 0.9)
#: Number of points of the price grid of the slices, on (B, PRICE_SLICE_UPPER_END].
PRICE_GRID_POINTS = 600
PRICE_SLICE_UPPER_END = 1.6

FORMULA_TEXT = (
    r"$\Phi_\theta=E'+h+d_{\partial_pQ}\Psi_\theta$, $E'=\chi(s)\,\Delta\,\mathrm{erf}(\xi')$, "
    r"$\xi'(s,t)=\frac{\ln(s/B')}{\sigma\sqrt{2(T-t)}}$, $B'=Be^{\eta}$ trained; $\Delta=K-B$, "
    r"$h=\pi-\chi\,\pi(B,\cdot)$, $d_{\partial_pQ}=(T-t)(s-B)$ on the contract's barrier $B$"
    "\n"
    r"Objective $\mathcal{J}=\mathrm{mean}\,(\mathcal{L}^{BS}\Phi_\theta)^2+\lambda_B\,\frac{1}{n}\sum_j\Phi_\theta(B,t_j)^2$; "
    r"control: analytic enrichment ($B'=B$ fixed, $\lambda_B=0$);  $V_{DO}$: Reiner-Rubinstein closed form"
)


def parse_run_arguments(run_arguments: list[str]) -> list[tuple[str, Path]]:
    """``LABEL=RUN_DIR`` pairs, split at the last ``=`` (labels may contain ``=``)."""
    runs = []
    for argument in run_arguments:
        label, _, directory = argument.rpartition("=")
        if not label or not directory:
            raise ValueError(f"--run expects LABEL=RUN_DIR; got {argument!r}")
        runs.append((label, Path(directory)))
    return runs


def collect_curves(runs: list[tuple[str, Path]]) -> dict:
    """Histories, summaries and price slices of every run, from the saved artefacts."""
    curves = {"runs": []}
    for label, run_directory in runs:
        meta = read_run_metadata(run_directory)
        with open(run_directory / "history_eps0.yaml") as handle:
            history = yaml.safe_load(handle)
        with open(run_directory / "summary_eps0.yaml") as handle:
            summary = yaml.safe_load(handle)
        K, B, r, sigma, T = (meta["contract"][key] for key in ("K", "B", "r", "sigma", "T"))
        if meta["hyperparameters"].get("dtype") == "float64":
            torch.set_default_dtype(torch.float64)
        model = load_trained_model(run_directory, 0.0, meta)
        price_grid = torch.linspace(B + 1e-4, PRICE_SLICE_UPPER_END, PRICE_GRID_POINTS, dtype=torch.float64)
        slices = []
        for slice_time in SLICE_TIMES:
            time_grid = torch.full_like(price_grid, slice_time)
            with torch.no_grad():
                inputs = torch.stack([price_grid, time_grid], dim=1).to(DEVICE).to(torch.get_default_dtype())
                learned = model(inputs).squeeze(-1).double().cpu()
            reference = reiner_rubinstein_down_and_out_put(price_grid, K, B, r, sigma, T - time_grid)
            slices.append({"time": slice_time, "learned": learned, "reference": reference})
        torch.set_default_dtype(torch.float32)
        curves["runs"].append({
            "label": label,
            "run_directory": str(run_directory),
            "contract": {"K": K, "B": B, "r": r, "sigma": sigma, "T": T},
            "corner_treatment": meta["hyperparameters"]["corner_treatment"],
            "barrier_trace_penalty_weight": meta["hyperparameters"].get("barrier_trace_penalty_weight"),
            "iterations": history["iter"],
            "learned_barrier": history.get("learned_barrier"),
            # The control run has no separate interior loss: its loss is the interior loss.
            "interior_loss": history.get("interior_loss", history["loss"]),
            "price_grid": price_grid,
            "slices": slices,
            "summary": {key: summary.get(key) for key in (
                "learned_barrier", "initial_learned_barrier", "learned_barrier_minus_barrier",
                "barrier_trace_sup_abs", "best_loss", "best_iter", "rel_l2_outside_corner", "rel_l2_corner",
                "max_abs_error_global")},
        })
        logger.info(f"[{label}] read {run_directory}: best loss {summary['best_loss']:.3e} at iteration "
                    f"{summary['best_iter']}, rel_l2_outside_corner {summary['rel_l2_outside_corner']:.3e}"
                    + (f", B' = {summary['learned_barrier']:.6f}" if summary.get("learned_barrier") is not None else ""))
    return curves


def plot_trajectories(curves: dict, out_path: Path) -> None:
    """B' against iteration (left) and the single-batch interior loss (right), every run."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    colours = plt.cm.tab10.colors
    barrier = curves["runs"][0]["contract"]["B"]
    for index, run in enumerate(curves["runs"]):
        colour = colours[index % len(colours)]
        if run["learned_barrier"] is not None:
            axes[0].plot(run["iterations"], run["learned_barrier"], color=colour, lw=1.5, label=run["label"])
        axes[1].semilogy(run["iterations"], run["interior_loss"], color=colour, lw=1.2, label=run["label"])
    axes[0].axhline(barrier, color="black", linestyle="--", lw=1.2, label=rf"Contract barrier $B={barrier:g}$")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel(r"Learned barrier $B'$")
    axes[0].set_title(r"Trajectory of $B'$")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel(r"$\mathrm{mean}\,(\mathcal{L}^{BS}\Phi_\theta)^2$ (single batch)")
    axes[1].set_title("Interior loss")
    for axis in axes:
        axis.grid(alpha=0.3, which="both")
    legends = [axis.legend(loc="upper left", bbox_to_anchor=(0.0, -0.18), fontsize=8) for axis in axes]
    fig.subplots_adjust(bottom=0.42, wspace=0.28)
    finalize_figure(fig, out_path, legends=legends, formula=FORMULA_TEXT, axes=list(axes))


def plot_price_slices(curves: dict, out_path: Path) -> None:
    """Phi_theta(., t) of every run against V_DO(., t) at the calendar times SLICE_TIMES."""
    fig, axes = plt.subplots(1, len(SLICE_TIMES), figsize=(16, 4.6), sharey=True)
    colours = plt.cm.tab10.colors
    for slice_index, axis in enumerate(axes):
        reference_drawn = False
        for index, run in enumerate(curves["runs"]):
            price_grid = run["price_grid"].numpy()
            current = run["slices"][slice_index]
            if not reference_drawn:
                axis.plot(price_grid, current["reference"].numpy(), color="black", linestyle="--", lw=1.8,
                          label=r"$V_{DO}$ (closed form)")
                reference_drawn = True
            axis.plot(price_grid, current["learned"].numpy(), color=colours[index % len(colours)], lw=1.3,
                      label=rf"$\Phi_\theta$, {run['label']}")
        barrier = curves["runs"][0]["contract"]["B"]
        axis.axvline(barrier, color="grey", linestyle=":", lw=1.0)
        axis.set_xlabel("Underlying price $s$")
        axis.set_title(f"$t = {SLICE_TIMES[slice_index]:g}$")
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("Price")
    legend = axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.subplots_adjust(right=0.78, bottom=0.3)
    finalize_figure(fig, out_path, legends=[legend], formula=FORMULA_TEXT, axes=list(axes))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", default=[], metavar="LABEL=RUN_DIR",
                        help="A run of pilot_down_and_out_put.py and its label in the figures; repeatable.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory (default: timestamped).")
    parser.add_argument("--replot", type=str, default=None, metavar="OUTPUT_DIR",
                        help="Rebuild the figures from OUTPUT_DIR/curves.pt without reading any run.")
    args = parser.parse_args()

    if args.replot is not None:
        out_dir = Path(args.replot)
    else:
        if not args.run:
            parser.error("give at least one --run LABEL=RUN_DIR (or --replot)")
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(args.out_dir) if args.out_dir else script_data_dir(__file__) / timestamp
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "compare.log")])
    logging.getLogger("matplotlib.mathtext").setLevel(logging.WARNING)
    logger.info(f"Command: {' '.join(sys.argv)}")
    logger.info(f"Python {platform.python_version()}, PyTorch {torch.__version__}; output directory {out_dir}")
    logger.info(f"Log file (follow in real time): {out_dir / 'compare.log'}")

    if args.replot is not None:
        curves = torch.load(out_dir / "curves.pt", weights_only=False)
        logger.info(f"--replot: curves read from {out_dir / 'curves.pt'}")
    else:
        curves = collect_curves(parse_run_arguments(args.run))
        torch.save(curves, out_dir / "curves.pt")
        summary = {
            "command": " ".join(sys.argv),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git": get_git_metadata(find_repo_root(Path(__file__).resolve())),
            "slice_times": list(SLICE_TIMES),
            "runs": [{"label": run["label"], "run_directory": run["run_directory"],
                      "corner_treatment": run["corner_treatment"],
                      "barrier_trace_penalty_weight": run["barrier_trace_penalty_weight"], **run["summary"]}
                     for run in curves["runs"]],
        }
        with open(out_dir / "summary.yaml", "w") as handle:
            yaml.dump(summary, handle, default_flow_style=False, sort_keys=False)
        logger.info(f"curves -> {out_dir / 'curves.pt'}; summary -> {out_dir / 'summary.yaml'}")

    plot_trajectories(curves, out_dir / "figures" / "learned_barrier_trajectories.png")
    plot_price_slices(curves, out_dir / "figures" / "price_slices.png")
    logger.info(f"figures -> {out_dir / 'figures'}")


if __name__ == "__main__":
    main()
