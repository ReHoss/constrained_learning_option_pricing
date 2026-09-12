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

Two further diagnostics are computed from the SAVED MODELS (no retraining;
the trained weights and the run's metadata are loaded through the pilot's
``load_trained_model``), on the pilot's own evaluation grid, with tau = T - t:

- ``window_shape_sweep/``: the relative L2 error on the complement of an
  excluded window, for three window SHAPES around the corner (B, T), each
  swept over a parameter --

      lozenge    N_w = { |s-B| + tau <= w },                w in {0.1, 0.2, 0.3, 0.5}
      parabola   N_c = { |s-B| <= c B sigma sqrt(tau) },    c in {0, 1, 2, 3}
      hyperbola  N_d = { tau (s-B) <= d },                  d in {0.005, 0.02, 0.05, 0.1}

  -- together with the fraction of the domain's area each window excludes.
  The error is plotted against the EXCLUDED AREA FRACTION, not against the
  parameter, so the three families are comparable on one abscissa (the
  lozenge at w = --corner-window reproduces ``rel_l2_outside_corner``).
- ``band_network_contribution/``: in the band 0.1 < |s-B| < 0.3 (all t), the
  L2 norms ||Phi_theta - V_DO|| and ||h_eps - V_DO||, where Phi_theta =
  g1 u_theta + g2 is the trained trial solution and h_eps = g2 is the
  terminal-function extension alone (no network). If the two are of the same
  order, the network contributes nothing in that band.

The across-seed part reads only the already-saved ``summary_eps<EPSILON>.yaml``
files. Runs trained before ``rel_l2_outside_corner`` existed must first be
refreshed with ``pilot_down_and_out_put.py --replot <RUN_DIR>`` (which
recomputes the evaluation from the saved model and writes the metric back).

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
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from learning_option_pricing.utils.figure_layout import finalize_figure  # noqa: E402
from learning_option_pricing.utils.run_context import script_data_dir  # noqa: E402
from pilot_down_and_out_put import (  # noqa: E402
    DEVICE, evaluate_against_closed_form, load_trained_model, read_run_metadata,
)

logger = logging.getLogger("aggregate_terminal_function_comparison")

PILOT_SCRIPT_PATH = Path(__file__).resolve().parent / "pilot_down_and_out_put.py"

# Excluded-window families around the corner (B, T), tau = T - t. Each mask
# takes the price grid ss, the time-to-maturity grid tau, the barrier B, the
# volatility sigma and the family parameter, and returns the EXCLUDED region.
WINDOW_FAMILIES: dict[str, dict] = {
    "lozenge": {
        "label": r"Lozenge $N_w=\{|s-B|+\tau\leq w\}$",
        "parameter_symbol": "w",
        "parameters": [0.1, 0.2, 0.3, 0.5],
        "mask": lambda ss, tau, B, sigma, p: (ss - B).abs() + tau <= p,
        "color": "tab:blue", "marker": "o",
    },
    "parabola": {
        "label": r"Parabola $N_c=\{|s-B|\leq c\,B\sigma\sqrt{\tau}\}$",
        "parameter_symbol": "c",
        "parameters": [0.0, 1.0, 2.0, 3.0],
        "mask": lambda ss, tau, B, sigma, p: (ss - B).abs() <= p * B * sigma * torch.sqrt(tau),
        "color": "tab:orange", "marker": "s",
    },
    "hyperbola": {
        "label": r"Hyperbola $N_d=\{\tau\,(s-B)\leq d\}$",
        "parameter_symbol": "d",
        "parameters": [0.005, 0.02, 0.05, 0.1],
        "mask": lambda ss, tau, B, sigma, p: tau * (ss - B) <= p,
        "color": "tab:green", "marker": "^",
    },
}

WINDOW_SHAPE_FORMULA_TEXT = (
    r"$\mathrm{rel}_{L^2}(\Omega\setminus N)=\|V_\theta-V_{DO}\|_{L^2(\Omega\setminus N)}"
    r"/\|V_{DO}\|_{L^2(\Omega\setminus N)}$ on the pilot's $300\times100$ grid of "
    r"$\Omega=(B,s_\infty)\times(0,T)$, $\tau=T-t$;  abscissa: $|N\cap\Omega|/|\Omega|$ (grid fraction)."
    "\n"
    r"$N_w=\{|s-B|+\tau\leq w\}$, $w\in\{0.1,0.2,0.3,0.5\}$;  "
    r"$N_c=\{|s-B|\leq c\,B\sigma\sqrt{\tau}\}$, $c\in\{0,1,2,3\}$;  "
    r"$N_d=\{\tau(s-B)\leq d\}$, $d\in\{0.005,0.02,0.05,0.1\}$.  "
    "Faint points: individual seeds; solid line: across-seed median; labels: parameter value."
)

BAND_FORMULA_TEXT = (
    r"Band $\mathcal{B}=\{(s,t): b_{lo}<|s-B|<b_{hi}\}$ (all $t$);  "
    r"$\|f\|_{L^2(\mathcal{B})}=\left(\sum_{\mathcal{B}} f^2\,\Delta s\,\Delta t\right)^{1/2}$ on the pilot's grid."
    "\n"
    r"$\Phi_\theta=g_1u_\theta+g_2$ (trained trial solution, solid points per seed, filled marker: median);  "
    r"$h_\varepsilon=g_2$ (terminal-function extension alone, no network; dashed);  "
    r"$\|V_{DO}\|_{L^2(\mathcal{B})}$ dotted."
)

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
    # Explicit margins: the rotated two-line tick labels and the formula box
    # below them need a reserved bottom band that tight_layout does not provide.
    fig.subplots_adjust(left=0.08, right=0.99, top=0.82, bottom=0.42, wspace=0.32)
    finalize_figure(fig, path, formula=FORMULA_TEXT, axes=list(axes), formula_fontsize=6.5)


def _grid_l2_norm(values: torch.Tensor, mask: torch.Tensor, cell_area: float) -> float:
    """Discrete L2 norm of ``values`` over the grid cells selected by ``mask``."""
    return float(torch.sqrt((values[mask] ** 2).sum() * cell_area))


def _evaluate_extension_on_grid(model, ss: torch.Tensor, tt: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """g2 = h_eps on the evaluation grid, in chunks (the split-semigroup g2 is
    an (n_points x n_quad) quadrature; the whole grid at once would allocate
    30000 x 8000 floats)."""
    s_flat = ss.reshape(-1).to(DEVICE).to(torch.get_default_dtype())
    t_flat = tt.reshape(-1).to(DEVICE).to(torch.get_default_dtype())
    pieces = []
    with torch.no_grad():
        for start in range(0, s_flat.numel(), chunk):
            pieces.append(model.g2(s_flat[start:start + chunk], t_flat[start:start + chunk]).double().cpu())
    return torch.cat(pieces).reshape(ss.shape)


def evaluate_run_on_grid(run_dir: Path, epsilon: float) -> dict:
    """Trained trial solution, closed form and terminal-function extension of
    one run on the pilot's evaluation grid, from the saved model only."""
    meta = read_run_metadata(run_dir)
    K, B, r, sigma, T = (meta["contract"][k] for k in ("K", "B", "r", "sigma", "T"))
    s_inf = meta["domain"]["s_inf"]
    corner_window = meta["hyperparameters"]["corner_window"]
    previous_dtype = torch.get_default_dtype()
    if meta["hyperparameters"].get("dtype") == "float64":
        torch.set_default_dtype(torch.float64)
    try:
        model = load_trained_model(run_dir, epsilon, meta)
        eval_result = evaluate_against_closed_form(model, K, B, r, sigma, T, s_inf, corner_window)
        ss, tt = torch.meshgrid(eval_result["s_grid"], eval_result["t_grid"], indexing="ij")
        extension = _evaluate_extension_on_grid(model, ss, tt)
    finally:
        torch.set_default_dtype(previous_dtype)
    s_grid, t_grid = eval_result["s_grid"], eval_result["t_grid"]
    return {
        "s_grid": s_grid, "t_grid": t_grid, "ss": ss, "tt": tt,
        "learned": eval_result["learned"], "reference": eval_result["reference"], "extension": extension,
        "B": B, "T": T, "sigma": sigma, "s_inf": s_inf, "corner_window": corner_window,
        "cell_area": float((s_grid[1] - s_grid[0]) * (t_grid[1] - t_grid[0])),
        "rel_l2_outside_corner_from_summary_refresh": eval_result["rel_l2_outside_corner"],
    }


def window_shape_sweep_for_run(grid: dict) -> dict:
    """For every window family and parameter: relative L2 error of the trained
    solution on the complement of the excluded window, and the excluded area
    fraction (grid fraction; the grid is uniform in s and t)."""
    ss, tt = grid["ss"], grid["tt"]
    tau = grid["T"] - tt
    error, reference = grid["learned"] - grid["reference"], grid["reference"]
    result: dict = {}
    for family, spec in WINDOW_FAMILIES.items():
        result[family] = {}
        for parameter in spec["parameters"]:
            excluded = spec["mask"](ss, tau, grid["B"], grid["sigma"], parameter)
            complement = ~excluded
            numerator = torch.linalg.vector_norm(error[complement])
            denominator = torch.linalg.vector_norm(reference[complement])
            result[family][parameter] = {
                "rel_l2_complement": float(numerator / denominator) if denominator > 0 else float("nan"),
                "excluded_area_fraction": float(excluded.double().mean()),
                "n_grid_points_complement": int(complement.sum()),
            }
    return result


def band_network_contribution_for_run(grid: dict, band_lo: float, band_hi: float) -> dict:
    """L2 norms over the band band_lo < |s-B| < band_hi (all t) of
    Phi_theta - V_DO (trained trial solution) and h_eps - V_DO (extension
    alone), plus the norm of V_DO itself for scale."""
    distance = (grid["ss"] - grid["B"]).abs()
    band = (distance > band_lo) & (distance < band_hi)
    cell_area = grid["cell_area"]
    trial_minus_reference = _grid_l2_norm(grid["learned"] - grid["reference"], band, cell_area)
    extension_minus_reference = _grid_l2_norm(grid["extension"] - grid["reference"], band, cell_area)
    reference_norm = _grid_l2_norm(grid["reference"], band, cell_area)
    return {
        "l2_trial_minus_reference": trial_minus_reference,
        "l2_extension_minus_reference": extension_minus_reference,
        "l2_reference": reference_norm,
        "ratio_trial_over_extension": (trial_minus_reference / extension_minus_reference
                                       if extension_minus_reference > 0 else float("nan")),
        "band_area_fraction": float(band.double().mean()),
        "n_grid_points_band": int(band.sum()),
    }


def model_based_diagnostics(runs: dict[str, dict[int, dict]], epsilon: float, band_lo: float, band_hi: float,
                            out_dir: Path) -> tuple[dict, dict]:
    """Evaluate every run's saved model once and derive both diagnostics.
    The per-run grids are saved under ``evaluation_grids/`` so that figures
    can be patched later without re-evaluating the models."""
    grids_dir = out_dir / "evaluation_grids"
    grids_dir.mkdir(exist_ok=True)
    sweep: dict = {}
    band: dict = {}
    for configuration in CONFIGURATION_LABELS:
        if configuration not in runs:
            continue
        sweep[configuration], band[configuration] = {}, {}
        for seed in sorted(runs[configuration]):
            run_dir = Path(runs[configuration][seed]["run_dir"])
            grid = evaluate_run_on_grid(run_dir, epsilon)
            torch.save({k: v for k, v in grid.items() if k not in ("ss", "tt")}, grids_dir / f"{run_dir.name}.pt")
            sweep[configuration][seed] = window_shape_sweep_for_run(grid)
            band[configuration][seed] = band_network_contribution_for_run(grid, band_lo, band_hi)
            lozenge_at_window = sweep[configuration][seed]["lozenge"].get(grid["corner_window"])
            consistency = ""
            if lozenge_at_window is not None:
                consistency = (f"; lozenge w={grid['corner_window']:g} -> "
                               f"{lozenge_at_window['rel_l2_complement']:.4e} vs summary refresh "
                               f"{grid['rel_l2_outside_corner_from_summary_refresh']:.4e}")
            b = band[configuration][seed]
            logger.info(
                f"  {configuration:<26s} seed {seed}: band ||Phi-V||={b['l2_trial_minus_reference']:.4e}  "
                f"||h_eps-V||={b['l2_extension_minus_reference']:.4e}  ratio={b['ratio_trial_over_extension']:.3f}"
                f"{consistency}"
            )
    return sweep, band


def _median_over_seeds(per_seed: dict, extract) -> float | None:
    values = [extract(v) for v in per_seed.values()]
    values = [v for v in values if v is not None and v == v]
    return statistics.median(values) if values else None


def plot_window_shape_sweep(sweep: dict, path: Path, iters: int, epsilon: float) -> None:
    configurations = list(sweep)
    fig, axes = plt.subplots(1, len(configurations), figsize=(4.6 * len(configurations), 6.0), sharey=True)
    axes = list(axes) if len(configurations) > 1 else [axes]
    handles = {}
    for ax, configuration in zip(axes, configurations):
        per_seed = sweep[configuration]
        for family, spec in WINDOW_FAMILIES.items():
            for seed_result in per_seed.values():
                xs = [seed_result[family][p]["excluded_area_fraction"] for p in spec["parameters"]]
                ys = [seed_result[family][p]["rel_l2_complement"] for p in spec["parameters"]]
                ax.scatter(xs, ys, s=16, alpha=0.3, color=spec["color"], marker=spec["marker"], zorder=2)
            median_x = [_median_over_seeds(per_seed, lambda r, f=family, p=p: r[f][p]["excluded_area_fraction"])
                        for p in spec["parameters"]]
            median_y = [_median_over_seeds(per_seed, lambda r, f=family, p=p: r[f][p]["rel_l2_complement"])
                        for p in spec["parameters"]]
            (line,) = ax.plot(median_x, median_y, "-", color=spec["color"], marker=spec["marker"],
                              markersize=6, markeredgecolor="black", zorder=3, label=spec["label"])
            handles[family] = line
            for x, y, p in zip(median_x, median_y, spec["parameters"]):
                if x is not None and y is not None:
                    ax.annotate(f"{spec['parameter_symbol']}={p:g}", (x, y), textcoords="offset points",
                                xytext=(4, 4), fontsize=6, color=spec["color"])
        ax.set_yscale("log")
        ax.margins(x=0.12)  # room for the parameter annotations at the right end of each curve
        ax.set_xlabel("Excluded area fraction $|N\\cap\\Omega|/|\\Omega|$")
        ax.set_title(CONFIGURATION_LABELS[configuration], fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("Relative $L^2$ error on the complement $\\Omega\\setminus N$")
    # Legend in the band between the x-axis labels and the formula box (which
    # finalize_figure draws at the bottom edge of the figure).
    legend = fig.legend(handles=list(handles.values()), loc="center", bbox_to_anchor=(0.5, 0.17),
                        ncol=len(handles), fontsize=8, title="Excluded window family (shape)")
    fig.suptitle(
        f"Down-and-out put — error vs. excluded area for three window shapes, {iters} iterations, "
        f"$\\varepsilon={epsilon:g}$, corner window excluded from collocation",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.34, wspace=0.12)
    finalize_figure(fig, path, legends=[legend], formula=WINDOW_SHAPE_FORMULA_TEXT, axes=axes, formula_fontsize=6.5)


def plot_band_network_contribution(band: dict, path: Path, iters: int, epsilon: float,
                                   band_lo: float, band_hi: float) -> None:
    configurations = list(band)
    fig, ax = plt.subplots(figsize=(4.8 + 1.4 * len(configurations), 4.6))
    positions = range(len(configurations))
    trial_handle = extension_handle = reference_handle = None
    for position, configuration in zip(positions, configurations):
        per_seed = band[configuration]
        trial = [v["l2_trial_minus_reference"] for v in per_seed.values()]
        extension = [v["l2_extension_minus_reference"] for v in per_seed.values()]
        reference = [v["l2_reference"] for v in per_seed.values()]
        ax.scatter([position] * len(trial), trial, s=28, alpha=0.45, color="tab:blue", zorder=2)
        trial_handle = ax.scatter([position], [statistics.median(trial)], s=70, marker="D", color="tab:blue",
                                  edgecolor="black", zorder=3,
                                  label=r"$\|\Phi_\theta-V_{DO}\|_{L^2(\mathcal{B})}$ (trained; median over seeds)")
        # h_eps does not depend on the network, so it is the same for every seed
        # of a configuration (up to float round-off); draw the median as a segment.
        extension_handle, = ax.plot([position - 0.3, position + 0.3], [statistics.median(extension)] * 2, "--",
                                    color="tab:red", zorder=3,
                                    label=r"$\|h_\varepsilon-V_{DO}\|_{L^2(\mathcal{B})}$ (extension alone)")
        reference_handle, = ax.plot([position - 0.3, position + 0.3], [statistics.median(reference)] * 2, ":",
                                    color="black", zorder=3, label=r"$\|V_{DO}\|_{L^2(\mathcal{B})}$ (scale)")
    ax.set_yscale("log")
    ax.set_xticks(list(positions))
    ax.set_xticklabels([CONFIGURATION_LABELS[c] for c in configurations], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(f"$L^2$ norm over the band ${band_lo:g}<|s-B|<{band_hi:g}$")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title(
        f"Down-and-out put — network contribution in the band ${band_lo:g}<|s-B|<{band_hi:g}$, "
        f"{iters} iterations, $\\varepsilon={epsilon:g}$",
        fontsize=9,
    )
    legend = ax.legend(handles=[trial_handle, extension_handle, reference_handle], loc="upper left",
                       bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.subplots_adjust(left=0.12, right=0.62, top=0.9, bottom=0.3)
    finalize_figure(fig, path, legends=[legend], formula=BAND_FORMULA_TEXT, axes=[ax], formula_fontsize=6.5)


def write_diagnostics_markdown(sweep: dict, band: dict, path: Path, band_lo: float, band_hi: float) -> None:
    lines = ["# Model-based diagnostics (from the saved models, no retraining)", "",
             f"## Network contribution in the band {band_lo:g} < |s-B| < {band_hi:g} (all t)", "",
             "Median [min, max] over seeds. Phi_theta = g1 u_theta + g2 (trained trial solution); "
             "h_eps = g2 (extension alone). Ratio close to 1: the network adds nothing in the band.", "",
             "| Configuration | ||Phi_theta - V_DO|| | ||h_eps - V_DO|| | ratio | ||V_DO|| |", "|---|---|---|---|---|"]
    for configuration, per_seed in band.items():
        def fmt(key):
            values = [v[key] for v in per_seed.values()]
            return f"{statistics.median(values):.3e} [{min(values):.3e}, {max(values):.3e}]"
        lines.append(f"| {CONFIGURATION_LABELS[configuration].replace(chr(10), ' ')} | "
                     f"{fmt('l2_trial_minus_reference')} | {fmt('l2_extension_minus_reference')} | "
                     f"{fmt('ratio_trial_over_extension')} | {fmt('l2_reference')} |")
    lines += ["", "## Window-shape sweep: relative L2 error on the complement of the excluded window", "",
              "Median over seeds of the relative L2 error [min, max], and excluded area fraction (median).", ""]
    for family, spec in WINDOW_FAMILIES.items():
        lines += [f"### {family}: {spec['label'].replace('$', '')}", "",
                  "| Configuration | " + " | ".join(f"{spec['parameter_symbol']}={p:g}" for p in spec["parameters"]) + " |",
                  "|---|" + "|".join("---" for _ in spec["parameters"]) + "|"]
        for configuration, per_seed in sweep.items():
            cells = []
            for p in spec["parameters"]:
                values = [r[family][p]["rel_l2_complement"] for r in per_seed.values()]
                areas = [r[family][p]["excluded_area_fraction"] for r in per_seed.values()]
                cells.append(f"{statistics.median(values):.3e} [{min(values):.3e}, {max(values):.3e}] "
                             f"(area {100 * statistics.median(areas):.1f}%)")
            lines.append(f"| {CONFIGURATION_LABELS[configuration].replace(chr(10), ' ')} | " + " | ".join(cells) + " |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


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
    parser.add_argument("--skip-model-diagnostics", action="store_true",
                        help="Only aggregate the saved summaries; skip the window-shape sweep and the band "
                             "network-contribution diagnostic (which load every saved model).")
    parser.add_argument("--band-lo", type=float, default=0.1,
                        help="Lower bound of the band |s-B| for the network-contribution diagnostic.")
    parser.add_argument("--band-hi", type=float, default=0.3,
                        help="Upper bound of the band |s-B| for the network-contribution diagnostic.")
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

    if args.skip_model_diagnostics:
        logger.info("  --skip-model-diagnostics: window-shape sweep and band diagnostic not computed.")
        return
    logger.info("Model-based diagnostics (loading every saved model; no retraining)")
    logger.info(f"  PyTorch: {torch.__version__}  device: {DEVICE}  band: {args.band_lo:g} < |s-B| < {args.band_hi:g}")
    for family, spec in WINDOW_FAMILIES.items():
        logger.info(f"  window family {family}: {spec['parameter_symbol']} in {spec['parameters']}")
    diagnostics_dir = out_dir / "model_based_diagnostics"
    diagnostics_dir.mkdir(exist_ok=True)
    (diagnostics_dir / "figures").mkdir(exist_ok=True)
    sweep, band = model_based_diagnostics(runs, args.epsilon, args.band_lo, args.band_hi, diagnostics_dir)
    with open(diagnostics_dir / "window_shape_sweep.yaml", "w") as f:
        yaml.dump({"families": {k: {"label": v["label"], "parameters": v["parameters"]}
                                for k, v in WINDOW_FAMILIES.items()},
                   "per_configuration_per_seed": sweep}, f, default_flow_style=False, sort_keys=False)
    with open(diagnostics_dir / "band_network_contribution.yaml", "w") as f:
        yaml.dump({"band_lo": args.band_lo, "band_hi": args.band_hi, "per_configuration_per_seed": band},
                  f, default_flow_style=False, sort_keys=False)
    write_diagnostics_markdown(sweep, band, diagnostics_dir / "diagnostics.md", args.band_lo, args.band_hi)
    logger.info(f"  Diagnostics saved -> {diagnostics_dir} (window_shape_sweep.yaml, "
                f"band_network_contribution.yaml, diagnostics.md, evaluation_grids/)")
    sweep_figure = diagnostics_dir / "figures" / "rel_l2_vs_excluded_area_by_window_shape.png"
    plot_window_shape_sweep(sweep, sweep_figure, args.iters, args.epsilon)
    logger.info(f"  Figure saved -> {sweep_figure}")
    band_figure = diagnostics_dir / "figures" / "band_network_contribution.png"
    plot_band_network_contribution(band, band_figure, args.iters, args.epsilon, args.band_lo, args.band_hi)
    logger.info(f"  Figure saved -> {band_figure}")


if __name__ == "__main__":
    main()
