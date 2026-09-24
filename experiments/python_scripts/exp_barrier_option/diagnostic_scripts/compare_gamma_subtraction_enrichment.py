r"""Gamma of the corner enrichment against Gamma of the exact subtraction, near the corner.

Reference: sections 15 to 17 of ``documents/methodology/barrier_option.md``, and
the working note's Sections 5.1 (Method 1) and 5.2 (Method 2).

Section 17.4 measures that the two analytic corner resolutions differ by the
interior forcing of their singular part: :math:`\mathcal L^{BS}S_{\mathrm{sub}}`
is exactly zero, :math:`\mathcal L^{BS}S_{\mathrm{enr}}` is of order one and
supported in the cutoff transition band :math:`[B+\delta_0, B+\delta_1]`.  The
second price derivative is the quantity most exposed to that forcing, and it is
the one to reduce if the enrichment is to be improved.

The existing corner-zoom profile figure
(``compare_corner_treatments_profiles.py``) plots the two analytic treatments
together with the smoothing run, whose second derivative oscillates over four
decades in the same panel; on that scale the two analytic curves lie on top of
each other and their difference cannot be read.  This script drops the smoothing
run and resolves the two analytic treatments alone, over every master seed of
the batch, so that the question "where, and by how much, is the enrichment's
Gamma worse, and is the gap reproducible across seeds" has an answer.

Computed, in float64, from the saved models only (no retraining):

- :math:`\partial_{ss}\Phi_\theta(s,t)` of each treatment and each seed on a
  dense price grid of the corner region, against the closed-form
  :math:`\partial_{ss}V_{DO}`;
- the signed and absolute Gamma error, per seed and median over seeds;
- :math:`|\mathcal L^{BS}S_{\mathrm{enr}}|`, the enrichment's own interior
  forcing, on the same grid, to test whether the Gamma error is supported where
  the forcing is;
- the :math:`L^2` norm of the Gamma error on four price bands against calendar
  time, which is the quantitative statement of where an improvement would pay.

Outputs (under ``data/compare_gamma_subtraction_enrichment/<timestamp>/``):

- ``figures/gamma_profiles_corner.png`` -- Gamma, signed error and absolute
  error along :math:`s` at fixed calendar times, with the forcing overlaid.
- ``figures/gamma_error_heatmaps.png`` -- :math:`|\partial_{ss}\Phi_\theta -
  \partial_{ss}V_{DO}|` over :math:`(s,t)` for the two treatments, and their
  ratio.
- ``figures/gamma_error_by_band.png`` -- band norms against calendar time.
- ``curves.pt`` and ``summary.yaml``; ``--replot`` rebuilds every figure from
  ``curves.pt``.
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
from matplotlib.colors import LogNorm, SymLogNorm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from learning_option_pricing.utils.figure_layout import finalize_figure  # noqa: E402
from learning_option_pricing.utils.run_context import script_data_dir  # noqa: E402
from compare_corner_treatments_profiles import (  # noqa: E402
    TREATMENT_COLOURS, corner_treatment_of, reference_profiles, trained_profiles,
)
from pilot_down_and_out_put import DEVICE, load_trained_model, read_run_metadata  # noqa: E402

logger = logging.getLogger("compare_gamma_subtraction_enrichment")

DEFAULT_CONFIGURATIONS = ["subtraction_blackscholes", "enrichment_blackscholes"]
DEFAULT_TIMES = (0.0, 0.5, 0.9, 0.99)
#: Price bands of the band-norm figure, as offsets from the barrier, closed on
#: the right. The first two are the cutoff's plateau and its transition strip;
#: the third runs to the strike and the fourth beyond it.
DEFAULT_BAND_EDGES = (0.0, 0.1, 0.3, 0.4, 0.6)

TREATMENT_DISPLAY = {
    "subtraction": "Exact subtraction (Method 1)",
    "enrichment": "Corner enrichment (Method 2)",
    "smoothing": r"Smoothing (corner layer, $\varepsilon=0.1$)",
}

FORMULA_GAMMA = (
    r"$\partial_{ss}\Phi_\theta$: two nested autograd passes on $g_1u_\theta$ + the closed-form "
    r"second derivative of $g_2$;  $\partial_{ss}V_{DO}$ = "
    r"reiner\_rubinstein\_down\_and\_out\_put\_gamma (closed form).  Pointwise, float64, no quadrature."
    "\n"
    r"Gamma error $e_\Gamma(s,t)=\partial_{ss}\Phi_\theta(s,t)-\partial_{ss}V_{DO}(s,t)$;  band norm "
    r"$\|e_\Gamma\|_{L^2(\mathcal{B})}(t)=(\int_{\mathcal{B}}e_\Gamma(s,t)^2\,\mathrm{d}s)^{1/2}$ on the "
    r"trapezoidal rule of the plotted grid."
    "\n"
    r"Forcing $\mathcal{L}^{BS}S_{\rm enr}$, $S_{\rm enr}=\chi(s)\,\Delta\,\mathrm{erf}(\xi)$, "
    r"$\xi=\ln(s/B)/(\sigma\sqrt{2(T-t)})$, $\Delta=K-B$: zero for the subtraction (Proposition 4), "
    r"nonzero for the enrichment (Proposition 5)."
    "\n"
    r"Dotted verticals: $s=B$, $s=B+\delta_0$, $s=B+\delta_1$, $s=K$. Faint curves: individual master "
    r"seeds; solid: median over seeds. The smoothing runs are deliberately absent: their "
    r"$\partial_{ss}\Phi_\theta$ spans four decades here and would set the scale."
)


def collect_models(aggregation_summary: dict, configurations: list[str], seeds: list[int] | None):
    """``{configuration: {seed: (model, metadata)}}`` from an aggregation's summary."""
    collected: dict[str, dict[int, tuple]] = {}
    for configuration in configurations:
        entry = aggregation_summary["configurations"].get(configuration)
        if entry is None:
            raise SystemExit(f"configuration {configuration!r} is not in the aggregation summary; "
                             f"available: {sorted(aggregation_summary['configurations'])}")
        collected[configuration] = {}
        for seed, run_dir in sorted(entry["runs"].items()):
            if seeds is not None and int(seed) not in seeds:
                continue
            run_path = Path(run_dir)
            metadata = read_run_metadata(run_path)
            epsilon = metadata["hyperparameters"]["epsilons"][0]
            model = load_trained_model(run_path, epsilon, metadata)
            collected[configuration][int(seed)] = (model, metadata)
            logger.info(f"  {configuration:<28} seed {seed}: {run_path.name}")
    return collected


def evaluate(collected: dict, s_grid: torch.Tensor, times: list[float], t_grid: torch.Tensor,
             contract: dict) -> dict:
    """Gamma and its error, per configuration and seed, on the slice grid (at
    ``times``) and on the ``(s, t)`` grid of the heatmaps."""
    K, B, r, sigma, T = (contract[k] for k in ("K", "B", "r", "sigma", "T"))
    curves: dict = {"s_grid": s_grid, "times": list(times), "t_grid": t_grid,
                    "contract": contract, "configurations": {}}

    curves["reference_slices"] = {}
    for t_value in times:
        t_column = torch.full_like(s_grid, float(t_value))
        curves["reference_slices"][t_value] = reference_profiles(s_grid, t_column, K, B, r, sigma, T)[2]

    reference_surface = torch.stack([
        reference_profiles(s_grid, torch.full_like(s_grid, float(t)), K, B, r, sigma, T)[2]
        for t in t_grid.tolist()])
    curves["reference_surface"] = reference_surface

    for configuration, per_seed in collected.items():
        entry: dict = {"slices": {}, "surface": {}, "forcing_slices": {}, "forcing_surface": None}
        for seed, (model, _) in sorted(per_seed.items()):
            entry["slices"][seed] = {}
            for t_value in times:
                t_column = torch.full_like(s_grid, float(t_value))
                entry["slices"][seed][t_value] = trained_profiles(model, s_grid, t_column)[2]
            entry["surface"][seed] = torch.stack([
                trained_profiles(model, s_grid, torch.full_like(s_grid, float(t)))[2]
                for t in t_grid.tolist()])
        # The interior forcing of the singular part is a property of the
        # construction, identical across seeds: read it from any one model.
        g2 = next(iter(per_seed.values()))[0].g2
        if hasattr(g2, "singular_residual"):
            for t_value in times:
                t_column = torch.full_like(s_grid, float(t_value))
                with torch.no_grad():
                    entry["forcing_slices"][t_value] = g2.singular_residual(s_grid, t_column)
            with torch.no_grad():
                entry["forcing_surface"] = torch.stack([
                    g2.singular_residual(s_grid, torch.full_like(s_grid, float(t)))
                    for t in t_grid.tolist()])
        curves["configurations"][configuration] = entry
    return curves


def _median_over_seeds(per_seed: dict) -> torch.Tensor:
    return torch.stack([per_seed[seed] for seed in sorted(per_seed)]).median(dim=0).values


def _mark_price_landmarks(ax, contract: dict, delta0: float, delta1: float) -> None:
    for position in (contract["B"], contract["B"] + delta0, contract["B"] + delta1, contract["K"]):
        ax.axvline(position, color="grey", linestyle=":", lw=0.9)


def plot_profiles(curves: dict, path: Path, delta0: float, delta1: float) -> None:
    """Gamma, signed error and absolute error along ``s`` at fixed calendar
    times, with the enrichment's interior forcing overlaid on the error row."""
    contract, s_grid, times = curves["contract"], curves["s_grid"], curves["times"]
    s = s_grid.numpy()
    fig, axes = plt.subplots(3, len(times), figsize=(4.6 * len(times), 11.5), squeeze=False)
    handles: list = []
    for column, t_value in enumerate(times):
        reference = curves["reference_slices"][t_value].numpy()
        top, middle, bottom = axes[0, column], axes[1, column], axes[2, column]
        # Drawn above the trained curves: both treatments track it closely here,
        # and underneath it would be hidden by whichever is plotted last.
        (reference_line,) = top.plot(s, reference, "k--", lw=1.8, zorder=5,
                                     label=r"$\partial_{ss}V_{DO}$ (closed form)")
        for configuration, entry in curves["configurations"].items():
            colour = TREATMENT_COLOURS[corner_treatment_of(configuration)]
            label = TREATMENT_DISPLAY[corner_treatment_of(configuration)]
            per_seed = entry["slices"][None] if None in entry["slices"] else entry["slices"]
            median = _median_over_seeds({seed: values[t_value] for seed, values in per_seed.items()}).numpy()
            for seed, values in sorted(per_seed.items()):
                top.plot(s, values[t_value].numpy(), lw=0.7, color=colour, alpha=0.3)
                middle.plot(s, (values[t_value] - curves["reference_slices"][t_value]).numpy(),
                            lw=0.7, color=colour, alpha=0.3)
                bottom.semilogy(s, np.abs((values[t_value] - curves["reference_slices"][t_value]).numpy()),
                                lw=0.7, color=colour, alpha=0.3)
            (line,) = top.plot(s, median, lw=1.7, color=colour, label=label)
            middle.plot(s, median - reference, lw=1.7, color=colour)
            bottom.semilogy(s, np.abs(median - reference), lw=1.7, color=colour)
            if column == 0:
                handles.append(line)

        forcing = next((entry["forcing_slices"].get(t_value)
                        for entry in curves["configurations"].values()
                        if corner_treatment_of_entry(curves, entry) == "enrichment"
                        and entry["forcing_slices"]), None)
        if forcing is not None:
            twin = bottom.twinx()
            magnitude = np.abs(forcing.numpy())
            # Beyond the outer cutoff radius the forcing is zero up to underflow;
            # left unbounded, the right axis would span fifty decades of values
            # that are not a measurement. Six decades below the maximum is the
            # range in which the forcing is actually present.
            ceiling = magnitude.max() if magnitude.size else 1.0
            floor = ceiling * 1e-6 if ceiling > 0.0 else 1e-6
            (forcing_line,) = twin.semilogy(
                s, np.ma.masked_where(magnitude < floor, magnitude),
                lw=1.3, color="#7B3294", linestyle=":",
                label=r"$|\mathcal{L}^{BS}S_{\rm enr}|$ (right axis, auxiliary; "
                      r"below $10^{-6}$ of its maximum: not drawn)")
            twin.set_ylim(floor, ceiling * 2.0)
            twin.set_ylabel(r"$|\mathcal{L}^{BS}S_{\rm enr}|$", color="#7B3294", fontsize=8)
            twin.tick_params(axis="y", labelcolor="#7B3294", labelsize=7)
            if column == 0:
                handles.append(forcing_line)

        middle.axhline(0.0, color="grey", lw=0.6)
        middle.set_yscale("symlog", linthresh=1e-4)
        for ax in (top, middle, bottom):
            _mark_price_landmarks(ax, contract, delta0, delta1)
            ax.grid(True, which="both", alpha=0.3)
            ax.set_xlim(float(s_grid[0]), float(s_grid[-1]))
        top.set_yscale("symlog", linthresh=0.1)
        top.set_title(f"$t = {t_value:g}$  ($T-t = {contract['T'] - t_value:g}$)", fontsize=10)
        bottom.set_xlabel("Underlying price $s$")
        if column == 0:
            top.set_ylabel(r"$\partial_{ss}\Phi_\theta(s,t)$ (symlog)")
            middle.set_ylabel(r"$e_\Gamma=\partial_{ss}\Phi_\theta-\partial_{ss}V_{DO}$ (symlog)")
            bottom.set_ylabel(r"$|e_\Gamma|$")
    handles.insert(0, reference_line)
    legend = fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.135),
                        ncol=min(4, len(handles)), fontsize=8)
    fig.suptitle("Down-and-out put — Gamma of the two analytic corner resolutions near the corner "
                 "(all master seeds)", fontsize=11)
    fig.subplots_adjust(left=0.07, right=0.94, top=0.94, bottom=0.21, wspace=0.34, hspace=0.22)
    finalize_figure(fig, path, legends=[legend], formula=FORMULA_GAMMA,
                    axes=list(axes.reshape(-1)), formula_fontsize=7)


def corner_treatment_of_entry(curves: dict, entry: dict) -> str:
    """The corner treatment of a configuration entry, by identity lookup."""
    for configuration, candidate in curves["configurations"].items():
        if candidate is entry:
            return corner_treatment_of(configuration)
    return "smoothing"


def plot_error_heatmaps(curves: dict, path: Path, delta0: float, delta1: float) -> None:
    """``|e_Gamma|`` over ``(s, t)`` for each treatment (shared logarithmic
    colour scale) and the ratio of the two."""
    contract, s_grid, t_grid = curves["contract"], curves["s_grid"], curves["t_grid"]
    extent = [float(s_grid[0]), float(s_grid[-1]), float(t_grid[0]), float(t_grid[-1])]
    errors = {}
    for configuration, entry in curves["configurations"].items():
        median = _median_over_seeds(entry["surface"])
        errors[configuration] = (median - curves["reference_surface"]).abs()
    finite = torch.cat([value[value > 0].reshape(-1) for value in errors.values()])
    vmin, vmax = float(finite.quantile(0.01)), float(finite.max())

    fig, axes = plt.subplots(1, len(errors) + 1, figsize=(5.6 * (len(errors) + 1), 5.4))
    for ax, (configuration, error) in zip(axes, errors.items()):
        image = ax.imshow(error.numpy(), origin="lower", aspect="auto", cmap="magma",
                          norm=LogNorm(vmin=max(vmin, vmax * 1e-8), vmax=vmax), extent=extent)
        ax.set_title(f"$|e_\\Gamma|$ — {TREATMENT_DISPLAY[corner_treatment_of(configuration)]}", fontsize=9)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        _mark_price_landmarks(ax, contract, delta0, delta1)
        ax.set_xlabel("Underlying price $s$")
        ax.set_ylabel("Calendar time $t$")

    names = list(errors)
    if len(names) == 2:
        floor = max(vmax * 1e-10, torch.finfo(torch.float64).tiny)
        ratio = (errors[names[1]].clamp_min(floor) / errors[names[0]].clamp_min(floor)).log10()
        limit = float(ratio.abs().quantile(0.99))
        image = axes[-1].imshow(ratio.numpy(), origin="lower", aspect="auto", cmap="RdBu_r",
                                norm=SymLogNorm(linthresh=0.1, vmin=-limit, vmax=limit), extent=extent)
        axes[-1].set_title(r"$\log_{10}\left(|e_\Gamma|_{\rm enr}\,/\,|e_\Gamma|_{\rm sub}\right)$"
                           "\n" + "red: the enrichment is worse there", fontsize=9)
        fig.colorbar(image, ax=axes[-1], fraction=0.046, pad=0.03)
        _mark_price_landmarks(axes[-1], contract, delta0, delta1)
        axes[-1].set_xlabel("Underlying price $s$")
        axes[-1].set_ylabel("Calendar time $t$")
    fig.suptitle("Down-and-out put — where the Gamma error of each analytic corner resolution is "
                 "(median over master seeds)", fontsize=11)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.86, bottom=0.34, wspace=0.26)
    finalize_figure(fig, path, formula=FORMULA_GAMMA, axes=list(axes), formula_fontsize=7)


def band_norms(curves: dict, band_edges: tuple) -> dict:
    """``{configuration: {band: {seed: tensor over t}}}``: the L2 norm in ``s``
    of the Gamma error on each price band, against calendar time."""
    contract, s_grid = curves["contract"], curves["s_grid"]
    offsets = s_grid - contract["B"]
    bands = {}
    for lower, upper in zip(band_edges[:-1], band_edges[1:]):
        mask = (offsets >= lower) & (offsets < upper)
        if mask.any():
            bands[(lower, upper)] = mask
    result: dict = {}
    for configuration, entry in curves["configurations"].items():
        result[configuration] = {band: {} for band in bands}
        for seed, surface in sorted(entry["surface"].items()):
            error = surface - curves["reference_surface"]
            for band, mask in bands.items():
                s_band = s_grid[mask]
                integrand = error[:, mask] ** 2
                norm = torch.trapezoid(integrand, s_band, dim=1).clamp_min(0.0).sqrt()
                result[configuration][band][seed] = norm
    return result


def plot_band_norms(curves: dict, norms: dict, path: Path, delta0: float, delta1: float) -> None:
    """One panel per price band: the band norm of the Gamma error against ``t``."""
    contract, t_grid = curves["contract"], curves["t_grid"]
    t = t_grid.numpy()
    bands = list(next(iter(norms.values())))
    fig, axes = plt.subplots(1, len(bands), figsize=(4.6 * len(bands), 5.0), squeeze=False)
    handles: list = []
    for column, band in enumerate(bands):
        ax = axes[0, column]
        lower, upper = band
        for configuration, per_band in norms.items():
            colour = TREATMENT_COLOURS[corner_treatment_of(configuration)]
            per_seed = per_band[band]
            for seed, values in sorted(per_seed.items()):
                ax.semilogy(t, values.numpy(), lw=0.7, color=colour, alpha=0.3)
            median = _median_over_seeds(per_seed).numpy()
            (line,) = ax.semilogy(t, median, lw=1.8, color=colour,
                                  label=TREATMENT_DISPLAY[corner_treatment_of(configuration)])
            if column == 0:
                handles.append(line)
        strike_offset = contract["K"] - contract["B"]
        role = {(0.0, delta0): r"cutoff plateau, $\chi\equiv1$",
                (delta0, delta1): r"cutoff transition, $\chi'\neq0$"}.get(band)
        if role is None:
            role = (r"beyond the cutoff ($\chi\equiv0$), up to the strike"
                    if upper <= strike_offset else r"beyond the cutoff, past the strike")
        ax.set_title(rf"$s-B\in[{lower:g}, {upper:g})$" "\n" + role, fontsize=9)
        ax.set_xlabel("Calendar time $t$")
        ax.grid(True, which="both", alpha=0.3)
    axes[0, 0].set_ylabel(r"$\|e_\Gamma(\cdot,t)\|_{L^2(\mathcal{B})}$")
    legend = fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.20), ncol=2, fontsize=8)
    fig.suptitle("Down-and-out put — Gamma error per price band against calendar time "
                 "(faint: master seeds; solid: median)", fontsize=11)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.85, bottom=0.34, wspace=0.26)
    finalize_figure(fig, path, legends=[legend], formula=FORMULA_GAMMA,
                    axes=list(axes.reshape(-1)), formula_fontsize=7)


def measured_summary(curves: dict, norms: dict) -> dict:
    """Band norms summarised over time and seeds, and the ratio between the two
    treatments, which is the number to reduce."""
    summary: dict = {"bands": {}}
    configurations = list(norms)
    for band in next(iter(norms.values())):
        key = f"s-B in [{band[0]:g}, {band[1]:g})"
        summary["bands"][key] = {}
        for configuration in configurations:
            stacked = torch.stack([norms[configuration][band][seed]
                                   for seed in sorted(norms[configuration][band])])
            median_over_seeds = stacked.median(dim=0).values
            summary["bands"][key][configuration] = {
                "median_over_seeds_time_mean": float(median_over_seeds.mean()),
                "median_over_seeds_max_over_time": float(median_over_seeds.max()),
                "across_seed_min": float(stacked.min()),
                "across_seed_max": float(stacked.max()),
            }
        if len(configurations) == 2:
            subtraction, enrichment = configurations
            if corner_treatment_of(subtraction) == "enrichment":
                subtraction, enrichment = enrichment, subtraction
            reference_value = summary["bands"][key][subtraction]["median_over_seeds_time_mean"]
            candidate = summary["bands"][key][enrichment]["median_over_seeds_time_mean"]
            summary["bands"][key]["ratio_enrichment_over_subtraction"] = (
                float(candidate / reference_value) if reference_value > 0.0 else None)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aggregation-dir", type=str, default=None,
                        help="Output directory of aggregate_terminal_function_comparison.py, whose "
                             "summary.yaml names the run directory of every configuration and seed.")
    parser.add_argument("--configurations", nargs="+", type=str, default=DEFAULT_CONFIGURATIONS,
                        help="Configuration keys to compare. The default is the two analytic corner "
                             "resolutions at the Black-Scholes terminal profile; the smoothing runs "
                             "are left out on purpose (their second derivative sets the scale).")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Master seeds to include (default: every seed of the aggregation).")
    parser.add_argument("--s-max", type=float, default=1.2,
                        help="Upper end of the price grid; the default reaches past the strike.")
    parser.add_argument("--n-s", type=int, default=800, help="Price nodes of the grid.")
    parser.add_argument("--n-t", type=int, default=200, help="Calendar-time nodes of the heatmaps.")
    parser.add_argument("--times", nargs="+", type=float, default=list(DEFAULT_TIMES),
                        help="Calendar times of the profile figure.")
    parser.add_argument("--band-edges", nargs="+", type=float, default=list(DEFAULT_BAND_EDGES),
                        help="Price-band edges of the band-norm figure, as offsets from the barrier.")
    parser.add_argument("--enrichment-delta0", type=float, default=0.1,
                        help="Inner cutoff radius, for the figure landmarks.")
    parser.add_argument("--enrichment-delta1", type=float, default=0.3,
                        help="Outer cutoff radius, for the figure landmarks.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory override.")
    parser.add_argument("--replot", type=str, default=None, metavar="OUT_DIR",
                        help="Rebuild the figures from a previous run's curves.pt; computes nothing.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("matplotlib.mathtext").setLevel(logging.WARNING)
    torch.set_default_dtype(torch.float64)

    if args.replot is not None:
        out_dir = Path(args.replot)
        curves = torch.load(out_dir / "curves.pt", weights_only=False)
    else:
        if args.aggregation_dir is None:
            raise SystemExit("--aggregation-dir is required unless --replot is given.")
        out_dir = Path(args.out_dir) if args.out_dir else script_data_dir(__file__) / (
            datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + "_".join(args.configurations))
        out_dir.mkdir(parents=True, exist_ok=True)
        curves = None

    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(out_dir / "diagnostic.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(file_handler)

    logger.info("Gamma of the corner enrichment against Gamma of the exact subtraction")
    logger.info(f"  Output directory: {out_dir.resolve()}")
    logger.info(f"  Log file (follow in real time): {(out_dir / 'diagnostic.log').resolve()}")
    logger.info(f"  Command: {' '.join([Path(sys.argv[0]).name] + sys.argv[1:])}")
    logger.info(f"  Host: {platform.node()}  Python: {platform.python_version()}  "
                f"PyTorch: {torch.__version__}  device: {DEVICE}")
    logger.info("  Evaluation dtype: float64. No RNG, no retraining: the saved models are read.")

    if curves is None:
        with open(Path(args.aggregation_dir) / "summary.yaml") as handle:
            aggregation_summary = yaml.safe_load(handle)
        logger.info(f"  Aggregation: {Path(args.aggregation_dir).resolve()}")
        collected = collect_models(aggregation_summary, args.configurations, args.seeds)
        first_metadata = next(iter(next(iter(collected.values())).values()))[1]
        contract = {k: first_metadata["contract"][k] for k in ("K", "B", "r", "sigma", "T")}
        logger.info("  Contract: " + ", ".join(f"{k}={v:g}" for k, v in contract.items()))
        logger.info(f"  Price grid: {args.n_s} nodes on ({contract['B']:g}, {args.s_max:g}); "
                    f"time grid: {args.n_t} nodes on [0, {contract['T']:g}), terminal slice excluded "
                    f"(the Gamma error vanishes there by construction)")
        logger.info(f"  Cutoff radii marked on the figures: delta0={args.enrichment_delta0:g}, "
                    f"delta1={args.enrichment_delta1:g}")
        s_grid = torch.linspace(contract["B"], args.s_max, args.n_s)
        # The terminal slice t = T is left out: there both the trial solution and
        # the closed form equal the payoff, whose second derivative vanishes away
        # from the strike, so the error is exactly zero by construction and would
        # put a spurious collapse to underflow at the right edge of every
        # logarithmic panel. The grid therefore covers [0, T).
        t_grid = torch.linspace(0.0, contract["T"], args.n_t + 1)[:-1]
        curves = evaluate(collected, s_grid, list(args.times), t_grid, contract)
        curves["band_edges"] = list(args.band_edges)
        curves["delta0"], curves["delta1"] = args.enrichment_delta0, args.enrichment_delta1
        curves["generated"] = datetime.now(timezone.utc).isoformat()
        torch.save(curves, out_dir / "curves.pt")
        logger.info(f"  Curves saved -> {(out_dir / 'curves.pt').resolve()} "
                    f"(--replot rebuilds every figure from this file)")

    delta0, delta1 = curves.get("delta0", 0.1), curves.get("delta1", 0.3)
    norms = band_norms(curves, tuple(curves.get("band_edges", DEFAULT_BAND_EDGES)))
    summary = measured_summary(curves, norms)
    logger.info("Gamma-error band norms (median over seeds, averaged over calendar time):")
    for band, per_configuration in summary["bands"].items():
        pieces = [f"{name}={values['median_over_seeds_time_mean']:.4e}"
                  for name, values in per_configuration.items() if isinstance(values, dict)]
        ratio = per_configuration.get("ratio_enrichment_over_subtraction")
        logger.info(f"    {band:<24} " + "  ".join(pieces)
                    + (f"  ratio={ratio:.2f}" if ratio is not None else ""))
    with open(out_dir / "summary.yaml", "w") as handle:
        yaml.safe_dump({"contract": curves["contract"], "delta0": delta0, "delta1": delta1,
                        "configurations": list(curves["configurations"]), "measured": summary},
                       handle, sort_keys=False)
    logger.info(f"  Summary saved -> {(out_dir / 'summary.yaml').resolve()}")

    plot_profiles(curves, out_dir / "figures" / "gamma_profiles_corner.png", delta0, delta1)
    plot_error_heatmaps(curves, out_dir / "figures" / "gamma_error_heatmaps.png", delta0, delta1)
    plot_band_norms(curves, norms, out_dir / "figures" / "gamma_error_by_band.png", delta0, delta1)
    logger.info(f"  Figures -> {(out_dir / 'figures').resolve()}")


if __name__ == "__main__":
    main()
