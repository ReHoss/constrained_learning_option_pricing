r"""Forcing energy of the extensions of the variable-coefficient cells (P1--P3).

Pre-registration:
``documents/methodology/2026-09-29_preregistration_variable_coefficient_split.md``
(sections 4--6).  On the circle, with the band-limited Bernoulli datum
:math:`g_K(x) = \sum_{k=1}^{K} \cos(kx)/(\pi^2 k^2)` (break point :math:`x^\star = 0`)
and the variable-coefficient generators

* LV2: :math:`a(x)\partial_{xx} + (r - a(x))\partial_x - r`,
  :math:`a(x) = 0.125\,(1 + \varepsilon\cos(x - \pi/4))`, :math:`r = 0.03`;
* LV4: :math:`-\beta(x)\partial_x^4 + 1.3\,\partial_x - 0.4`,
  :math:`\beta(x) = 0.05\,(1 + \varepsilon\cos(x - \pi/4))`,

the script evaluates, for each extension :math:`h`, the strip forcing energy

.. math::

    E(K) = \lVert \partial_t h + L h \rVert^2_{L^2(Q)}
         = 2\pi \sum_{|k| \le K+1} \int_0^T |\hat F_k(t)|^2\, dt ,

**in closed form** (no quadrature): the coefficient harmonic couples each
wavenumber to its two neighbours, so :math:`\hat F_k` is a combination of three
exponentials in time, whose squared modulus integrates exactly
(:mod:`learning_option_pricing.pde.variable_coefficient_periodic`).

Predictions under test (heuristic scaling, pre-registered): the energy stays
bounded as :math:`K \to \infty` iff :math:`p < 3/2 + \nu`, with :math:`2p` the order and
:math:`\nu` the vanishing order at :math:`x^\star` of the principal-coefficient
mismatch (:math:`\nu = 0` frozen at the mean, :math:`\nu \ge 1` frozen at :math:`x^\star`):

* P1: ``split_frozen_mean`` bounded on LV2, growing (linearly) on LV4;
* P2: ``split_frozen_singular`` bounded on LV2 and LV4;
* P3: ``constant_in_time`` and ``convex_raw`` growing like :math:`K` (LV2) and
  :math:`K^5` (LV4).

Checks performed before any value is saved (a failure raises):

* at :math:`\varepsilon = 0` the variable-coefficient closed forms reproduce the
  constant-coefficient library closed forms (``SplitSemigroupExtension``,
  ``ConstantInTimeExtension``, ``ConvexRawExtension``) to a relative tolerance
  of :math:`10^{-10}`;
* the Fourier--Galerkin reference solution of each trained cell
  (:math:`\varepsilon \in \{0.25, 0.75\}`) at truncation :math:`N` is compared with
  :math:`2N` at the eleven evaluation times of the trained runs; the largest
  relative deviation is recorded (it is reported, and a value above the
  stated tolerance invalidates the error measurements of that cell).

Artefacts (saved before plotting; ``--replot RUN_DIR`` rebuilds the figure
from them alone): ``energies.npz``, ``summary.yaml``, ``run_metadata.json``,
``command.txt``, ``run.log``, ``variable_coefficient_forcing_energy.png``.
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "exp_split_extension_trained"))
import _split_extension_catalogue as catalogue  # noqa: E402
from _figure_layout import finalize_figure  # noqa: E402
from learning_option_pricing.pde import (  # noqa: E402
    ConstantCoefficientGenerator,
    ConstantInTimeExtension,
    ConvexRawExtension,
    PeriodisedBernoulliDatum,
    SplitSemigroupExtension,
    symmetric_wavenumber_band,
    total_strip_forcing_squared,
)
from learning_option_pricing.pde.variable_coefficient_periodic import (  # noqa: E402
    BandLimitedDatum,
    GalerkinReferenceSolution,
    build_variable_coefficient_extension,
    galerkin_convergence_deviation,
    generator_from_specification,
    strip_forcing_energy,
)
from learning_option_pricing.utils.run_context import (  # noqa: E402
    collect_run_metadata,
    configure_cli_script_logging,
    find_repo_root,
    init_logging,
    log_parsed_args,
    script_data_dir,
    utc_timestamp,
    write_command_txt,
    write_json,
)

LOGGER = logging.getLogger(Path(__file__).stem)

# Smoke-test guard: the pre-registered range reaches K = 1024; a largest band
# edge below 256 without --debug is rejected.
SMOKE_TEST_MAXIMUM_BAND_EDGE_THRESHOLD = 256

EPSILON_ZERO_RELATIVE_TOLERANCE = 1.0e-10
# Tolerance on the N versus 2N deviation of the reference solution; a larger
# value invalidates the trained error measurements of the cell (section 6).
REFERENCE_CONVERGENCE_TOLERANCE = 1.0e-8

TERMINAL_TIME = 1.0
EVALUATION_GRID_SIZE = 1024          # the runner's evaluation grid
EVALUATION_TIME_SLICE_COUNT = 11     # the runner's evaluation time slices
VARIANT_NAMES = ("constant_in_time", "convex_raw", "split_frozen_singular", "split_frozen_mean")
ORDER_CELL_BUILDERS = {
    2: catalogue._local_volatility_cell,
    4: catalogue._variable_biharmonic_cell,
}
TRAINED_AMPLITUDE_RATIOS = tuple(catalogue.VARIABLE_COEFFICIENT_AMPLITUDE_RATIOS)

MAIN_FIGURE_FILENAME = "variable_coefficient_forcing_energy.png"
ENERGIES_FILENAME = "energies.npz"
SUMMARY_FILENAME = "summary.yaml"


def cell_configuration(order: int, amplitude_ratio: float) -> dict:
    """The catalogue configuration (single source of the coefficients)."""
    return ORDER_CELL_BUILDERS[order](amplitude_ratio)


def constant_coefficient_reference_energy(variant_name, cell_conf, band_edge) -> float:
    r"""At :math:`\varepsilon = 0`: the energy of the same extension through the
    constant-coefficient library closed forms (independent code path)."""
    coefficients = {
        int(order): (value["constant"] if isinstance(value, dict) else float(value))
        for order, value in cell_conf["generator_coefficients"].items()
    }
    generator = ConstantCoefficientGenerator(coefficients=coefficients, name="epsilon_zero")
    datum = PeriodisedBernoulliDatum(regularity_index=1)
    principal_order = max(o for o in coefficients if o % 2 == 0 and o > 0)
    if variant_name == "constant_in_time":
        extension = ConstantInTimeExtension(datum, generator, TERMINAL_TIME)
    elif variant_name == "convex_raw":
        extension = ConvexRawExtension(datum, generator, TERMINAL_TIME)
    else:  # both freezings coincide at epsilon = 0: the principal-part split
        extension = SplitSemigroupExtension(datum, generator, (principal_order,), TERMINAL_TIME)
    return total_strip_forcing_squared(extension, symmetric_wavenumber_band(band_edge))


def measure_energies(orders, amplitude_ratios, band_edges) -> dict:
    """Energy table ``[order][ratio][variant] -> array over band_edges``."""
    table: dict = {}
    for order in orders:
        table[order] = {}
        for ratio in amplitude_ratios:
            cell_conf = cell_configuration(order, ratio)
            generator = generator_from_specification(
                cell_conf["generator_coefficients"], f"LV{order}_eps{ratio}"
            )
            table[order][ratio] = {}
            for variant in VARIANT_NAMES:
                energies = []
                for band_edge in band_edges:
                    datum = BandLimitedDatum(PeriodisedBernoulliDatum(1), int(band_edge))
                    extension = build_variable_coefficient_extension(
                        variant, generator, datum, float(cell_conf["corner_point"]), TERMINAL_TIME
                    )
                    energy = strip_forcing_energy(extension)
                    if ratio == 0.0:
                        reference = constant_coefficient_reference_energy(
                            variant, cell_conf, int(band_edge)
                        )
                        deviation = abs(energy - reference) / abs(reference)
                        if deviation > EPSILON_ZERO_RELATIVE_TOLERANCE:
                            raise ValueError(
                                f"epsilon = 0 check failed: LV{order} {variant} K={band_edge}: "
                                f"variable-coefficient energy {energy:.12e} against the "
                                f"constant-coefficient closed form {reference:.12e} "
                                f"(relative deviation {deviation:.3e})"
                            )
                    energies.append(energy)
                table[order][ratio][variant] = np.asarray(energies)
                LOGGER.info(
                    "LV%d eps=%.2f %-22s E(K) = %s", order, ratio, variant,
                    ", ".join(f"{e:.4e}" for e in energies),
                )
    return table


def local_slopes(band_edges, energies) -> list[float]:
    """Log-log slopes between consecutive band edges (measured, not fitted)."""
    k = np.asarray(band_edges, dtype=np.float64)
    e = np.asarray(energies, dtype=np.float64)
    return [float(np.log(e[i + 1] / e[i]) / np.log(k[i + 1] / k[i])) for i in range(len(k) - 1)]


def check_reference_solutions(orders, galerkin_band, refined_band) -> dict:
    """Largest relative deviation of the reference at N versus 2N, per trained cell."""
    x = np.linspace(0.0, 2.0 * math.pi, EVALUATION_GRID_SIZE, endpoint=False)
    times = np.linspace(0.0, TERMINAL_TIME, EVALUATION_TIME_SLICE_COUNT)
    results: dict = {}
    for order in orders:
        for ratio in TRAINED_AMPLITUDE_RATIOS:
            cell_conf = cell_configuration(order, ratio)
            generator = generator_from_specification(
                cell_conf["generator_coefficients"], f"LV{order}_eps{ratio}"
            )
            datum = BandLimitedDatum(
                PeriodisedBernoulliDatum(1), int(cell_conf["truncation_wavenumber"])
            )
            start = time.perf_counter()
            deviation = galerkin_convergence_deviation(
                generator, datum, galerkin_band, refined_band, TERMINAL_TIME, times, x
            )
            key = f"lv{order}_eps{ratio:g}"
            results[key] = {
                "relative_deviation_N_versus_2N": deviation,
                "galerkin_band": galerkin_band,
                "refined_band": refined_band,
                "within_tolerance": bool(deviation <= REFERENCE_CONVERGENCE_TOLERANCE),
                "tolerance": REFERENCE_CONVERGENCE_TOLERANCE,
                "wall_time_s": time.perf_counter() - start,
            }
            LOGGER.info(
                "reference LV%d eps=%.2f: N=%d vs N=%d relative deviation %.3e (%s) in %.1f s",
                order, ratio, galerkin_band, refined_band, deviation,
                "within tolerance" if deviation <= REFERENCE_CONVERGENCE_TOLERANCE else "ABOVE TOLERANCE",
                results[key]["wall_time_s"],
            )
    return results


def study_reference_bands(orders, bands) -> dict:
    r"""Per-time relative deviations of the Galerkin reference across truncations.

    For each trained cell, the reference is computed at every truncation of
    ``bands`` on the evaluation grid and times, and the relative :math:`\ell^2`
    deviation of each truncation from every other is recorded per time.  A
    deviation that shrinks as both truncations decrease, and grows with the
    larger truncation, points to round-off in the matrix exponential of the
    larger (higher-norm) matrix; a deviation that shrinks as the truncation
    grows points to truncation error.  This separates the two causes before a
    truncation is retained for the trained cells.
    """
    x = np.linspace(0.0, 2.0 * math.pi, EVALUATION_GRID_SIZE, endpoint=False)
    times = np.linspace(0.0, TERMINAL_TIME, EVALUATION_TIME_SLICE_COUNT)
    study: dict = {}
    for order in orders:
        for ratio in TRAINED_AMPLITUDE_RATIOS:
            cell_conf = cell_configuration(order, ratio)
            generator = generator_from_specification(
                cell_conf["generator_coefficients"], f"LV{order}_eps{ratio}"
            )
            datum = BandLimitedDatum(
                PeriodisedBernoulliDatum(1), int(cell_conf["truncation_wavenumber"])
            )
            fields = {}
            for band in bands:
                start = time.perf_counter()
                reference = GalerkinReferenceSolution(generator, datum, int(band), TERMINAL_TIME)
                fields[int(band)] = np.stack([
                    reference.field(x, np.full_like(x, float(t))) for t in times
                ])
                LOGGER.info("band study LV%d eps=%.2f: N=%d computed in %.1f s",
                            order, ratio, band, time.perf_counter() - start)
            pairs = {}
            for i, first in enumerate(sorted(fields)):
                for second in sorted(fields)[i + 1:]:
                    per_time = [
                        float(np.linalg.norm(fields[first][j] - fields[second][j])
                              / np.linalg.norm(fields[second][j]))
                        for j in range(len(times))
                    ]
                    pairs[f"N{first}_vs_N{second}"] = {
                        "per_time": per_time,
                        "maximum": max(per_time),
                        "time_of_maximum": float(times[int(np.argmax(per_time))]),
                    }
                    LOGGER.info("band study LV%d eps=%.2f: N=%d vs N=%d max deviation %.3e at t=%.1f",
                                order, ratio, first, second, max(per_time),
                                float(times[int(np.argmax(per_time))]))
            study[f"lv{order}_eps{ratio:g}"] = {
                "times": [float(t) for t in times],
                "pairs": pairs,
            }
    return study


VARIANT_DISPLAY = {
    "constant_in_time": ("#1f77b4", "Constant in time"),
    "convex_raw": ("#2ca02c", "Convex raw"),
    "split_frozen_singular": ("#d62728", r"Split frozen at $x^\star$"),
    "split_frozen_mean": ("#9467bd", "Split frozen at the mean"),
}
AMPLITUDE_MARKERS = {0.0: "^", 0.25: "o", 0.75: "s"}


def render_main_figure(run_directory: Path) -> Path:
    """Rebuild the figure from ``energies.npz`` and ``summary.yaml`` only."""
    saved = np.load(run_directory / ENERGIES_FILENAME)
    with open(run_directory / SUMMARY_FILENAME, "r", encoding="utf-8") as handle:
        summary = yaml.safe_load(handle)
    band_edges = saved["band_edges"].astype(np.float64)
    orders = summary["parameters"]["orders"]
    shown_ratios = [r for r in summary["parameters"]["amplitude_ratios"] if r in AMPLITUDE_MARKERS]

    fig, axes = plt.subplots(1, len(orders), figsize=(12.0, 5.2), squeeze=False)
    axes = axes[0]
    for ax, order in zip(axes, orders):
        for variant in VARIANT_NAMES:
            colour, label = VARIANT_DISPLAY[variant]
            for ratio in shown_ratios:
                energies = saved[f"energy__lv{order}__eps{ratio:g}__{variant}"]
                ax.loglog(
                    band_edges, energies, "-", color=colour,
                    marker=AMPLITUDE_MARKERS[ratio], markersize=4, linewidth=1.2,
                    label=f"{label}, " + rf"$\varepsilon={ratio:g}$",
                )
        # Dotted guides: the pre-registered growth exponents (annotation only).
        k_guide = np.array([band_edges[0], band_edges[-1]])
        exponents = (1,) if order == 2 else (1, 5)
        for exponent in exponents:
            anchor = saved[f"energy__lv{order}__eps0.75__split_frozen_mean"][-1] if (
                order == 4 and exponent == 1
            ) else saved[f"energy__lv{order}__eps0.75__constant_in_time"][-1]
            ax.loglog(
                k_guide, anchor * (k_guide / band_edges[-1]) ** exponent, ":",
                color="0.35", linewidth=1.0,
            )
            ax.annotate(
                rf"$\propto K^{{{exponent}}}$", xy=(k_guide[0], anchor * (k_guide[0] / band_edges[-1]) ** exponent),
                fontsize=8, color="0.35", xytext=(4, 0), textcoords="offset points",
            )
        ax.set_title(rf"LV{order} (order {order})", fontsize=10)
        ax.set_xlabel(r"Datum band edge $K$")
        ax.set_ylabel(r"Forcing energy $\|\partial_t h+\mathcal{L}h\|^2_{L^2(Q)}$")
        ax.grid(True, which="both", alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.13),
        ncol=4, fontsize=6.5, frameon=True,
    )
    fig.tight_layout(rect=[0.0, 0.33, 1.0, 0.98])
    figure_path = run_directory / MAIN_FIGURE_FILENAME
    finalize_figure(
        fig, figure_path, legends=[legend], axes=list(axes),
        formula=(
            r"$\|\partial_t h+\mathcal{L}h\|^2_{L^2(Q)}=2\pi\sum_{|k|\leq K+1}\int_0^T|\hat F_k(t)|^2\,dt$"
            r" in closed form;  split: $h=e^{(T-t)A}g_K$, $A=c_{2p}(x_0)\,\partial_x^{2p}$ with "
            r"$x_0=x^\star=0$ or $c_{2p}$ replaced by its mean"
            "\n"
            r"LV2: $a(x)=0.125\,(1+\varepsilon\cos(x-\pi/4))$;  LV4: $\beta(x)=0.05\,"
            r"(1+\varepsilon\cos(x-\pi/4))$;  $g_K=\sum_{k\leq K}\cos(kx)/(\pi^2k^2)$;  "
            r"$\varepsilon=0$: constant coefficients ($G_2$, $G_3$)"
            "\n"
            r"dotted: pre-registered growth exponents (guides, not fits)"
        ),
        formula_fontsize=7.5,
    )
    return figure_path


def parse_arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--band-edges", type=int, nargs="+",
                        default=[32, 64, 128, 256, 512, 1024])
    parser.add_argument("--amplitude-ratios", type=float, nargs="+",
                        default=[0.0, 0.1, 0.25, 0.5, 0.75])
    parser.add_argument("--orders", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--galerkin-band", type=int, default=512)
    parser.add_argument("--galerkin-refined-band", type=int, default=1024)
    parser.add_argument("--skip-reference-check", action="store_true",
                        help="Skip the N versus 2N check of the reference solution.")
    parser.add_argument("--reference-bands", type=int, nargs="+", default=None,
                        help="Also compare the reference across these truncations, "
                             "per evaluation time (diagnostic of truncation versus "
                             "round-off error).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Master seed (recorded; the computation is deterministic).")
    parser.add_argument("--debug", action="store_true",
                        help="Mark as a smoke run (prefixes the folder with _debug_).")
    parser.add_argument("--replot", type=Path, default=None,
                        help="Rebuild the figure from an existing run directory.")
    arguments = parser.parse_args(argv)
    if arguments.replot is None and not arguments.debug and (
        max(arguments.band_edges) < SMOKE_TEST_MAXIMUM_BAND_EDGE_THRESHOLD
    ):
        parser.error(
            f"largest band edge {max(arguments.band_edges)} is below the smoke-test "
            f"threshold {SMOKE_TEST_MAXIMUM_BAND_EDGE_THRESHOLD}; pass --debug for a smoke run"
        )
    if 0.0 not in arguments.amplitude_ratios:
        parser.error("the amplitude ratios must include 0 (the constant-coefficient anchor)")
    return arguments


def main(argv=None) -> int:
    arguments = parse_arguments(argv)
    if arguments.replot is not None:
        configure_cli_script_logging(verbose=False)
        figure_path = render_main_figure(arguments.replot.resolve())
        LOGGER.info("Replotted figure from saved artefacts: %s", figure_path)
        return 0

    start_time = time.perf_counter()
    debug_prefix = "_debug_" if arguments.debug else ""
    config_tag = f"Kmax{max(arguments.band_edges)}_N{arguments.galerkin_band}"
    run_directory = script_data_dir(__file__) / f"{debug_prefix}{utc_timestamp()}_{config_tag}"
    run_directory.mkdir(parents=True, exist_ok=False)
    init_logging(run_dir=run_directory)
    LOGGER.info("Full command line: %s", " ".join(sys.argv))
    LOGGER.info(
        "Runtime: Python %s | numpy %s | matplotlib %s | torch not used",
        sys.version.split()[0], np.__version__, matplotlib.__version__,
    )
    log_parsed_args(LOGGER, arguments)
    LOGGER.info("Master seed %d (no random number generator is consumed)", arguments.seed)
    run_metadata = collect_run_metadata(
        run_dir=run_directory,
        repo_root=find_repo_root(Path(__file__)),
        script_name=Path(__file__).stem,
        command=list(sys.argv),
        params=dict(sorted(vars(arguments).items(), key=lambda item: item[0])),
    )
    write_json(run_directory / "run_metadata.json", run_metadata)
    write_command_txt(run_directory / "command.txt", list(sys.argv))
    LOGGER.info("Git commit %s (dirty: %s)", run_metadata["git"].get("commit"),
                run_metadata["git"].get("dirty"))

    band_edges = np.asarray(sorted(arguments.band_edges), dtype=np.int64)
    table = measure_energies(arguments.orders, arguments.amplitude_ratios, band_edges)

    reference_checks = {}
    if not arguments.skip_reference_check:
        reference_checks = check_reference_solutions(
            arguments.orders, arguments.galerkin_band, arguments.galerkin_refined_band
        )
    reference_band_study = {}
    if arguments.reference_bands is not None:
        reference_band_study = study_reference_bands(arguments.orders, arguments.reference_bands)

    payload = {"band_edges": band_edges}
    summary_cells: dict = {}
    for order in arguments.orders:
        for ratio in arguments.amplitude_ratios:
            for variant in VARIANT_NAMES:
                energies = table[order][ratio][variant]
                payload[f"energy__lv{order}__eps{ratio:g}__{variant}"] = energies
                summary_cells[f"lv{order}__eps{ratio:g}__{variant}"] = {
                    "energies": [float(e) for e in energies],
                    "local_log_log_slopes": local_slopes(band_edges, energies),
                    "ratio_largest_over_second_largest_band_edge": float(energies[-1] / energies[-2]),
                }
    summary = {
        "generated_by": Path(__file__).name,
        "parameters": {
            "band_edges": [int(k) for k in band_edges],
            "amplitude_ratios": [float(r) for r in arguments.amplitude_ratios],
            "orders": [int(o) for o in arguments.orders],
            "phase": catalogue.VARIABLE_COEFFICIENT_PHASE,
            "terminal_time": TERMINAL_TIME,
            "epsilon_zero_relative_tolerance": EPSILON_ZERO_RELATIVE_TOLERANCE,
        },
        "epsilon_zero_check": "passed (every energy at epsilon = 0 equals the "
                              "constant-coefficient closed form within tolerance)",
        "cells": summary_cells,
        "reference_solution_convergence": reference_checks,
        "reference_band_study": reference_band_study,
    }
    np.savez(run_directory / ENERGIES_FILENAME, **payload)
    with open(run_directory / SUMMARY_FILENAME, "w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)
    LOGGER.info("Saved artefacts: %s, %s", ENERGIES_FILENAME, SUMMARY_FILENAME)
    figure_path = render_main_figure(run_directory)
    LOGGER.info("Wrote main figure: %s", figure_path)
    LOGGER.info("Total wall-clock time: %.2f s; run directory: %s",
                time.perf_counter() - start_time, run_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
