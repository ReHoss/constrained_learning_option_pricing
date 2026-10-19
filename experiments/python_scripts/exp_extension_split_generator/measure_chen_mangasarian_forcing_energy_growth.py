r"""Growth of the band-truncated forcing energy of the Chen--Mangasarian extension.

Pre-registration:
``documents/methodology/2026-10-19_preregistration_chen_mangasarian_forcing_and_early_residual.md``
(part 1).

Question.  The Chen--Mangasarian (algebraic-kernel) extension of the paper
convolves the datum with the kernel of Fourier multiplier :math:`m(z) = z K_1(z)`
at a width that grows linearly away from the datum slice,
:math:`\varepsilon_0 = \sqrt{2 \nu T}`.  It leaves the component of wavenumber
:math:`k` almost unchanged over a time :math:`T/(\varepsilon_0 |k|)`, during
which its forcing is that of the constant-in-time extension.  The derivation of
the pre-registration gives, for the paper's datum
(:math:`|c_k| = 1/(2\pi^2 k^2)`),

.. math::

    E(K) = \frac{\gamma_{2p}^2 I}{\pi^3 \varepsilon_0}
    \sum_{k \le K} k^{4p-5}\,(1 + o(1)),
    \qquad I = \int_0^\infty (z K_1(z))^2\, dz = \frac{3\pi^2}{32},

so that :math:`E(K)` grows like :math:`\log K` at order two and like
:math:`K^4` at order four, whereas the highest-order split keeps :math:`E(K)`
bounded and the constant-in-time extension grows like :math:`K^{4p-3}`.

Cells and extensions.  The two circle cells of the paper, ``g2_bernoulli_full``
(:math:`p = 1`) and ``g3_bernoulli_full`` (:math:`p = 2`), with the extensions
built by the runner's ``build_closed_form_extension``, so that they are the
objects of the trained experiments: the Chen--Mangasarian extension, the
highest-order split and the constant-in-time extension.

Measured quantity.  The band-truncated forcing energy
:math:`E(K) = 2\pi \sum_{0 < |k| \le K} \int_0^T |\widehat{Lh}(k, t)|^2\, dt` at
dyadic band edges, accumulated by
``measure_forcing_energy_threshold_grid.energy_curve_at_dyadic_edges`` and
cross-checked at the smallest edge against
:func:`learning_option_pricing.pde.total_strip_forcing_squared`.

Predictions (pre-registered):

* F1: on ``g2``, every dyadic increment :math:`E(2^{j+1}) - E(2^j)`,
  :math:`15 \le j \le 19`, is positive and lies within ``INCREMENT_TOLERANCE`` of
  :math:`\ln 2\, \gamma_2^2 I / (\pi^3 \varepsilon_0)`;
* F2: on ``g3``, the log--log slope over :math:`[2^{16}, 2^{20}]` lies within
  ``SLOPE_TOLERANCE`` of 4 and :math:`E(2^{20}) / (C_2\, 2^{80})` within
  ``RATIO_TOLERANCE`` of 1, :math:`C_2 = \gamma_4^2 I / (4 \pi^3 \varepsilon_0)`;
* F3: the highest-order split has a relative increment over the last dyadic step
  below ``BOUNDED_RELATIVE_INCREMENT_TOLERANCE`` on both cells;
* F4: the constant-in-time extension has a log--log slope within
  ``SLOPE_TOLERANCE`` of :math:`4p - 3` on both cells.

Artefacts (saved before plotting; ``--replot RUN_DIR`` rebuilds the figure from
them alone): ``energy_curves.npz``, ``summary.yaml``, ``run_metadata.json``,
``command.txt``, ``run.log``, ``forcing_energy_against_band_edge.png``.

The computation is deterministic; the master seed is logged for convention
compliance, and no random number generator is consumed.
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

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "exp_split_extension_trained"))
from _figure_layout import finalize_figure  # noqa: E402
from measure_forcing_energy_threshold_grid import (  # noqa: E402
    cross_check_against_library_reference,
    energy_curve_at_dyadic_edges,
)
import _split_extension_catalogue as catalogue  # noqa: E402
from ablation_split_extension_trained import (  # noqa: E402
    build_closed_form_extension,
    build_problem,
    chen_mangasarian_initial_smoothing_scale,
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

# Smoke-test guard: the pre-registered run reaches the band edge 2^20; a largest
# band edge below 2^16 without --debug is rejected mechanically.
SMOKE_TEST_LOG2_MAXIMUM_BAND_EDGE_THRESHOLD = 16

# Pre-registered tolerances (section 3 of the pre-registration).
INCREMENT_TOLERANCE = 2.0e-2
SLOPE_TOLERANCE = 5.0e-2
RATIO_TOLERANCE = 2.0e-2
BOUNDED_RELATIVE_INCREMENT_TOLERANCE = 1.0e-3
# Dyadic steps j -> j + 1 tested by F1, and the fit range of F2 and F4 (log2).
F1_FIRST_STEP_LOG2 = 15
FIT_MINIMUM_LOG2 = 16

# Integral of (z K_1(z))^2 over (0, +infinity).
SQUARED_MULTIPLIER_INTEGRAL = 3.0 * math.pi**2 / 32.0

CELLS = ("g2_bernoulli_full", "g3_bernoulli_full")
CHEN_MANGASARIAN = "graded_chen_mangasarian"
CONSTANT_IN_TIME = "constant_in_time"
# The highest-order split has its own catalogue name on the order-four cell.
HIGHEST_ORDER_SPLIT = {"g2_bernoulli_full": "split_diffusion", "g3_bernoulli_full": "split_principal"}
EXTENSION_LABELS = {
    CHEN_MANGASARIAN: "Chen-Mangasarian (algebraic kernel)",
    "split": "Highest-order split",
    CONSTANT_IN_TIME: "Raw constant extension",
}
EXTENSION_COLOURS = {CHEN_MANGASARIAN: "#696969", "split": "#185d82", CONSTANT_IN_TIME: "#b2182b"}

ENERGY_CURVES_FILENAME = "energy_curves.npz"
SUMMARY_FILENAME = "summary.yaml"
ENERGY_FIGURE_FILENAME = "forcing_energy_against_band_edge.png"


def highest_even_order_coefficient(problem: dict) -> tuple[int, float]:
    """The half order p and the modulus of the coefficient of order 2p of the cell."""
    coefficients = problem["generator_coefficients"]
    top_order = max(order for order in coefficients if order % 2 == 0 and order > 0)
    return top_order // 2, abs(float(coefficients[top_order]))


def predicted_chen_mangasarian_constant(problem: dict) -> dict:
    r"""The constant of the pre-registered asymptotic law of the cell.

    :math:`\gamma_{2p}^2 I / (\pi^3 \varepsilon_0)` multiplies
    :math:`\sum_{k \le K} k^{4p-5}`: at :math:`p = 1` it is the coefficient of
    :math:`\ln K`, at :math:`p = 2` the coefficient of :math:`K^4` is a quarter of it.
    """
    half_order, top_coefficient = highest_even_order_coefficient(problem)
    variant = catalogue.variant_by_name(problem["cell_name"], CHEN_MANGASARIAN)
    initial_smoothing_scale = chen_mangasarian_initial_smoothing_scale(variant, problem)
    sum_coefficient = top_coefficient**2 * SQUARED_MULTIPLIER_INTEGRAL / (
        math.pi**3 * initial_smoothing_scale
    )
    return {
        "half_order": half_order,
        "top_coefficient_modulus": top_coefficient,
        "initial_smoothing_scale": initial_smoothing_scale,
        "sum_coefficient": sum_coefficient,
    }


def log_log_slope(band_edges: np.ndarray, energy_values: np.ndarray, minimum_edge: int) -> float:
    """Least-squares slope of log E against log K over the edges at or above ``minimum_edge``."""
    selected = band_edges >= minimum_edge
    return float(np.polyfit(np.log(band_edges[selected]), np.log(energy_values[selected]), 1)[0])


def analyse_cell(cell_name: str, problem: dict, band_edges: np.ndarray, curves: dict) -> dict:
    """The pre-registered verdicts F1-F4 of one cell."""
    prediction = predicted_chen_mangasarian_constant(problem)
    half_order = prediction["half_order"]
    chen = curves[CHEN_MANGASARIAN]
    split = curves["split"]
    constant = curves[CONSTANT_IN_TIME]
    summary = {
        "prediction": prediction,
        "energy_at_band_edge_4096": {
            key: float(values[band_edges == 4096][0]) for key, values in curves.items()
            if np.any(band_edges == 4096)
        },
        "energy_at_largest_band_edge": {key: float(values[-1]) for key, values in curves.items()},
    }
    if half_order == 1:
        predicted_increment = math.log(2.0) * prediction["sum_coefficient"]
        increments = {}
        for step in range(F1_FIRST_STEP_LOG2, int(np.log2(band_edges[-1]))):
            lower, upper = 2**step, 2 ** (step + 1)
            increment = float(chen[band_edges == upper][0] - chen[band_edges == lower][0])
            increments[f"{step}->{step + 1}"] = {
                "increment": increment,
                "relative_deviation_from_prediction": abs(increment - predicted_increment)
                / predicted_increment,
            }
        summary["f1"] = {
            "predicted_increment": predicted_increment,
            "increments": increments,
            "holds": bool(increments) and all(
                value["increment"] > 0.0
                and value["relative_deviation_from_prediction"] <= INCREMENT_TOLERANCE
                for value in increments.values()
            ),
        }
    if half_order == 2:
        predicted_leading_constant = prediction["sum_coefficient"] / 4.0
        slope = log_log_slope(band_edges, chen, 2**FIT_MINIMUM_LOG2)
        ratio = float(chen[-1] / (predicted_leading_constant * float(band_edges[-1]) ** 4))
        summary["f2"] = {
            "predicted_leading_constant": predicted_leading_constant,
            "fitted_log_log_slope": slope,
            "ratio_measured_over_predicted_at_largest_band_edge": ratio,
            "holds": abs(slope - 4.0) <= SLOPE_TOLERANCE and abs(ratio - 1.0) <= RATIO_TOLERANCE,
        }
    split_increment = float((split[-1] - split[-2]) / split[-1])
    summary["f3"] = {
        "relative_increment_over_last_dyadic_step": split_increment,
        "holds": split_increment < BOUNDED_RELATIVE_INCREMENT_TOLERANCE,
    }
    constant_slope = log_log_slope(band_edges, constant, 2**FIT_MINIMUM_LOG2)
    summary["f4"] = {
        "predicted_slope": 4 * half_order - 3,
        "fitted_log_log_slope": constant_slope,
        "holds": abs(constant_slope - (4 * half_order - 3)) <= SLOPE_TOLERANCE,
    }
    LOGGER.info("%s: p = %d, epsilon_0 = %.6f, sum coefficient %.6e", cell_name, half_order,
                prediction["initial_smoothing_scale"], prediction["sum_coefficient"])
    for key in ("f1", "f2", "f3", "f4"):
        if key in summary:
            LOGGER.info("%s: %s %s", cell_name, key.upper(), summary[key])
    return summary


def render_energy_figure(run_directory: Path) -> Path:
    """E(K) against K for the three extensions on both cells, with the predicted laws."""
    curves = np.load(run_directory / ENERGY_CURVES_FILENAME)
    summary = yaml.safe_load((run_directory / SUMMARY_FILENAME).read_text())
    band_edges = curves["dyadic_band_edges"].astype(np.float64)
    figure, axes = plt.subplots(1, len(CELLS), figsize=(10.0, 4.0))
    titles = {"g2_bernoulli_full": "(a) Order two, $G_1$", "g3_bernoulli_full": "(b) Order four, $G_2$"}
    for axis, cell_name in zip(axes, CELLS):
        for key in (CHEN_MANGASARIAN, "split", CONSTANT_IN_TIME):
            # Solid: closed-form energies of the extensions, the measured objects here.
            axis.loglog(band_edges, curves[f"energy_curve__{cell_name}__{key}"], "-o", markersize=3,
                        color=EXTENSION_COLOURS[key], label=EXTENSION_LABELS[key])
        prediction = summary["cells"][cell_name]["prediction"]
        positive_edges = band_edges[band_edges >= 2**8]
        if prediction["half_order"] == 1:
            anchor = curves[f"energy_curve__{cell_name}__{CHEN_MANGASARIAN}"][band_edges == 2**F1_FIRST_STEP_LOG2][0]
            law = anchor + prediction["sum_coefficient"] * np.log(positive_edges / 2**F1_FIRST_STEP_LOG2)
            law_label = r"Predicted law $C\,\ln K$ + const"
        else:
            law = prediction["sum_coefficient"] / 4.0 * positive_edges**4
            law_label = r"Predicted law $\frac{C}{4}K^4$"
        # Dashed: the analytical asymptotic law of the pre-registration.
        axis.loglog(positive_edges, law, "--", color="black", linewidth=1.0, label=law_label)
        axis.set_title(titles[cell_name])
        axis.set_xlabel("Band edge $K$")
        axis.set_ylabel("Forcing energy $E(K)$")
        axis.grid(True, which="both", alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    legend = figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4,
                           fontsize=8, frameon=True)
    formula = (r"$E(K)=2\pi\sum_{0<|k|\leq K}\int_0^T|\widehat{\mathcal{L}h}(k,t)|^2\,dt$;"
               r"  Chen-Mangasarian: $\hat h(k,t)=m(\varepsilon(t)|k|)\,c_k$, $m(z)=zK_1(z)$,"
               r" $\varepsilon$ linear in time, $\varepsilon_0=\sqrt{2\nu T}$;"
               "\n"
               r"predicted $E(K)=\frac{\gamma_{2p}^2 I}{\pi^3\varepsilon_0}\sum_{k\leq K}k^{4p-5}(1+o(1))$,"
               r" $I=\int_0^\infty (zK_1(z))^2dz=3\pi^2/32$, $C=\gamma_{2p}^2I/(\pi^3\varepsilon_0)$")
    figure.tight_layout(rect=[0, 0.12, 1, 1])
    path = run_directory / ENERGY_FIGURE_FILENAME
    finalize_figure(figure, path, legends=[legend], formula=formula, axes=list(axes))
    return path


def parse_arguments(argv=None) -> argparse.Namespace:
    """Parse and validate the command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log2-minimum-band-edge", type=int, default=4)
    parser.add_argument("--log2-maximum-band-edge", type=int, default=20)
    parser.add_argument("--chunk-length", type=int, default=2**13,
                        help="maximal number of positive wavenumbers per chunk (default 2**13; the "
                        "Chen-Mangasarian quadrature evaluates 352 nodes per wavenumber)")
    parser.add_argument("--seed", type=int, default=0,
                        help="master seed, logged only: the measurement consumes no random number generator")
    parser.add_argument("--debug", action="store_true",
                        help="prepend '_debug_' to the output folder name (smoke runs)")
    parser.add_argument("--replot", type=Path, default=None, metavar="RUN_DIR",
                        help="rebuild the figure from the saved artefacts of a run directory")
    arguments = parser.parse_args(argv)
    if arguments.replot is not None:
        return arguments
    if arguments.log2_maximum_band_edge < SMOKE_TEST_LOG2_MAXIMUM_BAND_EDGE_THRESHOLD and not arguments.debug:
        parser.error(
            f"smoke-test guard: --log2-maximum-band-edge {arguments.log2_maximum_band_edge} < "
            f"{SMOKE_TEST_LOG2_MAXIMUM_BAND_EDGE_THRESHOLD} requires --debug"
        )
    if not 1 <= arguments.log2_minimum_band_edge < arguments.log2_maximum_band_edge:
        parser.error("--log2-minimum-band-edge must lie in [1, --log2-maximum-band-edge)")
    return arguments


def main(argv=None) -> int:
    """Run the measurement (or the replot path) and write the artefacts."""
    arguments = parse_arguments(argv)
    if arguments.replot is not None:
        configure_cli_script_logging(verbose=False)
        LOGGER.info("Replotted figure from saved artefacts: %s", render_energy_figure(arguments.replot.resolve()))
        return 0

    start_time = time.perf_counter()
    debug_prefix = "_debug_" if arguments.debug else ""
    run_directory = script_data_dir(__file__) / f"{debug_prefix}{utc_timestamp()}_log2K{arguments.log2_maximum_band_edge}"
    run_directory.mkdir(parents=True, exist_ok=False)
    init_logging(run_dir=run_directory)
    LOGGER.info("Full command line: %s", " ".join(sys.argv))
    LOGGER.info("Runtime: Python %s | numpy %s | matplotlib %s | torch not used (closed-form measurement)",
                sys.version.split()[0], np.__version__, matplotlib.__version__)
    log_parsed_args(LOGGER, arguments)
    LOGGER.info("Master seed %d (deterministic measurement: no random number generator is consumed)", arguments.seed)
    LOGGER.info("Pre-registered tolerances: increment %.1e, slope %.1e, ratio %.1e, bounded increment %.1e",
                INCREMENT_TOLERANCE, SLOPE_TOLERANCE, RATIO_TOLERANCE, BOUNDED_RELATIVE_INCREMENT_TOLERANCE)
    run_metadata = collect_run_metadata(
        run_dir=run_directory, repo_root=find_repo_root(Path(__file__)), script_name=Path(__file__).stem,
        command=list(sys.argv), params=dict(sorted(vars(arguments).items(), key=lambda item: item[0])),
    )
    write_json(run_directory / "run_metadata.json", run_metadata)
    write_command_txt(run_directory / "command.txt", list(sys.argv))
    LOGGER.info("Git commit %s (dirty: %s)", run_metadata["git"].get("commit"), run_metadata["git"].get("dirty"))

    band_edges = 2 ** np.arange(arguments.log2_minimum_band_edge, arguments.log2_maximum_band_edge + 1, dtype=np.int64)
    LOGGER.info("Dyadic band edges 2^%d .. 2^%d; chunk length %d",
                arguments.log2_minimum_band_edge, arguments.log2_maximum_band_edge, arguments.chunk_length)
    energy_curves: dict[str, np.ndarray] = {"dyadic_band_edges": band_edges}
    cell_summaries: dict[str, dict] = {}
    for cell_name in CELLS:
        problem = build_problem(cell_name)
        LOGGER.info("%s: generator coefficients %s, T = %g", cell_name, problem["generator_coefficients"],
                    problem["terminal_time"])
        curves = {}
        for key, variant_name in ((CHEN_MANGASARIAN, CHEN_MANGASARIAN), ("split", HIGHEST_ORDER_SPLIT[cell_name]),
                                  (CONSTANT_IN_TIME, CONSTANT_IN_TIME)):
            extension_start = time.perf_counter()
            extension = build_closed_form_extension(catalogue.variant_by_name(cell_name, variant_name), problem)
            values = energy_curve_at_dyadic_edges(extension, band_edges, arguments.chunk_length)
            discrepancy = cross_check_against_library_reference(extension, int(band_edges[0]), float(values[0]))
            curves[key] = values
            energy_curves[f"energy_curve__{cell_name}__{key}"] = values
            LOGGER.info("%s / %-24s E(2^%d) = %.6e | E(2^%d) = %.6e | cross-check %.1e | %.1f s", cell_name,
                        variant_name, arguments.log2_minimum_band_edge, values[0], arguments.log2_maximum_band_edge,
                        values[-1], discrepancy, time.perf_counter() - extension_start)
        cell_summaries[cell_name] = analyse_cell(cell_name, problem, band_edges, curves)

    verdicts = {key: [cell_summaries[cell][key]["holds"] for cell in CELLS if key in cell_summaries[cell]]
                for key in ("f1", "f2", "f3", "f4")}
    summary = {
        "parameters": {**{key: value for key, value in vars(arguments).items() if key != "replot"},
                       "increment_tolerance": INCREMENT_TOLERANCE, "slope_tolerance": SLOPE_TOLERANCE,
                       "ratio_tolerance": RATIO_TOLERANCE,
                       "bounded_relative_increment_tolerance": BOUNDED_RELATIVE_INCREMENT_TOLERANCE,
                       "squared_multiplier_integral": SQUARED_MULTIPLIER_INTEGRAL},
        "cells": cell_summaries,
        "verdicts": {key: all(values) if values else None for key, values in verdicts.items()},
    }
    np.savez(run_directory / ENERGY_CURVES_FILENAME, **energy_curves)
    with open(run_directory / SUMMARY_FILENAME, "w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)
    LOGGER.info("Saved %s and %s under %s", ENERGY_CURVES_FILENAME, SUMMARY_FILENAME, run_directory)
    LOGGER.info("Saved figure %s", render_energy_figure(run_directory))
    LOGGER.info("Verdicts: %s", summary["verdicts"])
    LOGGER.info("Total wall-clock time %.2f s", time.perf_counter() - start_time)
    return 0


if __name__ == "__main__":
    sys.exit(main())
