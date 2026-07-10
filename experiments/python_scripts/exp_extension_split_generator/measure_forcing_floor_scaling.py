r"""Measured operator-channel floor law on the circle (prediction P2, Propositions 3--5).

Proposition under test.  On the circle :math:`[0, 2\pi)` with the Fourier
convention :math:`c_k = \frac{1}{2\pi}\int_0^{2\pi} g(x) e^{-ikx}\,dx`, the
band-truncated operator-channel floor of a generator with symbol
:math:`a(k)` acting on a terminal datum :math:`g` is

.. math::

    \mathrm{floor}(K) = 2\pi \sum_{0 < |k| \le K} |a(k)|^2 |c_k|^2 .

For a periodised Bernoulli datum (single break point, regularity index
:math:`\rho \in \{0, 1, 2\}`, jump :math:`J` of the :math:`\rho`-th
derivative) and a generator of maximal order :math:`2p` with principal
constant :math:`a_0`, the predicted law is: with growth exponent
:math:`e = 4p - 2\rho - 1`,

* if :math:`e > 0`:
  :math:`\mathrm{floor}(K) = C_{\mathrm{pred}} K^{e} (1 + o(1))` as
  :math:`K \to \infty`, with
  :math:`C_{\mathrm{pred}} = a_0^2 J^2 / (\pi e)`;
* if :math:`e < 0`: the floor saturates to its finite total sum (a plateau
  is reported instead of a slope).

Measurement grid.  Generators G1 (advection--diffusion--reaction,
:math:`p = 1`) and G3 (biharmonic--advection--reaction, :math:`p = 2`)
crossed with :math:`\rho \in \{0, 1, 2\}`, plus (G2 Black--Scholes
log-price, :math:`\rho = 1`); expected exponents: G1 gives
:math:`e = 3, 1, -1`, G3 gives :math:`e = 7, 5, 3`, G2 at :math:`\rho = 1`
gives :math:`e = 1`.  The square-wave datum on G1 is included as the
multi-singularity extension case (two break points): the single-break-point
law does not apply verbatim, so that curve is measured only, with no
prediction line.

Measurement policy.  Every floor value is computed from the exact analytic
Fourier coefficients evaluated over integer wavenumber arrays (never from
an FFT of sampled values), by vectorised cumulative sums over the band
:math:`0 < |k| \le 2^{22}` accumulated in chunks so the peak memory stays
bounded; the accumulation is cross-checked against the library function
``operator_channel_floor`` at the smallest band edge and a relative
discrepancy above :math:`10^{-12}` raises ``ValueError``.  For each grid
cell the saved quantities are the floor curve at the dyadic band edges
:math:`K = 2^7, \dots, 2^{22}`, the fitted log--log slope over
:math:`K \in [2^{10}, 2^{22}]`, the predicted exponent and constant, and
the ratio :math:`\mathrm{floor}(K_{\max}) / (C_{\mathrm{pred}}
K_{\max}^{e})` (expected to approach 1 within a few percent).

Artefacts (all saved before any plotting; the figure is rebuilt from the
saved artefacts alone via ``--replot RUN_DIR``): ``floor_curves.npz`` (the
dyadic band edges and one floor curve per cell), ``summary.yaml``
(per-cell scalars and the run parameters), ``run_metadata.json``,
``command.txt``, ``run.log``, and the main figure
``forcing_floor_scaling.png`` (log--log measured curves, solid, colour by
grid cell on a viridis scale; dashed predicted power laws with their
predicted constants; the saturating cell flattening onto its dotted
plateau marker; formula textbox below the axes).

The computation is deterministic: the master seed is exposed and logged
for convention compliance, but no random number generator is consumed.
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
from _figure_layout import finalize_figure  # noqa: E402
from learning_option_pricing.pde import (  # noqa: E402
    PeriodisedBernoulliDatum,
    SquareWaveDatum,
    advection_diffusion_reaction,
    biharmonic_advection_reaction,
    black_scholes_log_price,
    operator_channel_floor,
    predicted_floor_exponent,
    predicted_operator_channel_floor_constant,
)
from learning_option_pricing.pde.periodic_spectral_toolbox import TWO_PI  # noqa: E402
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

LOGGER = logging.getLogger("measure_forcing_floor_scaling")

# Smoke-test guard: a real run sums the band up to 2^22; any band edge below
# 2^16 without --debug is mechanically rejected (see the repository
# convention on unflagged smoke runs).
SMOKE_TEST_LOG2_MAXIMUM_BAND_EDGE_THRESHOLD = 16

# Relative tolerance for the cross-check of the chunked accumulation against
# the library reference at the smallest band edge.  A violation raises
# ValueError (never a silent acceptance).
CROSS_CHECK_RELATIVE_TOLERANCE = 1.0e-12

MAIN_FIGURE_FILENAME = "forcing_floor_scaling.png"
FLOOR_CURVES_FILENAME = "floor_curves.npz"
SUMMARY_FILENAME = "summary.yaml"


# ---------------------------------------------------------------------------
# Measurement grid
# ---------------------------------------------------------------------------


def build_measurement_cells() -> list[dict]:
    """Return the ordered list of measurement cells of the grid.

    The grid is {G1, G3} x {rho = 0, 1, 2} plus (G2, rho = 1) for the
    single-break-point Bernoulli data, followed by the square-wave datum on
    G1 (the multi-singularity extension case, measured only).

    Returns:
        List of dictionaries with keys ``cell_key`` (identifier used in the
        artefact files), ``display_label`` (legend text), ``generator``
        (a ``ConstantCoefficientGenerator``) and ``datum``.
    """
    generator_g1 = advection_diffusion_reaction()
    generator_g2 = black_scholes_log_price()
    generator_g3 = biharmonic_advection_reaction()

    measurement_cells: list[dict] = []
    for generator_tag, generator in (("G1", generator_g1), ("G3", generator_g3)):
        for regularity_index in (0, 1, 2):
            measurement_cells.append(
                {
                    "cell_key": f"{generator_tag}_bernoulli_rho{regularity_index}",
                    "display_label": (
                        f"{generator_tag} {generator.name.replace('_', '–')}, "
                        f"Bernoulli $\\rho={regularity_index}$"
                    ),
                    "generator": generator,
                    "datum": PeriodisedBernoulliDatum(regularity_index),
                }
            )
    measurement_cells.append(
        {
            "cell_key": "G2_bernoulli_rho1",
            "display_label": (
                f"G2 {generator_g2.name.replace('_', '–')}, Bernoulli $\\rho=1$"
            ),
            "generator": generator_g2,
            "datum": PeriodisedBernoulliDatum(1),
        }
    )
    measurement_cells.append(
        {
            "cell_key": "G1_square_wave",
            "display_label": (
                f"G1 {generator_g1.name.replace('_', '–')}, square wave "
                "(two break points, measured only)"
            ),
            "generator": generator_g1,
            "datum": SquareWaveDatum(),
        }
    )
    return measurement_cells


# ---------------------------------------------------------------------------
# Chunked exact floor curve
# ---------------------------------------------------------------------------


def measured_floor_curve_at_dyadic_edges(
    generator,
    datum,
    dyadic_band_edges: np.ndarray,
    chunk_length: int,
) -> np.ndarray:
    r"""Floor curve :math:`\mathrm{floor}(K)` at every dyadic band edge.

    The sum :math:`2\pi \sum_{0 < |k| \le K} |a(k)|^2 |c_k|^2` is
    accumulated over consecutive segments of positive wavenumbers, each
    segment processed in chunks of at most ``chunk_length`` positive
    wavenumbers; every chunk evaluates the exact symbol and the exact datum
    coefficients on both the positive and the negative wavenumbers of the
    chunk (no symmetry shortcut), so the accumulated value at each dyadic
    edge is the exact band-truncated sum in floating point.

    Args:
        generator: Constant-coefficient generator exposing ``symbol``.
        datum: Terminal datum exposing ``fourier_coefficients``.
        dyadic_band_edges: Increasing array of band edges
            :math:`K = 2^{j}` at which the cumulative floor is recorded.
        chunk_length: Maximal number of positive wavenumbers evaluated at
            once (memory bound: each chunk allocates arrays of length
            ``2 * chunk_length``).

    Returns:
        ``float64`` array of the same length as ``dyadic_band_edges``.
    """
    floor_values = np.empty(len(dyadic_band_edges), dtype=np.float64)
    accumulated_sum = 0.0
    segment_start = 1
    for edge_index, band_edge in enumerate(dyadic_band_edges):
        chunk_start = segment_start
        while chunk_start <= band_edge:
            chunk_stop = min(chunk_start + chunk_length - 1, int(band_edge))
            positive_wavenumbers = np.arange(
                chunk_start, chunk_stop + 1, dtype=np.int64
            )
            chunk_band = np.concatenate(
                [-positive_wavenumbers, positive_wavenumbers]
            )
            symbol_values = generator.symbol(chunk_band)
            coefficient_values = datum.fourier_coefficients(chunk_band)
            accumulated_sum += float(
                np.sum(
                    np.abs(symbol_values) ** 2 * np.abs(coefficient_values) ** 2
                )
            )
            chunk_start = chunk_stop + 1
        floor_values[edge_index] = TWO_PI * accumulated_sum
        segment_start = int(band_edge) + 1
    return floor_values


def cross_check_against_library_reference(
    generator, datum, smallest_band_edge: int, chunked_floor_value: float
) -> float:
    """Validate the chunked accumulation against ``operator_channel_floor``.

    Args:
        generator: The generator of the cell.
        datum: The datum of the cell.
        smallest_band_edge: The first dyadic band edge.
        chunked_floor_value: The chunked cumulative value at that edge.

    Returns:
        The relative discrepancy between the two evaluations.

    Raises:
        ValueError: If the relative discrepancy exceeds
            ``CROSS_CHECK_RELATIVE_TOLERANCE`` (reported with both values).
    """
    library_floor_value = operator_channel_floor(
        generator, datum, smallest_band_edge
    )
    relative_discrepancy = abs(chunked_floor_value - library_floor_value) / abs(
        library_floor_value
    )
    if relative_discrepancy > CROSS_CHECK_RELATIVE_TOLERANCE:
        raise ValueError(
            "chunked floor accumulation disagrees with the library reference "
            f"at K = {smallest_band_edge}: chunked = {chunked_floor_value!r}, "
            f"library = {library_floor_value!r}, relative discrepancy = "
            f"{relative_discrepancy:.3e} > {CROSS_CHECK_RELATIVE_TOLERANCE:.0e}"
        )
    return relative_discrepancy


# ---------------------------------------------------------------------------
# Per-cell analysis
# ---------------------------------------------------------------------------


def analyse_measurement_cell(
    cell: dict,
    dyadic_band_edges: np.ndarray,
    floor_values: np.ndarray,
    fit_minimum_band_edge: int,
) -> dict:
    """Fit and compare one measured floor curve against the prediction.

    For a Bernoulli cell with positive predicted exponent the fitted
    log--log slope over the fit range and the ratio
    ``floor(K_max) / (C_pred * K_max**e)`` are recorded; for a saturating
    cell (negative exponent) the plateau value and the relative increment
    over the last dyadic step are recorded instead of a slope; for the
    square-wave cell the slope is recorded as a measured quantity only,
    with the prediction fields explicitly null.

    Args:
        cell: Measurement-cell dictionary from :func:`build_measurement_cells`.
        dyadic_band_edges: The dyadic band edges of the curve.
        floor_values: The measured floor values at those edges.
        fit_minimum_band_edge: Lower edge of the slope-fit range.

    Returns:
        Dictionary of per-cell scalars (plain Python types, ready for YAML).
    """
    generator = cell["generator"]
    datum = cell["datum"]
    largest_band_edge = int(dyadic_band_edges[-1])
    fit_mask = dyadic_band_edges >= fit_minimum_band_edge
    if int(np.sum(fit_mask)) < 2:
        raise ValueError(
            "the slope-fit range contains fewer than two dyadic band edges: "
            f"fit minimum {fit_minimum_band_edge}, largest edge "
            f"{largest_band_edge}"
        )

    cell_summary: dict = {
        "display_label": cell["display_label"],
        "generator_name": generator.name,
        "generator_coefficients": {
            int(order): float(value)
            for order, value in sorted(generator.coefficients.items())
        },
        "generator_half_order": int(generator.half_order),
        "generator_principal_constant": float(generator.principal_constant),
        "floor_at_largest_band_edge": float(floor_values[-1]),
        "fit_band_edges": [
            int(dyadic_band_edges[fit_mask][0]),
            int(dyadic_band_edges[fit_mask][-1]),
        ],
    }

    fitted_slope = float(
        np.polyfit(
            np.log(dyadic_band_edges[fit_mask].astype(np.float64)),
            np.log(floor_values[fit_mask]),
            1,
        )[0]
    )

    if isinstance(datum, SquareWaveDatum):
        # Multi-singularity extension case: two break points, so the
        # single-break-point law is not applied; the curve and its slope
        # are measured only.
        cell_summary.update(
            {
                "datum": "square_wave",
                "datum_regularity_index": int(datum.regularity_index),
                "datum_break_point_jumps": {
                    float(point): float(jump)
                    for point, jump in datum.break_point_jumps.items()
                },
                "predicted_exponent": None,
                "predicted_constant": None,
                "fitted_log_log_slope": fitted_slope,
                "ratio_measured_over_predicted_at_largest_band_edge": None,
                "note": (
                    "two break points: the single-break-point floor law is "
                    "not applied verbatim; the curve is measured only"
                ),
            }
        )
        return cell_summary

    growth_exponent = predicted_floor_exponent(generator, datum)
    cell_summary.update(
        {
            "datum": "periodised_bernoulli",
            "datum_regularity_index": int(datum.regularity_index),
            "datum_jump_of_rho_derivative": float(datum.jump_of_rho_derivative),
            "predicted_exponent": int(growth_exponent),
        }
    )
    if growth_exponent > 0:
        predicted_constant = predicted_operator_channel_floor_constant(
            generator, datum
        )
        ratio_at_largest_edge = float(
            floor_values[-1]
            / (predicted_constant * float(largest_band_edge) ** growth_exponent)
        )
        cell_summary.update(
            {
                "predicted_constant": float(predicted_constant),
                "fitted_log_log_slope": fitted_slope,
                "ratio_measured_over_predicted_at_largest_band_edge": (
                    ratio_at_largest_edge
                ),
            }
        )
    else:
        # Saturation regime (e < 0): the floor tends to its finite total
        # sum; the plateau value is reported instead of a slope.
        relative_increment_over_last_dyadic_step = float(
            (floor_values[-1] - floor_values[-2]) / floor_values[-1]
        )
        cell_summary.update(
            {
                "predicted_constant": None,
                "fitted_log_log_slope": None,
                "ratio_measured_over_predicted_at_largest_band_edge": None,
                "plateau_value": float(floor_values[-1]),
                "relative_increment_over_last_dyadic_step": (
                    relative_increment_over_last_dyadic_step
                ),
                "note": (
                    "saturation regime (predicted exponent "
                    f"{growth_exponent} < 0): the plateau value is the floor "
                    "at the largest band edge, reported instead of a slope"
                ),
            }
        )
    return cell_summary


# ---------------------------------------------------------------------------
# Figure (built from the saved artefacts only)
# ---------------------------------------------------------------------------


def render_main_figure(run_directory: Path) -> Path:
    """Rebuild the main figure from ``floor_curves.npz`` and ``summary.yaml``.

    The function reads only the saved artefacts (no recomputation), so the
    ``--replot`` path and the in-run plotting path are the same code.

    Args:
        run_directory: Directory containing the saved artefacts.

    Returns:
        Path of the written figure.
    """
    saved_curves = np.load(run_directory / FLOOR_CURVES_FILENAME)
    with open(run_directory / SUMMARY_FILENAME, "r", encoding="utf-8") as handle:
        summary = yaml.safe_load(handle)

    dyadic_band_edges = saved_curves["dyadic_band_edges"].astype(np.float64)
    cell_order = summary["cell_order"]
    fit_minimum_band_edge = summary["parameters"]["fit_minimum_band_edge"]
    cell_colours = plt.cm.viridis(np.linspace(0.0, 0.92, len(cell_order)))

    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    prediction_line_handle = None
    plateau_line_handle = None
    for cell_index, cell_key in enumerate(cell_order):
        cell_summary = summary["cells"][cell_key]
        floor_values = saved_curves[f"floor_curve__{cell_key}"]
        colour = cell_colours[cell_index]

        fitted_slope = cell_summary["fitted_log_log_slope"]
        predicted_exponent = cell_summary["predicted_exponent"]
        if predicted_exponent is None:
            legend_label = (
                f"{cell_summary['display_label']}; measured slope "
                f"{fitted_slope:.3f}"
            )
        elif fitted_slope is None:
            legend_label = (
                f"{cell_summary['display_label']}; $e={predicted_exponent}$, "
                f"plateau {cell_summary['plateau_value']:.3e}"
            )
        else:
            legend_label = (
                f"{cell_summary['display_label']}; $e={predicted_exponent}$, "
                f"measured slope {fitted_slope:.3f}, ratio "
                f"{cell_summary['ratio_measured_over_predicted_at_largest_band_edge']:.3f}"
            )
        # Solid stroke: measured curve.
        ax.loglog(
            dyadic_band_edges,
            floor_values,
            "-o",
            color=colour,
            linewidth=1.6,
            markersize=3.5,
            label=legend_label,
        )
        # Dashed stroke: the analytical prediction C_pred * K^e with its
        # predicted constant (not anchored to the measurement), drawn over
        # the slope-fit range.
        if cell_summary.get("predicted_constant") is not None:
            fit_mask = dyadic_band_edges >= fit_minimum_band_edge
            predicted_values = cell_summary["predicted_constant"] * (
                dyadic_band_edges[fit_mask] ** predicted_exponent
            )
            (prediction_line_handle,) = ax.loglog(
                dyadic_band_edges[fit_mask],
                predicted_values,
                "--",
                color=colour,
                linewidth=1.2,
            )
        # Dotted stroke: auxiliary plateau marker for the saturating cell.
        if cell_summary.get("plateau_value") is not None:
            plateau_line_handle = ax.axhline(
                cell_summary["plateau_value"],
                linestyle=":",
                color=colour,
                linewidth=1.2,
            )

    ax.set_xlabel("Band edge $K$")
    # The defining formula of the floor is stated in the textbox below the
    # figure; the axis label stays short so it fits the canvas.
    ax.set_ylabel(r"Operator-channel floor $\mathrm{floor}(K)$")
    ax.set_title(
        "Operator-channel floor against the band edge: measured curves "
        "(solid) and predicted power laws (dashed)",
        fontsize=10,
    )
    ax.grid(True, which="both", alpha=0.3)

    legend_handles, legend_labels = ax.get_legend_handles_labels()
    if prediction_line_handle is not None:
        legend_handles.append(prediction_line_handle)
        legend_labels.append(
            r"Predicted $C_{\mathrm{pred}}K^{e}$ (dashed, per cell colour)"
        )
    if plateau_line_handle is not None:
        legend_handles.append(plateau_line_handle)
        legend_labels.append("Saturation plateau (dotted)")
    # Figure-level legend anchored just above the bottom formula box (the
    # legend is outside the axes data area, below, in two columns).
    legend = fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.115),
        ncol=2,
        fontsize=6.5,
        frameon=True,
    )
    fig.tight_layout(rect=[0.0, 0.30, 1.0, 0.98])

    figure_path = run_directory / MAIN_FIGURE_FILENAME
    finalize_figure(
        fig,
        figure_path,
        legends=[legend],
        axes=[ax],
        formula=(
            r"$\mathrm{floor}(K)=2\pi\sum_{0<|k|\leq K}|a(k)|^2|c_k|^2$;"
            r"  predicted law: $\mathrm{floor}(K)=\frac{a_0^2 J^2}{\pi e}"
            r"K^{e}\,(1+o(1))$ as $K\to\infty$, with $e=4p-2\rho-1$"
            "\n"
            r"$a_0$ = principal constant of the generator, $J$ = jump of the "
            r"$\rho$-th derivative at the single break point;"
            r" $e<0$: saturation to the finite total sum (dotted plateau);"
            "\n"
            r"square wave: two break points, measured only "
            r"(single-break-point law not applied)"
        ),
        formula_fontsize=7.5,
    )
    return figure_path


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def parse_arguments(argv=None) -> argparse.Namespace:
    """Parse and validate the command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--log2-minimum-band-edge",
        type=int,
        default=7,
        help="smallest dyadic band edge is 2 to this power (default 7)",
    )
    parser.add_argument(
        "--log2-maximum-band-edge",
        type=int,
        default=22,
        help="largest dyadic band edge is 2 to this power (default 22)",
    )
    parser.add_argument(
        "--log2-fit-minimum-band-edge",
        type=int,
        default=10,
        help=(
            "the log-log slope is fitted over band edges from 2 to this "
            "power up to the largest edge (default 10)"
        ),
    )
    parser.add_argument(
        "--chunk-length",
        type=int,
        default=2**19,
        help=(
            "maximal number of positive wavenumbers evaluated per chunk "
            "(memory bound; default 2**19)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "master seed, logged for convention compliance; the measurement "
            "is deterministic and consumes no random number generator"
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="prepend '_debug_' to the output folder name (smoke runs)",
    )
    parser.add_argument(
        "--replot",
        type=Path,
        default=None,
        metavar="RUN_DIR",
        help=(
            "rebuild the figure from the saved artefacts of an existing run "
            "directory, without any recomputation"
        ),
    )
    arguments = parser.parse_args(argv)

    if arguments.replot is not None:
        return arguments
    if (
        arguments.log2_maximum_band_edge
        < SMOKE_TEST_LOG2_MAXIMUM_BAND_EDGE_THRESHOLD
        and not arguments.debug
    ):
        parser.error(
            "smoke-test guard: --log2-maximum-band-edge "
            f"{arguments.log2_maximum_band_edge} < "
            f"{SMOKE_TEST_LOG2_MAXIMUM_BAND_EDGE_THRESHOLD} requires --debug "
            "(unflagged smoke runs are rejected mechanically)"
        )
    if arguments.log2_minimum_band_edge < 1:
        parser.error("--log2-minimum-band-edge must be at least 1")
    if arguments.log2_maximum_band_edge <= arguments.log2_minimum_band_edge:
        parser.error(
            "--log2-maximum-band-edge must exceed --log2-minimum-band-edge"
        )
    if not (
        arguments.log2_minimum_band_edge
        <= arguments.log2_fit_minimum_band_edge
        <= arguments.log2_maximum_band_edge - 1
    ):
        parser.error(
            "--log2-fit-minimum-band-edge must lie between "
            "--log2-minimum-band-edge and --log2-maximum-band-edge - 1 "
            "(at least two dyadic points in the fit range)"
        )
    if arguments.chunk_length < 1:
        parser.error("--chunk-length must be a positive integer")
    return arguments


def main(argv=None) -> int:
    """Run the measurement (or the replot path) and write the artefacts."""
    arguments = parse_arguments(argv)

    if arguments.replot is not None:
        configure_cli_script_logging(verbose=False)
        run_directory = arguments.replot.resolve()
        if not (run_directory / FLOOR_CURVES_FILENAME).exists():
            raise FileNotFoundError(
                f"no {FLOOR_CURVES_FILENAME} under {run_directory}; "
                "--replot expects a completed run directory"
            )
        figure_path = render_main_figure(run_directory)
        LOGGER.info("Replotted figure from saved artefacts: %s", figure_path)
        return 0

    start_time = time.perf_counter()
    debug_prefix = "_debug_" if arguments.debug else ""
    config_tag = f"log2K{arguments.log2_maximum_band_edge}"
    run_directory = (
        script_data_dir(__file__) / f"{debug_prefix}{utc_timestamp()}_{config_tag}"
    )
    run_directory.mkdir(parents=True, exist_ok=False)

    init_logging(run_dir=run_directory)
    LOGGER.info("Full command line: %s", " ".join(sys.argv))
    LOGGER.info(
        "Runtime: Python %s | numpy %s | matplotlib %s | torch not used "
        "(pure-numpy measurement)",
        sys.version.split()[0],
        np.__version__,
        matplotlib.__version__,
    )
    log_parsed_args(LOGGER, arguments)
    LOGGER.info(
        "Master seed %d (deterministic measurement: no random number "
        "generator is consumed)",
        arguments.seed,
    )
    repository_root = find_repo_root(Path(__file__))
    run_metadata = collect_run_metadata(
        run_dir=run_directory,
        repo_root=repository_root,
        script_name=Path(__file__).stem,
        command=list(sys.argv),
        params=dict(sorted(vars(arguments).items(), key=lambda item: item[0])),
    )
    write_json(run_directory / "run_metadata.json", run_metadata)
    write_command_txt(run_directory / "command.txt", list(sys.argv))
    LOGGER.info(
        "Git commit %s (dirty: %s)",
        run_metadata["git"].get("commit"),
        run_metadata["git"].get("dirty"),
    )

    dyadic_band_edges = 2 ** np.arange(
        arguments.log2_minimum_band_edge,
        arguments.log2_maximum_band_edge + 1,
        dtype=np.int64,
    )
    fit_minimum_band_edge = int(2 ** arguments.log2_fit_minimum_band_edge)
    LOGGER.info(
        "Dyadic band edges: 2^%d .. 2^%d (%d edges); slope-fit range "
        "[%d, %d]; chunk length %d",
        arguments.log2_minimum_band_edge,
        arguments.log2_maximum_band_edge,
        len(dyadic_band_edges),
        fit_minimum_band_edge,
        int(dyadic_band_edges[-1]),
        arguments.chunk_length,
    )

    measurement_cells = build_measurement_cells()
    summary: dict = {
        "parameters": {
            "log2_minimum_band_edge": int(arguments.log2_minimum_band_edge),
            "log2_maximum_band_edge": int(arguments.log2_maximum_band_edge),
            "log2_fit_minimum_band_edge": int(
                arguments.log2_fit_minimum_band_edge
            ),
            "fit_minimum_band_edge": fit_minimum_band_edge,
            "chunk_length": int(arguments.chunk_length),
            "seed": int(arguments.seed),
            "debug": bool(arguments.debug),
            "cross_check_relative_tolerance": CROSS_CHECK_RELATIVE_TOLERANCE,
        },
        "cell_order": [cell["cell_key"] for cell in measurement_cells],
        "cells": {},
    }
    floor_curves_payload: dict = {"dyadic_band_edges": dyadic_band_edges}

    for cell in measurement_cells:
        cell_start_time = time.perf_counter()
        floor_values = measured_floor_curve_at_dyadic_edges(
            cell["generator"],
            cell["datum"],
            dyadic_band_edges,
            arguments.chunk_length,
        )
        cross_check_discrepancy = cross_check_against_library_reference(
            cell["generator"],
            cell["datum"],
            int(dyadic_band_edges[0]),
            float(floor_values[0]),
        )
        cell_summary = analyse_measurement_cell(
            cell, dyadic_band_edges, floor_values, fit_minimum_band_edge
        )
        cell_summary["cross_check_relative_discrepancy_at_smallest_edge"] = float(
            cross_check_discrepancy
        )
        summary["cells"][cell["cell_key"]] = cell_summary
        floor_curves_payload[f"floor_curve__{cell['cell_key']}"] = floor_values

        if cell_summary["fitted_log_log_slope"] is None:
            LOGGER.info(
                "%-24s e = %s; plateau = %.6e (relative increment over the "
                "last dyadic step %.3e); cross-check %.2e  [%.2f s]",
                cell["cell_key"],
                cell_summary["predicted_exponent"],
                cell_summary["plateau_value"],
                cell_summary["relative_increment_over_last_dyadic_step"],
                cross_check_discrepancy,
                time.perf_counter() - cell_start_time,
            )
        else:
            LOGGER.info(
                "%-24s e = %s; fitted slope = %.4f; ratio at K_max = %s; "
                "cross-check %.2e  [%.2f s]",
                cell["cell_key"],
                cell_summary["predicted_exponent"],
                cell_summary["fitted_log_log_slope"],
                (
                    f"{cell_summary['ratio_measured_over_predicted_at_largest_band_edge']:.4f}"
                    if cell_summary[
                        "ratio_measured_over_predicted_at_largest_band_edge"
                    ]
                    is not None
                    else "n/a (measured only)"
                ),
                cross_check_discrepancy,
                time.perf_counter() - cell_start_time,
            )

    # Save every measured curve and the scalar summary BEFORE any plotting,
    # so the figure is reproducible from the artefacts alone.
    np.savez(run_directory / FLOOR_CURVES_FILENAME, **floor_curves_payload)
    with open(run_directory / SUMMARY_FILENAME, "w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)
    LOGGER.info(
        "Saved artefacts: %s, %s", FLOOR_CURVES_FILENAME, SUMMARY_FILENAME
    )

    figure_path = render_main_figure(run_directory)
    LOGGER.info("Wrote main figure: %s", figure_path)
    LOGGER.info(
        "Total wall-clock time: %.2f s; run directory: %s",
        time.perf_counter() - start_time,
        run_directory,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
