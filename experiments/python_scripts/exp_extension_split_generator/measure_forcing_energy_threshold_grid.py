r"""Threshold of finite forcing energy in the half order p and the remainder order m_B.

Pre-registration:
``documents/methodology/2026-10-11_preregistration_forcing_energy_threshold_grid.md``.

Propositions under test.  Propositions 4.2(iii) and 4.3 of the paper
(``RHB_notes/paper/2026-09-29_boundary_paper_draft.tex`` of the report
repository): for a parabolic retained operator :math:`A` with constant
coefficients and a remainder :math:`B = \mathcal L^X - A` of order
:math:`m_B`, the forcing energy :math:`E_h` of the retained evolution is finite
if and only if :math:`g \in H^{s_B}`, with :math:`s_B = \max(m_B - p, 0)`.  For
a datum that belongs to :math:`H^s` exactly for :math:`s < \sigma`, finiteness
therefore holds if and only if :math:`m_B - p < \sigma`.

Grid.  On the circle, with :math:`T = 1`:

* generators :math:`\mathcal L^X_p`, :math:`p \in \{1, 2, 3\}`, with a coefficient
  of modulus one at every order :math:`0 \le j \le 2p`, dissipative at every
  even order (:func:`complete_unit_coefficient_generator`);
* data :math:`g_\alpha(x) = \sum_{k \ge 1} \cos(kx) / (\pi^2 k^\alpha)`,
  :math:`\alpha \in \{5/4, 2, 3\}`, continuous, with Sobolev critical index
  :math:`\sigma = \alpha - 1/2 \in \{3/4, 3/2, 5/2\}` (:math:`\alpha = 2` is the
  datum of the paper);
* for every :math:`m \in \{0, \dots, 2p - 1\}`, the truncation
  :math:`A = \sum_{j > m} \gamma_j \partial^j / \partial x^j`, whose remainder
  has order :math:`m_B = m`, and the constant-in-time extension :math:`h = g`
  (empty subset, :math:`m_B = 2p`, no smoothing).

Measured quantity.  The band-truncated forcing energy, in closed form (the
time integral of every wavenumber is exact, no quadrature),

.. math::

    E(K) = 2\pi \sum_{0 < |k| \le K} |b(k)|^2 |c_k|^2\,
    \varphi(2 \operatorname{Re} a_A(k)),

the forcing energy of the extension of the band-limited datum.  By Parseval's
identity and monotone convergence, :math:`E_h(g_\alpha) = \lim_K E(K)`, so
:math:`E_h` is finite exactly when :math:`E(K)` is bounded.

Predictions (pre-registered, with the exponent
:math:`e = 2(m_B - s_A - \sigma)`, :math:`s_A = p` for a truncation and
:math:`s_A = 0` for :math:`h = g`; see
:func:`learning_option_pricing.pde.predicted_split_forcing_energy_exponent`):

* R1: :math:`E(K)` is bounded if and only if :math:`e < 0`; operationally, the
  relative increment over the last dyadic step is below
  ``BOUNDED_RELATIVE_INCREMENT_TOLERANCE`` for every cell with :math:`e < 0`,
  and above ``GROWING_RELATIVE_INCREMENT_THRESHOLD`` for every cell with
  :math:`e > 0`;
* R2: for every cell with :math:`e > 0`, the fitted log--log slope over the fit
  range lies within ``SLOPE_TOLERANCE`` of :math:`e`, and
  :math:`E(K_{\max}) / (C K_{\max}^e)` lies within ``RATIO_TOLERANCE`` of 1.

The verdicts are recorded per cell and counted; they never abort the run.

Checks performed before any value is saved (a failure raises): every energy
value is finite, and the chunked accumulation at the smallest band edge equals
:func:`learning_option_pricing.pde.total_strip_forcing_squared` to a relative
tolerance of ``CROSS_CHECK_RELATIVE_TOLERANCE``.

Artefacts (saved before plotting; ``--replot RUN_DIR`` rebuilds the figures
from them alone): ``energy_curves.npz``, ``summary.yaml``,
``run_metadata.json``, ``command.txt``, ``run.log``,
``forcing_energy_against_band_edge.png``, ``threshold_table.png``.

The computation is deterministic: the master seed is exposed and logged for
convention compliance, but no random number generator is consumed.
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _figure_layout import finalize_figure  # noqa: E402
from learning_option_pricing.pde import (  # noqa: E402
    ConstantCoefficientGenerator,
    PowerLawCosineDatum,
    SplitSemigroupExtension,
    predicted_split_forcing_energy_constant,
    predicted_split_forcing_energy_exponent,
    symmetric_wavenumber_band,
    total_strip_forcing_squared,
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

LOGGER = logging.getLogger(Path(__file__).stem)

# Smoke-test guard: the pre-registered run sums the band up to 2^22; a largest
# band edge below 2^16 without --debug is rejected mechanically.
SMOKE_TEST_LOG2_MAXIMUM_BAND_EDGE_THRESHOLD = 16

# Pre-registered tolerances (section 3 of the pre-registration).
BOUNDED_RELATIVE_INCREMENT_TOLERANCE = 1.0e-3
GROWING_RELATIVE_INCREMENT_THRESHOLD = 1.0e-1
SLOPE_TOLERANCE = 5.0e-2
RATIO_TOLERANCE = 2.0e-2

# Relative tolerance of the cross-check of the chunked accumulation against
# the library function at the smallest band edge (a violation raises).
CROSS_CHECK_RELATIVE_TOLERANCE = 1.0e-12

HALF_ORDERS = (1, 2, 3)
# Decay exponents alpha of the data g_alpha; the Sobolev critical index is
# alpha - 1/2.  The value 2 gives the datum of the paper.
DECAY_EXPONENTS = (1.25, 2.0, 3.0)

ENERGY_CURVES_FILENAME = "energy_curves.npz"
SUMMARY_FILENAME = "summary.yaml"
ENERGY_FIGURE_FILENAME = "forcing_energy_against_band_edge.png"
THRESHOLD_TABLE_FIGURE_FILENAME = "threshold_table.png"


# ---------------------------------------------------------------------------
# Measurement grid
# ---------------------------------------------------------------------------


def complete_unit_coefficient_generator(half_order: int) -> ConstantCoefficientGenerator:
    r"""Generator of order :math:`2p` with a coefficient of modulus one at every order.

    The coefficient of an odd order is :math:`+1`.  The coefficient of an even
    order :math:`j` is :math:`(-1)^{j/2 + 1}`, so that
    :math:`\gamma_j (ik)^j = -k^j`: every even-order term is dissipative, and
    every subset of orders that contains :math:`2p` defines a parabolic
    retained operator.  Coefficients of equal modulus make the relative
    corrections to the leading behaviour of :math:`|b(k)|^2` of order
    :math:`k^{-2}` with a constant of order one.

    Args:
        half_order: The half order :math:`p \ge 1`.

    Returns:
        The generator, named ``complete_unit_coefficient_order_<2p>``.
    """
    coefficients = {}
    for order in range(2 * half_order + 1):
        if order % 2 == 1:
            coefficients[order] = 1.0
        else:
            coefficients[order] = float((-1) ** (order // 2 + 1))
    return ConstantCoefficientGenerator(
        coefficients, name=f"complete_unit_coefficient_order_{2 * half_order}"
    )


def decay_exponent_tag(decay_exponent: float) -> str:
    """File-safe tag of a decay exponent, e.g. ``1p25`` for 1.25 and ``2`` for 2.0."""
    if float(decay_exponent).is_integer():
        return str(int(decay_exponent))
    return f"{decay_exponent:g}".replace(".", "p")


def build_measurement_cells() -> list[dict]:
    """Return the ordered list of the 45 measurement cells.

    Order: decay exponent, then half order, then remainder order from 0 to
    :math:`2p` (the last one being the constant-in-time extension).

    Returns:
        List of dictionaries with keys ``cell_key``, ``half_order``,
        ``decay_exponent``, ``remainder_order``, ``subset_orders``,
        ``extension_kind`` (``"truncation"`` or ``"constant_in_time"``),
        ``generator`` and ``datum``.
    """
    measurement_cells: list[dict] = []
    for decay_exponent in DECAY_EXPONENTS:
        datum = PowerLawCosineDatum(decay_exponent)
        for half_order in HALF_ORDERS:
            generator = complete_unit_coefficient_generator(half_order)
            for remainder_order in range(2 * half_order + 1):
                subset_orders = tuple(
                    order for order in range(remainder_order + 1, 2 * half_order + 1)
                )
                measurement_cells.append(
                    {
                        "cell_key": (
                            f"alpha{decay_exponent_tag(decay_exponent)}"
                            f"_p{half_order}_mB{remainder_order}"
                        ),
                        "half_order": half_order,
                        "decay_exponent": decay_exponent,
                        "remainder_order": remainder_order,
                        "subset_orders": subset_orders,
                        "extension_kind": (
                            "constant_in_time" if not subset_orders else "truncation"
                        ),
                        "generator": generator,
                        "datum": datum,
                    }
                )
    return measurement_cells


# ---------------------------------------------------------------------------
# Chunked exact energy curve
# ---------------------------------------------------------------------------


def energy_curve_at_dyadic_edges(
    extension: SplitSemigroupExtension,
    dyadic_band_edges: np.ndarray,
    chunk_length: int,
) -> np.ndarray:
    r"""Band-truncated forcing energy :math:`E(K)` at every dyadic band edge.

    The sum :math:`2\pi \sum_{0 < |k| \le K}` of the closed-form per-wavenumber
    time integrals is accumulated in increasing :math:`|k|`, in chunks of at
    most ``chunk_length`` positive wavenumbers; every chunk evaluates the
    positive and the negative wavenumbers (no symmetry shortcut).

    Args:
        extension: The split extension of the cell.
        dyadic_band_edges: Increasing array of band edges.
        chunk_length: Maximal number of positive wavenumbers per chunk.

    Returns:
        ``float64`` array of the same length as ``dyadic_band_edges``.

    Raises:
        ValueError: If a chunk produces a non-finite value.
    """
    energy_values = np.empty(len(dyadic_band_edges), dtype=np.float64)
    accumulated_sum = 0.0
    segment_start = 1
    for edge_index, band_edge in enumerate(dyadic_band_edges):
        chunk_start = segment_start
        while chunk_start <= band_edge:
            chunk_stop = min(chunk_start + chunk_length - 1, int(band_edge))
            positive_wavenumbers = np.arange(chunk_start, chunk_stop + 1, dtype=np.int64)
            chunk_band = np.concatenate([-positive_wavenumbers, positive_wavenumbers])
            chunk_integrals = extension.squared_forcing_time_integral(chunk_band)
            if not np.all(np.isfinite(chunk_integrals)):
                raise ValueError(
                    "non-finite per-wavenumber time integral in the chunk "
                    f"[{chunk_start}, {chunk_stop}]"
                )
            accumulated_sum += float(np.sum(chunk_integrals))
            chunk_start = chunk_stop + 1
        energy_values[edge_index] = TWO_PI * accumulated_sum
        segment_start = int(band_edge) + 1
    if not np.all(np.isfinite(energy_values)):
        raise ValueError("non-finite accumulated energy")
    return energy_values


def cross_check_against_library_reference(
    extension: SplitSemigroupExtension, smallest_band_edge: int, chunked_value: float
) -> float:
    """Compare the chunked value with ``total_strip_forcing_squared``.

    Returns:
        The relative discrepancy.

    Raises:
        ValueError: If the discrepancy exceeds ``CROSS_CHECK_RELATIVE_TOLERANCE``.
    """
    library_value = total_strip_forcing_squared(
        extension, symmetric_wavenumber_band(smallest_band_edge)
    )
    relative_discrepancy = abs(chunked_value - library_value) / abs(library_value)
    if relative_discrepancy > CROSS_CHECK_RELATIVE_TOLERANCE:
        raise ValueError(
            "chunked energy accumulation differs from the library value at "
            f"K = {smallest_band_edge}: chunked = {chunked_value!r}, library = "
            f"{library_value!r}, relative discrepancy {relative_discrepancy:.3e}"
        )
    return relative_discrepancy


# ---------------------------------------------------------------------------
# Per-cell analysis
# ---------------------------------------------------------------------------


def analyse_measurement_cell(
    cell: dict,
    dyadic_band_edges: np.ndarray,
    energy_values: np.ndarray,
    fit_minimum_band_edge: int,
    terminal_time: float,
) -> dict:
    """Compare one measured energy curve with the predictions R1 and R2.

    Returns:
        Dictionary of per-cell scalars (plain Python types, ready for YAML).
    """
    generator = cell["generator"]
    datum = cell["datum"]
    subset_orders = cell["subset_orders"]
    predicted_exponent = predicted_split_forcing_energy_exponent(
        generator, subset_orders, datum
    )
    relative_increment_over_last_dyadic_step = float(
        (energy_values[-1] - energy_values[-2]) / energy_values[-1]
    )
    fit_mask = dyadic_band_edges >= fit_minimum_band_edge
    fitted_slope = float(
        np.polyfit(
            np.log(dyadic_band_edges[fit_mask].astype(np.float64)),
            np.log(energy_values[fit_mask]),
            1,
        )[0]
    )
    predicted_bounded = bool(predicted_exponent < 0.0)
    if predicted_bounded:
        verdict_boundedness = bool(
            relative_increment_over_last_dyadic_step < BOUNDED_RELATIVE_INCREMENT_TOLERANCE
        )
    else:
        verdict_boundedness = bool(
            relative_increment_over_last_dyadic_step > GROWING_RELATIVE_INCREMENT_THRESHOLD
        )

    cell_summary: dict = {
        "half_order": int(cell["half_order"]),
        "decay_exponent": float(cell["decay_exponent"]),
        "sobolev_critical_index": float(datum.sobolev_critical_index),
        "remainder_order": int(cell["remainder_order"]),
        "sobolev_index_of_remainder": (
            int(max(cell["remainder_order"] - cell["half_order"], 0))
            if cell["extension_kind"] == "truncation"
            else None
        ),
        "subset_orders": [int(order) for order in subset_orders],
        "extension_kind": cell["extension_kind"],
        "generator_name": generator.name,
        "generator_coefficients": {
            int(order): float(value) for order, value in sorted(generator.coefficients.items())
        },
        "predicted_exponent": float(predicted_exponent),
        "predicted_bounded": predicted_bounded,
        "energy_at_largest_band_edge": float(energy_values[-1]),
        "relative_increment_over_last_dyadic_step": relative_increment_over_last_dyadic_step,
        "fitted_log_log_slope": fitted_slope,
        "fit_band_edges": [
            int(dyadic_band_edges[fit_mask][0]),
            int(dyadic_band_edges[fit_mask][-1]),
        ],
        "verdict_r1_boundedness": verdict_boundedness,
    }
    if predicted_bounded:
        cell_summary.update(
            {
                "predicted_constant": None,
                "ratio_measured_over_predicted_at_largest_band_edge": None,
                "verdict_r2_rate": None,
            }
        )
    else:
        predicted_constant = predicted_split_forcing_energy_constant(
            generator, subset_orders, datum, terminal_time
        )
        ratio = float(
            energy_values[-1]
            / (predicted_constant * float(dyadic_band_edges[-1]) ** predicted_exponent)
        )
        cell_summary.update(
            {
                "predicted_constant": float(predicted_constant),
                "ratio_measured_over_predicted_at_largest_band_edge": ratio,
                "verdict_r2_rate": bool(
                    abs(fitted_slope - predicted_exponent) <= SLOPE_TOLERANCE
                    and abs(ratio - 1.0) <= RATIO_TOLERANCE
                ),
            }
        )
    return cell_summary


# ---------------------------------------------------------------------------
# Figures (built from the saved artefacts only)
# ---------------------------------------------------------------------------


def _load_artefacts(run_directory: Path) -> tuple[dict, dict]:
    saved_curves = dict(np.load(run_directory / ENERGY_CURVES_FILENAME))
    with open(run_directory / SUMMARY_FILENAME, "r", encoding="utf-8") as handle:
        summary = yaml.safe_load(handle)
    return saved_curves, summary


def render_energy_figure(run_directory: Path) -> Path:
    """Rebuild the 3 x 3 panel figure of E(K) from the saved artefacts."""
    saved_curves, summary = _load_artefacts(run_directory)
    dyadic_band_edges = saved_curves["dyadic_band_edges"].astype(np.float64)
    fit_minimum_band_edge = summary["parameters"]["fit_minimum_band_edge"]
    half_orders = summary["parameters"]["half_orders"]
    decay_exponents = summary["parameters"]["decay_exponents"]
    maximal_remainder_order = 2 * max(half_orders)
    # Sequential palette: the axis of the sweep is the remainder order m_B.
    remainder_colours = plt.cm.viridis(np.linspace(0.0, 0.92, maximal_remainder_order + 1))

    fig, axes = plt.subplots(
        len(half_orders), len(decay_exponents), figsize=(13.0, 11.0), squeeze=False
    )
    for row_index, half_order in enumerate(half_orders):
        for column_index, decay_exponent in enumerate(decay_exponents):
            ax = axes[row_index, column_index]
            for cell_key in summary["cell_order"]:
                cell_summary = summary["cells"][cell_key]
                if (
                    cell_summary["half_order"] != half_order
                    or cell_summary["decay_exponent"] != decay_exponent
                ):
                    continue
                remainder_order = cell_summary["remainder_order"]
                colour = remainder_colours[remainder_order]
                marker = "s" if cell_summary["extension_kind"] == "constant_in_time" else "o"
                # Solid stroke: closed-form measured curve.
                ax.loglog(
                    dyadic_band_edges,
                    saved_curves[f"energy_curve__{cell_key}"],
                    "-",
                    marker=marker,
                    color=colour,
                    linewidth=1.4,
                    markersize=2.8,
                )
                # Dashed stroke: predicted power law C K^e over the fit range.
                if cell_summary["predicted_constant"] is not None:
                    fit_mask = dyadic_band_edges >= fit_minimum_band_edge
                    ax.loglog(
                        dyadic_band_edges[fit_mask],
                        cell_summary["predicted_constant"]
                        * dyadic_band_edges[fit_mask] ** cell_summary["predicted_exponent"],
                        "--",
                        color="black",
                        linewidth=0.9,
                    )
            sobolev_critical_index = decay_exponent - 0.5
            ax.set_title(
                rf"$p={half_order}$, $\alpha={decay_exponent:g}$ "
                rf"($g_\alpha\in H^s$ iff $s<{sobolev_critical_index:g}$)",
                fontsize=9,
            )
            ax.grid(True, which="both", alpha=0.3)
            if row_index == len(half_orders) - 1:
                ax.set_xlabel(r"Band edge $K$")
            if column_index == 0:
                ax.set_ylabel(r"Forcing energy $E(K)$")

    legend_handles = [
        Line2D([0], [0], color=remainder_colours[order], linewidth=1.6, label=rf"$m_B={order}$")
        for order in range(maximal_remainder_order + 1)
    ]
    legend_handles.append(
        Line2D([0], [0], color="grey", marker="o", linewidth=1.4, label="Truncation of the generator (circles)")
    )
    legend_handles.append(
        Line2D([0], [0], color="grey", marker="s", linewidth=1.4, label=r"Constant in time $h=g$, $m_B=2p$ (squares)")
    )
    legend_handles.append(
        Line2D([0], [0], color="black", linestyle="--", linewidth=0.9, label=r"Predicted $C\,K^{e}$ (dashed)")
    )
    legend = fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.085),
        ncol=5,
        fontsize=8,
        frameon=True,
    )
    fig.suptitle(
        "Band-truncated forcing energy of the truncations of the generator, closed form",
        fontsize=11,
    )
    fig.tight_layout(rect=[0.0, 0.17, 1.0, 0.97])
    figure_path = run_directory / ENERGY_FIGURE_FILENAME
    finalize_figure(
        fig,
        figure_path,
        legends=[legend],
        axes=list(axes.ravel()),
        formula=(
            r"$E(K)=2\pi\sum_{0<|k|\leq K}|b(k)|^2|c_k|^2\,\varphi(2\,\mathrm{Re}\,a_A(k))$, "
            r"$\varphi(z)=(e^{zT}-1)/z$, $c_k=1/(2\pi^2|k|^\alpha)$, "
            r"$A=\sum_{j>m_B}\gamma_{\mathcal{L}^X,j}\,\partial^j/\partial x^j$, $B=\mathcal{L}^X-A$"
            "\n"
            r"predicted: $E(K)=C\,K^{e}(1+o(1))$ as $K\to\infty$ with "
            r"$e=2(m_B-s_A-\alpha+1/2)$, $s_A=p$ (truncation) or $0$ ($h=g$); "
            r"$E(K)$ bounded iff $e<0$"
        ),
        formula_fontsize=8,
    )
    return figure_path


def render_threshold_table_figure(run_directory: Path) -> Path:
    """Rebuild the table of measured boundedness against p and m_B, one panel per datum."""
    _, summary = _load_artefacts(run_directory)
    half_orders = summary["parameters"]["half_orders"]
    decay_exponents = summary["parameters"]["decay_exponents"]
    maximal_remainder_order = 2 * max(half_orders)
    bounded_colour = "#a6cee3"
    growing_colour = "#fb9a99"

    fig, axes = plt.subplots(1, len(decay_exponents), figsize=(15.0, 4.8), squeeze=False)
    for column_index, decay_exponent in enumerate(decay_exponents):
        ax = axes[0, column_index]
        for cell_key in summary["cell_order"]:
            cell_summary = summary["cells"][cell_key]
            if cell_summary["decay_exponent"] != decay_exponent:
                continue
            row = cell_summary["half_order"]
            column = cell_summary["remainder_order"]
            measured_bounded = (
                cell_summary["relative_increment_over_last_dyadic_step"]
                < BOUNDED_RELATIVE_INCREMENT_TOLERANCE
            )
            ax.add_patch(
                plt.Rectangle(
                    (column - 0.5, row - 0.5),
                    1.0,
                    1.0,
                    facecolor=bounded_colour if measured_bounded else growing_colour,
                    edgecolor="white",
                    hatch="//" if cell_summary["extension_kind"] == "constant_in_time" else None,
                )
            )
            if measured_bounded:
                measured_text = "bounded"
            else:
                measured_text = f"slope {cell_summary['fitted_log_log_slope']:.2f}"
            verdict = cell_summary["verdict_r1_boundedness"] and (
                cell_summary["verdict_r2_rate"] in (None, True)
            )
            ax.text(
                column,
                row,
                f"{measured_text}\n$e={cell_summary['predicted_exponent']:g}$"
                + ("" if verdict else "\nPREDICTION FAILS"),
                ha="center",
                va="center",
                fontsize=7,
            )
        # Dotted stroke: predicted boundary m_B - p = sigma between the last
        # bounded and the first growing truncation of each row, drawn only
        # when a growing truncation exists (truncations have m_B <= 2p - 1).
        sobolev_critical_index = decay_exponent - 0.5
        for half_order in half_orders:
            boundary_position = half_order + sobolev_critical_index
            if boundary_position < 2 * half_order - 1:
                ax.plot(
                    [boundary_position, boundary_position],
                    [half_order - 0.5, half_order + 0.5],
                    ":",
                    color="black",
                    linewidth=1.6,
                )
        ax.set_xlim(-0.5, maximal_remainder_order + 0.5)
        ax.set_ylim(max(half_orders) + 0.5, min(half_orders) - 0.5)
        ax.set_xticks(range(maximal_remainder_order + 1))
        ax.set_yticks(half_orders)
        ax.set_xlabel(r"Remainder order $m_B$")
        ax.set_ylabel(r"Half order $p$")
        ax.set_title(
            rf"$\alpha={decay_exponent:g}$: $g_\alpha\in H^s$ iff $s<{sobolev_critical_index:g}$",
            fontsize=9,
        )
        ax.set_aspect("equal")

    legend = fig.legend(
        handles=[
            Patch(facecolor=bounded_colour, label=r"Measured: $E(K)$ bounded"),
            Patch(facecolor=growing_colour, label=r"Measured: $E(K)$ grows (fitted slope)"),
            Patch(facecolor="white", edgecolor="black", hatch="//", label=r"Constant in time $h=g$ ($m_B=2p$)"),
            Line2D([0], [0], color="black", linestyle=":", linewidth=1.6, label=r"Predicted boundary $m_B-p=\alpha-1/2$"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.13),
        ncol=4,
        fontsize=8,
        frameon=True,
    )
    fig.tight_layout(rect=[0.0, 0.25, 1.0, 1.0])
    figure_path = run_directory / THRESHOLD_TABLE_FIGURE_FILENAME
    finalize_figure(
        fig,
        figure_path,
        legends=[legend],
        axes=list(axes.ravel()),
        formula=(
            r"measured bounded: $(E(K_{\max})-E(K_{\max}/2))/E(K_{\max})<10^{-3}$; "
            r"$e=2(m_B-s_A-\alpha+1/2)$ predicted exponent, $s_A=p$ (truncation) or $0$ ($h=g$)"
            "\n"
            r"Propositions 4.2(iii) and 4.3: for a truncation, $E_h<+\infty$ iff "
            r"$g\in H^{s_B}$ with $s_B=\max(m_B-p,0)$, that is iff $m_B-p<\alpha-1/2$"
        ),
        formula_fontsize=8,
    )
    return figure_path


def render_all_figures(run_directory: Path) -> list[Path]:
    """Rebuild every figure of the run from its saved artefacts."""
    return [render_energy_figure(run_directory), render_threshold_table_figure(run_directory)]


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def parse_arguments(argv=None) -> argparse.Namespace:
    """Parse and validate the command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--log2-minimum-band-edge", type=int, default=4,
                        help="smallest dyadic band edge is 2 to this power (default 4)")
    parser.add_argument("--log2-maximum-band-edge", type=int, default=22,
                        help="largest dyadic band edge is 2 to this power (default 22)")
    parser.add_argument("--log2-fit-minimum-band-edge", type=int, default=16,
                        help="the log-log slope is fitted from 2 to this power up to the largest edge (default 16)")
    parser.add_argument("--chunk-length", type=int, default=2**19,
                        help="maximal number of positive wavenumbers per chunk (default 2**19)")
    parser.add_argument("--terminal-time", type=float, default=1.0,
                        help="horizon T (default 1.0)")
    parser.add_argument("--seed", type=int, default=0,
                        help="master seed, logged only: the measurement consumes no random number generator")
    parser.add_argument("--debug", action="store_true",
                        help="prepend '_debug_' to the output folder name (smoke runs)")
    parser.add_argument("--replot", type=Path, default=None, metavar="RUN_DIR",
                        help="rebuild the figures from the saved artefacts of a run directory")
    arguments = parser.parse_args(argv)

    if arguments.replot is not None:
        return arguments
    if (
        arguments.log2_maximum_band_edge < SMOKE_TEST_LOG2_MAXIMUM_BAND_EDGE_THRESHOLD
        and not arguments.debug
    ):
        parser.error(
            f"smoke-test guard: --log2-maximum-band-edge {arguments.log2_maximum_band_edge} < "
            f"{SMOKE_TEST_LOG2_MAXIMUM_BAND_EDGE_THRESHOLD} requires --debug"
        )
    if arguments.log2_minimum_band_edge < 1:
        parser.error("--log2-minimum-band-edge must be at least 1")
    if not (
        arguments.log2_minimum_band_edge
        <= arguments.log2_fit_minimum_band_edge
        <= arguments.log2_maximum_band_edge - 1
    ):
        parser.error(
            "--log2-fit-minimum-band-edge must lie between --log2-minimum-band-edge and "
            "--log2-maximum-band-edge - 1"
        )
    if arguments.chunk_length < 1:
        parser.error("--chunk-length must be a positive integer")
    if not arguments.terminal_time > 0.0:
        parser.error("--terminal-time must be positive")
    return arguments


def main(argv=None) -> int:
    """Run the measurement (or the replot path) and write the artefacts."""
    arguments = parse_arguments(argv)

    if arguments.replot is not None:
        configure_cli_script_logging(verbose=False)
        run_directory = arguments.replot.resolve()
        if not (run_directory / ENERGY_CURVES_FILENAME).exists():
            raise FileNotFoundError(
                f"no {ENERGY_CURVES_FILENAME} under {run_directory}; --replot expects a completed run"
            )
        for figure_path in render_all_figures(run_directory):
            LOGGER.info("Replotted figure from saved artefacts: %s", figure_path)
        return 0

    start_time = time.perf_counter()
    debug_prefix = "_debug_" if arguments.debug else ""
    config_tag = f"log2K{arguments.log2_maximum_band_edge}"
    run_directory = script_data_dir(__file__) / f"{debug_prefix}{utc_timestamp()}_{config_tag}"
    run_directory.mkdir(parents=True, exist_ok=False)

    init_logging(run_dir=run_directory)
    LOGGER.info("Full command line: %s", " ".join(sys.argv))
    LOGGER.info(
        "Runtime: Python %s | numpy %s | matplotlib %s | torch not used (pure-numpy measurement)",
        sys.version.split()[0],
        np.__version__,
        matplotlib.__version__,
    )
    log_parsed_args(LOGGER, arguments)
    LOGGER.info("Master seed %d (deterministic measurement: no random number generator is consumed)", arguments.seed)
    LOGGER.info(
        "Pre-registered tolerances: bounded increment < %.1e, growing increment > %.1e, "
        "slope within %.1e, ratio within %.1e",
        BOUNDED_RELATIVE_INCREMENT_TOLERANCE,
        GROWING_RELATIVE_INCREMENT_THRESHOLD,
        SLOPE_TOLERANCE,
        RATIO_TOLERANCE,
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
        "Git commit %s (dirty: %s)", run_metadata["git"].get("commit"), run_metadata["git"].get("dirty")
    )

    dyadic_band_edges = 2 ** np.arange(
        arguments.log2_minimum_band_edge, arguments.log2_maximum_band_edge + 1, dtype=np.int64
    )
    fit_minimum_band_edge = int(2**arguments.log2_fit_minimum_band_edge)
    LOGGER.info(
        "Dyadic band edges 2^%d .. 2^%d (%d edges); fit range [%d, %d]; chunk length %d; T = %g",
        arguments.log2_minimum_band_edge,
        arguments.log2_maximum_band_edge,
        len(dyadic_band_edges),
        fit_minimum_band_edge,
        int(dyadic_band_edges[-1]),
        arguments.chunk_length,
        arguments.terminal_time,
    )

    measurement_cells = build_measurement_cells()
    for half_order in HALF_ORDERS:
        LOGGER.info(
            "Generator p = %d: coefficients %s",
            half_order,
            complete_unit_coefficient_generator(half_order).coefficients,
        )
    LOGGER.info("Data: decay exponents %s (Sobolev critical index alpha - 1/2)", DECAY_EXPONENTS)
    LOGGER.info("Measurement grid: %d cells", len(measurement_cells))

    energy_curves: dict[str, np.ndarray] = {"dyadic_band_edges": dyadic_band_edges}
    cell_summaries: dict[str, dict] = {}
    for cell in measurement_cells:
        cell_start_time = time.perf_counter()
        extension = SplitSemigroupExtension(
            cell["datum"], cell["generator"], cell["subset_orders"], arguments.terminal_time
        )
        energy_values = energy_curve_at_dyadic_edges(
            extension, dyadic_band_edges, arguments.chunk_length
        )
        cross_check_discrepancy = cross_check_against_library_reference(
            extension, int(dyadic_band_edges[0]), float(energy_values[0])
        )
        cell_summary = analyse_measurement_cell(
            cell, dyadic_band_edges, energy_values, fit_minimum_band_edge, arguments.terminal_time
        )
        cell_summary["cross_check_relative_discrepancy"] = float(cross_check_discrepancy)
        energy_curves[f"energy_curve__{cell['cell_key']}"] = energy_values
        cell_summaries[cell["cell_key"]] = cell_summary
        LOGGER.info(
            "%-22s subset %-22s e = %+5.2f | E(Kmax) = %.6e | last-step increment %.3e | "
            "slope %+.4f | ratio %s | R1 %s | R2 %s | %.2f s",
            cell["cell_key"],
            str(cell["subset_orders"]),
            cell_summary["predicted_exponent"],
            cell_summary["energy_at_largest_band_edge"],
            cell_summary["relative_increment_over_last_dyadic_step"],
            cell_summary["fitted_log_log_slope"],
            (
                "-"
                if cell_summary["ratio_measured_over_predicted_at_largest_band_edge"] is None
                else f"{cell_summary['ratio_measured_over_predicted_at_largest_band_edge']:.5f}"
            ),
            cell_summary["verdict_r1_boundedness"],
            cell_summary["verdict_r2_rate"],
            time.perf_counter() - cell_start_time,
        )

    verdict_counts = {
        "r1_cells": len(cell_summaries),
        "r1_holds": sum(1 for summary in cell_summaries.values() if summary["verdict_r1_boundedness"]),
        "r2_cells": sum(1 for summary in cell_summaries.values() if summary["verdict_r2_rate"] is not None),
        "r2_holds": sum(1 for summary in cell_summaries.values() if summary["verdict_r2_rate"] is True),
    }
    summary = {
        "parameters": {
            "log2_minimum_band_edge": arguments.log2_minimum_band_edge,
            "log2_maximum_band_edge": arguments.log2_maximum_band_edge,
            "fit_minimum_band_edge": fit_minimum_band_edge,
            "chunk_length": arguments.chunk_length,
            "terminal_time": arguments.terminal_time,
            "half_orders": list(HALF_ORDERS),
            "decay_exponents": list(DECAY_EXPONENTS),
            "bounded_relative_increment_tolerance": BOUNDED_RELATIVE_INCREMENT_TOLERANCE,
            "growing_relative_increment_threshold": GROWING_RELATIVE_INCREMENT_THRESHOLD,
            "slope_tolerance": SLOPE_TOLERANCE,
            "ratio_tolerance": RATIO_TOLERANCE,
            "seed": arguments.seed,
            "debug": arguments.debug,
        },
        "cell_order": [cell["cell_key"] for cell in measurement_cells],
        "cells": cell_summaries,
        "verdict_counts": verdict_counts,
    }
    np.savez(run_directory / ENERGY_CURVES_FILENAME, **energy_curves)
    with open(run_directory / SUMMARY_FILENAME, "w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)
    LOGGER.info("Saved %s and %s under %s", ENERGY_CURVES_FILENAME, SUMMARY_FILENAME, run_directory)

    for figure_path in render_all_figures(run_directory):
        LOGGER.info("Saved figure %s", figure_path)

    LOGGER.info(
        "Verdicts: R1 holds in %d of %d cells; R2 holds in %d of %d cells",
        verdict_counts["r1_holds"],
        verdict_counts["r1_cells"],
        verdict_counts["r2_holds"],
        verdict_counts["r2_cells"],
    )
    LOGGER.info("Total wall-clock time %.2f s", time.perf_counter() - start_time)
    return 0


if __name__ == "__main__":
    sys.exit(main())
