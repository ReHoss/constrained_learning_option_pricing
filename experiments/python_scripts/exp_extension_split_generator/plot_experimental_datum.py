r"""The two experimental data of the methodology report, side by side.

Purpose.  The report "On boundary-constrained learning of partial differential
equations" (repository ``2026_01_29_constrained_learning_pde_lehalle_hosseinkhan``,
file ``boundary_constrained_learning_problem.tex``, section "Experimental setting
for the illustrations") uses **two different terminal data**, and the distinction
carries the whole of the stage-one result:

* **stage one** measures on the periodised second Bernoulli polynomial

  .. math::

      g(x) = B_2\!\left(\frac{x}{2\pi}\right)
           = \sum_{k \ge 1} \frac{\cos(k x)}{\pi^2 k^2},
      \qquad
      c_0 = 0, \quad c_k = \frac{1}{2\pi^2 k^2} \ (k \ne 0),

  a piecewise-quadratic, continuous function whose **first derivative jumps** at
  the single break point :math:`x^\star = 0`.  It is *not* band-limited, and the
  forcing of the raw extensions built from it is not square-integrable;

* **stage two** trains on its band-limited truncation at the datum-band edge
  :math:`K_g = 128`,

  .. math::

      g_{K_g}(x) = \sum_{k=1}^{K_g} \frac{\cos(k x)}{\pi^2 k^2},

  a trigonometric polynomial -- hence **analytic**, with no discontinuous
  derivative of any order.  Its forcing is band-limited, every floor is finite,
  and the Monte-Carlo objective is well defined.

The two are indistinguishable to the eye at the scale of the circle, which is
precisely why a figure is needed: the difference lives in the derivative and in
the tail of the spectrum, and the report's claims are about exactly those.

What the figure shows (four panels).

* The two data on the circle: they coincide to plotting accuracy.
* The corner window :math:`\{|x - x^\star| \le \pi/16\}` -- the window on which
  the stage-two relative error is measured -- where the truncation rounds the
  corner and oscillates.
* The first derivative: :math:`g'` jumps by :math:`-1/\pi` at :math:`x^\star`,
  while :math:`g_{K_g}'` is continuous and rings (the Gibbs oscillation of a
  truncated Fourier series).
* The exact coefficients :math:`|c_k| = 1/(2\pi^2 k^2)`, and where the truncation
  cuts them.

Verification performed, and logged.  The jump of :math:`g'` measured from the
closed form is compared with the exact value carried by the library datum class
(:attr:`PeriodisedBernoulliDatum.jump_of_rho_derivative`, equal to
:math:`-(\rho+1)!\,(2\pi)^{-\rho} = -1/\pi` at :math:`\rho = 1`); the run aborts
if they disagree beyond ``JUMP_AGREEMENT_TOLERANCE``.

Plot conventions (repository-wide).  Dashed = the analytical reference (the exact
datum :math:`g`); solid = what the trained runs actually use (the truncation
:math:`g_{K_g}`); dotted = auxiliary annotation (the break point :math:`x^\star`,
the datum-band edge :math:`K_g`).

Reproducibility.  Every array plotted is saved to ``datum_arrays.npz`` and every
scalar to ``summary.yaml``; ``--replot RUN_DIR`` rebuilds the figure from those
artefacts with no recomputation.
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

from learning_option_pricing.pde.periodic_spectral_toolbox import (  # noqa: E402
    PeriodisedBernoulliDatum,
)
from learning_option_pricing.utils.figure_layout import finalize_figure  # noqa: E402
from learning_option_pricing.utils.run_context import (  # noqa: E402
    collect_run_metadata,
    find_repo_root,
    init_logging,
    log_parsed_args,
    log_runtime_versions,
    script_data_dir,
    utc_timestamp,
    write_command_txt,
    write_json,
)

logger = logging.getLogger(__name__)

# The regularity order of the experimental datum: the first derivative is the
# lowest discontinuous one.
BERNOULLI_REGULARITY_INDEX = 1

# The break point of the periodised Bernoulli polynomial, and the half-width of
# the corner window on which the stage-two relative error is measured.
BREAK_POINT = 0.0
CORNER_WINDOW_HALF_WIDTH = math.pi / 16.0

# The measured jump of g' must reproduce the class's exact value to this
# tolerance; a disagreement means the closed form below and the library have
# drifted apart, and the run must not produce a figure.
JUMP_AGREEMENT_TOLERANCE = 1.0e-12

# A truncation far below this is an exploratory picture, not the datum of the
# report, and must be flagged with --debug.
SMOKE_TEST_DATUM_BAND_EDGE_THRESHOLD = 16

FIGURE_NAME = "experimental_datum.png"
ARRAYS_NAME = "datum_arrays.npz"
SUMMARY_NAME = "summary.yaml"

FORMULA_BOX = (
    r"$g(x)=B_2(x/2\pi)=\sum_{k\geq1}\frac{\cos(kx)}{\pi^2k^2}$"
    r"$\qquad c_0=0,\;\; c_k=\frac{1}{2\pi^2k^2}\;(k\neq0)$"
    r"$\qquad g_{K_g}(x)=\sum_{k=1}^{K_g}\frac{\cos(kx)}{\pi^2k^2}$"
    "\n"
    r"$g'(x)=\frac{x}{2\pi^2}-\frac{1}{2\pi}$ on $(0,2\pi)$, so "
    r"$[g']_{x^\star}=-\frac{1}{\pi}$ at $x^\star=0$"
    r"$\qquad g_{K_g}'(x)=-\sum_{k=1}^{K_g}\frac{\sin(kx)}{\pi^2k}$, continuous"
)
# The box formerly restated the stroke convention on a third line; it duplicated
# the legend verbatim and overlapped it at the foot of the canvas. The legend
# carries the convention, so the line is gone.


def exact_datum_values(spatial_points: np.ndarray) -> np.ndarray:
    """Closed-form values of the periodised second Bernoulli polynomial.

    Delegates to the library datum class, which carries the exact closed form.

    Args:
        spatial_points: Points on the circle, any shape.

    Returns:
        The values of :math:`g` at those points.
    """
    datum = PeriodisedBernoulliDatum(BERNOULLI_REGULARITY_INDEX)
    return np.asarray(datum.pointwise_values(spatial_points), dtype=float)


def exact_datum_derivative(spatial_points: np.ndarray) -> np.ndarray:
    r"""Closed-form first derivative of the exact datum, with its jump intact.

    On :math:`(0, 2\pi)`, :math:`g(x) = x^2/(4\pi^2) - x/(2\pi) + 1/6`, whence
    :math:`g'(x) = x/(2\pi^2) - 1/(2\pi)`.  The derivative is evaluated on the
    principal period, so the jump at the break point is reproduced exactly rather
    than smoothed by a finite difference.

    Args:
        spatial_points: Points on the circle, any shape.

    Returns:
        The values of :math:`g'` at those points; the value AT the break point
        itself is the right-hand limit.
    """
    reduced = np.mod(np.asarray(spatial_points, dtype=float), 2.0 * math.pi)
    return reduced / (2.0 * math.pi**2) - 1.0 / (2.0 * math.pi)


def truncated_datum_and_derivative(
    spatial_points: np.ndarray, datum_band_edge: int
) -> tuple[np.ndarray, np.ndarray]:
    r"""The band-limited truncation and its derivative, from the exact coefficients.

    The coefficients are taken from the library datum class rather than
    re-derived here, so that the truncation cannot drift from the exact datum it
    truncates.  With :math:`c_k` real and even, the real form of the partial sum
    is :math:`g_{K}(x) = \sum_{k=1}^{K} 2 c_k \cos(kx)` and its derivative
    :math:`g_{K}'(x) = -\sum_{k=1}^{K} 2 c_k k \sin(kx)`.

    Args:
        spatial_points: Points on the circle, shape ``(n,)``.
        datum_band_edge: The truncation wavenumber :math:`K_g`.

    Returns:
        The pair ``(g_K, g_K')`` evaluated at the points.
    """
    datum = PeriodisedBernoulliDatum(BERNOULLI_REGULARITY_INDEX)
    wavenumbers = np.arange(1, datum_band_edge + 1)
    coefficients = np.real(datum.fourier_coefficients(wavenumbers))
    phases = np.outer(np.asarray(spatial_points, dtype=float), wavenumbers)
    truncated_values = 2.0 * (coefficients * np.cos(phases)).sum(axis=1)
    truncated_derivative = -2.0 * (
        coefficients * wavenumbers * np.sin(phases)
    ).sum(axis=1)
    return truncated_values, truncated_derivative


def compute_arrays(datum_band_edge: int, number_of_points: int) -> dict:
    """Every array the figure needs, and the verification of the jump.

    Args:
        datum_band_edge: The truncation wavenumber :math:`K_g`.
        number_of_points: Number of points on the circle.

    Returns:
        A dictionary of arrays and scalars.

    Raises:
        ValueError: If the jump of :math:`g'` measured from the closed form
            disagrees with the exact value carried by the library datum class.
    """
    datum = PeriodisedBernoulliDatum(BERNOULLI_REGULARITY_INDEX)

    # The circle, centred on the break point so that the corner is in the middle.
    spatial_points = np.linspace(-math.pi, math.pi, number_of_points, endpoint=False)
    exact_values = exact_datum_values(spatial_points)
    exact_derivative = exact_datum_derivative(spatial_points)
    truncated_values, truncated_derivative = truncated_datum_and_derivative(
        spatial_points, datum_band_edge
    )

    # The jump of g' at the break point, from the one-sided limits of the closed
    # form, against the exact value the library carries.
    right_limit = exact_datum_derivative(np.asarray([1.0e-12]))[0]
    left_limit = exact_datum_derivative(np.asarray([-1.0e-12]))[0]
    measured_jump = float(right_limit - left_limit)
    exact_jump = float(datum.jump_of_rho_derivative)
    jump_deviation = abs(measured_jump - exact_jump)
    if jump_deviation > JUMP_AGREEMENT_TOLERANCE:
        raise ValueError(
            "the jump of g' measured from the closed form "
            f"({measured_jump!r}) disagrees with the exact value carried by "
            f"PeriodisedBernoulliDatum ({exact_jump!r}); deviation "
            f"{jump_deviation:.3e} exceeds {JUMP_AGREEMENT_TOLERANCE:.1e}. The "
            "closed form and the library have drifted apart and no figure is "
            "produced."
        )

    # The exact coefficients, and the truncation's support.
    coefficient_wavenumbers = np.arange(1, 4 * datum_band_edge + 1)
    exact_coefficients = np.abs(
        datum.fourier_coefficients(coefficient_wavenumbers)
    )
    truncated_coefficients = np.where(
        coefficient_wavenumbers <= datum_band_edge, exact_coefficients, 0.0
    )

    # The supremum distance between the two data, and between their derivatives.
    supremum_distance = float(np.abs(exact_values - truncated_values).max())
    supremum_derivative_distance = float(
        np.abs(exact_derivative - truncated_derivative).max()
    )

    return {
        "spatial_points": spatial_points,
        "exact_values": exact_values,
        "truncated_values": truncated_values,
        "exact_derivative": exact_derivative,
        "truncated_derivative": truncated_derivative,
        "coefficient_wavenumbers": coefficient_wavenumbers,
        "exact_coefficients": exact_coefficients,
        "truncated_coefficients": truncated_coefficients,
        "datum_band_edge": np.asarray([datum_band_edge]),
        "measured_jump": np.asarray([measured_jump]),
        "exact_jump": np.asarray([exact_jump]),
        "jump_deviation": np.asarray([jump_deviation]),
        "supremum_distance": np.asarray([supremum_distance]),
        "supremum_derivative_distance": np.asarray([supremum_derivative_distance]),
    }


def build_figure(arrays: dict, figure_path: Path) -> None:
    """Build and save the four-panel figure from the saved arrays.

    Args:
        arrays: The dictionary returned by :func:`compute_arrays`, or the
            contents of ``datum_arrays.npz``.
        figure_path: Where to write the PNG.
    """
    spatial_points = np.asarray(arrays["spatial_points"])
    exact_values = np.asarray(arrays["exact_values"])
    truncated_values = np.asarray(arrays["truncated_values"])
    exact_derivative = np.asarray(arrays["exact_derivative"])
    truncated_derivative = np.asarray(arrays["truncated_derivative"])
    coefficient_wavenumbers = np.asarray(arrays["coefficient_wavenumbers"])
    exact_coefficients = np.asarray(arrays["exact_coefficients"])
    datum_band_edge = int(np.asarray(arrays["datum_band_edge"]).reshape(-1)[0])
    exact_jump = float(np.asarray(arrays["exact_jump"]).reshape(-1)[0])

    exact_label = r"Exact datum $g$ (stage one)"
    truncated_label = rf"Truncation $g_{{K_g}}$, $K_g={datum_band_edge}$ (stage two)"

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4))

    # (0, 0) the two data on the circle: they coincide to plotting accuracy.
    axis = axes[0, 0]
    axis.plot(spatial_points, exact_values, "--", color="black", lw=1.6,
              label=exact_label)
    axis.plot(spatial_points, truncated_values, "-", color="tab:blue", lw=1.2,
              label=truncated_label)
    axis.axvline(BREAK_POINT, ls=":", color="grey", lw=1.0,
                 label=r"Break point $x^\star=0$")
    axis.set_xlabel(r"Spatial point $x$")
    axis.set_ylabel(r"Datum")
    axis.set_title("On the circle: the two coincide to plotting accuracy",
                   fontsize=9)
    axis.grid(True, alpha=0.3)

    # (0, 1) the corner window: the window on which the stage-two error is read.
    axis = axes[0, 1]
    window = np.abs(spatial_points - BREAK_POINT) <= CORNER_WINDOW_HALF_WIDTH
    axis.plot(spatial_points[window], exact_values[window], "--", color="black",
              lw=1.6)
    axis.plot(spatial_points[window], truncated_values[window], "-",
              color="tab:blue", lw=1.2)
    axis.axvline(BREAK_POINT, ls=":", color="grey", lw=1.0)
    axis.set_xlabel(r"Spatial point $x$")
    axis.set_ylabel(r"Datum")
    axis.set_title(
        r"Corner window $\{|x-x^\star|\leq\pi/16\}$: the corner, and its rounding",
        fontsize=9,
    )
    axis.grid(True, alpha=0.3)

    # (1, 0) the first derivative: the jump, and its absence.
    axis = axes[1, 0]
    axis.plot(spatial_points, exact_derivative, "--", color="black", lw=1.6)
    axis.plot(spatial_points, truncated_derivative, "-", color="tab:blue", lw=1.2)
    axis.axvline(BREAK_POINT, ls=":", color="grey", lw=1.0)
    axis.annotate(
        rf"$[g']_{{x^\star}}={exact_jump:.4f}=-1/\pi$",
        xy=(0.04, 0.08), xycoords="axes fraction", fontsize=8,
    )
    axis.set_xlabel(r"Spatial point $x$")
    axis.set_ylabel(r"First derivative")
    axis.set_title(
        r"$g'$ jumps at $x^\star$; $g_{K_g}'$ is continuous and rings",
        fontsize=9,
    )
    axis.grid(True, alpha=0.3)

    # (1, 1) the coefficients: the exact decay, and where the truncation cuts.
    axis = axes[1, 1]
    axis.loglog(coefficient_wavenumbers, exact_coefficients, "--", color="black",
                lw=1.6)
    inside = coefficient_wavenumbers <= datum_band_edge
    axis.loglog(coefficient_wavenumbers[inside], exact_coefficients[inside], "-",
                color="tab:blue", lw=1.2)
    axis.axvline(datum_band_edge, ls=":", color="grey", lw=1.0,
                 label=rf"Datum-band edge $K_g={datum_band_edge}$")
    axis.set_xlabel(r"Wavenumber $k$")
    axis.set_ylabel(r"$|c_k|$")
    axis.set_title(
        r"$|c_k|=1/(2\pi^2k^2)$; the truncation is zero beyond $K_g$",
        fontsize=9,
    )
    axis.grid(True, which="both", alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    band_handle, band_label = axes[1, 1].get_legend_handles_labels()
    legend = fig.legend(
        handles + band_handle,
        labels + band_label,
        loc="upper center",
        # Anchored ABOVE the formula box, which sits at the very bottom of the
        # canvas (figure_layout.formula_box places it at y = 0.012, growing
        # upwards). The layout checker warns when the two collide; the anchor and
        # the reserved strip below are set so that they do not.
        bbox_to_anchor=(0.5, 0.20),
        ncol=4,
        fontsize=8,
        frameon=True,
    )
    fig.tight_layout(rect=[0, 0.24, 1, 1])
    finalize_figure(fig, figure_path, legends=[legend], formula=FORMULA_BOX,
                    axes=list(axes.ravel()))


def write_artefacts(run_directory: Path, arrays: dict) -> None:
    """Save every array and scalar the figure is built from."""
    np.savez_compressed(run_directory / ARRAYS_NAME, **arrays)
    summary = {
        "datum": "periodised_bernoulli",
        "regularity_index": BERNOULLI_REGULARITY_INDEX,
        "break_point": BREAK_POINT,
        "corner_window_half_width": CORNER_WINDOW_HALF_WIDTH,
        "datum_band_edge": int(np.asarray(arrays["datum_band_edge"]).reshape(-1)[0]),
        "jump_of_first_derivative": {
            "measured_from_closed_form": float(
                np.asarray(arrays["measured_jump"]).reshape(-1)[0]
            ),
            "exact_from_library_class": float(
                np.asarray(arrays["exact_jump"]).reshape(-1)[0]
            ),
            "deviation": float(
                np.asarray(arrays["jump_deviation"]).reshape(-1)[0]
            ),
            "agreement": True,
        },
        "supremum_distance_between_the_two_data": float(
            np.asarray(arrays["supremum_distance"]).reshape(-1)[0]
        ),
        "supremum_distance_between_their_derivatives": float(
            np.asarray(arrays["supremum_derivative_distance"]).reshape(-1)[0]
        ),
    }
    with open(run_directory / SUMMARY_NAME, "w") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)


def regenerate_figures(run_directory: Path) -> Path:
    """Rebuild the figure from the saved arrays, with no recomputation."""
    arrays_path = run_directory / ARRAYS_NAME
    if not arrays_path.is_file():
        raise FileNotFoundError(f"no saved arrays at {arrays_path}")
    with np.load(arrays_path) as saved:
        arrays = {key: saved[key] for key in saved.files}
    figure_path = run_directory / FIGURE_NAME
    build_figure(arrays, figure_path)
    return figure_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Illustrate the two experimental data of the methodology report: "
            "the exact periodised Bernoulli datum used at stage one, and its "
            "band-limited truncation used at stage two."
        )
    )
    parser.add_argument(
        "--datum-band-edge",
        type=int,
        default=128,
        help="The truncation wavenumber K_g of the stage-two datum (default 128, "
        "the value used by the trained ablation).",
    )
    parser.add_argument(
        "--number-of-points",
        type=int,
        default=4096,
        help="Number of points on the circle at which the closed forms are "
        "evaluated (default 4096).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Master seed, recorded for the run-log contract; every computation "
        "in this script is deterministic and consumes no random number.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Prepend '_debug_' to the output folder name (exploratory runs; "
        "mandatory below the smoke-test band-edge threshold).",
    )
    parser.add_argument(
        "--replot",
        metavar="RUN_DIR",
        type=str,
        default=None,
        help="Rebuild the figure from the saved artefacts of an existing run "
        "directory, without any recomputation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.replot is not None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)sZ %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
            force=True,
        )
        run_directory = Path(args.replot).resolve()
        if not run_directory.is_dir():
            parser.error(f"--replot: run directory does not exist: {run_directory}")
        figure_path = regenerate_figures(run_directory)
        logger.info("Replotted figure from saved artefacts: %s", figure_path)
        return 0

    # Smoke-test guard: a truncation far below the report's is an exploratory
    # picture and must sort apart from the real ones.
    if (
        args.datum_band_edge < SMOKE_TEST_DATUM_BAND_EDGE_THRESHOLD
        and not args.debug
    ):
        parser.error(
            f"--datum-band-edge {args.datum_band_edge} is below the smoke-test "
            f"threshold {SMOKE_TEST_DATUM_BAND_EDGE_THRESHOLD}; pass --debug for "
            "an exploratory run"
        )

    start_wall_clock = time.perf_counter()
    debug_prefix = "_debug_" if args.debug else ""
    config_tag = f"Kg{args.datum_band_edge}_n{args.number_of_points}"
    run_directory = (
        script_data_dir(__file__) / f"{debug_prefix}{utc_timestamp()}_{config_tag}"
    )
    run_directory.mkdir(parents=True, exist_ok=False)
    init_logging(run_dir=run_directory)

    logger.info("Command line: %s", " ".join(sys.argv))
    log_runtime_versions(logger)
    log_parsed_args(logger, args)
    logger.info(
        "Master seed: %d (every computation here is deterministic and consumes "
        "no random number; the seed is recorded for the run-log contract)",
        args.seed,
    )

    repo_root = find_repo_root(Path(__file__))
    metadata = collect_run_metadata(
        run_dir=run_directory,
        repo_root=repo_root,
        script_name=Path(__file__).stem,
        command=sys.argv,
        params=dict(sorted(vars(args).items())),
        extra={
            "bernoulli_regularity_index": BERNOULLI_REGULARITY_INDEX,
            "break_point": BREAK_POINT,
            "corner_window_half_width": CORNER_WINDOW_HALF_WIDTH,
        },
    )
    write_json(run_directory / "run_metadata.json", metadata)
    write_command_txt(run_directory / "command.txt", sys.argv)

    arrays = compute_arrays(args.datum_band_edge, args.number_of_points)

    logger.info(
        "Jump of g' at the break point: measured %.12f, exact %.12f "
        "(-1/pi = %.12f); deviation %.3e, below the tolerance %.1e",
        float(arrays["measured_jump"][0]),
        float(arrays["exact_jump"][0]),
        -1.0 / math.pi,
        float(arrays["jump_deviation"][0]),
        JUMP_AGREEMENT_TOLERANCE,
    )
    logger.info(
        "Supremum distance between the two data: %.3e; between their "
        "derivatives: %.3e. The data are indistinguishable to the eye; the "
        "derivatives are not.",
        float(arrays["supremum_distance"][0]),
        float(arrays["supremum_derivative_distance"][0]),
    )

    write_artefacts(run_directory, arrays)
    figure_path = run_directory / FIGURE_NAME
    build_figure(arrays, figure_path)
    logger.info("Figure written: %s", figure_path)

    logger.info(
        "Total wall-clock time: %.2f s", time.perf_counter() - start_wall_clock
    )
    logger.info("Run directory: %s", run_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
