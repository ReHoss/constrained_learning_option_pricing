r"""The heat kernel, and the mollification it performs on the datum.

Purpose.  The extension catalogue of the methodology report (repository
``2026_01_29_constrained_learning_pde_lehalle_hosseinkhan``, file
``boundary_constrained_learning_problem.tex``, section "Experimental setting")
is given in real space by a single object: the periodic heat kernel

.. math::

    \mathcal{H}_s(x) = \frac{1}{2\pi} \sum_{k \in \mathbb{Z}} e^{-s k^2} e^{ikx},
    \qquad s > 0,

the third Jacobi theta function, of unit mass, converging to the Dirac mass as
:math:`s \to 0^+`.  Two of the four extensions are convolutions with it:

* the **graded-Gaussian** extension is
  :math:`h(\cdot, t) = \mathcal{H}_{\nu_c (T-t)} * g` -- the datum mollified at a
  bandwidth :math:`\sqrt{2 \nu_c (T - t)}` that VANISHES at the terminal slice, so
  that the terminal condition survives, and widens backwards in time;
* the **split** extension with :math:`A = \{\partial_{xx}\}` is the same formula
  at the matched bandwidth :math:`\nu_c = \nu`.  It does not merely resemble a
  mollifier; it IS one, and it is the one whose bandwidth is the generator's own
  diffusivity.

Everything the construction does is in that sentence, and none of it is legible
from the spectral form :math:`\hat h(k, t) = e^{-\nu_c k^2 (T-t)} c_k`.  This
script draws it.

What the figure shows (three panels).

* The kernel :math:`\mathcal{H}_s` at the bandwidths the extension passes through
  as :math:`t \to T`: it concentrates, without bound, into the Dirac mass.  That
  is why the terminal condition is met exactly.
* The mollified datum :math:`\mathcal{H}_{\nu(T-t)} * g` near the break point, at
  those same times: the corner is rounded far from the slice and recovered
  sharply at it.  The extension MEETS the datum at :math:`T` and smooths away
  from it.
* The matched bandwidth against the mis-matched one at a fixed time: what
  :math:`\nu_c = \nu / 2` actually costs is that the datum is smoothed half as
  much, and the second-order channel of the forcing is therefore not cancelled.

Truncation, and why it is inert.  Both the kernel and the convolution are
evaluated as finite Fourier sums truncated at ``SUM_TRUNCATION``.  The truncation
is provably inert rather than merely small: the run asserts that
:math:`e^{-s_{\min} K^2}` is below the double-precision epsilon at the smallest
bandwidth drawn, so no retained term is discarded and no discarded term is
representable.

Plot conventions.  Dashed black is the analytical reference (the exact datum
:math:`g`); the bandwidth is a hyperparameter axis and is therefore encoded in
COLOUR with the stroke kept solid; dotted grey is the break point.
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

BERNOULLI_REGULARITY_INDEX = 1
BREAK_POINT = 0.0

# The diffusivity of the Black-Scholes generator G2, sigma^2 / 2 at sigma = 0.5,
# and the terminal time of the study.
DIFFUSIVITY = 0.125
TERMINAL_TIME = 1.0

# The times at which the extension is drawn, as fractions of the terminal time.
# The last is close to T, where the kernel is at its most concentrated.
TIME_FRACTIONS = (0.0, 0.5, 0.9, 0.99)

# Half-width of the window on which the corner is shown.
CORNER_WINDOW_HALF_WIDTH = math.pi / 8.0

# Truncation of every Fourier sum. Its inertness is ASSERTED at run time against
# the smallest bandwidth drawn, not assumed: see check_truncation_is_inert.
SUM_TRUNCATION = 1024
DOUBLE_PRECISION_EPSILON = 2.3e-16

FIGURE_NAME = "heat_kernel_mollification.png"
ARRAYS_NAME = "kernel_arrays.npz"
SUMMARY_NAME = "summary.yaml"

FORMULA_BOX = (
    r"$\mathcal{H}_s(x)=\frac{1}{2\pi}\sum_{k\in\mathbb{Z}}e^{-sk^2}e^{ikx}$,  "
    r"$\int_0^{2\pi}\mathcal{H}_s=1$,  $\mathcal{H}_s\to\delta$ as $s\to0^+$"
    r"$\qquad h(\cdot,t)=\mathcal{H}_{\nu_c(T-t)}*g$,  bandwidth "
    r"$\sqrt{2\nu_c(T-t)}\to0$ as $t\to T$"
    "\n"
    r"Matched $\nu_c=\nu$: the split $A=\{\partial_{xx}\}$ extension.  "
    r"Mis-matched $\nu_c=\nu/2$.  "
    r"$g(x)=\sum_{k\geq1}\frac{\cos(kx)}{\pi^2k^2}$, "
    r"$c_k=\frac{1}{2\pi^2k^2}$"
)


def check_truncation_is_inert(smallest_bandwidth: float) -> float:
    """Assert that the Fourier truncation discards nothing representable.

    Args:
        smallest_bandwidth: The smallest :math:`s` at which the kernel is drawn.

    Returns:
        The magnitude of the first discarded term.

    Raises:
        ValueError: If that magnitude is at or above the double-precision
            epsilon, in which case the truncation would alter the drawing and
            must not pass silently.
    """
    first_discarded = math.exp(-smallest_bandwidth * (SUM_TRUNCATION + 1) ** 2)
    if first_discarded >= DOUBLE_PRECISION_EPSILON:
        raise ValueError(
            f"the Fourier truncation at K = {SUM_TRUNCATION} is NOT inert at the "
            f"smallest bandwidth s = {smallest_bandwidth!r}: the first discarded "
            f"term has magnitude {first_discarded:.3e}, at or above the "
            f"double-precision epsilon {DOUBLE_PRECISION_EPSILON:.1e}. Raise "
            "SUM_TRUNCATION rather than let a truncation alter the figure."
        )
    return first_discarded


def heat_kernel(spatial_points: np.ndarray, bandwidth: float) -> np.ndarray:
    r"""The periodic heat kernel :math:`\mathcal{H}_s` on the circle.

    Args:
        spatial_points: Points on the circle, shape ``(n,)``.
        bandwidth: The scale :math:`s > 0`.

    Returns:
        :math:`\mathcal{H}_s` at those points.
    """
    wavenumbers = np.arange(1, SUM_TRUNCATION + 1)
    decay = np.exp(-bandwidth * wavenumbers.astype(float) ** 2)
    phases = np.outer(np.asarray(spatial_points, dtype=float), wavenumbers)
    return (1.0 + 2.0 * (decay * np.cos(phases)).sum(axis=1)) / (2.0 * math.pi)


def mollified_datum(spatial_points: np.ndarray, bandwidth: float) -> np.ndarray:
    r"""The datum convolved with the heat kernel, :math:`\mathcal{H}_s * g`.

    The coefficients are taken from the library datum class, so the mollification
    cannot drift from the datum it smooths; convolution multiplies them by
    :math:`e^{-sk^2}`.

    Args:
        spatial_points: Points on the circle, shape ``(n,)``.
        bandwidth: The scale :math:`s \ge 0`; zero returns the datum itself.

    Returns:
        :math:`(\mathcal{H}_s * g)` at those points.
    """
    datum = PeriodisedBernoulliDatum(BERNOULLI_REGULARITY_INDEX)
    wavenumbers = np.arange(1, SUM_TRUNCATION + 1)
    coefficients = np.real(datum.fourier_coefficients(wavenumbers))
    decay = np.exp(-bandwidth * wavenumbers.astype(float) ** 2)
    phases = np.outer(np.asarray(spatial_points, dtype=float), wavenumbers)
    return 2.0 * (coefficients * decay * np.cos(phases)).sum(axis=1)


def compute_arrays(number_of_points: int) -> dict:
    """Every array the figure needs, and the truncation check."""
    bandwidths = [DIFFUSIVITY * (TERMINAL_TIME - f * TERMINAL_TIME)
                  for f in TIME_FRACTIONS]
    first_discarded = check_truncation_is_inert(min(b for b in bandwidths if b > 0))

    spatial_points = np.linspace(-math.pi, math.pi, number_of_points, endpoint=False)
    kernels = np.stack([heat_kernel(spatial_points, s) for s in bandwidths])
    extensions = np.stack([mollified_datum(spatial_points, s) for s in bandwidths])
    exact = np.asarray(
        PeriodisedBernoulliDatum(BERNOULLI_REGULARITY_INDEX)
        .pointwise_values(spatial_points),
        dtype=float,
    )
    # Matched against mis-matched, at the inception time t = 0.
    matched = mollified_datum(spatial_points, DIFFUSIVITY * TERMINAL_TIME)
    mismatched = mollified_datum(spatial_points, 0.5 * DIFFUSIVITY * TERMINAL_TIME)

    return {
        "spatial_points": spatial_points,
        "bandwidths": np.asarray(bandwidths),
        "time_fractions": np.asarray(TIME_FRACTIONS),
        "kernels": kernels,
        "extensions": extensions,
        "exact_datum": exact,
        "matched_at_inception": matched,
        "mismatched_at_inception": mismatched,
        "first_discarded_term": np.asarray([first_discarded]),
        "kernel_masses": np.asarray(
            [float(np.trapezoid(k, spatial_points) + k[0] * 0.0) for k in kernels]
        ),
    }


def build_figure(arrays: dict, figure_path: Path) -> None:
    """The three panels."""
    x = np.asarray(arrays["spatial_points"])
    bandwidths = np.asarray(arrays["bandwidths"]).reshape(-1)
    fractions = np.asarray(arrays["time_fractions"]).reshape(-1)
    kernels = np.asarray(arrays["kernels"])
    extensions = np.asarray(arrays["extensions"])
    exact = np.asarray(arrays["exact_datum"])
    matched = np.asarray(arrays["matched_at_inception"])
    mismatched = np.asarray(arrays["mismatched_at_inception"])

    # The bandwidth is a hyperparameter axis: colour, not stroke.
    colours = [plt.cm.viridis(v) for v in np.linspace(0.12, 0.82, len(bandwidths))]
    window = np.abs(x - BREAK_POINT) <= CORNER_WINDOW_HALF_WIDTH

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.1))

    # (0) the kernel concentrating into the Dirac mass
    axis = axes[0]
    for kern, s, f, colour in zip(kernels, bandwidths, fractions, colours):
        axis.plot(x, kern, "-", color=colour, lw=1.4,
                  label=rf"$t/T={f:g}$,  $s=\nu(T-t)={s:.4g}$")
    axis.axvline(BREAK_POINT, ls=":", color="grey", lw=1.0)
    axis.set_xlabel(r"Spatial point $x$")
    axis.set_ylabel(r"$\mathcal{H}_s(x)$")
    axis.set_title(r"The kernel concentrates into $\delta$ as $t\to T$", fontsize=9)
    axis.grid(True, alpha=0.3)

    # (1) the extension: the corner, rounded away from T and recovered at it
    axis = axes[1]
    for ext, colour in zip(extensions, colours):
        axis.plot(x[window], ext[window], "-", color=colour, lw=1.4)
    axis.plot(x[window], exact[window], "--", color="black", lw=1.6,
              label=r"Exact datum $g$")
    axis.axvline(BREAK_POINT, ls=":", color="grey", lw=1.0,
                 label=r"Break point $x^\star$")
    axis.set_xlabel(r"Spatial point $x$")
    axis.set_ylabel(r"$h(x,t)=(\mathcal{H}_{\nu(T-t)}*g)(x)$")
    axis.set_title(
        r"The extension meets $g$ at $T$, and smooths away from it", fontsize=9
    )
    axis.grid(True, alpha=0.3)

    # (2) what the mis-matched bandwidth actually does
    axis = axes[2]
    axis.plot(x[window], mismatched[window], "-", color=plt.cm.viridis(0.62), lw=1.5,
              label=r"Mis-matched $\nu_c=\nu/2$")
    axis.plot(x[window], matched[window], "-", color=plt.cm.viridis(0.12), lw=1.5,
              label=r"Matched $\nu_c=\nu$ (the split $A=\{\partial_{xx}\}$)")
    axis.plot(x[window], exact[window], "--", color="black", lw=1.6)
    axis.axvline(BREAK_POINT, ls=":", color="grey", lw=1.0)
    axis.set_xlabel(r"Spatial point $x$")
    axis.set_ylabel(r"$h(x,0)$")
    axis.set_title(
        r"At $t=0$: half the bandwidth, half the smoothing", fontsize=9
    )
    axis.grid(True, alpha=0.3)

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l
    # The formula box grows upwards from y = 0.012; the legend is anchored clear of
    # it. The layout checker warns when they collide, and it is read.
    legend = fig.legend(handles, labels, loc="upper center",
                        bbox_to_anchor=(0.5, 0.36), ncol=3, fontsize=8, frameon=True)
    fig.tight_layout(rect=[0, 0.40, 1, 1])
    finalize_figure(fig, figure_path, legends=[legend], formula=FORMULA_BOX,
                    axes=list(axes))


def write_artefacts(run_directory: Path, arrays: dict) -> None:
    np.savez_compressed(run_directory / ARRAYS_NAME, **arrays)
    summary = {
        "diffusivity": DIFFUSIVITY,
        "terminal_time": TERMINAL_TIME,
        "time_fractions": list(TIME_FRACTIONS),
        "bandwidths": [float(b) for b in np.asarray(arrays["bandwidths"]).reshape(-1)],
        "sum_truncation": SUM_TRUNCATION,
        "first_discarded_term": float(
            np.asarray(arrays["first_discarded_term"]).reshape(-1)[0]
        ),
        "truncation_is_inert": True,
        "kernel_masses": [
            float(m) for m in np.asarray(arrays["kernel_masses"]).reshape(-1)
        ],
    }
    with open(run_directory / SUMMARY_NAME, "w") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)


def regenerate_figures(run_directory: Path) -> Path:
    with np.load(run_directory / ARRAYS_NAME) as saved:
        arrays = {k: saved[k] for k in saved.files}
    path = run_directory / FIGURE_NAME
    build_figure(arrays, path)
    return path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Draw the periodic heat kernel and the mollification it "
        "performs on the datum: the real-space content of the graded-Gaussian "
        "and split extensions."
    )
    parser.add_argument("--number-of-points", type=int, default=4096,
                        help="points on the circle (default 4096)")
    parser.add_argument("--seed", type=int, default=0,
                        help="master seed, recorded for the run-log contract; "
                        "this script is deterministic and consumes no random number")
    parser.add_argument("--debug", action="store_true",
                        help="prepend '_debug_' to the output folder name")
    parser.add_argument("--replot", metavar="RUN_DIR", type=str, default=None,
                        help="rebuild the figure from a run's saved artefacts")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.replot is not None:
        logging.basicConfig(level=logging.INFO, force=True,
                            format="%(asctime)sZ %(levelname)s [%(name)s] %(message)s",
                            datefmt="%Y-%m-%dT%H:%M:%S")
        run_directory = Path(args.replot).resolve()
        if not run_directory.is_dir():
            parser.error(f"--replot: no such run directory: {run_directory}")
        logger.info("Replotted: %s", regenerate_figures(run_directory))
        return 0

    start = time.perf_counter()
    debug_prefix = "_debug_" if args.debug else ""
    config_tag = f"nu{DIFFUSIVITY}_n{args.number_of_points}"
    run_directory = (
        script_data_dir(__file__) / f"{debug_prefix}{utc_timestamp()}_{config_tag}"
    )
    run_directory.mkdir(parents=True, exist_ok=False)
    init_logging(run_dir=run_directory)

    logger.info("Command line: %s", " ".join(sys.argv))
    log_runtime_versions(logger)
    log_parsed_args(logger, args)
    logger.info("Master seed: %d (deterministic; no random number is consumed)",
                args.seed)

    repo_root = find_repo_root(Path(__file__))
    metadata = collect_run_metadata(
        run_dir=run_directory, repo_root=repo_root,
        script_name=Path(__file__).stem, command=sys.argv,
        params=dict(sorted(vars(args).items())),
        extra={"diffusivity": DIFFUSIVITY, "terminal_time": TERMINAL_TIME,
               "time_fractions": list(TIME_FRACTIONS),
               "sum_truncation": SUM_TRUNCATION},
    )
    write_json(run_directory / "run_metadata.json", metadata)
    write_command_txt(run_directory / "command.txt", sys.argv)

    arrays = compute_arrays(args.number_of_points)
    logger.info(
        "Fourier truncation at K = %d is INERT: the first discarded term has "
        "magnitude %.3e, below the double-precision epsilon %.1e, at the "
        "smallest bandwidth drawn.",
        SUM_TRUNCATION,
        float(arrays["first_discarded_term"][0]),
        DOUBLE_PRECISION_EPSILON,
    )
    masses = np.asarray(arrays["kernel_masses"]).reshape(-1)
    logger.info(
        "Kernel masses (each should be 1): %s -- maximum deviation %.2e",
        ", ".join(f"{m:.6f}" for m in masses),
        float(np.abs(masses - 1.0).max()),
    )

    write_artefacts(run_directory, arrays)
    build_figure(arrays, run_directory / FIGURE_NAME)
    logger.info("Figure written: %s", run_directory / FIGURE_NAME)
    logger.info("Total wall-clock time: %.2f s", time.perf_counter() - start)
    logger.info("Run directory: %s", run_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
