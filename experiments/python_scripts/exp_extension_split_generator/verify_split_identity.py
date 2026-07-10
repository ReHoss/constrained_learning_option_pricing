r"""Verification of the split-generator forcing identity (predictions P1 and P1b).

Proposition under test.  Proposition 7(i) ("Split-generator extension removes
the floor") of the methodology report "On boundary-constrained learning of
partial differential equations" (repository
2026_01_29_constrained_learning_pde_lehalle_hosseinkhan, file
``boundary_constrained_learning_problem.tex``): for a
constant-coefficient generator with Fourier symbol :math:`a(k)` split as
:math:`a(k) = a_A(k) + b(k)` (subset symbol :math:`a_A`, defect symbol
:math:`b`), the split-semigroup extension
:math:`\hat h(k, t) = e^{(T-t) a_A(k)} c_k` of a terminal datum with exact
Fourier coefficients :math:`c_k` satisfies the identity

.. math::

    \widehat{Lh}(k, t) = \partial_t \hat h(k, t) + a(k)\, \hat h(k, t)
                       = b(k)\, \hat h(k, t),

i.e. the forcing of the extension reduces to the defect symbol applied to the
extension coefficient (:math:`Lh = Bh`).

Checks performed (each with a PASS/FAIL line against the stated tolerance):

1. Spectral identity residuals (P1).  For the splits with subset orders
   :math:`A = \{2\}` (defect order :math:`m = 1`) and :math:`A = \{2, 1\}`
   (:math:`m = 0`) on the generators G1 (advection–diffusion–reaction) and G2
   (Black–Scholes log-price), and :math:`A = \{4\}` and :math:`A = \{4, 1\}`
   on G3 (biharmonic–advection–reaction), with the periodised Bernoulli datum
   of regularity index :math:`\rho = 1`, the relative residual
   :math:`\max_k |(\partial_t \hat h + a \hat h) - b \hat h| /
   \max_k |b \hat h|` is measured at :math:`t \in \{0, T/2, 0.99\,T\}` over
   the symmetric wavenumber band and must not exceed :math:`10^{-13}`.

2. Finite-difference-in-time convergence (P1).  The analytic time derivative
   :math:`\partial_t \hat h(k, t)` is compared at :math:`t = T/2` against the
   central difference
   :math:`[\hat h(k, t + \delta) - \hat h(k, t - \delta)] / (2\delta)` over a
   dyadic sweep :math:`\delta = 2^{-j}`; the relative error, normalised by
   :math:`\max_k |\partial_t \hat h(k, t)|`, satisfies
   :math:`E(\delta) = O(\delta^2)` as :math:`\delta \to 0`, and the fitted
   log-log slope must belong to :math:`[1.9, 2.1]`.

3. Gaussian-smoothed put cancellation (P1b, torch autograd, ``float64``, on
   the real line).  The matched-variance Gaussian smoothing of the put payoff
   has the closed form
   :math:`h(x, \tau) = K\,\Phi(d) - e^{x + \sigma^2 \tau / 2}\,
   \Phi(d - \sigma\sqrt{\tau})` with
   :math:`d = (\ln K - x) / (\sigma\sqrt{\tau})`, :math:`\tau = T - t`, and
   :math:`\Phi` the standard normal cumulative distribution function.  Since
   :math:`\partial_t h = -\partial_\tau h` and :math:`h` solves the heat
   equation in :math:`\tau`, the second-order part of the Black–Scholes
   operator cancels:
   :math:`\partial_t h + \tfrac{\sigma^2}{2} \partial_{xx} h = 0`, so
   :math:`\partial_t h + \tfrac{\sigma^2}{2} \partial_{xx} h +
   (r - \tfrac{\sigma^2}{2}) \partial_x h - r h =
   (r - \tfrac{\sigma^2}{2}) \partial_x h - r h`.  The cancellation is
   verified with ``torch.autograd`` on the grid
   :math:`x \in [\ln 60, \ln 140]`, :math:`\tau \in [0.05, 1]`, with
   :math:`K = 100`, :math:`\sigma = 0.5`, :math:`r = 0.03`, to a relative
   residual of at most :math:`10^{-10}`.  Normalisation note: the residual is
   normalised by the grid-wide scale
   :math:`\max(\max |\partial_t h|, \max |\tfrac{\sigma^2}{2}
   \partial_{xx} h|)`, because the two terms vanish simultaneously along the
   zero-crossing curve of :math:`\partial_{xx} h`, where a pointwise quotient
   is an ill-conditioned :math:`0/0` form.  A complementary pointwise
   relative residual, restricted to the grid points where the local scale
   :math:`\max(|\partial_t h|, |\tfrac{\sigma^2}{2} \partial_{xx} h|)`
   exceeds :math:`10^{-3}` times its grid maximum, is recorded in the summary
   and logged; the grid-normalised quantity remains the pass/fail criterion.

4. Exact-solution extension.  For
   :math:`\hat h(k, t) = e^{(T-t) a(k)} c_k` the forcing
   :math:`\partial_t \hat h + a(k) \hat h` vanishes identically; the maximum
   of :math:`|\widehat{Lh}(k, t)|` over the band and the three evaluation
   times is asserted to equal ``0.0`` exactly in floating point (the two
   assembled terms are the same computed product with opposite signs).

Measurement policy.  Every spectral quantity is evaluated from the exact
analytic Fourier coefficients over integer wavenumber arrays (vectorised
``complex128``); no FFT of sampled values is used anywhere in this script.

Artefacts.  All residual tables are saved to ``summary.yaml`` and ``.npz``
files before any figure is produced; the single figure (finite-difference
convergence, log-log, solid measured curves and a dashed slope-2 reference)
is rebuilt from the saved artefacts alone, so ``--replot RUN_DIR``
regenerates it without recomputation.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _figure_layout import finalize_figure  # noqa: E402
from learning_option_pricing.pde.periodic_spectral_toolbox import (  # noqa: E402
    ConstantCoefficientGenerator,
    PeriodisedBernoulliDatum,
    advection_diffusion_reaction,
    biharmonic_advection_reaction,
    black_scholes_log_price,
    symmetric_wavenumber_band,
)
from learning_option_pricing.pde.terminal_data_extensions import (  # noqa: E402
    ExactSolutionExtension,
    SplitSemigroupExtension,
)
from learning_option_pricing.utils.run_context import (  # noqa: E402
    collect_run_metadata,
    find_repo_root,
    init_logging,
    log_parsed_args,
    log_runtime_versions,
    script_data_dir,
    utc_timestamp,
    write_json,
)

import logging  # noqa: E402

logger = logging.getLogger(Path(__file__).stem)

# ---------------------------------------------------------------------------
# Constants of the study (single source of truth: the mathematical
# specification of the spectral forcing study).
# ---------------------------------------------------------------------------

TERMINAL_TIME = 1.0
BERNOULLI_REGULARITY_INDEX = 1

# Tolerances of the PASS/FAIL criteria (stated in the module docstring).
SPLIT_IDENTITY_RELATIVE_TOLERANCE = 1.0e-13
FINITE_DIFFERENCE_SLOPE_INTERVAL = (1.9, 2.1)
GAUSSIAN_SMOOTHED_PUT_RELATIVE_TOLERANCE = 1.0e-10

# Evaluation times of the spectral identity check: t in {0, T/2, 0.99 T}.
IDENTITY_EVALUATION_TIMES = (0.0, 0.5 * TERMINAL_TIME, 0.99 * TERMINAL_TIME)

# Evaluation time of the finite-difference sweep (interior point, so that
# t +/- delta stays inside [0, T] for every dyadic step of the sweep).
FINITE_DIFFERENCE_EVALUATION_TIME = 0.5 * TERMINAL_TIME

# Black–Scholes parameters of the Gaussian-smoothed put check (P1b).
PUT_STRIKE = 100.0
PUT_VOLATILITY = 0.5
PUT_RISK_FREE_RATE = 0.03
PUT_LOG_PRICE_LOWER = math.log(60.0)
PUT_LOG_PRICE_UPPER = math.log(140.0)
PUT_TIME_TO_MATURITY_LOWER = 0.05
PUT_TIME_TO_MATURITY_UPPER = 1.0

# Smoke-test guard: a real run must use a band at least this wide; anything
# smaller is an exploratory smoke test and must be flagged with --debug.
SMOKE_TEST_MAXIMUM_WAVENUMBER_THRESHOLD = 64


# ---------------------------------------------------------------------------
# Split configurations under test
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitConfiguration:
    """One (generator, subset-of-orders) pair whose identity is verified.

    Attributes:
        display_name: Human-readable generator name used in tables, legends
            and log lines.
        generator: The constant-coefficient generator being split.
        subset_orders: Differential orders defining the subset symbol
            :math:`a_A`.
    """

    display_name: str
    generator: ConstantCoefficientGenerator
    subset_orders: tuple[int, ...]


def build_split_configurations() -> list[SplitConfiguration]:
    """Return the six split configurations of prediction P1.

    Splits with subset orders ``(2,)`` (defect order 1) and ``(2, 1)``
    (defect order 0) on G1 and G2, and ``(4,)`` and ``(4, 1)`` on G3.
    """
    generator_one = advection_diffusion_reaction()
    generator_two = black_scholes_log_price()
    generator_three = biharmonic_advection_reaction()
    return [
        SplitConfiguration("Advection–diffusion–reaction (G1)", generator_one, (2,)),
        SplitConfiguration("Advection–diffusion–reaction (G1)", generator_one, (2, 1)),
        SplitConfiguration("Black–Scholes log-price (G2)", generator_two, (2,)),
        SplitConfiguration("Black–Scholes log-price (G2)", generator_two, (2, 1)),
        SplitConfiguration("Biharmonic–advection–reaction (G3)", generator_three, (4,)),
        SplitConfiguration("Biharmonic–advection–reaction (G3)", generator_three, (4, 1)),
    ]


def configuration_label(configuration: SplitConfiguration, defect_order: int) -> str:
    """Legend/table label naming the generator, the subset and the defect order."""
    subset_rendering = ",".join(str(order) for order in configuration.subset_orders)
    return (
        f"{configuration.display_name}, "
        f"$A=\\{{{subset_rendering}\\}}$, $m={defect_order}$"
    )


# ---------------------------------------------------------------------------
# Check 1: spectral identity residuals
# ---------------------------------------------------------------------------


def measure_split_identity_residuals(
    configurations: list[SplitConfiguration],
    datum: PeriodisedBernoulliDatum,
    wavenumber_band: np.ndarray,
) -> list[dict]:
    r"""Relative residual of the identity :math:`\partial_t \hat h + a \hat h = b \hat h`.

    For every configuration and every evaluation time the residual

    .. math::

        \frac{\max_k |(\partial_t \hat h + a \hat h) - b \hat h|}
             {\max_k |b \hat h|}

    is measured over the supplied wavenumber band.  The numerator uses the
    library's assembled forcing coefficient (which computes
    :math:`\partial_t \hat h + a \hat h`); the comparison value is the defect
    symbol multiplied by the extension coefficient.

    Returns:
        One row per (configuration, time) with the residual, the tolerance
        and the PASS/FAIL outcome.
    """
    residual_rows: list[dict] = []
    for configuration in configurations:
        extension = SplitSemigroupExtension(
            datum,
            configuration.generator,
            configuration.subset_orders,
            terminal_time=TERMINAL_TIME,
        )
        defect_symbol_values = extension.generator_split.defect_symbol(wavenumber_band)
        defect_order = int(extension.defect_order)
        for evaluation_time in IDENTITY_EVALUATION_TIMES:
            assembled_forcing = extension.forcing_coefficient(
                wavenumber_band, evaluation_time
            )
            defect_times_extension = defect_symbol_values * extension.extension_coefficient(
                wavenumber_band, evaluation_time
            )
            numerator = float(
                np.max(np.abs(assembled_forcing - defect_times_extension))
            )
            denominator = float(np.max(np.abs(defect_times_extension)))
            relative_residual = numerator / denominator
            passed = relative_residual <= SPLIT_IDENTITY_RELATIVE_TOLERANCE
            residual_rows.append(
                {
                    "generator": configuration.generator.name,
                    "display_name": configuration.display_name,
                    "subset_orders": list(configuration.subset_orders),
                    "defect_order": defect_order,
                    "evaluation_time": float(evaluation_time),
                    "relative_residual": relative_residual,
                    "tolerance": SPLIT_IDENTITY_RELATIVE_TOLERANCE,
                    "passed": bool(passed),
                }
            )
            logger.info(
                "%s | split identity | %s A=%s m=%d t=%.2f | residual %.3e <= %.0e",
                "PASS" if passed else "FAIL",
                configuration.generator.name,
                list(configuration.subset_orders),
                defect_order,
                evaluation_time,
                relative_residual,
                SPLIT_IDENTITY_RELATIVE_TOLERANCE,
            )
    return residual_rows


# ---------------------------------------------------------------------------
# Check 2: finite-difference-in-time convergence of the analytic derivative
# ---------------------------------------------------------------------------


def measure_finite_difference_convergence(
    configurations: list[SplitConfiguration],
    datum: PeriodisedBernoulliDatum,
    wavenumber_band: np.ndarray,
    time_step_values: np.ndarray,
) -> tuple[list[dict], np.ndarray]:
    r"""Order of convergence of central differences towards :math:`\partial_t \hat h`.

    At the fixed interior time :math:`t = T/2` the relative error

    .. math::

        E(\delta) = \frac{\max_k |[\hat h(k, t+\delta) - \hat h(k, t-\delta)]
        / (2\delta) - \partial_t \hat h(k, t)|}{\max_k |\partial_t \hat h(k, t)|}

    is measured over the dyadic sweep of time steps, and the log-log slope is
    fitted; central differences predict :math:`E(\delta) = O(\delta^2)` as
    :math:`\delta \to 0`, so the fitted slope must belong to ``[1.9, 2.1]``.

    Returns:
        The per-configuration result rows and the error matrix of shape
        ``(number_of_configurations, number_of_time_steps)``.
    """
    evaluation_time = FINITE_DIFFERENCE_EVALUATION_TIME
    relative_error_matrix = np.zeros(
        (len(configurations), time_step_values.size), dtype=np.float64
    )
    convergence_rows: list[dict] = []
    for configuration_index, configuration in enumerate(configurations):
        extension = SplitSemigroupExtension(
            datum,
            configuration.generator,
            configuration.subset_orders,
            terminal_time=TERMINAL_TIME,
        )
        analytic_time_derivative = extension.extension_coefficient_time_derivative(
            wavenumber_band, evaluation_time
        )
        normalisation_scale = float(np.max(np.abs(analytic_time_derivative)))
        for time_step_index, time_step in enumerate(time_step_values):
            central_difference = (
                extension.extension_coefficient(
                    wavenumber_band, evaluation_time + time_step
                )
                - extension.extension_coefficient(
                    wavenumber_band, evaluation_time - time_step
                )
            ) / (2.0 * time_step)
            relative_error_matrix[configuration_index, time_step_index] = float(
                np.max(np.abs(central_difference - analytic_time_derivative))
                / normalisation_scale
            )
        fitted_slope, _ = np.polyfit(
            np.log(time_step_values),
            np.log(relative_error_matrix[configuration_index]),
            1,
        )
        slope_lower, slope_upper = FINITE_DIFFERENCE_SLOPE_INTERVAL
        passed = slope_lower <= fitted_slope <= slope_upper
        defect_order = int(extension.defect_order)
        convergence_rows.append(
            {
                "generator": configuration.generator.name,
                "display_name": configuration.display_name,
                "subset_orders": list(configuration.subset_orders),
                "defect_order": defect_order,
                "evaluation_time": float(evaluation_time),
                "fitted_slope": float(fitted_slope),
                "slope_interval": list(FINITE_DIFFERENCE_SLOPE_INTERVAL),
                "time_step_values": [float(value) for value in time_step_values],
                "relative_errors": [
                    float(value)
                    for value in relative_error_matrix[configuration_index]
                ],
                "passed": bool(passed),
            }
        )
        logger.info(
            "%s | finite-difference order | %s A=%s m=%d | fitted slope %.4f in [%.1f, %.1f]",
            "PASS" if passed else "FAIL",
            configuration.generator.name,
            list(configuration.subset_orders),
            defect_order,
            fitted_slope,
            slope_lower,
            slope_upper,
        )
    return convergence_rows, relative_error_matrix


# ---------------------------------------------------------------------------
# Check 3 (P1b): Gaussian-smoothed put — the second-order part cancels
# ---------------------------------------------------------------------------


def _standard_normal_cdf(argument: torch.Tensor) -> torch.Tensor:
    r"""Standard normal cumulative distribution function :math:`\Phi` (``float64``)."""
    return 0.5 * (1.0 + torch.erf(argument / math.sqrt(2.0)))


def measure_gaussian_smoothed_put_cancellation(
    number_of_log_price_points: int,
    number_of_time_to_maturity_points: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    r"""Autograd verification that :math:`\partial_t h + \tfrac{\sigma^2}{2} \partial_{xx} h = 0`.

    The matched-variance Gaussian smoothing of the put payoff,

    .. math::

        h(x, \tau) = K\,\Phi(d) - e^{x + \sigma^2 \tau / 2}\,
        \Phi(d - \sigma\sqrt{\tau}),
        \qquad
        d = \frac{\ln K - x}{\sigma\sqrt{\tau}},
        \qquad
        \tau = T - t,

    solves the heat equation :math:`\partial_\tau h = \tfrac{\sigma^2}{2}
    \partial_{xx} h`; with :math:`\partial_t h = -\partial_\tau h` the
    second-order part of the Black–Scholes operator therefore cancels.  All
    derivatives are obtained by ``torch.autograd`` in ``float64``.

    The residual is normalised by the grid-wide scale
    :math:`\max(\max |\partial_t h|, \max |\tfrac{\sigma^2}{2}
    \partial_{xx} h|)` because both terms vanish simultaneously along the
    zero-crossing curve of :math:`\partial_{xx} h`, where a pointwise
    quotient is an ill-conditioned :math:`0/0` form.

    Returns:
        The summary row (scalars) and the field arrays saved for later
        replotting or inspection.
    """
    log_strike = math.log(PUT_STRIKE)
    log_price_axis = torch.linspace(
        PUT_LOG_PRICE_LOWER,
        PUT_LOG_PRICE_UPPER,
        number_of_log_price_points,
        dtype=torch.float64,
    )
    time_to_maturity_axis = torch.linspace(
        PUT_TIME_TO_MATURITY_LOWER,
        PUT_TIME_TO_MATURITY_UPPER,
        number_of_time_to_maturity_points,
        dtype=torch.float64,
    )
    log_price_grid, time_to_maturity_grid = torch.meshgrid(
        log_price_axis, time_to_maturity_axis, indexing="ij"
    )
    log_price_grid = log_price_grid.clone().requires_grad_(True)
    time_to_maturity_grid = time_to_maturity_grid.clone().requires_grad_(True)

    volatility_times_root_time = PUT_VOLATILITY * torch.sqrt(time_to_maturity_grid)
    gaussian_argument = (log_strike - log_price_grid) / volatility_times_root_time
    smoothed_put = PUT_STRIKE * _standard_normal_cdf(gaussian_argument) - torch.exp(
        log_price_grid + 0.5 * PUT_VOLATILITY**2 * time_to_maturity_grid
    ) * _standard_normal_cdf(gaussian_argument - volatility_times_root_time)

    (first_log_price_derivative,) = torch.autograd.grad(
        smoothed_put.sum(), log_price_grid, create_graph=True
    )
    (second_log_price_derivative,) = torch.autograd.grad(
        first_log_price_derivative.sum(), log_price_grid, retain_graph=True
    )
    (time_to_maturity_derivative,) = torch.autograd.grad(
        smoothed_put.sum(), time_to_maturity_grid
    )

    # tau = T - t, hence the time derivative is the negated tau derivative.
    time_derivative = (-time_to_maturity_derivative).detach().numpy()
    second_order_term = (
        (0.5 * PUT_VOLATILITY**2 * second_log_price_derivative).detach().numpy()
    )
    first_order_part = (
        (
            (PUT_RISK_FREE_RATE - 0.5 * PUT_VOLATILITY**2)
            * first_log_price_derivative
            - PUT_RISK_FREE_RATE * smoothed_put
        )
        .detach()
        .numpy()
    )
    residual_field = time_derivative + second_order_term
    # The quantity "full operator versus first-order part" was removed as
    # circular: the full-operator field was constructed as residual_field +
    # first_order_part, so the difference equals the residual up to
    # re-association rounding and measures nothing new.

    normalisation_scale = max(
        float(np.max(np.abs(time_derivative))),
        float(np.max(np.abs(second_order_term))),
    )
    max_absolute_residual = float(np.max(np.abs(residual_field)))
    relative_residual = max_absolute_residual / normalisation_scale
    # Complementary diagnostic: pointwise relative residual restricted to the
    # grid points where the local scale max(|dt h|, |(sigma^2/2) dxx h|)
    # exceeds 1e-3 times its grid maximum (the well-scaled points).
    local_scale_field = np.maximum(
        np.abs(time_derivative), np.abs(second_order_term)
    )
    well_scaled_relative_threshold = 1.0e-3
    well_scaled_mask = local_scale_field > (
        well_scaled_relative_threshold * float(np.max(local_scale_field))
    )
    max_pointwise_relative_residual_on_well_scaled_points = float(
        np.max(
            np.abs(residual_field[well_scaled_mask])
            / local_scale_field[well_scaled_mask]
        )
    )
    # The grid-normalised residual stays the primary pass/fail criterion: the
    # local scale crosses zero (along the zero-crossing curve of dxx h), where
    # a pointwise quotient is an ill-conditioned 0/0 form.
    passed = relative_residual <= GAUSSIAN_SMOOTHED_PUT_RELATIVE_TOLERANCE

    summary_row = {
        "strike": PUT_STRIKE,
        "volatility": PUT_VOLATILITY,
        "risk_free_rate": PUT_RISK_FREE_RATE,
        "log_price_interval": [PUT_LOG_PRICE_LOWER, PUT_LOG_PRICE_UPPER],
        "time_to_maturity_interval": [
            PUT_TIME_TO_MATURITY_LOWER,
            PUT_TIME_TO_MATURITY_UPPER,
        ],
        "number_of_log_price_points": int(number_of_log_price_points),
        "number_of_time_to_maturity_points": int(number_of_time_to_maturity_points),
        "max_absolute_residual": max_absolute_residual,
        "normalisation_scale": normalisation_scale,
        "relative_residual": relative_residual,
        "well_scaled_relative_threshold": well_scaled_relative_threshold,
        "number_of_well_scaled_grid_points": int(np.sum(well_scaled_mask)),
        "max_pointwise_relative_residual_on_well_scaled_points": (
            max_pointwise_relative_residual_on_well_scaled_points
        ),
        "tolerance": GAUSSIAN_SMOOTHED_PUT_RELATIVE_TOLERANCE,
        "passed": bool(passed),
    }
    field_arrays = {
        "log_price_axis": log_price_axis.numpy(),
        "time_to_maturity_axis": time_to_maturity_axis.numpy(),
        "time_derivative_field": time_derivative,
        "second_order_term_field": second_order_term,
        "residual_field": residual_field,
        "first_order_part_field": first_order_part,
    }
    logger.info(
        "%s | Gaussian-smoothed put (P1b) | grid-normalised relative residual "
        "%.3e <= %.0e (scale %.3e; primary criterion) | pointwise relative "
        "residual on the %d well-scaled grid points (local scale > %.0e of "
        "its grid maximum): %.3e",
        "PASS" if passed else "FAIL",
        relative_residual,
        GAUSSIAN_SMOOTHED_PUT_RELATIVE_TOLERANCE,
        normalisation_scale,
        int(np.sum(well_scaled_mask)),
        well_scaled_relative_threshold,
        max_pointwise_relative_residual_on_well_scaled_points,
    )
    return summary_row, field_arrays


# ---------------------------------------------------------------------------
# Check 4: exact-solution extension has identically zero forcing
# ---------------------------------------------------------------------------


def measure_exact_extension_forcing(
    generators: list[tuple[str, ConstantCoefficientGenerator]],
    datum: PeriodisedBernoulliDatum,
    wavenumber_band: np.ndarray,
) -> list[dict]:
    r"""Assert :math:`\max_{k, t} |\widehat{Lh}(k, t)| = 0` for the exact extension.

    The forcing of :math:`\hat h(k, t) = e^{(T-t) a(k)} c_k` is assembled as
    :math:`\partial_t \hat h + a(k) \hat h`; the two terms are the same
    computed product with opposite signs, so the maximum must equal ``0.0``
    exactly in floating point (not merely below a tolerance).

    Returns:
        One row per generator with the measured maximum and the PASS/FAIL
        outcome.
    """
    zero_forcing_rows: list[dict] = []
    for display_name, generator in generators:
        extension = ExactSolutionExtension(datum, generator, terminal_time=TERMINAL_TIME)
        max_absolute_forcing = 0.0
        for evaluation_time in IDENTITY_EVALUATION_TIMES:
            forcing_values = extension.forcing_coefficient(
                wavenumber_band, evaluation_time
            )
            max_absolute_forcing = max(
                max_absolute_forcing, float(np.max(np.abs(forcing_values)))
            )
        passed = max_absolute_forcing == 0.0
        zero_forcing_rows.append(
            {
                "generator": generator.name,
                "display_name": display_name,
                "max_absolute_forcing": max_absolute_forcing,
                "passed": bool(passed),
            }
        )
        logger.info(
            "%s | exact-extension zero forcing | %s | max |forcing| = %.1e (required exactly 0.0)",
            "PASS" if passed else "FAIL",
            generator.name,
            max_absolute_forcing,
        )
    return zero_forcing_rows


# ---------------------------------------------------------------------------
# Figure (rebuilt from the saved artefacts alone)
# ---------------------------------------------------------------------------


def regenerate_figures(run_directory: Path) -> Path:
    """Rebuild the finite-difference convergence figure from the saved ``.npz``.

    Reads ``finite_difference_convergence.npz`` only — no quantity is
    recomputed — so this function serves both the fresh run and the
    ``--replot`` path.

    Args:
        run_directory: A run directory produced by this script.

    Returns:
        The path of the written figure.
    """
    artefact_path = run_directory / "finite_difference_convergence.npz"
    payload = np.load(artefact_path, allow_pickle=False)
    time_step_values = payload["time_step_values"]
    relative_error_matrix = payload["relative_error_matrix"]
    fitted_slopes = payload["fitted_slopes"]
    labels = [str(label) for label in payload["configuration_labels"]]
    evaluation_time = float(payload["evaluation_time"])

    figure, axes = plt.subplots(figsize=(8.6, 7.0))
    colour_map = plt.get_cmap("viridis")
    number_of_configurations = len(labels)
    for configuration_index, label in enumerate(labels):
        # Solid stroke: measured quantity (repository plot convention);
        # the configuration index is encoded in a viridis colour.
        colour = colour_map(
            0.9 * configuration_index / max(number_of_configurations - 1, 1)
        )
        axes.loglog(
            time_step_values,
            relative_error_matrix[configuration_index],
            "-o",
            color=colour,
            lw=1.6,
            ms=3.5,
            label=f"{label} (fitted slope {fitted_slopes[configuration_index]:.3f})",
        )
    # Dashed stroke: analytical prediction (slope-2 power law), anchored to
    # the largest measured error at the coarsest time step.
    reference_anchor = float(np.max(relative_error_matrix[:, 0]))
    reference_curve = reference_anchor * (time_step_values / time_step_values[0]) ** 2
    axes.loglog(
        time_step_values,
        reference_curve,
        "--",
        color="black",
        lw=1.2,
        label=r"Slope-2 reference $\delta \mapsto C\,\delta^{2}$ (predicted order)",
    )
    axes.set_xlabel(r"Time step $\delta$")
    axes.set_ylabel(r"Relative error $E(\delta)$")
    axes.set_title(
        "Central-difference convergence towards the analytic "
        r"$\partial_t \hat h(k, t)$ at $t = "
        f"{evaluation_time:g}$",
        fontsize=10,
    )
    axes.grid(True, which="both", alpha=0.3)
    handles, handle_labels = axes.get_legend_handles_labels()
    legend = figure.legend(
        handles,
        handle_labels,
        loc="lower center",
        ncol=2,
        fontsize=7.5,
        frameon=True,
        bbox_to_anchor=(0.5, 0.135),
    )
    figure.tight_layout(rect=[0, 0.33, 1, 0.96])
    figure_path = run_directory / "finite_difference_convergence.png"
    finalize_figure(
        figure,
        figure_path,
        legends=[legend],
        axes=[axes],
        formula=(
            r"Split extension: $\hat h(k,t)=e^{(T-t)a_A(k)}c_k$, "
            r"$\partial_t\hat h(k,t)=-a_A(k)\,\hat h(k,t)$; central difference "
            r"$D_\delta\hat h(k,t)=[\hat h(k,t+\delta)-\hat h(k,t-\delta)]/(2\delta)$."
            "\n"
            r"Measured error $E(\delta)=\max_k|D_\delta\hat h(k,t)-\partial_t\hat h(k,t)|"
            r"/\max_k|\partial_t\hat h(k,t)| = O(\delta^2)$ as $\delta\to 0$; "
            r"Bernoulli datum with $\rho=1$, $T=1$."
        ),
        formula_fontsize=8,
    )
    return figure_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    """Command-line interface of the verification script."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--maximum-wavenumber",
        type=int,
        default=512,
        help="Band edge K of the symmetric wavenumber band used for the "
        "spectral identity and exact-extension checks (default 512).",
    )
    parser.add_argument(
        "--finite-difference-maximum-wavenumber",
        type=int,
        default=64,
        help="Band edge of the (smaller) band used in the finite-difference "
        "convergence sweep (default 64).",
    )
    parser.add_argument(
        "--finite-difference-min-exponent",
        type=int,
        default=4,
        help="Smallest dyadic exponent j of the sweep delta = 2**(-j); the "
        "coarsest time step is 2**(-j) (default 4).",
    )
    parser.add_argument(
        "--finite-difference-max-exponent",
        type=int,
        default=10,
        help="Largest dyadic exponent j of the sweep delta = 2**(-j); the "
        "finest time step is 2**(-j) (default 10).",
    )
    parser.add_argument(
        "--put-log-price-points",
        type=int,
        default=41,
        help="Number of log-price grid points of the Gaussian-smoothed put "
        "check (default 41).",
    )
    parser.add_argument(
        "--put-time-to-maturity-points",
        type=int,
        default=21,
        help="Number of time-to-maturity grid points of the Gaussian-smoothed "
        "put check (default 21).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Master seed, logged for the run-log contract; every computation "
        "in this script is deterministic, and only torch.manual_seed is "
        "applied (no numpy random number generator is consumed).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Prepend '_debug_' to the output folder name (exploratory / "
        "smoke-test runs; mandatory below the smoke-test band threshold).",
    )
    parser.add_argument(
        "--replot",
        metavar="RUN_DIR",
        type=str,
        default=None,
        help="Rebuild every figure of an existing run directory from its "
        "saved artefacts, without any recomputation.",
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

    # Smoke-test guard: a band below the threshold is an exploratory run and
    # must be flagged with --debug so its output folder sorts apart.
    if (
        args.maximum_wavenumber < SMOKE_TEST_MAXIMUM_WAVENUMBER_THRESHOLD
        and not args.debug
    ):
        parser.error(
            f"--maximum-wavenumber {args.maximum_wavenumber} is below the "
            f"smoke-test threshold {SMOKE_TEST_MAXIMUM_WAVENUMBER_THRESHOLD}; "
            "pass --debug for an exploratory run"
        )
    if args.finite_difference_min_exponent < 2:
        parser.error(
            "--finite-difference-min-exponent must be at least 2 so that "
            "t +/- delta stays inside [0, T] at t = T/2"
        )
    if args.finite_difference_max_exponent < args.finite_difference_min_exponent + 2:
        parser.error(
            "--finite-difference-max-exponent must exceed the minimum "
            "exponent by at least 2 (at least three sweep points are needed "
            "for a meaningful slope fit)"
        )

    start_wall_clock = time.perf_counter()
    debug_prefix = "_debug_" if args.debug else ""
    config_tag = (
        f"K{args.maximum_wavenumber}"
        f"_fdK{args.finite_difference_maximum_wavenumber}"
    )
    run_directory = (
        script_data_dir(__file__) / f"{debug_prefix}{utc_timestamp()}_{config_tag}"
    )
    run_directory.mkdir(parents=True, exist_ok=False)
    init_logging(run_dir=run_directory)

    logger.info("Command line: %s", " ".join(sys.argv))
    log_runtime_versions(logger)
    log_parsed_args(logger, args)
    logger.info(
        "Master seed: %d (every computation here is deterministic; the seed "
        "is recorded for the run-log contract, and only torch.manual_seed is "
        "applied)",
        args.seed,
    )
    torch.manual_seed(args.seed)

    repo_root = find_repo_root(Path(__file__))
    metadata = collect_run_metadata(
        run_dir=run_directory,
        repo_root=repo_root,
        script_name=Path(__file__).stem,
        command=sys.argv,
        params=dict(sorted(vars(args).items())),
        extra={
            "terminal_time": TERMINAL_TIME,
            "bernoulli_regularity_index": BERNOULLI_REGULARITY_INDEX,
            "identity_evaluation_times": list(IDENTITY_EVALUATION_TIMES),
            "finite_difference_evaluation_time": FINITE_DIFFERENCE_EVALUATION_TIME,
        },
    )
    write_json(run_directory / "run_metadata.json", metadata)
    logger.info(
        "Git commit: %s (dirty: %s)",
        metadata["git"].get("commit"),
        metadata["git"].get("dirty"),
    )

    datum = PeriodisedBernoulliDatum(BERNOULLI_REGULARITY_INDEX)
    configurations = build_split_configurations()
    identity_band = symmetric_wavenumber_band(args.maximum_wavenumber)
    finite_difference_band = symmetric_wavenumber_band(
        args.finite_difference_maximum_wavenumber
    )
    time_step_values = 2.0 ** (
        -np.arange(
            args.finite_difference_min_exponent,
            args.finite_difference_max_exponent + 1,
            dtype=np.float64,
        )
    )
    logger.info(
        "Bands: identity |k| <= %d, finite-difference |k| <= %d; dyadic time "
        "steps 2^-%d .. 2^-%d; Bernoulli datum rho = %d",
        args.maximum_wavenumber,
        args.finite_difference_maximum_wavenumber,
        args.finite_difference_min_exponent,
        args.finite_difference_max_exponent,
        BERNOULLI_REGULARITY_INDEX,
    )

    # --- Check 1: spectral identity residuals -----------------------------
    identity_rows = measure_split_identity_residuals(
        configurations, datum, identity_band
    )

    # --- Check 2: finite-difference convergence ---------------------------
    convergence_rows, relative_error_matrix = measure_finite_difference_convergence(
        configurations, datum, finite_difference_band, time_step_values
    )

    # --- Check 3 (P1b): Gaussian-smoothed put cancellation -----------------
    put_row, put_fields = measure_gaussian_smoothed_put_cancellation(
        args.put_log_price_points, args.put_time_to_maturity_points
    )

    # --- Check 4: exact-solution extension has zero forcing ----------------
    named_generators = [
        (configurations[0].display_name, configurations[0].generator),
        (configurations[2].display_name, configurations[2].generator),
        (configurations[4].display_name, configurations[4].generator),
    ]
    exact_rows = measure_exact_extension_forcing(named_generators, datum, identity_band)

    # --- Save all measured tables (npz + summary.yaml) BEFORE plotting -----
    labels = [
        configuration_label(configuration, int(row["defect_order"]))
        for configuration, row in zip(configurations, convergence_rows)
    ]
    np.savez(
        run_directory / "finite_difference_convergence.npz",
        time_step_values=time_step_values,
        relative_error_matrix=relative_error_matrix,
        fitted_slopes=np.array([row["fitted_slope"] for row in convergence_rows]),
        configuration_labels=np.array(labels),
        evaluation_time=np.float64(FINITE_DIFFERENCE_EVALUATION_TIME),
    )
    np.savez(
        run_directory / "split_identity_residuals.npz",
        residual_matrix=np.array(
            [row["relative_residual"] for row in identity_rows]
        ).reshape(len(configurations), len(IDENTITY_EVALUATION_TIMES)),
        evaluation_times=np.array(IDENTITY_EVALUATION_TIMES),
        configuration_labels=np.array(labels),
        tolerance=np.float64(SPLIT_IDENTITY_RELATIVE_TOLERANCE),
    )
    np.savez(run_directory / "gaussian_smoothed_put_fields.npz", **put_fields)

    overall_pass = (
        all(row["passed"] for row in identity_rows)
        and all(row["passed"] for row in convergence_rows)
        and put_row["passed"]
        and all(row["passed"] for row in exact_rows)
    )
    wall_clock_seconds = time.perf_counter() - start_wall_clock
    summary = {
        "parameters": dict(sorted(vars(args).items())),
        "constants": {
            "terminal_time": TERMINAL_TIME,
            "bernoulli_regularity_index": BERNOULLI_REGULARITY_INDEX,
            "identity_evaluation_times": list(IDENTITY_EVALUATION_TIMES),
            "finite_difference_evaluation_time": FINITE_DIFFERENCE_EVALUATION_TIME,
        },
        "split_identity": {
            "tolerance": SPLIT_IDENTITY_RELATIVE_TOLERANCE,
            "rows": identity_rows,
            "all_passed": bool(all(row["passed"] for row in identity_rows)),
        },
        "finite_difference_convergence": {
            "slope_interval": list(FINITE_DIFFERENCE_SLOPE_INTERVAL),
            "rows": convergence_rows,
            "all_passed": bool(all(row["passed"] for row in convergence_rows)),
        },
        "gaussian_smoothed_put": put_row,
        "exact_solution_extension": {
            "rows": exact_rows,
            "all_passed": bool(all(row["passed"] for row in exact_rows)),
        },
        "overall_pass": bool(overall_pass),
        "wall_clock_seconds": float(wall_clock_seconds),
    }
    summary_path = run_directory / "summary.yaml"
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        yaml.safe_dump(summary, summary_file, sort_keys=False)
    logger.info("Saved summary to %s", summary_path)

    # --- Figure (from the saved artefacts alone) ---------------------------
    figure_path = regenerate_figures(run_directory)
    logger.info("Saved figure to %s", figure_path)

    logger.info(
        "OVERALL: %s | wall clock %.2f s | run directory %s",
        "PASS" if overall_pass else "FAIL",
        wall_clock_seconds,
        run_directory,
    )
    logger.info("Follow the log in real time: tail -f %s", run_directory / "run.log")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
