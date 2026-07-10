r"""Measured forcing spectra and strip norms of terminal-data extensions (P3, P4).

Propositions under test.  This script measures two predictions of the
extension-comparison analysis (Proposition 6(iii) of the report draft and the
accompanying extension comparison) for the periodised Bernoulli datum of
regularity index :math:`\rho = 1` (exact coefficients
:math:`|c_k| = 2/(2\pi k)^2`, single break point at :math:`x^\star = 0`) under
the two second-order generators

* G1, advection–diffusion–reaction: :math:`a(k) = -0.7 k^2 + 1.3\,ik - 0.4`;
* G2, Black–Scholes in the log-price coordinate
  (:math:`\sigma = 0.5`, :math:`r = 0.03`):
  :math:`a(k) = -0.125 k^2 - 0.095\,ik - 0.03`,

with terminal time :math:`T = 1` and the extension catalogue: convex raw
(:math:`\hat h(k,t) = (t/T)\,c_k`, linear terminal-distance factor),
constant-in-time (:math:`\hat h = c_k`), split semigroup with subset
:math:`A = \{\partial_{xx}\}` (defect order :math:`m = 1`) and
:math:`A = \{\partial_{xx}, \partial_x\}` (:math:`m = 0`), graded Gaussian
matched (:math:`\nu_c = \nu`, which coincides with the split
:math:`\{\partial_{xx}\}`) and mismatched (:math:`\nu_c = \nu/2`), and the
exact solution (forcing identically zero).

Prediction P3 (forcing spectra at fixed time).  With
:math:`\widehat{Lh}(k,t) = \partial_t \hat h(k,t) + a(k)\hat h(k,t)`, the
near-terminal envelopes for :math:`\rho = 1`, :math:`p = 1` are

* constant-in-time and convex raw at :math:`t \to T`: flat (white),
  value :math:`a_0 |c_k| k^2 = a_0/(2\pi^2)` (convex raw scaled by
  :math:`t/T`);
* split of defect order :math:`m`:
  :math:`|b_m| |c_k| k^m \propto k^{m-\rho-1}` (decaying for
  :math:`m \le \rho`);
* graded Gaussian mismatched: flat with the power reduced by the factor
  :math:`((\nu-\nu_c)/\nu)^2 = 0.25` relative to constant-in-time, verified
  numerically at a large wavenumber at :math:`t = T`;
* exact solution: identically zero (asserted exactly, not to a tolerance).

At :math:`t < T` the split / graded moduli equal the near-terminal envelope
multiplied by the semigroup factor :math:`e^{-(T-t)\nu k^2}` (respectively
:math:`e^{-(T-t)\nu_c k^2}`); the departure of the measured curve below the
dashed envelope beyond :math:`k \sim ((T-t)\nu)^{-1/2}` is this factor, not a
violation of the envelope law.

Prediction P4 (total strip forcing versus the band edge).  The squared strip
norm :math:`2\pi \sum_{0<|k|\le K_{\max}} \int_0^T |\widehat{Lh}(k,t)|^2 dt`
(closed-form time integrals, no time quadrature) is tabulated for
:math:`K_{\max} \in \{2^{12}, 2^{16}, 2^{20}\}`.  The specification predicts
unbounded (linear) growth for constant-in-time, convex raw and the mismatched
graded extension, and convergence for the splits with :math:`m \le \rho` and
the matched graded extension.  The measured classification is reported next
to the predicted one; a disagreement is recorded with its analytic account
rather than suppressed (for the mismatched graded extension the closed-form
per-wavenumber time integral has the tail
:math:`((\nu-\nu_c)^2/(2\nu_c))\,|c_k|^2 k^2 = O(k^{-2})` as
:math:`|k| \to \infty`, because the flat spectral component of the forcing at
:math:`t = T` is confined to a temporal boundary layer of width
:math:`(2\nu_c k^2)^{-1}`; the wavenumber sum therefore converges even though
the fixed-time norm at :math:`t = T` grows without bound).

Measurement policy.  Every quantity is evaluated from the exact analytic
Fourier coefficients over integer wavenumber arrays (vectorised
``complex128``); the FFT of sampled values is never used.  All artefacts
(``forcing_spectra_measurements.npz`` with every measured curve and every
predicted envelope, plus ``summary.yaml``) are written before any figure is
drawn, and ``--replot RUN_DIR`` rebuilds every figure from the saved
artefacts alone, without recomputation.
"""
from __future__ import annotations

import argparse
import logging
import math
import shlex
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _figure_layout import finalize_figure  # noqa: E402
from learning_option_pricing.pde import (  # noqa: E402
    ConstantCoefficientGenerator,
    ConstantInTimeExtension,
    ConvexRawExtension,
    ExactSolutionExtension,
    GradedGaussianExtension,
    PeriodisedBernoulliDatum,
    SplitSemigroupExtension,
    TerminalDataExtension,
    advection_diffusion_reaction,
    black_scholes_log_price,
    symmetric_wavenumber_band,
    total_strip_forcing_squared,
)
from learning_option_pricing.utils.run_context import (  # noqa: E402
    collect_run_metadata,
    configure_cli_script_logging,
    find_repo_root,
    get_git_metadata,
    init_logging,
    log_parsed_args,
    script_data_dir,
    utc_timestamp,
    write_command_txt,
    write_json,
)

TERMINAL_TIME = 1.0
REGULARITY_INDEX = 1  # rho = 1: first-derivative discontinuity of the datum

MEASUREMENTS_FILENAME = "forcing_spectra_measurements.npz"
SUMMARY_FILENAME = "summary.yaml"

# Smoke-test guard thresholds: values below these require --debug, so an
# exploratory run cannot land in the real-run namespace (mechanical
# enforcement, per the repository convention).
SMOKE_TEST_SPECTRUM_MAX_WAVENUMBER_THRESHOLD = 512
SMOKE_TEST_MINIMUM_STRIP_BAND_EDGE_THRESHOLD = 2**12
SMOKE_TEST_RATIO_WAVENUMBER_THRESHOLD = 2**16

GENERATOR_ORDER = ("advection_diffusion_reaction", "black_scholes_log_price")
GENERATOR_DISPLAY_LABELS = {
    "advection_diffusion_reaction": (
        r"G1: advection–diffusion–reaction, $a(k)=-0.7k^2+1.3\,ik-0.4$"
    ),
    "black_scholes_log_price": (
        r"G2: Black–Scholes log-price, $a(k)=-0.125k^2-0.095\,ik-0.03$"
    ),
}

EXTENSION_ORDER = (
    "convex_raw",
    "constant_in_time",
    "split_diffusion",
    "split_diffusion_advection",
    "graded_gaussian_matched",
    "graded_gaussian_mismatched",
    "exact_solution",
)
EXTENSION_DISPLAY_LABELS = {
    "convex_raw": r"Convex raw $\hat h=(t/T)\,c_k$",
    "constant_in_time": r"Constant-in-time $\hat h=c_k$",
    "split_diffusion": r"Split $A=\{\partial_{xx}\}$ ($m=1$)",
    "split_diffusion_advection": r"Split $A=\{\partial_{xx},\partial_x\}$ ($m=0$)",
    "graded_gaussian_matched": (
        r"Graded Gaussian $\nu_c=\nu$ (coincides with split $\{\partial_{xx}\}$)"
    ),
    "graded_gaussian_mismatched": r"Graded Gaussian $\nu_c=\nu/2$",
    "exact_solution": r"Exact solution ($\widehat{Lh}\equiv 0$)",
}
EXTENSION_COLOURS = {
    "convex_raw": "#d1495b",
    "constant_in_time": "#e08a00",
    "split_diffusion": "#1b6ca8",
    "split_diffusion_advection": "#66a182",
    "graded_gaussian_matched": "#7b5aa6",
    "graded_gaussian_mismatched": "#c05aa0",
    "exact_solution": "#555555",
}

TIME_TAGS = ("t_initial", "t_near_terminal", "t_terminal")

# Predicted strip-norm behaviour per extension, as stated in the
# specification of prediction P4 (recorded verbatim; the measured
# classification is reported next to it and disagreements are flagged).
PREDICTED_STRIP_CLASSIFICATION = {
    "convex_raw": "divergent_linear_in_band_edge",
    "constant_in_time": "divergent_linear_in_band_edge",
    "split_diffusion": "convergent",
    "split_diffusion_advection": "convergent",
    "graded_gaussian_matched": "convergent",
    "graded_gaussian_mismatched": "divergent_linear_in_band_edge",
    "exact_solution": "identically_zero",
}

MISMATCHED_GRADED_DISAGREEMENT_NOTE = (
    "The specification of P4 predicts unbounded linear growth for the "
    "mismatched graded extension; the measurement shows convergence. "
    "Analytic account: the closed-form per-wavenumber time integral "
    "|a(k) + nu_c k^2|^2 |c_k|^2 phi(-2 nu_c k^2) has the tail "
    "((nu - nu_c)^2 / (2 nu_c)) |c_k|^2 k^2 = O(k^-2) as |k| -> infinity "
    "(rho = 1), because the flat spectral component of the forcing at t = T "
    "is confined to a temporal boundary layer of width (2 nu_c k^2)^-1; the "
    "wavenumber sum therefore converges, even though the fixed-time "
    "L^2(0, 2*pi) norm of the forcing at t = T grows without bound "
    "(flat spectrum, prediction P3)."
)


# ---------------------------------------------------------------------------
# Extension catalogue and predicted envelopes
# ---------------------------------------------------------------------------


def build_generator_catalogue() -> dict[str, ConstantCoefficientGenerator]:
    """Return the two named generators of the study, keyed by their names."""
    return {
        "advection_diffusion_reaction": advection_diffusion_reaction(),
        "black_scholes_log_price": black_scholes_log_price(),
    }


def build_extension_catalogue(
    datum: PeriodisedBernoulliDatum,
    generator: ConstantCoefficientGenerator,
) -> dict[str, TerminalDataExtension]:
    """Instantiate the seven extensions of the comparison for one generator.

    Args:
        datum: The terminal datum (periodised Bernoulli, ``rho = 1``).
        generator: A second-order constant-coefficient generator whose
            order-2 coefficient is the diffusivity ``nu``.

    Returns:
        Mapping from extension key (in :data:`EXTENSION_ORDER`) to the
        extension instance.
    """
    diffusivity = generator.coefficients[2]
    return {
        "convex_raw": ConvexRawExtension(datum, generator, TERMINAL_TIME),
        "constant_in_time": ConstantInTimeExtension(datum, generator, TERMINAL_TIME),
        "split_diffusion": SplitSemigroupExtension(
            datum, generator, subset_orders=[2], terminal_time=TERMINAL_TIME
        ),
        "split_diffusion_advection": SplitSemigroupExtension(
            datum, generator, subset_orders=[2, 1], terminal_time=TERMINAL_TIME
        ),
        "graded_gaussian_matched": GradedGaussianExtension(
            datum,
            generator,
            comparison_diffusivity=diffusivity,
            terminal_time=TERMINAL_TIME,
        ),
        "graded_gaussian_mismatched": GradedGaussianExtension(
            datum,
            generator,
            comparison_diffusivity=diffusivity / 2.0,
            terminal_time=TERMINAL_TIME,
        ),
        "exact_solution": ExactSolutionExtension(datum, generator, TERMINAL_TIME),
    }


def defect_polynomial_coefficients(
    extension: TerminalDataExtension,
) -> dict[int, float] | None:
    r"""Coefficients (in the :math:`(ik)^j` basis) of the extension's defect symbol.

    The defect symbol is the multiplier :math:`b(k)` with
    :math:`\widehat{Lh}(k, t) = b(k)\,\hat h(k, t)` at :math:`t = T` (where
    the extension coefficient equals :math:`c_k`):

    * constant-in-time: :math:`b = a` (the full symbol);
    * convex raw: :math:`b = a` up to the bounded term :math:`1/T`, which
      does not alter the leading order (the time scaling :math:`t/T` is
      applied by the caller);
    * split semigroup: the defect symbol of the split;
    * graded Gaussian: :math:`a(k) + \nu_c k^2`, whose order-2 coefficient in
      the :math:`(ik)^2 = -k^2` basis is :math:`a_2 - \nu_c`;
    * exact solution: ``None`` (the forcing vanishes identically).

    Returns:
        Mapping from differential order to the real coefficient, or ``None``
        for the exact solution.
    """
    generator = extension.generator
    if isinstance(extension, ExactSolutionExtension):
        return None
    if isinstance(extension, SplitSemigroupExtension):
        return {
            order: generator.coefficients[order]
            for order in extension.generator_split.defect_orders
        }
    if isinstance(extension, GradedGaussianExtension):
        defect_coefficients = dict(generator.coefficients)
        defect_coefficients[2] = (
            defect_coefficients.get(2, 0.0) - extension.comparison_diffusivity
        )
        return defect_coefficients
    # Constant-in-time and convex raw: the defect is the full symbol.
    return dict(generator.coefficients)


def leading_defect_term(
    defect_coefficients: dict[int, float],
) -> tuple[int, float] | None:
    """Leading (highest-order, nonzero) term of a defect polynomial.

    Returns:
        Pair ``(leading_order, absolute_coefficient)``, or ``None`` when
        every coefficient vanishes (empty defect).
    """
    nonzero_terms = {
        order: value for order, value in defect_coefficients.items() if value != 0.0
    }
    if not nonzero_terms:
        return None
    leading_order = max(nonzero_terms)
    return leading_order, abs(nonzero_terms[leading_order])


def predicted_envelope_modulus(
    extension_key: str,
    extension: TerminalDataExtension,
    datum: PeriodisedBernoulliDatum,
    positive_wavenumbers: np.ndarray,
    time_value: float,
) -> np.ndarray | None:
    r"""Predicted near-terminal envelope of :math:`|\widehat{Lh}(k, t)|`.

    With the exact Bernoulli modulus
    :math:`|c_k| = (\rho+1)!/(2\pi)^{\rho+1}\, k^{-\rho-1}` and a defect
    symbol of leading term :math:`|b_m| k^m`, the envelope is

    .. math::

        |b_m|\, \frac{(\rho+1)!}{(2\pi)^{\rho+1}}\, k^{m-\rho-1},

    scaled by :math:`t/T` for the convex raw extension (whose forcing at
    :math:`t = 0` is exactly :math:`|c_k|/T`, the special case returned when
    ``time_value == 0.0``).

    Returns:
        ``float64`` envelope array over ``positive_wavenumbers``, or ``None``
        when no envelope is predicted (exact solution).
    """
    regularity_index = datum.regularity_index
    envelope_prefactor = math.factorial(regularity_index + 1) / (
        2.0 * math.pi
    ) ** (regularity_index + 1)
    wavenumber_array = np.asarray(positive_wavenumbers, dtype=np.float64)
    if extension_key == "convex_raw" and time_value == 0.0:
        # At t = 0 the convex-raw forcing is c_k / T exactly.
        return (
            envelope_prefactor
            / extension.terminal_time
            * wavenumber_array ** (-(regularity_index + 1))
        )
    defect_coefficients = defect_polynomial_coefficients(extension)
    if defect_coefficients is None:
        return None
    leading_term = leading_defect_term(defect_coefficients)
    if leading_term is None:
        return None
    leading_order, leading_coefficient = leading_term
    time_scale = (
        time_value / extension.terminal_time if extension_key == "convex_raw" else 1.0
    )
    return (
        time_scale
        * leading_coefficient
        * envelope_prefactor
        * wavenumber_array ** (leading_order - regularity_index - 1)
    )


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def classify_measured_growth(
    strip_values: list[float], band_edges: list[int]
) -> tuple[str, list[float]]:
    """Classify the measured strip-norm growth across the band edges.

    The classification is empirical, from the last measured growth factor
    ``strip(K_last) / strip(K_previous)`` compared against the band-edge
    factor ``K_last / K_previous``:

    * every value zero -> ``identically_zero``;
    * growth factor at least half the band-edge factor ->
      ``divergent_linear_in_band_edge``;
    * growth factor at most 1.05 -> ``convergent``;
    * otherwise -> ``intermediate``.

    Returns:
        Pair ``(classification, growth_factors)`` where ``growth_factors``
        lists the successive ratios (``nan`` where the denominator is zero).
    """
    growth_factors = [
        strip_values[index] / strip_values[index - 1]
        if strip_values[index - 1] > 0.0
        else float("nan")
        for index in range(1, len(strip_values))
    ]
    if all(value == 0.0 for value in strip_values):
        return "identically_zero", growth_factors
    last_growth_factor = growth_factors[-1]
    band_edge_factor = band_edges[-1] / band_edges[-2]
    if math.isnan(last_growth_factor):
        return "intermediate", growth_factors
    if last_growth_factor >= 0.5 * band_edge_factor:
        return "divergent_linear_in_band_edge", growth_factors
    if last_growth_factor <= 1.05:
        return "convergent", growth_factors
    return "intermediate", growth_factors


def run_measurements(arguments: argparse.Namespace, run_directory: Path) -> None:
    """Measure the spectra (P3) and the strip-norm table (P4); save the artefacts.

    Writes ``forcing_spectra_measurements.npz`` (every measured curve and
    every predicted envelope) and ``summary.yaml`` into ``run_directory``
    before any figure is drawn.
    """
    logger = logging.getLogger(Path(__file__).stem)
    datum = PeriodisedBernoulliDatum(regularity_index=REGULARITY_INDEX)
    generators = build_generator_catalogue()

    near_terminal_time = arguments.near_terminal_fraction * TERMINAL_TIME
    time_points = {
        "t_initial": 0.0,
        "t_near_terminal": near_terminal_time,
        "t_terminal": TERMINAL_TIME,
    }
    positive_wavenumbers = np.arange(
        1, arguments.spectrum_max_wavenumber + 1, dtype=np.int64
    )
    # The moduli are even in k for real-coefficient generators
    # (a(-k) is the complex conjugate of a(k) and |c_{-k}| = |c_k|), so the
    # spectra are measured over the positive wavenumbers only.
    strip_band_edges = list(arguments.strip_band_edges)

    saved_arrays: dict[str, np.ndarray] = {
        "spectrum_wavenumbers": positive_wavenumbers,
        "strip_band_edges": np.asarray(strip_band_edges, dtype=np.int64),
    }
    summary: dict = {
        "parameters": {
            "terminal_time": TERMINAL_TIME,
            "regularity_index": REGULARITY_INDEX,
            "spectrum_max_wavenumber": int(arguments.spectrum_max_wavenumber),
            "strip_band_edges": [int(edge) for edge in strip_band_edges],
            "ratio_wavenumber": int(arguments.ratio_wavenumber),
            "near_terminal_fraction": float(arguments.near_terminal_fraction),
            "seed": int(arguments.seed),
            "debug": bool(arguments.debug),
        },
        "time_points": {tag: float(value) for tag, value in time_points.items()},
        "generator_order": list(GENERATOR_ORDER),
        "extension_order": list(EXTENSION_ORDER),
        "datum": {
            "class": "PeriodisedBernoulliDatum",
            "regularity_index": REGULARITY_INDEX,
            "coefficient_modulus": "|c_k| = 2 / (2*pi*k)^2 (exact)",
            "jump_of_rho_derivative": float(datum.jump_of_rho_derivative),
        },
        "generators": {},
    }

    for generator_key in GENERATOR_ORDER:
        generator = generators[generator_key]
        diffusivity = generator.coefficients[2]
        extensions = build_extension_catalogue(datum, generator)
        generator_summary: dict = {
            "name": generator.name,
            "coefficients": {
                int(order): float(value)
                for order, value in generator.coefficients.items()
            },
            "half_order": int(generator.half_order),
            "principal_constant": float(generator.principal_constant),
            "diffusivity": float(diffusivity),
            "comparison_diffusivity_matched": float(diffusivity),
            "comparison_diffusivity_mismatched": float(diffusivity / 2.0),
        }

        # ---- P3: forcing spectra at the three time points --------------
        for extension_key in EXTENSION_ORDER:
            extension = extensions[extension_key]
            for time_tag in TIME_TAGS:
                forcing_modulus = np.abs(
                    extension.forcing_coefficient(
                        positive_wavenumbers, time_points[time_tag]
                    )
                )
                saved_arrays[
                    f"forcing_modulus__{generator_key}__{extension_key}__{time_tag}"
                ] = forcing_modulus
            # Predicted envelopes: at t = 0 only for the extensions without a
            # semigroup factor (constant-in-time; convex raw, whose t = 0
            # modulus is |c_k|/T exactly); near the terminal time for every
            # extension with a nonempty defect.
            for time_tag in ("t_initial", "t_near_terminal"):
                if time_tag == "t_initial" and extension_key not in (
                    "constant_in_time",
                    "convex_raw",
                ):
                    continue
                envelope = predicted_envelope_modulus(
                    extension_key,
                    extension,
                    datum,
                    positive_wavenumbers,
                    time_points[time_tag],
                )
                if envelope is not None:
                    saved_arrays[
                        f"predicted_envelope__{generator_key}__{extension_key}__{time_tag}"
                    ] = envelope

        # ---- Exact-solution assertion (identically zero, not tolerance) --
        exact_maximum_modulus = max(
            float(
                np.max(
                    saved_arrays[
                        f"forcing_modulus__{generator_key}__exact_solution__{time_tag}"
                    ]
                )
            )
            for time_tag in TIME_TAGS
        )
        if exact_maximum_modulus != 0.0:
            raise AssertionError(
                "the exact-solution extension must have identically zero "
                f"forcing, but max |forcing| = {exact_maximum_modulus!r} for "
                f"generator '{generator_key}'"
            )
        generator_summary["exact_extension_max_forcing_modulus"] = 0.0
        logger.info(
            "[%s] exact-solution extension: max |forcing| over the band and "
            "the three time points equals 0.0 exactly (assertion passed).",
            generator_key,
        )

        # ---- Mismatch power ratio at a large wavenumber, t = T ----------
        ratio_band = np.asarray([arguments.ratio_wavenumber], dtype=np.int64)
        mismatched_power = float(
            np.abs(
                extensions["graded_gaussian_mismatched"].forcing_coefficient(
                    ratio_band, TERMINAL_TIME
                )
            )[0]
            ** 2
        )
        constant_power = float(
            np.abs(
                extensions["constant_in_time"].forcing_coefficient(
                    ratio_band, TERMINAL_TIME
                )
            )[0]
            ** 2
        )
        measured_ratio = mismatched_power / constant_power
        predicted_ratio = (
            (diffusivity - diffusivity / 2.0) / diffusivity
        ) ** 2
        generator_summary["mismatch_power_ratio"] = {
            "predicted": float(predicted_ratio),
            "measured": float(measured_ratio),
            "measured_at_wavenumber": int(arguments.ratio_wavenumber),
            "measured_at_time": float(TERMINAL_TIME),
            "relative_deviation": float(
                abs(measured_ratio - predicted_ratio) / predicted_ratio
            ),
        }
        logger.info(
            "[%s] mismatch power ratio at k = %d, t = T: measured %.12f, "
            "predicted ((nu - nu_c)/nu)^2 = %.2f, relative deviation %.3e.",
            generator_key,
            arguments.ratio_wavenumber,
            measured_ratio,
            predicted_ratio,
            abs(measured_ratio - predicted_ratio) / predicted_ratio,
        )

        # ---- Coincidence of the matched graded and the diffusion split ---
        matched_versus_split_spectra = max(
            float(
                np.max(
                    np.abs(
                        saved_arrays[
                            f"forcing_modulus__{generator_key}__graded_gaussian_matched__{time_tag}"
                        ]
                        - saved_arrays[
                            f"forcing_modulus__{generator_key}__split_diffusion__{time_tag}"
                        ]
                    )
                )
            )
            for time_tag in TIME_TAGS
        )
        generator_summary["graded_matched_equals_split_diffusion"] = {
            "max_absolute_spectra_difference": matched_versus_split_spectra,
        }

        # ---- P4: total strip forcing versus the band edge ----------------
        strip_table: dict[str, dict] = {}
        for extension_key in EXTENSION_ORDER:
            extension = extensions[extension_key]
            strip_values = [
                total_strip_forcing_squared(
                    extension, symmetric_wavenumber_band(band_edge)
                )
                for band_edge in strip_band_edges
            ]
            saved_arrays[f"strip_forcing__{generator_key}__{extension_key}"] = (
                np.asarray(strip_values, dtype=np.float64)
            )
            classification, growth_factors = classify_measured_growth(
                strip_values, strip_band_edges
            )
            predicted_classification = PREDICTED_STRIP_CLASSIFICATION[extension_key]
            entry: dict = {
                "values": [float(value) for value in strip_values],
                "growth_factors": [float(value) for value in growth_factors],
                "measured_classification": classification,
                "predicted_classification": predicted_classification,
                "agreement": classification == predicted_classification,
            }
            if (
                extension_key == "graded_gaussian_mismatched"
                and classification != predicted_classification
            ):
                entry["note"] = MISMATCHED_GRADED_DISAGREEMENT_NOTE
            strip_table[extension_key] = entry
        generator_summary["strip_forcing"] = {
            "band_edges": [int(edge) for edge in strip_band_edges],
            "per_extension": strip_table,
        }
        matched_strip = saved_arrays[
            f"strip_forcing__{generator_key}__graded_gaussian_matched"
        ]
        split_strip = saved_arrays[
            f"strip_forcing__{generator_key}__split_diffusion"
        ]
        generator_summary["graded_matched_equals_split_diffusion"][
            "max_relative_strip_difference"
        ] = float(np.max(np.abs(matched_strip - split_strip) / split_strip))

        # ---- Log the P4 table --------------------------------------------
        header = (
            f"{'Extension':32s}"
            + "".join(f"{'K=' + str(edge):>16s}" for edge in strip_band_edges)
            + f"{'Last growth':>14s}  Classification (measured | predicted)"
        )
        logger.info("[%s] total strip forcing ||Lh||^2 versus K_max:", generator_key)
        logger.info("%s", header)
        for extension_key in EXTENSION_ORDER:
            entry = strip_table[extension_key]
            last_growth = (
                f"{entry['growth_factors'][-1]:.6f}"
                if entry["growth_factors"]
                and not math.isnan(entry["growth_factors"][-1])
                else "-"
            )
            logger.info(
                "%s",
                f"{extension_key:32s}"
                + "".join(f"{value:>16.6e}" for value in entry["values"])
                + f"{last_growth:>14s}  {entry['measured_classification']}"
                + f" | {entry['predicted_classification']}"
                + ("" if entry["agreement"] else "  [DISAGREEMENT]"),
            )

        summary["generators"][generator_key] = generator_summary

    # ---- Save artefacts before plotting ------------------------------
    measurements_path = run_directory / MEASUREMENTS_FILENAME
    np.savez_compressed(measurements_path, **saved_arrays)
    logger.info("Saved measured curves and envelopes to %s", measurements_path)
    summary_path = run_directory / SUMMARY_FILENAME
    with open(summary_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)
    logger.info("Saved summary to %s", summary_path)


# ---------------------------------------------------------------------------
# Figures (built from the saved artefacts only)
# ---------------------------------------------------------------------------


def build_all_figures(run_directory: Path) -> None:
    """Rebuild every figure from the saved ``.npz`` and ``summary.yaml`` alone.

    This function performs no measurement: it is shared verbatim by the
    in-run plotting path and by ``--replot``.
    """
    logger = logging.getLogger(Path(__file__).stem)
    measurements = np.load(run_directory / MEASUREMENTS_FILENAME)
    with open(run_directory / SUMMARY_FILENAME, "r", encoding="utf-8") as handle:
        summary = yaml.safe_load(handle)

    positive_wavenumbers = measurements["spectrum_wavenumbers"]
    strip_band_edges = measurements["strip_band_edges"]
    time_points = summary["time_points"]

    # ---- Spectra figures (one per generator) --------------------------
    panel_time_tags = ("t_initial", "t_near_terminal")
    panel_titles = {
        "t_initial": r"Initial time $t = 0$",
        "t_near_terminal": (
            rf"Near-terminal time"
            rf" $t = {summary['parameters']['near_terminal_fraction']:g}\,T$"
        ),
    }
    for generator_key in summary["generator_order"]:
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharey=True)
        for axis, time_tag in zip(axes, panel_time_tags):
            for extension_key in summary["extension_order"]:
                modulus_key = (
                    f"forcing_modulus__{generator_key}__{extension_key}__{time_tag}"
                )
                forcing_modulus = measurements[modulus_key]
                # Figure-only truncation: entries below 1e-16 lie under the
                # axis floor of 1e-14 and only inflate the autoscaled range
                # (the semigroup factors decay to the underflow threshold).
                # The saved .npz retains the full curve; no measured value
                # is altered.
                positive_mask = forcing_modulus > 1.0e-16
                if extension_key == "exact_solution" or not np.any(positive_mask):
                    # Identically zero: not representable on logarithmic axes
                    # (stated in the formula box).
                    continue
                line_width = (
                    2.6 if extension_key == "split_diffusion" else 1.4
                )
                axis.loglog(
                    positive_wavenumbers[positive_mask],
                    forcing_modulus[positive_mask],
                    "-",
                    color=EXTENSION_COLOURS[extension_key],
                    lw=line_width,
                )
                envelope_key = (
                    f"predicted_envelope__{generator_key}__{extension_key}__{time_tag}"
                )
                if envelope_key in measurements:
                    axis.loglog(
                        positive_wavenumbers,
                        measurements[envelope_key],
                        "--",
                        color=EXTENSION_COLOURS[extension_key],
                        lw=1.0,
                        alpha=0.85,
                    )
            axis.set_xlabel(r"Wavenumber $k$")
            axis.set_title(panel_titles[time_tag], fontsize=10)
            axis.grid(True, which="both", alpha=0.3)
            axis.set_ylim(bottom=1e-14)
        axes[0].set_ylabel(r"Forcing modulus $|\widehat{Lh}(k,t)|$")
        legend_handles = [
            Line2D(
                [],
                [],
                color=EXTENSION_COLOURS[extension_key],
                ls="-",
                lw=2.6 if extension_key == "split_diffusion" else 1.4,
                label=EXTENSION_DISPLAY_LABELS[extension_key],
            )
            for extension_key in summary["extension_order"]
        ]
        legend_handles.append(
            Line2D(
                [],
                [],
                color="black",
                ls="--",
                lw=1.0,
                label=r"Predicted envelope (dashed, valid as $t\to T$)",
            )
        )
        legend = fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=2,
            fontsize=8,
            frameon=True,
            bbox_to_anchor=(0.5, 0.14),
        )
        fig.suptitle(
            "Forcing spectra per extension — "
            + GENERATOR_DISPLAY_LABELS[generator_key],
            fontsize=11,
        )
        fig.tight_layout(rect=[0, 0.34, 1, 0.94])
        finalize_figure(
            fig,
            run_directory / f"forcing_spectra__{generator_key}.png",
            legends=[legend],
            axes=list(axes),
            formula=(
                r"$\widehat{Lh}(k,t)=\partial_t\hat h(k,t)+a(k)\,\hat h(k,t)$;"
                r" datum $|c_k|=2/(2\pi k)^2$ ($\rho=1$). Envelopes at $t\to T$:"
                r" constant-in-time $a_0|c_k|k^2$ (flat); convex raw"
                r" $(t/T)\,a_0|c_k|k^2$ (equal to $|c_k|/T$ at $t=0$);"
                "\n"
                r"split of defect order $m$: $|b_m||c_k|k^{m}$; graded"
                r" mismatched $(\nu-\nu_c)|c_k|k^2$ (flat, power ratio"
                r" $((\nu-\nu_c)/\nu)^2=0.25$). At $t<T$ the split / graded"
                r" moduli equal the envelope multiplied by $e^{-(T-t)\nu k^2}$"
                r" (resp. $e^{-(T-t)\nu_c k^2}$):"
                "\n"
                r"the departure beyond $k\sim((T-t)\nu)^{-1/2}$ is this factor,"
                r" not a violation. Exact solution: $\widehat{Lh}\equiv 0$,"
                r" omitted from the logarithmic axes."
            ),
            formula_fontsize=7,
        )
        logger.info(
            "Wrote %s", run_directory / f"forcing_spectra__{generator_key}.png"
        )

    # ---- Strip-forcing-versus-band-edge figure ------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, generator_key in zip(axes, summary["generator_order"]):
        for extension_key in summary["extension_order"]:
            strip_values = measurements[
                f"strip_forcing__{generator_key}__{extension_key}"
            ]
            if np.all(strip_values == 0.0):
                continue  # exact solution: identically zero, stated in the box
            # The split {d_xx} curve is widened so that it remains visible
            # under the exactly coincident matched graded Gaussian curve.
            line_width = 3.0 if extension_key == "split_diffusion" else 1.5
            axis.loglog(
                strip_band_edges,
                strip_values,
                "-o",
                color=EXTENSION_COLOURS[extension_key],
                lw=line_width,
                ms=4,
            )
        constant_values = measurements[
            f"strip_forcing__{generator_key}__constant_in_time"
        ]
        linear_reference = constant_values[0] * (
            strip_band_edges / strip_band_edges[0]
        )
        axis.loglog(
            strip_band_edges,
            linear_reference,
            "--",
            color="black",
            lw=1.1,
        )
        axis.set_xlabel(r"Band edge $K_{\max}$")
        axis.set_title(GENERATOR_DISPLAY_LABELS[generator_key], fontsize=9)
        axis.grid(True, which="both", alpha=0.3)
    # The full norm, with its function spaces, is stated in the formula box;
    # a long mathematical y-label would be clipped at the canvas top.
    axes[0].set_ylabel(r"Total strip forcing $\|Lh\|^2$")
    legend_handles = [
        Line2D(
            [],
            [],
            color=EXTENSION_COLOURS[extension_key],
            ls="-",
            marker="o",
            ms=4,
            lw=3.0 if extension_key == "split_diffusion" else 1.5,
            label=EXTENSION_DISPLAY_LABELS[extension_key],
        )
        for extension_key in summary["extension_order"]
        if extension_key != "exact_solution"
    ]
    legend_handles.append(
        Line2D(
            [],
            [],
            color="black",
            ls="--",
            lw=1.1,
            label=r"Linear growth $\propto K_{\max}$ (predicted, anchored to"
            r" constant-in-time)",
        )
    )
    legend = fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        fontsize=8,
        frameon=True,
        bbox_to_anchor=(0.5, 0.19),
    )
    fig.suptitle(
        "Total strip forcing versus the band edge (prediction P4)", fontsize=11
    )
    fig.tight_layout(rect=[0, 0.40, 1, 0.94])
    finalize_figure(
        fig,
        run_directory / "strip_forcing_vs_band_edge.png",
        legends=[legend],
        axes=list(axes),
        formula=(
            r"$\|Lh\|^2_{L^2((0,T);L^2(0,2\pi))}=2\pi\sum_{0<|k|\leq K_{\max}}"
            r"\int_0^T|\widehat{Lh}(k,t)|^2\,dt$ (closed-form time integrals,"
            r" no quadrature)."
            "\n"
            r"Measured: linear growth for constant-in-time and convex raw;"
            r" convergent for the splits ($m\leq\rho$), the matched graded"
            r" ($\nu_c=\nu$) and the mismatched graded extension"
            "\n"
            r"($\int_0^T|\widehat{Lh}(k,t)|^2\,dt=O(k^{-2})$ as"
            r" $|k|\to\infty$; see summary.yaml). Exact solution: identically"
            r" zero, not representable on the logarithmic axes."
        ),
        formula_fontsize=7,
    )
    logger.info("Wrote %s", run_directory / "strip_forcing_vs_band_edge.png")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def parse_arguments(argv=None) -> argparse.Namespace:
    """Parse the command line, enforcing the smoke-test guard."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--spectrum-max-wavenumber",
        type=int,
        default=4096,
        help="Largest positive wavenumber of the plotted spectra (default 4096).",
    )
    parser.add_argument(
        "--strip-band-edges",
        type=int,
        nargs="+",
        default=[2**12, 2**16, 2**20],
        help="Band edges K_max of the strip-forcing table (default 4096 65536 1048576).",
    )
    parser.add_argument(
        "--ratio-wavenumber",
        type=int,
        default=2**20,
        help="Large wavenumber at which the mismatch power ratio is measured "
        "at t = T (default 1048576).",
    )
    parser.add_argument(
        "--near-terminal-fraction",
        type=float,
        default=0.99,
        help="Fraction f of the terminal time for the near-terminal spectra "
        "panel, t = f*T (default 0.99).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Master seed, logged for the run-log contract; the measurement "
        "is deterministic and instantiates no random number generator.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Prepend '_debug_' to the output folder name (exploratory runs).",
    )
    parser.add_argument(
        "--replot",
        type=Path,
        default=None,
        metavar="RUN_DIR",
        help="Rebuild every figure from the saved artefacts of an existing "
        "run directory, without recomputation.",
    )
    arguments = parser.parse_args(argv)

    if arguments.replot is None:
        if len(arguments.strip_band_edges) < 2:
            parser.error("--strip-band-edges requires at least two band edges")
        if any(
            later <= earlier
            for earlier, later in zip(
                arguments.strip_band_edges, arguments.strip_band_edges[1:]
            )
        ):
            parser.error(
                "--strip-band-edges must be strictly increasing, received "
                f"{arguments.strip_band_edges}"
            )
        if not 0.0 < arguments.near_terminal_fraction < 1.0:
            parser.error(
                "--near-terminal-fraction must belong to (0, 1), received "
                f"{arguments.near_terminal_fraction}"
            )
        smoke_sized = (
            arguments.spectrum_max_wavenumber
            < SMOKE_TEST_SPECTRUM_MAX_WAVENUMBER_THRESHOLD
            or min(arguments.strip_band_edges)
            < SMOKE_TEST_MINIMUM_STRIP_BAND_EDGE_THRESHOLD
            or arguments.ratio_wavenumber < SMOKE_TEST_RATIO_WAVENUMBER_THRESHOLD
        )
        if smoke_sized and not arguments.debug:
            parser.error(
                "smoke-test guard: a spectrum band below "
                f"{SMOKE_TEST_SPECTRUM_MAX_WAVENUMBER_THRESHOLD}, a strip band "
                f"edge below {SMOKE_TEST_MINIMUM_STRIP_BAND_EDGE_THRESHOLD}, or "
                f"a ratio wavenumber below "
                f"{SMOKE_TEST_RATIO_WAVENUMBER_THRESHOLD} requires --debug so "
                "the exploratory run is flagged in the output namespace"
            )
    return arguments


def main(argv=None) -> int:
    """Entry point: measure and plot, or replot from saved artefacts."""
    arguments = parse_arguments(argv)
    script_stem = Path(__file__).stem

    if arguments.replot is not None:
        configure_cli_script_logging(verbose=False)
        logger = logging.getLogger(script_stem)
        run_directory = arguments.replot.resolve()
        if not (run_directory / MEASUREMENTS_FILENAME).exists():
            raise FileNotFoundError(
                f"--replot: {run_directory / MEASUREMENTS_FILENAME} not found"
            )
        logger.info("Replot mode: rebuilding figures from %s", run_directory)
        build_all_figures(run_directory)
        return 0

    start_time = time.perf_counter()
    debug_prefix = "_debug_" if arguments.debug else ""
    config_tag = (
        f"rho{REGULARITY_INDEX}"
        f"_specK{arguments.spectrum_max_wavenumber}"
        f"_stripK{max(arguments.strip_band_edges)}"
    )
    run_directory = (
        script_data_dir(__file__) / f"{debug_prefix}{utc_timestamp()}_{config_tag}"
    )
    run_directory.mkdir(parents=True, exist_ok=False)

    init_logging(run_dir=run_directory)
    logger = logging.getLogger(script_stem)
    logger.info("Command line: %s", " ".join(shlex.quote(token) for token in sys.argv))
    logger.info(
        "Runtime: Python %s | numpy %s | matplotlib %s | PyYAML %s",
        sys.version.split()[0],
        np.__version__,
        matplotlib.__version__,
        yaml.__version__,
    )
    repository_root = find_repo_root(Path(__file__))
    git_metadata = get_git_metadata(repository_root)
    logger.info(
        "Git: commit %s | branch %s | dirty %s",
        git_metadata.get("commit"),
        git_metadata.get("branch"),
        git_metadata.get("dirty"),
    )
    log_parsed_args(logger, arguments)
    logger.info(
        "Master seed: %d (deterministic measurement — no random number "
        "generator is instantiated; the seed is recorded for the run-log "
        "contract).",
        arguments.seed,
    )
    logger.info("Run directory: %s", run_directory)
    write_json(
        run_directory / "run_metadata.json",
        collect_run_metadata(
            run_dir=run_directory,
            repo_root=repository_root,
            script_name=script_stem,
            command=sys.argv,
            params={key: str(value) for key, value in vars(arguments).items()},
        ),
    )
    write_command_txt(run_directory / "command.txt", list(sys.argv))

    run_measurements(arguments, run_directory)
    build_all_figures(run_directory)

    elapsed_seconds = time.perf_counter() - start_time
    logger.info("Total wall-clock time: %.2f s", elapsed_seconds)
    logger.info(
        "Follow the log in real time with: tail -f %s", run_directory / "run.log"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
