r"""Spectral concentration of the extension forcing below a reachable cutoff.

Hypothesis under test (reviewer's spectral-gap hypothesis).  A terminal-data
extension :math:`h` is favourable for the boundary-constrained learning
problem when the spectral mass of its forcing :math:`Lh` — the per-wavenumber
time integrals :math:`I_k` defined below, whose band sum is the squared strip
norm up to the factor :math:`2\pi` — is concentrated below the cutoff
:math:`K_\star` reachable by the trained network: the network can reduce the
residual only on the spectral components it represents, so the fraction of
the forcing at wavenumbers above the cutoff is not removable within the
representable band.  This script measures the two exact spectral quantities
that the hypothesis compares — the unreachable fraction and the bandwidth
demand — for the seven extensions of the comparison; the link between these
quantities and trained-network behaviour is the reviewer's conjecture and is
not measured here.

Setting.  Generators G1 (advection–diffusion–reaction,
:math:`a(k) = -0.7 k^2 + 1.3\,ik - 0.4`, diffusivity :math:`\nu = 0.7`) and
G2 (Black–Scholes in the log-price coordinate,
:math:`a(k) = -0.125 k^2 - 0.095\,ik - 0.03`, :math:`\nu = 0.125`); the
periodised Bernoulli datum of regularity index :math:`\rho = 1` (exact
coefficients :math:`|c_k| = 2/(2\pi k)^2`, single break point); terminal time
:math:`T = 1`; the extension catalogue of ``measure_forcing_spectra.py``:
convex raw (linear terminal-distance factor :math:`d_T(t) = 1 - t/T`),
constant-in-time, split semigroup with subset :math:`A = \{\partial_{xx}\}`
(defect order :math:`m = 1`) and :math:`A = \{\partial_{xx}, \partial_x\}`
(:math:`m = 0`), graded Gaussian matched (:math:`\nu_c = \nu`) and mismatched
(:math:`\nu_c = \nu/2`), and the exact solution.

Quantities (all exact; no sampling, no time quadrature, no FFT).

1. The per-wavenumber time integral
   :math:`I_k = \int_0^T |\widehat{Lh}(k, t)|^2\, dt` over the working band
   :math:`0 < |k| \le K_{\mathrm{band}}` (default
   :math:`K_{\mathrm{band}} = 2^{20}`), from the library closed forms
   (``squared_forcing_time_integral``).  For the real-coefficient generators
   the integrand is even in :math:`k` (:math:`I_{-k} = I_k`), so the
   evaluation runs over the positive wavenumbers; the doubled positive-side
   sum is validated against the symmetric-band library reference
   ``total_strip_forcing_squared`` at :math:`K = \min(2^{10},
   K_{\mathrm{band}})` to a relative tolerance of :math:`10^{-12}`, raising
   ``ValueError`` on violation (never a silent acceptance).
2. The unreachable fraction over a dyadic grid of cutoffs
   :math:`K_\star = 2^3, \dots, K_{\mathrm{band}}/2`:

   .. math::

       \mathrm{unreachable}(K_\star)
       = \Bigl(\sum_{K_\star < |k| \le K_{\mathrm{band}}} I_k\Bigr)
         \Big/ \Bigl(\sum_{0 < |k| \le K_{\mathrm{band}}} I_k\Bigr),

   with the tail sums accumulated smallest-terms-first through a reversed
   cumulative sum.
3. The bandwidth demand: for each tolerance
   :math:`\gamma \in \{10^{-1}, 10^{-2}, 10^{-3}, 10^{-6}\}`, the smallest
   dyadic :math:`K_\star` with
   :math:`\mathrm{unreachable}(K_\star) \le \gamma`; when no cutoff of the
   grid satisfies the bound, the entry states "not reached within the band"
   explicitly (never a placeholder value).

Derived predictions (dashed envelopes; :math:`\rho = 1`, so
:math:`|c_k|^2 = k^{-4}/(4\pi^4)` exactly).

* Split :math:`A = \{\partial_{xx}\}` (defect
  :math:`b(k) = a_1\,ik + a_0`, with
  :math:`|b(k)|^2 = a_1^2 k^2\,(1 + O(k^{-2}))` as :math:`|k| \to \infty`)
  and matched graded Gaussian (identical defect
  :math:`a(k) + \nu k^2 = a_1\,ik + a_0`): with
  :math:`\varphi(-2\nu k^2) = (2\nu k^2)^{-1}(1 + o(1))` as
  :math:`|k| \to \infty`,
  :math:`I_k = \frac{a_1^2}{2\nu}\,|c_k|^2\,(1 + o(1))`, proportional to
  :math:`k^{-4}`; hence
  :math:`\mathrm{unreachable}(K_\star) = C\,K_\star^{-3}(1 + o(1))` as
  :math:`K_\star \to \infty` (within an unbounded band; band caveat below).
* Split :math:`A = \{\partial_{xx}, \partial_x\}` (defect
  :math:`b = a_0`): :math:`I_k = \frac{a_0^2}{2\nu}\,|c_k|^2\,k^{-2}
  (1 + o(1))` as :math:`|k| \to \infty`, proportional to :math:`k^{-6}`;
  tail proportional to :math:`K_\star^{-5}`.
* Mismatched graded Gaussian (:math:`\nu_c = \nu/2`):
  :math:`|a(k) + \nu_c k^2|^2 = (\nu - \nu_c)^2 k^4\,(1 + O(k^{-2}))` and
  :math:`\varphi(-2\nu_c k^2) = (2\nu_c k^2)^{-1}(1 + o(1))` as
  :math:`|k| \to \infty`, so
  :math:`I_k = \frac{(\nu-\nu_c)^2}{2\nu_c}\,|c_k|^2\,k^2\,(1 + o(1))`,
  proportional to :math:`k^{-2}`; tail proportional to
  :math:`K_\star^{-1}`.
* Constant-in-time: :math:`I_k = T\,|a(k)|^2 |c_k|^2 =
  \frac{\nu^2 T}{4\pi^4}\,(1 + O(k^{-2}))` as :math:`|k| \to \infty`: the
  per-wavenumber integral tends to a positive constant (flat spectrum;
  :math:`\rho = 1`, :math:`p = 1`), so no power-law tail exists and the
  fraction is band-limited: for a :math:`k`-independent :math:`I_k` it
  equals :math:`1 - K_\star/K_{\mathrm{band}}` exactly (integer counts), it
  depends on the working band :math:`K_{\mathrm{band}}`, and it tends to 1
  as :math:`K_{\mathrm{band}} \to \infty` at fixed :math:`K_\star`.
* Convex raw: :math:`I_k = |c_k|^2\,(1/T + \operatorname{Re} a(k) +
  |a(k)|^2 T/3) = \frac{\nu^2 T}{12\pi^4}\,(1 + O(k^{-2}))` as
  :math:`|k| \to \infty`: the same flat regime as constant-in-time, at the
  level reduced by the factor 3 originating from the time polynomial
  :math:`\int_0^T (t/T)^2\, dt = T/3`; the same band-limited statement
  applies.
* Exact solution: :math:`I_k = 0` identically; the total is reported as
  identically zero and the extension is excluded from every ratio through an
  explicit branch (no ratio is formed; no epsilon).

Fit and agreement.  For every extension with a positive total, the tail
exponent is fitted as the log–log slope of
:math:`\mathrm{unreachable}(K_\star)` over the dyadic cutoffs in the decade
below :math:`K_{\mathrm{band}}/8` (:math:`K_{\mathrm{band}}/80 \le K_\star
\le K_{\mathrm{band}}/8`) and compared against the predicted value with the
absolute tolerance 0.1: predicted :math:`-3, -3, -5, -1` for the four
convergent extensions; for the two band-limited extensions the predicted
value is the fitted log–log slope of :math:`1 - K_\star/K_{\mathrm{band}}`
over the same cutoffs.  The tolerance covers the deterministic finite-band
corrections at the fit decade: the Euler–Maclaurin remainders (relative size
of order :math:`1/K_\star` at the fit cutoffs, below :math:`10^{-3}` at the
default sizes and a few times :math:`10^{-2}` at the smallest band edges the
guard admits under ``--debug``), and, for the mismatched graded extension,
the band truncation, which multiplies the unbounded-band tail
:math:`K_\star^{-1}` by :math:`1 - K_\star/K_{\mathrm{band}}` and thereby
lowers the fitted slope by
:math:`(\ln(1 - 1/8) - \ln(1 - 1/80))/\ln 10 = -5.3\times10^{-2}`.  Any
disagreement is flagged with a ``[DISAGREEMENT]`` tag in the log and
recorded in ``summary.yaml`` (same convention as
``measure_forcing_spectra.py``), never suppressed.

Measurement policy.  Every quantity is evaluated from the exact analytic
Fourier coefficients and the closed-form per-wavenumber time integrals over
integer wavenumber arrays (vectorised, chunked so the peak memory stays
bounded); the FFT of sampled values is never used and no time quadrature is
performed.

Artefacts (all written before any figure; ``--replot RUN_DIR`` rebuilds every
figure from the saved artefacts alone, without recomputation):
``spectral_gap_measurements.npz`` (per-wavenumber integrals over
:math:`0 < k \le K_{\mathrm{band}}` — even in :math:`k` — the
unreachable-fraction curves and the predicted dashed curves),
``summary.yaml`` (totals, the fraction at every dyadic cutoff, bandwidth
demands, fitted-versus-predicted tail exponents with agreement booleans),
``bandwidth_demand_table.txt`` (the printed bandwidth-demand table),
``run_metadata.json``, ``command.txt``, ``run.log``, and one figure per
generator (``unreachable_fraction__<generator>.png``: log–log, solid
measured curves, dashed predicted tails, viridis colour per extension,
legend outside below the axes, formula textbox).
"""
from __future__ import annotations

import argparse
import logging
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
from learning_option_pricing.pde.periodic_spectral_toolbox import TWO_PI  # noqa: E402
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

LOGGER = logging.getLogger(Path(__file__).stem)

TERMINAL_TIME = 1.0
REGULARITY_INDEX = 1  # rho = 1: first-derivative discontinuity of the datum

MEASUREMENTS_FILENAME = "spectral_gap_measurements.npz"
SUMMARY_FILENAME = "summary.yaml"
BANDWIDTH_DEMAND_TABLE_FILENAME = "bandwidth_demand_table.txt"

# Smoke-test guard: a real run uses the working band |k| <= 2^20; any band
# edge below 2^14 without --debug is mechanically rejected (see the
# repository convention on unflagged smoke runs).
SMOKE_TEST_LOG2_BAND_EDGE_THRESHOLD = 14

# Relative tolerance for the cross-check of the doubled positive-wavenumber
# accumulation against the symmetric-band library reference.  A violation
# raises ValueError (never a silent acceptance).
CROSS_CHECK_RELATIVE_TOLERANCE = 1.0e-12
CROSS_CHECK_BAND_EDGE = 2**10

# Absolute tolerance for the fitted-versus-predicted tail-exponent
# comparison.  The predicted exponents differ pairwise by at least 2, and
# the deterministic finite-band corrections at the fit decade (see the
# module docstring) stay below this tolerance for every band edge the
# smoke-test guard admits.
EXPONENT_AGREEMENT_ABSOLUTE_TOLERANCE = 0.1

# Tolerances gamma of the bandwidth demand: the demand at gamma is the
# smallest dyadic cutoff whose unreachable fraction does not exceed gamma.
UNREACHABLE_FRACTION_THRESHOLDS = (1.0e-1, 1.0e-2, 1.0e-3, 1.0e-6)

# Tail-exponent fit range: the dyadic cutoffs in the decade below
# K_band / FIT_UPPER_EDGE_DIVISOR, i.e. cutoffs K_star with
# K_band / (FIT_UPPER_EDGE_DIVISOR * FIT_DECADE_FACTOR) <= K_star
# <= K_band / FIT_UPPER_EDGE_DIVISOR.
FIT_UPPER_EDGE_DIVISOR = 8
FIT_DECADE_FACTOR = 10

# Explicit report strings (never a silent placeholder).
NOT_REACHED_MESSAGE = "not reached within the band"
IDENTICALLY_ZERO_MESSAGE = "identically zero (no forcing)"

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

# Predicted tail behaviour of the unreachable fraction per extension.  The
# derivations are stated in the module docstring; with rho = 1 the datum has
# |c_k|^2 = k^{-4}/(4 pi^4) exactly.
#
# "power_law": the per-wavenumber integral satisfies
# I_k = C k^{q} (1 + o(1)) as |k| -> infinity with q < -1, so the tail sum
# over |k| > K_star scales as K_star^{q+1} and the unreachable fraction has
# the tail exponent q + 1:
#   split {d_xx} and matched graded (q = -4)  ->  exponent -3;
#   split {d_xx, d_x}                (q = -6)  ->  exponent -5;
#   mismatched graded                (q = -2)  ->  exponent -1.
# "band_limited_flat": I_k tends to a positive constant (q = 0); the tail
# sum grows with the band, the fraction equals 1 - K_star/K_band for a
# k-independent I_k, and no power-law tail exists — the predicted value for
# the fit is the log-log slope of that band-limited curve over the same fit
# cutoffs, and the fraction tends to 1 as K_band -> infinity at fixed
# K_star.
# "identically_zero": the exact solution has I_k = 0; no ratio is formed
# (explicit branch) and the extension is excluded from the figure.
PREDICTED_TAIL = {
    "convex_raw": {"kind": "band_limited_flat"},
    "constant_in_time": {"kind": "band_limited_flat"},
    "split_diffusion": {"kind": "power_law", "exponent": -3},
    "split_diffusion_advection": {"kind": "power_law", "exponent": -5},
    "graded_gaussian_matched": {"kind": "power_law", "exponent": -3},
    "graded_gaussian_mismatched": {"kind": "power_law", "exponent": -1},
    "exact_solution": {"kind": "identically_zero"},
}


# ---------------------------------------------------------------------------
# Extension catalogue (mirrors measure_forcing_spectra.build_extension_catalogue)
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


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------


def dyadic_cutoff_grid(
    log2_minimum_cutoff: int, log2_band_edge: int
) -> np.ndarray:
    r"""Dyadic cutoffs :math:`K_\star = 2^{j}` for :math:`j` from the minimum up to :math:`K_{\mathrm{band}}/2`.

    Args:
        log2_minimum_cutoff: Exponent of the smallest cutoff (default grid
            starts at :math:`2^3`).
        log2_band_edge: Exponent of the working band edge; the grid ends at
            :math:`2^{\text{log2\_band\_edge} - 1} = K_{\mathrm{band}}/2`.

    Returns:
        Increasing ``int64`` array of dyadic cutoffs.
    """
    return 2 ** np.arange(
        log2_minimum_cutoff, log2_band_edge, dtype=np.int64
    )


def fit_cutoff_mask(dyadic_cutoffs: np.ndarray, band_edge: int) -> np.ndarray:
    r"""Boolean mask of the fit cutoffs: the decade below :math:`K_{\mathrm{band}}/8`.

    The tail-exponent fit runs over the dyadic cutoffs :math:`K_\star` with
    :math:`K_{\mathrm{band}}/80 \le K_\star \le K_{\mathrm{band}}/8`.
    """
    fit_upper_edge = band_edge // FIT_UPPER_EDGE_DIVISOR
    fit_lower_edge = fit_upper_edge / FIT_DECADE_FACTOR
    return (dyadic_cutoffs >= fit_lower_edge) & (dyadic_cutoffs <= fit_upper_edge)


def compute_per_wavenumber_integrals(
    extension: TerminalDataExtension,
    band_edge: int,
    chunk_length: int,
) -> np.ndarray:
    r"""Closed-form :math:`I_k = \int_0^T |\widehat{Lh}(k, t)|^2 dt` for :math:`k = 1, \dots, K_{\mathrm{band}}`.

    The library closed form is evaluated over consecutive chunks of at most
    ``chunk_length`` positive wavenumbers so the peak memory stays bounded.
    Only the positive wavenumbers are evaluated: for the real-coefficient
    generators of this study the integrand is even in :math:`k`
    (:math:`I_{-k} = I_k`), and the evenness is validated against the
    symmetric-band library reference by
    :func:`cross_check_against_library_reference`.

    Args:
        extension: A :class:`TerminalDataExtension` instance.
        band_edge: The working band edge :math:`K_{\mathrm{band}}`.
        chunk_length: Maximal number of positive wavenumbers evaluated at
            once.

    Returns:
        ``float64`` array of length ``band_edge``; entry ``i`` is
        :math:`I_{i+1}`.
    """
    per_wavenumber_integrals = np.empty(band_edge, dtype=np.float64)
    chunk_start = 1
    while chunk_start <= band_edge:
        chunk_stop = min(chunk_start + chunk_length - 1, band_edge)
        chunk_wavenumbers = np.arange(chunk_start, chunk_stop + 1, dtype=np.int64)
        per_wavenumber_integrals[chunk_start - 1 : chunk_stop] = (
            extension.squared_forcing_time_integral(chunk_wavenumbers)
        )
        chunk_start = chunk_stop + 1
    return per_wavenumber_integrals


def cross_check_against_library_reference(
    extension: TerminalDataExtension,
    per_wavenumber_integrals: np.ndarray,
    check_band_edge: int,
) -> dict:
    r"""Validate the doubled positive-side sum against ``total_strip_forcing_squared``.

    The check compares :math:`2\pi \cdot 2 \sum_{0 < k \le K} I_k` (evenness
    assumption plus chunked accumulation) against the symmetric-band library
    sum at the same band edge :math:`K`.  The identically-zero case (exact
    solution) is handled by an explicit branch: both values must equal
    ``0.0`` exactly and no ratio is formed.

    Args:
        extension: The extension under check.
        per_wavenumber_integrals: The positive-wavenumber integrals.
        check_band_edge: The band edge of the comparison.

    Returns:
        Mapping with keys ``band_edge``, ``relative_discrepancy`` (``None``
        when both values are exactly zero) and ``both_exactly_zero``.

    Raises:
        ValueError: If exactly one of the two values vanishes, or if the
            relative discrepancy exceeds
            ``CROSS_CHECK_RELATIVE_TOLERANCE`` (reported with both values).
    """
    doubled_positive_total = TWO_PI * 2.0 * float(
        np.sum(per_wavenumber_integrals[:check_band_edge])
    )
    library_total = total_strip_forcing_squared(
        extension, symmetric_wavenumber_band(check_band_edge)
    )
    if doubled_positive_total == 0.0 or library_total == 0.0:
        if not (doubled_positive_total == 0.0 and library_total == 0.0):
            raise ValueError(
                "cross-check inconsistency at K = "
                f"{check_band_edge}: doubled positive-side total = "
                f"{doubled_positive_total!r}, library symmetric-band total = "
                f"{library_total!r} (exactly one of the two vanishes)"
            )
        return {
            "band_edge": int(check_band_edge),
            "relative_discrepancy": None,
            "both_exactly_zero": True,
        }
    relative_discrepancy = abs(doubled_positive_total - library_total) / abs(
        library_total
    )
    if relative_discrepancy > CROSS_CHECK_RELATIVE_TOLERANCE:
        raise ValueError(
            "the doubled positive-wavenumber accumulation disagrees with the "
            f"symmetric-band library reference at K = {check_band_edge}: "
            f"doubled positive-side total = {doubled_positive_total!r}, "
            f"library total = {library_total!r}, relative discrepancy = "
            f"{relative_discrepancy:.3e} > "
            f"{CROSS_CHECK_RELATIVE_TOLERANCE:.0e}"
        )
    return {
        "band_edge": int(check_band_edge),
        "relative_discrepancy": float(relative_discrepancy),
        "both_exactly_zero": False,
    }


def bandwidth_demand_mapping(
    dyadic_cutoffs: np.ndarray, unreachable_fractions: np.ndarray
) -> dict[str, int | str]:
    r"""Smallest dyadic cutoff with :math:`\mathrm{unreachable}(K_\star) \le \gamma`, per tolerance.

    Returns:
        Mapping from the formatted tolerance (e.g. ``"1e-03"``) to the
        smallest satisfying cutoff, or to the explicit string
        ``"not reached within the band"`` when no cutoff of the grid
        satisfies the bound (never a silent placeholder).
    """
    demand: dict[str, int | str] = {}
    for gamma in UNREACHABLE_FRACTION_THRESHOLDS:
        satisfied = unreachable_fractions <= gamma
        if bool(np.any(satisfied)):
            demand[f"{gamma:.0e}"] = int(dyadic_cutoffs[int(np.argmax(satisfied))])
        else:
            demand[f"{gamma:.0e}"] = NOT_REACHED_MESSAGE
    return demand


def compose_bandwidth_demand_table(summary: dict) -> list[str]:
    """Compose the bandwidth-demand table lines from the summary mapping.

    The table is built from the saved ``bandwidth_demand`` fields (never from
    a recomputation), so the printed table restates the artefact.

    Args:
        summary: The summary mapping of a completed measurement.

    Returns:
        List of text lines (header, then one block per generator).
    """
    gamma_labels = [
        f"{gamma:.0e}"
        for gamma in summary["parameters"]["unreachable_fraction_thresholds"]
    ]
    log2_band_edge = summary["parameters"]["log2_band_edge"]
    lines = [
        "Bandwidth demand: smallest dyadic cutoff K_star with "
        "unreachable_fraction(K_star) <= gamma",
        f"Working band 0 < |k| <= K_band = 2^{log2_band_edge} = "
        f"{summary['parameters']['band_edge']}; cutoff grid K_star = "
        f"2^{summary['parameters']['log2_minimum_cutoff']} .. "
        f"2^{log2_band_edge - 1} (dyadic).",
    ]
    for generator_key in summary["generator_order"]:
        lines.append("")
        lines.append(f"Generator: {generator_key}")
        lines.append(
            f"{'Extension':30s}"
            + "".join(f"{'Gamma ' + label:>30s}" for label in gamma_labels)
        )
        for extension_key in summary["extension_order"]:
            demand = summary["generators"][generator_key]["per_extension"][
                extension_key
            ]["bandwidth_demand"]
            lines.append(
                f"{extension_key:30s}"
                + "".join(f"{str(demand[label]):>30s}" for label in gamma_labels)
            )
    return lines


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def run_measurements(arguments: argparse.Namespace, run_directory: Path) -> None:
    """Measure the spectral-gap quantities and save every artefact.

    Writes ``spectral_gap_measurements.npz`` (per-wavenumber integrals,
    unreachable-fraction curves, predicted dashed curves), ``summary.yaml``
    and ``bandwidth_demand_table.txt`` into ``run_directory`` before any
    figure is drawn.
    """
    datum = PeriodisedBernoulliDatum(regularity_index=REGULARITY_INDEX)
    generators = build_generator_catalogue()
    band_edge = 2**arguments.log2_band_edge
    dyadic_cutoffs = dyadic_cutoff_grid(
        arguments.log2_minimum_cutoff, arguments.log2_band_edge
    )
    fit_mask = fit_cutoff_mask(dyadic_cutoffs, band_edge)
    fit_cutoffs = dyadic_cutoffs[fit_mask]
    check_band_edge = min(CROSS_CHECK_BAND_EDGE, band_edge)
    LOGGER.info(
        "Working band 0 < |k| <= 2^%d = %d; %d dyadic cutoffs 2^%d .. 2^%d; "
        "tail-exponent fit over the %d cutoffs in [%d, %d] (the decade below "
        "K_band/%d); cross-check at K = %d; chunk length %d",
        arguments.log2_band_edge,
        band_edge,
        len(dyadic_cutoffs),
        arguments.log2_minimum_cutoff,
        arguments.log2_band_edge - 1,
        len(fit_cutoffs),
        int(fit_cutoffs[0]),
        int(fit_cutoffs[-1]),
        FIT_UPPER_EDGE_DIVISOR,
        check_band_edge,
        arguments.chunk_length,
    )

    saved_arrays: dict[str, np.ndarray] = {
        "positive_wavenumbers": np.arange(1, band_edge + 1, dtype=np.int64),
        "dyadic_cutoffs": dyadic_cutoffs,
        "fit_cutoffs": fit_cutoffs,
    }
    summary: dict = {
        "parameters": {
            "terminal_time": TERMINAL_TIME,
            "regularity_index": REGULARITY_INDEX,
            "log2_band_edge": int(arguments.log2_band_edge),
            "band_edge": int(band_edge),
            "log2_minimum_cutoff": int(arguments.log2_minimum_cutoff),
            "chunk_length": int(arguments.chunk_length),
            "seed": int(arguments.seed),
            "debug": bool(arguments.debug),
            "unreachable_fraction_thresholds": [
                float(gamma) for gamma in UNREACHABLE_FRACTION_THRESHOLDS
            ],
            "fit_upper_edge_divisor": FIT_UPPER_EDGE_DIVISOR,
            "fit_decade_factor": FIT_DECADE_FACTOR,
            "exponent_agreement_absolute_tolerance": (
                EXPONENT_AGREEMENT_ABSOLUTE_TOLERANCE
            ),
            "cross_check_band_edge": int(check_band_edge),
            "cross_check_relative_tolerance": CROSS_CHECK_RELATIVE_TOLERANCE,
        },
        "dyadic_cutoffs": [int(cutoff) for cutoff in dyadic_cutoffs],
        "fit_cutoffs": [int(cutoff) for cutoff in fit_cutoffs],
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
        per_extension_summaries: dict[str, dict] = {}

        for extension_key in EXTENSION_ORDER:
            extension_start_time = time.perf_counter()
            extension = extensions[extension_key]
            per_wavenumber_integrals = compute_per_wavenumber_integrals(
                extension, band_edge, arguments.chunk_length
            )
            saved_arrays[
                f"per_wavenumber_integral__{generator_key}__{extension_key}"
            ] = per_wavenumber_integrals
            cross_check_record = cross_check_against_library_reference(
                extension, per_wavenumber_integrals, check_band_edge
            )

            if extension_key == "exact_solution":
                # Identically-zero branch: the total vanishes, so no ratio is
                # formed (explicit branch, no epsilon) and the extension is
                # excluded from the fraction curves and the figure.
                maximum_integral = float(np.max(per_wavenumber_integrals))
                if maximum_integral != 0.0:
                    raise AssertionError(
                        "the exact-solution extension must have identically "
                        "zero per-wavenumber integrals, but max I_k = "
                        f"{maximum_integral!r} for generator "
                        f"'{generator_key}'"
                    )
                per_extension_summaries[extension_key] = {
                    "total_strip_forcing_squared_over_band": 0.0,
                    "identically_zero": True,
                    "cross_check": cross_check_record,
                    "unreachable_fraction_at_cutoffs": None,
                    "bandwidth_demand": {
                        f"{gamma:.0e}": IDENTICALLY_ZERO_MESSAGE
                        for gamma in UNREACHABLE_FRACTION_THRESHOLDS
                    },
                    "tail_exponent": {
                        "fitted": None,
                        "predicted_kind": "identically_zero",
                        "predicted_asymptotic_exponent": None,
                        "predicted_for_fit": None,
                        "absolute_deviation": None,
                        "tolerance": EXPONENT_AGREEMENT_ABSOLUTE_TOLERANCE,
                        "agreement": None,
                        "fit_cutoff_range": [
                            int(fit_cutoffs[0]),
                            int(fit_cutoffs[-1]),
                        ],
                    },
                    "note": (
                        "the forcing of the exact solution vanishes "
                        "identically; no ratio is formed (explicit branch, "
                        "no epsilon) and the extension is excluded from the "
                        "figure"
                    ),
                }
                LOGGER.info(
                    "[%s] %-28s total ||Lh||^2 over the band = 0.0 exactly "
                    "(identically zero; excluded from every ratio)  [%.2f s]",
                    generator_key,
                    extension_key,
                    time.perf_counter() - extension_start_time,
                )
                continue

            # Tail sums accumulated smallest-terms-first (reversed cumulative
            # sum), so the deep tails are not formed as differences of large
            # totals.
            reversed_cumulative_sum = np.cumsum(per_wavenumber_integrals[::-1])
            total_positive_sum = float(reversed_cumulative_sum[-1])
            if not total_positive_sum > 0.0:
                raise ValueError(
                    f"the non-exact extension '{extension_key}' returned a "
                    f"non-positive total {total_positive_sum!r} for generator "
                    f"'{generator_key}'; the unreachable fraction is undefined"
                )
            tail_sums = reversed_cumulative_sum[band_edge - dyadic_cutoffs - 1]
            unreachable_fractions = tail_sums / total_positive_sum
            if not bool(np.all(unreachable_fractions[fit_mask] > 0.0)):
                raise ValueError(
                    "the tail-exponent fit requires strictly positive "
                    f"unreachable fractions, but extension '{extension_key}' "
                    f"on generator '{generator_key}' has a vanishing fraction "
                    "inside the fit range"
                )
            saved_arrays[
                f"unreachable_fraction__{generator_key}__{extension_key}"
            ] = unreachable_fractions
            total_strip_forcing = TWO_PI * 2.0 * total_positive_sum

            fitted_tail_exponent = float(
                np.polyfit(
                    np.log(fit_cutoffs.astype(np.float64)),
                    np.log(unreachable_fractions[fit_mask]),
                    1,
                )[0]
            )
            prediction = PREDICTED_TAIL[extension_key]
            extension_note: str | None = None
            if prediction["kind"] == "power_law":
                predicted_asymptotic_exponent: int | None = int(
                    prediction["exponent"]
                )
                predicted_exponent_for_fit = float(prediction["exponent"])
                # Dashed envelope: the predicted power-law tail, anchored to
                # the measured fraction at the lower edge of the fit range
                # (the slope is the prediction; the level is anchored, and
                # the anchoring is stated in the figure caption).
                anchor_index = int(np.argmax(fit_mask))
                anchor_cutoff = float(dyadic_cutoffs[anchor_index])
                anchor_fraction = float(unreachable_fractions[anchor_index])
                predicted_cutoffs = dyadic_cutoffs[dyadic_cutoffs >= anchor_cutoff]
                predicted_fraction_curve = anchor_fraction * (
                    predicted_cutoffs.astype(np.float64) / anchor_cutoff
                ) ** predicted_exponent_for_fit
            else:  # band_limited_flat
                predicted_asymptotic_exponent = None
                # Dashed envelope: the band-limited curve 1 - K_star/K_band,
                # exact for a k-independent per-wavenumber integral (integer
                # counts); no anchoring is involved.
                predicted_cutoffs = dyadic_cutoffs
                predicted_fraction_curve = (
                    1.0 - dyadic_cutoffs.astype(np.float64) / float(band_edge)
                )
                predicted_exponent_for_fit = float(
                    np.polyfit(
                        np.log(fit_cutoffs.astype(np.float64)),
                        np.log(
                            1.0
                            - fit_cutoffs.astype(np.float64) / float(band_edge)
                        ),
                        1,
                    )[0]
                )
                extension_note = (
                    "the per-wavenumber integral tends to a positive "
                    "constant, so no power-law tail exists: the fraction "
                    "follows the band-limited curve 1 - K_star/K_band, "
                    "depends on the working band, and tends to 1 as K_band "
                    "grows at fixed K_star; the predicted value compared "
                    "against the fit is the log-log slope of that curve over "
                    "the same fit cutoffs"
                )
            saved_arrays[
                f"predicted_fraction__{generator_key}__{extension_key}"
            ] = predicted_fraction_curve
            saved_arrays[
                f"predicted_fraction_cutoffs__{generator_key}__{extension_key}"
            ] = predicted_cutoffs

            absolute_deviation = abs(
                fitted_tail_exponent - predicted_exponent_for_fit
            )
            agreement = (
                absolute_deviation <= EXPONENT_AGREEMENT_ABSOLUTE_TOLERANCE
            )
            tail_exponent_entry: dict = {
                "fitted": float(fitted_tail_exponent),
                "predicted_kind": prediction["kind"],
                "predicted_asymptotic_exponent": predicted_asymptotic_exponent,
                "predicted_for_fit": float(predicted_exponent_for_fit),
                "absolute_deviation": float(absolute_deviation),
                "tolerance": EXPONENT_AGREEMENT_ABSOLUTE_TOLERANCE,
                "agreement": bool(agreement),
                "fit_cutoff_range": [int(fit_cutoffs[0]), int(fit_cutoffs[-1])],
            }
            if not agreement:
                # Disagreement mechanism (same convention as
                # measure_forcing_spectra.py): the summary records the
                # conflict explicitly and the log line below appends a
                # [DISAGREEMENT] tag; a disagreement is never suppressed —
                # its analytic account must be recorded before the result is
                # cited.
                tail_exponent_entry["note"] = (
                    "the fitted tail exponent disagrees with the prediction; "
                    "see the [DISAGREEMENT] line in run.log and record the "
                    "analytic account before citing this result"
                )

            entry: dict = {
                "total_strip_forcing_squared_over_band": float(
                    total_strip_forcing
                ),
                "identically_zero": False,
                "cross_check": cross_check_record,
                "unreachable_fraction_at_cutoffs": [
                    float(value) for value in unreachable_fractions
                ],
                "bandwidth_demand": bandwidth_demand_mapping(
                    dyadic_cutoffs, unreachable_fractions
                ),
                "tail_exponent": tail_exponent_entry,
            }
            if extension_note is not None:
                entry["note"] = extension_note
            per_extension_summaries[extension_key] = entry

            LOGGER.info(
                "[%s] %-28s total ||Lh||^2 over the band = %.6e; tail "
                "exponent fitted %.4f vs predicted %.4f (%s); cross-check "
                "%.2e; agreement %s%s  [%.2f s]",
                generator_key,
                extension_key,
                total_strip_forcing,
                fitted_tail_exponent,
                predicted_exponent_for_fit,
                prediction["kind"],
                cross_check_record["relative_discrepancy"],
                agreement,
                "" if agreement else "  [DISAGREEMENT]",
                time.perf_counter() - extension_start_time,
            )

        # Coincidence of the matched graded Gaussian and the split
        # {d_xx}: the two defect symbols are identical analytically, so the
        # per-wavenumber integrals must agree to rounding.
        split_integrals = saved_arrays[
            f"per_wavenumber_integral__{generator_key}__split_diffusion"
        ]
        matched_integrals = saved_arrays[
            f"per_wavenumber_integral__{generator_key}__graded_gaussian_matched"
        ]
        maximum_relative_difference = float(
            np.max(np.abs(matched_integrals - split_integrals) / split_integrals)
        )
        generator_summary["graded_matched_equals_split_diffusion"] = {
            "max_relative_per_wavenumber_integral_difference": (
                maximum_relative_difference
            ),
        }
        LOGGER.info(
            "[%s] matched graded Gaussian versus split {d_xx}: max relative "
            "per-wavenumber integral difference %.3e",
            generator_key,
            maximum_relative_difference,
        )

        generator_summary["per_extension"] = per_extension_summaries
        summary["generators"][generator_key] = generator_summary

    # ---- Bandwidth-demand table: printed to the log and saved -----------
    table_lines = compose_bandwidth_demand_table(summary)
    for line in table_lines:
        LOGGER.info("%s", line)
    table_path = run_directory / BANDWIDTH_DEMAND_TABLE_FILENAME
    table_path.write_text("\n".join(table_lines) + "\n", encoding="utf-8")
    LOGGER.info("Saved bandwidth-demand table to %s", table_path)

    # ---- Save artefacts before plotting ---------------------------------
    measurements_path = run_directory / MEASUREMENTS_FILENAME
    np.savez_compressed(measurements_path, **saved_arrays)
    LOGGER.info(
        "Saved per-wavenumber integrals, fraction curves and predicted "
        "curves to %s",
        measurements_path,
    )
    summary_path = run_directory / SUMMARY_FILENAME
    with open(summary_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)
    LOGGER.info("Saved summary to %s", summary_path)


# ---------------------------------------------------------------------------
# Figures (built from the saved artefacts only)
# ---------------------------------------------------------------------------


def compose_extension_legend_label(
    extension_key: str, per_extension_entry: dict
) -> str:
    """Legend label of one extension, restating the saved fit comparison.

    The fitted and predicted values are read from the saved summary entry
    (never recomputed), and a disagreement is flagged in the label itself.
    """
    tail_exponent_entry = per_extension_entry["tail_exponent"]
    base_label = EXTENSION_DISPLAY_LABELS[extension_key]
    if tail_exponent_entry["predicted_kind"] == "power_law":
        label = (
            f"{base_label}; exponent fitted "
            f"{tail_exponent_entry['fitted']:.3f} | predicted "
            f"{tail_exponent_entry['predicted_asymptotic_exponent']}"
        )
    else:
        label = (
            f"{base_label}; band-limited, slope fitted "
            f"{tail_exponent_entry['fitted']:.3f} | predicted curve "
            f"{tail_exponent_entry['predicted_for_fit']:.3f}"
        )
    if tail_exponent_entry["agreement"] is False:
        label += "  [DISAGREEMENT]"
    return label


def build_all_figures(run_directory: Path) -> None:
    """Rebuild every figure from the saved ``.npz`` and ``summary.yaml`` alone.

    This function performs no measurement: it is shared verbatim by the
    in-run plotting path and by ``--replot``.
    """
    measurements = np.load(run_directory / MEASUREMENTS_FILENAME)
    with open(run_directory / SUMMARY_FILENAME, "r", encoding="utf-8") as handle:
        summary = yaml.safe_load(handle)

    dyadic_cutoffs = measurements["dyadic_cutoffs"].astype(np.float64)
    log2_band_edge = summary["parameters"]["log2_band_edge"]
    anchor_cutoff = summary["fit_cutoffs"][0]

    formula_text = (
        r"$\mathrm{unreachable}(K_\star)=\left(\sum_{|k|>K_\star}I_k\right)"
        r"/\left(\sum_{0<|k|\leq K_{\mathrm{band}}}I_k\right)$ with"
        r" $I_k=\int_0^T|\widehat{Lh}(k,t)|^2\,dt$ the closed-form time"
        r" integral (no quadrature); working band"
        r" $K_{\mathrm{band}}=2^{" + str(log2_band_edge) + r"}$."
        "\n"
        r"Dashed predicted tails, anchored to the measured fraction at"
        r" $K_\star=" + str(anchor_cutoff) + r"$:"
        r" $\propto K_\star^{-3}$ split $\{\partial_{xx}\}$ and matched"
        r" graded ($I_k\propto k^{-4}$);"
        r" $\propto K_\star^{-5}$ split $\{\partial_{xx},\partial_x\}$"
        r" ($I_k\propto k^{-6}$);"
        r" $\propto K_\star^{-1}$ mismatched graded ($I_k\propto k^{-2}$)."
        "\n"
        r"Constant-in-time / convex raw: $I_k$ tends to a positive constant,"
        r" so the fraction is band-limited — dashed reference"
        r" $1-K_\star/K_{\mathrm{band}}$ (exact for $k$-independent $I_k$,"
        r" no anchoring);"
        "\n"
        r"it depends on the working band and tends to 1 as"
        r" $K_{\mathrm{band}}\to\infty$ at fixed $K_\star$."
        r" Exact solution: $I_k=0$ identically; excluded (no ratio is"
        r" formed)."
        "\n"
        r"Tail exponents fitted over the dyadic cutoffs in"
        r" $[K_{\mathrm{band}}/80,\,K_{\mathrm{band}}/8]$."
    )

    for generator_key in summary["generator_order"]:
        per_extension_summaries = summary["generators"][generator_key][
            "per_extension"
        ]
        # The exclusion of the identically-zero extension is driven by the
        # saved flag, not by the extension name.
        plotted_extension_keys = [
            extension_key
            for extension_key in summary["extension_order"]
            if not per_extension_summaries[extension_key]["identically_zero"]
        ]
        extension_colours = {
            extension_key: colour
            for extension_key, colour in zip(
                plotted_extension_keys,
                plt.cm.viridis(
                    np.linspace(0.0, 0.92, len(plotted_extension_keys))
                ),
            )
        }

        fig, ax = plt.subplots(figsize=(12.5, 7.0))
        predicted_line_handle = None
        for extension_key in plotted_extension_keys:
            unreachable_fractions = measurements[
                f"unreachable_fraction__{generator_key}__{extension_key}"
            ]
            # The split {d_xx} curve is widened so that it remains visible
            # under the exactly coincident matched graded Gaussian curve.
            line_width = 3.0 if extension_key == "split_diffusion" else 1.5
            # Solid stroke: measured curve.
            ax.loglog(
                dyadic_cutoffs,
                unreachable_fractions,
                "-o",
                color=extension_colours[extension_key],
                linewidth=line_width,
                markersize=3.5,
                label=compose_extension_legend_label(
                    extension_key, per_extension_summaries[extension_key]
                ),
            )
            # Dashed stroke: the predicted tail (power-law tail anchored at
            # the fit lower edge, or the band-limited reference curve).
            (predicted_line_handle,) = ax.loglog(
                measurements[
                    f"predicted_fraction_cutoffs__{generator_key}__{extension_key}"
                ].astype(np.float64),
                measurements[
                    f"predicted_fraction__{generator_key}__{extension_key}"
                ],
                "--",
                color=extension_colours[extension_key],
                linewidth=1.2,
                alpha=0.9,
            )
        ax.set_xlabel(r"Cutoff $K_\star$")
        # The defining formula of the fraction is stated in the textbox
        # below the figure; the axis label stays short so it fits the canvas.
        ax.set_ylabel(r"Unreachable fraction $\mathrm{unreachable}(K_\star)$")
        ax.set_title(
            "Unreachable fraction of the strip forcing versus the cutoff\n"
            + GENERATOR_DISPLAY_LABELS[generator_key],
            fontsize=10,
        )
        ax.grid(True, which="both", alpha=0.3)

        legend_handles, legend_labels = ax.get_legend_handles_labels()
        if predicted_line_handle is not None:
            legend_handles.append(
                Line2D([], [], color="black", ls="--", lw=1.2)
            )
            legend_labels.append(
                "Predicted tail (dashed, per extension colour; anchoring "
                "stated in the box below)"
            )
        # Figure-level legend anchored above the bottom formula box (outside
        # the axes data area, below, in two columns).
        legend = fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.145),
            ncol=2,
            fontsize=6.5,
            frameon=True,
        )
        fig.tight_layout(rect=[0.0, 0.33, 1.0, 0.98])

        figure_path = run_directory / f"unreachable_fraction__{generator_key}.png"
        finalize_figure(
            fig,
            figure_path,
            legends=[legend],
            axes=[ax],
            formula=formula_text,
            formula_fontsize=7,
        )
        LOGGER.info("Wrote %s", figure_path)


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
        "--log2-band-edge",
        type=int,
        default=20,
        help="the working band retains 0 < |k| <= 2**this "
        "(default 20, i.e. K_band = 1048576)",
    )
    parser.add_argument(
        "--log2-minimum-cutoff",
        type=int,
        default=3,
        help="smallest dyadic cutoff K_star is 2**this (default 3); the "
        "cutoff grid ends at K_band/2",
    )
    parser.add_argument(
        "--chunk-length",
        type=int,
        default=2**19,
        help="maximal number of positive wavenumbers evaluated per chunk "
        "(memory bound; default 2**19)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="master seed, logged for the run-log contract; the measurement "
        "is deterministic and instantiates no random number generator",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="prepend '_debug_' to the output folder name (exploratory runs)",
    )
    parser.add_argument(
        "--replot",
        type=Path,
        default=None,
        metavar="RUN_DIR",
        help="rebuild every figure from the saved artefacts of an existing "
        "run directory, without recomputation",
    )
    arguments = parser.parse_args(argv)

    if arguments.replot is not None:
        return arguments
    if (
        arguments.log2_band_edge < SMOKE_TEST_LOG2_BAND_EDGE_THRESHOLD
        and not arguments.debug
    ):
        parser.error(
            "smoke-test guard: --log2-band-edge "
            f"{arguments.log2_band_edge} < "
            f"{SMOKE_TEST_LOG2_BAND_EDGE_THRESHOLD} requires --debug "
            "(unflagged smoke runs are rejected mechanically)"
        )
    if arguments.log2_minimum_cutoff < 1:
        parser.error("--log2-minimum-cutoff must be at least 1")
    if arguments.log2_minimum_cutoff > arguments.log2_band_edge - 1:
        parser.error(
            "--log2-minimum-cutoff must not exceed --log2-band-edge - 1 "
            "(the cutoff grid ends at K_band/2)"
        )
    if arguments.chunk_length < 1:
        parser.error("--chunk-length must be a positive integer")
    validation_cutoffs = dyadic_cutoff_grid(
        arguments.log2_minimum_cutoff, arguments.log2_band_edge
    )
    validation_fit_mask = fit_cutoff_mask(
        validation_cutoffs, 2**arguments.log2_band_edge
    )
    if int(np.sum(validation_fit_mask)) < 2:
        parser.error(
            "the tail-exponent fit range (the decade below K_band/8, i.e. "
            "cutoffs in [K_band/80, K_band/8]) contains fewer than two "
            "dyadic cutoffs; increase --log2-band-edge or lower "
            "--log2-minimum-cutoff"
        )
    return arguments


def main(argv=None) -> int:
    """Entry point: measure and plot, or replot from saved artefacts."""
    arguments = parse_arguments(argv)
    script_stem = Path(__file__).stem

    if arguments.replot is not None:
        configure_cli_script_logging(verbose=False)
        run_directory = arguments.replot.resolve()
        if not (run_directory / MEASUREMENTS_FILENAME).exists():
            raise FileNotFoundError(
                f"--replot: {run_directory / MEASUREMENTS_FILENAME} not found"
            )
        LOGGER.info("Replot mode: rebuilding figures from %s", run_directory)
        build_all_figures(run_directory)
        return 0

    start_time = time.perf_counter()
    debug_prefix = "_debug_" if arguments.debug else ""
    config_tag = f"rho{REGULARITY_INDEX}_log2K{arguments.log2_band_edge}"
    run_directory = (
        script_data_dir(__file__) / f"{debug_prefix}{utc_timestamp()}_{config_tag}"
    )
    run_directory.mkdir(parents=True, exist_ok=False)

    init_logging(run_dir=run_directory)
    LOGGER.info(
        "Command line: %s", " ".join(shlex.quote(token) for token in sys.argv)
    )
    LOGGER.info(
        "Runtime: Python %s | numpy %s | matplotlib %s | PyYAML %s | torch "
        "not used (pure-numpy measurement)",
        sys.version.split()[0],
        np.__version__,
        matplotlib.__version__,
        yaml.__version__,
    )
    repository_root = find_repo_root(Path(__file__))
    git_metadata = get_git_metadata(repository_root)
    LOGGER.info(
        "Git: commit %s | branch %s | dirty %s",
        git_metadata.get("commit"),
        git_metadata.get("branch"),
        git_metadata.get("dirty"),
    )
    log_parsed_args(LOGGER, arguments)
    LOGGER.info(
        "Master seed: %d (deterministic measurement — no random number "
        "generator is instantiated; the seed is recorded for the run-log "
        "contract).",
        arguments.seed,
    )
    LOGGER.info("Run directory: %s", run_directory)
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
    LOGGER.info("Total wall-clock time: %.2f s", elapsed_seconds)
    LOGGER.info(
        "Follow the log in real time with: tail -f %s", run_directory / "run.log"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
