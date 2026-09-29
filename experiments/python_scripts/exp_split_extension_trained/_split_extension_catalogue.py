r"""Torch-free catalogue for the stage-2 trained split-extension ablation.

This module is imported on cluster login nodes during the ``--init-only``
phase of the array launcher, so it must remain free of any heavy import (no
``torch``, no ``numpy``): it only defines plain-Python configuration
dictionaries.  The authoritative specification is
``documents/methodology/stage2_trained_ablation_specification.md``.

Study layout (specification Section 1).  Three *cells*, each a
(generator, datum) pair on the circle ``[0, 2*pi)`` over the strip
``(0, T) x [0, 2*pi)`` with ``T = 1``:

* ``g1_bernoulli_bandlimited`` — generator ``A = 0.7 d_xx + 1.3 d_x - 0.4``
  (stage-1 G1), band-limited Bernoulli datum
  ``g(x) = sum_{k=1}^{128} cos(k x) / (pi^2 k^2)``;
* ``g2_bernoulli_bandlimited`` — generator
  ``A = 0.125 d_xx - 0.095 d_x - 0.03`` (stage-1 G2, the Black--Scholes
  log-price generator at volatility 0.5 and risk-free rate 0.03), same datum;
* ``heat_sine_single_component`` — pure-heat generator ``A = 0.125 d_xx``
  with the single-spectral-component datum ``g(x) = sin x`` (the control
  cell of specification Section 1.3).

The generator cells compare seven *variants* whose only intervention is the
terminal-data extension ``Psi`` of the hard-constrained trial solution
``u_hat = (1 - lambda(t)) Phi_theta + Psi``; the control cell compares the
matched exponential interpolation factor against the linear convex baseline.
Soft forms are excluded (specification decision D2): the stage-2 axis is the
theta-independent extension forcing ``P Psi``, on which the soft forms are
silent.

Schema (specification Section 1.4 item 1).  Every variant entry has exactly
the fields ``name``, ``form`` (one of the ``FORMS`` of
``learning_option_pricing.models.terminal_ansatz``), ``interpolation``,
``extension`` (a **registry key** of
``learning_option_pricing.pde.EXTENSION_FIELD_REGISTRY``, resolved to torch
callables at build time inside the runner), ``comparison_diffusivity_ratio``
(graded Gaussian variants only), ``smoothing_scale_ratio`` (graded
Chen--Mangasarian variants only; the initial scale is
``epsilon_0 = smoothing_scale_ratio * sqrt(2 nu T)``, the standard deviation
of the heat kernel of the cell's own diffusivity at ``s = T``),
``exponential_rate_gamma`` (control cell only), ``color`` and ``label``.

The runner must assert ``RUNNER_SCRIPT_STEM == Path(__file__).stem`` so the
output-folder-from-filename invariant cannot silently drift.
"""
from __future__ import annotations

import math

# The runner script's filename stem (without extension).  Asserted by the
# runner at startup so the data folder cannot drift from the script name.
RUNNER_SCRIPT_STEM = "ablation_split_extension_trained"


# ---------------------------------------------------------------------------
# Variants of the generator cells (specification Section 1.2, V1--V7)
# ---------------------------------------------------------------------------
# All trained variants use the linear interpolation coefficient
# lambda(t) = t / T, the same network, the same sampler, and the same seeds
# (shared-seed policy); the intervention is the extension alone.

GENERATOR_CELL_VARIANTS: list[dict] = [
    {
        # V1 — existing convex baseline.  Extension Psi = lambda(t) g(x); the
        # analytic-derivative bypass is NOT applied (it remains on the
        # autograd route, per the specification's V1 paragraph).
        "name": "convex_raw",
        "form": "hard_convex",
        "interpolation": "linear",
        "extension": None,
        "comparison_diffusivity_ratio": None,
        "smoothing_scale_ratio": None,
        "exponential_rate_gamma": None,
        "color": "#2ca02c",  # green
        "label": r"convex raw: $\Psi=\lambda(t)\,g$, linear $\lambda$",
    },
    {
        # V2 — existing constant baseline.  Extension Psi = g(x); autograd
        # route (extension=None resolves to the datum path of
        # TerminalAnsatz.extension).
        "name": "constant_in_time",
        "form": "hard_constant",
        "interpolation": "linear",
        "extension": None,
        "comparison_diffusivity_ratio": None,
        "smoothing_scale_ratio": None,
        "exponential_rate_gamma": None,
        "color": "#1f77b4",  # blue
        "label": r"constant-in-time: $\Psi=g$",
    },
    {
        # V3 — split semigroup extension, subset {d_xx}; the forcing
        # satisfies P Psi = mu d_x Psi + r_0 Psi (defect order 1).
        "name": "split_diffusion",
        "form": "hard_constant",
        "interpolation": "linear",
        "extension": "split_diffusion",
        "comparison_diffusivity_ratio": None,
        "smoothing_scale_ratio": None,
        "exponential_rate_gamma": None,
        "color": "#d62728",  # red
        "label": r"diffusion split $\{\partial_{xx}\}$: $P\Psi=\mu\,\partial_x\Psi+r_0\Psi$",
    },
    {
        # V4 — split semigroup extension, subset {d_xx, d_x}; the forcing
        # satisfies P Psi = r_0 Psi (defect order 0).
        "name": "split_diffusion_advection",
        "form": "hard_constant",
        "interpolation": "linear",
        "extension": "split_diffusion_advection",
        "comparison_diffusivity_ratio": None,
        "smoothing_scale_ratio": None,
        "exponential_rate_gamma": None,
        "color": "#ff7f0e",  # orange
        "label": "diffusion–advection split " + r"$\{\partial_{xx},\partial_x\}$: $P\Psi=r_0\Psi$",
    },
    {
        # V5 — graded Gaussian extension with comparison diffusivity
        # nu_c = nu.  Mathematically identical to V3; retained as a
        # plumbing-consistency control of the graded code path
        # (specification decision D3; agreement asserted at build time).
        "name": "graded_gaussian_matched",
        "form": "hard_constant",
        "interpolation": "linear",
        "extension": "graded_gaussian",
        "comparison_diffusivity_ratio": 1.0,
        "smoothing_scale_ratio": None,
        "exponential_rate_gamma": None,
        "color": "#9467bd",  # purple
        "label": r"graded Gaussian, $\nu_c=\nu$ (control of the split $\{\partial_{xx}\}$)",
    },
    {
        # V6 — graded Gaussian extension with comparison diffusivity
        # nu_c = nu / 2 (mis-specified comparison semigroup).
        "name": "graded_gaussian_mismatched",
        "form": "hard_constant",
        "interpolation": "linear",
        "extension": "graded_gaussian",
        "comparison_diffusivity_ratio": 0.5,
        "smoothing_scale_ratio": None,
        "exponential_rate_gamma": None,
        "color": "#8c564b",  # brown
        "label": r"graded Gaussian, $\nu_c=\nu/2$ (mis-specified)",
    },
    {
        # V7 — the exact solution as extension (zero-forcing control): the
        # trained loss measures the optimiser-noise floor of the pipeline.
        "name": "exact_solution",
        "form": "hard_constant",
        "interpolation": "linear",
        "extension": "exact_solution",
        "comparison_diffusivity_ratio": None,
        "smoothing_scale_ratio": None,
        "exponential_rate_gamma": None,
        "color": "#7f7f7f",  # grey
        "label": r"exact solution: $\Psi=u^\star$, $P\Psi=0$",
    },
    {
        # V8 — linearly graded Chen--Mangasarian extension (added 2026-09-28,
        # the arm missing from the July campaign): the datum convolved with
        # the Chen--Mangasarian kernel at scale eps(t) = eps_0 (T - t) / T,
        # eps_0 = sqrt(2 nu T) (the heat-kernel width at s = T).  It meets the
        # datum exactly at t = T but is not the semigroup of any operator, so
        # its forcing at the slice is A g (that of constant_in_time).  Linear
        # grading because the parabolic one makes d_t Psi diverge
        # logarithmically at the slice.
        "name": "graded_chen_mangasarian",
        "form": "hard_constant",
        "interpolation": "linear",
        "extension": "graded_chen_mangasarian",
        "comparison_diffusivity_ratio": None,
        "smoothing_scale_ratio": 1.0,
        "exponential_rate_gamma": None,
        "color": "#e377c2",  # pink
        "label": r"graded Chen--Mangasarian, $\varepsilon_0=\sqrt{2\nu T}$",
    },
    {
        # V9 — as V8 with half the initial scale (sensitivity to eps_0).
        "name": "graded_chen_mangasarian_narrow",
        "form": "hard_constant",
        "interpolation": "linear",
        "extension": "graded_chen_mangasarian",
        "comparison_diffusivity_ratio": None,
        "smoothing_scale_ratio": 0.5,
        "exponential_rate_gamma": None,
        "color": "#bcbd22",  # olive
        "label": r"graded Chen--Mangasarian, $\varepsilon_0=\frac{1}{2}\sqrt{2\nu T}$",
    },
]


# ---------------------------------------------------------------------------
# Variants of the fourth-order cell (added 2026-09-29)
# ---------------------------------------------------------------------------
# The generator G3 = -0.05 d_x^4 + 1.3 d_x - 0.4 has no order-2 term, so the
# split variants retain the principal (order-4) part.  The scale-dependent
# extensions use the reference diffusivity nu_ref = (|c_4| T)^{1/2} / T
# (runner: reference_diffusivity), which equals nu for an order-2 generator,
# so the convention of V5/V8 is unchanged there.  None of the graded
# (Gaussian or Chen--Mangasarian) extensions is the semigroup of the
# biharmonic principal part, whose kernel changes sign.

SPLIT_PRINCIPAL_VARIANT: dict = {
    # Split semigroup extension, subset {d_x^4}; forcing P Psi = mu d_x Psi + r_0 Psi.
    "name": "split_principal",
    "form": "hard_constant",
    "interpolation": "linear",
    "extension": "split_principal",
    "comparison_diffusivity_ratio": None,
    "smoothing_scale_ratio": None,
    "exponential_rate_gamma": None,
    "color": "#d62728",  # red
    "label": r"principal split $\{\partial_x^4\}$: $P\Psi=\mu\,\partial_x\Psi+r_0\Psi$",
}
SPLIT_PRINCIPAL_ADVECTION_VARIANT: dict = {
    # Split semigroup extension, subset {d_x^4, d_x}; forcing P Psi = r_0 Psi.
    "name": "split_principal_advection",
    "form": "hard_constant",
    "interpolation": "linear",
    "extension": "split_principal_advection",
    "comparison_diffusivity_ratio": None,
    "smoothing_scale_ratio": None,
    "exponential_rate_gamma": None,
    "color": "#ff7f0e",  # orange
    "label": r"principal--advection split $\{\partial_x^4,\partial_x\}$: $P\Psi=r_0\Psi$",
}
GRADED_GAUSSIAN_WIDTH_MATCHED_VARIANT: dict = {
    # Heat-kernel (Gaussian) grading at nu_c = nu_ref: width-matched to the
    # biharmonic kernel at s = T, but the semigroup of d_xx, not of d_x^4.
    "name": "graded_gaussian_width_matched",
    "form": "hard_constant",
    "interpolation": "linear",
    "extension": "graded_gaussian",
    "comparison_diffusivity_ratio": 1.0,
    "smoothing_scale_ratio": None,
    "exponential_rate_gamma": None,
    "color": "#8c564b",  # brown
    "label": r"graded Gaussian, $\nu_c=\nu_{\mathrm{ref}}$ (heat kernel)",
}


def _generator_variant(name: str) -> dict:
    for variant in GENERATOR_CELL_VARIANTS:
        if variant["name"] == name:
            return variant
    raise KeyError(name)


FOURTH_ORDER_CELL_VARIANTS: list[dict] = [
    _generator_variant("convex_raw"),
    _generator_variant("constant_in_time"),
    SPLIT_PRINCIPAL_VARIANT,
    SPLIT_PRINCIPAL_ADVECTION_VARIANT,
    GRADED_GAUSSIAN_WIDTH_MATCHED_VARIANT,
    _generator_variant("graded_chen_mangasarian"),
    _generator_variant("exact_solution"),
]


# ---------------------------------------------------------------------------
# Variable-coefficient cells (pre-registration
# documents/methodology/2026-09-29_preregistration_variable_coefficient_split.md)
# ---------------------------------------------------------------------------
# Coefficients c(x) = c_0 + c_1 cos(x - phi) with phi = pi/4, so that at the
# datum's break point x* = 0 the principal coefficient is neither stationary nor
# equal to its mean (section 3.1).  Base values inherited from G2 (LV2, local
# volatility in log-price, sigma(x)^2 / 2 = a(x)) and G3 (LV4); the amplitude
# ratio epsilon is swept.  The two splits retain the principal term only, frozen
# at x* or at the mean; no exact-solution control (no closed form here).

VARIABLE_COEFFICIENT_PHASE = math.pi / 4.0
VARIABLE_COEFFICIENT_AMPLITUDE_RATIOS = (0.25, 0.75)

SPLIT_FROZEN_SINGULAR_VARIANT: dict = {
    # h = e^{(T-t) A*} g, A* = c_{2p}(x*) d_x^{2p} (principal coefficient frozen
    # at the singular point); forcing (c_{2p}(x) - c_{2p}(x*)) d_x^{2p} h + lower.
    "name": "split_frozen_singular",
    "form": "hard_constant",
    "interpolation": "linear",
    "extension": "split_frozen_singular",
    "comparison_diffusivity_ratio": None,
    "smoothing_scale_ratio": None,
    "exponential_rate_gamma": None,
    "color": "#d62728",  # red
    "label": r"split frozen at the singular point $x^\star$",
}
SPLIT_FROZEN_MEAN_VARIANT: dict = {
    # h = e^{(T-t) bar A} g, bar A = bar c_{2p} d_x^{2p} (frozen at the mean).
    "name": "split_frozen_mean",
    "form": "hard_constant",
    "interpolation": "linear",
    "extension": "split_frozen_mean",
    "comparison_diffusivity_ratio": None,
    "smoothing_scale_ratio": None,
    "exponential_rate_gamma": None,
    "color": "#9467bd",  # purple (blue is constant_in_time)
    "label": r"split frozen at the mean",
}

VARIABLE_COEFFICIENT_CELL_VARIANTS: list[dict] = [
    _generator_variant("convex_raw"),
    _generator_variant("constant_in_time"),
    SPLIT_FROZEN_SINGULAR_VARIANT,
    SPLIT_FROZEN_MEAN_VARIANT,
]


def _amplitude_tag(amplitude_ratio: float) -> str:
    return f"eps{amplitude_ratio:.2f}".replace(".", "p")


def _local_volatility_cell(amplitude_ratio: float) -> dict:
    diffusivity_mean, risk_free_rate = 0.125, 0.03
    amplitude = diffusivity_mean * amplitude_ratio
    return {
        # a(x) = 0.125 (1 + eps cos(x - pi/4)); drift r - a(x); reaction -r.
        "generator_coefficients": {
            2: {"constant": diffusivity_mean, "amplitude": amplitude,
                "phase": VARIABLE_COEFFICIENT_PHASE},
            1: {"constant": risk_free_rate - diffusivity_mean, "amplitude": -amplitude,
                "phase": VARIABLE_COEFFICIENT_PHASE},
            0: -risk_free_rate,
        },
        "variable_coefficients": True,
        "datum": "bernoulli_bandlimited",
        "truncation_wavenumber": 128,
        "terminal_time": 1.0,
        "corner_point": 0.0,  # the datum's break point x*, where A* is frozen
        "variant_set": "variable_coefficient",
        "short_label": rf"LV2, $\varepsilon={amplitude_ratio:g}$",
        "label": (
            r"$g(x)=\sum_{k=1}^{128}\frac{\cos(kx)}{\pi^2k^2}$,  "
            r"$A=a(x)\,\partial_{xx}+(r-a(x))\,\partial_x-r$,  "
            rf"$a(x)=0.125\,(1+{amplitude_ratio:g}\cos(x-\pi/4))$, $r=0.03$ "
            r"(LV2, local volatility),  $T=1$"
        ),
    }


def _variable_biharmonic_cell(amplitude_ratio: float) -> dict:
    beta_mean = 0.05
    return {
        # beta(x) = 0.05 (1 + eps cos(x - pi/4)); A = -beta(x) d_x^4 + 1.3 d_x - 0.4.
        "generator_coefficients": {
            4: {"constant": -beta_mean, "amplitude": -beta_mean * amplitude_ratio,
                "phase": VARIABLE_COEFFICIENT_PHASE},
            1: 1.3,
            0: -0.4,
        },
        "variable_coefficients": True,
        "datum": "bernoulli_bandlimited",
        "truncation_wavenumber": 128,
        "terminal_time": 1.0,
        "corner_point": 0.0,
        "variant_set": "variable_coefficient",
        "short_label": rf"LV4, $\varepsilon={amplitude_ratio:g}$",
        "label": (
            r"$g(x)=\sum_{k=1}^{128}\frac{\cos(kx)}{\pi^2k^2}$,  "
            r"$A=-\beta(x)\,\partial_x^4+1.3\,\partial_x-0.4$,  "
            rf"$\beta(x)=0.05\,(1+{amplitude_ratio:g}\cos(x-\pi/4))$ "
            r"(LV4),  $T=1$"
        ),
    }

# Rate gamma = nu k_0^2 = 0.125 * 1^2 of the control cell, passed EXPLICITLY
# (specification decision D10): the library default of
# make_interpolation_coefficient is the eigenvalue-matched value of the
# unit-interval family, sigma^2 pi^2 / 2, which on the circle would silently
# mismatch the factor by pi^2.  The runner asserts this value against
# learning_option_pricing.pde.sine_cell_matched_exponential_rate at build
# time (a violation raises).
CONTROL_CELL_MATCHED_EXPONENTIAL_RATE = 0.125

MATCHED_EXPONENTIAL_FACTOR_VARIANT: dict = {
    # C1 — matched exponential interpolation factor on the control cell:
    # Psi = lambda(t) g = e^{-nu k_0^2 (T - t)} sin x = u^star, so the
    # forcing vanishes identically.
    "name": "matched_exponential_factor",
    "form": "hard_convex",
    "interpolation": "exponential",
    "extension": None,
    "comparison_diffusivity_ratio": None,
    "smoothing_scale_ratio": None,
    "exponential_rate_gamma": CONTROL_CELL_MATCHED_EXPONENTIAL_RATE,
    "color": "#17becf",  # cyan
    "label": r"matched exponential factor: $\lambda(t)=e^{-\nu k_0^2(T-t)}$, $\Psi=\lambda g=u^\star$",
}

# C2 — the non-zero-forcing contrast within the control cell; schema-identical
# to V1 (the same dict object is reused so the two cannot drift).
CONTROL_CELL_VARIANTS: list[dict] = [
    MATCHED_EXPONENTIAL_FACTOR_VARIANT,
    GENERATOR_CELL_VARIANTS[0],  # convex_raw
]

# The full variant catalogue with unique names (specification Section 1.4
# item 1 refers to this list as METHOD_VARIANTS).  The per-cell variant sets
# are selected through variants_for_cell().
METHOD_VARIANTS: list[dict] = GENERATOR_CELL_VARIANTS + [
    SPLIT_PRINCIPAL_VARIANT,
    SPLIT_PRINCIPAL_ADVECTION_VARIANT,
    GRADED_GAUSSIAN_WIDTH_MATCHED_VARIANT,
    SPLIT_FROZEN_SINGULAR_VARIANT,
    SPLIT_FROZEN_MEAN_VARIANT,
    MATCHED_EXPONENTIAL_FACTOR_VARIANT
]


# ---------------------------------------------------------------------------
# Cell configurations (specification Section 1.1)
# ---------------------------------------------------------------------------

CELL_CONFIGS: dict[str, dict] = {
    "g1_bernoulli_bandlimited": {
        # Stage-1 generator G1 (advection–diffusion–reaction):
        # A = 0.7 d_xx + 1.3 d_x - 0.4, symbol a(k) = -0.7 k^2 + 1.3 i k - 0.4.
        "generator_coefficients": {2: 0.7, 1: 1.3, 0: -0.4},
        "datum": "bernoulli_bandlimited",
        # Band edge K_g = 128: the largest bandwidth demand of any convergent
        # extension in the stage-1 P5 table (gamma = 1e-6 for the split
        # {d_xx}); it keeps every extension evaluation a cheap finite sum.
        "truncation_wavenumber": 128,
        "terminal_time": 1.0,
        # The full datum's break point x^star = 0 becomes, after truncation,
        # the point of maximal oscillation concentration; the corner-window
        # metrics are centred there.
        "corner_point": 0.0,
        "variant_set": "generator",
        # Short display name for figure tick labels. It must be the name the
        # report uses in its prose -- a reader who reads "on the Black-Scholes
        # generator" has to find that string on the axis, not "g2".
        "short_label": r"$G_1$",
        "label": (
            r"$g(x)=\sum_{k=1}^{128}\frac{\cos(kx)}{\pi^2k^2}$,  "
            r"$A=0.7\,\partial_{xx}+1.3\,\partial_x-0.4$ (G1),  "
            r"$Pu=\partial_t u+Au$,  $T=1$"
        ),
    },
    "g2_bernoulli_bandlimited": {
        # Stage-1 generator G2 (Black–Scholes log-price, sigma = 0.5,
        # r = 0.03): A = 0.125 d_xx - 0.095 d_x - 0.03.
        "generator_coefficients": {2: 0.125, 1: -0.095, 0: -0.03},
        "datum": "bernoulli_bandlimited",
        "truncation_wavenumber": 128,
        "terminal_time": 1.0,
        "corner_point": 0.0,
        "variant_set": "generator",
        "short_label": r"$G_2$, Black–Scholes",
        "label": (
            r"$g(x)=\sum_{k=1}^{128}\frac{\cos(kx)}{\pi^2k^2}$,  "
            r"$A=0.125\,\partial_{xx}-0.095\,\partial_x-0.03$ "
            r"(G2, Black–Scholes log-price, $\sigma=0.5$, $r=0.03$),  $T=1$"
        ),
    },
    "g3_bernoulli_bandlimited": {
        # Fourth-order generator (added 2026-09-29, the report's analytical
        # G3): A = -0.05 d_x^4 + 1.3 d_x - 0.4, symbol
        # a(k) = -0.05 k^4 + 1.3 i k - 0.4.  Same datum as G1/G2.
        "generator_coefficients": {4: -0.05, 1: 1.3, 0: -0.4},
        "datum": "bernoulli_bandlimited",
        "truncation_wavenumber": 128,
        "terminal_time": 1.0,
        "corner_point": 0.0,
        "variant_set": "fourth_order",
        "short_label": r"$G_3$, fourth order",
        "label": (
            r"$g(x)=\sum_{k=1}^{128}\frac{\cos(kx)}{\pi^2k^2}$,  "
            r"$A=-0.05\,\partial_x^4+1.3\,\partial_x-0.4$ (G3, fourth order),  "
            r"$Pu=\partial_t u+Au$,  $T=1$"
        ),
    },
    **{
        f"lv2_bernoulli_bandlimited_{_amplitude_tag(ratio)}": _local_volatility_cell(ratio)
        for ratio in VARIABLE_COEFFICIENT_AMPLITUDE_RATIOS
    },
    **{
        f"lv4_bernoulli_bandlimited_{_amplitude_tag(ratio)}": _variable_biharmonic_cell(ratio)
        for ratio in VARIABLE_COEFFICIENT_AMPLITUDE_RATIOS
    },
    "heat_sine_single_component": {
        # Control cell: pure heat at the G2 diffusivity with the
        # single-spectral-component datum g(x) = sin x (k_0 = 1); exact
        # solution u^star(x, t) = e^{-nu (T - t)} sin x.  The pure-heat
        # generator is confined to this cell because under pure heat the
        # split {d_xx} extension IS the exact solution (no non-trivial split
        # comparison exists).
        "generator_coefficients": {2: 0.125},
        "datum": "sine_single_component",
        "sine_wavenumber": 1,
        "sine_amplitude": 1.0,
        "terminal_time": 1.0,
        "corner_point": 0.0,
        "variant_set": "control",
        "short_label": r"$G_0$, heat (control)",
        "label": (
            r"$g(x)=\sin x$,  $A=0.125\,\partial_{xx}$,  "
            r"$u^\star(x,t)=e^{-0.125(T-t)}\sin x$,  $T=1$"
        ),
    },
}


# ---------------------------------------------------------------------------
# Default optimisation hyperparameters (specification Section 4, pinned to
# the ansatz_forms_cross_seed_summary reference)
# ---------------------------------------------------------------------------

DEFAULT_HPARAMS: dict = {
    # ResNet backbone (reference values; d_in = 3 through the periodic
    # feature map (x, t) -> (cos x, sin x, 2 t / T - 1), specification
    # Section 2 and decision D5 — predicted parameter count 33601, to be
    # confirmed against the run log's measured count).
    "net_width": 64,
    "net_blocks": 4,
    "net_layers_per_block": 2,
    # Optimisation: Adam + cosine annealing over the full budget.
    "learning_rate": 1e-3,
    "num_iterations": 20000,
    # Collocation.  n_boundary = 0 deviates from the reference value 256
    # (specification decision D4): the periodic feature map makes the
    # lateral identification exact, so the boundary sampler and the
    # boundary-drift diagnostic are removed.
    "n_interior": 4096,
    "n_terminal": 1024,  # terminal points (diagnostic only for hard forms)
    "n_boundary": 0,
}


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

def cell_names() -> list[str]:
    """Return the available cell identifiers."""
    return list(CELL_CONFIGS.keys())


def cell_short_label(name: str) -> str:
    """The configuration's short display name, for a figure tick label.

    The report names its configurations in prose ("on the Black-Scholes
    generator"); a figure axis that says "g2" instead leaves the reader unable to
    locate the value the prose quotes. This is the single source of that name.
    """
    return cell_by_name(name)["short_label"]


def cell_by_name(name: str) -> dict:
    """Return the cell configuration for ``name`` (raises if unknown)."""
    if name not in CELL_CONFIGS:
        raise KeyError(f"Unknown cell {name!r}. Available: {cell_names()}")
    return CELL_CONFIGS[name]


def variants_for_cell(cell_name: str) -> list[dict]:
    """Return the variant list of a cell (7 generator / 2 control entries)."""
    cell_conf = cell_by_name(cell_name)
    if cell_conf["variant_set"] == "generator":
        return list(GENERATOR_CELL_VARIANTS)
    if cell_conf["variant_set"] == "fourth_order":
        return list(FOURTH_ORDER_CELL_VARIANTS)
    if cell_conf["variant_set"] == "variable_coefficient":
        return list(VARIABLE_COEFFICIENT_CELL_VARIANTS)
    return list(CONTROL_CELL_VARIANTS)


def variant_names(cell_name: str) -> list[str]:
    """Return the variant names of a cell, in catalogue order."""
    return [v["name"] for v in variants_for_cell(cell_name)]


def all_variant_names() -> list[str]:
    """Return every variant name across cells (unique, catalogue order)."""
    return [v["name"] for v in METHOD_VARIANTS]


def variant_by_name(cell_name: str, variant_name: str) -> dict:
    """Return the variant dict of a cell (raises if unknown for that cell)."""
    for variant in variants_for_cell(cell_name):
        if variant["name"] == variant_name:
            return variant
    raise KeyError(
        f"Unknown variant {variant_name!r} for cell {cell_name!r}. "
        f"Available: {variant_names(cell_name)}"
    )
