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
(graded variants only), ``exponential_rate_gamma`` (control cell only),
``color`` and ``label``.

The runner must assert ``RUNNER_SCRIPT_STEM == Path(__file__).stem`` so the
output-folder-from-filename invariant cannot silently drift.
"""
from __future__ import annotations

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
        "exponential_rate_gamma": None,
        "color": "#d62728",  # red
        "label": r"split $\{\partial_{xx}\}$: $P\Psi=\mu\,\partial_x\Psi+r_0\Psi$",
    },
    {
        # V4 — split semigroup extension, subset {d_xx, d_x}; the forcing
        # satisfies P Psi = r_0 Psi (defect order 0).
        "name": "split_diffusion_advection",
        "form": "hard_constant",
        "interpolation": "linear",
        "extension": "split_diffusion_advection",
        "comparison_diffusivity_ratio": None,
        "exponential_rate_gamma": None,
        "color": "#ff7f0e",  # orange
        "label": r"split $\{\partial_{xx},\partial_x\}$: $P\Psi=r_0\Psi$",
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
        "exponential_rate_gamma": None,
        "color": "#7f7f7f",  # grey
        "label": r"exact solution: $\Psi=u^\star$, $P\Psi=0$",
    },
]

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
        "label": (
            r"$g(x)=\sum_{k=1}^{128}\frac{\cos(kx)}{\pi^2k^2}$,  "
            r"$A=0.125\,\partial_{xx}-0.095\,\partial_x-0.03$ "
            r"(G2, Black–Scholes log-price, $\sigma=0.5$, $r=0.03$),  $T=1$"
        ),
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
