r"""Torch-free catalogue for the Bermudan free-boundary extension ablation.

The ablation isolates the choice of **terminal-data extension** at the moving free
boundary, in the smallest Bermudan chain in which a free boundary exists
(:math:`M = 2` exercise dates).  The post-exercise stage is reduced to the
closed-form Black--Scholes European put, so the exact continuation value
:math:`C^{\star}_{1}`, the free boundary
:math:`\Gamma_{1} = \{\payoff = C^{\star}_{1}\}` and the jump :math:`J_{1}` of the
stage datum's first derivative there are all **exactly known**; the single trained
stage is :math:`[0, t_{1}]`, whose terminal datum is
:math:`V_{1} = \max(\payoff, C^{\star}_{1})`.

Every variant shares the trial-solution family, the architecture, the optimiser,
the collocation budget and the seed axis (shared seeding: identical initialisation
and sampler trajectory across variants, so a difference between variants reflects
the intervention and not the random-number generator).  The variants differ **only**
in the extension, and the ablation axis is therefore the extension alone.

The five variants are the rows of the paper's ledger.  What each is predicted to
do, and by which proposition:

===============================  ===========  ===================  ==================
Variant                          Slice bias   ``sup|L h|``, s -> 0  Target ``Psi*``
===============================  ===========  ===================  ==================
``convex_constant_smoothed``     eps_0 / 2    bounded              bounded
``convex_exact_datum``           0            +infinity            **unbounded**
``split_matched``                **0**        **bounded**          **bounded**
``split_mismatched``             0            O(s^{-1/2})          unbounded
``graded_mollifier_linear``      0            O(s^{-1})            unbounded
===============================  ===========  ===================  ==================

Two readings the ablation is designed to test, and which are *not* obvious:

* **exactness at the slice does not discriminate.**  Four of the five variants
  impose the datum exactly and therefore have zero slice bias; the property rests
  only on a semigroup at parameter zero being the identity, which every semigroup
  has.  A measured slice bias of zero for ``split_matched`` confirms nothing that
  ``split_mismatched`` does not also confirm.
* **nor does square-integrability of the forcing.**  The mis-specified split has a
  finite forcing floor.  The property that singles out the matched split is
  **boundedness** --- membership of every :math:`\mathrm{L}^{q}`, hence a
  finite-variance Monte-Carlo estimator of the objective, hence a bounded target
  for the network.  The discriminating measurement is therefore the growth of
  :math:`\sup|\mathcal{L} h|` as the slice is approached, and the seed dispersion
  of the objective --- not the floor.

Two variants are *unfair* on any measured floor and are labelled as such wherever
a comparison is drawn.  ``convex_exact_datum``'s forcing contains a Dirac mass at
the free boundary: it is not square-integrable, yet the divergence is supported on
a null set, so a collocation estimator never samples it and the *observed* loss is
finite, stable, and meaningless as a floor --- its real defect is the unbounded
target.  ``graded_mollifier_linear``'s forcing is genuinely non-integrable and its
estimator has infinite variance, so its reported value is a function of the
collocation draw and the seed dispersion is the actual signal.

The generator is the **genuine Black--Scholes** operator in log-price coordinates,
with a strictly positive rate.  This is not incidental: on the pure-heat generator
used by the companion induction experiments the split
:math:`\mathcal{L}^{X} = \mathcal{A} + \mathcal{B}` has defect
:math:`\mathcal{B} = 0`, the matched split extension coincides with the exact stage
solution, and the ablation would be degenerate.
"""

from __future__ import annotations

# The runner script's filename stem (without extension).  Asserted by the runner so
# that the two cannot drift: the output folder is derived from the runner's own
# filename, and this catalogue is imported on login nodes where torch is absent.
RUNNER_SCRIPT_STEM = "bermudan_free_boundary_extension_ablation"

# ---------------------------------------------------------------------------
# Problem geometry.  Fixed; not an ablation axis.
# ---------------------------------------------------------------------------

#: Strike.
STRIKE = 100.0
#: Volatility.
VOLATILITY = 0.25
#: Risk-free rate.  STRICTLY POSITIVE: with r = 0 the defect part of the split is
#: B = -nu d_x, and with the *pure heat* generator it would be zero outright, which
#: would collapse the matched split onto the exact solution.  A positive rate also
#: makes the put's exercise region non-empty, hence the free boundary exist.
RISK_FREE_RATE = 0.05
#: Maturity.
MATURITY = 1.0
#: The single intermediate exercise date; the trained stage is [0, t_1].
INTERMEDIATE_EXERCISE_DATE = 0.5

#: Log-price training window (spot 20 to 200), matching the companion chains.
LOG_PRICE_LO, LOG_PRICE_HI = 2.995732273553991, 5.298317366548036  # ln 20 .. ln 200
#: Evaluation window (where every reported error is measured): spot 60 to 140.
EVALUATION_LO, EVALUATION_HI = 4.0943445622221, 4.941642422609304  # ln 60 .. ln 140
#: Quadrature support for the Gaussian convolutions.  It must cover the training
#: window padded by at least ``5 * sigma * sqrt(Delta)`` (here 0.884), and the datum
#: must be supplied there by its ANALYTIC far-field values -- never by zero-padding
#: (the put datum tends to the strike, not to zero, as x -> -infinity, so zero-padding
#: inserts a spurious jump of size K at the window edge whose convolution is a
#: first-order source diverging like s^{-1/2}) and never by extrapolating the trained
#: network.  Both the payoff and the exact European continuation are closed-form on the
#: whole line, so the analytic extension is available here at no cost.
QUADRATURE_LO, QUADRATURE_HI = -2.995732273553991, 6.907755278982137  # ln 0.05 .. ln 1000

#: Constant smoothing scale of the received baseline, matching the measured chains.
CONSTANT_SMOOTHING_SCALE = 2.0

#: Quadrature nodes for the extension convolutions.
EXTENSION_QUADRATURE_NODES = 6000
#: Quadrature nodes for the exact Bermudan reference (accuracy, not speed).
REFERENCE_QUADRATURE_NODES = 20000

#: Width of the terminal sliver excluded from the interior collocation sampler.
#:
#: The extension's fixed-grid quadrature cannot resolve the Gaussian kernel when its
#: standard deviation ``sigma_c * sqrt(T - t)`` falls below a few grid spacings, i.e.
#: for ``T - t`` below ``tau_floor = (5 * dy / sigma_c)^2``.  Sampling the residual
#: there would silently return the un-smoothed (kinked) datum and the second-derivative
#: channel of the residual would be wrong.
#:
#: The exclusion is a SINGLE constant, applied to EVERY variant identically -- including
#: those with no semigroup extension, whose own floor is zero.  Were each variant to
#: exclude only its own floor, the samplers would differ across the ablation axis and
#: the comparison would be unfair: the mis-specified split (whose smaller ``sigma_c``
#: gives the largest floor) would be the only one spared the sliver where the graded
#: mollifier's forcing diverges.  The runner asserts that this constant dominates every
#: variant's own floor.
#:
#: Consequence, and it is intended: the measured forcing floor of the linear-graded
#: mollifier is a function of this width (its true floor is infinite), so that column is
#: a diagnostic of the sampler and is labelled as such wherever it is compared.
EXCLUDED_TERMINAL_SLIVER = 2.5e-3

#: Half-widths of the corner windows on which the localised diagnostics are reported.
CORNER_WINDOW_HALF_WIDTHS = (0.05, 0.1, 0.2)

#: Smoke-test guard: below this iteration count the run must carry --debug.
SMOKE_TEST_NUM_ITERATIONS_THRESHOLD = 700


# ---------------------------------------------------------------------------
# The ablation axis: the extension
# ---------------------------------------------------------------------------

#: Variant keys.
#:   ``form``                        trial-solution branch of ``TerminalAnsatz``.
#:   ``interpolation``               kind of the interpolation coefficient.
#:   ``datum``                       how the stage terminal datum is glued:
#:                                   ``chen_mangasarian_constant`` (a bias at the
#:                                   slice) or ``exact_maximum`` (no bias).
#:   ``extension``                   ``None`` (the datum is its own extension,
#:                                   weighted by the interpolation coefficient),
#:                                   ``gaussian_semigroup`` or
#:                                   ``graded_chen_mangasarian``.
#:   ``comparison_volatility_ratio`` sigma_c / sigma for the semigroup extension;
#:                                   1.0 is matched, 1/sqrt(2) gives nu_c = nu / 2.
#:   ``grading_exponent``            q of eps(t) = eps_0 ((T-t)/T)^q.
#:   ``analytic_derivatives``        whether the extension supplies closed-form
#:                                   dt/dx/dxx, so that L h is assembled without the
#:                                   catastrophic cancellation of the autograd route.
#:   ``fair_floor``                  False when the measured forcing floor is NOT a
#:                                   property of the method (see the module
#:                                   docstring); such a floor is reported but must be
#:                                   labelled wherever it is compared.
METHOD_VARIANTS: list[dict] = [
    {
        # (a) The received baseline, matching the measured chains: the datum is the
        # constant-scale smoothed max, so the max is NOT imposed at the slice and the
        # error recursion charges a bias in [0, eps_0/2] at the exercise date.
        "name": "convex_constant_smoothed",
        "form": "hard_convex",
        "interpolation": "linear",
        "datum": "chen_mangasarian_constant",
        "extension": None,
        "comparison_volatility_ratio": None,
        "grading_exponent": None,
        "analytic_derivatives": False,
        "fair_floor": True,
        "color": "#2ca02c",
        "label": r"convex, constant smoothing ($\varepsilon_0=2$)",
    },
    {
        # (b) The exact datum with no interior profile of its own: the max is imposed
        # exactly (no slice bias), but L h contains a Dirac at the free boundary and
        # the exact minimiser the network must represent has a first-derivative
        # discontinuity whose amplitude DIVERGES as the slice is approached.
        "name": "convex_exact_datum",
        "form": "hard_convex",
        "interpolation": "linear",
        "datum": "exact_maximum",
        "extension": None,
        "comparison_volatility_ratio": None,
        "grading_exponent": None,
        "analytic_derivatives": False,
        "fair_floor": False,
        "color": "#d62728",
        "label": r"convex, exact datum $\max(g,C)$ (unbounded target)",
    },
    {
        # (c) THE PROPOSITION: exact at the slice, bounded forcing, bounded target.
        "name": "split_matched",
        "form": "hard_constant",
        "interpolation": "linear",
        "datum": "exact_maximum",
        "extension": "gaussian_semigroup",
        "comparison_volatility_ratio": 1.0,
        "grading_exponent": None,
        "analytic_derivatives": True,
        "fair_floor": True,
        "color": "#1f77b4",
        "label": r"split, matched $\nu_c=\nu$",
    },
    {
        # (d) Mis-specified split: still exact at the slice, still square-integrable,
        # but the second-order channel survives and the forcing is UNBOUNDED.
        # sigma_c = sigma / sqrt(2) gives nu_c = nu / 2.
        "name": "split_mismatched",
        "form": "hard_constant",
        "interpolation": "linear",
        "datum": "exact_maximum",
        "extension": "gaussian_semigroup",
        "comparison_volatility_ratio": 0.7071067811865476,
        "grading_exponent": None,
        "analytic_derivatives": True,
        "fair_floor": True,
        "color": "#ff7f0e",
        "label": r"split, mis-specified $\nu_c=\nu/2$",
    },
    {
        # (e) Linear-graded mollifier: exact at the slice, and the marginal
        # logarithmically-divergent member of the graded family. Its estimator has
        # infinite variance, so its measured floor is a diagnostic of the sampler.
        "name": "graded_mollifier_linear",
        "form": "hard_constant",
        "interpolation": "linear",
        "datum": "exact_maximum",
        "extension": "graded_chen_mangasarian",
        "comparison_volatility_ratio": None,
        "grading_exponent": 1.0,
        "analytic_derivatives": False,
        "fair_floor": False,
        "color": "#9467bd",
        "label": r"graded mollifier, $q=1$ (divergent forcing)",
    },
]


DEFAULT_HPARAMS: dict = {
    # ResNet backbone (matching the companion Bermudan chains).
    "net_width": 64,
    "net_blocks": 4,
    "net_layers_per_block": 2,
    # Optimisation.
    "learning_rate": 1e-3,
    "num_iterations": 20000,
    # Collocation.
    "n_interior": 4096,
}

#: The seed axis, shared across variants.
SEEDS = (0, 1, 2)


def variant_names() -> list[str]:
    """The available variant identifiers."""
    return [v["name"] for v in METHOD_VARIANTS]


def variant_by_name(name: str) -> dict:
    """Look a variant up by name."""
    for v in METHOD_VARIANTS:
        if v["name"] == name:
            return v
    raise KeyError(f"Unknown variant {name!r}. Available: {variant_names()}")


def array_tasks() -> list[dict]:
    """Enumerate the ``(variant, seed)`` task descriptors for the job array."""
    return [
        {"variant": v["name"], "seed": seed}
        for v in METHOD_VARIANTS
        for seed in SEEDS
    ]
