"""Torch-free catalogue for the four-form / three-IC ansatz study.

This module is imported on cluster login nodes during the ``--init-only`` phase
of the array launcher, so it must remain free of any heavy import (no ``torch``,
no ``numpy``): it only defines plain-Python configuration dictionaries.

The study compares four terminal-condition enforcement *forms* on three
backward-heat *initial (terminal) conditions*:

Forms (all trained -> solid stroke per the repository plot convention):
    * ``hard_constant`` -- u = (1 - lambda) Phi + g          (eq:bermudan-ansatz)
    * ``hard_convex``  -- u = (1 - lambda) Phi + lambda g    (eq:bermudan-ansatz-alt)
    * ``soft_pinn``     -- u = Phi, terminal mismatch penalised in the loss
    * ``pure_nn``       -- u = Phi, no terminal handling (non-identifiable control)

The ``linear`` / ``exponential`` interpolation-coefficient sweep applies only to the two hard
forms; the soft forms carry ``interpolation = None``.

Initial (terminal) conditions, all under the pure heat operator
``P u = d_t u + (sigma^2 / 2) d_xx u``:
    * ``sine``   -- g(x) = sin(pi x) + c sin(f pi x) on [0, 1]      (Dirichlet)
    * ``theta3`` -- g(x) = 1 + 2 sum e^{-n^2} cos(pi n x) on [-1, 1] (cosine series)
    * ``call``   -- smoothed call payoff in x = ln S on [ln 60, ln 160]

The runner must assert ``RUNNER_SCRIPT_STEM == Path(__file__).stem`` so the
output-folder-from-filename invariant cannot silently drift.
"""
from __future__ import annotations

import math

# The runner script's filename stem (without extension).  Asserted by the runner.
RUNNER_SCRIPT_STEM = "ablation_ansatz_forms"


# ---------------------------------------------------------------------------
# Method variants (identical across the three initial conditions)
# ---------------------------------------------------------------------------

METHOD_VARIANTS: list[dict] = [
    {
        "name": "hard_constant_linear",
        "form": "hard_constant",
        "interpolation": "linear",
        "color": "#1f77b4",  # blue
        "label": r"hard, $\Psi=g$, linear $\lambda$",
    },
    {
        "name": "hard_constant_exp",
        "form": "hard_constant",
        "interpolation": "exponential",
        "color": "#17becf",  # cyan
        "label": r"hard, $\Psi=g$, exp.\ $\lambda$",
    },
    {
        "name": "hard_convex_linear",
        "form": "hard_convex",
        "interpolation": "linear",
        "color": "#2ca02c",  # green
        "label": r"hard, $\Psi=\lambda g$, linear $\lambda$",
    },
    {
        "name": "hard_convex_exp",
        "form": "hard_convex",
        "interpolation": "exponential",
        "color": "#8c564b",  # brown
        "label": r"hard, $\Psi=\lambda g$, exp.\ $\lambda$",
    },
    {
        # Matched-diffusion Gaussian-semigroup (split) extension, mirroring the
        # split_matched row of _bermudan_extension_catalogue.  The stage datum is the
        # exact glued maximum; the extension is the Gaussian convolution of that datum
        # at the matched comparison volatility (nu_c = nu), so the extension forcing
        # P Psi collapses to the bounded first-order defect part.  The analytic
        # extension derivatives are threaded into the ansatz, so P Psi is assembled
        # without the catastrophic cancellation of the autograd route.  Validated
        # against the exact Black--Scholes Bermudan reference (rate R), not the heat
        # reference.  Trained -> solid stroke per the repository plot convention.
        "name": "split_matched",
        "form": "hard_constant",
        "interpolation": "linear",
        "datum": "exact_maximum",
        "extension": "gaussian_semigroup",
        "comparison_volatility_ratio": 1.0,
        "grading_exponent": None,
        "analytic_derivatives": True,
        "fair_floor": True,
        "color": "#1f77b4",  # blue
        "label": r"split, matched $\nu_c=\nu$ (Gaussian semigroup)",
    },
    {
        # Black--Scholes convex baseline, mirroring the convex_exact_datum row of
        # _bermudan_extension_catalogue.  The stage datum is the exact glued maximum
        # max(g, C), imposed exactly by the convex-combination form (no smoothing
        # bias at the slice), but with NO interior extension of its own: the datum is
        # its own extension, weighted by the interpolation coefficient lambda(t).  The
        # extension forcing P Psi therefore contains a Dirac mass at the free boundary
        # and the exact minimiser the network must represent has a first-derivative
        # discontinuity whose amplitude diverges as the slice is approached (unbounded
        # target).  This is the baseline the matched split is designed to beat, on the
        # SAME Black--Scholes generator (rate R) as split_matched; validated against the
        # exact Black--Scholes Bermudan reference.  Trained -> solid stroke per the
        # repository plot convention.
        "name": "convex_exact_datum",
        "form": "hard_convex",
        "interpolation": "linear",
        "datum": "exact_maximum",
        "extension": None,
        "comparison_volatility_ratio": None,
        "grading_exponent": None,
        "analytic_derivatives": False,
        "fair_floor": False,
        "color": "#d62728",  # red
        "label": r"convex, exact datum $\max(g,C)$ (unbounded target)",
    },
    {
        "name": "soft_pinn",
        "form": "soft_pinn",
        "interpolation": None,
        "color": "#ff7f0e",  # orange
        "label": r"soft PINN (terminal penalty)",
    },
    {
        "name": "pure_nn",
        "form": "pure_nn",
        "interpolation": None,
        "color": "#d62728",  # red
        "label": r"pure NN (no terminal handling)",
    },
]


# ---------------------------------------------------------------------------
# Initial-condition (problem) configurations
# ---------------------------------------------------------------------------

IC_CONFIGS: dict[str, dict] = {
    # The PDE is posed on all of R; ``x_lo``/``x_hi`` only delimit the support of
    # the Monte-Carlo sampling measure for the residual + terminal terms (no
    # lateral boundary condition is imposed).  Accuracy metrics are reported on
    # the inner evaluation window ``x_eval_lo``/``x_eval_hi``, leaving a buffer of
    # a few diffusion lengths so the edge under-determination does not pollute
    # them.  For the periodic ICs the window is a natural period cell, so the
    # eval window can be the full cell.
    "sine": {
        "reference": "sine",
        "sigma": 1.0,
        "T": 0.1,
        "x_lo": 0.0,
        "x_hi": 1.0,
        "x_eval_lo": 0.0,
        "x_eval_hi": 1.0,
        # Terminal datum g(x) = sin(pi x) + c sin(f pi x); integer f keeps the
        # exact solution homogeneous-Dirichlet (u = 0) on the period boundary.
        "params": {"c": 0.5, "f": 4.0},
        "label": (
            r"$g(x)=\sin(\pi x)+0.5\,\sin(4\pi x)$, "
            r"$\mathcal{P}u=\partial_t u+\frac{\sigma^2}{2}\partial_{xx}u$, $\sigma=1$"
        ),
    },
    "theta3": {
        "reference": "theta3",
        "sigma": 1.0,
        "T": 0.1,
        "x_lo": -1.0,
        "x_hi": 1.0,
        "x_eval_lo": -1.0,
        "x_eval_hi": 1.0,
        "params": {"n_modes": 6},
        "label": (
            r"$g(x)=\vartheta_3(x/2,e^{-1})=1+2\sum_{n\geq1}e^{-n^2}\cos(\pi n x)$, "
            r"$\sigma=1$"
        ),
    },
    "call": {
        "reference": "call",
        "sigma": 0.25,
        "T": 1.0,
        # Sampling window wider than the evaluation window (genuine truncation
        # of R): buffer ~ 4 sigma sqrt(T) = 1.0 in log-price on the left.
        "x_lo": math.log(20.0),
        "x_hi": math.log(200.0),
        "x_eval_lo": math.log(60.0),
        "x_eval_hi": math.log(140.0),
        "K": 100.0,
        # beta controls the softplus smoothing of the (e^x - K)^+ payoff.
        "params": {"beta": 100.0},
        "label": (
            r"$g_\beta(x)=\mathrm{softplus}_\beta(e^x-K)-\frac{\log 2}{\beta}$, "
            r"$K=100$, $\beta=100$, $\sigma=0.25$"
        ),
    },
    "call_cm": {
        # Same call problem, but the hard-form extension is a time-dependent
        # one-sided Chen--Mangasarian smoothing with vanishing bandwidth
        # eps(t)=eps0 (T-t)/T: exact payoff at t=T (no terminal bias) and a
        # smooth, bounded-residual extension for t<T.  Compare against "call"
        # (static softplus) to separate terminal-bias from the forcing-floor.
        "reference": "call_cm",
        "sigma": 0.25,
        "T": 1.0,
        "x_lo": math.log(20.0),
        "x_hi": math.log(200.0),
        "x_eval_lo": math.log(60.0),
        "x_eval_hi": math.log(140.0),
        "K": 100.0,
        "params": {"eps0": 10.0},
        "label": (
            r"$\Psi(x,t)=\frac{1}{2}[(e^x-K)+\sqrt{(e^x-K)^2+\varepsilon(t)^2}]$, "
            r"$\varepsilon(t)=\varepsilon_0\frac{T-t}{T}$, $\varepsilon_0=10$, "
            r"$K=100$, $\sigma=0.25$"
        ),
    },
    "bermudan_put": {
        # One backward-induction step of a Bermudan put: stage [0, t1] under the
        # pure heat operator, terminal datum at t1 the Chen--Mangasarian-smoothed
        # gluing of the exercise payoff (K-e^x)^+ and the European continuation
        # value C(x) (analytic post-exercise stage [t1, T_option]).  The exact
        # reference is the Gaussian convolution of that datum (no binomial tree).
        # The framework "T" is the stage terminal = exercise date t1.
        "reference": "bermudan_put",
        "sigma": 0.25,
        "T": 0.5,
        "x_lo": math.log(20.0),
        "x_hi": math.log(200.0),
        "x_eval_lo": math.log(60.0),
        "x_eval_hi": math.log(140.0),
        "K": 100.0,
        "params": {"T_option": 1.0, "eps": 2.0},
        "label": (
            r"Bermudan put, stage $[0,t_1]$: $g(x)=M_\varepsilon((K-e^x)^+,\,C(x))$, "
            r"$C=$ European put at $t_1$; $K=100$, $\sigma=0.25$, $t_1=0.5$, "
            r"$T=1$, $\varepsilon=2$"
        ),
    },
}


# ---------------------------------------------------------------------------
# Default optimisation hyperparameters (shared across forms for fair comparison)
# ---------------------------------------------------------------------------

DEFAULT_HPARAMS: dict = {
    # ResNet backbone
    "net_width": 64,
    "net_blocks": 4,
    "net_layers_per_block": 2,
    # Optimisation
    "learning_rate": 1e-3,
    "num_iterations": 20000,
    # Collocation
    "n_interior": 4096,  # PDE residual collocation points per step
    "n_terminal": 1024,  # terminal-condition points (soft form / diagnostics)
    "n_boundary": 256,  # spatial-boundary points (Dirichlet enforcement)
    # Loss weight.  For the hard forms and the pure-NN control only the PDE
    # residual is active.  For the soft form the total is
    # a * L_pde + (1 - a) * L_tc  (the note's weighted convention).  No
    # spatial-boundary term enters the loss; boundary drift is monitored as a
    # diagnostic only.
    "soft_pde_weight_a": 0.5,
}


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

def ic_names() -> list[str]:
    """Return the available initial-condition identifiers."""
    return list(IC_CONFIGS.keys())


def variant_names() -> list[str]:
    """Return the available method-variant names."""
    return [v["name"] for v in METHOD_VARIANTS]


# Variants kept in the data / summary but omitted from comparison plots: the
# pure-NN control sits at rel L2 ~ 1 (non-identifiable, by design) and would
# compress the log scale of the informative forms.  Its numbers stay in
# summary.yaml / metrics; only the figures drop it.
PLOT_EXCLUDE = ("pure_nn",)


def plotted_variant_names() -> list[str]:
    """Method-variant names shown in comparison plots (excludes PLOT_EXCLUDE)."""
    return [v["name"] for v in METHOD_VARIANTS if v["name"] not in PLOT_EXCLUDE]


def variant_by_name(name: str) -> dict:
    """Return the method-variant dict for ``name`` (raises if unknown)."""
    for v in METHOD_VARIANTS:
        if v["name"] == name:
            return v
    raise KeyError(f"Unknown variant {name!r}. Available: {variant_names()}")


def ic_by_name(name: str) -> dict:
    """Return the IC config dict for ``name`` (raises if unknown)."""
    if name not in IC_CONFIGS:
        raise KeyError(f"Unknown IC {name!r}. Available: {ic_names()}")
    return IC_CONFIGS[name]


def array_tasks(ic_filter: list[str] | None = None) -> list[dict]:
    """Enumerate ``(ic, variant)`` task descriptors for the job array.

    Each descriptor is ``{"ic": <ic_name>, "variant": <variant_name>}``; the
    seed axis is applied on top by the launcher (one array per seed, or a seed
    column).  Pass ``ic_filter`` to restrict to a subset of ICs.

    Returns:
        A list of task descriptors, ICs in catalogue order, variants nested.
    """
    ics = ic_filter if ic_filter is not None else ic_names()
    tasks: list[dict] = []
    for ic in ics:
        if ic not in IC_CONFIGS:
            raise KeyError(f"Unknown IC {ic!r}. Available: {ic_names()}")
        for v in METHOD_VARIANTS:
            tasks.append({"ic": ic, "variant": v["name"]})
    return tasks
