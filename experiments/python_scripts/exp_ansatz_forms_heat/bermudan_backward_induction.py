r"""Learned Bermudan-put backward induction with the terminal-condition ansatz.

This validates the *procedure*, not a single PDE solve: it chains ``m`` learned
stages through the dynamic program and measures how the per-stage learning error
**propagates** down the induction toward the inception price.

Geometry.  Exercise dates :math:`t_1 < \dots < t_m = T` (the last is maturity),
valuation at :math:`t_0 = 0`.  Between consecutive dates the value solves the pure
heat equation, so each interval :math:`I_k = [t_k, t_{k+1}]` is one learned stage
(local time :math:`s \in [0, \Delta_k]`, terminal datum at :math:`s = \Delta_k`).

Induction (top-down, so the stage above is already trained when its value is
needed):

* **stage** :math:`k = m-1` (interval ending at maturity): terminal datum is the
  Chen--Mangasarian-smoothed payoff :math:`M_\varepsilon((K-e^x)^+, 0)`;
* **stage** :math:`k < m-1`: terminal datum is
  :math:`M_\varepsilon\bigl((K-e^x)^+,\ C_{k+1}(x)\bigr)`, where the continuation
  :math:`C_{k+1}(x) = \hat u_{k+1}(x, s{=}0)` is the **learned** value at the
  bottom of the interval above (a frozen network, differentiable in :math:`x` so
  the hard-form forcing :math:`\mathcal{P}\Psi` is exact).

Each stage is trained with the chosen :class:`TerminalAnsatz` form via the shared
``train_variant`` of ``ablation_ansatz_forms``.  After all stages are trained, the
learned value at every global time :math:`t_k` is compared against the exact
multi-stage reference :func:`bermudan_put_value_exact` (chained Gaussian
convolution, exact max-gluing); the relative :math:`L^2` error per :math:`t_k`,
from maturity down to inception, is the error-propagation curve.

Usage::

    # smoke (flagged exploratory)
    python bermudan_backward_induction.py --m 2 --num-iterations 300 --debug
    # real run
    python bermudan_backward_induction.py --m 2 --variant hard_convex_linear --num-iterations 20000
    # redraw figures from saved artefacts
    python bermudan_backward_induction.py --replot data/bermudan_backward_induction/<run>
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ansatz_forms_catalogue as cat  # noqa: E402
# Torch-free split-extension catalogue: reused for the matched-diffusion split
# variant (wide quadrature support, node count, excluded terminal sliver and the
# strictly-positive Black--Scholes risk-free rate).  See _make_split_stage_problem.
import _bermudan_extension_catalogue as bext_cat  # noqa: E402

logger = logging.getLogger("bermudan_induction")

# Below this iteration count a run must be flagged --debug (mechanical smoke-test
# guard, ~3% of the 20k real-run budget); bypassed only for --replot.
SMOKE_TEST_NUM_ITERATIONS_THRESHOLD = 700

# Fixed problem geometry (the Bermudan put of the ansatz-form study).
K = 100.0
SIGMA = 0.25
MATURITY = 1.0
EPS = 2.0
X_LO, X_HI = None, None  # set from log-spot domain below
import math  # noqa: E402

X_LO, X_HI = math.log(20.0), math.log(200.0)
X_EVAL_LO, X_EVAL_HI = math.log(60.0), math.log(140.0)
Y_LO, Y_HI = math.log(5.0), math.log(600.0)  # convolution support for the exact ref

# Risk-free rate for the matched-diffusion split variant.  The heat variants use the
# pure backward-heat operator (rate zero); the split is validated against the genuine
# Black--Scholes reference, whose rate is single-sourced from the M=2 split ablation
# catalogue so the two studies share the same generator.
R = bext_cat.RISK_FREE_RATE


# ---------------------------------------------------------------------------
# Coarse exact interpolant (cheap per-iteration boundary diagnostic)
# ---------------------------------------------------------------------------

class CoarseExact2D:
    """Bilinear interpolant of the exact stage solution on a regular (x, s) grid.

    The exact multi-stage reference is expensive (nested convolution); the
    per-iteration training diagnostic only needs a cheap approximation, so we
    sample it once on a coarse grid and bilinearly interpolate.  Real validation
    uses the exact reference directly (post-training).
    """

    def __init__(self, xs, ss, vals):
        import torch
        self.xs, self.ss, self.vals = xs, ss, vals
        self.x0, self.dx, self.nx = float(xs[0]), float(xs[1] - xs[0]), len(xs)
        self.s0 = float(ss[0])
        self.ds = float(ss[1] - ss[0]) if len(ss) > 1 else 1.0
        self.ns = len(ss)
        self._torch = torch

    def __call__(self, x, t):
        # The grid (self.vals) lives on CPU in float64; the query may arrive on a
        # GPU (training collocation).  Evaluate on the grid's device and return on
        # the query's device so this works as a diagnostic under either backend.
        torch = self._torch
        dev, dt = self.vals.device, self.vals.dtype
        xq = x.reshape(-1).to(device=dev, dtype=dt)
        tq = t.reshape(-1).to(device=dev, dtype=dt)
        fx = ((xq - self.x0) / self.dx).clamp(0, self.nx - 1)
        ft = ((tq - self.s0) / self.ds).clamp(0, self.ns - 1)
        i0 = fx.floor().long(); i1 = (i0 + 1).clamp(max=self.nx - 1); wx = fx - i0
        j0 = ft.floor().long(); j1 = (j0 + 1).clamp(max=self.ns - 1); wt = ft - j0
        V = self.vals
        v = ((1 - wx) * (1 - wt) * V[i0, j0] + wx * (1 - wt) * V[i1, j0]
             + (1 - wx) * wt * V[i0, j1] + wx * wt * V[i1, j1])
        return v.reshape(x.shape).to(device=x.device, dtype=x.dtype)


def _build_coarse_exact(exercise_times, t_k, delta_k, *, n_xg=121, n_sg=13, n_quad=1500):
    """Sample the exact reference on a coarse (x, s_local) grid for stage k."""
    import torch
    from learning_option_pricing.pde import bermudan_put_value_exact

    xs = torch.linspace(X_LO, X_HI, n_xg, dtype=torch.float64)
    ss = torch.linspace(0.0, delta_k, n_sg, dtype=torch.float64)
    vals = torch.empty((n_xg, n_sg), dtype=torch.float64)
    for j, s in enumerate(ss):
        t_global = torch.full_like(xs, t_k + float(s))
        vals[:, j] = bermudan_put_value_exact(
            xs, t_global, exercise_times=exercise_times, K=K, sigma=SIGMA,
            y_lo=Y_LO, y_hi=Y_HI, n_quad=n_quad)
    return CoarseExact2D(xs, ss, vals)


# ---------------------------------------------------------------------------
# Stage problem construction
# ---------------------------------------------------------------------------

def _is_split_variant(variant: dict) -> bool:
    """Whether ``variant`` is the matched-diffusion Gaussian-semigroup split.

    The split variant carries the extra catalogue keys mirrored from the M=2
    ablation (``extension == "gaussian_semigroup"``); the four heat forms
    (``hard_convex_linear`` etc.) have no ``extension`` key at all.
    """
    return variant is not None and variant.get("extension") == "gaussian_semigroup"


def _far_field_composite_datum(continuation_fn, K, x_lo, x_hi, *, n_tabulate=4096):
    r"""Composite stage datum valid on the WIDE convolution grid.

    The Gaussian-semigroup extension convolves the stage datum on the wide
    quadrature support :math:`[y_{\mathrm{lo}}, y_{\mathrm{hi}}]`, which extends far
    beyond the training window :math:`[x_{\mathrm{lo}}, x_{\mathrm{hi}}]` on which the
    trained continuation network is valid.  Extrapolating the network onto that
    support is forbidden; the analytic put payoff is used instead outside the
    window.  The composite is

    .. math::

        V(y) =
        \begin{cases}
            \max\bigl((K - e^{y})^{+},\ C_{k}(y)\bigr), & y \in [x_{\mathrm{lo}},\,
                x_{\mathrm{hi}}], \\
            (K - e^{y})^{+}, & \text{otherwise}.
        \end{cases}

    The replacement is exact in both far-field regions: below :math:`x_{\mathrm{lo}}`
    the put is deep in-the-money and lies in the exercise region, where the maximum
    equals the payoff; above :math:`x_{\mathrm{hi}}` the put payoff is zero and the
    continuation is negligible, so the maximum again equals the payoff (zero).  On the
    training window the composite equals the exact glued maximum
    :math:`\max((K - e^{y})^{+}, C_{k}(y))`.  For the top stage the continuation is
    identically zero, so :math:`V(y) = (K - e^{y})^{+}` everywhere.

    Args:
        continuation_fn:  Callable ``C(y) -> Tensor`` (the frozen stage-above
                          network at local time ``s = 0``, or ``zeros`` for the top
                          stage), differentiable in ``y`` on the window.
        K:                Strike.
        x_lo, x_hi:       Training-window bounds delimiting where the continuation is
                          used; outside them the analytic payoff is used.

    Returns:
        A callable ``V(y) -> Tensor`` (torch, differentiable in ``y`` on the window).
    """
    import torch

    from learning_option_pricing.pde import heat_put_payoff

    # The frozen continuation C_k is, for a lower stage, itself a split-extended
    # network whose evaluation triggers the stage-above Gaussian convolution.  The
    # extension convolves this datum on a fixed n_quad grid at every training
    # iteration, so calling C_k directly would nest one convolution inside another
    # (an O(n_quad^2) cost per step).  Because C_k is frozen, it is tabulated once
    # per (device, dtype) on a dense window grid and linearly interpolated
    # thereafter: only its VALUES enter the convolution (the extension's
    # derivatives come from the analytic kernel, not from differentiating the
    # datum), and the tabulation grid is finer than the quadrature grid, so this is
    # exact to quadrature resolution while removing the nesting.
    grid_step = (x_hi - x_lo) / (n_tabulate - 1)
    tabulated: dict = {}

    def _continuation_values(y: "torch.Tensor") -> "torch.Tensor":
        key = (y.device, y.dtype)
        values = tabulated.get(key)
        if values is None:
            grid = torch.linspace(
                x_lo, x_hi, n_tabulate, dtype=y.dtype, device=y.device
            )
            with torch.no_grad():
                values = continuation_fn(grid).reshape(-1)
            tabulated[key] = values
        query = y.reshape(-1)
        position = ((query - x_lo) / grid_step).clamp(0.0, n_tabulate - 1 - 1e-6)
        left_index = position.floor().long()
        fraction = position - left_index.to(y.dtype)
        interpolated = (
            values[left_index] * (1.0 - fraction)
            + values[left_index + 1] * fraction
        )
        return interpolated.reshape(y.shape)

    def datum(y: "torch.Tensor") -> "torch.Tensor":
        payoff = heat_put_payoff(y, K)
        inside_window = (y >= x_lo) & (y <= x_hi)
        glued_on_window = torch.maximum(payoff, _continuation_values(y))
        return torch.where(inside_window, glued_on_window, payoff)

    return datum


def _split_free_boundary_and_jump(continuation_fn):
    r"""Free boundary :math:`x^{\star}` and datum-derivative jump for logging.

    Both are used only in the training log lines of the split ``train_variant``.  A
    poorly trained continuation (a smoke run of a few iterations) may fail to cross
    the payoff on the window; the crossing is then reported as ``nan`` rather than
    aborting the run, since these quantities are diagnostic only.
    """
    import torch

    from learning_option_pricing.pde import (
        bermudan_exercise_boundary, heat_put_payoff)

    try:
        free_boundary = bermudan_exercise_boundary(
            continuation_fn, K=K, x_lo=X_LO, x_hi=X_HI)
    except ValueError:
        return float("nan"), float("nan")

    point = torch.tensor([free_boundary], dtype=torch.float64, requires_grad=True)
    difference = heat_put_payoff(point, K) - continuation_fn(point)
    (slope,) = torch.autograd.grad(difference, point, torch.ones_like(difference))
    return free_boundary, abs(float(slope))


def _make_split_stage_problem(k, tau, delta_k, cont_above_fn, variant):
    """Build the ``train_variant``-compatible problem dict for the split stage k.

    Mirrors :func:`bermudan_free_boundary_extension_ablation.build_problem` for the
    ``split_matched`` variant, but the extension is built from the far-field composite
    datum (:func:`_far_field_composite_datum`) so an arbitrary trained continuation can
    be convolved on the wide quadrature grid without being extrapolated.
    """
    from learning_option_pricing.pde import (
        GaussianSemigroupExtensionField,
        black_scholes_generator_coefficients,
    )

    composite_datum = _far_field_composite_datum(cont_above_fn, K, X_LO, X_HI)

    comparison_volatility = float(variant["comparison_volatility_ratio"]) * SIGMA
    extension_field = GaussianSemigroupExtensionField(
        composite_datum,
        terminal_time=delta_k,
        comparison_volatility=comparison_volatility,
        y_lo=bext_cat.QUADRATURE_LO,
        y_hi=bext_cat.QUADRATURE_HI,
        n_quad=bext_cat.EXTENSION_QUADRATURE_NODES,
        name=f"{variant['name']}_stage{k}",
    )
    own_quadrature_floor = extension_field.time_to_terminal_floor
    if own_quadrature_floor > bext_cat.EXCLUDED_TERMINAL_SLIVER:
        raise ValueError(
            f"split stage {k}: unresolved-quadrature floor "
            f"{own_quadrature_floor:.3e} exceeds the excluded terminal sliver "
            f"{bext_cat.EXCLUDED_TERMINAL_SLIVER:.3e}; raise EXTENSION_QUADRATURE_NODES "
            "or the excluded sliver in _bermudan_extension_catalogue.")

    free_boundary, derivative_jump = _split_free_boundary_and_jump(cont_above_fn)

    return {
        # Keys consumed by the split ablation's build_ansatz / make_interior_sampler /
        # train_variant (imported and reused, not duplicated).
        "sigma": SIGMA,
        "T": delta_k,
        "stage_terminal_time": delta_k,
        "x_lo": X_LO, "x_hi": X_HI,
        "x_eval_lo": X_EVAL_LO, "x_eval_hi": X_EVAL_HI,
        "generator_coefficients": black_scholes_generator_coefficients(
            volatility=SIGMA, risk_free_rate=R),
        # Exact glued maximum on the window (not the Chen--Mangasarian-smoothed max).
        "terminal_datum": composite_datum,
        "extension_field": extension_field,
        "extension_fn": extension_field.field,
        "extension_derivative_fns": extension_field.derivative_callables(),
        "free_boundary": free_boundary,
        "derivative_jump": derivative_jump,
        "excluded_terminal_sliver": bext_cat.EXCLUDED_TERMINAL_SLIVER,
        "own_quadrature_floor": own_quadrature_floor,
        "ic_name": f"bermudan_split_stage{k}",
        "label": f"Bermudan split stage {k}: [{tau[k]:.3f}, {tau[k+1]:.3f}]",
    }


def _make_stage_problem(k, exercise_times, tau, cont_above_fn, coarse_exact,
                        variant=None):
    """Build a ``train_variant``-compatible problem dict for stage k (interval
    ``[tau[k], tau[k+1]]`` in local time ``s in [0, delta_k]``).

    When ``variant`` is the matched-diffusion split (``extension ==
    "gaussian_semigroup"``) the problem carries the Gaussian-semigroup extension of
    the far-field composite datum and the Black--Scholes generator; otherwise the
    hard heat forms use the time-constant Chen--Mangasarian-smoothed datum
    (``extension_fn=None``), exactly as before.
    """
    import torch

    from learning_option_pricing.pde import chen_mangasarian_max, heat_put_payoff

    delta_k = float(tau[k + 1] - tau[k])

    if _is_split_variant(variant):
        return _make_split_stage_problem(k, tau, delta_k, cont_above_fn, variant)

    def terminal_datum(x):
        # Value at the top of the interval (global tau[k+1]): smoothed max of the
        # exercise payoff and the continuation coming from the stage above.
        return chen_mangasarian_max(heat_put_payoff(x, K), cont_above_fn(x), EPS)

    def exact(x, t):  # t is stage-local s in [0, delta_k]
        return coarse_exact(x, t)

    return {
        "sigma": SIGMA,
        "T": delta_k,
        "x_lo": X_LO, "x_hi": X_HI,
        "x_eval_lo": X_EVAL_LO, "x_eval_hi": X_EVAL_HI,
        "terminal_datum": terminal_datum,
        "extension_fn": None,  # hard forms use the time-constant CM-smoothed g
        "exact": exact,
        "ic_name": f"bermudan_stage{k}",
        "label": f"Bermudan stage {k}: [{tau[k]:.3f}, {tau[k+1]:.3f}]",
    }


# ---------------------------------------------------------------------------
# Chain reconstruction from saved artefacts (revalidation / post-hoc analysis)
# ---------------------------------------------------------------------------

def rebuild_models(run_dir: Path, meta: dict) -> dict:
    """Rebuild the frozen stage chain top-down from the saved per-stage state
    dicts, exactly mirroring the training-time construction (each stage's
    terminal datum closes over the frozen stage above).  Returns ``{k: model}``
    in float64 eval mode with gradients disabled."""
    import torch

    variant = cat.variant_by_name(meta["variant"])
    is_split = _is_split_variant(variant)
    if is_split:
        torch.set_default_dtype(torch.float64)
        from bermudan_free_boundary_extension_ablation import (
            build_ansatz as build_ansatz_fn)
    else:
        from ablation_ansatz_forms import build_ansatz as build_ansatz_fn

    hparams = dict(cat.DEFAULT_HPARAMS)
    hparams["num_iterations"] = meta["num_iterations"]
    exercise_times, tau, m = meta["exercise_times"], meta["tau"], meta["m"]

    models: dict = {}
    cont_above_fn = lambda x: torch.zeros_like(x)  # noqa: E731  (nothing above maturity)
    dummy_exact = lambda x, t: torch.zeros_like(x)  # noqa: E731  (unused at evaluation)
    for k in range(m - 1, -1, -1):
        problem = _make_stage_problem(
            k, exercise_times, tau, cont_above_fn, dummy_exact, variant=variant)
        ansatz = build_ansatz_fn(variant, problem, hparams, model_seed=0)
        state = torch.load(run_dir / f"stage{k}" / "model.pt", map_location="cpu")
        ansatz.load_state_dict(state)
        ansatz.double().eval()
        for p in ansatz.parameters():
            p.requires_grad_(False)
        models[k] = ansatz

        def make_cont(frozen):
            def cont(x):
                xc = x.reshape(-1, 1).to(torch.float64)
                xt = torch.cat([xc, torch.zeros_like(xc)], dim=1)
                return frozen(xt).reshape(x.shape).to(x.dtype)
            return cont
        cont_above_fn = make_cont(ansatz)
    return models


# ---------------------------------------------------------------------------
# Validation against the exact reference
# ---------------------------------------------------------------------------

def _validate(models, exercise_times, tau, *, variant=None, n_x=400, n_quad=4000):
    """Per-global-time learned-vs-exact comparison (the error-propagation curve).

    For the matched-diffusion split variant the reference is the exact Black--Scholes
    Bermudan value :func:`bermudan_put_value_exact_black_scholes` (genuine generator,
    strictly positive rate ``R``); for the heat variants it stays the pure backward-heat
    chained-convolution reference :func:`bermudan_put_value_exact`.
    """
    import numpy as np
    import torch

    from learning_option_pricing.pde import (
        bermudan_put_value_exact, heat_put_exact, heat_put_payoff)

    is_split = _is_split_variant(variant)
    if is_split:
        from learning_option_pricing.pde import (
            bermudan_put_value_exact_black_scholes, black_scholes_put_exact)

    x = torch.linspace(X_EVAL_LO, X_EVAL_HI, n_x, dtype=torch.float64)
    payoff = heat_put_payoff(x, K)
    m = len(exercise_times)

    stages = []
    for k in range(m):  # stage k -> value at global tau[k]
        # Evaluate the trained stage on its native device/dtype: for the hard
        # forms the forward calls the extension, which chains to the frozen
        # stage-above model (possibly on GPU); moving only this stage to CPU would
        # mix devices. Bring just the result to CPU float64 for the CPU reference.
        model = models[k]
        model.eval()
        p0 = next(model.parameters())
        xq = x.to(device=p0.device, dtype=p0.dtype)
        with torch.no_grad():
            cont_dev = model(torch.stack([xq, torch.zeros_like(xq)], dim=1)).squeeze(-1)
        cont = cont_dev.to(device="cpu", dtype=torch.float64)
        v_net = cont if k == 0 else torch.maximum(payoff, cont)
        if is_split:
            v_exact = bermudan_put_value_exact_black_scholes(
                x, torch.full_like(x, float(tau[k])), exercise_times=exercise_times,
                K=K, volatility=SIGMA, risk_free_rate=R,
                y_lo=bext_cat.QUADRATURE_LO, y_hi=bext_cat.QUADRATURE_HI,
                n_quad=bext_cat.REFERENCE_QUADRATURE_NODES)
        else:
            v_exact = bermudan_put_value_exact(
                x, torch.full_like(x, float(tau[k])), exercise_times=exercise_times,
                K=K, sigma=SIGMA, y_lo=Y_LO, y_hi=Y_HI, n_quad=n_quad)
        rel_l2 = float((v_net - v_exact).norm() / v_exact.norm())
        stages.append({
            "k": k, "t_global": float(tau[k]),
            "cont_net": cont.numpy(), "v_net": v_net.numpy(),
            "v_exact": v_exact.numpy(), "rel_l2": rel_l2,
        })
        logger.info("stage k=%d  t=%.3f  rel_l2(value vs exact)=%.3e", k, tau[k], rel_l2)

    if is_split:
        european0 = black_scholes_put_exact(
            x, torch.zeros_like(x), K=K, T=MATURITY, volatility=SIGMA, risk_free_rate=R)
    else:
        european0 = heat_put_exact(x, torch.zeros_like(x), K=K, T=MATURITY, sigma=SIGMA)
    return {
        "x": x.numpy(), "S": torch.exp(x).numpy(), "payoff": payoff.numpy(),
        "european0": european0.numpy(), "stages": stages,
        "tau": np.asarray([float(s) for s in tau]),
        "exercise_times": np.asarray([float(s) for s in exercise_times]),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot(val, out_dir, variant_label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from _figure_layout import finalize_figure

    S = val["S"]
    stages = val["stages"]
    inception = stages[0]

    # --- Figure 1: inception price slice (learned vs exact) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(S, inception["v_net"], "-", color="#1b6ca8", lw=1.6,
            label=r"learned $\hat V(\cdot,0)$")
    ax.plot(S, inception["v_exact"], "--", color="black", lw=1.6,
            label=r"exact Bermudan $V(\cdot,0)$")
    ax.plot(S, val["european0"], "--", color="#888888", lw=1.2,
            label=r"European put (no early exercise)")
    ax.plot(S, val["payoff"], ":", color="#d1495b", lw=1.4,
            label=r"payoff $(K-S)^+$")
    ax.set_xlabel("spot $S$"); ax.set_ylabel("value")
    ax.set_title(f"Bermudan put inception price ({variant_label})", fontsize=10)
    ax.grid(True, alpha=0.3)
    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=True)
    fig.tight_layout(rect=[0, 0.13, 0.80, 1])
    finalize_figure(
        fig, out_dir / "inception_price.png", legends=[leg], axes=[ax],
        formula=(r"learned backward induction (solid) vs exact chained-convolution "
                 r"reference (dashed); inception rel $L^2$ = "
                 f"{inception['rel_l2']:.2e}"), dpi=140, formula_fontsize=8)

    # --- Figure 2: error-propagation curve ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = [s["k"] for s in stages]
    ts = [s["t_global"] for s in stages]
    errs = [s["rel_l2"] for s in stages]
    ax.semilogy(ts, errs, "-o", color="#1b6ca8", lw=1.6)
    for s in stages:
        ax.annotate(f"k={s['k']}", (s["t_global"], s["rel_l2"]),
                    textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
    ax.set_xlabel("global time $t_k$ (0 = inception, $T$ = maturity)")
    ax.set_ylabel(r"relative $L^2$ error vs exact")
    ax.set_title("Error propagation through the backward induction", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.invert_xaxis()  # induction proceeds from maturity (right) to inception (left)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    finalize_figure(
        fig, out_dir / "error_propagation.png", axes=[ax],
        formula=(r"error at each exercise date; induction runs right$\to$left "
                 r"(maturity $\to$ inception). Growth quantifies error compounding "
                 r"across learned stages."), dpi=140, formula_fontsize=8)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _save_arrays(val, out_dir):
    import numpy as np
    flat = {"x": val["x"], "S": val["S"], "payoff": val["payoff"],
            "european0": val["european0"], "tau": val["tau"],
            "exercise_times": val["exercise_times"]}
    for s in val["stages"]:
        k = s["k"]
        flat[f"stage{k}_t_global"] = np.asarray([s["t_global"]])
        flat[f"stage{k}_cont_net"] = s["cont_net"]
        flat[f"stage{k}_v_net"] = s["v_net"]
        flat[f"stage{k}_v_exact"] = s["v_exact"]
        flat[f"stage{k}_rel_l2"] = np.asarray([s["rel_l2"]])
    np.savez(out_dir / "validation.npz", **flat)


def _load_arrays(npz_path):
    import numpy as np
    z = np.load(npz_path)
    n_stages = sum(1 for kk in z.files if kk.endswith("_rel_l2"))
    stages = []
    for k in range(n_stages):
        stages.append({
            "k": k, "t_global": float(z[f"stage{k}_t_global"][0]),
            "cont_net": z[f"stage{k}_cont_net"], "v_net": z[f"stage{k}_v_net"],
            "v_exact": z[f"stage{k}_v_exact"], "rel_l2": float(z[f"stage{k}_rel_l2"][0]),
        })
    return {"x": z["x"], "S": z["S"], "payoff": z["payoff"],
            "european0": z["european0"], "tau": z["tau"],
            "exercise_times": z["exercise_times"], "stages": stages}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--m", type=int, default=2,
                   help="number of exercise dates incl. maturity (m=2: one intermediate)")
    p.add_argument("--variant", choices=cat.variant_names(), default="hard_convex_linear",
                   help="ansatz form for every stage")
    p.add_argument("--num-iterations", type=int, default=cat.DEFAULT_HPARAMS["num_iterations"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--debug", action="store_true",
                   help="prefix the output dir with _debug_ (exploratory runs)")
    p.add_argument("--replot", type=str, default=None,
                   help="redraw figures from a saved run dir and exit")
    p.add_argument("--revalidate", type=str, default=None,
                   help="rebuild the chain from a saved run dir, recompute the "
                        "validation against the exact reference (overwriting "
                        "validation.npz, figures, and the metadata error fields), "
                        "and exit; no training")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)sZ %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S", force=True)

    from learning_option_pricing.utils.run_context import (
        script_data_dir, utc_timestamp, log_runtime_versions)

    if args.replot is not None:
        run_dir = Path(args.replot)
        val = _load_arrays(run_dir / "validation.npz")
        import yaml
        meta = yaml.safe_load(open(run_dir / "metadata.yaml"))
        _plot(val, run_dir, meta.get("variant_label", meta.get("variant", "")))
        logger.info("replotted into %s", run_dir)
        return 0

    if args.revalidate is not None:
        import yaml
        run_dir = Path(args.revalidate)
        meta = yaml.safe_load(open(run_dir / "metadata.yaml"))
        logger.info("revalidating %s (m=%d, variant=%s) against the exact reference",
                    run_dir.name, meta["m"], meta["variant"])
        models = rebuild_models(run_dir, meta)
        exercise_times = [float(s) for s in meta["exercise_times"]]
        tau = [float(s) for s in meta["tau"]]
        val = _validate(models, exercise_times, tau,
                        variant=cat.variant_by_name(meta["variant"]))
        _save_arrays(val, run_dir)
        _plot(val, run_dir, meta.get("variant_label", meta.get("variant", "")))
        meta["rel_l2_per_stage"] = {f"k{s['k']}_t{s['t_global']:.3f}": s["rel_l2"]
                                    for s in val["stages"]}
        meta["inception_rel_l2"] = val["stages"][0]["rel_l2"]
        meta["revalidated"] = True
        with open(run_dir / "metadata.yaml", "w") as f:
            yaml.dump(meta, f, default_flow_style=False, sort_keys=False)
        logger.info("revalidated: inception rel_l2=%.3e (metadata + npz + figures updated)",
                    val["stages"][0]["rel_l2"])
        return 0

    if args.num_iterations < SMOKE_TEST_NUM_ITERATIONS_THRESHOLD and not args.debug:
        raise SystemExit(
            f"--num-iterations {args.num_iterations} is below the smoke-test "
            f"threshold ({SMOKE_TEST_NUM_ITERATIONS_THRESHOLD}); pass --debug "
            f"to flag this as exploratory, or raise the iteration count.")

    import torch
    import yaml

    from ablation_ansatz_forms import derive_seed

    device = torch.device(args.device)
    variant = cat.variant_by_name(args.variant)
    is_split = _is_split_variant(variant)
    # The split path reuses the (tested) build/train functions of the M=2 split
    # ablation, which thread the analytic extension derivatives into the ansatz,
    # assemble P Psi with the Black--Scholes generator, and exclude the unresolved
    # terminal sliver from the interior sampler.  The heat variants keep the shared
    # ablation_ansatz_forms.train_variant unchanged.  The split ablation is validated
    # in float64, so the whole split run is set to float64 for a consistent frozen
    # continuation chain and the tightest analytic-vs-autograd cross-check.
    if is_split:
        torch.set_default_dtype(torch.float64)
        from bermudan_free_boundary_extension_ablation import (
            train_variant as train_variant_fn)
    else:
        from ablation_ansatz_forms import train_variant as train_variant_fn

    hparams = dict(cat.DEFAULT_HPARAMS)
    hparams["num_iterations"] = args.num_iterations

    # Geometry: equally spaced exercise dates t_j = MATURITY * j / m, j=1..m.
    m = args.m
    exercise_times = [MATURITY * j / m for j in range(1, m + 1)]
    tau = [0.0] + exercise_times  # interval endpoints; m intervals

    debug_prefix = "_debug_" if args.debug else ""
    out_dir = script_data_dir(__file__) / (
        f"{debug_prefix}{utc_timestamp()}_m{m}_{args.variant}_iters{args.num_iterations}_seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("command: %s", " ".join(sys.argv))
    log_runtime_versions(logger)
    logger.info("variant=%s m=%d exercise_times=%s iters=%d seed=%d device=%s eps=%g",
                args.variant, m, exercise_times, args.num_iterations, args.seed,
                args.device, EPS)

    t_start = time.time()

    # Backward induction: train from the maturity interval (k=m-1) down to k=0.
    models: dict = {}
    cont_above_fn = (lambda x: torch.zeros_like(x))  # nothing above maturity
    for k in range(m - 1, -1, -1):
        logger.info("=== training stage k=%d  interval [%.3f, %.3f] ===",
                    k, tau[k], tau[k + 1])
        # The split train path does not use the coarse exact diagnostic; the heat path
        # samples it once per stage for the cheap per-iteration boundary comparison.
        coarse_exact = None if is_split else _build_coarse_exact(
            exercise_times, tau[k], float(tau[k + 1] - tau[k]))
        problem = _make_stage_problem(
            k, exercise_times, tau, cont_above_fn, coarse_exact, variant=variant)
        stage_seed = derive_seed(args.seed, f"stage{k}")
        model, history = train_variant_fn(
            variant, problem, hparams, num_iterations=args.num_iterations,
            seed=stage_seed, device=device, log_every=args.log_every)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        models[k] = model
        # checkpoint the stage
        stage_dir = out_dir / f"stage{k}"
        stage_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), stage_dir / "model.pt")

        # freeze this stage's continuation for the stage below.  The terminal
        # datum may call this with x shaped (N,) or (N,1); build a (N,2) input
        # at local time s=0 and return the value reshaped to match x.
        def make_cont(frozen_model):
            dtype = next(frozen_model.parameters()).dtype
            def cont(x):
                xc = x.reshape(-1, 1).to(dtype)
                xt = torch.cat([xc, torch.zeros_like(xc)], dim=1)
                return frozen_model(xt).reshape(x.shape).to(x.dtype)
            return cont
        cont_above_fn = make_cont(model)

    # Validate against the exact reference + plot
    val = _validate(models, exercise_times, tau, variant=variant)
    _save_arrays(val, out_dir)
    _plot(val, out_dir, variant["label"] if "label" in variant else args.variant)

    wall = time.time() - t_start
    meta = {
        "variant": args.variant, "variant_label": variant.get("label", args.variant),
        "m": m, "exercise_times": exercise_times, "tau": tau,
        "num_iterations": args.num_iterations, "seed": args.seed,
        "eps": EPS, "sigma": SIGMA, "K": K, "maturity": MATURITY,
        "wall_time_s": wall,
        "rel_l2_per_stage": {f"k{s['k']}_t{s['t_global']:.3f}": s["rel_l2"]
                             for s in val["stages"]},
        "inception_rel_l2": val["stages"][0]["rel_l2"],
    }
    with open(out_dir / "metadata.yaml", "w") as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)

    logger.info("DONE in %.1fs. inception rel_l2=%.3e. artefacts in %s",
                wall, val["stages"][0]["rel_l2"], out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
