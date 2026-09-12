r"""Down-and-out put: Delta and Gamma at the strike of the saved corner-excluded models.

Re-applies the Phase 2 metric (report ``rapports/strike_g2``, section
"Métriques" of ``rapports/selection_g2_strike``) to the saved models of the
terminal-function comparison trained with the corner window excluded from
collocation: for every configuration (Black-Scholes payoff through the
ordinary autograd route, Black-Scholes payoff through the two-term
analytic-residual route, split-semigroup profile) and every master seed,

    Delta:  d/ds   Phi_theta(K, t)   against  d/ds   V_DO(K, t)
    Gamma:  d2/ds2 Phi_theta(K, t)   against  d2/ds2 V_DO(K, t)

at the strike s = K and calendar times t in {0, 0.25, 0.5, 0.75, 0.9}, reported
as the pointwise relative error |numerical - exact| / |exact|, median over the
seeds with [min, max].

Trained side. Phi_theta = h_eps + g1 u_theta with g1 = (T - t)(s - B); the
Greeks are those of the WHOLE trial solution. The neural-manifold part
g1 u_theta is differentiated by two nested autograd passes on
``ETCNN.forward_neural_manifold`` (the Phase 2 path). The extension h_eps = g2
is differentiated analytically when the extension object exposes
``first_price_derivative`` / ``second_price_derivative`` (split-semigroup
mode; its ``__call__`` is a fixed-grid quadrature whose autograd derivative
is the derivative of the discretisation), and otherwise by the same two
nested autograd passes on g2 with gradients explicitly enabled. Pitfall
avoided on purpose: the pilot's ``compute_loss`` evaluates h_eps under
``torch.no_grad()`` in the two-term route; reusing that path would silently
zero every derivative of h_eps. As a check, the full-model nested autograd
(``model(x)``, exactly the Phase 2 code path) is also computed and its
discrepancy with the decomposed evaluation is logged: it is round-off for the
Black-Scholes modes and the quadrature-autograd error for the split mode.

Reference side. V_DO is the Reiner-Rubinstein closed form of
``learning_option_pricing.pricing.barrier.reiner_rubinstein_down_and_out_put``
transcribed into sympy and differentiated SYMBOLICALLY (``sympy.diff``), then
evaluated in arbitrary precision (mpmath) -- never a finite difference. The
symbolic derivatives are cross-checked against the package's own closed forms
(autograd Delta of the torch price; ``reiner_rubinstein_down_and_out_put_gamma``)
and the maximum relative discrepancy is logged.

Sanity checks, printed and saved:
  1. d2/ds2 h_eps(K, t) != 0 for each configuration (the extension does
     contribute curvature at the strike; a zero here means its derivative was
     computed under no_grad or the wrong object was differentiated);
  2. the exact Gamma grows like tau^(-1/2) as t -> T: least-squares slope of
     log Gamma_exact(K, tau) against log tau on a small-tau grid (expected
     -0.5), plus the slope on the evaluation grid itself for information;
  3. Phi_theta(B, t) ~ 0 for every t (the barrier condition holds by
     construction through g1; anything far above round-off is a loading bug).

Outputs (under ``data/evaluate_greeks_no_corner/<timestamp>_iters<ITERS>_eps<EPS>_nocorner/``):
  - ``greeks_no_corner_table.md``: one row per (configuration, t) with
    err_rel_Delta and err_rel_Gamma as median [min, max] over seeds;
  - ``greeks_no_corner.yaml``: every per-seed value, the references, the
    sanity-check results and the run directories;
  - ``figures/greeks_no_corner.png``: err_rel_Gamma against tau = T - t, one
    curve per configuration (median over seeds; faint points: seeds), log-log.

No retraining, no relative-L2 metric, no window sweep: evaluation of the
existing checkpoints only.

Usage:
    python3 experiments/python_scripts/exp_barrier_option/evaluate_greeks_no_corner.py \
        --iters 20000 --epsilon 0.1
"""
from __future__ import annotations

import argparse
import logging
import statistics
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from learning_option_pricing.pricing.barrier import (  # noqa: E402
    reiner_rubinstein_down_and_out_put,
    reiner_rubinstein_down_and_out_put_gamma,
)
from learning_option_pricing.utils.figure_layout import finalize_figure  # noqa: E402
from learning_option_pricing.utils.run_context import script_data_dir  # noqa: E402
from aggregate_terminal_function_comparison import (  # noqa: E402
    CONFIGURATION_LABELS, PILOT_SCRIPT_PATH, collect_runs,
)
from pilot_down_and_out_put import DEVICE, load_trained_model, read_run_metadata  # noqa: E402

logger = logging.getLogger("evaluate_greeks_no_corner")

DEFAULT_TIMES = [0.0, 0.25, 0.5, 0.75, 0.9]
# Small-tau grid for the tau^(-1/2) growth check of the exact Gamma (sanity
# check 2): well inside the asymptotic regime, unlike the evaluation times.
ASYMPTOTIC_TAU_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

FORMULA_TEXT = (
    r"$\Phi_\theta=h_\varepsilon+g_1u_\theta$, $g_1=(T-t)(s-B)$;  "
    r"$\mathrm{err}_{\mathrm{rel}}\,\Gamma(t)=|\partial_{ss}\Phi_\theta(K,t)-\partial_{ss}V_{DO}(K,t)|"
    r"\,/\,|\partial_{ss}V_{DO}(K,t)|$, $\tau=T-t$."
    "\n"
    r"$\partial_{ss}(g_1u_\theta)$: two nested autograd passes on the frozen network;  "
    r"$\partial_{ss}h_\varepsilon$: analytic (split) or autograd with gradients enabled (Black-Scholes);  "
    r"$\partial_{ss}V_{DO}$: sympy.diff of the Reiner-Rubinstein closed form, evaluated with mpmath."
    "\n"
    "Solid line: median over master seeds; faint points: individual seeds."
)


# ---------------------------------------------------------------------------
# Reference: symbolic Delta and Gamma of the Reiner-Rubinstein closed form
# ---------------------------------------------------------------------------

def build_symbolic_reference(K: float, B: float, r: float, sigma: float):
    """Transcribe ``reiner_rubinstein_down_and_out_put`` into sympy and return
    mpmath-evaluable callables ``(price, delta, gamma)`` of ``(s, tau)``.

    The expression mirrors the torch implementation term by term: truncated
    put TP(s) = P(s, K) - K e^{-r tau} N(-d_-(s, B)) + s N(-d_+(s, B)) with
    P the put-style price, and V_DO(s) = TP(s) - (B/s)^{2r/sigma^2 - 1} TP(B^2/s).
    """
    s, tau = sympy.symbols("s tau", positive=True)
    K_, B_, r_, sigma_ = (sympy.Rational(str(v)) if float(v).is_integer() else sympy.Float(v, 30)
                          for v in (K, B, r, sigma))

    def normal_cdf(x):
        return (1 + sympy.erf(x / sympy.sqrt(2))) / 2

    def d_plus(x, strike):
        return (sympy.log(x / strike) + (r_ + sigma_**2 / 2) * tau) / (sigma_ * sympy.sqrt(tau))

    def put_style_price(x, strike):
        dp = d_plus(x, strike)
        dm = dp - sigma_ * sympy.sqrt(tau)
        return strike * sympy.exp(-r_ * tau) * normal_cdf(-dm) - x * normal_cdf(-dp)

    def truncated_put(x):
        return (put_style_price(x, K_)
                - K_ * sympy.exp(-r_ * tau) * normal_cdf(
                    -((sympy.log(x / B_) + (r_ - sigma_**2 / 2) * tau) / (sigma_ * sympy.sqrt(tau))))
                + x * normal_cdf(
                    -((sympy.log(x / B_) + (r_ + sigma_**2 / 2) * tau) / (sigma_ * sympy.sqrt(tau)))))

    exponent = 2 * r_ / sigma_**2 - 1
    price_expression = truncated_put(s) - (B_ / s) ** exponent * truncated_put(B_**2 / s)
    delta_expression = sympy.diff(price_expression, s)
    gamma_expression = sympy.diff(price_expression, s, 2)
    price = sympy.lambdify((s, tau), price_expression, modules="mpmath")
    delta = sympy.lambdify((s, tau), delta_expression, modules="mpmath")
    gamma = sympy.lambdify((s, tau), gamma_expression, modules="mpmath")
    return price, delta, gamma


def cross_check_symbolic_reference(reference, K, B, r, sigma, tau_values: list[float]) -> dict:
    """Maximum relative discrepancy between the symbolic derivatives and the
    package's own closed forms (autograd Delta of the torch price, closed-form
    Gamma), in float64. Expected: round-off (1e-13 or below)."""
    price, delta, gamma = reference
    worst = {"price": 0.0, "delta": 0.0, "gamma": 0.0}
    for tau in tau_values:
        s_t = torch.tensor([K], dtype=torch.float64, requires_grad=True)
        tau_t = torch.tensor([tau], dtype=torch.float64)
        v = reiner_rubinstein_down_and_out_put(s_t, K, B, r, sigma, tau_t)
        torch_delta = torch.autograd.grad(v.sum(), s_t)[0].item()
        torch_gamma = reiner_rubinstein_down_and_out_put_gamma(
            torch.tensor([K], dtype=torch.float64), K, B, r, sigma, tau_t).item()
        for key, torch_value, sym_value in (("price", v.item(), float(price(K, tau))),
                                            ("delta", torch_delta, float(delta(K, tau))),
                                            ("gamma", torch_gamma, float(gamma(K, tau)))):
            worst[key] = max(worst[key], abs(torch_value - sym_value) / abs(sym_value))
    return worst


# ---------------------------------------------------------------------------
# Trained side: Greeks of the whole trial solution at (K, t)
# ---------------------------------------------------------------------------

def _nested_autograd_greeks(function, s_value: float, t_value: float) -> tuple[float, float, float]:
    """``(value, d/ds, d2/ds2)`` of ``function(s, t)`` at one point by two nested
    autograd passes, with gradients explicitly enabled."""
    with torch.enable_grad():
        s = torch.tensor([s_value], dtype=torch.get_default_dtype(), device=DEVICE, requires_grad=True)
        t = torch.tensor([t_value], dtype=torch.get_default_dtype(), device=DEVICE)
        value = function(s, t)
        first = torch.autograd.grad(value.sum(), s, create_graph=True)[0]
        second = torch.autograd.grad(first.sum(), s)[0]
    return value.item(), first.item(), second.item()


def trained_greeks_at_strike(model, K: float, t_value: float) -> dict:
    """Delta and Gamma of Phi_theta = g1 u_theta + h_eps at (K, t), the two
    parts differentiated separately (see the module docstring), plus the
    full-model Phase 2 path for the consistency check."""
    def neural_manifold(s, t):
        return model.forward_neural_manifold(torch.stack([s, t], dim=1)).squeeze(-1)

    def full_model(s, t):
        return model(torch.stack([s, t], dim=1)).squeeze(-1)

    manifold_value, manifold_delta, manifold_gamma = _nested_autograd_greeks(neural_manifold, K, t_value)
    g2 = model.g2
    if hasattr(g2, "second_price_derivative"):
        with torch.no_grad():  # analytic derivatives: no graph needed
            s = torch.tensor([K], dtype=torch.get_default_dtype(), device=DEVICE)
            t = torch.tensor([t_value], dtype=torch.get_default_dtype(), device=DEVICE)
            extension_value = g2(s, t).item()
            extension_delta = g2.first_price_derivative(s, t).item()
            extension_gamma = g2.second_price_derivative(s, t).item()
        extension_route = "analytic"
    else:
        extension_value, extension_delta, extension_gamma = _nested_autograd_greeks(g2, K, t_value)
        extension_route = "autograd (gradients enabled)"
    full_value, full_delta, full_gamma = _nested_autograd_greeks(full_model, K, t_value)
    return {
        "value": manifold_value + extension_value,
        "delta": manifold_delta + extension_delta,
        "gamma": manifold_gamma + extension_gamma,
        "extension_value": extension_value,
        "extension_delta": extension_delta,
        "extension_gamma": extension_gamma,
        "extension_route": extension_route,
        "manifold_gamma": manifold_gamma,
        "phase2_full_model_delta": full_delta,
        "phase2_full_model_gamma": full_gamma,
    }


def barrier_condition_residual(model, B: float, T: float, n_t: int = 41) -> float:
    """max_t |Phi_theta(B, t)| on a uniform t grid (sanity check 3)."""
    with torch.no_grad():
        t = torch.linspace(0.0, T, n_t, dtype=torch.get_default_dtype(), device=DEVICE)
        s = torch.full_like(t, B)
        return float(model(torch.stack([s, t], dim=1)).abs().max())


def log_log_slope(x: list[float], y: list[float]) -> float:
    slope, _ = np.polyfit(np.log(np.asarray(x)), np.log(np.asarray(y)), 1)
    return float(slope)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _stats(values: list[float]) -> dict:
    return {"median": statistics.median(values), "min": min(values), "max": max(values), "n": len(values)}


def write_table(results: dict, times: list[float], T: float, path: Path, iters: int, epsilon: float) -> None:
    lines = [
        f"# Delta and Gamma at the strike — corner-excluded runs, {iters} iterations, epsilon = {epsilon:g}",
        "",
        "Pointwise relative error |numerical - exact| / |exact| at s = K; median over master seeds [min, max].",
        "Reference: sympy symbolic derivative of the Reiner-Rubinstein closed form (mpmath evaluation).",
        "",
        "The exact Greeks are listed because the relative error is ill-conditioned where the exact Gamma "
        "is close to a sign change (a small denominator inflates err_rel_Gamma without the numerical "
        "Gamma being worse in absolute terms); such rows are marked with (*).",
        "",
        "| Configuration | t | tau = T - t | Delta exact | Gamma exact | n seeds | err_rel_Delta | err_rel_Gamma |",
        "|---|---|---|---|---|---|---|---|",
    ]
    gamma_scale = max(abs(next(iter(per_time[t].values()))["exact_gamma"])
                      for per_time in results.values() for t in times)
    for configuration, per_time in results.items():
        for t_value in times:
            delta = _stats([v["err_rel_delta"] for v in per_time[t_value].values()])
            gamma = _stats([v["err_rel_gamma"] for v in per_time[t_value].values()])
            any_seed = next(iter(per_time[t_value].values()))
            exact_gamma = any_seed["exact_gamma"]
            ill_conditioned = " (*)" if abs(exact_gamma) < 0.1 * gamma_scale else ""
            lines.append(
                f"| {CONFIGURATION_LABELS[configuration].replace(chr(10), ' ')} | {t_value:g} | {T - t_value:g} | "
                f"{any_seed['exact_delta']:+.4e} | {exact_gamma:+.4e}{ill_conditioned} | "
                f"{delta['n']} | {delta['median']:.3e} [{delta['min']:.3e}, {delta['max']:.3e}] | "
                f"{gamma['median']:.3e} [{gamma['min']:.3e}, {gamma['max']:.3e}] |"
            )
    lines += ["", f"(*) |Gamma exact| below 10 % of its largest value over the evaluation times "
                  f"({gamma_scale:.3e}): relative error ill-conditioned."]
    path.write_text("\n".join(lines) + "\n")


def plot_gamma_error(results: dict, times: list[float], T: float, path: Path, iters: int, epsilon: float) -> None:
    fig, ax = plt.subplots(figsize=(13.0, 5.2))
    markers = {"blackscholes": "o", "blackscholes_analyticres": "s", "split": "^"}
    colors = {"blackscholes": "tab:blue", "blackscholes_analyticres": "tab:orange", "split": "tab:green"}
    handles = []
    for configuration, per_time in results.items():
        taus = [T - t for t in times]
        color = colors.get(configuration, None)
        for t_value, tau in zip(times, taus):
            seeds_values = [v["err_rel_gamma"] for v in per_time[t_value].values()]
            ax.scatter([tau] * len(seeds_values), seeds_values, s=16, alpha=0.3, color=color,
                       marker=markers.get(configuration, "o"), zorder=2)
        medians = [statistics.median(v["err_rel_gamma"] for v in per_time[t].values()) for t in times]
        (line,) = ax.plot(taus, medians, "-", color=color, marker=markers.get(configuration, "o"),
                          markersize=6, markeredgecolor="black", zorder=3,
                          label=CONFIGURATION_LABELS[configuration].replace("\n", " "))
        handles.append(line)
    # Mark the evaluation times where the exact Gamma is close to its sign
    # change (small denominator: the relative error is ill-conditioned there).
    any_configuration = next(iter(results.values()))
    exact_gammas = {t: next(iter(any_configuration[t].values()))["exact_gamma"] for t in times}
    gamma_scale = max(abs(g) for g in exact_gammas.values())
    ill_conditioned = [(T - t, g) for t, g in exact_gammas.items() if abs(g) < 0.1 * gamma_scale]
    for tau, _ in ill_conditioned:
        marker_line = ax.axvline(tau, linestyle=":", color="grey", zorder=1)
    if ill_conditioned:
        marker_line.set_label(
            r"$\tau$ where $|\partial_{ss}V_{DO}(K,t)|<10\%$ of its maximum over the evaluation times"
            "\n(" + ", ".join(f"$\\tau={tau:g}$: ${g:+.2e}$" for tau, g in ill_conditioned)
            + "; near a sign change: relative error ill-conditioned)")
        handles.append(marker_line)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Time to maturity $\tau = T - t$")
    ax.set_ylabel(r"Relative error of $\Gamma$ at the strike, $\mathrm{err}_{\mathrm{rel}}\,\Gamma(t)$")
    ax.set_title(f"Down-and-out put — Gamma error at $s=K$, corner-excluded runs, {iters} iterations, "
                 f"$\\varepsilon={epsilon:g}$", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    legend = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8,
                       title="Terminal-function configuration")
    fig.subplots_adjust(left=0.08, right=0.52, top=0.9, bottom=0.3)
    finalize_figure(fig, path, legends=[legend], formula=FORMULA_TEXT, axes=[ax], formula_fontsize=6.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iters", type=int, required=True, help="Iteration budget of the runs to evaluate.")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Corner-layer bandwidth of the runs.")
    parser.add_argument("--base-dir", type=str, default=None,
                        help="Directory holding the pilot's run directories (default: the pilot's data directory).")
    parser.add_argument("--times", nargs="+", type=float, default=DEFAULT_TIMES,
                        help="Calendar times t at which the Greeks are evaluated.")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"],
                        help="Evaluation dtype. The models were trained in float32; float64 evaluation loads "
                             "the same weights and removes float32 round-off from the nested autograd passes.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory override.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir) if args.base_dir is not None else script_data_dir(PILOT_SCRIPT_PATH)
    out_dir = (Path(args.out_dir) if args.out_dir is not None else script_data_dir(__file__) / (
        f"{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}_iters{args.iters}_eps{args.epsilon:g}_nocorner"
    ))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "evaluate_greeks.log")],
    )
    logger.info("Delta and Gamma at the strike — corner-excluded terminal-function comparison")
    logger.info(f"  Command: {' '.join(sys.argv)}")
    logger.info(f"  Python: {sys.version.split()[0]}  PyTorch: {torch.__version__}  sympy: {sympy.__version__}  "
                f"device: {DEVICE}  evaluation dtype: {args.dtype}")
    logger.info(f"  Run directories read from: {base_dir}")
    logger.info(f"  Output directory: {out_dir}")
    logger.info(f"  times t = {args.times}")
    torch.set_default_dtype(torch.float64 if args.dtype == "float64" else torch.float32)

    runs = collect_runs(base_dir, args.iters, args.epsilon, require_nocorner=True)
    if not runs:
        logger.error("No matching corner-excluded run directory found.")
        sys.exit(1)

    # Contract: read from the first run's metadata and required to be shared.
    first_run_dir = Path(next(iter(next(iter(runs.values())).values()))["run_dir"])
    contract = read_run_metadata(first_run_dir)["contract"]
    K, B, r, sigma, T = (contract[k] for k in ("K", "B", "r", "sigma", "T"))
    logger.info(f"  Contract (from {first_run_dir.name}/metadata.yaml): K={K:g} B={B:g} r={r:g} sigma={sigma:g} T={T:g}")
    for configuration, per_seed in runs.items():
        for seed, summary in per_seed.items():
            other = read_run_metadata(Path(summary["run_dir"]))["contract"]
            if other != contract:
                logger.error(f"  {configuration} seed {seed}: contract {other} differs from {contract}; aborting.")
                sys.exit(1)
        logger.info(f"  {configuration}: seeds {sorted(per_seed)}")

    # ---- reference ----------------------------------------------------------
    logger.info("Building the symbolic Reiner-Rubinstein reference (sympy.diff, mpmath evaluation)")
    reference = build_symbolic_reference(K, B, r, sigma)
    _, reference_delta, reference_gamma = reference
    taus = [T - t for t in args.times]
    worst = cross_check_symbolic_reference(reference, K, B, r, sigma, taus)
    logger.info(f"  cross-check vs package closed forms (max relative discrepancy at s=K over the evaluation "
                f"taus): price {worst['price']:.2e}, Delta (autograd of torch price) {worst['delta']:.2e}, "
                f"Gamma (reiner_rubinstein_down_and_out_put_gamma) {worst['gamma']:.2e}")
    exact = {t: {"delta": float(reference_delta(K, T - t)), "gamma": float(reference_gamma(K, T - t))}
             for t in args.times}
    for t in args.times:
        logger.info(f"  exact at (K, t={t:g}, tau={T - t:g}): Delta={exact[t]['delta']:+.10e}  "
                    f"Gamma={exact[t]['gamma']:+.10e}")

    # ---- sanity check 2: tau^(-1/2) growth of the exact Gamma ---------------
    asymptotic_gammas = [float(reference_gamma(K, tau)) for tau in ASYMPTOTIC_TAU_GRID]
    slope_asymptotic = log_log_slope(ASYMPTOTIC_TAU_GRID, asymptotic_gammas)
    evaluation_gammas = [exact[t]["gamma"] for t in args.times]
    sign_change = min(evaluation_gammas) < 0 < max(evaluation_gammas)
    slope_evaluation = log_log_slope(taus, [abs(g) for g in evaluation_gammas])
    logger.info(f"SANITY CHECK 2 — exact Gamma(K, tau) ~ tau^(-1/2) as t -> T: log-log slope on "
                f"tau in {ASYMPTOTIC_TAU_GRID}: {slope_asymptotic:+.4f} (expected -0.5); on |Gamma| over the "
                f"evaluation taus {[round(x, 6) for x in taus]}: {slope_evaluation:+.4f} (not asymptotic, for information"
                + ("; NOTE: the exact Gamma changes sign over these taus, so the relative error is "
                   "ill-conditioned near the zero crossing)" if sign_change else "") + ")")
    if abs(slope_asymptotic + 0.5) > 0.05:
        logger.warning("  SANITY CHECK 2 FAILED: slope differs from -0.5 by more than 0.05.")

    # ---- trained models -------------------------------------------------------
    results: dict = {}
    sanity: dict = {"extension_gamma_at_strike": {}, "barrier_condition_max_abs": {},
                    "phase2_full_model_vs_decomposed_max_rel": {}}
    for configuration in CONFIGURATION_LABELS:
        if configuration not in runs:
            continue
        results[configuration] = {t: {} for t in args.times}
        sanity["extension_gamma_at_strike"][configuration] = {}
        sanity["barrier_condition_max_abs"][configuration] = {}
        sanity["phase2_full_model_vs_decomposed_max_rel"][configuration] = {}
        for seed in sorted(runs[configuration]):
            run_dir = Path(runs[configuration][seed]["run_dir"])
            model = load_trained_model(run_dir, args.epsilon)
            barrier_residual = barrier_condition_residual(model, B, T)
            sanity["barrier_condition_max_abs"][configuration][seed] = barrier_residual
            worst_phase2 = 0.0
            for t in args.times:
                greeks = trained_greeks_at_strike(model, K, t)
                err_delta = abs(greeks["delta"] - exact[t]["delta"]) / abs(exact[t]["delta"])
                err_gamma = abs(greeks["gamma"] - exact[t]["gamma"]) / abs(exact[t]["gamma"])
                results[configuration][t][seed] = {
                    **greeks, "exact_delta": exact[t]["delta"], "exact_gamma": exact[t]["gamma"],
                    "err_rel_delta": err_delta, "err_rel_gamma": err_gamma, "run_dir": str(run_dir),
                }
                worst_phase2 = max(worst_phase2,
                                   abs(greeks["phase2_full_model_gamma"] - greeks["gamma"]) / abs(greeks["gamma"]),
                                   abs(greeks["phase2_full_model_delta"] - greeks["delta"]) / abs(greeks["delta"]))
                logger.info(
                    f"  {configuration:<26s} seed {seed} t={t:<5g} Delta={greeks['delta']:+.6e} "
                    f"(err {err_delta:.3e})  Gamma={greeks['gamma']:+.6e} (err {err_gamma:.3e})  "
                    f"[d2 h_eps={greeks['extension_gamma']:+.4e}, d2 (g1 u)={greeks['manifold_gamma']:+.4e}]"
                )
            sanity["extension_gamma_at_strike"][configuration][seed] = {
                t: results[configuration][t][seed]["extension_gamma"] for t in args.times}
            sanity["phase2_full_model_vs_decomposed_max_rel"][configuration][seed] = worst_phase2
            route = results[configuration][args.times[0]][seed]["extension_route"]
            logger.info(f"  {configuration:<26s} seed {seed}: h_eps derivatives by {route}; "
                        f"Phase 2 full-model autograd vs decomposed evaluation: max rel {worst_phase2:.2e}; "
                        f"max_t |Phi_theta(B,t)| = {barrier_residual:.2e}")

    # ---- sanity checks 1 and 3 ---------------------------------------------
    for configuration in results:
        gammas = sanity["extension_gamma_at_strike"][configuration]
        smallest = min(abs(v) for per_t in gammas.values() for v in per_t.values())
        seed0 = sorted(gammas)[0]
        logger.info(f"SANITY CHECK 1 — d2/ds2 h_eps(K, t) for {configuration}: "
                    + ", ".join(f"t={t:g}: {gammas[seed0][t]:+.4e}" for t in args.times)
                    + f"  (min |.| over seeds and t: {smallest:.3e}) -> {'OK' if smallest > 0 else 'FAILED (zero)'}")
        if smallest == 0.0:
            logger.warning(f"  SANITY CHECK 1 FAILED for {configuration}: the extension's Gamma is zero at the strike.")
        residuals = sanity["barrier_condition_max_abs"][configuration]
        worst_residual = max(residuals.values())
        tolerance = 1e-12 if args.dtype == "float64" else 1e-6
        logger.info(f"SANITY CHECK 3 — max_t |Phi_theta(B, t)| for {configuration}: "
                    + ", ".join(f"seed {s}: {v:.2e}" for s, v in residuals.items())
                    + f" -> {'OK' if worst_residual < tolerance else 'FAILED'} (tolerance {tolerance:g})")
        if worst_residual >= tolerance:
            logger.warning(f"  SANITY CHECK 3 FAILED for {configuration}: barrier condition violated at loading.")

    # ---- outputs ----------------------------------------------------------------
    with open(out_dir / "greeks_no_corner.yaml", "w") as f:
        yaml.dump({
            "command": " ".join(sys.argv), "iters": args.iters, "epsilon": args.epsilon,
            "contract": contract, "times": args.times, "evaluation_dtype": args.dtype,
            "reference": {"kind": "sympy.diff of the Reiner-Rubinstein closed form, mpmath evaluation",
                          "cross_check_max_rel_discrepancy_vs_package": worst,
                          "exact_at_strike": exact},
            "sanity_checks": {
                "1_extension_gamma_at_strike": sanity["extension_gamma_at_strike"],
                "2_exact_gamma_log_log_slope": {"asymptotic_tau_grid": ASYMPTOTIC_TAU_GRID,
                                                "slope_asymptotic": slope_asymptotic,
                                                "slope_on_abs_gamma_over_evaluation_taus": slope_evaluation,
                                                "exact_gamma_changes_sign_over_evaluation_taus": sign_change,
                                                "expected": -0.5},
                "3_barrier_condition_max_abs": sanity["barrier_condition_max_abs"],
                "phase2_full_model_vs_decomposed_max_rel": sanity["phase2_full_model_vs_decomposed_max_rel"],
            },
            "results": results,
        }, f, default_flow_style=False, sort_keys=False)
    write_table(results, args.times, T, out_dir / "greeks_no_corner_table.md", args.iters, args.epsilon)
    plot_gamma_error(results, args.times, T, out_dir / "figures" / "greeks_no_corner.png", args.iters, args.epsilon)
    logger.info(f"  Saved: {out_dir / 'greeks_no_corner_table.md'}, {out_dir / 'greeks_no_corner.yaml'}, "
                f"{out_dir / 'figures' / 'greeks_no_corner.png'}")
    for configuration, per_time in results.items():
        for t in args.times:
            g = _stats([v["err_rel_gamma"] for v in per_time[t].values()])
            d = _stats([v["err_rel_delta"] for v in per_time[t].values()])
            logger.info(f"  {configuration:<26s} t={t:<5g} err_rel_Delta={d['median']:.3e} [{d['min']:.3e}, {d['max']:.3e}]  "
                        f"err_rel_Gamma={g['median']:.3e} [{g['min']:.3e}, {g['max']:.3e}]")


if __name__ == "__main__":
    main()
