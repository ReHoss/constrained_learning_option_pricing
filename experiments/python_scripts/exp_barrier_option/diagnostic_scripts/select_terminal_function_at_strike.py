r"""Selection of the terminal-function mode that best resolves the strike singularity.

Decides, on measured quantities, which of the terminal-function modes of
:mod:`learning_option_pricing.pricing.barrier` best resolves the payoff's
first-derivative discontinuity at ``s=K``.  Unlike
``compare_payoff_modes_at_strike.py``, which compares exactly four runs at one
fixed value of the Chen-Mangasarian bandwidth, this script compares each mode
**at its own best bandwidth** -- the envelope-against-envelope comparison a
fair ranking requires, since two of the five modes have a free hyperparameter
and three have none.

Reads already-trained runs only; never retrains.  Every quantity below comes
from a forward pass or an autograd pass through a frozen trained model, or
from a closed form.

The conflicting-corner ``(B,T)`` is EXCLUDED from every metric here: the
signed-error heat maps of ``compare_payoff_modes_at_strike.py`` (analysis 9)
measured that the corner carries 47 to 63 per cent of the squared error of
every smoothed mode, which masks the strike ranking entirely.  The corner is a
separate problem, treated separately; the exclusion is by the ell^1 mask
``(s-B) + (T-t) <= corner_window``.  Note that the corner is excluded from the
metrics only -- the training that produced these runs sampled it (uniform
collocation over the whole domain, about 8.5 of 4096 points per iteration
inside the corner layer at epsilon=0.1).

Metrics, all restricted to the strike band ``|s-K| <= delta`` with the corner
mask removed:

- **Primary** -- relative L^2 error of the price,
  ``||Phi_theta - V_DO|| / ||V_DO||``.  Reported for several ``delta`` so the
  sensitivity of the ranking to that arbitrary width is visible rather than
  hidden (the ranking is known to reverse between the two best modes at
  ``delta=0.2``, where the band stops being a neighbourhood of the strike).
- **Secondary** -- relative L^2 error of the Gamma against the exact
  Reiner-Rubinstein Gamma, the quantity the strike singularity actually
  damages; minimum price over the band (a negative price is an arbitrage
  violation and disqualifies a mode outright); and, for context only, the
  share of the mode's total squared error carried by the corner.

Fairness label, to be carried into any report: the ``black_scholes`` mode
inserts the closed-form European price of the very operator being solved.  It
is an ORACLE bound on what is achievable, not a generally applicable method
(no closed form exists for a general contract).  The ``split`` mode is the
generic counterpart of the same idea (mollification by the Gaussian
semigroup) and carries no such advantage.

Figures (the two recommended in review, made quantitative):

1. ``envelope_<metric>.png`` -- each metric against the Chen-Mangasarian
   bandwidth eps0, one curve per grading, with the no-hyperparameter modes as
   horizontal reference lines: the envelope-against-envelope comparison.
2. ``price_delta_gamma_profiles.png`` -- price, Delta and Gamma against the
   underlying price s at fixed calendar times, each mode at its selected
   bandwidth, each against its own exact closed-form counterpart.
3. ``error_field_heat_maps.png`` -- the signed error field Phi_theta - V_DO
   over (s,t), full domain and corner zoom, one panel per selected mode.

Usage:
    python3 experiments/python_scripts/exp_barrier_option/diagnostic_scripts/\
select_terminal_function_at_strike.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402

import pilot_down_and_out_put as pilot  # noqa: E402
import compare_payoff_modes_at_strike as compare  # noqa: E402
from learning_option_pricing.pricing.barrier import (  # noqa: E402
    reiner_rubinstein_down_and_out_put,
    reiner_rubinstein_down_and_out_put_gamma,
)
from learning_option_pricing.utils.figure_layout import finalize_figure  # noqa: E402
from learning_option_pricing.utils.run_context import script_data_dir  # noqa: E402

logger = logging.getLogger("select_terminal_function_at_strike")

# Modes without a free hyperparameter, drawn as horizontal reference lines on
# the envelope figures.  "black_scholes" is the ORACLE (see module docstring).
FIXED_PARAMETER_MODES = ["raw", "black_scholes", "black_scholes_two_term", "split"]
FIXED_MODE_STYLE = {
    "raw": ("tab:brown", "-", "Raw payoff $(K-s)^+$ (negative control)"),
    "black_scholes": ("tab:blue", "-", "Black-Scholes $V^e$ (ORACLE), one-graph residual"),
    "black_scholes_two_term": ("tab:purple", "-", "Black-Scholes $V^e$ (ORACLE), two-term residual"),
    "split": ("tab:red", "-", "Split semigroup (generic counterpart)"),
}
GRADING_STYLE = {
    "constant": ("tab:green", r"Chen-Mangasarian, constant $\varepsilon_0$"),
    "time_graded": ("tab:orange", r"Chen-Mangasarian, time-graded $\varepsilon_0(t)$"),
}


def classify_run(meta: dict) -> tuple[str, float | None]:
    """Return ``(mode, free_hyperparameter)``; the hyperparameter is None for
    a mode that has none."""
    hp = meta["hyperparameters"]
    if hp.get("smoothed_payoff", False):
        return f"mangasarian_{hp['grading']}", float(hp["eps0"])
    if hp.get("black_scholes_payoff", False):
        # Same field, two training routes: the ordinary one-graph autograd
        # residual, and the two-term assembly the analytic residual enables.
        # They are distinct entries here -- that difference is the control arm.
        return ("black_scholes_two_term" if hp.get("analytic_residual", False)
                else "black_scholes"), None
    if hp.get("split_payoff", False):
        return "split", None
    return "raw", None


def run_display_name(mode: str, hyperparameter: float | None) -> str:
    if hyperparameter is None:
        return FIXED_MODE_STYLE[mode][2] if mode in FIXED_MODE_STYLE else mode
    grading = mode.replace("mangasarian_", "")
    return f"{GRADING_STYLE[grading][1]}, $\\varepsilon_0={hyperparameter:g}$"


def discover_runs(base_dir: Path, epsilon: float, iters: int, seed: int) -> list[dict]:
    """Every completed run at the given (epsilon, iters, seed), one per
    terminal-function mode and, for the Chen-Mangasarian modes, per
    bandwidth.  A run without its per-epsilon summary file is still training
    and is skipped with a warning."""
    runs, skipped = [], []
    for run_dir in sorted(base_dir.glob(f"*iters{iters}_eps{epsilon:g}_seed{seed}*")):
        metadata_path = run_dir / "metadata.yaml"
        if not metadata_path.exists() or run_dir.name.startswith("_debug_"):
            continue
        with open(metadata_path) as f:
            meta = yaml.safe_load(f)
        if not (run_dir / f"summary_eps{epsilon:g}.yaml").exists():
            skipped.append(run_dir.name)
            continue
        mode, hyperparameter = classify_run(meta)
        if any(r["mode"] == mode and r["hyperparameter"] == hyperparameter for r in runs):
            logger.warning(f"skipped (duplicate configuration already collected): {run_dir.name}")
            continue
        runs.append({"run_dir": run_dir, "meta": meta, "mode": mode, "hyperparameter": hyperparameter})
    for name in skipped:
        logger.warning(f"skipped (still training, no summary file): {name}")
    return runs


def load_trained_model(run_dir: Path, meta: dict, dtype: torch.dtype = torch.float64) -> torch.nn.Module:
    """Rebuild the ansatz from metadata.yaml and load the saved weights.

    Extends ``compare_payoff_modes_at_strike.load_trained_model`` with the
    split-semigroup mode, whose extension needs its quadrature support and
    node count restored exactly as trained.
    """
    hp, contract = meta["hyperparameters"], meta["contract"]
    epsilon = hp["epsilons"][0]
    model = pilot.build_model(
        contract["K"], contract["B"], contract["T"], epsilon, model_seed=0,
        smoothed_payoff=hp.get("smoothed_payoff", False),
        eps0=hp.get("eps0", pilot.DEFAULT_EPS0),
        grading=hp.get("grading", pilot.DEFAULT_GRADING),
        black_scholes_payoff=hp.get("black_scholes_payoff", False),
        r=contract["r"], sigma=contract["sigma"],
        split_payoff=hp.get("split_payoff", False),
        s_inf=meta["domain"]["s_inf"],
        comparison_volatility=hp.get("comparison_volatility"),
        split_y_lo=hp.get("split_y_lo"), split_y_hi=hp.get("split_y_hi"),
        split_n_quad=hp.get("split_n_quad", pilot.DEFAULT_SPLIT_N_QUAD),
    )
    model.load_state_dict(torch.load(run_dir / f"models/model_eps{epsilon:g}.pt",
                                     map_location="cpu", weights_only=True))
    return model.to(dtype).eval()


# The helpers reused from compare_payoff_modes_at_strike (error fields, price
# and Greek profiles) call *its* model loader, which predates the
# split-semigroup mode and silently rebuilds a split run as a raw one -- the
# weights then load into the wrong ansatz and every figure drawn for that mode
# is wrong while its metric table, computed here, is right.  Redirect that
# module's loader to the split-aware one above so both paths agree.
compare.load_trained_model = load_trained_model


# ---------------------------------------------------------------------------
# Metrics on the strike band, corner excluded
# ---------------------------------------------------------------------------

def compute_metrics(
    run: dict, s_grid: torch.Tensor, t_grid: torch.Tensor,
    strike_windows: list[float], corner_window: float, terminal_margins: list[float],
) -> dict:
    """Every decision metric for one run, on one shared (s,t) grid.

    The price and its Gamma come from the trained model (forward pass, and two
    nested autograd passes); the references are the closed form and its exact
    Gamma.  Every band metric excludes the ell^1 corner window.
    """
    contract = run["meta"]["contract"]
    K, B, r, sigma, T = contract["K"], contract["B"], contract["r"], contract["sigma"], contract["T"]
    model = load_trained_model(run["run_dir"], run["meta"])

    ss, tt = torch.meshgrid(s_grid, t_grid, indexing="ij")
    reference_price = reiner_rubinstein_down_and_out_put(ss, K, B, r, sigma, T - tt)
    with torch.no_grad():
        learned_price = model(torch.stack([ss.reshape(-1), tt.reshape(-1)], dim=1)).squeeze(-1).reshape(ss.shape)
    price_error = (learned_price - reference_price).numpy()

    # Gamma: one autograd pass per time slice (the graph is per-slice)
    learned_delta = np.zeros(ss.shape)
    reference_delta = np.zeros(ss.shape)
    learned_gamma = np.zeros(ss.shape)
    reference_gamma = np.zeros(ss.shape)
    for time_index, t_value in enumerate(t_grid.tolist()):
        s = s_grid.clone().requires_grad_(True)
        x = torch.stack([s, torch.full_like(s, t_value)], dim=1)
        value = model(x).squeeze(-1)
        first = torch.autograd.grad(value.sum(), s, create_graph=True)[0]
        learned_delta[:, time_index] = first.detach().numpy()
        learned_gamma[:, time_index] = torch.autograd.grad(first.sum(), s)[0].detach().numpy()
        # Reference Delta by autograd on the CLOSED FORM (exact analytic
        # derivative, not a finite difference); the reference Gamma has its
        # own independently implemented closed form, which cross-checks it.
        s_reference = s_grid.clone().requires_grad_(True)
        tau_reference = torch.full_like(s_reference, T - t_value)
        value_reference = reiner_rubinstein_down_and_out_put(s_reference, K, B, r, sigma, tau_reference)
        reference_delta[:, time_index] = torch.autograd.grad(
            value_reference.sum(), s_reference)[0].detach().numpy()
        reference_gamma[:, time_index] = reiner_rubinstein_down_and_out_put_gamma(
            s_grid, K, B, r, sigma, torch.full_like(s_grid, T - t_value)).numpy()
    delta_error = learned_delta - reference_delta
    gamma_error = learned_gamma - reference_gamma

    corner_mask = ((ss - B).abs() + (T - tt) <= corner_window).numpy()
    reference_price_numpy = reference_price.numpy()
    metrics = {
        "mode": run["mode"], "hyperparameter": run["hyperparameter"],
        "run_dir": str(run["run_dir"]),
        "corner_share_of_squared_error": float((price_error[corner_mask] ** 2).sum() / (price_error ** 2).sum()),
    }
    # The exact Gamma is unbounded as t -> T; the closed form only returns a
    # finite value there because of its own tau floor (_TAU_EPS = 1e-8 in
    # pricing.barrier), which yields Gamma(K,T) = 1.33e4 -- an artefact of that
    # clamp, not a physical value.  A single such slice dominates any L^2 norm
    # over the band and drives every mode's relative Gamma error to 1.000.
    # Every metric is therefore evaluated on t <= T - margin, and reported for
    # several margins so the approach to maturity is visible rather than hidden.
    time_grid_numpy = tt.numpy()

    # PRIMARY DOMAIN: the whole truncated domain MINUS the corner window.  The
    # goal is the extension that gives the best solution overall, the corner
    # being a separate problem treated separately; a strike band of arbitrary
    # half-width delta is a diagnostic of where the constructions differ, not
    # the criterion.  Price and Delta over t in [0,T]; Gamma with its margin.
    outside_corner = ~corner_mask
    metrics["price_rel_l2_domain"] = float(
        np.linalg.norm(price_error[outside_corner]) / np.linalg.norm(reference_price_numpy[outside_corner]))
    metrics["delta_rel_l2_domain"] = float(
        np.linalg.norm(delta_error[outside_corner]) / np.linalg.norm(reference_delta[outside_corner]))
    for margin in terminal_margins:
        region = outside_corner & (time_grid_numpy <= T - margin)
        metrics[f"gamma_rel_l2_domain_margin{margin:g}"] = float(
            np.linalg.norm(gamma_error[region]) / np.linalg.norm(reference_gamma[region]))
    metrics["min_price_domain"] = float(learned_price.numpy()[outside_corner].min())

    # Same, but excluding the whole corner LAYER {s <= B+epsilon} rather than
    # only the ell^1 corner window: the window is 0.38 per cent of the area, so
    # a "domain minus corner" metric built on it is still dominated by the
    # near-barrier zone, where the cutoff zeta varies and every mode is poor.
    # Outside the layer zeta is identically 1 and g2 is the pure terminal
    # function -- this is the region where the constructions are comparable
    # on their own terms.
    outside_layer = (ss > B + run["meta"]["hyperparameters"]["epsilons"][0]).numpy()
    metrics["price_rel_l2_outside_layer"] = float(
        np.linalg.norm(price_error[outside_layer]) / np.linalg.norm(reference_price_numpy[outside_layer]))
    metrics["delta_rel_l2_outside_layer"] = float(
        np.linalg.norm(delta_error[outside_layer]) / np.linalg.norm(reference_delta[outside_layer]))
    for margin in terminal_margins:
        region = outside_layer & (time_grid_numpy <= T - margin)
        metrics[f"gamma_rel_l2_outside_layer_margin{margin:g}"] = float(
            np.linalg.norm(gamma_error[region]) / np.linalg.norm(reference_gamma[region]))
    metrics["min_price_outside_layer"] = float(learned_price.numpy()[outside_layer].min())

    for delta in strike_windows:
        band_all_times = ((ss - K).abs() <= delta).numpy() & ~corner_mask

        # Price and Delta are BOUNDED at t=T, so they are measured over the
        # whole time range INCLUDING the terminal slice -- which is where the
        # modes differ structurally (there g_1(s,T)=0, so Phi_theta(s,T) is
        # the extension alone and the terminal condition Phi=payoff either
        # holds exactly or does not).  Only the Gamma needs a margin, the
        # exact Gamma being unbounded as t -> T (clamp 1, see the module
        # docstring): applying that margin to the price too, as an earlier
        # version did, hid the terminal behaviour of every mode.
        metrics[f"price_rel_l2_delta{delta:g}"] = float(
            np.linalg.norm(price_error[band_all_times]) / np.linalg.norm(reference_price_numpy[band_all_times]))
        metrics[f"delta_rel_l2_delta{delta:g}"] = float(
            np.linalg.norm(delta_error[band_all_times]) / np.linalg.norm(reference_delta[band_all_times]))
        metrics[f"min_price_delta{delta:g}"] = float(learned_price.numpy()[band_all_times].min())

        # Terminal-trace error: ||Phi_theta(.,T) - payoff|| on the band.  With
        # g_1(s,T)=0 this is a property of the extension alone, no network
        # involved -- it is the metric that answers "at t=T, does the priced
        # value equal the payoff?".
        terminal_band = band_all_times & (time_grid_numpy >= T - 1e-12)
        if terminal_band.any():
            payoff_terminal = np.clip(K - ss.numpy()[terminal_band], 0.0, None)
            learned_terminal = learned_price.numpy()[terminal_band]
            metrics[f"terminal_trace_rel_l2_delta{delta:g}"] = float(
                np.linalg.norm(learned_terminal - payoff_terminal) / np.linalg.norm(payoff_terminal))
            metrics[f"terminal_trace_max_delta{delta:g}"] = float(
                np.abs(learned_terminal - payoff_terminal).max())

        for margin in terminal_margins:
            band = band_all_times & (time_grid_numpy <= T - margin)
            metrics[f"gamma_rel_l2_delta{delta:g}_margin{margin:g}"] = float(
                np.linalg.norm(gamma_error[band]) / np.linalg.norm(reference_gamma[band]))

    band_key = f"delta{strike_windows[1]:g}"
    logger.info(
        f"[{run['mode']}" + (f", eps0={run['hyperparameter']:g}" if run["hyperparameter"] is not None else "")
        + f"] DOMAINE prix {metrics['price_rel_l2_domain']:.4e} delta {metrics['delta_rel_l2_domain']:.4e} "
        f"gamma {metrics[f'gamma_rel_l2_domain_margin{terminal_margins[0]:g}']:.4e} | bande: "
        f"prix {metrics[f'price_rel_l2_{band_key}']:.4e}; "
        f"Delta {metrics[f'delta_rel_l2_{band_key}']:.4e}; "
        f"Gamma {metrics[f'gamma_rel_l2_{band_key}_margin{terminal_margins[0]:g}']:.4e}; "
        f"terminal trace {metrics.get(f'terminal_trace_rel_l2_{band_key}', float('nan')):.4e}; "
        f"min price {metrics[f'min_price_{band_key}']:.4e}; "
        f"corner {metrics['corner_share_of_squared_error'] * 100:.1f} %"
    )
    return metrics


def select_best_per_mode(rows: list[dict], primary_key: str) -> dict[str, dict]:
    """Each mode at its own best hyperparameter, on the primary metric."""
    best: dict[str, dict] = {}
    for row in rows:
        current = best.get(row["mode"])
        if current is None or row[primary_key] < current[primary_key]:
            best[row["mode"]] = row
    return best


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_envelope(
    rows: list[dict], metric_key: str, metric_label: str, out_path: Path, log_y: bool = True,
) -> None:
    """One metric against the Chen-Mangasarian bandwidth, one curve per
    grading, the no-hyperparameter modes as horizontal lines."""
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for grading, (color, label) in GRADING_STYLE.items():
        mode = f"mangasarian_{grading}"
        points = sorted([(row["hyperparameter"], row[metric_key]) for row in rows if row["mode"] == mode])
        if not points:
            continue
        x, y = zip(*points)
        ax.plot(x, y, color=color, marker="o", lw=1.6, label=label)
    for mode in FIXED_PARAMETER_MODES:
        row = next((r for r in rows if r["mode"] == mode), None)
        if row is None:
            continue
        color, style, label = FIXED_MODE_STYLE[mode]
        ax.axhline(row[metric_key], color=color, linestyle=style, lw=1.6, label=label)
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(r"Chen-Mangasarian bandwidth $\varepsilon_0$ (log scale)")
    ax.set_ylabel(metric_label)
    ax.grid(alpha=0.3, which="both")
    legend = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    formula = (
        r"Every metric is restricted to the strike band $|s-K|\leq\delta$ with the $\ell^1$ corner window "
        r"$(s-B)+(T-t)\leq w$ REMOVED, so the conflicting corner $(B,T)$ does not enter the ranking"
        "\n"
        r"Horizontal lines: the modes with no free bandwidth. Black-Scholes is an ORACLE -- it inserts the "
        r"closed-form European price of the operator being solved, unavailable for a general contract"
    )
    fig.subplots_adjust(right=0.58, bottom=0.30)
    finalize_figure(fig, out_path, legends=[legend], formula=formula, axes=[ax])


def render_metric_table(rows: list[dict], strike_windows: list[float], terminal_margins: list[float]) -> str:
    """Full metric table, one line per run, ordered by the primary metric.

    Columns show the primary metric's sensitivity to both arbitrary choices it
    contains: the strike half-width delta, and the terminal margin (the exact
    Gamma is unbounded as t -> T, so a margin is unavoidable).
    """
    primary = f"price_rel_l2_delta{strike_windows[1]:g}"
    ordered = sorted(rows, key=lambda r: r[primary])
    header = (["Mode", "$\\varepsilon_0$"]
              + [f"Price rel $L^2$, $\\delta={d:g}$" for d in strike_windows]
              + [f"Delta rel $L^2$, $\\delta={strike_windows[1]:g}$"]
              + [f"Gamma rel $L^2$, margin ${m:g}$" for m in terminal_margins]
              + ["Terminal trace", "Min price", "Corner share"])
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---:" for _ in header) + "|"]
    for row in ordered:
        cells = [run_display_name(row["mode"], row["hyperparameter"]),
                 "--" if row["hyperparameter"] is None else f"{row['hyperparameter']:g}"]
        cells += [f"{row[f'price_rel_l2_delta{d:g}']:.4e}" for d in strike_windows]
        cells += [f"{row[f'delta_rel_l2_delta{strike_windows[1]:g}']:.4e}"]
        cells += [f"{row[f'gamma_rel_l2_delta{strike_windows[1]:g}_margin{m:g}']:.4e}" for m in terminal_margins]
        cells += [f"{row.get(f'terminal_trace_rel_l2_delta{strike_windows[1]:g}', float('nan')):.4e}",
                  f"{row[f'min_price_delta{strike_windows[1]:g}']:.4e}",
                  f"{row['corner_share_of_squared_error'] * 100:.1f} %"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_error_against_time(
    runs: list[dict], mode_order: list[str], s_grid: torch.Tensor, t_grid: torch.Tensor,
    corner_layer_edge: float, terminal_margin: float, out_path: Path,
) -> dict:
    r"""Relative L^2 error of the price, Delta and Gamma over s, slice by slice
    in calendar time, with the ansatz weight :math:`g_1` overlaid.

    This is the figure that explains why every mode is accurate near maturity
    and inaccurate at early times, and why a metric restricted to a
    neighbourhood of the strike reports smaller numbers than one over the whole
    domain.  The trial solution is :math:`\Phi_\theta=g_1u_\theta+g_2` with
    :math:`g_1(s,t)=(T-t)(s-B)`, so the network's error is multiplied by
    :math:`g_1`: it vanishes as :math:`t\to T`, where
    :math:`\Phi_\theta\to g_2` whose terminal trace is exact, and is fully
    expressed at :math:`t=0` where :math:`g_1` is largest.
    """
    contract = runs[0]["meta"]["contract"]
    K, B, r, sigma, T = contract["K"], contract["B"], contract["r"], contract["sigma"], contract["T"]
    s_values = s_grid[s_grid > corner_layer_edge]
    t_numpy = t_grid.numpy()

    series = {label: {key: [] for key in ("price", "delta", "gamma")} for label in mode_order}
    models = {run["label"]: load_trained_model(run["run_dir"], run["meta"]) for run in runs}
    for t_value in t_grid.tolist():
        tau = torch.full_like(s_values, T - t_value)
        reference_price = reiner_rubinstein_down_and_out_put(s_values, K, B, r, sigma, tau)
        s_reference = s_values.clone().requires_grad_(True)
        value_reference = reiner_rubinstein_down_and_out_put(
            s_reference, K, B, r, sigma, torch.full_like(s_reference, T - t_value))
        reference_delta = torch.autograd.grad(value_reference.sum(), s_reference)[0].detach()
        reference_gamma = reiner_rubinstein_down_and_out_put_gamma(s_values, K, B, r, sigma, tau)
        for label in mode_order:
            model = models[label]
            s = s_values.clone().requires_grad_(True)
            x = torch.stack([s, torch.full_like(s, t_value)], dim=1)
            value = model(x).squeeze(-1)
            first = torch.autograd.grad(value.sum(), s, create_graph=True)[0]
            second = torch.autograd.grad(first.sum(), s)[0]
            for key, learned, reference in (("price", value.detach(), reference_price),
                                            ("delta", first.detach(), reference_delta),
                                            ("gamma", second.detach(), reference_gamma)):
                series[label][key].append(float(torch.linalg.vector_norm(learned - reference)
                                                / torch.linalg.vector_norm(reference)))

    quantity_title = {"price": r"Prix, $\|\Phi_\theta-V_{DO}\|/\|V_{DO}\|$",
                      "delta": r"Delta, $\|\partial_s\Phi_\theta-\partial_sV_{DO}\|/\|\partial_sV_{DO}\|$",
                      "gamma": r"Gamma, $\|\partial_{ss}\Phi_\theta-\partial_{ss}V_{DO}\|/\|\partial_{ss}V_{DO}\|$"}
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    legend = None
    for ax, key in zip(axes, ("price", "delta", "gamma")):
        # The margin applies to all three panels: at t=T exactly, g_1=0 and
        # Phi_theta = g_2, so an exact-trace mode has identically zero error
        # there -- a single degenerate point that costs six decades of the
        # logarithmic axis and hides the trend being shown.
        keep = t_numpy <= T - terminal_margin
        for label in mode_order:
            ax.semilogy(t_numpy[keep], np.asarray(series[label][key])[keep],
                        color=compare.MODE_COLOR[label], lw=1.9, alpha=0.9,
                        label=compare.MODE_DISPLAY_NAME[label])
        weight_axis = ax.twinx()
        weight_axis.plot(t_numpy, (T - t_numpy) * (K - B), color="grey", linestyle=":", lw=2.0)
        weight_axis.set_ylim(bottom=0.0)
        if key == "price":
            weight_axis.set_ylabel(r"$g_1(K,t)=(T-t)(K-B)$", color="grey", fontsize=9)
            weight_axis.tick_params(axis="y", labelcolor="grey")
        else:
            weight_axis.set_yticklabels([])
        ax.set_xlabel("Temps calendaire $t$")
        ax.set_title(quantity_title[key], fontsize=10)
        ax.grid(alpha=0.3, which="both")
        if key == "price":
            ax.set_ylabel("Erreur relative $L^2$ sur $s$ (échelle log)")
        if key == "gamma":
            legend = ax.legend(loc="upper left", bbox_to_anchor=(1.14, 1.0), fontsize=8)

    formula = (
        r"Erreur relative $L^2$ prise tranche par tranche en $t$, sur $s>" + f"{corner_layer_edge:g}" +
        r"$ (le domaine de mesure). Pointillé gris, axe de droite : le poids $g_1(K,t)=(T-t)(K-B)$"
        "\n"
        r"$\Phi_\theta=g_1u_\theta+g_2$ : l'erreur du réseau est multipliée par $g_1$. Cet "
        r"amortissement domine pour le prix ; pour les grecques il est concurrencé par l'affinement de $\partial_sV_{DO}$ et $\partial_{ss}V_{DO}$ quand $t\to T$, d'où leur forme en U"
        "\n"
        r"Panneau Gamma tronqué à $t\leq T-" + f"{terminal_margin:g}" +
        r"$ : le Gamma exact n'est pas borné en $t=T$"
    )
    fig.subplots_adjust(right=0.72, bottom=0.30, wspace=0.42)
    finalize_figure(fig, out_path, legends=[legend] if legend is not None else [],
                    formula=formula, axes=list(axes))
    return series


def plot_where_the_gamma_error_lives(
    run: dict, s_grid: torch.Tensor, t_grid: torch.Tensor,
    corner_layer_edge: float, strike_window: float, terminal_margin: float, out_path: Path,
) -> dict:
    r"""Where, in the underlying price, the Gamma error of one mode is located.

    Answers a question the other figures cannot: the relative Gamma error over
    the measurement domain and the one over a narrow strike band differ by two
    orders of magnitude near maturity, and the reason is spatial -- the error
    concentrates in a thin strip just above the corner layer, which the strike
    band never sees.

    Left panel: the absolute Gamma error over (s,t) on a logarithmic colour
    scale.  Right panel: the same relative error computed on three regions,
    against calendar time.
    """
    contract = run["meta"]["contract"]
    K, B, r, sigma, T = contract["K"], contract["B"], contract["r"], contract["sigma"], contract["T"]
    s_values = s_grid[s_grid > corner_layer_edge]
    model = load_trained_model(run["run_dir"], run["meta"])
    t_kept = t_grid[t_grid <= T - terminal_margin]

    error_field = np.zeros((len(s_values), len(t_kept)))
    reference_field = np.zeros_like(error_field)
    for time_index, t_value in enumerate(t_kept.tolist()):
        s = s_values.clone().requires_grad_(True)
        value = model(torch.stack([s, torch.full_like(s, t_value)], dim=1)).squeeze(-1)
        first = torch.autograd.grad(value.sum(), s, create_graph=True)[0]
        learned = torch.autograd.grad(first.sum(), s)[0].detach()
        reference = reiner_rubinstein_down_and_out_put_gamma(
            s_values, K, B, r, sigma, torch.full_like(s_values, T - t_value))
        error_field[:, time_index] = (learned - reference).numpy()
        reference_field[:, time_index] = reference.numpy()

    s_numpy, t_numpy = s_values.numpy(), t_kept.numpy()
    near_barrier = (s_numpy >= corner_layer_edge) & (s_numpy < corner_layer_edge + 0.10)
    strike_band = np.abs(s_numpy - K) <= strike_window
    regions = {"near_barrier": near_barrier, "strike_band": strike_band,
               "whole_domain": np.ones_like(near_barrier)}
    series = {name: [float(np.linalg.norm(error_field[mask, k]) / np.linalg.norm(reference_field[mask, k]))
                     for k in range(len(t_numpy))] for name, mask in regions.items()}

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))

    # LEFT PANEL: the composition of the report's own metric.  The relative
    # L^2 error is a single number per region -- an integral -- so a map needs
    # a localised quantity.  Rather than invent one, the panel shows how the
    # integrand of that very norm, (Phi_theta - V_DO)^2, is distributed over
    # the underlying price: the share of the total squared error carried by
    # each band of s, at each date.  The shares sum to 100 per cent by
    # construction, so the panel reads directly as "where does the number in
    # the table come from".
    bands = [(0.70, 0.80), (0.80, 0.95), (0.95, 1.05), (1.05, 1.30), (1.30, 2.00)]
    band_colours = ["tab:red", "tab:orange", "tab:green", "tab:blue", "tab:purple"]
    squared = error_field ** 2
    total = squared.sum(axis=0)
    shares = []
    for lo, hi in bands:
        mask = (s_numpy >= lo) & (s_numpy < hi)
        shares.append(100.0 * squared[mask, :].sum(axis=0) / total)
    axes[0].stackplot(t_numpy, *shares, colors=band_colours,
                      labels=[(r"$s\in[%.2f,%.2f[$" % (lo, hi)) + ("  (strike)" if lo == 0.95 else
                              ("  (barrière)" if lo == 0.70 else "")) for lo, hi in bands])
    axes[0].set_xlim(t_numpy[0], t_numpy[-1])
    axes[0].set_ylim(0, 100)
    axes[0].set_xlabel("Temps calendaire $t$")
    axes[0].set_ylabel("Part de l'erreur quadratique totale (\\%)")
    axes[0].set_title("D'où vient l'erreur $L^2$ du Gamma, par bande de prix", fontsize=10)
    band_legend = axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, -0.16), fontsize=8, ncol=2)

    style = {"near_barrier": ("tab:red", f"Près de la barrière, $s\\in[{corner_layer_edge:g},{corner_layer_edge+0.10:g}[$"),
             "strike_band": ("tab:green", f"Bande du strike, $|s-K|\\leq{strike_window:g}$"),
             "whole_domain": ("black", "Domaine entier, $s>" + f"{corner_layer_edge:g}$")}
    for name in ("whole_domain", "near_barrier", "strike_band"):
        color, label = style[name]
        axes[1].semilogy(t_numpy, series[name], color=color, lw=2.0,
                         linestyle="--" if name == "whole_domain" else "-", label=label)
    axes[1].set_xlabel("Temps calendaire $t$")
    axes[1].set_ylabel("Erreur relative $L^2$ du Gamma (échelle log)")
    axes[1].set_title("La même erreur, sur trois régions", fontsize=10)
    axes[1].grid(alpha=0.3, which="both")
    legend = axes[1].legend(loc="upper left", bbox_to_anchor=(0.02, -0.16), fontsize=8)

    formula = (
        r"Configuration retenue : Black-Scholes, résidu assemblé en deux termes. Les deux panneaux "
        r"décomposent la MÊME grandeur que le tableau du rapport : l'erreur relative $L^2$ du Gamma"
        "\n"
        r"Gauche : part de l'erreur quadratique totale portée par chaque bande de prix, à chaque date ; "
        r"les parts somment à 100\% par construction"
        "\n"
        r"Droite : la même erreur relative, calculée sur trois régions. La courbe du domaine entier suit "
        r"celle de la barrière : c'est cette zone qui fixe le niveau global, non le strike"
    )
    fig.subplots_adjust(bottom=0.36, wspace=0.28)
    finalize_figure(fig, out_path, legends=[legend, band_legend], formula=formula, axes=list(axes))
    return {"t_grid": t_numpy.tolist(), **series}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the terminal-function mode that best resolves the strike singularity, "
                    "each mode at its own best bandwidth. Reads trained runs only; never retrains.")
    parser.add_argument("--base-dir", type=str, default=None,
                        help="Directory holding the run subdirectories (default: the pilot's own data directory).")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Corner bandwidth shared by every compared run.")
    parser.add_argument("--iters", type=int, default=20000, help="Iteration budget shared by every compared run.")
    parser.add_argument("--seed", type=int, default=0, help="Master seed shared by every compared run.")
    parser.add_argument("--strike-windows", nargs="+", type=float, default=[0.02, 0.05, 0.10, 0.20],
                        help="Half-widths delta of the strike band; the SECOND is the primary metric's, "
                             "the others quantify the ranking's sensitivity to that arbitrary choice.")
    parser.add_argument("--terminal-margins", nargs="+", type=float, default=[0.01, 0.05, 0.2],
                        help="Metrics are evaluated on t <= T - margin. A margin is unavoidable: the exact "
                             "Gamma is unbounded as t -> T and the closed form only returns a finite value "
                             "there through its own tau floor (_TAU_EPS=1e-8), giving Gamma(K,T)=1.33e4, "
                             "which alone dominates any L^2 norm over the band. The FIRST value defines the "
                             "primary metric; the others quantify the ranking's sensitivity to it.")
    parser.add_argument("--corner-window", type=float, default=0.1,
                        help="Half-width of the ell^1 corner window removed from every metric.")
    parser.add_argument("--n-s", type=int, default=400, help="Number of s grid points of the metric grid.")
    parser.add_argument("--n-t", type=int, default=200, help="Number of t grid points of the metric grid.")
    parser.add_argument("--s-max", type=float, default=2.0, help="Upper end of the metric grid's s-range.")
    parser.add_argument("--profile-times", nargs="+", type=float, default=[0.0, 0.5, 0.9, 0.99],
                        help="Calendar times at which the price / Delta / Gamma profiles are drawn.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("matplotlib.mathtext").setLevel(logging.WARNING)

    base_dir = Path(args.base_dir) if args.base_dir else script_data_dir(Path(pilot.__file__).resolve())
    runs = discover_runs(base_dir, args.epsilon, args.iters, args.seed)
    if not runs:
        logger.error(f"no completed run found under {base_dir} at eps={args.epsilon:g}, "
                     f"iters={args.iters}, seed={args.seed}.")
        sys.exit(1)
    logger.info(f"{len(runs)} completed runs compared (epsilon={args.epsilon:g}, iters={args.iters}, seed={args.seed}):")
    for run in runs:
        suffix = "" if run["hyperparameter"] is None else f", eps0={run['hyperparameter']:g}"
        logger.info(f"  {run['mode']}{suffix:<14} -> {run['run_dir'].name}")

    out_dir = Path(args.out_dir) if args.out_dir else base_dir / (
        datetime.now().astimezone().strftime("%Y%m%d_%H%M%S") + "_select_terminal_function_strike")
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    s_grid = torch.linspace(runs[0]["meta"]["contract"]["B"], args.s_max, args.n_s, dtype=torch.float64)
    terminal_margins = sorted(args.terminal_margins)
    # The grid reaches t = T: the price and Delta metrics need the terminal
    # slice, and the Gamma metric masks it off by its own margin.
    t_grid = torch.linspace(0.0, runs[0]["meta"]["contract"]["T"], args.n_t, dtype=torch.float64)
    logger.info(f"metric grid: s in [{float(s_grid[0]):g}, {float(s_grid[-1]):g}] x {args.n_s}, "
                f"t in [0, {float(t_grid[-1]):g}] x {args.n_t} (maturity margin {terminal_margins[0]:g}); "
                f"corner window {args.corner_window:g} removed")

    rows = [compute_metrics(run, s_grid, t_grid, args.strike_windows, args.corner_window, terminal_margins)
            for run in runs]
    # Same metric the report ranks by: the whole domain outside the corner
    # layer.  Selecting on the narrow strike band instead would draw, for a
    # given family, a different bandwidth from the one the table ranks first.
    primary_key = "price_rel_l2_outside_layer"

    table_text = render_metric_table(rows, args.strike_windows, terminal_margins)
    print("\n=== Strike-band metrics, corner excluded (ordered by the primary metric) ===\n" + table_text + "\n")
    with open(out_dir / "strike_selection_metrics.md", "w") as f:
        f.write(table_text + "\n")
    with open(out_dir / "strike_selection_metrics.yaml", "w") as f:
        yaml.dump(rows, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Metric table saved -> {out_dir / 'strike_selection_metrics.md'} / .yaml")

    for metric_key, metric_label in (
        (primary_key, f"Price rel. $L^2$ error on $|s-K|\\leq{args.strike_windows[1]:g}$ (log scale)"),
        (f"gamma_rel_l2_delta{args.strike_windows[1]:g}_margin{terminal_margins[0]:g}",
         f"Gamma rel. $L^2$ error on $|s-K|\\leq{args.strike_windows[1]:g}$ (log scale)"),
    ):
        path = figures_dir / f"envelope_{metric_key}.png"
        plot_envelope(rows, metric_key, metric_label, path)
        logger.info(f"Figure saved -> {path}")

    best = select_best_per_mode(rows, primary_key)
    logger.info("Selected bandwidth per mode (primary metric):")
    for mode, row in sorted(best.items(), key=lambda kv: kv[1][primary_key]):
        suffix = "" if row["hyperparameter"] is None else f" at eps0={row['hyperparameter']:g}"
        logger.info(f"  {mode}{suffix}: {primary_key} = {row[primary_key]:.4e}")
    with open(out_dir / "selected_per_mode.yaml", "w") as f:
        yaml.dump({mode: row for mode, row in best.items()}, f, default_flow_style=False, sort_keys=False)

    # The two figures recommended in review, drawn for the selected runs only.
    selected_runs = []
    for mode, row in best.items():
        run = next(r for r in runs if str(r["run_dir"]) == row["run_dir"])
        label = mode if row["hyperparameter"] is None else f"{mode}_eps0{row['hyperparameter']:g}"
        compare.MODE_DISPLAY_NAME[label] = run_display_name(mode, row["hyperparameter"])
        compare.MODE_COLOR[label] = (FIXED_MODE_STYLE[mode][0] if mode in FIXED_MODE_STYLE
                                     else GRADING_STYLE[mode.replace("mangasarian_", "")][0])
        selected_runs.append({**run, "label": label})
    selected_order = [r["label"] for r in sorted(
        selected_runs, key=lambda r: best[r["mode"]][primary_key])]

    contract = runs[0]["meta"]["contract"]
    # The profiles are drawn on the SAME domain the metrics use: the corner
    # layer [B, B+epsilon] is excluded.  Drawn from s=B, as an earlier version
    # did, the near-barrier oscillations of every mode span several decades and
    # compress the strike region of the symlog axis to invisibility -- and the
    # figure then disagrees with the protocol, which excludes the corner.
    profile_s_min = contract["B"] + args.epsilon
    profiles = compare.compute_price_and_greek_profiles_vs_price(
        selected_runs, torch.linspace(profile_s_min, args.s_max, 801, dtype=torch.float64), args.profile_times)
    profile_path = figures_dir / "price_delta_gamma_profiles.png"
    compare.plot_price_delta_gamma_profiles_vs_price(
        profiles, selected_order, contract["K"], contract["B"], args.epsilon, 1e-1, profile_path)
    logger.info(f"Figure saved -> {profile_path} (s from {profile_s_min:g}, corner layer excluded)")

    # Strike zoom, on exactly the band |s-K| <= delta the primary metric uses.
    strike_window = args.strike_windows[1]
    zoom_profiles = compare.compute_price_and_greek_profiles_vs_price(
        selected_runs,
        torch.linspace(contract["K"] - strike_window, contract["K"] + strike_window, 601, dtype=torch.float64),
        args.profile_times)
    zoom_path = figures_dir / "price_delta_gamma_profiles_strike_zoom.png"
    compare.plot_price_delta_gamma_profiles_vs_price(
        zoom_profiles, selected_order, contract["K"], contract["B"], args.epsilon, 1e-2, zoom_path)
    logger.info(f"Figure saved -> {zoom_path} (band |s-K| <= {strike_window:g})")

    fields = compare.compute_error_fields(selected_runs, s_grid, t_grid)
    heat_map_path = figures_dir / "error_field_heat_maps.png"
    compare.plot_error_heat_maps(
        fields, selected_order, s_grid, t_grid, contract["K"], contract["B"], contract["T"],
        args.epsilon, args.corner_window, args.strike_windows[1], heat_map_path)
    logger.info(f"Figure saved -> {heat_map_path}")

    time_figure_path = figures_dir / "erreur_vs_temps.png"
    time_series = plot_error_against_time(
        selected_runs, selected_order, s_grid, t_grid,
        contract["B"] + args.epsilon, terminal_margins[0], time_figure_path)
    logger.info(f"Figure saved -> {time_figure_path}")
    with open(out_dir / "erreur_vs_temps.yaml", "w") as f:
        yaml.dump({"t_grid": t_grid.tolist(), **time_series}, f, default_flow_style=False, sort_keys=False)

    retained = next((r for r in selected_runs if r["mode"] == "black_scholes_two_term"), selected_runs[0])
    localisation_path = figures_dir / "ou_vit_l_erreur_gamma.png"
    localisation = plot_where_the_gamma_error_lives(
        retained, s_grid, t_grid, contract["B"] + args.epsilon,
        args.strike_windows[1], terminal_margins[0], localisation_path)
    logger.info(f"Figure saved -> {localisation_path}")
    with open(out_dir / "ou_vit_l_erreur_gamma.yaml", "w") as f:
        yaml.dump(localisation, f, default_flow_style=False, sort_keys=False)

    np.savez(out_dir / "selection_fields.npz", s_values=s_grid.numpy(), t_values=t_grid.numpy(),
             **{label: fields[label] for label in selected_order})
    logger.info(f"All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
