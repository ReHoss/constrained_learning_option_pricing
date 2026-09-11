r"""Down-and-out put: comparison of the four g2 terminal-function modes near the strike.

Compares four already-trained runs of ``pilot_down_and_out_put.py`` at the
same corner-regularisation bandwidth epsilon and the same master seed, each
using a different terminal-function mode for g2:

- ``raw``:                      the discontinuous payoff (K-s)^+
  (:func:`learning_option_pricing.pricing.barrier.make_corner_regularised_extension`).
- ``mangasarian_time_graded``:  Chen-Mangasarian smoothed payoff,
  smoothing bandwidth epsilon_0(t) = epsilon_0*(T-t)/T (0 exactly at t=T).
- ``mangasarian_constant``:     Chen-Mangasarian smoothed payoff,
  epsilon_0(t) = epsilon_0 for every t.
- ``black_scholes``:            the exact Black-Scholes European put price.

Does not retrain or resample anything: every analysis below either reads an
already-saved artefact (``training.log`` for the loss history, the saved
model checkpoints for a forward/autograd evaluation) or recomputes an
evaluation metric on a dense grid via a forward pass through the frozen
trained model -- never a gradient step. This is the same "evaluation only"
category as ``pilot_down_and_out_put.py --replot``.

Analyses, each producing one figure (and, for several, one table).  The
numbering below covers only the first four; analyses 5 to 8 are documented at
their own section headers further down:

1. Relative L^2 error, global vs. excluding a neighbourhood of the barrier
   (``s < B + margin``, margin configurable, default 0.1) -- does excluding
   this near-barrier zone change the ranking of the four modes?
2. Loss (interior PDE residual, the only loss term in this pilot -- see
   ``pilot_down_and_out_put.py::compute_loss``) vs. iteration, log scale,
   the four modes overlaid.
3. ``||d^2U/ds^2||_{L^2}`` (discrete L^2 norm over a small s-window around
   the strike K) vs. calendar time t, the four modes overlaid, log scale.
   Computed via two nested ``torch.autograd.grad`` calls on the full trained
   ansatz Phi_theta = g1*u_theta + g2, the full trained price (not on g2 alone,
   and not to be confused with u_theta, the bare network). A fifth, reference
   curve overlays the same L^2 norm computed from the exact closed-form Gamma
   of the true down-and-out price
   (:func:`~learning_option_pricing.pricing.barrier.reiner_rubinstein_down_and_out_put_gamma`,
   analytic, no network involved) -- the ground truth each mode's learned
   curvature is implicitly trying to reproduce.
4. A strike-zoom, LINEAR-scale price slice (not log: log scale cannot
   represent a negative value, and the pilot's existing ``log_slice_*.png``
   floors negative prices to 1e-12 before taking the log, which visually
   hides any negative-price artefact) -- directly shows whether prices near
   the strike dip below zero, and by how much, for each mode.

Analyses 3, 5 and 6 hold the underlying price fixed (at or near the strike)
and sweep the calendar time.  Analysis 8 is their transpose: the calendar time
is held fixed at a few values and the whole s-profile of the second price
derivative of the *full* trained price Phi_theta = g1*u_theta + g2 is drawn,
over an s-range spanning both the barrier corner layer and the strike, against
the exact Reiner-Rubinstein Gamma.  A companion figure splits that profile
into its network term partial_ss(g1*u_theta) and its extension term
partial_ss(g2) = partial_ss(h_epsilon), so that the profile of the extension
alone is read against the profile of the price it is part of.

Caveat on analysis 3 for the ``raw`` mode (stated once here, not repeated in
the figure): the raw g2 has a genuine C^0 kink at s=K (a first-derivative
discontinuity), but that kink is a single point of measure zero. Autograd
sampled at generic grid points away from that exact point differentiates the
locally linear branch of g2 (whose second derivative is exactly 0 there), so
the curvature this analysis reports for the ``raw`` mode is entirely the
compensating curvature the trained network u_theta has learned in an attempt
to approximate the kink -- not the (formally infinite, delta-function) true
curvature of the kink itself. A finite-difference sweep straddling s=K (as
used in ``test/pricing/test_barrier.py``'s C^2 checks) is the right tool to
see the kink directly; this script does not repeat that here because the
user-specified method for this diagnostic is autograd on the saved model.

Usage:
    python3 experiments/python_scripts/exp_barrier_option/diagnostic_scripts/\
compare_payoff_modes_at_strike.py \
        --run-dir-prefix 20260825_014738_iters20000_eps0.1_seed0
"""
from __future__ import annotations

import argparse
import logging
import re
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
from learning_option_pricing.pricing.barrier import (  # noqa: E402
    reiner_rubinstein_down_and_out_put,
    reiner_rubinstein_down_and_out_put_gamma,
)
from learning_option_pricing.utils.figure_layout import finalize_figure  # noqa: E402
from learning_option_pricing.utils.run_context import script_data_dir  # noqa: E402

logger = logging.getLogger("compare_payoff_modes_at_strike")

# Canonical mode order and per-mode plotting style, shared across all figures
# so a given mode always carries the same colour and can be cross-referenced
# at a glance between figures.
MODE_ORDER = ["raw", "mangasarian_time_graded", "mangasarian_constant", "black_scholes"]
MODE_DISPLAY_NAME = {
    "raw": "Raw payoff $(K-s)^+$",
    "mangasarian_time_graded": r"Mangasarian, time-graded $\varepsilon_0(t)$",
    "mangasarian_constant": r"Mangasarian, constant $\varepsilon_0$",
    "black_scholes": "Black-Scholes $V^e(s,t)$",
}
MODE_COLOR = {
    "raw": "black",
    "mangasarian_time_graded": "tab:orange",
    "mangasarian_constant": "tab:green",
    "black_scholes": "tab:blue",
}
REFERENCE_LABEL = "Reiner-Rubinstein (exact, closed form)"
REFERENCE_COLOR = "black"

_LOSS_LINE_RE = re.compile(r"iter\s+(\d+)/\d+\s+loss=([0-9.eE+\-]+)")


# ---------------------------------------------------------------------------
# Run discovery and model loading
# ---------------------------------------------------------------------------

def mode_label(meta: dict) -> str:
    """Classify a run's metadata.yaml into one of MODE_ORDER."""
    hp = meta["hyperparameters"]
    if hp.get("smoothed_payoff", False):
        return f"mangasarian_{hp['grading']}"
    if hp.get("black_scholes_payoff", False):
        return "black_scholes"
    return "raw"


def discover_runs(base_dir: Path, run_dir_prefix: str) -> list[dict]:
    """Find the 4 run directories sharing *run_dir_prefix* and label each by mode.

    Raises:
        SystemExit: If the discovered set is not exactly the 4 expected modes
            (missing run, duplicate mode, or an unrelated directory sharing
            the prefix).
    """
    candidates = sorted(p for p in base_dir.glob(f"{run_dir_prefix}*") if (p / "metadata.yaml").exists())
    runs = []
    for run_dir in candidates:
        with open(run_dir / "metadata.yaml") as f:
            meta = yaml.safe_load(f)
        runs.append({"run_dir": run_dir, "meta": meta, "label": mode_label(meta)})

    labels = [r["label"] for r in runs]
    if sorted(labels) != sorted(MODE_ORDER):
        logger.error(
            f"Expected exactly the 4 modes {MODE_ORDER} under {base_dir}/{run_dir_prefix}*, "
            f"found {labels} across {len(runs)} directories: {[str(r['run_dir']) for r in runs]}"
        )
        sys.exit(1)
    runs.sort(key=lambda r: MODE_ORDER.index(r["label"]))
    return runs


def load_trained_model(run_dir: Path, meta: dict, dtype: torch.dtype = torch.float32) -> torch.nn.Module:
    """Rebuild the ETCNN ansatz from metadata.yaml and load the saved final weights.

    Forward-pass / autograd evaluation only -- no optimiser state is touched,
    no training step is taken.
    """
    hp = meta["hyperparameters"]
    contract = meta["contract"]
    epsilon = hp["epsilons"][0]
    model = pilot.build_model(
        contract["K"], contract["B"], contract["T"], epsilon, model_seed=0,
        smoothed_payoff=hp.get("smoothed_payoff", False),
        eps0=hp.get("eps0", pilot.DEFAULT_EPS0),
        grading=hp.get("grading", pilot.DEFAULT_GRADING),
        black_scholes_payoff=hp.get("black_scholes_payoff", False),
        r=contract["r"], sigma=contract["sigma"],
    )
    model_path = run_dir / "models" / f"model_eps{epsilon:g}.pt"
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model = model.to(dtype)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Analysis 1 -- relative L^2 error, global vs. excluding a near-barrier zone
# ---------------------------------------------------------------------------

def _rel_l2(error: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor) -> float:
    num = torch.linalg.vector_norm(error[mask])
    den = torch.linalg.vector_norm(reference[mask])
    return float(num / den) if den > 0 else float("nan")


def compute_error_table(runs: list[dict], margin: float) -> list[dict]:
    """For each run, evaluate on the same dense grid as pilot_down_and_out_put.py's
    own evaluate_against_closed_form, then recompute the relative L^2 error
    restricted to s >= B + margin (the "excluding a neighbourhood of the
    barrier" region), alongside the already-known full-global value.
    """
    rows = []
    for run in runs:
        meta, run_dir = run["meta"], run["run_dir"]
        contract, domain, hp = meta["contract"], meta["domain"], meta["hyperparameters"]
        # float32: matches torch.get_default_dtype() used internally by
        # evaluate_against_closed_form (and the dtype the model was trained
        # in, per metadata.yaml's dtype=float32), so the recomputed
        # rel_l2_global cross-checks exactly against the saved summary.
        model = load_trained_model(run_dir, meta, dtype=torch.float32)
        eval_result = pilot.evaluate_against_closed_form(
            model, contract["K"], contract["B"], contract["r"], contract["sigma"], contract["T"],
            domain["s_inf"], hp["corner_window"],
        )
        ss, tt = torch.meshgrid(eval_result["s_grid"], eval_result["t_grid"], indexing="ij")
        error = eval_result["learned"] - eval_result["reference"]
        reference = eval_result["reference"]

        rel_l2_global_full = _rel_l2(error, reference, torch.ones_like(ss, dtype=torch.bool))
        mask_excl_barrier = ss >= (contract["B"] + margin)
        rel_l2_excl_barrier = _rel_l2(error, reference, mask_excl_barrier)

        with open(run_dir / f"summary_eps{hp['epsilons'][0]:g}.yaml") as f:
            saved_summary = yaml.safe_load(f)
        cross_check_diff = abs(rel_l2_global_full - saved_summary["rel_l2_global"])
        if cross_check_diff > 1e-3:
            logger.warning(
                f"[{run['label']}] recomputed rel_l2_global={rel_l2_global_full:.4e} differs from "
                f"saved summary value {saved_summary['rel_l2_global']:.4e} by {cross_check_diff:.2e} "
                f"(expected: exact match up to dtype/rounding)."
            )

        rows.append({
            "mode": run["label"],
            "rel_l2_global_full": rel_l2_global_full,
            "rel_l2_global_excl_barrier": rel_l2_excl_barrier,
            "margin": margin,
        })
        logger.info(
            f"[{run['label']}] rel_L2 global: full={rel_l2_global_full:.4e}  "
            f"excl. s<B+{margin:g}: {rel_l2_excl_barrier:.4e}"
        )
    return rows


def render_error_table(rows: list[dict]) -> str:
    header = f"| {'Mode':<28} | {'rel_L2 global (full)':>22} | {'rel_L2 global (s>=B+margin)':>28} | {'Ratio':>8} |"
    sep = "|" + "-" * (len(header) - 2) + "|"
    lines = [header, sep]
    for row in rows:
        ratio = row["rel_l2_global_excl_barrier"] / row["rel_l2_global_full"] if row["rel_l2_global_full"] else float("nan")
        lines.append(
            f"| {MODE_DISPLAY_NAME[row['mode']]:<28} | {row['rel_l2_global_full']:>22.4e} "
            f"| {row['rel_l2_global_excl_barrier']:>28.4e} | {ratio:>8.3f} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analysis 2 -- loss history from training.log
# ---------------------------------------------------------------------------

def parse_loss_history(training_log_path: Path) -> tuple[list[int], list[float]]:
    iters, losses = [], []
    with open(training_log_path) as f:
        for line in f:
            match = _LOSS_LINE_RE.search(line)
            if match:
                iters.append(int(match.group(1)))
                losses.append(float(match.group(2)))
    return iters, losses


def plot_loss_history(runs: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for run in runs:
        iters, losses = parse_loss_history(run["run_dir"] / "training.log")
        ax.semilogy(iters, losses, color=MODE_COLOR[run["label"]], lw=1.5, label=MODE_DISPLAY_NAME[run["label"]])
    ax.set_xlabel("Training iteration")
    ax.set_ylabel(r"Loss $=\mathrm{mean}(\mathcal{F}(\Phi_\theta)^2)$  (log scale)")
    ax.set_title("Down-and-out put: interior PDE residual loss vs. iteration", fontsize=11)
    ax.grid(alpha=0.3, which="both")
    legend = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    formula = (
        r"$\mathcal{F}(U)=\partial_tU+\frac{1}{2}\sigma^2s^2\partial_{ss}U+rs\partial_sU-rU$"
        "\n"
        r"loss $=\mathrm{mean}_{(s,t)\sim\mathrm{collocation}}\mathcal{F}(\Phi_\theta)^2$ (interior residual only)"
        "\n"
        r"no terminal/barrier loss term: both hold exactly by construction, $\Phi_\theta=g_1u_\theta+g_2$"
    )
    fig.subplots_adjust(right=0.62, bottom=0.32)
    finalize_figure(fig, out_path, legends=[legend], formula=formula, axes=[ax])


# ---------------------------------------------------------------------------
# Analysis 3 -- ||d^2U/ds^2||_{L^2(strike window)} vs. calendar time t
# ---------------------------------------------------------------------------

def compute_curvature_vs_time(
    model: torch.nn.Module, K: float, T: float, strike_window: float, n_s_window: int, t_grid: torch.Tensor,
) -> list[float]:
    s_values = torch.linspace(K - strike_window, K + strike_window, n_s_window, dtype=torch.float64)
    curvature_norm = []
    for t_val in t_grid.tolist():
        s = s_values.clone().requires_grad_(True)
        t = torch.full_like(s, t_val)
        x = torch.stack([s, t], dim=1)
        U = model(x).squeeze()
        grad_s = torch.autograd.grad(U.sum(), s, create_graph=True)[0]
        grad_ss = torch.autograd.grad(grad_s.sum(), s)[0]
        l2_norm = torch.sqrt(torch.trapz(grad_ss**2, s_values))
        curvature_norm.append(float(l2_norm))
    return curvature_norm


def compute_reference_curvature_vs_time(
    K: float, B: float, r: float, sigma: float, T: float, strike_window: float, n_s_window: int, t_grid: torch.Tensor,
) -> list[float]:
    """Same L^2(K-delta,K+delta) norm as compute_curvature_vs_time, but of the
    exact closed-form Gamma (reiner_rubinstein_down_and_out_put_gamma) of the
    true down-and-out price V_DO -- the "ground truth" curvature that a
    perfectly-trained ansatz would reproduce, independent of any network."""
    s_values = torch.linspace(K - strike_window, K + strike_window, n_s_window, dtype=torch.float64)
    curvature_norm = []
    for t_val in t_grid.tolist():
        tau = torch.full_like(s_values, T - t_val)
        gamma = reiner_rubinstein_down_and_out_put_gamma(s_values, K, B, r, sigma, tau)
        l2_norm = torch.sqrt(torch.trapz(gamma**2, s_values))
        curvature_norm.append(float(l2_norm))
    return curvature_norm


def plot_curvature_vs_time(runs: list[dict], strike_window: float, out_path: Path) -> tuple[torch.Tensor, dict[str, list[float]]]:
    """Builds the curvature-vs-time figure and returns (t_grid, curvature_by_label)
    -- the label "reference" holds the exact Reiner-Rubinstein Gamma norm --
    so the selected-times table can be built from the same data without
    recomputing it."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    contract = runs[0]["meta"]["contract"]
    T, K, B, r, sigma = contract["T"], contract["K"], contract["B"], contract["r"], contract["sigma"]
    t_grid = torch.linspace(0.0, T - 1e-4, 60, dtype=torch.float64)
    curvature_by_label: dict[str, list[float]] = {}

    reference_curvature = compute_reference_curvature_vs_time(K, B, r, sigma, T, strike_window, n_s_window=21, t_grid=t_grid)
    curvature_by_label["reference"] = reference_curvature
    ax.semilogy(t_grid.numpy(), reference_curvature, color=REFERENCE_COLOR, lw=2.5, linestyle="--",
                label=REFERENCE_LABEL, zorder=5)
    logger.info(
        f"[reference] curvature norm: min={min(reference_curvature):.3e}  max={max(reference_curvature):.3e}  "
        f"at t={t_grid[int(np.argmax(reference_curvature))]:.4f}"
    )

    for run in runs:
        model = load_trained_model(run["run_dir"], run["meta"], dtype=torch.float64)
        curvature = compute_curvature_vs_time(model, K, T, strike_window, n_s_window=21, t_grid=t_grid)
        curvature_by_label[run["label"]] = curvature
        ax.semilogy(t_grid.numpy(), curvature, color=MODE_COLOR[run["label"]], lw=1.5,
                    label=MODE_DISPLAY_NAME[run["label"]])
        logger.info(
            f"[{run['label']}] curvature norm: min={min(curvature):.3e}  max={max(curvature):.3e}  "
            f"at t={t_grid[int(np.argmax(curvature))]:.4f}"
        )
    ax.set_xlabel("Calendar time $t$")
    ax.set_ylabel(r"$\|\partial_{ss}\Phi_\theta\|_{L^2(K-\delta,K+\delta)}$  (log scale)")
    ax.set_title(f"Down-and-out put: curvature near the strike vs. time ($\\delta={strike_window:g}$)", fontsize=11)
    ax.grid(alpha=0.3, which="both")
    legend = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    formula = (
        r"$\|\partial_{ss}\Phi_\theta\|_{L^2(K-\delta,K+\delta)}(t)=\sqrt{\int_{K-\delta}^{K+\delta}"
        r"(\partial_{ss}\Phi_\theta(s,t))^2ds}$,  $\Phi_\theta=g_1u_\theta+g_2$, $\delta=$ --strike-window"
        "\n"
        r"Reference (dashed): exact $\partial_{ss}V_{DO}$, reiner\_rubinstein\_down\_and\_out\_put\_gamma"
        "\n"
        r"Raw-mode caveat: off $s=K$, raw $g_2$ has zero curvature (piecewise-linear) --"
        "\n"
        r"the ``raw'' curve is the network's compensating response, not the kink's own curvature"
    )
    fig.subplots_adjust(right=0.62, bottom=0.36)
    finalize_figure(fig, out_path, legends=[legend], formula=formula, axes=[ax])
    return t_grid, curvature_by_label


def render_curvature_table(t_grid: torch.Tensor, curvature_by_label: dict[str, list[float]], selected_times: list[float]) -> str:
    """Table of curvature (reference + each mode) at a few selected times,
    picking the closest available grid point to each requested time."""
    columns = ["reference"] + MODE_ORDER
    column_names = {"reference": REFERENCE_LABEL, **MODE_DISPLAY_NAME}
    header = "| t | " + " | ".join(column_names[c] for c in columns) + " |"
    sep = "|---|" + "|".join("---:" for _ in columns) + "|"
    lines = [header, sep]
    t_grid_np = t_grid.numpy()
    for t_target in selected_times:
        idx = int(np.argmin(np.abs(t_grid_np - t_target)))
        t_actual = t_grid_np[idx]
        row = [f"{t_actual:.4g}"] + [f"{curvature_by_label[c][idx]:.4e}" for c in columns]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analysis 4 -- strike-zoom, LINEAR-scale slice (reveals negative prices;
# log scale, used by the pilot's own log_slice_*.png, cannot).
# ---------------------------------------------------------------------------

def plot_strike_zoom_linear(runs: list[dict], strike_window: float, t_values: list[float], out_path: Path) -> None:
    K = runs[0]["meta"]["contract"]["K"]
    s = torch.linspace(K - strike_window, K + strike_window, 401, dtype=torch.float64)

    fig, axes = plt.subplots(1, len(t_values), figsize=(4.2 * len(t_values), 4.2), sharey=True)
    if len(t_values) == 1:
        axes = [axes]

    min_price_rows = []
    for run in runs:
        model = load_trained_model(run["run_dir"], run["meta"], dtype=torch.float64)
        for ax, t_val in zip(axes, t_values):
            t = torch.full_like(s, t_val)
            x = torch.stack([s, t], dim=1)
            with torch.no_grad():
                U = model(x).squeeze()
            ax.plot(s.numpy(), U.numpy(), color=MODE_COLOR[run["label"]], lw=1.5,
                    label=MODE_DISPLAY_NAME[run["label"]])
        # separately, scan a (s, t) grid for the minimum price near the strike
        t_scan = torch.linspace(0.0, run["meta"]["contract"]["T"] - 1e-4, 60, dtype=torch.float64)
        ss, tt = torch.meshgrid(s, t_scan, indexing="ij")
        with torch.no_grad():
            x_scan = torch.stack([ss.reshape(-1), tt.reshape(-1)], dim=1)
            U_scan = model(x_scan).squeeze().reshape(ss.shape)
        min_val = float(U_scan.min())
        min_idx = int(U_scan.argmin())
        min_s = float(ss.reshape(-1)[min_idx])
        min_t = float(tt.reshape(-1)[min_idx])
        min_price_rows.append({"mode": run["label"], "min_price": min_val, "at_s": min_s, "at_t": min_t})
        logger.info(f"[{run['label']}] min price in strike window: {min_val:.4e} at (s={min_s:.4f}, t={min_t:.4f})")

    for ax, t_val in zip(axes, t_values):
        ax.axhline(0.0, color="grey", linestyle=":", linewidth=1.0)
        ax.axvline(K, color="grey", linestyle=":", linewidth=1.0)
        ax.set_xlabel("Underlying price $s$")
        ax.set_title(f"$t = {t_val:g}$")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("$\\Phi_\\theta(s,t)$  (linear scale)")
    legend = axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    formula = (
        r"Linear scale (not log): a log-scale slice cannot represent a negative price, and the pilot's own "
        r"log\_slice\_*.png floors negative values to $10^{-12}$ before plotting, which hides sign. "
        r"Dotted lines mark $\Phi_\theta=0$ and $s=K$."
    )
    fig.subplots_adjust(right=0.80, bottom=0.30)
    finalize_figure(fig, out_path, legends=[legend], formula=formula, axes=list(axes))
    return min_price_rows


# ---------------------------------------------------------------------------
# Analysis 5 -- pointwise (not L^2-averaged) second derivative at exact
# points near the strike, decomposed into Phi_theta, g2, and g1*u_theta, plus
# an explicit finite-difference-step sensitivity test at s=K exactly.
#
# Deliberately uses central finite differences on the model's own forward
# pass (model(x) for Phi_theta, model.forward_neural_manifold(x) for
# g1*u_theta, and g2 = Phi_theta - g1*u_theta by exact linearity of the
# finite-difference operator -- no separate g2 callable needed) rather than
# autograd: a genuine C^0 kink (raw mode, s=K) is only visible to a finite
# difference straddling the kink, not to autograd sampled at a generic point
# (see the analysis-3 caveat: autograd there differentiates the specific
# smooth branch it lands on and never sees the kink).
# ---------------------------------------------------------------------------

def _pointwise_second_difference(model: torch.nn.Module, s_val: float, t_val: float, h: float) -> tuple[float, float, float]:
    """Central second difference, step h, of (Phi_theta, g1*u_theta, g2) at (s_val, t_val)."""
    s_col = torch.tensor([[s_val - h], [s_val], [s_val + h]], dtype=torch.float64)
    t_col = torch.full_like(s_col, t_val)
    x = torch.cat([s_col, t_col], dim=1)
    with torch.no_grad():
        phi = model(x).squeeze(-1)
        g1u = model.forward_neural_manifold(x).squeeze(-1)
    g2 = phi - g1u
    second_diff = lambda v: float((v[2] - 2 * v[1] + v[0]) / h**2)
    return second_diff(phi), second_diff(g1u), second_diff(g2)


def compute_h_sensitivity_at_strike(
    run: dict, t_values: list[float], h_values: list[float],
) -> list[dict]:
    """At s=K exactly, sweep the finite-difference step h and report d^2 Phi_theta/ds^2
    (full trial solution) for each (t, h) -- the table that shows the raw
    mode's pointwise second derivative changing (diverging) with h, unlike a
    smoothed mode's, which stabilises."""
    K = run["meta"]["contract"]["K"]
    model = load_trained_model(run["run_dir"], run["meta"], dtype=torch.float64)
    rows = []
    for t_val in t_values:
        row = {"mode": run["label"], "t": t_val}
        for h in h_values:
            phi_dd, _, _ = _pointwise_second_difference(model, K, t_val, h)
            row[f"h={h:g}"] = phi_dd
        rows.append(row)
    return rows


def render_h_sensitivity_table(rows: list[dict], h_values: list[float]) -> str:
    h_cols = [f"h={h:g}" for h in h_values]
    header = "| Mode | t | " + " | ".join(h_cols) + " |"
    sep = "|---|---|" + "|".join("---:" for _ in h_cols) + "|"
    lines = [header, sep]
    for row in rows:
        cells = [f"{row[c]:.4e}" for c in h_cols]
        lines.append(f"| {MODE_DISPLAY_NAME[row['mode']]} | {row['t']:g} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def compute_pointwise_by_offset(
    run: dict, s_offsets: list[float], h: float, t_grid: torch.Tensor,
) -> dict[float, dict[str, list[float]]]:
    """For each s = K + offset, returns {"phi": [...], "g1u": [...], "g2": [...]} vs t_grid."""
    K = run["meta"]["contract"]["K"]
    model = load_trained_model(run["run_dir"], run["meta"], dtype=torch.float64)
    results: dict[float, dict[str, list[float]]] = {}
    for offset in s_offsets:
        s_val = K + offset
        series = {"phi": [], "g1u": [], "g2": []}
        for t_val in t_grid.tolist():
            phi_dd, g1u_dd, g2_dd = _pointwise_second_difference(model, s_val, t_val, h)
            series["phi"].append(phi_dd)
            series["g1u"].append(g1u_dd)
            series["g2"].append(g2_dd)
        results[offset] = series
    return results


def plot_pointwise_second_derivatives(
    pointwise_runs: list[dict], s_offsets: list[float], h: float, t_grid: torch.Tensor, out_path: Path,
) -> dict[str, dict[float, dict[str, list[float]]]]:
    """2 (mode) x 3 (Phi_theta, g1*u_theta, g2) grid, each panel showing the
    pointwise second derivative vs t at every s=K+offset (colour-coded,
    sequential palette: the sweep axis here is the s-offset)."""
    n_modes = len(pointwise_runs)
    fig, axes = plt.subplots(n_modes, 3, figsize=(15.0, 4.6 * n_modes), squeeze=False)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(s_offsets)))
    quantity_key = ["phi", "g1u", "g2"]
    quantity_title = {
        "phi": r"$\partial_{ss}\Phi_\theta$ (full trial solution)",
        "g1u": r"$\partial_{ss}(g_1u_\theta)$ (network term)",
        "g2": r"$\partial_{ss}g_2$ (terminal-function term)",
    }
    data_by_mode: dict[str, dict[float, dict[str, list[float]]]] = {}

    for row_idx, run in enumerate(pointwise_runs):
        by_offset = compute_pointwise_by_offset(run, s_offsets, h, t_grid)
        data_by_mode[run["label"]] = by_offset
        for col_idx, key in enumerate(quantity_key):
            ax = axes[row_idx][col_idx]
            for color, offset in zip(colors, s_offsets):
                sign = "+" if offset >= 0 else ""
                ax.plot(t_grid.numpy(), by_offset[offset][key], color=color, lw=1.5,
                        label=fr"$s=K{sign}{offset:g}$")
            ax.set_yscale("symlog", linthresh=1e-2)
            ax.set_xlabel("Calendar time $t$")
            if col_idx == 0:
                ax.set_ylabel(f"{MODE_DISPLAY_NAME[run['label']]}\n\nvalue (symlog scale)")
            ax.set_title(quantity_title[key], fontsize=10)
            ax.grid(alpha=0.3)
            if row_idx == 0 and col_idx == 2:
                legend = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)

    formula = (
        r"Central 2nd difference, step $h$ (fixed), of the model's own forward pass: "
        r"$\partial_{ss}\Phi_\theta$, $\partial_{ss}(g_1u_\theta)$ direct; $\partial_{ss}g_2$ by "
        r"subtraction (exact, linearity of finite differences), $h=$" + f"{h:g}"
        "\n"
        r"symlog $y$-axis (linear near 0, log beyond $10^{-2}$): needed because raw-mode $\partial_{ss}g_2$ "
        r"at $s=K$ is orders of magnitude above all other curves"
    )
    fig.subplots_adjust(right=0.82, bottom=0.10, hspace=0.35, wspace=0.30)
    finalize_figure(fig, out_path, legends=[legend], formula=formula, axes=[a for row in axes for a in row])
    return data_by_mode


def render_l2_vs_pointwise_table(
    curvature_by_label: dict[str, list[float]], t_grid_l2: torch.Tensor,
    pointwise_data_by_mode: dict[str, dict[float, dict[str, list[float]]]], t_grid_pointwise: torch.Tensor,
    selected_times: list[float],
) -> str:
    """At each selected t, compares the L^2(K-delta,K+delta) window-average
    curvature (analysis 3) against the pointwise value exactly at s=K
    (offset 0.0) from this analysis, for each mode present in both."""
    modes = [m for m in pointwise_data_by_mode if 0.0 in pointwise_data_by_mode[m]]
    header = "| t | " + " | ".join(f"{MODE_DISPLAY_NAME[m]} -- L2 norm | {MODE_DISPLAY_NAME[m]} -- pointwise at s=K" for m in modes) + " |"
    sep = "|---|" + "|".join("---:|---:" for _ in modes) + "|"
    lines = [header, sep]
    t_grid_l2_np = t_grid_l2.numpy()
    t_grid_pw_np = t_grid_pointwise.numpy()
    for t_target in selected_times:
        idx_l2 = int(np.argmin(np.abs(t_grid_l2_np - t_target)))
        idx_pw = int(np.argmin(np.abs(t_grid_pw_np - t_target)))
        cells = []
        for m in modes:
            l2_val = curvature_by_label[m][idx_l2]
            pw_val = pointwise_data_by_mode[m][0.0]["phi"][idx_pw]
            cells.append(f"{l2_val:.4e} | {pw_val:.4e}")
        lines.append(f"| {t_grid_l2_np[idx_l2]:.4g} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analysis 6 -- is the network's compensating curvature at s=K physically
# sensible? Compares the pointwise total d^2 Phi_theta/ds^2 (well-converged,
# small fixed h) against the exact Reiner-Rubinstein Gamma at the same (s=K,
# t) point. Only meaningful for a mode whose pointwise second derivative is
# itself well-defined (h-independent) -- i.e. a smoothed mode, not raw (see
# analysis 5's h-sensitivity table: raw's pointwise value has no h->0 limit).
# ---------------------------------------------------------------------------

def compute_pointwise_phi_vs_true_gamma(
    run: dict, t_grid: torch.Tensor, h: float = 1e-4,
) -> dict[str, list[float]]:
    contract = run["meta"]["contract"]
    K, B, r, sigma, T = contract["K"], contract["B"], contract["r"], contract["sigma"], contract["T"]
    model = load_trained_model(run["run_dir"], run["meta"], dtype=torch.float64)
    s = torch.tensor([K], dtype=torch.float64)
    phi_values, true_gamma_values = [], []
    for t_val in t_grid.tolist():
        phi_dd, _, _ = _pointwise_second_difference(model, K, t_val, h)
        tau = torch.tensor([T - t_val], dtype=torch.float64)
        true_gamma = float(reiner_rubinstein_down_and_out_put_gamma(s, K, B, r, sigma, tau))
        phi_values.append(phi_dd)
        true_gamma_values.append(true_gamma)
    return {"phi": phi_values, "true_gamma": true_gamma_values}


def plot_pointwise_phi_vs_true_gamma(run: dict, t_grid: torch.Tensor, data: dict[str, list[float]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(t_grid.numpy(), data["true_gamma"], color="black", lw=2.5, linestyle="--",
             label=r"True Gamma, $\partial_{ss}V_{DO}(K,t)$ (exact)")
    ax.plot(t_grid.numpy(), data["phi"], color=MODE_COLOR[run["label"]], lw=1.5,
             label=fr"$\partial_{{ss}}\Phi_\theta(K,t)$ ({MODE_DISPLAY_NAME[run['label']]})")
    ax.axhline(0.0, color="grey", lw=0.7)
    ax.set_yscale("symlog", linthresh=1e-1)
    ax.set_xlabel("Calendar time $t$")
    ax.set_ylabel(r"$\partial_{ss}\cdot(K,t)$  (symlog scale)")
    ax.set_title(f"Pointwise curvature at $s=K$: trained total vs. true Gamma ({run['label']})", fontsize=11)
    ax.grid(alpha=0.3, which="both")
    legend = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    formula = (
        r"$\partial_{ss}\Phi_\theta(K,t)$: central 2nd difference, $h=10^{-4}$ (converged, see analysis-5 "
        r"h-sensitivity table), on the trained model's own forward pass"
        "\n"
        r"True Gamma: reiner\_rubinstein\_down\_and\_out\_put\_gamma$(K,t)$, exact closed form, no network"
    )
    fig.subplots_adjust(right=0.62, bottom=0.30)
    finalize_figure(fig, out_path, legends=[legend], formula=formula, axes=[ax])


def render_phi_vs_true_gamma_table(t_grid: torch.Tensor, data: dict[str, list[float]], selected_times: list[float]) -> str:
    header = "| t | True Gamma (exact) | $\\partial_{ss}\\Phi_\\theta$ (trained, pointwise) | Abs. diff | Ratio |"
    sep = "|---|---:|---:|---:|---:|"
    lines = [header, sep]
    t_grid_np = t_grid.numpy()
    for t_target in selected_times:
        idx = int(np.argmin(np.abs(t_grid_np - t_target)))
        true_gamma = data["true_gamma"][idx]
        phi = data["phi"][idx]
        ratio = phi / true_gamma if abs(true_gamma) > 1e-6 else float("nan")
        lines.append(f"| {t_grid_np[idx]:.4g} | {true_gamma:.4e} | {phi:.4e} | {phi - true_gamma:.4e} | {ratio:.3f} |")
    return "\n".join(lines)


def render_combined_phi_vs_true_gamma_table(
    t_grid: torch.Tensor, data_by_mode: dict[str, dict[str, list[float]]], mode_order: list[str], selected_times: list[float],
) -> str:
    """One table, all requested modes side by side against the same True Gamma
    column -- the direct "which g2 tracks the true Gamma best" comparison."""
    header_cells = ["t", "True Gamma (exact)"] + [f"{MODE_DISPLAY_NAME[m]} -- $\\Phi_\\theta$" for m in mode_order] + [f"{MODE_DISPLAY_NAME[m]} -- abs. diff" for m in mode_order]
    header = "| " + " | ".join(header_cells) + " |"
    sep = "|" + "|".join("---:" for _ in header_cells) + "|"
    lines = [header, sep]
    t_grid_np = t_grid.numpy()
    for t_target in selected_times:
        idx = int(np.argmin(np.abs(t_grid_np - t_target)))
        true_gamma = data_by_mode[mode_order[0]]["true_gamma"][idx]  # identical across modes (same contract)
        phi_cells = [f"{data_by_mode[m]['phi'][idx]:.4e}" for m in mode_order]
        diff_cells = [f"{data_by_mode[m]['phi'][idx] - true_gamma:.4e}" for m in mode_order]
        lines.append(f"| {t_grid_np[idx]:.4g} | {true_gamma:.4e} | " + " | ".join(phi_cells) + " | " + " | ".join(diff_cells) + " |")
    return "\n".join(lines)


def render_mean_abs_error_summary(
    t_grid: torch.Tensor, data_by_mode: dict[str, dict[str, list[float]]], mode_order: list[str], t_max: float,
) -> str:
    """Single-number-per-mode summary: mean |Phi_theta - true Gamma| over
    t in [0, t_max] -- a compact score for "which g2 tracks the true Gamma
    best" across the whole range, complementing the pointwise table."""
    t_grid_np = t_grid.numpy()
    mask = t_grid_np <= t_max
    header = "| Mode | Mean $|\\Phi_\\theta - \\text{true Gamma}|$ over $t\\in[0," + f"{t_max:g}" + "]$ |"
    sep = "|---|---:|"
    lines = [header, sep]
    for m in mode_order:
        phi = np.array(data_by_mode[m]["phi"])[mask]
        true_gamma = np.array(data_by_mode[m]["true_gamma"])[mask]
        mae = float(np.mean(np.abs(phi - true_gamma)))
        lines.append(f"| {MODE_DISPLAY_NAME[m]} | {mae:.4e} |")
    return "\n".join(lines)


def plot_combined_phi_vs_true_gamma(
    t_grid: torch.Tensor, data_by_mode: dict[str, dict[str, list[float]]], mode_order: list[str], out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(t_grid.numpy(), data_by_mode[mode_order[0]]["true_gamma"], color=REFERENCE_COLOR, lw=2.5, linestyle="--",
             label=r"True Gamma, $\partial_{ss}V_{DO}(K,t)$ (exact)", zorder=5)
    for m in mode_order:
        ax.plot(t_grid.numpy(), data_by_mode[m]["phi"], color=MODE_COLOR[m], lw=1.5,
                 label=fr"$\partial_{{ss}}\Phi_\theta(K,t)$ ({MODE_DISPLAY_NAME[m]})")
    ax.axhline(0.0, color="grey", lw=0.7)
    ax.set_yscale("symlog", linthresh=1e-1)
    ax.set_xlabel("Calendar time $t$")
    ax.set_ylabel(r"$\partial_{ss}\cdot(K,t)$  (symlog scale)")
    ax.set_title("Pointwise curvature at $s=K$: which g2 tracks the true Gamma best?", fontsize=11)
    ax.grid(alpha=0.3, which="both")
    legend = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    formula = (
        r"$\partial_{ss}\Phi_\theta(K,t)$: central 2nd difference, $h=10^{-4}$, on each trained model's own "
        r"forward pass; True Gamma: reiner\_rubinstein\_down\_and\_out\_put\_gamma$(K,t)$, exact, no network"
    )
    fig.subplots_adjust(right=0.62, bottom=0.28)
    finalize_figure(fig, out_path, legends=[legend], formula=formula, axes=[ax])


# ---------------------------------------------------------------------------
# Analysis 7 -- mechanism: why does the raw mode's L^2(K-delta,K+delta)
# curvature extinguish towards t=T while the time-graded mode's explodes?
# The L^2-norm figure of analysis 3 shows *that* this happens but not *why*;
# this analysis decomposes the norm into its g2 (closed form where possible)
# and g1*u_theta (autograd, no closed form for a trained network) pieces.
#
# Along the way this uncovered a genuine numerical defect in the analysis-3
# figure: its L^2 norm is a 21-point trapezoidal quadrature over a fixed
# window (K-delta,K+delta); once the Mangasarian bandwidth epsilon(t) shrinks
# below the ~delta/10 grid spacing (t gtrsim 0.9 for time-graded), that grid
# no longer resolves the g2 bump at all, and the quadrature *overestimates*
# the true L^2 norm by an order of magnitude or more (quantified below via an
# exact closed form for g2's contribution, cross-checked against a 2-million-
# point quadrature to machine precision).
# ---------------------------------------------------------------------------

def mangasarian_g2_l2_norm_exact(eps: float, delta: float) -> float:
    r"""Exact closed form of :math:`\|\partial_{ss}g_{\varepsilon_0}\|_{L^2(-\delta,\delta)}`
    for the Chen-Mangasarian smoothed payoff (see the report's derivation:
    substitution :math:`x=\varepsilon\tan\phi`, then the standard reduction
    formula for :math:`\int\cos^4\phi\,d\phi`). No quadrature involved --
    this is the true continuum L^2 norm, not a discretised approximation.

    .. math::

        \|\partial_{ss}g_{\varepsilon_0}\|_{L^2(-\delta,\delta)}^2
        = \frac1{4\varepsilon}\left(\frac{3\phi_0}4+\frac{\sin2\phi_0}2+\frac{\sin4\phi_0}{16}\right),
        \qquad \phi_0=\arctan(\delta/\varepsilon).

    Cross-checked during development against a 2,000,001-point trapezoidal
    quadrature of the same integral: relative difference <= 3e-14 (machine
    precision) for eps spanning 5e-2 down to 5e-6.
    """
    phi0 = np.arctan(delta / eps)
    integral = (3.0 * phi0 / 4.0) + (np.sin(2.0 * phi0) / 2.0) + (np.sin(4.0 * phi0) / 16.0)
    integral /= 4.0 * eps
    return float(np.sqrt(integral))


def black_scholes_g2_l2_norm_quadrature(K: float, r: float, sigma: float, tau: float, delta: float, n: int = 4001) -> float:
    """Fine (n-point, converged) trapezoidal quadrature of the exact vanilla
    Gamma over (K-delta,K+delta) -- no closed form is derived for this
    windowed L^2 norm (unlike the Mangasarian case), but n=4001 resolves it
    to convergence for the tau values used here (tau not shrinking as fast
    as the Mangasarian epsilon(t), so no under-resolution risk)."""
    tau_safe = max(tau, 1e-8)
    x = np.linspace(K - delta, K + delta, n)
    d_plus = (np.log(x / K) + (r + 0.5 * sigma**2) * tau_safe) / (sigma * np.sqrt(tau_safe))
    gamma = np.exp(-0.5 * d_plus**2) / np.sqrt(2.0 * np.pi) / (x * sigma * np.sqrt(tau_safe))
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(np.sqrt(trapz(gamma**2, x)))


def compute_l2_decomposition_vs_time(
    run: dict, strike_window: float, n_s_window: int, t_grid: torch.Tensor,
) -> dict[str, list[float]]:
    """Decomposes the L^2(K-delta,K+delta) curvature norm into:

    - ``g2_exact``: the g2-alone contribution, in closed form wherever one is
      available -- identically 0 for raw (Prop.: (K-s)^+ is affine a.e., the
      single non-differentiable point has Lebesgue measure zero and does not
      contribute to the integral), :func:`mangasarian_g2_l2_norm_exact` for a
      smoothed mode, :func:`black_scholes_g2_l2_norm_quadrature` for
      black_scholes.
    - ``g1u``: the g1*u_theta contribution, via autograd on
      ``model.forward_neural_manifold`` (no closed form exists for a trained
      network).
    - ``phi``: the total, via autograd on ``model`` directly -- reproduces
      analysis 3's own number, included here as an internal cross-check.
    """
    meta = run["meta"]
    hp = meta["hyperparameters"]
    contract = meta["contract"]
    K, B, r, sigma, T = contract["K"], contract["B"], contract["r"], contract["sigma"], contract["T"]
    model = load_trained_model(run["run_dir"], meta, dtype=torch.float64)
    s_values = torch.linspace(K - strike_window, K + strike_window, n_s_window, dtype=torch.float64)

    g2_exact, g1u_l2, phi_l2 = [], [], []
    for t_val in t_grid.tolist():
        s = s_values.clone().requires_grad_(True)
        t = torch.full_like(s, t_val)
        x = torch.stack([s, t], dim=1)

        g1u_val = model.forward_neural_manifold(x).squeeze(-1)
        grad_s = torch.autograd.grad(g1u_val.sum(), s, create_graph=True)[0]
        grad_ss = torch.autograd.grad(grad_s.sum(), s)[0]
        g1u_l2.append(float(torch.sqrt(torch.trapz(grad_ss**2, s_values))))

        s2 = s_values.clone().requires_grad_(True)
        t2 = torch.full_like(s2, t_val)
        x2 = torch.stack([s2, t2], dim=1)
        phi_val = model(x2).squeeze(-1)
        grad_s2 = torch.autograd.grad(phi_val.sum(), s2, create_graph=True)[0]
        grad_ss2 = torch.autograd.grad(grad_s2.sum(), s2)[0]
        phi_l2.append(float(torch.sqrt(torch.trapz(grad_ss2**2, s_values))))

        if hp.get("smoothed_payoff", False):
            eps_t = hp["eps0"] * (T - t_val) / T if hp["grading"] == "time_graded" else hp["eps0"]
            g2_exact.append(mangasarian_g2_l2_norm_exact(eps_t, strike_window))
        elif hp.get("black_scholes_payoff", False):
            g2_exact.append(black_scholes_g2_l2_norm_quadrature(K, r, sigma, T - t_val, strike_window))
        else:
            g2_exact.append(0.0)

    return {"g2_exact": g2_exact, "g1u": g1u_l2, "phi": phi_l2}


def plot_mechanism_figure(
    runs: list[dict], strike_window: float, t_grid: torch.Tensor, out_path: Path,
) -> dict[str, dict[str, list[float]]]:
    """4-panel figure explaining the mechanism behind analysis 3's L^2-norm
    plot: (a) g1(K,t), the network's multiplicative weight, decaying to 0 at
    t=T -- the ONLY source of curvature left for raw once g2's contribution
    vanishes a.e.; (b) epsilon(t) and its reciprocal 1/(2 epsilon(t)), the
    Mangasarian time-graded g2's own pointwise peak at s=K, exploding
    independently of the network; (c)/(d) the L^2-norm decomposition
    (g2_exact / g1u / phi) vs t, one panel per mode, so the dominant term is
    visible directly."""
    contract = runs[0]["meta"]["contract"]
    K, B, T = contract["K"], contract["B"], contract["T"]
    t_np = t_grid.numpy()

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.5))
    ax_g1, ax_eps, ax_raw, ax_tg = axes[0][0], axes[0][1], axes[1][0], axes[1][1]

    g1_at_K = (T - t_np) * (K - B)
    ax_g1.plot(t_np, g1_at_K, color="black", lw=2)
    ax_g1.set_xlabel("Calendar time $t$")
    ax_g1.set_ylabel("$g_1(K,t)=(T-t)(K-B)$")
    ax_g1.set_title("(a) Network weight $g_1$ -- the raw mode's only curvature source", fontsize=10)
    ax_g1.grid(alpha=0.3)

    time_graded_run = next((r for r in runs if r["label"] == "mangasarian_time_graded"), None)
    if time_graded_run is not None:
        eps0 = time_graded_run["meta"]["hyperparameters"]["eps0"]
        eps_t = eps0 * (T - t_np) / T
        ax_eps2 = ax_eps.twinx()
        l1, = ax_eps.plot(t_np, eps_t, color="tab:orange", lw=2, label=r"$\varepsilon(t)$")
        l2, = ax_eps2.plot(t_np, 1.0 / (2.0 * np.maximum(eps_t, 1e-12)), color="tab:red", lw=2, linestyle="--",
                            label=r"$1/(2\varepsilon(t))=\partial_{ss}g_{\varepsilon_0}(K,t)$")
        ax_eps2.set_yscale("log")
        ax_eps.set_xlabel("Calendar time $t$")
        ax_eps.set_ylabel(r"$\varepsilon(t)$", color="tab:orange")
        ax_eps2.set_ylabel(r"$1/(2\varepsilon(t))$  (log scale)", color="tab:red")
        ax_eps.set_title("(b) Time-graded bandwidth $\\to0$ and its pointwise peak $\\to\\infty$", fontsize=10)
        ax_eps.legend(handles=[l1, l2], loc="upper center", fontsize=8)
        ax_eps.grid(alpha=0.3)

    data_by_mode: dict[str, dict[str, list[float]]] = {}
    for ax, label in [(ax_raw, "raw"), (ax_tg, "mangasarian_time_graded")]:
        run = next((r for r in runs if r["label"] == label), None)
        if run is None:
            continue
        data = compute_l2_decomposition_vs_time(run, strike_window, 21, t_grid)
        data_by_mode[label] = data
        ax.semilogy(t_np, np.maximum(np.abs(data["g2_exact"]), 1e-12), color="tab:green", lw=1.5,
                    label=r"$\|\partial_{ss}g_2\|_{L^2}$ (exact)")
        ax.semilogy(t_np, np.abs(data["g1u"]), color="tab:purple", lw=1.5,
                    label=r"$\|\partial_{ss}(g_1u_\theta)\|_{L^2}$ (autograd)")
        ax.semilogy(t_np, data["phi"], color=MODE_COLOR[label], lw=2.0, linestyle=":",
                    label=r"$\|\partial_{ss}\Phi_\theta\|_{L^2}$ (total, autograd)")
        ax.set_xlabel("Calendar time $t$")
        ax.set_ylabel("$L^2(K-\\delta,K+\\delta)$ norm  (log scale)")
        ax.set_title(f"({'c' if label == 'raw' else 'd'}) L^2-norm decomposition -- {MODE_DISPLAY_NAME[label]}", fontsize=10)
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="upper left", fontsize=7)

    formula = (
        r"(c)/(d): $g_2$ exact (raw: $0$ a.e., Prop. -- see report; Mangasarian: closed form "
        r"$\|\partial_{ss}g_{\varepsilon_0}\|_{L^2(-\delta,\delta)}$, no quadrature); $g_1u_\theta$/$\Phi_\theta$: "
        r"autograd, $\delta=$" + f"{strike_window:g}"
    )
    fig.subplots_adjust(hspace=0.35, wspace=0.30, bottom=0.08)
    finalize_figure(fig, out_path, formula=formula, axes=list(axes.flat))
    return data_by_mode


def render_l2_norm_correction_table(t_grid: torch.Tensor, eps0: float, T: float, delta: float, n_crude: int, selected_times: list[float]) -> str:
    """Quantifies the crude-quadrature defect identified above: exact
    (closed-form) vs. crude (n_crude-point trapz, analysis 3's own
    resolution) L^2 norm of the time-graded g2 alone, at selected t."""
    header = "| t | $\\varepsilon(t)$ | Exact $\\|\\partial_{ss}g_2\\|_{L^2}$ | Crude (n=" + f"{n_crude}" + f") $\\|\\partial_{{ss}}g_2\\|_{{L^2}}$ | Ratio crude/exact |"
    sep = "|---|---:|---:|---:|---:|"
    lines = [header, sep]
    trapz = getattr(np, "trapezoid", None) or np.trapz
    for t_val in selected_times:
        eps_t = eps0 * (T - t_val) / T
        exact = mangasarian_g2_l2_norm_exact(eps_t, delta)
        x = np.linspace(-delta, delta, n_crude)
        g = eps_t**2 / (2.0 * (x**2 + eps_t**2)**1.5)
        crude = float(np.sqrt(trapz(g**2, x)))
        lines.append(f"| {t_val:g} | {eps_t:.3e} | {exact:.4f} | {crude:.4f} | {crude/exact:.3f} |")
    return "\n".join(lines)


def render_exact_gamma_at_strike_table(runs: list[dict], selected_times: list[float]) -> str:
    """Closed-form (no autograd, no finite difference, no network) pointwise
    d^2 g2/ds^2 at s=K for every mode where such a formula exists, plus the
    true reference Gamma -- the "exact values" table."""
    contract = runs[0]["meta"]["contract"]
    K, B, r, sigma, T = contract["K"], contract["B"], contract["r"], contract["sigma"], contract["T"]
    header = "| t | Brut (kink) | Mangasarian time-graded, $1/(2\\varepsilon(t))$ | Mangasarian constant, $1/(2\\varepsilon_0)$ | Black-Scholes, $\\Gamma(K,K)$ | Référence $\\partial_{ss}V_{DO}(K,t)$ |"
    sep = "|---|---|---:|---:|---:|---:|"
    lines = [header, sep]
    eps0_tg = next((r["meta"]["hyperparameters"]["eps0"] for r in runs if r["label"] == "mangasarian_time_graded"), 0.05)
    eps0_const = next((r["meta"]["hyperparameters"]["eps0"] for r in runs if r["label"] == "mangasarian_constant"), 0.05)
    for t_val in selected_times:
        tau = T - t_val
        eps_tg = eps0_tg * tau / T
        mang_tg = 1.0 / (2.0 * eps_tg)
        mang_const = 1.0 / (2.0 * eps0_const)
        tau_safe = max(tau, 1e-8)
        d_plus_KK = (r + 0.5 * sigma**2) * np.sqrt(tau_safe) / sigma
        gamma_bs = float(np.exp(-0.5 * d_plus_KK**2) / np.sqrt(2.0 * np.pi) / (K * sigma * np.sqrt(tau_safe)))
        s_t = torch.tensor([K], dtype=torch.float64)
        tau_t = torch.tensor([tau], dtype=torch.float64)
        gamma_true = float(reiner_rubinstein_down_and_out_put_gamma(s_t, K, B, r, sigma, tau_t))
        lines.append(f"| {t_val:g} | non défini (Dirac) | {mang_tg:.4f} | {mang_const:.4f} | {gamma_bs:.4f} | {gamma_true:.4f} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analysis 8 -- Gamma profiles as a function of the underlying price s, at a
# few FIXED calendar times t.
#
# Analyses 3, 5 and 6 all hold s fixed (at the strike, or at a small set of
# offsets from it) and sweep the calendar time t. This analysis is their
# transpose: t is held fixed and the whole s-profile of the second price
# derivative is drawn, over a range spanning the barrier corner layer
# {B <= s <= B+epsilon} and the strike.
#
# The quantity drawn is the second price derivative of the FULL trial
# solution, Phi_theta = g_1 u_theta + g_2 -- the trained price -- not of the
# corner-regularised extension h_epsilon = g_2 alone. The two differ by
# partial_ss(g_1 u_theta), which is exactly the network's contribution; the
# companion decomposition figure draws all three terms separately so the
# split between "what the extension supplies" and "what the network has
# learned" is visible pointwise in s.
#
# Method: two nested torch.autograd.grad calls on the model's own forward
# pass -- model(x) for partial_ss Phi_theta, model.forward_neural_manifold(x)
# for partial_ss(g_1 u_theta) -- and partial_ss g_2 by subtraction, exact by
# linearity of the differential operator. The exact Gamma of the true
# down-and-out price (reiner_rubinstein_down_and_out_put_gamma, closed form,
# no network) overlays the total as the dashed reference.
#
# Caveat for the raw mode (same one as analysis 3): the raw g_2 = (K-s)^+ has
# a first-derivative discontinuity at s=K, but that point has Lebesgue
# measure zero and the s-grid used here does not, in general, contain it.
# Autograd evaluated at the surrounding grid points differentiates the
# locally affine branch, whose second derivative is exactly 0; the raw curve
# therefore reports the network's compensating curvature, never the
# (distributional, Dirac) curvature of the kink itself. Analysis 5's
# finite-difference h-sweep is the tool that exhibits the kink directly.
# ---------------------------------------------------------------------------

def _value_and_price_derivatives_by_autograd(
    value_fn, s_values: torch.Tensor, t_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Value, first and second derivative in s of ``value_fn(x)`` at fixed
    ``t_value``, by two nested autograd passes over the whole ``s_values``
    grid at once."""
    s = s_values.clone().requires_grad_(True)
    t = torch.full_like(s, t_value)
    x = torch.stack([s, t], dim=1)
    value = value_fn(x).squeeze(-1)
    first_derivative = torch.autograd.grad(value.sum(), s, create_graph=True)[0]
    second_derivative = torch.autograd.grad(first_derivative.sum(), s)[0]
    return (value.detach().numpy(), first_derivative.detach().numpy(),
            second_derivative.detach().numpy())


def _reference_value_and_price_derivatives(
    s_values: torch.Tensor, t_value: float, K: float, B: float, r: float, sigma: float, T: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Exact :math:`V_{DO}`, :math:`\partial_sV_{DO}` and
    :math:`\partial_{ss}V_{DO}` at fixed ``t_value``.

    The value is the closed form
    :func:`~learning_option_pricing.pricing.barrier.reiner_rubinstein_down_and_out_put`;
    the two derivatives are obtained by autograd **applied to that closed
    form** -- not to any network and not by a finite difference -- so they are
    the exact analytic derivatives up to floating-point round-off.  Verified
    during development against the independently implemented closed-form
    Gamma
    (:func:`~learning_option_pricing.pricing.barrier.reiner_rubinstein_down_and_out_put_gamma`):
    relative L^2 discrepancy 4.7e-16 to 9.5e-16 over s in [0.62, 2] at
    t in {0, 0.5, 0.9, 0.99}, i.e. machine precision.
    """
    s = s_values.clone().requires_grad_(True)
    tau = torch.full_like(s, T - t_value)
    value = reiner_rubinstein_down_and_out_put(s, K, B, r, sigma, tau)
    first_derivative = torch.autograd.grad(value.sum(), s, create_graph=True)[0]
    second_derivative = torch.autograd.grad(first_derivative.sum(), s)[0]
    return (value.detach().numpy(), first_derivative.detach().numpy(),
            second_derivative.detach().numpy())


PROFILE_QUANTITIES = ["value", "delta", "gamma"]
PROFILE_QUANTITY_TITLE = {
    "value": r"$\Phi_\theta(s,t)$ (price)",
    "delta": r"$\partial_s\Phi_\theta(s,t)$ (Delta)",
    "gamma": r"$\partial_{ss}\Phi_\theta(s,t)$ (Gamma)",
}
PROFILE_REFERENCE_TITLE = {
    "value": r"$V_{DO}(s,t)$",
    "delta": r"$\partial_sV_{DO}(s,t)$",
    "gamma": r"$\partial_{ss}V_{DO}(s,t)$",
}


def compute_price_and_greek_profiles_vs_price(
    runs: list[dict], s_values: torch.Tensor, t_values: list[float],
) -> dict:
    """Price, Delta and Gamma of the full trial solution -- and, for the
    Gamma, its split into the network term and the extension term -- as
    functions of s, at each fixed t.

    Returns:
        A dict with keys ``"s_values"`` (array, n_s), ``"t_values"`` (list,
        n_t), ``"reference"`` (a dict with ``"value"``/``"delta"``/``"gamma"``,
        the exact closed-form quantities) and, for every mode label, a dict
        with ``"value"``, ``"delta"``, ``"gamma"``, ``"gamma_g1u"`` and
        ``"gamma_g2"``, each an (n_t x n_s) array.
    """
    contract = runs[0]["meta"]["contract"]
    K, B, r, sigma, T = contract["K"], contract["B"], contract["r"], contract["sigma"], contract["T"]
    s_numpy = s_values.numpy()
    n_t, n_s = len(t_values), len(s_values)

    reference = {key: np.zeros((n_t, n_s)) for key in PROFILE_QUANTITIES}
    for time_index, t_value in enumerate(t_values):
        value, delta, gamma = _reference_value_and_price_derivatives(s_values, t_value, K, B, r, sigma, T)
        reference["value"][time_index], reference["delta"][time_index], reference["gamma"][time_index] = value, delta, gamma
        logger.info(
            f"[reference] t={t_value:g}: V_DO in [{value.min():.4e}, {value.max():.4e}], "
            f"Delta in [{delta.min():.4e}, {delta.max():.4e}], Gamma in [{gamma.min():.4e}, {gamma.max():.4e}]"
        )

    profiles: dict = {"s_values": s_numpy, "t_values": list(t_values), "reference": reference}
    for run in runs:
        model = load_trained_model(run["run_dir"], run["meta"], dtype=torch.float64)
        by_quantity = {key: np.zeros((n_t, n_s)) for key in PROFILE_QUANTITIES + ["gamma_g1u", "gamma_g2"]}
        for time_index, t_value in enumerate(t_values):
            value, delta, gamma = _value_and_price_derivatives_by_autograd(model, s_values, t_value)
            _, _, gamma_g1u = _value_and_price_derivatives_by_autograd(model.forward_neural_manifold, s_values, t_value)
            by_quantity["value"][time_index] = value
            by_quantity["delta"][time_index] = delta
            by_quantity["gamma"][time_index] = gamma
            by_quantity["gamma_g1u"][time_index] = gamma_g1u
            by_quantity["gamma_g2"][time_index] = gamma - gamma_g1u  # exact: linearity of d^2/ds^2
            index_at_strike = int(np.argmin(np.abs(s_numpy - K)))
            peak_index = int(np.argmax(np.abs(gamma)))
            logger.info(
                f"[{run['label']}] t={t_value:g}: at s=K -> price {value[index_at_strike]:.4e}, "
                f"Delta {delta[index_at_strike]:.4e}, Gamma {gamma[index_at_strike]:.4e}; "
                f"max |Gamma| = {abs(gamma[peak_index]):.4e} at s={s_numpy[peak_index]:.4f}"
            )
        profiles[run["label"]] = by_quantity
    return profiles


def _annotate_price_landmarks(ax, K: float, B: float, epsilon: float) -> None:
    """Dotted vertical markers at the barrier, the corner-layer edge and the strike.

    Only the landmarks inside the current x-range are drawn: on a strike zoom
    the barrier is far off-range and its marker would silently rescale the axis.
    """
    x_lo, x_hi = ax.get_xlim()
    for landmark in (B, B + epsilon, K):
        if x_lo <= landmark <= x_hi:
            ax.axvline(landmark, color="grey", linestyle=":", linewidth=1.0)
    ax.axhline(0.0, color="grey", linewidth=0.7)


def plot_gamma_profiles_vs_price(
    profiles: dict, mode_order: list[str], K: float, B: float, epsilon: float,
    linear_threshold: float, out_path: Path,
) -> None:
    """One panel per fixed t: the second price derivative of the full trained
    price Phi_theta as a function of s, all modes overlaid, against the exact
    Reiner-Rubinstein Gamma."""
    t_values = profiles["t_values"]
    s_numpy = profiles["s_values"]
    fig, axes = plt.subplots(1, len(t_values), figsize=(4.6 * len(t_values), 4.6), squeeze=False)
    axes = list(axes[0])

    for time_index, (ax, t_value) in enumerate(zip(axes, t_values)):
        ax.plot(s_numpy, profiles["reference"]["gamma"][time_index], color=REFERENCE_COLOR, lw=2.2,
                linestyle="--", label=REFERENCE_LABEL, zorder=5)
        for label in mode_order:
            ax.plot(s_numpy, profiles[label]["gamma"][time_index], color=MODE_COLOR[label], lw=1.4,
                    label=MODE_DISPLAY_NAME[label])
        _annotate_price_landmarks(ax, K, B, epsilon)
        ax.set_yscale("symlog", linthresh=linear_threshold)
        ax.set_xlabel("Underlying price $s$")
        ax.set_title(f"$t = {t_value:g}$", fontsize=11)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel(r"$\partial_{ss}\Phi_\theta(s,t)$  (symlog scale)")
    legend = axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)

    formula = (
        r"$\partial_{ss}\Phi_\theta(s,t)$, $\Phi_\theta=g_1u_\theta+g_2$ -- the second price derivative of the "
        r"FULL trained price, not of the extension $h_\varepsilon=g_2$ alone; two nested autograd passes on the "
        r"trained model's own forward pass"
        "\n"
        r"Dashed: exact $\partial_{ss}V_{DO}(s,t)$, reiner\_rubinstein\_down\_and\_out\_put\_gamma (closed form, no "
        r"network). Dotted verticals: $s=B$, $s=B+\varepsilon$ (corner-layer edge), $s=K$; $\varepsilon=$"
        + f"{epsilon:g}"
        "\n"
        r"symlog $y$-axis, linear below " + f"{linear_threshold:g}" +
        r" (the profiles change sign near the barrier, which a log axis cannot represent)"
        "\n"
        r"Raw-mode caveat: $\partial_{ss}(K-s)^+=0$ away from $s=K$ and the grid misses that measure-zero point, so "
        r"the raw curve is the network's compensating curvature, not the kink's own"
    )
    fig.subplots_adjust(right=0.78, bottom=0.34, wspace=0.28)
    finalize_figure(fig, out_path, legends=[legend], formula=formula, axes=axes)


def plot_gamma_profile_decomposition_vs_price(
    profiles: dict, mode_order: list[str], K: float, B: float, epsilon: float,
    linear_threshold: float, out_path: Path,
) -> None:
    """3 (quantity) x n_t (fixed time) grid: the same s-profiles split into the
    full price, the network term and the extension term, so the pointwise
    origin of the total curvature is visible."""
    t_values = profiles["t_values"]
    s_numpy = profiles["s_values"]
    quantity_key = ["gamma", "gamma_g1u", "gamma_g2"]
    quantity_title = {
        "gamma": r"$\partial_{ss}\Phi_\theta$ (full trained price)",
        "gamma_g1u": r"$\partial_{ss}(g_1u_\theta)$ (network term)",
        "gamma_g2": r"$\partial_{ss}g_2=\partial_{ss}h_\varepsilon$ (extension term)",
    }
    fig, axes = plt.subplots(3, len(t_values), figsize=(4.6 * len(t_values), 11.0), squeeze=False)
    legend = None

    for row_index, key in enumerate(quantity_key):
        for time_index, t_value in enumerate(t_values):
            ax = axes[row_index][time_index]
            if key == "gamma":
                ax.plot(s_numpy, profiles["reference"]["gamma"][time_index], color=REFERENCE_COLOR, lw=2.2,
                        linestyle="--", label=REFERENCE_LABEL, zorder=5)
            for label in mode_order:
                ax.plot(s_numpy, profiles[label][key][time_index], color=MODE_COLOR[label], lw=1.4,
                        label=MODE_DISPLAY_NAME[label])
            _annotate_price_landmarks(ax, K, B, epsilon)
            ax.set_yscale("symlog", linthresh=linear_threshold)
            ax.grid(alpha=0.3, which="both")
            if row_index == 0:
                ax.set_title(f"$t = {t_value:g}$", fontsize=11)
            if row_index == len(quantity_key) - 1:
                ax.set_xlabel("Underlying price $s$")
            if time_index == 0:
                ax.set_ylabel(f"{quantity_title[key]}\n\nvalue (symlog scale)", fontsize=9)
            if row_index == 0 and time_index == len(t_values) - 1:
                legend = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)

    formula = (
        r"$\Phi_\theta=g_1u_\theta+g_2$; $\partial_{ss}\Phi_\theta$ and $\partial_{ss}(g_1u_\theta)$ by two nested "
        r"autograd passes on the trained model, $\partial_{ss}g_2$ by subtraction (exact, linearity of "
        r"$\partial_{ss}$)"
        "\n"
        r"Dashed (top row only): exact $\partial_{ss}V_{DO}(s,t)$, reiner\_rubinstein\_down\_and\_out\_put\_gamma. "
        r"Dotted verticals: $s=B$, $s=B+\varepsilon$, $s=K$; $\varepsilon=$" + f"{epsilon:g}"
        "\n"
        r"symlog $y$-axis, linear below " + f"{linear_threshold:g}"
    )
    fig.subplots_adjust(right=0.80, bottom=0.13, hspace=0.28, wspace=0.30)
    finalize_figure(fig, out_path, legends=[legend] if legend is not None else [], formula=formula,
                    axes=[a for row in axes for a in row])


def render_gamma_profile_table(profiles: dict, mode_order: list[str], K: float, B: float, epsilon: float) -> str:
    """At each fixed t and at three landmark prices (the corner-layer edge
    s=B+epsilon, the strike s=K, and the midpoint between them), the exact
    Gamma against each mode's total second price derivative."""
    s_numpy = profiles["s_values"]
    landmarks = [("$B+\\varepsilon$", B + epsilon), ("$(B+\\varepsilon+K)/2$", 0.5 * (B + epsilon + K)), ("$K$", K)]
    header_cells = ["t", "s", "Exact $\\partial_{ss}V_{DO}$"] + [f"{MODE_DISPLAY_NAME[m]}" for m in mode_order]
    lines = ["| " + " | ".join(header_cells) + " |", "|" + "|".join("---:" for _ in header_cells) + "|"]
    for time_index, t_value in enumerate(profiles["t_values"]):
        for landmark_name, s_target in landmarks:
            index = int(np.argmin(np.abs(s_numpy - s_target)))
            cells = [f"{t_value:g}", f"{landmark_name} $={s_numpy[index]:.4g}$",
                     f"{profiles['reference']['gamma'][time_index][index]:.4e}"]
            cells += [f"{profiles[m]['gamma'][time_index][index]:.4e}" for m in mode_order]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_price_delta_gamma_profiles_vs_price(
    profiles: dict, mode_order: list[str], K: float, B: float, epsilon: float,
    linear_threshold: float, out_path: Path,
) -> None:
    """3 (price, Delta, Gamma) x n_t (fixed time) grid: the full trained trial
    solution and its first two price derivatives as functions of s, all modes
    overlaid, each against its own exact closed-form counterpart.

    The price row is drawn on a linear y-axis (a price is O(1) and may be
    negative -- the raw mode's is; a log axis would hide the sign, and a
    symlog axis would compress the very range being compared).  The Delta and
    Gamma rows use the symlog axis, both changing sign and spanning several
    decades.
    """
    t_values = profiles["t_values"]
    s_numpy = profiles["s_values"]
    fig, axes = plt.subplots(len(PROFILE_QUANTITIES), len(t_values),
                             figsize=(4.6 * len(t_values), 3.7 * len(PROFILE_QUANTITIES)), squeeze=False)
    legend = None

    for row_index, key in enumerate(PROFILE_QUANTITIES):
        for time_index, t_value in enumerate(t_values):
            ax = axes[row_index][time_index]
            ax.plot(s_numpy, profiles["reference"][key][time_index], color=REFERENCE_COLOR, lw=2.2,
                    linestyle="--", label=f"{PROFILE_REFERENCE_TITLE[key]}, {REFERENCE_LABEL}", zorder=5)
            for label in mode_order:
                ax.plot(s_numpy, profiles[label][key][time_index], color=MODE_COLOR[label], lw=1.9,
                        alpha=0.9, label=MODE_DISPLAY_NAME[label])
            _annotate_price_landmarks(ax, K, B, epsilon)
            if key != "value":
                ax.set_yscale("symlog", linthresh=linear_threshold)
            ax.grid(alpha=0.3, which="both")
            if row_index == 0:
                ax.set_title(f"$t = {t_value:g}$", fontsize=11)
            if row_index == len(PROFILE_QUANTITIES) - 1:
                ax.set_xlabel("Underlying price $s$")
            if time_index == 0:
                scale_note = "linear scale" if key == "value" else "symlog scale"
                ax.set_ylabel(f"{PROFILE_QUANTITY_TITLE[key]}\n\nvalue ({scale_note})", fontsize=9)
            if row_index == 0 and time_index == len(t_values) - 1:
                legend = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)

    formula = (
        r"Trained (solid): $\Phi_\theta=g_1u_\theta+g_2$, and $\partial_s\Phi_\theta$, "
        r"$\partial_{ss}\Phi_\theta$ by two nested autograd passes on the model's own forward pass"
        "\n"
        r"Reference (dashed): $V_{DO}$ = reiner\_rubinstein\_down\_and\_out\_put, its $\partial_s$ and "
        r"$\partial_{ss}$ by autograd applied to that CLOSED FORM (no network, no finite difference; "
        r"cross-checked against the closed-form Gamma to $<10^{-15}$ relative)"
        "\n"
        r"Price row: linear $y$-axis (a negative price must remain visible). Delta / Gamma rows: symlog, "
        r"linear below " + f"{linear_threshold:g}" +
        r". Dotted verticals: $s=B$, $s=B+\varepsilon$, $s=K$; $\varepsilon=$" + f"{epsilon:g}"
    )
    fig.subplots_adjust(right=0.79, bottom=0.12, hspace=0.28, wspace=0.30)
    finalize_figure(fig, out_path, legends=[legend] if legend is not None else [], formula=formula,
                    axes=[a for row in axes for a in row])


# ---------------------------------------------------------------------------
# Analysis 9 -- where does the error actually live?  Signed error field
# Phi_theta - V_DO over the whole (s,t) domain, one panel per mode, with a
# corner zoom, and the regional split of the squared L^2 error.
#
# This is the measurement that decides whether the strike singularity and the
# corner singularity can be studied separately: if the error mass carried by
# the corner window and the error mass carried by the strike band are both
# substantial, no metric restricted to one of them characterises a mode.
# ---------------------------------------------------------------------------

def compute_error_fields(
    runs: list[dict], s_grid: torch.Tensor, t_grid: torch.Tensor,
) -> dict[str, np.ndarray]:
    """Signed error Phi_theta - V_DO on a full (s,t) grid, per mode.

    Forward passes only -- no autograd, no gradient step.
    """
    contract = runs[0]["meta"]["contract"]
    K, B, r, sigma, T = contract["K"], contract["B"], contract["r"], contract["sigma"], contract["T"]
    ss, tt = torch.meshgrid(s_grid, t_grid, indexing="ij")
    reference = reiner_rubinstein_down_and_out_put(ss, K, B, r, sigma, T - tt)

    fields: dict[str, np.ndarray] = {"reference": reference.numpy()}
    x = torch.stack([ss.reshape(-1), tt.reshape(-1)], dim=1)
    for run in runs:
        model = load_trained_model(run["run_dir"], run["meta"], dtype=torch.float64)
        with torch.no_grad():
            learned = model(x).squeeze(-1).reshape(ss.shape)
        fields[run["label"]] = (learned - reference).numpy()
        logger.info(
            f"[{run['label']}] signed error field: min={fields[run['label']].min():.4e}, "
            f"max={fields[run['label']].max():.4e}, RMS={np.sqrt((fields[run['label']]**2).mean()):.4e}"
        )
    return fields


def _error_region_masks(
    s_grid: torch.Tensor, t_grid: torch.Tensor, K: float, B: float, T: float,
    corner_window: float, strike_window: float,
) -> dict[str, np.ndarray]:
    """The three regions the squared error is split over: the ell^1 corner
    window N_eps around (B,T), the strike band |s-K|<=delta with the corner
    window removed, and everything else."""
    ss, tt = torch.meshgrid(s_grid, t_grid, indexing="ij")
    corner = ((ss - B).abs() + (T - tt) <= corner_window).numpy()
    strike = ((ss - K).abs() <= strike_window).numpy() & ~corner
    return {"corner": corner, "strike": strike, "rest": ~(corner | strike)}


def plot_error_heat_maps(
    fields: dict[str, np.ndarray], mode_order: list[str], s_grid: torch.Tensor, t_grid: torch.Tensor,
    K: float, B: float, T: float, epsilon: float, corner_window: float, strike_window: float,
    out_path: Path,
) -> None:
    """2 rows x n_modes: the signed error field Phi_theta - V_DO over the full
    domain (top) and zoomed on the corner (bottom), on a shared diverging,
    symmetric-logarithmic colour scale so the panels are directly comparable."""
    from matplotlib.colors import SymLogNorm

    s_numpy, t_numpy = s_grid.numpy(), t_grid.numpy()
    largest = max(float(np.abs(fields[m]).max()) for m in mode_order)
    norm = SymLogNorm(linthresh=1e-3, vmin=-largest, vmax=largest, base=10)
    extent = [t_numpy[0], t_numpy[-1], s_numpy[0], s_numpy[-1]]

    fig, axes = plt.subplots(2, len(mode_order), figsize=(3.9 * len(mode_order), 8.2), squeeze=False)
    image = None
    for column, label in enumerate(mode_order):
        for row, zoomed in enumerate((False, True)):
            ax = axes[row][column]
            image = ax.imshow(fields[label], origin="lower", aspect="auto", extent=extent,
                              cmap="RdBu_r", norm=norm)
            # the ell^1 corner window boundary: (s-B) + (T-t) = corner_window
            t_edge = np.linspace(T - corner_window, T, 100)
            ax.plot(t_edge, B + corner_window - (T - t_edge), color="black", lw=1.4, linestyle="--")
            ax.axhline(K, color="black", lw=0.9, linestyle=":")
            ax.axhline(B + epsilon, color="black", lw=0.9, linestyle=":")
            if zoomed:
                ax.set_xlim(T - 4.0 * corner_window, T)
                ax.set_ylim(B, B + 4.0 * corner_window)
                ax.set_xlabel("Calendar time $t$")
            else:
                ax.set_title(MODE_DISPLAY_NAME[label], fontsize=9)
            if column == 0:
                ax.set_ylabel(("Corner zoom\n\n" if zoomed else "Full domain\n\n") + "Underlying price $s$",
                              fontsize=9)
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    colorbar.set_label(r"$\Phi_\theta(s,t)-V_{DO}(s,t)$  (signed, symlog colour scale)", fontsize=9)

    formula = (
        r"Signed error of the trained price against the exact closed form; diverging palette, symmetric "
        r"symlog colour scale shared by all panels (linear below $10^{-3}$), so blue and red are the two "
        r"signs of the same magnitude"
        "\n"
        r"Dashed black: boundary of the $\ell^1$ corner window $(s-B)+(T-t)=$" + f"{corner_window:g}" +
        r". Dotted black: $s=K$ (strike) and $s=B+\varepsilon$ (corner-layer edge), $\varepsilon=$" + f"{epsilon:g}"
        "\n"
        r"Bottom row: the same fields restricted to $[B,B+4\cdot" + f"{corner_window:g}" +
        r"]\times[T-4\cdot" + f"{corner_window:g}" + r",T]$"
    )
    fig.subplots_adjust(bottom=0.16, hspace=0.20, wspace=0.30, right=0.88)
    finalize_figure(fig, out_path, formula=formula, axes=[a for row in axes for a in row])


def render_error_budget_table(
    fields: dict[str, np.ndarray], mode_order: list[str], masks: dict[str, np.ndarray],
    corner_window: float, strike_window: float,
) -> str:
    """Share of the total squared L^2 error carried by each region, per mode.

    The (s,t) grid is uniform, so every cell has the same area and the share
    of the squared L^2 norm is the share of the sum of squares -- no
    quadrature weight is needed.  The region areas differ, so the shares are
    reported next to each region's share of the domain area, which is the
    share a spatially uniform error would produce.
    """
    total_cells = next(iter(masks.values())).size
    area_share = {name: mask.sum() / total_cells for name, mask in masks.items()}
    region_order = ["corner", "strike", "rest"]
    region_name = {
        "corner": f"Corner window $(s-B)+(T-t)\\le{corner_window:g}$",
        "strike": f"Strike band $|s-K|\\le{strike_window:g}$ (corner removed)",
        "rest": "Rest of the domain",
    }
    header = "| Region | Share of the domain area | " + " | ".join(MODE_DISPLAY_NAME[m] for m in mode_order) + " |"
    lines = [header, "|---|---:|" + "|".join("---:" for _ in mode_order) + "|"]
    for region in region_order:
        cells = []
        for label in mode_order:
            squared = fields[label] ** 2
            cells.append(f"{squared[masks[region]].sum() / squared.sum() * 100:.2f} %")
        lines.append(f"| {region_name[region]} | {area_share[region] * 100:.2f} % | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("| Concentration ratio (share of squared error / share of area) | | "
                 + " | ".join("" for _ in mode_order) + " |")
    for region in region_order:
        cells = []
        for label in mode_order:
            squared = fields[label] ** 2
            share = squared[masks[region]].sum() / squared.sum()
            cells.append(f"{share / area_share[region]:.2f}" if area_share[region] > 0 else "n/a")
        lines.append(f"| {region_name[region]} | 1.00 | " + " | ".join(cells) + " |")
    return "\n".join(lines)



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the 4 g2 terminal-function modes of pilot_down_and_out_put.py near the strike. "
                     "Reads already-trained runs only; never retrains.",
    )
    parser.add_argument("--run-dir-prefix", type=str, default="20260825_014738_iters20000_eps0.1_seed0",
                         help="Shared prefix of the 4 run directories to compare (one per g2 mode).")
    parser.add_argument("--base-dir", type=str, default=None,
                         help="Directory containing the run subdirectories (default: pilot_down_and_out_put.py's own data directory).")
    parser.add_argument("--corner-margin", type=float, default=0.1,
                         help="Excludes s < B + margin from the 'global excluding near-barrier zone' rel_L2 metric (analysis 1).")
    parser.add_argument("--strike-window", type=float, default=0.05,
                         help="Half-width delta of the s-window (K-delta, K+delta) around the strike used in analyses 3 and 4.")
    parser.add_argument("--strike-zoom-times", nargs="+", type=float, default=[0.0, 0.5, 0.9, 0.99],
                         help="Calendar times t at which to draw the strike-zoom linear slice (analysis 4).")
    parser.add_argument("--curvature-table-times", nargs="+", type=float, default=[0.0, 0.5, 0.9],
                         help="Calendar times t at which to tabulate the curvature-vs-time values (analysis 3).")
    parser.add_argument("--pointwise-modes", nargs="+", type=str, default=["raw", "mangasarian_time_graded"],
                         help="Modes analysed pointwise (analysis 5): finite-difference-step sensitivity at "
                              "s=K, and the Phi_theta/g1*u_theta/g2 decomposition vs t at several s offsets.")
    parser.add_argument("--pointwise-s-offsets", nargs="+", type=float, default=[-0.05, -0.02, 0.0, 0.02, 0.05],
                         help="s - K offsets at which analysis 5's pointwise second derivatives are evaluated.")
    parser.add_argument("--pointwise-h", type=float, default=1e-3,
                         help="Fixed finite-difference step used for analysis 5's Phi_theta/g1*u_theta/g2 "
                              "decomposition figure (the h-sensitivity table below sweeps h explicitly instead).")
    parser.add_argument("--pointwise-h-sensitivity-values", nargs="+", type=float, default=[1e-2, 1e-3, 1e-4],
                         help="Finite-difference steps h swept at s=K exactly, to show whether the pointwise "
                              "second derivative converges (smoothed modes) or diverges (raw mode's kink).")
    parser.add_argument("--pointwise-h-sensitivity-times", nargs="+", type=float, default=[0.0, 0.5, 0.9],
                         help="Calendar times t at which the h-sensitivity table (analysis 5) is evaluated.")
    parser.add_argument("--true-gamma-comparison-modes", nargs="+", type=str,
                         default=["mangasarian_time_graded", "mangasarian_constant", "black_scholes"],
                         help="Modes for which analysis 6 compares the pointwise trained d^2 Phi_theta/ds^2 at "
                              "s=K against the exact Reiner-Rubinstein Gamma. Only meaningful for a mode whose "
                              "pointwise value is itself well-defined (h-independent) -- excludes raw by default.")
    parser.add_argument("--true-gamma-comparison-times", nargs="+", type=float, default=[0.0, 0.2, 0.3, 0.5, 0.7, 0.9, 0.98],
                         help="Calendar times t tabulated in analysis 6's comparison table.")
    parser.add_argument("--mechanism-crude-n", type=int, default=21,
                         help="Grid resolution analysis 3 used for its L2-norm quadrature -- analysis 7's "
                              "correction table compares this crude resolution against the exact closed form.")
    parser.add_argument("--exact-gamma-table-times", nargs="+", type=float,
                         default=[0.0, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999],
                         help="Calendar times t tabulated in analysis 7's exact (closed-form, no network) "
                              "pointwise Gamma table.")
    parser.add_argument("--gamma-profile-times", nargs="+", type=float, default=[0.0, 0.5, 0.9, 0.99],
                         help="Calendar times t held FIXED in analysis 8, whose s-profiles of the second price "
                              "derivative of the full trained price are drawn.")
    parser.add_argument("--gamma-profile-s-min", type=float, default=None,
                         help="Lower end of the s-range of analysis 8 (default: the barrier B).")
    parser.add_argument("--gamma-profile-s-max", type=float, default=None,
                         help="Upper end of the s-range of analysis 8 (default: 2K, i.e. twice the strike).")
    parser.add_argument("--gamma-profile-n-s", type=int, default=801,
                         help="Number of s grid points of analysis 8's profiles.")
    parser.add_argument("--gamma-profile-linear-threshold", type=float, default=1e-1,
                         help="Half-width of the linear region of analysis 8's symlog y-axis (the profiles change "
                              "sign near the barrier, so a pure log axis cannot represent them).")
    parser.add_argument("--error-field-s-max", type=float, default=2.0,
                         help="Upper end of the s-range of analysis 9's error-field heat maps.")
    parser.add_argument("--error-field-n-s", type=int, default=400,
                         help="Number of s grid points of analysis 9's error field.")
    parser.add_argument("--error-field-n-t", type=int, default=400,
                         help="Number of t grid points of analysis 9's error field.")
    parser.add_argument("--error-field-corner-window", type=float, default=0.1,
                         help="Half-width of the ell^1 corner window (s-B)+(T-t) <= w used both to draw the "
                              "corner boundary on analysis 9's heat maps and to split its error budget.")
    parser.add_argument("--out-dir", type=str, default=None,
                         help="Output directory (default: <base-dir>/<timestamp>_compare_payoff_modes_strike/).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("matplotlib.mathtext").setLevel(logging.WARNING)

    pilot_script_path = Path(pilot.__file__).resolve()
    base_dir = Path(args.base_dir) if args.base_dir is not None else script_data_dir(pilot_script_path)

    runs = discover_runs(base_dir, args.run_dir_prefix)
    logger.info(f"Comparing {len(runs)} runs under {base_dir}/{args.run_dir_prefix}*:")
    for run in runs:
        logger.info(f"  {run['label']:<28} -> {run['run_dir']}")

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        out_dir = base_dir / f"{timestamp}_compare_payoff_modes_strike"
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info(f"Analysis 1: rel_L2 error, global vs. excluding s < B + {args.corner_margin:g}")
    logger.info("=" * 70)
    error_rows = compute_error_table(runs, args.corner_margin)
    table_text = render_error_table(error_rows)
    print("\n" + table_text + "\n")
    with open(out_dir / "error_table.yaml", "w") as f:
        yaml.dump(error_rows, f, default_flow_style=False, sort_keys=False)
    with open(out_dir / "error_table.md", "w") as f:
        f.write(table_text + "\n")
    logger.info(f"Table saved -> {out_dir / 'error_table.md'} / .yaml")

    logger.info("=" * 70)
    logger.info("Analysis 2: loss history vs. iteration")
    logger.info("=" * 70)
    plot_loss_history(runs, figures_dir / "loss_history.png")
    logger.info(f"Figure saved -> {figures_dir / 'loss_history.png'}")

    logger.info("=" * 70)
    logger.info(f"Analysis 3: curvature near the strike (delta={args.strike_window:g}) vs. time")
    logger.info("=" * 70)
    t_grid, curvature_by_label = plot_curvature_vs_time(runs, args.strike_window, figures_dir / "curvature_vs_time.png")
    logger.info(f"Figure saved -> {figures_dir / 'curvature_vs_time.png'}")
    curvature_table_text = render_curvature_table(t_grid, curvature_by_label, args.curvature_table_times)
    print("\n" + curvature_table_text + "\n")
    with open(out_dir / "curvature_at_selected_times.md", "w") as f:
        f.write(curvature_table_text + "\n")
    with open(out_dir / "curvature_vs_time.yaml", "w") as f:
        yaml.dump({"t_grid": t_grid.tolist(), **curvature_by_label}, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Curvature table saved -> {out_dir / 'curvature_at_selected_times.md'}")
    logger.info(f"Full curvature-vs-time data saved -> {out_dir / 'curvature_vs_time.yaml'}")

    logger.info("=" * 70)
    logger.info(f"Analysis 4: strike-zoom linear-scale slice (delta={args.strike_window:g})")
    logger.info("=" * 70)
    min_price_rows = plot_strike_zoom_linear(runs, args.strike_window, args.strike_zoom_times, figures_dir / "strike_zoom_linear.png")
    with open(out_dir / "min_price_near_strike.yaml", "w") as f:
        yaml.dump(min_price_rows, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Figure saved -> {figures_dir / 'strike_zoom_linear.png'}")
    logger.info(f"Min-price table saved -> {out_dir / 'min_price_near_strike.yaml'}")

    logger.info("=" * 70)
    logger.info(f"Analysis 5: pointwise second derivative near the strike ({args.pointwise_modes})")
    logger.info("=" * 70)
    pointwise_runs = [r for r in runs if r["label"] in args.pointwise_modes]
    if len(pointwise_runs) != len(args.pointwise_modes):
        found = [r["label"] for r in pointwise_runs]
        logger.warning(f"--pointwise-modes {args.pointwise_modes} : only found {found} among the discovered runs.")

    h_sensitivity_rows: list[dict] = []
    for run in pointwise_runs:
        h_sensitivity_rows += compute_h_sensitivity_at_strike(run, args.pointwise_h_sensitivity_times, args.pointwise_h_sensitivity_values)
    h_sensitivity_text = render_h_sensitivity_table(h_sensitivity_rows, args.pointwise_h_sensitivity_values)

    # Reference column for any smoothed (mangasarian) mode present: the exact
    # closed form 1/(2*eps(t)) that a well-defined (h-independent) pointwise
    # second derivative at s=K should converge to.
    reference_lines = []
    for run in pointwise_runs:
        hp = run["meta"]["hyperparameters"]
        if hp.get("smoothed_payoff", False):
            T = run["meta"]["contract"]["T"]
            eps0, grading = hp["eps0"], hp["grading"]
            for t_val in args.pointwise_h_sensitivity_times:
                eps_t = eps0 * (T - t_val) / T if grading == "time_graded" else eps0
                reference_lines.append(f"  [{run['label']}] analytic 1/(2*eps(t)) at t={t_val:g}: {1.0 / (2.0 * eps_t):.4e}")
    print("\n" + h_sensitivity_text + "\n")
    if reference_lines:
        print("Analytic reference (smoothed modes only, converged pointwise value at s=K):")
        print("\n".join(reference_lines) + "\n")
    with open(out_dir / "h_sensitivity_at_strike.md", "w") as f:
        f.write(h_sensitivity_text + "\n")
        if reference_lines:
            f.write("\nAnalytic reference (smoothed modes only, converged pointwise value at s=K):\n\n")
            f.write("\n".join(reference_lines) + "\n")
    with open(out_dir / "h_sensitivity_at_strike.yaml", "w") as f:
        yaml.dump(h_sensitivity_rows, f, default_flow_style=False, sort_keys=False)
    logger.info(f"h-sensitivity table saved -> {out_dir / 'h_sensitivity_at_strike.md'} / .yaml")

    T_contract = runs[0]["meta"]["contract"]["T"]
    t_grid_pointwise = torch.linspace(0.0, T_contract - 1e-4, 60, dtype=torch.float64)
    pointwise_data_by_mode = plot_pointwise_second_derivatives(
        pointwise_runs, args.pointwise_s_offsets, args.pointwise_h, t_grid_pointwise,
        figures_dir / "pointwise_second_derivative_decomposition.png",
    )
    logger.info(f"Figure saved -> {figures_dir / 'pointwise_second_derivative_decomposition.png'}")

    l2_vs_pointwise_text = render_l2_vs_pointwise_table(
        curvature_by_label, t_grid, pointwise_data_by_mode, t_grid_pointwise, args.curvature_table_times,
    )
    print("\n" + l2_vs_pointwise_text + "\n")
    with open(out_dir / "l2_norm_vs_pointwise_at_strike.md", "w") as f:
        f.write(l2_vs_pointwise_text + "\n")
    logger.info(f"L2-norm-vs-pointwise table saved -> {out_dir / 'l2_norm_vs_pointwise_at_strike.md'}")

    logger.info("=" * 70)
    logger.info(f"Analysis 6: pointwise trained curvature vs. true Reiner-Rubinstein Gamma at s=K ({args.true_gamma_comparison_modes})")
    logger.info("=" * 70)
    gamma_comparison_runs = [r for r in runs if r["label"] in args.true_gamma_comparison_modes]
    gamma_data_by_mode: dict[str, dict[str, list[float]]] = {}
    for run in gamma_comparison_runs:
        label = run["label"]
        gamma_figure_path = figures_dir / f"pointwise_phi_vs_true_gamma_{label}.png"
        gamma_table_path = out_dir / f"phi_vs_true_gamma_{label}.md"

        gamma_data = compute_pointwise_phi_vs_true_gamma(run, t_grid_pointwise)
        gamma_data_by_mode[label] = gamma_data
        plot_pointwise_phi_vs_true_gamma(run, t_grid_pointwise, gamma_data, gamma_figure_path)
        gamma_table_text = render_phi_vs_true_gamma_table(t_grid_pointwise, gamma_data, args.true_gamma_comparison_times)
        print(f"\n[{label}]\n" + gamma_table_text + "\n")
        with open(gamma_table_path, "w") as f:
            f.write(gamma_table_text + "\n")
        logger.info(f"[{label}] figure saved -> {gamma_figure_path}")
        logger.info(f"[{label}] table saved -> {gamma_table_path}")

    if len(gamma_data_by_mode) > 1:
        mode_order = [m for m in MODE_ORDER if m in gamma_data_by_mode]
        combined_table_text = render_combined_phi_vs_true_gamma_table(t_grid_pointwise, gamma_data_by_mode, mode_order, args.true_gamma_comparison_times)
        mae_summary_text = render_mean_abs_error_summary(t_grid_pointwise, gamma_data_by_mode, mode_order, t_max=T_contract - 1e-4)
        print("\n=== Combined: which g2 tracks the true Gamma best? ===\n" + combined_table_text + "\n")
        print(mae_summary_text + "\n")
        combined_figure_path = figures_dir / "pointwise_phi_vs_true_gamma_combined.png"
        plot_combined_phi_vs_true_gamma(t_grid_pointwise, gamma_data_by_mode, mode_order, combined_figure_path)
        with open(out_dir / "phi_vs_true_gamma_combined.md", "w") as f:
            f.write(combined_table_text + "\n\n" + mae_summary_text + "\n")
        logger.info(f"Combined figure saved -> {combined_figure_path}")
        logger.info(f"Combined table saved -> {out_dir / 'phi_vs_true_gamma_combined.md'}")

    logger.info("=" * 70)
    logger.info("Analysis 7: mechanism (why raw's L2 curvature extinguishes, time-graded's explodes) + exact-value corrections")
    logger.info("=" * 70)
    mechanism_runs = [r for r in runs if r["label"] in ("raw", "mangasarian_time_graded")]
    mechanism_figure_path = figures_dir / "mechanism_extinction_vs_explosion.png"
    mechanism_data = plot_mechanism_figure(mechanism_runs, args.strike_window, t_grid_pointwise, mechanism_figure_path)
    logger.info(f"Mechanism figure saved -> {mechanism_figure_path}")

    time_graded_run = next((r for r in runs if r["label"] == "mangasarian_time_graded"), None)
    if time_graded_run is not None:
        eps0_tg = time_graded_run["meta"]["hyperparameters"]["eps0"]
        correction_table_text = render_l2_norm_correction_table(
            t_grid_pointwise, eps0_tg, T_contract, args.strike_window, args.mechanism_crude_n, args.curvature_table_times,
        )
        print("\n=== L2-norm quadrature correction (time-graded g2 alone, exact vs. crude) ===\n" + correction_table_text + "\n")
        with open(out_dir / "l2_norm_quadrature_correction.md", "w") as f:
            f.write(correction_table_text + "\n")
        logger.info(f"L2-norm correction table saved -> {out_dir / 'l2_norm_quadrature_correction.md'}")

    exact_gamma_table_text = render_exact_gamma_at_strike_table(runs, args.exact_gamma_table_times)
    print("\n=== Exact (closed-form, no network) pointwise Gamma at s=K ===\n" + exact_gamma_table_text + "\n")
    with open(out_dir / "exact_gamma_at_strike.md", "w") as f:
        f.write(exact_gamma_table_text + "\n")
    logger.info(f"Exact-Gamma table saved -> {out_dir / 'exact_gamma_at_strike.md'}")

    with open(out_dir / "l2_decomposition_by_component.yaml", "w") as f:
        yaml.dump({"t_grid": t_grid_pointwise.tolist(), **mechanism_data}, f, default_flow_style=False, sort_keys=False)
    logger.info(f"L2 decomposition data saved -> {out_dir / 'l2_decomposition_by_component.yaml'}")

    logger.info("=" * 70)
    logger.info(f"Analysis 8: Gamma of the full trained price vs. s, at fixed times {args.gamma_profile_times}")
    logger.info("=" * 70)
    contract = runs[0]["meta"]["contract"]
    epsilon_corner = runs[0]["meta"]["hyperparameters"]["epsilons"][0]
    s_min = args.gamma_profile_s_min if args.gamma_profile_s_min is not None else contract["B"]
    s_max = args.gamma_profile_s_max if args.gamma_profile_s_max is not None else 2.0 * contract["K"]
    s_profile_grid = torch.linspace(s_min, s_max, args.gamma_profile_n_s, dtype=torch.float64)
    logger.info(
        f"s-grid: [{s_min:g}, {s_max:g}], {args.gamma_profile_n_s} points "
        f"(spacing {(s_max - s_min) / (args.gamma_profile_n_s - 1):.4e}); corner-layer bandwidth epsilon={epsilon_corner:g}"
    )
    gamma_profiles = compute_price_and_greek_profiles_vs_price(runs, s_profile_grid, args.gamma_profile_times)

    price_greek_figure_path = figures_dir / "price_delta_gamma_profiles_vs_price.png"
    plot_price_delta_gamma_profiles_vs_price(
        gamma_profiles, MODE_ORDER, contract["K"], contract["B"], epsilon_corner,
        args.gamma_profile_linear_threshold, price_greek_figure_path,
    )
    logger.info(f"Figure saved -> {price_greek_figure_path}")

    gamma_profile_figure_path = figures_dir / "gamma_profiles_vs_price.png"
    plot_gamma_profiles_vs_price(
        gamma_profiles, MODE_ORDER, contract["K"], contract["B"], epsilon_corner,
        args.gamma_profile_linear_threshold, gamma_profile_figure_path,
    )
    logger.info(f"Figure saved -> {gamma_profile_figure_path}")

    gamma_decomposition_figure_path = figures_dir / "gamma_profiles_vs_price_decomposition.png"
    plot_gamma_profile_decomposition_vs_price(
        gamma_profiles, MODE_ORDER, contract["K"], contract["B"], epsilon_corner,
        args.gamma_profile_linear_threshold, gamma_decomposition_figure_path,
    )
    logger.info(f"Figure saved -> {gamma_decomposition_figure_path}")

    gamma_profile_table_text = render_gamma_profile_table(
        gamma_profiles, MODE_ORDER, contract["K"], contract["B"], epsilon_corner,
    )
    print("\n=== Gamma of the full trained price at landmark prices, per fixed t ===\n" + gamma_profile_table_text + "\n")
    with open(out_dir / "gamma_profiles_at_landmark_prices.md", "w") as f:
        f.write(gamma_profile_table_text + "\n")
    logger.info(f"Table saved -> {out_dir / 'gamma_profiles_at_landmark_prices.md'}")

    # Saved so both analysis-8 figures can be redrawn from the artefact
    # without re-running any autograd pass over the trained models.
    gamma_profile_arrays = {
        "s_values": gamma_profiles["s_values"],
        "t_values": np.asarray(gamma_profiles["t_values"]),
    }
    for key in PROFILE_QUANTITIES:
        gamma_profile_arrays[f"reference__{key}"] = gamma_profiles["reference"][key]
    for label in MODE_ORDER:
        for key in PROFILE_QUANTITIES + ["gamma_g1u", "gamma_g2"]:
            gamma_profile_arrays[f"{label}__{key}"] = gamma_profiles[label][key]
    np.savez(out_dir / "price_delta_gamma_profiles_vs_price.npz", **gamma_profile_arrays)
    logger.info(f"Profile data saved -> {out_dir / 'price_delta_gamma_profiles_vs_price.npz'}")

    logger.info("=" * 70)
    logger.info("Analysis 9: signed error field Phi_theta - V_DO, heat maps and regional error budget")
    logger.info("=" * 70)
    s_field_grid = torch.linspace(contract["B"], args.error_field_s_max, args.error_field_n_s, dtype=torch.float64)
    t_field_grid = torch.linspace(0.0, contract["T"], args.error_field_n_t, dtype=torch.float64)
    logger.info(
        f"error field grid: s in [{contract['B']:g}, {args.error_field_s_max:g}] x {args.error_field_n_s} points, "
        f"t in [0, {contract['T']:g}] x {args.error_field_n_t} points"
    )
    error_fields = compute_error_fields(runs, s_field_grid, t_field_grid)

    error_heat_map_path = figures_dir / "error_field_heat_maps.png"
    plot_error_heat_maps(
        error_fields, MODE_ORDER, s_field_grid, t_field_grid, contract["K"], contract["B"], contract["T"],
        epsilon_corner, args.error_field_corner_window, args.strike_window, error_heat_map_path,
    )
    logger.info(f"Figure saved -> {error_heat_map_path}")

    region_masks = _error_region_masks(
        s_field_grid, t_field_grid, contract["K"], contract["B"], contract["T"],
        args.error_field_corner_window, args.strike_window,
    )
    error_budget_text = render_error_budget_table(
        error_fields, MODE_ORDER, region_masks, args.error_field_corner_window, args.strike_window,
    )
    print("\n=== Where the squared L2 error lives (share per region, per mode) ===\n" + error_budget_text + "\n")
    with open(out_dir / "error_budget_by_region.md", "w") as f:
        f.write(error_budget_text + "\n")
    np.savez(out_dir / "error_fields.npz",
             s_values=s_field_grid.numpy(), t_values=t_field_grid.numpy(),
             **{label: error_fields[label] for label in MODE_ORDER})
    logger.info(f"Error-budget table saved -> {out_dir / 'error_budget_by_region.md'}")
    logger.info(f"Error-field data saved -> {out_dir / 'error_fields.npz'}")

    logger.info("=" * 70)
    logger.info(f"All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
