r"""Down-and-out put: where is the error, and which excluded region separates the treatments?

Answers two questions the corner-window metric of
``aggregate_terminal_function_comparison.py`` cannot answer on its own.

**1. Which region is excluded?** The comparison metric removes the ell^1 corner
window N_w = {|s-B| + (T-t) <= w}, a small lozenge that touches the corner only.
The smoothing constructions, however, give up the exact terminal trace on the
WHOLE strip where their cutoff differs from one,

    S_eps = {(s, t) : s - B < eps}    (all t),

because zeta((s-B)/eps) = 1 exactly only for s - B >= eps. Removing N_w
therefore leaves most of the region the smoothing sacrifices inside the metric,
while the analytic corner treatments (exact subtraction, corner enrichment)
sacrifice nothing there. This script recomputes the relative L2 error on the
complement of several regions -- none, N_w, S_eps, a strike band
B_delta = {|s - K| < delta}, and their unions -- so that the comparison can be
read on a region every construction treats identically.

**2. How is the error distributed?** For each configuration it computes, from the
saved models and with no retraining:

- the error-energy density along s, E(s) = int_0^T |Phi_theta - V_DO|^2 dt, and
  its cumulative share, which localise the error in s without any window choice;
- the interior PDE residual |L^BS Phi_theta|^2 on the grid, assembled through
  the SAME route the training loss used (two-term analytic when g2 exposes
  black_scholes_residual, one autograd graph otherwise), and its share per
  region -- i.e. where the loss actually sees an error, as opposed to where the
  solution is wrong;
- the fraction of a uniform collocation batch that falls in each region, next to
  that region's share of the residual energy: the ratio says whether uniform
  sampling under-weights the region that dominates the residual.

Figures (all rebuilt from ``grids.pt`` with ``--replot``):

- ``comparison_by_exclusion_region.png``: the terminal-function comparison
  figure, one panel per excluded region, configurations on the abscissa,
  individual seeds and across-seed median.
- ``error_energy_along_s.png``: E(s) and its cumulative share, one curve per
  configuration, with the region boundaries marked.
- ``error_and_residual_maps.png``: |Phi - V_DO| and |L^BS Phi| in the (t, s)
  plane, one row per configuration (one seed).

Usage:
    python3 experiments/python_scripts/exp_barrier_option/diagnose_error_localisation.py \
        --aggregation-dir data/aggregate_terminal_function_comparison/<dir>
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from learning_option_pricing.pricing.barrier import reiner_rubinstein_down_and_out_put  # noqa: E402
from learning_option_pricing.pricing.terminal import bsm_operator  # noqa: E402
from learning_option_pricing.utils.figure_layout import finalize_figure  # noqa: E402
from learning_option_pricing.utils.run_context import find_repo_root, script_data_dir  # noqa: E402
from aggregate_terminal_function_comparison import (  # noqa: E402
    CONFIGURATION_LABELS, FAR_FIELD_DIRICHLET_SUFFIX,
)
from compare_corner_treatments_profiles import (  # noqa: E402
    FAR_FIELD_DIRICHLET_LABEL, TREATMENT_LABELS, TERMINAL_PROFILE_LABELS,
    corner_treatment_of, curve_colour, far_field_dirichlet_of, terminal_profile_of,
)
from pilot_down_and_out_put import DEVICE, load_trained_model, read_run_metadata  # noqa: E402

logger = logging.getLogger("diagnose_error_localisation")

#: Every configuration key the aggregation can hold, in figure order. The
#: ``*_corner_included`` keys are the smoothing runs trained on the whole domain
#: (methodology section 17.2), the only ones comparable to the analytic
#: treatments without a collocation-domain confound.
DEFAULT_CONFIGURATIONS = list(CONFIGURATION_LABELS)


#: Compact tick labels: the full CONFIGURATION_LABELS carry the collocation-domain
#: annotation, constant across this figure and stated in its caption instead.
COMPACT_LABELS: dict[str, str] = {
    "raw_corner_included": "Smoothing,\nraw payoff",
    "smoothed_corner_included": "Smoothing,\nChen-Mangasarian",
    "blackscholes_corner_included": "Smoothing,\nBlack-Scholes",
    "blackscholes_analyticres_corner_included": "Smoothing, Black-Scholes\n(two-term route)",
    "split_corner_included": "Smoothing,\nsplit-semigroup",
    "raw": "Smoothing, raw payoff\n(corner excluded)",
    "blackscholes": "Smoothing, Black-Scholes\n(corner excluded)",
    "blackscholes_analyticres": "Smoothing, BS two-term\n(corner excluded)",
    "split": "Smoothing, split\n(corner excluded)",
    "subtraction_raw": "Subtraction,\nraw payoff",
    "subtraction_blackscholes": "Subtraction,\nBlack-Scholes",
    "subtraction_split": "Subtraction,\nsplit-semigroup",
    "enrichment_raw": "Enrichment,\nraw payoff",
    "enrichment_blackscholes": "Enrichment,\nBlack-Scholes",
    "enrichment_split": "Enrichment,\nsplit-semigroup",
}


#: Far-field twin of every compact label, so that an aggregation holding both
#: arms of the condition (``--far-field any``) keeps short tick labels. The
#: second line names the condition, which is what such a figure compares.
COMPACT_LABELS |= {
    configuration + FAR_FIELD_DIRICHLET_SUFFIX: label + "\n+ far-field Dirichlet"
    for configuration, label in COMPACT_LABELS.items()
}


def compact_label(configuration: str) -> str:
    return COMPACT_LABELS.get(configuration, CONFIGURATION_LABELS.get(configuration, configuration))


def exclusion_regions(ss: torch.Tensor, tt: torch.Tensor, K: float, B: float, T: float,
                      corner_window: float, cutoff_epsilon: float, strike_delta: float,
                      far_field_start: float) -> dict[str, dict]:
    """``{name: {"mask": kept points, "label": ..., "definition": ...}}``.

    Each mask selects the points KEPT by that metric (the complement of the
    excluded region).
    """
    corner = (ss - B).abs() + (T - tt) <= corner_window
    strip = (ss - B) < cutoff_epsilon
    strike = (ss - K).abs() < strike_delta
    far = ss >= far_field_start
    return {
        "full": {
            "mask": torch.ones_like(ss, dtype=torch.bool),
            "label": r"$\Omega$ (nothing excluded)",
            "definition": "Omega = (B, s_inf) x (0, T)",
        },
        "minus_corner": {
            "mask": ~corner,
            "label": rf"$\Omega\setminus \mathcal{{N}}_{{{corner_window:g}}}$ (corner lozenge)",
            "definition": f"excludes |s-B| + (T-t) <= {corner_window:g}",
        },
        "minus_cutoff_strip": {
            "mask": ~strip,
            "label": rf"$\Omega\setminus \mathcal{{Z}}_{{{cutoff_epsilon:g}}}$ (cutoff cylinder)",
            "definition": f"excludes s - B < {cutoff_epsilon:g} (all t): where zeta != 1",
        },
        "minus_strip_and_strike": {
            "mask": ~(strip | strike),
            "label": rf"$\Omega\setminus(\mathcal{{Z}}_{{{cutoff_epsilon:g}}}\cup \mathcal{{K}}_{{{strike_delta:g}}})$",
            "definition": f"excludes s - B < {cutoff_epsilon:g} and |s - K| < {strike_delta:g}",
        },
        "minus_strip_strike_far": {
            "mask": ~(strip | strike | far),
            "label": rf"$\Omega\setminus(\mathcal{{Z}}_{{{cutoff_epsilon:g}}}\cup \mathcal{{K}}_{{{strike_delta:g}}}\cup \mathcal{{W}}_{{{far_field_start:g}}})$",
            "definition": (f"excludes s - B < {cutoff_epsilon:g}, |s - K| < {strike_delta:g} "
                           f"and s >= {far_field_start:g}"),
        },
    }


def interior_residual(model, ss: torch.Tensor, tt: torch.Tensor, r: float, sigma: float,
                      chunk: int = 4096) -> torch.Tensor:
    """``L^BS Phi_theta`` on the grid, through the SAME route the training loss
    used: the two-term analytic assembly when ``g2`` exposes
    ``black_scholes_residual`` (subtraction, enrichment, split, and the
    Black-Scholes two-term arm), one autograd graph on the full trial solution
    otherwise. Evaluated in chunks; the returned tensor has the grid's shape."""
    s_flat = ss.reshape(-1).to(DEVICE).to(torch.get_default_dtype())
    t_flat = tt.reshape(-1).to(DEVICE).to(torch.get_default_dtype())
    g2 = model.g2
    analytic = hasattr(g2, "black_scholes_residual")
    pieces = []
    for start in range(0, s_flat.numel(), chunk):
        s_chunk = s_flat[start:start + chunk].clone().requires_grad_(True)
        t_chunk = t_flat[start:start + chunk].clone().requires_grad_(True)
        x = torch.stack([s_chunk, t_chunk], dim=1)
        with torch.enable_grad():
            if analytic:
                manifold = model.forward_neural_manifold(x).squeeze(-1)
                residual = bsm_operator(manifold, s_chunk, t_chunk, r, 0.0, sigma)
                with torch.no_grad():
                    residual = residual + g2.black_scholes_residual(s_chunk.detach(), t_chunk.detach(), r, sigma)
            else:
                value = model(x).squeeze(-1)
                residual = bsm_operator(value, s_chunk, t_chunk, r, 0.0, sigma)
        pieces.append(residual.detach().double().cpu())
    return torch.cat(pieces).reshape(ss.shape)


def evaluate_grid(run_dir: Path, epsilon: float, n_s: int, n_t: int, with_residual: bool) -> dict:
    """Trained field, closed form and (optionally) the interior residual of one run."""
    meta = read_run_metadata(run_dir)
    K, B, r, sigma, T = (meta["contract"][k] for k in ("K", "B", "r", "sigma", "T"))
    s_inf = meta["domain"]["s_inf"]
    s_grid = torch.linspace(B + 1e-4, s_inf, n_s, dtype=torch.float64)
    t_grid = torch.linspace(0.0, T - 1e-4, n_t, dtype=torch.float64)
    ss, tt = torch.meshgrid(s_grid, t_grid, indexing="ij")
    previous_dtype = torch.get_default_dtype()
    if meta["hyperparameters"].get("dtype") == "float64":
        torch.set_default_dtype(torch.float64)
    try:
        model = load_trained_model(run_dir, epsilon, meta)
        with torch.no_grad():
            x = torch.stack([ss.reshape(-1), tt.reshape(-1)], dim=1).to(DEVICE).to(torch.get_default_dtype())
            learned = model(x).squeeze().double().cpu().reshape(ss.shape)
        residual = interior_residual(model, ss, tt, r, sigma) if with_residual else None
    finally:
        torch.set_default_dtype(previous_dtype)
    reference = reiner_rubinstein_down_and_out_put(ss, K, B, r, sigma, T - tt)
    return {
        "s_grid": s_grid, "t_grid": t_grid, "learned": learned, "reference": reference,
        "residual": residual, "contract": {"K": K, "B": B, "r": r, "sigma": sigma, "T": T},
        "s_inf": s_inf, "cell_area": float((s_grid[1] - s_grid[0]) * (t_grid[1] - t_grid[0])),
    }


def region_metrics(grid: dict, regions: dict) -> dict:
    """Relative L2 error, residual energy share and uniform-collocation area
    share, per region."""
    error = grid["learned"] - grid["reference"]
    reference = grid["reference"]
    residual = grid["residual"]
    total_residual_energy = float((residual**2).sum()) if residual is not None else None
    out = {}
    for name, region in regions.items():
        mask = region["mask"]
        numerator = float(error[mask].norm())
        denominator = float(reference[mask].norm())
        entry = {
            "definition": region["definition"],
            "rel_l2": numerator / denominator if denominator > 0 else float("nan"),
            "abs_l2": numerator * grid["cell_area"] ** 0.5,
            "kept_area_fraction": float(mask.double().mean()),
        }
        if residual is not None:
            entry["residual_energy_share_kept"] = float((residual[mask] ** 2).sum()) / total_residual_energy
        out[name] = entry
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

FORMULA_COMPARISON = (
    r"$\mathrm{rel}_{L^2}(A)=\|\Phi_\theta-V_{DO}\|_{L^2(A)}/\|V_{DO}\|_{L^2(A)}$ on the kept region $A$, a "
    r"SPACE-TIME region $A\subset\Omega=(B,s_\infty)\times(0,T)$: both norms integrate over $A$ in "
    r"$\mathrm{d}s\,\mathrm{d}t$, so a region named by a price range below stands for that range times $(0,T)$; "
    r"$\mathcal{N}_w=\{(s,t):|s-B|+(T-t)\leq w\}$, the corner lozenge, which is NOT a product; the three "
    r"others are space-time cylinders $I\times(0,T)$ over a price set $I$: $\mathcal{Z}_\varepsilon$ over "
    r"$(B,B+\varepsilon)$, where the smoothing cutoff $\zeta((s-B)/\varepsilon)\neq1$ and the smoothing "
    r"constructions give up the exact terminal trace; $\mathcal{K}_\delta$ over $(K-\delta,K+\delta)$; "
    r"$\mathcal{W}_a$ over $[a,s_\infty)$."
    "\n"
    r"Points: individual master seeds; filled marker: across-seed median. The analytic corner treatments "
    r"(subtraction, enrichment) have no cutoff in $s$ and give up nothing on $\mathcal{Z}_\varepsilon$; excluding it "
    r"removes the region the smoothing constructions sacrifice, not a region where they merely perform badly."
)
FORMULA_ENERGY = (
    r"$E(s)=\int_0^T|\Phi_\theta(s,t)-V_{DO}(s,t)|^2\,\mathrm{d}t$, evaluated as the Riemann sum "
    r"$\sum_j |\Phi_\theta(s,t_j)-V_{DO}(s,t_j)|^2\,\Delta t$ over the $200$ uniformly spaced times "
    r"$t_j\in[0,T-10^{-4}]$ of the evaluation grid (top row, log scale), and its cumulative share "
    r"$\int_B^sE/\int_B^{s_\infty}E$ (bottom row), from the saved models."
    "\n"
    r"Colour: corner treatment, in a darker shade of the same hue for an arm trained with the far-field "
    r"Dirichlet condition on $\Sigma_\infty$. Vertical lines: $s=B$ and $s=K$ (dotted), $s=B+\varepsilon$ (dashed, edge of "
    r"the cutoff cylinder $\mathcal{Z}_\varepsilon$). A treatment whose error is caused by the strike singularity has $E$ "
    r"peaked at $s=K$ and its cumulative share stepping up there; one whose error is caused by the cutoff has "
    r"both inside $\mathcal{Z}_\varepsilon$; one whose error is the far-field component has the step at $s\geq2$."
)
FORMULA_RESIDUAL_ENERGY = (
    r"$R(s)=\int_0^T|\mathcal{L}^{BS}\Phi_\theta(s,t)|^2\,\mathrm{d}t$, the density along the price axis of "
    r"the interior residual the training loss samples, assembled through the route used in training; same "
    r"Riemann sum over the $200$ uniform times as $E(s)$ (top row, log scale), and its cumulative share "
    r"(bottom row). One seed."
    "\n"
    r"Colour: corner treatment, in a darker shade of the same hue for an arm trained with the far-field "
    r"Dirichlet condition on $\Sigma_\infty$. Vertical lines: $s=B$ and $s=K$ (dotted), $s=B+\varepsilon$ (dashed). The "
    r"cutoff residual is supported by the whole cylinder $\mathcal{Z}_\varepsilon$ over $(B,B+\varepsilon)$, where $\zeta'$ and "
    r"$\zeta''$ are nonzero, not by its edge: the smoothing curves rise at $s=B$ and fall back at "
    r"$s=B+\varepsilon$. Compare with $E(s)$: the loss is large where the error is not, and conversely."
)
FORMULA_MAPS = (
    r"Left: $|\Phi_\theta(s,t)-V_{DO}(s,t)|$; right: $|\mathcal{L}^{BS}\Phi_\theta(s,t)|$, the interior PDE "
    r"residual the training loss samples, assembled through the same route as in training (two-term analytic "
    r"when $g_2$ exposes a closed-form residual, one autograd graph otherwise). Both on a logarithmic colour "
    r"scale, shared across configurations; in both colour maps a DARK pixel is a LARGE value and a pale one "
    r"a small value."
    "\n"
    r"Thin grey reference marks (not features of the field): dashed $s=B+\varepsilon$, the far edge of the "
    r"cutoff cylinder $\mathcal{Z}_\varepsilon$ over $(B,B+\varepsilon)$; dotted $s=K$. The error map says where the solution is "
    r"wrong, the residual map where the loss can see it: a region with a large error and a small residual is "
    r"one the interior loss does not penalise."
)


def plot_comparison_by_region(per_configuration: dict, regions: dict, path: Path, iters: int) -> None:
    configurations = [c for c in DEFAULT_CONFIGURATIONS if c in per_configuration]
    names = list(regions)
    n_cols = 3
    n_rows = -(-len(names) // n_cols)
    fig, axes_grid = plt.subplots(n_rows, n_cols, figsize=(max(4.6, 0.78 * len(configurations)) * n_cols,
                                                           5.6 * n_rows), squeeze=False)
    axes = list(axes_grid.reshape(-1))
    for ax in axes[len(names):]:
        ax.set_visible(False)
    positions = range(len(configurations))
    for ax, name in zip(axes, names):
        for position, configuration in zip(positions, configurations):
            values = [seed_metrics[name]["rel_l2"] for seed_metrics in per_configuration[configuration].values()]
            ax.scatter([position] * len(values), values, s=28, alpha=0.45, color="tab:blue", zorder=2)
            ax.scatter([position], [float(np.median(values))], s=70, marker="D", color="tab:blue",
                       edgecolor="black", zorder=3)
        ax.set_yscale("log")
        ax.set_xticks(list(positions))
        ax.set_xticklabels([compact_label(c) for c in configurations], rotation=40, ha="right", fontsize=6.5)
        ax.set_title(regions[name]["label"], fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
    for ax in axes[::n_cols]:
        ax.set_ylabel(r"$\mathrm{rel}_{L^2}$ on the kept region")
    fig.suptitle(f"Down-and-out put — relative $L^2$ error per excluded region, {iters} iterations, "
                 "5 seeds per configuration", fontsize=11)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.2, wspace=0.22, hspace=0.9)
    finalize_figure(fig, path, formula=FORMULA_COMPARISON, axes=axes[:len(names)], formula_fontsize=6.5)


def plot_energy_along_s(grids: dict, path: Path, cutoff_epsilon: float, *, field: str,
                        title: str, density_label: str, cumulative_label: str, formula: str) -> None:
    """One column per terminal function, the three corner treatments superposed
    inside it (colour by treatment): the raw-payoff curves of the subtraction and
    of the enrichment lie on top of each other to plotting accuracy, so a single
    panel holding all ten configurations hides one under the other.

    ``field`` selects what is integrated over time at fixed price: ``"error"``
    for ``Phi_theta - V_DO``, ``"residual"`` for ``L^BS Phi_theta``. Top row: the
    density; bottom row: its cumulative share.
    """
    # Group the two Black-Scholes smoothing routes into one column: they share a
    # terminal function and differ only in how the residual is assembled.
    column_of = {"raw": "raw", "blackscholes": "blackscholes",
                 "blackscholes_analyticres": "blackscholes", "split": "split",
                 "smoothed": "smoothed"}
    columns: dict[str, list[str]] = {}
    for configuration in grids:
        key = column_of.get(terminal_profile_of(configuration), terminal_profile_of(configuration))
        columns.setdefault(key, []).append(configuration)
    order = [k for k in ("raw", "blackscholes", "smoothed", "split") if k in columns]
    first = next(iter(grids.values()))
    K, B = first["contract"]["K"], first["contract"]["B"]

    fig, axes_grid = plt.subplots(2, len(order), figsize=(5.0 * len(order), 8.4), squeeze=False)
    handles: dict[str, object] = {}
    for column, profile in enumerate(order):
        ax_density, ax_cumulative = axes_grid[0, column], axes_grid[1, column]
        for configuration in columns[profile]:
            grid = grids[configuration]
            if field == "residual" and grid.get("residual") is None:
                continue
            s_axis = grid["s_grid"].numpy()
            dt = float(grid["t_grid"][1] - grid["t_grid"][0])
            values = (grid["learned"] - grid["reference"]) if field == "error" else grid["residual"]
            energy = (values ** 2).sum(dim=1).numpy() * dt
            treatment = corner_treatment_of(configuration)
            style = {"color": curve_colour(configuration), "lw": 1.6}
            if terminal_profile_of(configuration) == "blackscholes_analyticres":
                style |= {"linestyle": (0, (4, 1.5)), "lw": 1.3}
            (line,) = ax_density.semilogy(s_axis, np.maximum(energy, 1e-18), **style)
            label = TREATMENT_LABELS[treatment] + (
                " — two-term route" if terminal_profile_of(configuration) == "blackscholes_analyticres" else "")
            if far_field_dirichlet_of(configuration):
                label += f" — {FAR_FIELD_DIRICHLET_LABEL}"
            handles.setdefault(label, line)
            ax_cumulative.plot(s_axis, np.cumsum(energy) / energy.sum(), **style)
        for ax in (ax_density, ax_cumulative):
            ax.axvline(B, color="grey", linestyle=":", lw=1)
            ax.axvline(B + cutoff_epsilon, color="black", linestyle="--", lw=0.9)
            ax.axvline(K, color="grey", linestyle=":", lw=1)
            ax.grid(alpha=0.3, which="both")
        ax_density.set_title(f"Terminal function: {TERMINAL_PROFILE_LABELS.get(profile, profile)}", fontsize=8.5)
        ax_cumulative.set_xlabel("Underlying price $s$")
        ax_cumulative.set_ylim(0, 1.02)
    axes_grid[0, 0].set_ylabel(density_label)
    axes_grid[1, 0].set_ylabel(cumulative_label)
    # One shared vertical scale per row, so the columns are comparable.
    for row in range(2):
        limits = [axes_grid[row, c].get_ylim() for c in range(len(order))]
        low, high = min(l[0] for l in limits), max(l[1] for l in limits)
        for c in range(len(order)):
            axes_grid[row, c].set_ylim(low, high)
    legend = fig.legend(handles=list(handles.values()), labels=list(handles), loc="lower center",
                        bbox_to_anchor=(0.5, 0.12), ncol=2, fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.27, wspace=0.22, hspace=0.25)
    finalize_figure(fig, path, legends=[legend], formula=formula,
                    axes=list(axes_grid.reshape(-1)), formula_fontsize=7)


def plot_error_and_residual_maps(grids: dict, path: Path, cutoff_epsilon: float, s_plot_max: float) -> None:
    configurations = list(grids)
    fig, axes_grid = plt.subplots(len(configurations), 2, figsize=(11, 3.0 * len(configurations)), squeeze=False)
    first = next(iter(grids.values()))
    K, B = first["contract"]["K"], first["contract"]["B"]
    error_floor, residual_floor = 1e-9, 1e-9
    error_max = max(float((g["learned"] - g["reference"]).abs().max()) for g in grids.values())
    residual_max = max(float(g["residual"].abs().max()) for g in grids.values() if g["residual"] is not None)
    for row, configuration in enumerate(configurations):
        grid = grids[configuration]
        s, t = grid["s_grid"].numpy(), grid["t_grid"].numpy()
        keep = s <= s_plot_max
        fields = [
            (np.abs((grid["learned"] - grid["reference"]).numpy()), error_floor, error_max, "magma_r",
             r"$|\Phi_\theta-V_{DO}|$"),
            (np.abs(grid["residual"].numpy()) if grid["residual"] is not None else None,
             residual_floor, residual_max, "viridis_r", r"$|\mathcal{L}^{BS}\Phi_\theta|$"),
        ]
        for column, (field, floor, vmax, cmap, label) in enumerate(fields):
            ax = axes_grid[row, column]
            if field is None:
                ax.set_visible(False)
                continue
            mesh = ax.pcolormesh(t, s[keep], np.maximum(field[keep], floor), shading="auto", cmap=cmap,
                                 norm=plt.matplotlib.colors.LogNorm(vmin=floor, vmax=vmax))
            # Reference marks, drawn thin and grey so they cannot be mistaken for
            # a feature of the field: the cutoff strip is a BAND (B, B+epsilon),
            # not a line at its edge.
            ax.axhline(B + cutoff_epsilon, color="0.55", linestyle="--", lw=0.7, alpha=0.8)
            ax.axhline(K, color="0.55", linestyle=":", lw=0.7, alpha=0.8)
            fig.colorbar(mesh, ax=ax, label=label)
            if row == len(configurations) - 1:
                ax.set_xlabel("Calendar time $t$")
            if column == 0:
                ax.set_ylabel(compact_label(configuration), fontsize=7)
    fig.suptitle("Down-and-out put — error and interior residual in the $(t,s)$ plane (one seed)", fontsize=11)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.96, bottom=0.1, wspace=0.25, hspace=0.25)
    finalize_figure(fig, path, formula=FORMULA_MAPS, axes=list(axes_grid.reshape(-1)), formula_fontsize=7)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aggregation-dir", type=str, default=None,
                        help="Aggregation directory whose summary.yaml lists the runs.")
    parser.add_argument("--configurations", nargs="+", type=str, default=DEFAULT_CONFIGURATIONS)
    parser.add_argument("--map-seed", type=int, default=0, help="Seed whose maps and energy curves are drawn.")
    parser.add_argument("--corner-window", type=float, default=0.1, help="Half-width w of the ell^1 corner lozenge.")
    parser.add_argument("--cutoff-epsilon", type=float, default=0.1,
                        help="Bandwidth epsilon of the smoothing cutoff: the strip s - B < epsilon is the region "
                             "where zeta != 1, i.e. where the smoothing constructions give up the exact trace.")
    parser.add_argument("--strike-delta", type=float, default=0.1, help="Half-width of the excluded strike band.")
    parser.add_argument("--far-field-start", type=float, default=2.0, help="Lower end of the excluded far field.")
    parser.add_argument("--n-s", type=int, default=400,
                        help="Grid points in s. The metrics are integrals of a continuous field: refining "
                             "300x100 to 2400x800 moves them by at most 1.3 per cent (methodology 15.5), so "
                             "the default trades resolution against the cost of the smoothing split runs, whose "
                             "g2 is a fixed-grid quadrature re-evaluated at every point.")
    parser.add_argument("--n-t", type=int, default=200, help="Grid points in t.")
    parser.add_argument("--s-plot-max", type=float, default=2.0, help="Upper end of the plotted s range on the maps.")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--replot", type=str, default=None, metavar="OUT_DIR",
                        help="Rebuild the figures from a previous run's grids.pt and metrics.yaml.")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    if args.replot:
        out_dir = Path(args.replot)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
        payload = torch.load(out_dir / "grids.pt", weights_only=False)
        per_configuration, grids, regions_meta, iters = (
            payload["per_configuration"], payload["grids"], payload["regions"], payload["iters"])
    else:
        if args.aggregation_dir is None:
            parser.error("--aggregation-dir is required unless --replot is given.")
        out_dir = Path(args.out_dir) if args.out_dir else script_data_dir(__file__) / (
            datetime.now().astimezone().strftime("%Y%m%d_%H%M%S"))
        out_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
                            handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "diagnose.log")])
        logger.info(f"Command: {' '.join(sys.argv)}")
        logger.info(f"PyTorch {torch.__version__}, device {DEVICE}, float64, grid {args.n_s}x{args.n_t}")
        with open(Path(args.aggregation_dir) / "summary.yaml") as f:
            summary = yaml.safe_load(f)
        iters = summary["iters"]
        repo_root = find_repo_root(Path(__file__).resolve())
        per_configuration: dict[str, dict[int, dict]] = {}
        grids: dict[str, dict] = {}
        regions_meta: dict[str, dict] = {}
        for configuration in args.configurations:
            entry = summary["configurations"].get(configuration)
            if entry is None:
                continue
            per_configuration[configuration] = {}
            for seed, run_dir_text in sorted(entry["runs"].items()):
                run_dir = Path(run_dir_text)
                if not run_dir.is_absolute():
                    run_dir = repo_root / run_dir
                seed = int(seed)
                is_map_seed = seed == args.map_seed
                started = time.time()
                grid = evaluate_grid(run_dir, _run_epsilon(run_dir), args.n_s, args.n_t,
                                     with_residual=is_map_seed)
                contract = grid["contract"]
                ss, tt = torch.meshgrid(grid["s_grid"], grid["t_grid"], indexing="ij")
                regions = exclusion_regions(ss, tt, contract["K"], contract["B"], contract["T"],
                                            args.corner_window, args.cutoff_epsilon, args.strike_delta,
                                            args.far_field_start)
                regions_meta = {name: {"label": region["label"], "definition": region["definition"],
                                       "kept_area_fraction": float(region["mask"].double().mean())}
                                for name, region in regions.items()}
                per_configuration[configuration][seed] = region_metrics(grid, regions)
                if is_map_seed:
                    grids[configuration] = grid
                logger.info(f"  {configuration:<26s} seed {seed} ({time.time() - started:.0f}s): " + "  ".join(
                    f"{name}={values['rel_l2']:.3e}" for name, values in per_configuration[configuration][seed].items()))
        torch.save({"per_configuration": per_configuration, "grids": grids, "regions": regions_meta,
                    "iters": iters}, out_dir / "grids.pt")
        with open(out_dir / "metrics.yaml", "w") as f:
            yaml.dump({"command": " ".join(sys.argv), "iters": iters, "regions": regions_meta,
                       "metrics": per_configuration}, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Metrics saved -> {out_dir / 'metrics.yaml'}")

    (out_dir / "figures").mkdir(exist_ok=True)
    regions_for_plot = {name: {"label": meta["label"]} for name, meta in regions_meta.items()}
    plot_comparison_by_region(per_configuration, regions_for_plot,
                              out_dir / "figures" / "comparison_by_exclusion_region.png", iters)
    if grids:
        plot_energy_along_s(
            grids, out_dir / "figures" / "error_energy_along_s.png", args.cutoff_epsilon, field="error",
            title="Down-and-out put — distribution of the squared error along the price axis (one seed)",
            density_label=r"$E(s)=\int_0^T|\Phi_\theta-V_{DO}|^2\,\mathrm{d}t$",
            cumulative_label=r"cumulative share $\int_B^s E\,/\int_B^{s_\infty} E$",
            formula=FORMULA_ENERGY,
        )
        plot_energy_along_s(
            grids, out_dir / "figures" / "residual_energy_along_s.png", args.cutoff_epsilon, field="residual",
            title="Down-and-out put — distribution of the squared interior residual along the price axis (one seed)",
            density_label=r"$R(s)=\int_0^T|\mathcal{L}^{BS}\Phi_\theta|^2\,\mathrm{d}t$",
            cumulative_label=r"cumulative share $\int_B^s R\,/\int_B^{s_\infty} R$",
            formula=FORMULA_RESIDUAL_ENERGY,
        )
        plot_error_and_residual_maps(grids, out_dir / "figures" / "error_and_residual_maps.png",
                                     args.cutoff_epsilon, args.s_plot_max)
    logger.info(f"Figures -> {out_dir / 'figures'}")


def _run_epsilon(run_dir: Path) -> float:
    import re
    match = re.search(r"_eps([0-9.]+)_seed", run_dir.name)
    return float(match.group(1)) if match else 0.0


if __name__ == "__main__":
    main()
