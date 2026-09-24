r"""Why the corner enrichment is behind the exact subtraction: the two singular parts side by side.

Reference: working note "A rigorous statement of exact-constraint learning at a
conflicting constraint corner: the knock-out barrier option" (S. Ouaissi
internship, 2026-06-24), Sections 5.1 (Method 1) and 5.2 (Method 2), and
sections 15 to 17 of ``documents/methodology/barrier_option.md``.

Both analytic corner resolutions write the extension as a singular part that
reproduces the corner jump plus a regular remainder,

.. math::

    g_2(s,t) = S(s,t) + h(s,t),\qquad \Delta = K - B,

and differ only in the choice of :math:`S`:

- **Exact subtraction** (Definition 7): :math:`S_{\mathrm{sub}} = \Delta\,V_{DOD}(s,t)`
  with :math:`V_{DOD}` the down-and-out digital of the same contract, and
  :math:`h = \pi - \pi(B,\cdot)`.  The digital is an exact solution of the
  Black-Scholes operator, so :math:`\mathcal L^{BS}S_{\mathrm{sub}} = 0`
  identically (Proposition 4).
- **Corner enrichment** (Definition 8): :math:`S_{\mathrm{enr}} = \chi(s)\,\Delta\,\operatorname{erf}(\xi)`
  with :math:`\xi = \ln(s/B)/(\sigma\sqrt{2(T-t)})` the similarity variable of
  the short-time limit and :math:`\chi` a fixed :math:`C^\infty` cutoff in
  :math:`s` (radii :math:`\delta_0`, :math:`\delta_1`), and
  :math:`h = \pi - \chi(s)\,\pi(B,\cdot)`.  The similarity profile solves the
  heat equation of the leading-order corner problem, not the full operator, so
  :math:`\mathcal L^{BS}S_{\mathrm{enr}}` is square-integrable but nonzero
  (Proposition 5); the cutoff adds its commutator terms.

The two extensions reproduce the same two traces exactly, so the trained prices
are not separated by their constraints.  What separates them is the forcing the
network has to absorb.  This script evaluates, on a grid of the training domain
and on a corner zoom, in float64 and without any trained network:

- the two singular parts and their pointwise difference,
- the interior residual :math:`\mathcal L^{BS}S` of each,
- the difference of the two complete extensions :math:`g_2`, which is what a
  trained trial solution actually sees.

Everything is written to ``grids.pt`` so the figures can be rebuilt with
``--replot`` without recomputation.

Outputs (under ``data/compare_singular_parts_subtraction_enrichment/<timestamp>/``):

- ``figures/singular_parts_and_difference.png`` -- six heatmaps over
  :math:`(s,t)`: the two singular parts, their difference, the two residuals,
  and the difference of the complete extensions.
- ``figures/singular_parts_corner_zoom.png`` -- the same difference and the two
  residuals on a corner zoom.
- ``figures/singular_parts_slices.png`` -- slices along :math:`s` at several
  calendar times, where the magnitudes are read off quantitatively.
- ``summary.yaml`` -- the measured maxima and norms quoted in the note.
"""
from __future__ import annotations

import argparse
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from matplotlib.colors import Normalize, SymLogNorm, TwoSlopeNorm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from learning_option_pricing.pricing.barrier import (  # noqa: E402
    make_corner_enriched_extension,
    make_subtracted_digital_extension,
)
from learning_option_pricing.utils.figure_layout import finalize_figure  # noqa: E402
from learning_option_pricing.utils.run_context import script_data_dir  # noqa: E402

logger = logging.getLogger("compare_singular_parts_subtraction_enrichment")

#: Contract of the 50000-iteration batches (sections 15.4, 16.4 and 17.3).
DEFAULT_CONTRACT = {"K": 1.0, "B": 0.6, "r": 0.03, "sigma": 0.3, "T": 1.0}
DEFAULT_S_INF = 3.0
#: Cutoff radii of the corner enrichment used by those batches.
DEFAULT_DELTA0, DEFAULT_DELTA1 = 0.1, 0.3
#: Calendar times of the slice figure, chosen to straddle the corner layer.
DEFAULT_SLICE_TIMES = (0.0, 0.5, 0.9, 0.99)

SEQUENTIAL_COLOURMAP = "viridis"
DIVERGING_COLOURMAP = "RdBu_r"

FORMULA_SINGULAR = (
    r"$\Delta=K-B$;  exact subtraction: $S_{\rm sub}(s,t)=\Delta\,V_{DOD}(s,t)$, "
    r"$V_{DOD}$ = down-and-out digital (method of images), $\mathcal{L}^{BS}S_{\rm sub}=0$ "
    r"exactly (Proposition 4)."
    "\n"
    r"Corner enrichment: $S_{\rm enr}(s,t)=\chi(s)\,\Delta\,\mathrm{erf}(\xi)$, "
    r"$\xi=\ln(s/B)/(\sigma\sqrt{2(T-t)})$, $\chi=1$ on $s-B\leq\delta_0$ and $0$ on "
    r"$s-B\geq\delta_1$;  $\mathcal{L}^{BS}S_{\rm enr}\neq0$, square-integrable (Proposition 5)."
    "\n"
    r"$\mathcal{L}^{BS}V=\partial_tV+\frac{1}{2}\sigma^2s^2\partial_{ss}V+rs\,\partial_sV-rV$;  "
    r"complete extension $g_2=S+h$, $h=\pi-\chi\,\pi(B,\cdot)$ ($\chi\equiv1$ for the subtraction), "
    r"$\pi$ = Black-Scholes put price."
    "\n"
    r"Dotted verticals: $s=B$, $s=B+\delta_0$, $s=B+\delta_1$, $s=K$. Evaluated in float64, no "
    r"trained network."
)


def build_extensions(contract: dict, delta0: float, delta1: float, terminal_profile: str):
    """The exact-subtraction and corner-enrichment extensions of the batches."""
    K, B, r, sigma, T = (contract[k] for k in ("K", "B", "r", "sigma", "T"))
    profile_options = dict(comparison_volatility=sigma, y_lo=None, y_hi=None,
                           n_quad=8000, split_profile="closed_form")
    subtraction = make_subtracted_digital_extension(K, B, r, sigma, T, terminal_profile, **profile_options)
    enrichment = make_corner_enriched_extension(K, B, r, sigma, T, terminal_profile,
                                                delta0=delta0, delta1=delta1, **profile_options)
    return subtraction, enrichment


def evaluate_on_grid(subtraction, enrichment, s_values: torch.Tensor, t_values: torch.Tensor) -> dict:
    """Every field of the comparison on the tensor product ``s_values x t_values``.

    Returns arrays of shape ``(len(t_values), len(s_values))``, the orientation
    ``imshow`` expects with time on the vertical axis.
    """
    s_mesh, t_mesh = torch.meshgrid(s_values, t_values, indexing="ij")
    s_flat, t_flat = s_mesh.reshape(-1), t_mesh.reshape(-1)

    def as_grid(flat: torch.Tensor) -> torch.Tensor:
        return flat.reshape(s_mesh.shape).T.contiguous()

    fields = {
        "singular_subtraction": subtraction.singular_part(s_flat, t_flat),
        "singular_enrichment": enrichment.singular_part(s_flat, t_flat),
        "residual_singular_subtraction": subtraction.singular_residual(s_flat, t_flat),
        "residual_singular_enrichment": enrichment.singular_residual(s_flat, t_flat),
        "extension_subtraction": subtraction(s_flat, t_flat),
        "extension_enrichment": enrichment(s_flat, t_flat),
    }
    grids = {name: as_grid(value) for name, value in fields.items()}
    grids["singular_difference"] = grids["singular_enrichment"] - grids["singular_subtraction"]
    grids["extension_difference"] = grids["extension_enrichment"] - grids["extension_subtraction"]
    grids["s_values"], grids["t_values"] = s_values, t_values
    return grids


def _symmetric_diverging_norm(values: np.ndarray) -> TwoSlopeNorm:
    """Diverging norm centred on zero, so that the sign of a difference is read
    from the colour and not from the colour-bar ticks."""
    extent = float(np.nanmax(np.abs(values)))
    extent = extent if extent > 0.0 else 1.0
    return TwoSlopeNorm(vmin=-extent, vcenter=0.0, vmax=extent)


def _symmetric_log_norm(values: np.ndarray) -> SymLogNorm:
    """Diverging norm on a symmetric-logarithmic scale, for a residual whose
    magnitude spans several decades between the corner and the far field."""
    extent = float(np.nanmax(np.abs(values)))
    extent = extent if extent > 0.0 else 1.0
    positive = np.abs(values[np.abs(values) > 0.0])
    linear_threshold = float(np.percentile(positive, 5)) if positive.size else extent * 1e-6
    return SymLogNorm(linthresh=max(linear_threshold, extent * 1e-8), vmin=-extent, vmax=extent)


def _draw_panel(ax, grid: torch.Tensor, s_values: torch.Tensor, t_values: torch.Tensor,
                title: str, colourmap: str, norm, contract: dict, delta0: float, delta1: float):
    values = grid.numpy()
    if not np.any(values):
        # An identically-zero field has no colour scale to show: a diverging norm
        # would invent a range and the panel would read as "small", not as "zero".
        ax.text(0.5, 0.5, "identically zero\n(every grid value exactly $0$ in float64)",
                transform=ax.transAxes, ha="center", va="center", fontsize=10)
        ax.set_xlim(float(s_values[0]), float(s_values[-1]))
        ax.set_ylim(float(t_values[0]), float(t_values[-1]))
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Underlying price $s$")
        ax.set_ylabel("Calendar time $t$")
        return None
    image = ax.imshow(
        values, origin="lower", aspect="auto", cmap=colourmap, norm=norm,
        extent=[float(s_values[0]), float(s_values[-1]), float(t_values[0]), float(t_values[-1])],
    )
    for position in (contract["B"], contract["B"] + delta0, contract["B"] + delta1, contract["K"]):
        if float(s_values[0]) <= position <= float(s_values[-1]):
            ax.axvline(position, color="white", linestyle=":", lw=0.9, alpha=0.8)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Underlying price $s$")
    ax.set_ylabel("Calendar time $t$")
    return image


def plot_heatmaps(grids: dict, path: Path, contract: dict, delta0: float, delta1: float,
                  title: str) -> None:
    """Six heatmaps: the two singular parts, their difference, the two interior
    residuals, and the difference of the complete extensions."""
    s_values, t_values = grids["s_values"], grids["t_values"]
    shared_extent = max(float(grids["singular_subtraction"].abs().max()),
                        float(grids["singular_enrichment"].abs().max()))
    difference_extent = float(grids["singular_difference"].abs().max())
    panels = [
        ("singular_subtraction",
         r"(a) Exact subtraction: $S_{\rm sub}=\Delta\,V_{DOD}$", SEQUENTIAL_COLOURMAP,
         Normalize(vmin=0.0, vmax=shared_extent)),
        ("singular_enrichment",
         r"(b) Corner enrichment: $S_{\rm enr}=\chi\,\Delta\,\mathrm{erf}(\xi)$"
         "\n" + rf"(panels (a) and (b) share the scale $[0, {shared_extent:.3g}]$)",
         SEQUENTIAL_COLOURMAP, Normalize(vmin=0.0, vmax=shared_extent)),
        ("singular_difference",
         rf"(c) $S_{{\rm enr}}-S_{{\rm sub}}$, reaching ${difference_extent:.3g}$ in magnitude",
         DIVERGING_COLOURMAP, _symmetric_diverging_norm(grids["singular_difference"].numpy())),
        ("residual_singular_subtraction",
         r"(d) $\mathcal{L}^{BS}S_{\rm sub}$ (zero by Proposition 4)",
         DIVERGING_COLOURMAP, _symmetric_diverging_norm(grids["residual_singular_subtraction"].numpy())),
        ("residual_singular_enrichment",
         r"(e) $\mathcal{L}^{BS}S_{\rm enr}$ (symlog): the forcing the network must absorb",
         DIVERGING_COLOURMAP, _symmetric_log_norm(grids["residual_singular_enrichment"].numpy())),
        ("extension_difference",
         r"(f) $g_2^{\rm enr}-g_2^{\rm sub}$: what the trial solution sees",
         DIVERGING_COLOURMAP, _symmetric_diverging_norm(grids["extension_difference"].numpy())),
    ]
    fig, axes_grid = plt.subplots(2, 3, figsize=(16.5, 9.0))
    for ax, (name, panel_title, colourmap, norm) in zip(axes_grid.reshape(-1), panels):
        image = _draw_panel(ax, grids[name], s_values, t_values, panel_title, colourmap, norm,
                            contract, delta0, delta1)
        if image is not None:
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(title, fontsize=11)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.26, wspace=0.28, hspace=0.38)
    finalize_figure(fig, path, formula=FORMULA_SINGULAR, axes=list(axes_grid.reshape(-1)),
                    formula_fontsize=7)


def plot_corner_zoom(grids: dict, path: Path, contract: dict, delta0: float, delta1: float,
                     title: str) -> None:
    """The difference and the two residuals on the corner zoom alone."""
    s_values, t_values = grids["s_values"], grids["t_values"]
    panels = [
        ("singular_difference", r"$S_{\rm enr}-S_{\rm sub}$", DIVERGING_COLOURMAP,
         _symmetric_diverging_norm(grids["singular_difference"].numpy())),
        ("residual_singular_enrichment", r"$\mathcal{L}^{BS}S_{\rm enr}$ (symlog)",
         DIVERGING_COLOURMAP, _symmetric_log_norm(grids["residual_singular_enrichment"].numpy())),
        ("extension_difference", r"$g_2^{\rm enr}-g_2^{\rm sub}$", DIVERGING_COLOURMAP,
         _symmetric_diverging_norm(grids["extension_difference"].numpy())),
    ]
    fig, axes_grid = plt.subplots(1, 3, figsize=(16.5, 5.2))
    for ax, (name, panel_title, colourmap, norm) in zip(axes_grid, panels):
        image = _draw_panel(ax, grids[name], s_values, t_values, panel_title, colourmap, norm,
                            contract, delta0, delta1)
        if image is not None:
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(title, fontsize=11)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.86, bottom=0.36, wspace=0.28)
    finalize_figure(fig, path, formula=FORMULA_SINGULAR, axes=list(axes_grid), formula_fontsize=7)


def plot_slices(slices: dict, path: Path, contract: dict, delta0: float, delta1: float,
                title: str) -> None:
    """Slices along ``s`` at several calendar times: the two singular parts, their
    difference, and the two interior residuals."""
    times = slices["times"]
    s_values = slices["s_values"].numpy()
    fig, axes_grid = plt.subplots(3, len(times), figsize=(4.4 * len(times), 10.5), squeeze=False)
    handles: list = []
    for column, t_value in enumerate(times):
        entry = slices[t_value]
        top, middle, bottom = axes_grid[0, column], axes_grid[1, column], axes_grid[2, column]

        (line_sub,) = top.plot(s_values, entry["singular_subtraction"].numpy(), lw=1.6,
                               color="#009E73", label=r"$S_{\rm sub}=\Delta\,V_{DOD}$ (exact subtraction)")
        (line_enr,) = top.plot(s_values, entry["singular_enrichment"].numpy(), lw=1.6,
                               color="#7B3294", label=r"$S_{\rm enr}=\chi\,\Delta\,\mathrm{erf}(\xi)$ (enrichment)")
        top.set_ylabel(r"Singular part $S(s,t)$")

        (line_diff,) = middle.plot(s_values, entry["singular_difference"].numpy(), lw=1.6,
                                   color="#D55E00", label=r"$S_{\rm enr}-S_{\rm sub}$")
        (line_ext,) = middle.plot(s_values, entry["extension_difference"].numpy(), lw=1.4,
                                  color="#0072B2", label=r"$g_2^{\rm enr}-g_2^{\rm sub}$")
        middle.axhline(0.0, color="grey", lw=0.6)
        middle.set_ylabel("Difference")

        enrichment_residual = np.abs(entry["residual_singular_enrichment"].numpy())
        subtraction_residual = np.abs(entry["residual_singular_subtraction"].numpy())
        # A logarithmic axis cannot show an exact zero. Rather than shifting the
        # series by an epsilon -- which would draw a curve where there is no
        # value and set the axis range from that epsilon -- the identically-zero
        # series is stated in the panel and only the nonzero one is drawn.
        (line_enrichment,) = bottom.semilogy(
            s_values, np.ma.masked_where(enrichment_residual == 0.0, enrichment_residual),
            lw=1.6, color="#7B3294", label=r"$|\mathcal{L}^{BS}S_{\rm enr}|$ (corner enrichment)")
        line_zero = line_enrichment
        if np.any(subtraction_residual):
            (line_zero,) = bottom.semilogy(
                s_values, np.ma.masked_where(subtraction_residual == 0.0, subtraction_residual),
                lw=1.6, color="#009E73", label=r"$|\mathcal{L}^{BS}S_{\rm sub}|$ (exact subtraction)")
        else:
            bottom.text(0.5, 0.06, r"$\mathcal{L}^{BS}S_{\rm sub} = 0$ at every grid point (float64)",
                        transform=bottom.transAxes, ha="center", va="bottom", fontsize=8,
                        color="#009E73")
        bottom.set_ylabel(r"$|\mathcal{L}^{BS}S|$")
        bottom.set_xlabel("Underlying price $s$")

        for ax in (top, middle, bottom):
            for position in (contract["B"], contract["B"] + delta0,
                             contract["B"] + delta1, contract["K"]):
                ax.axvline(position, color="grey", linestyle=":", lw=0.9)
            ax.grid(True, which="both", alpha=0.3)
            # The residual row's data support is narrower than the others';
            # letting it autoscale would put three different price axes in one
            # column and invite reading the rows against each other wrongly.
            ax.set_xlim(float(s_values[0]), float(s_values[-1]))
        top.set_title(f"$t = {t_value:g}$  ($T-t = {contract['T'] - t_value:g}$)", fontsize=9)
        if column == 0:
            handles = [line_sub, line_enr, line_diff, line_ext, line_enrichment]
            if line_zero is not line_enrichment:
                handles.append(line_zero)

    legend = fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.135), ncol=3, fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.21, wspace=0.28, hspace=0.25)
    finalize_figure(fig, path, legends=[legend], formula=FORMULA_SINGULAR,
                    axes=list(axes_grid.reshape(-1)), formula_fontsize=7)


def measured_summary(grids: dict, zoom: dict, slices: dict) -> dict:
    """The numbers the note quotes, read off the evaluated grids."""

    def statistics(grid: torch.Tensor) -> dict:
        values = grid.reshape(-1)
        return {
            "max_abs": float(values.abs().max()),
            "mean_abs": float(values.abs().mean()),
            "l2": float(torch.linalg.vector_norm(values) / values.numel() ** 0.5),
        }

    summary = {
        "whole_domain": {name: statistics(grids[name]) for name in (
            "singular_subtraction", "singular_enrichment", "singular_difference",
            "residual_singular_subtraction", "residual_singular_enrichment",
            "extension_difference")},
        "corner_zoom": {name: statistics(zoom[name]) for name in (
            "singular_difference", "residual_singular_subtraction",
            "residual_singular_enrichment", "extension_difference")},
    }
    s_values = grids["s_values"]
    difference = grids["singular_difference"]
    flat_index = int(difference.abs().reshape(-1).argmax())
    row, column = divmod(flat_index, difference.shape[1])
    summary["argmax_singular_difference"] = {
        "s": float(s_values[column]), "t": float(grids["t_values"][row]),
        "value": float(difference[row, column]),
    }
    summary["slice_times"] = list(slices["times"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-s", type=int, default=600, help="Price nodes of the whole-domain grid.")
    parser.add_argument("--n-t", type=int, default=400, help="Time nodes of the whole-domain grid.")
    parser.add_argument("--s-max", type=float, default=DEFAULT_S_INF,
                        help="Upper end of the price axis of the whole-domain grid.")
    parser.add_argument("--zoom-width", type=float, default=0.4,
                        help="Width in s of the corner zoom, measured from the barrier.")
    parser.add_argument("--zoom-n-s", type=int, default=400, help="Price nodes of the corner zoom.")
    parser.add_argument("--zoom-n-t", type=int, default=400, help="Time nodes of the corner zoom.")
    parser.add_argument("--enrichment-delta0", type=float, default=DEFAULT_DELTA0,
                        help="Inner cutoff radius of the corner enrichment, in price units.")
    parser.add_argument("--enrichment-delta1", type=float, default=DEFAULT_DELTA1,
                        help="Outer cutoff radius of the corner enrichment, in price units.")
    parser.add_argument("--terminal-profile", type=str, default="black_scholes",
                        choices=["raw", "black_scholes", "split"],
                        help="Terminal profile pi of the regular part. Only the complete extension "
                             "g2 depends on it; the singular parts do not.")
    parser.add_argument("--slice-times", nargs="+", type=float, default=list(DEFAULT_SLICE_TIMES),
                        help="Calendar times of the slice figure.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory override.")
    parser.add_argument("--replot", type=str, default=None, metavar="OUT_DIR",
                        help="Rebuild the figures from the grids.pt of a previous run; computes nothing.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("matplotlib.mathtext").setLevel(logging.WARNING)

    if args.replot is not None:
        out_dir = Path(args.replot)
        payload = torch.load(out_dir / "grids.pt", weights_only=False)
        logger.info(f"--replot: grids read from {out_dir / 'grids.pt'} (nothing recomputed)")
    else:
        out_dir = Path(args.out_dir) if args.out_dir else script_data_dir(__file__) / (
            datetime.now().strftime("%Y%m%d_%H%M%S")
            + f"_{args.terminal_profile}_d0{args.enrichment_delta0:g}_d1{args.enrichment_delta1:g}")
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = None

    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(out_dir / "diagnostic.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(file_handler)

    logger.info("Singular parts of the two analytic corner resolutions, side by side")
    logger.info(f"  Output directory: {out_dir.resolve()}")
    logger.info(f"  Log file (follow in real time): {(out_dir / 'diagnostic.log').resolve()}")
    logger.info(f"  Command: {' '.join([Path(sys.argv[0]).name] + sys.argv[1:])}")
    logger.info(f"  Host: {platform.node()}")
    logger.info(f"  Python: {platform.python_version()}")
    logger.info(f"  PyTorch: {torch.__version__}")
    logger.info("  No RNG is used: every field is a closed-form evaluation, no trained network.")

    if payload is None:
        torch.set_default_dtype(torch.float64)
        contract = dict(DEFAULT_CONTRACT)
        subtraction, enrichment = build_extensions(
            contract, args.enrichment_delta0, args.enrichment_delta1, args.terminal_profile)
        logger.info(f"  Contract: " + ", ".join(f"{k}={v:g}" for k, v in contract.items()))
        logger.info(f"  Corner jump Delta = K - B = {contract['K'] - contract['B']:g}")
        logger.info(f"  Enrichment cutoff radii: delta0={args.enrichment_delta0:g}, "
                    f"delta1={args.enrichment_delta1:g} (in price units, from the barrier)")
        logger.info(f"  Terminal profile of the regular part: {args.terminal_profile}")
        logger.info(f"  Whole-domain grid: {args.n_s} x {args.n_t} on "
                    f"({contract['B']:g}, {args.s_max:g}) x (0, {contract['T']:g})")
        logger.info(f"  Corner zoom: {args.zoom_n_s} x {args.zoom_n_t} on "
                    f"({contract['B']:g}, {contract['B'] + args.zoom_width:g}) x (0, {contract['T']:g})")

        s_values = torch.linspace(contract["B"], args.s_max, args.n_s)
        t_values = torch.linspace(0.0, contract["T"], args.n_t)
        grids = evaluate_on_grid(subtraction, enrichment, s_values, t_values)

        s_zoom = torch.linspace(contract["B"], contract["B"] + args.zoom_width, args.zoom_n_s)
        t_zoom = torch.linspace(0.0, contract["T"], args.zoom_n_t)
        zoom = evaluate_on_grid(subtraction, enrichment, s_zoom, t_zoom)

        slices: dict = {"times": list(args.slice_times), "s_values": s_values}
        for t_value in args.slice_times:
            t_column = torch.full_like(s_values, float(t_value))
            entry = evaluate_on_grid(subtraction, enrichment, s_values, torch.tensor([float(t_value)]))
            slices[t_value] = {name: entry[name][0] for name in (
                "singular_subtraction", "singular_enrichment", "singular_difference",
                "residual_singular_subtraction", "residual_singular_enrichment",
                "extension_difference")}
            del t_column

        payload = {"grids": grids, "zoom": zoom, "slices": slices, "contract": contract,
                   "delta0": args.enrichment_delta0, "delta1": args.enrichment_delta1,
                   "terminal_profile": args.terminal_profile,
                   "generated": datetime.now(timezone.utc).isoformat()}
        torch.save(payload, out_dir / "grids.pt")
        logger.info(f"  Grids saved -> {(out_dir / 'grids.pt').resolve()} "
                    f"(--replot rebuilds every figure from this file)")

    grids, zoom, slices = payload["grids"], payload["zoom"], payload["slices"]
    contract, delta0, delta1 = payload["contract"], payload["delta0"], payload["delta1"]

    summary = measured_summary(grids, zoom, slices)
    logger.info("Measured on the whole domain:")
    for name, stats in summary["whole_domain"].items():
        logger.info(f"    {name:<34} max|.|={stats['max_abs']:.4e}  mean|.|={stats['mean_abs']:.4e}  "
                    f"rms={stats['l2']:.4e}")
    logger.info("Measured on the corner zoom:")
    for name, stats in summary["corner_zoom"].items():
        logger.info(f"    {name:<34} max|.|={stats['max_abs']:.4e}  mean|.|={stats['mean_abs']:.4e}  "
                    f"rms={stats['l2']:.4e}")
    argmax = summary["argmax_singular_difference"]
    logger.info(f"  Largest |S_enr - S_sub|: {argmax['value']:.4e} at s={argmax['s']:.4f}, "
                f"t={argmax['t']:.4f}")

    with open(out_dir / "summary.yaml", "w") as handle:
        yaml.safe_dump({"contract": contract, "delta0": delta0, "delta1": delta1,
                        "terminal_profile": payload["terminal_profile"], "measured": summary},
                       handle, sort_keys=False)
    logger.info(f"  Summary saved -> {(out_dir / 'summary.yaml').resolve()}")

    plot_heatmaps(grids, out_dir / "figures" / "singular_parts_and_difference.png",
                  contract, delta0, delta1,
                  "Down-and-out put — singular parts of the exact subtraction and of the corner "
                  "enrichment, and what separates them")
    plot_corner_zoom(zoom, out_dir / "figures" / "singular_parts_corner_zoom.png",
                     contract, delta0, delta1,
                     "Down-and-out put — the same differences on a corner zoom "
                     f"($B \\leq s \\leq B + {float(zoom['s_values'][-1]) - contract['B']:g}$)")
    plot_slices(slices, out_dir / "figures" / "singular_parts_slices.png",
                contract, delta0, delta1,
                "Down-and-out put — singular parts, their difference and their interior residuals "
                "along $s$ at fixed calendar times")
    logger.info(f"  Figures -> {(out_dir / 'figures').resolve()}")


if __name__ == "__main__":
    main()
