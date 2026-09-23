r"""Down-and-out put: price, Delta and Gamma profiles of the three corner treatments.

Loads ONE saved model per configuration of an aggregation (the seed given by
``--seed``) and evaluates, in float64 and without retraining, along the price
axis at fixed calendar times and along the time axis at the strike:

    Phi_theta(s, t),   d_s Phi_theta(s, t),   d_ss Phi_theta(s, t)

against the Reiner-Rubinstein closed form V_DO, its Delta (autograd of the
closed form) and its Gamma (``reiner_rubinstein_down_and_out_put_gamma``).
The trained side is differentiated as in ``evaluate_greeks_no_corner.py``: the
network manifold g1 u_theta by two nested autograd passes, the extension g2
analytically when it exposes ``first_price_derivative`` /
``second_price_derivative`` (split, subtraction, enrichment) and by autograd
with gradients enabled otherwise. Every derivative is pointwise -- no grid
quadrature of a curvature enters any number here, so the strike peak's width
is not a resolution issue (see the methodology document, section 15.5).

Three figures, all from the same saved curves (``curves.pt``, replot with
``--replot``):

- ``profiles_price_delta_gamma.png``: rows Phi, d_s Phi, d_ss Phi; columns t in
  --times; one line per configuration, closed form dashed.
- ``absolute_errors_along_s.png``: |Phi - V_DO|, |d_s Phi - d_s V_DO|,
  |d_ss Phi - d_ss V_DO| along s (log scale) -- WHERE each treatment's error is.
- ``greeks_at_strike_vs_time.png``: Delta(K, t) and Gamma(K, t), numerical and
  exact, against t, plus their relative errors.
- ``profiles_price_delta_gamma_corner_zoom.png``: the first figure restricted to
  the corner region s in (B, B + --zoom-width), evaluated on its own dense grid
  (--zoom-n-s points), where the Gamma of the smoothing runs oscillates.

Usage:
    python3 experiments/python_scripts/exp_barrier_option/compare_corner_treatments_profiles.py \
        --aggregation-dir data/aggregate_terminal_function_comparison/<dir> \
        --configurations blackscholes subtraction_blackscholes enrichment_blackscholes
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from learning_option_pricing.pricing.barrier import (  # noqa: E402
    reiner_rubinstein_down_and_out_put,
    reiner_rubinstein_down_and_out_put_gamma,
)
from learning_option_pricing.utils.figure_layout import finalize_figure  # noqa: E402
from learning_option_pricing.utils.run_context import find_repo_root, script_data_dir  # noqa: E402
from aggregate_terminal_function_comparison import CONFIGURATION_LABELS  # noqa: E402
from pilot_down_and_out_put import DEVICE, load_trained_model, read_run_metadata  # noqa: E402

logger = logging.getLogger("compare_corner_treatments_profiles")

DEFAULT_TIMES = [0.0, 0.5, 0.9, 0.99]
DEFAULT_STRIKE_TIMES = np.linspace(0.0, 0.99, 100).tolist()
DEFAULT_CONFIGURATIONS = [
    "blackscholes", "split",
    "subtraction_blackscholes", "subtraction_split",
    "enrichment_blackscholes", "enrichment_split",
]
COLOURS = {
    "raw": "tab:brown", "smoothed": "tab:olive",
    "blackscholes": "tab:blue", "blackscholes_analyticres": "tab:purple", "split": "tab:red",
    "subtraction_raw": "peru", "subtraction_blackscholes": "tab:green", "subtraction_split": "tab:orange",
    "enrichment_raw": "tan", "enrichment_blackscholes": "tab:cyan", "enrichment_split": "tab:pink",
}

FORMULA_PROFILES = (
    r"Trained (solid): $\Phi_\theta=g_1u_\theta+g_2$; $\partial_s\Phi_\theta$, $\partial_{ss}\Phi_\theta$ = "
    r"two nested autograd passes on $g_1u_\theta$ + analytic derivatives of $g_2$ (split, subtraction, "
    r"enrichment) or autograd on $g_2$ (Black-Scholes smoothing). Pointwise, no grid quadrature."
    "\n"
    r"Reference (dashed): $V_{DO}$ = Reiner-Rubinstein closed form; $\partial_sV_{DO}$ by autograd of the "
    r"closed form; $\partial_{ss}V_{DO}$ = reiner_rubinstein_down_and_out_put_gamma (closed form). "
    r"Dotted verticals: $s=B$, $s=K$. Price row linear; Delta/Gamma rows symlog (linear below 0.1)."
)
FORMULA_ERRORS = (
    r"$e_0(s,t)=|\Phi_\theta(s,t)-V_{DO}(s,t)|$,  $e_1(s,t)=|\partial_s\Phi_\theta(s,t)-\partial_sV_{DO}(s,t)|$,  "
    r"$e_2(s,t)=|\partial_{ss}\Phi_\theta(s,t)-\partial_{ss}V_{DO}(s,t)|$, pointwise along $s$ at fixed $t$ "
    r"(log scale; values below $10^{-9}$ clipped)."
    "\n"
    r"Derivatives as in profiles_price_delta_gamma.png. Dotted verticals: $s=B$, $s=K$; grey band: "
    r"$\ell^1$ corner window $|s-B|+(T-t)\leq 0.1$ at that $t$ (empty when $T-t>0.1$)."
)
FORMULA_STRIKE = (
    r"Top: $\partial_s\Phi_\theta(K,t)$ and $\partial_{ss}\Phi_\theta(K,t)$ against $t$ (solid), exact "
    r"$\partial_sV_{DO}(K,t)$, $\partial_{ss}V_{DO}(K,t)$ (dashed).  Bottom: "
    r"$\mathrm{err}_{\mathrm{rel}}(t)=|\partial^{k}_s\Phi_\theta(K,t)-\partial^{k}_sV_{DO}(K,t)|\,/\,|\partial^{k}_sV_{DO}(K,t)|$, "
    r"$k=1$ (Delta), $k=2$ (Gamma)."
    "\n"
    r"The exact Gamma changes sign near $t\approx0.3$ (grey vertical): the relative Gamma error is "
    r"ill-conditioned there (small denominator), not the numerical Gamma."
)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _nested_autograd(function, s: torch.Tensor, t: torch.Tensor):
    """``(value, d/ds, d2/ds2)`` of ``function(s, t)`` pointwise, for a vector of
    ``s`` (each output depends on its own ``s`` only, so the summed gradients
    are the pointwise derivatives)."""
    with torch.enable_grad():
        s = s.detach().clone().requires_grad_(True)
        value = function(s, t)
        first = torch.autograd.grad(value.sum(), s, create_graph=True)[0]
        second = torch.autograd.grad(first.sum(), s)[0]
    return value.detach(), first.detach(), second.detach()


def trained_profiles(model, s: torch.Tensor, t: torch.Tensor):
    """``(Phi, d_s Phi, d_ss Phi)`` of the trained trial solution at the points
    ``(s, t)`` (same shape), the two parts differentiated separately."""
    def manifold(s_, t_):
        return model.forward_neural_manifold(torch.stack([s_, t_], dim=1)).squeeze(-1)

    value_m, first_m, second_m = _nested_autograd(manifold, s, t)
    g2 = model.g2
    if hasattr(g2, "second_price_derivative"):
        with torch.no_grad():
            value_e = g2(s, t)
            first_e = g2.first_price_derivative(s, t)
            second_e = g2.second_price_derivative(s, t)
    else:
        value_e, first_e, second_e = _nested_autograd(g2, s, t)
    return value_m + value_e, first_m + first_e, second_m + second_e


def reference_profiles(s: torch.Tensor, t: torch.Tensor, K, B, r, sigma, T):
    """``(V_DO, d_s V_DO, d_ss V_DO)`` from the closed forms (Delta by autograd of the closed form)."""
    with torch.enable_grad():
        s_ = s.detach().clone().requires_grad_(True)
        value = reiner_rubinstein_down_and_out_put(s_, K, B, r, sigma, T - t)
        first = torch.autograd.grad(value.sum(), s_)[0]
    second = reiner_rubinstein_down_and_out_put_gamma(s, K, B, r, sigma, T - t)
    return value.detach(), first.detach(), second


def evaluate_curves(runs: dict[str, Path], times: list[float], strike_times: list[float],
                    n_s: int, s_max: float, zoom_width: float, zoom_n_s: int) -> dict:
    """All curves of all configurations; contract read from the first run. The
    corner zoom is evaluated on its own dense grid ``(B, B + zoom_width)``."""
    first_meta = read_run_metadata(next(iter(runs.values())))
    K, B, r, sigma, T = (first_meta["contract"][k] for k in ("K", "B", "r", "sigma", "T"))
    s_grid = torch.linspace(B + 1e-4, s_max, n_s, dtype=torch.float64)
    s_zoom = torch.linspace(B + 1e-5, B + zoom_width, zoom_n_s, dtype=torch.float64)
    curves: dict = {
        "contract": {"K": K, "B": B, "r": r, "sigma": sigma, "T": T},
        "s_grid": s_grid, "s_zoom": s_zoom, "times": times, "strike_times": strike_times,
        "reference": {}, "reference_zoom": {}, "configurations": {},
    }
    for t_value in times:
        t = torch.full_like(s_grid, t_value)
        curves["reference"][t_value] = torch.stack(reference_profiles(s_grid, t, K, B, r, sigma, T))
        t_zoom = torch.full_like(s_zoom, t_value)
        curves["reference_zoom"][t_value] = torch.stack(reference_profiles(s_zoom, t_zoom, K, B, r, sigma, T))
    strike_t = torch.tensor(strike_times, dtype=torch.float64)
    strike_s = torch.full_like(strike_t, K)
    curves["reference"]["strike"] = torch.stack(reference_profiles(strike_s, strike_t, K, B, r, sigma, T))

    for configuration, run_dir in runs.items():
        meta = read_run_metadata(run_dir)
        if meta["contract"] != curves["contract"]:
            raise SystemExit(f"{configuration}: contract {meta['contract']} differs from {curves['contract']}")
        with open(run_dir / f"summary_eps{_run_epsilon(run_dir):g}.yaml") as f:
            epsilon = yaml.safe_load(f)["epsilon"]
        model = load_trained_model(run_dir, epsilon, meta)
        entry = {"run_dir": str(run_dir), "zoom": {}}
        for t_value in times:
            t = torch.full_like(s_grid, t_value)
            entry[t_value] = torch.stack(trained_profiles(model, s_grid.to(DEVICE), t.to(DEVICE))).cpu()
            t_zoom = torch.full_like(s_zoom, t_value)
            entry["zoom"][t_value] = torch.stack(trained_profiles(model, s_zoom.to(DEVICE), t_zoom.to(DEVICE))).cpu()
        entry["strike"] = torch.stack(trained_profiles(model, strike_s.to(DEVICE), strike_t.to(DEVICE))).cpu()
        curves["configurations"][configuration] = entry
        logger.info(f"  {configuration:<26s} {run_dir.name}: evaluated at {len(times)} times x {n_s} prices "
                    f"+ {len(strike_times)} strike times")
    return curves


def _run_epsilon(run_dir: Path) -> float:
    import re
    match = re.search(r"_eps([0-9.]+)_seed", run_dir.name)
    return float(match.group(1)) if match else 0.0


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

ROW_LABELS = [r"$\Phi_\theta(s,t)$ (price)", r"$\partial_s\Phi_\theta(s,t)$ (Delta)", r"$\partial_{ss}\Phi_\theta(s,t)$ (Gamma)"]


def plot_profiles(curves: dict, path: Path, s_plot_max: float, zoom: bool = False) -> None:
    """Rows Phi, d_s Phi, d_ss Phi; columns t. ``zoom`` uses the dense corner
    grid ``s_zoom`` and its own reference (the whole grid is plotted)."""
    times = curves["times"]
    s = (curves["s_zoom"] if zoom else curves["s_grid"]).numpy()
    keep = np.ones_like(s, dtype=bool) if zoom else s <= s_plot_max
    K, B, T = curves["contract"]["K"], curves["contract"]["B"], curves["contract"]["T"]
    fig, axes = plt.subplots(3, len(times), figsize=(4.4 * len(times), 11), squeeze=False)
    handles = []
    for j, t_value in enumerate(times):
        reference = (curves["reference_zoom"] if zoom else curves["reference"])[t_value].numpy()
        for i in range(3):
            ax = axes[i, j]
            for configuration, entry in curves["configurations"].items():
                trained = (entry["zoom"][t_value] if zoom else entry[t_value]).numpy()
                (line,) = ax.plot(s[keep], trained[i][keep], lw=1.5,
                                  color=COLOURS.get(configuration, None),
                                  label=CONFIGURATION_LABELS[configuration].replace("\n", " "))
                if i == 0 and j == 0:
                    handles.append(line)
            (ref_line,) = ax.plot(s[keep], reference[i][keep], "k--", lw=1.8, label=r"$V_{DO}$ (Reiner-Rubinstein, exact)")
            if i == 0 and j == 0:
                handles.insert(0, ref_line)
            for x in (B, K):
                ax.axvline(x, color="grey", linestyle=":", lw=1)
            if zoom:
                # Diffusion length of the corner layer at this t: B sigma sqrt(2 tau).
                layer = B * curves["contract"]["sigma"] * np.sqrt(2.0 * max(T - t_value, 0.0))
                ax.axvline(B + layer, color="tab:grey", linestyle="--", lw=0.8)
                ax.set_xlim(s[0], s[-1])
            if i > 0:
                ax.set_yscale("symlog", linthresh=0.1)
            ax.axhline(0, color="grey", lw=0.6)
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(f"$t = {t_value:g}$")
            if i == 2:
                ax.set_xlabel("Underlying price $s$")
            if j == 0:
                ax.set_ylabel(ROW_LABELS[i])
    legend = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.095), ncol=4, fontsize=8)
    title = ("Down-and-out put — price, Delta and Gamma profiles near the corner (one seed)" if zoom
             else "Down-and-out put — price, Delta and Gamma profiles of the corner treatments (one seed)")
    formula = FORMULA_PROFILES + (
        "\n" + r"Zoom on the corner region; dashed grey vertical: $s = B + B\sigma\sqrt{2(T-t)}$, "
        "the diffusion length of the corner layer at that $t$." if zoom else "")
    fig.suptitle(title, fontsize=11)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=0.21, wspace=0.25, hspace=0.3)
    finalize_figure(fig, path, legends=[legend], formula=formula, axes=list(axes.reshape(-1)), formula_fontsize=7)


def plot_absolute_errors(curves: dict, path: Path, s_plot_max: float, window: float = 0.1) -> None:
    times = curves["times"]
    s = curves["s_grid"].numpy()
    keep = s <= s_plot_max
    K, B, T = curves["contract"]["K"], curves["contract"]["B"], curves["contract"]["T"]
    labels = [r"$e_0=|\Phi_\theta-V_{DO}|$", r"$e_1=|\partial_s\Phi_\theta-\partial_sV_{DO}|$",
              r"$e_2=|\partial_{ss}\Phi_\theta-\partial_{ss}V_{DO}|$"]
    fig, axes = plt.subplots(3, len(times), figsize=(4.4 * len(times), 11), squeeze=False)
    handles = []
    for j, t_value in enumerate(times):
        reference = curves["reference"][t_value].numpy()
        for i in range(3):
            ax = axes[i, j]
            for configuration, entry in curves["configurations"].items():
                error = np.abs(entry[t_value].numpy()[i] - reference[i])
                (line,) = ax.semilogy(s[keep], np.maximum(error[keep], 1e-9), lw=1.4,
                                      color=COLOURS.get(configuration, None),
                                      label=CONFIGURATION_LABELS[configuration].replace("\n", " "))
                if i == 0 and j == 0:
                    handles.append(line)
            half_width = window - (T - t_value)
            if half_width > 0:
                ax.axvspan(B, B + half_width, color="grey", alpha=0.15, lw=0)
            for x in (B, K):
                ax.axvline(x, color="grey", linestyle=":", lw=1)
            ax.grid(alpha=0.3, which="both")
            if i == 0:
                ax.set_title(f"$t = {t_value:g}$")
            if i == 2:
                ax.set_xlabel("Underlying price $s$")
            if j == 0:
                ax.set_ylabel(labels[i])
    legend = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.095), ncol=4, fontsize=8)
    fig.suptitle("Down-and-out put — where the error of each corner treatment is (absolute, pointwise, one seed)", fontsize=11)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=0.21, wspace=0.25, hspace=0.3)
    finalize_figure(fig, path, legends=[legend], formula=FORMULA_ERRORS, axes=list(axes.reshape(-1)), formula_fontsize=7)


def plot_greeks_at_strike(curves: dict, path: Path) -> None:
    t = np.array(curves["strike_times"])
    reference = curves["reference"]["strike"].numpy()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    handles = []
    sign_change = np.where(np.diff(np.sign(reference[2])) != 0)[0]
    for column, k in enumerate((1, 2)):
        ax_value, ax_error = axes[0, column], axes[1, column]
        for configuration, entry in curves["configurations"].items():
            numerical = entry["strike"].numpy()[k]
            (line,) = ax_value.plot(t, numerical, lw=1.5, color=COLOURS.get(configuration, None),
                                    label=CONFIGURATION_LABELS[configuration].replace("\n", " "))
            if column == 0:
                handles.append(line)
            ax_error.semilogy(t, np.abs(numerical - reference[k]) / np.abs(reference[k]), lw=1.5,
                              color=COLOURS.get(configuration, None))
        (ref_line,) = ax_value.plot(t, reference[k], "k--", lw=1.8, label="exact (closed form)")
        if column == 0:
            handles.insert(0, ref_line)
        for idx in sign_change:
            for ax in (ax_value, ax_error):
                ax.axvline(t[idx], color="grey", lw=0.8)
        ax_value.set_ylabel(r"$\partial_s\Phi_\theta(K,t)$" if k == 1 else r"$\partial_{ss}\Phi_\theta(K,t)$")
        ax_error.set_ylabel(r"$\mathrm{err}_{\mathrm{rel}}\,\Delta(K,t)$" if k == 1 else r"$\mathrm{err}_{\mathrm{rel}}\,\Gamma(K,t)$")
        ax_error.set_xlabel("Calendar time $t$")
        ax_value.set_title("Delta at the strike" if k == 1 else "Gamma at the strike")
        for ax in (ax_value, ax_error):
            ax.grid(alpha=0.3, which="both")
    legend = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 0.95), fontsize=8)
    fig.suptitle("Down-and-out put — Delta and Gamma at the strike against calendar time (one seed)", fontsize=11)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.92, bottom=0.2, wspace=0.3, hspace=0.3)
    finalize_figure(fig, path, legends=[legend], formula=FORMULA_STRIKE, axes=list(axes.reshape(-1)), formula_fontsize=7)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aggregation-dir", type=str, default=None,
                        help="Aggregation directory whose summary.yaml lists the runs (one per configuration and seed).")
    parser.add_argument("--configurations", nargs="+", type=str, default=DEFAULT_CONFIGURATIONS,
                        help="Configuration keys to compare (see aggregate_terminal_function_comparison.CONFIGURATION_LABELS).")
    parser.add_argument("--seed", type=int, default=0, help="Master seed of the runs to load.")
    parser.add_argument("--times", nargs="+", type=float, default=DEFAULT_TIMES, help="Calendar times of the s-profiles.")
    parser.add_argument("--n-s", type=int, default=800, help="Number of s points of the profiles.")
    parser.add_argument("--s-max", type=float, default=3.0, help="Upper end of the evaluated s range.")
    parser.add_argument("--s-plot-max", type=float, default=2.0, help="Upper end of the PLOTTED s range.")
    parser.add_argument("--zoom-width", type=float, default=0.3, help="Width s - B of the corner zoom.")
    parser.add_argument("--zoom-n-s", type=int, default=600, help="Number of s points of the corner zoom grid.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory override.")
    parser.add_argument("--replot", type=str, default=None, metavar="OUT_DIR",
                        help="Rebuild the figures from a previous run's curves.pt, no evaluation.")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    if args.replot:
        out_dir = Path(args.replot)
        curves = torch.load(out_dir / "curves.pt", weights_only=False)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
        logger.info(f"--replot from {out_dir / 'curves.pt'}")
    else:
        if args.aggregation_dir is None:
            parser.error("--aggregation-dir is required unless --replot is given.")
        out_dir = Path(args.out_dir) if args.out_dir else script_data_dir(__file__) / (
            f"{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}_seed{args.seed}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
                            handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "compare.log")])
        logger.info(f"Command: {' '.join(sys.argv)}")
        logger.info(f"PyTorch {torch.__version__}, device {DEVICE}, evaluation dtype float64")
        with open(Path(args.aggregation_dir) / "summary.yaml") as f:
            summary = yaml.safe_load(f)
        repo_root = find_repo_root(Path(__file__).resolve())
        runs: dict[str, Path] = {}
        for configuration in args.configurations:
            entry = summary["configurations"].get(configuration)
            if entry is None:
                logger.warning(f"  {configuration}: not in the aggregation; skipped")
                continue
            run_dir_text = entry["runs"].get(args.seed) or entry["runs"].get(str(args.seed))
            if run_dir_text is None:
                logger.warning(f"  {configuration}: no run for seed {args.seed}; skipped")
                continue
            run_dir = Path(run_dir_text)
            runs[configuration] = run_dir if run_dir.is_absolute() else repo_root / run_dir
        curves = evaluate_curves(runs, args.times, DEFAULT_STRIKE_TIMES, args.n_s, args.s_max,
                                 args.zoom_width, args.zoom_n_s)
        torch.save(curves, out_dir / "curves.pt")
        with open(out_dir / "runs.yaml", "w") as f:
            yaml.dump({"command": " ".join(sys.argv), "seed": args.seed,
                       "runs": {c: str(p) for c, p in runs.items()}, "contract": curves["contract"]}, f)
        logger.info(f"Curves saved -> {out_dir / 'curves.pt'}")

    (out_dir / "figures").mkdir(exist_ok=True)
    plot_profiles(curves, out_dir / "figures" / "profiles_price_delta_gamma.png", args.s_plot_max)
    if "s_zoom" in curves:
        plot_profiles(curves, out_dir / "figures" / "profiles_price_delta_gamma_corner_zoom.png", args.s_plot_max, zoom=True)
    else:
        logger.warning("curves.pt predates the corner zoom; re-run without --replot to produce it.")
    plot_absolute_errors(curves, out_dir / "figures" / "absolute_errors_along_s.png", args.s_plot_max)
    plot_greeks_at_strike(curves, out_dir / "figures" / "greeks_at_strike_vs_time.png")
    logger.info(f"Figures -> {out_dir / 'figures'}")


if __name__ == "__main__":
    main()
