r"""Selection-report figure "erreur_vs_temps" aggregated across master seeds.

``select_terminal_function_at_strike.py`` draws, for one master seed, the relative
L^2 error over s (slice by slice in calendar time, on the measurement domain
s > B + epsilon_corner) of the price, the Delta and the Gamma of every
construction of the terminal extension g_2, and saves the curves to
``erreur_vs_temps.yaml``. This script reads those files for several seeds
(one sub-directory per seed, ``seed<k>/erreur_vs_temps.yaml``) and draws

- ``erreur_vs_temps_median_over_seeds.png``: the report's three-panel figure
  with the across-seed median as the solid line and every seed as a faint
  curve, one colour per construction;
- ``erreur_vs_temps_per_construction.png``: one row per construction, the
  three quantities in columns, the five seeds overlaid, so that the
  across-seed dispersion of each construction can be read directly.

Reads saved curves only: no model evaluation, no training.

Usage:
    python3 experiments/python_scripts/exp_barrier_option/diagnostic_scripts/\
plot_selection_error_vs_time_across_seeds.py \
        --selection-dir data/select_terminal_function_at_strike/20260914_50k_nocorner_republique
"""
from __future__ import annotations

import argparse
import logging
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from learning_option_pricing.utils.figure_layout import finalize_figure  # noqa: E402

logger = logging.getLogger("plot_selection_error_vs_time_across_seeds")

QUANTITY_TITLE = {
    "price": r"Prix, $\|\Phi_\theta-V_{DO}\|_{\ell^2}/\|V_{DO}\|_{\ell^2}$",
    "delta": r"Delta, $\|\partial_s\Phi_\theta-\partial_sV_{DO}\|_{\ell^2}/\|\partial_sV_{DO}\|_{\ell^2}$",
    "gamma": r"Gamma, $\|\partial_{ss}\Phi_\theta-\partial_{ss}V_{DO}\|_{\ell^2}/\|\partial_{ss}V_{DO}\|_{\ell^2}$",
}
CONSTRUCTION_LABEL = {
    "black_scholes": r"Black-Scholes $V^e$, assemblage ordinaire",
    "black_scholes_two_term": r"Black-Scholes $V^e$, assemblage à deux termes",
    "split": r"Split-semigroupe $\pi$, assemblage à deux termes",
    "raw": r"Payoff brut $(K-s)^+$",
}
CONSTRUCTION_COLOR = {"black_scholes": "tab:blue", "black_scholes_two_term": "tab:purple", "split": "tab:red", "raw": "black"}
FORMULA_TEXT = (
    r"Erreur relative $\ell^2$ sur les noeuds $s_i>B+\varepsilon_{\mathrm{coin}}$ d'une tranche $t_j$ "
    r"(grille $s\in[B,2]$, $400$ noeuds ; $t\in[0,T]$, $200$ noeuds), courbes de select_terminal_function_at_strike.py."
    "\n"
    r"$\Phi_\theta=g_1u_\theta+g_2$ ; références : $V_{DO}$ (Reiner-Rubinstein), $\partial_sV_{DO}$ (autograd de la forme close), "
    r"$\partial_{ss}V_{DO}$ (forme close). Panneau Gamma tronqué à $t\leq T-0.01$."
    "\n"
    "Trait plein : médiane sur les graines ; traits fins : graines individuelles."
)


def read_series(selection_dir: Path) -> tuple[np.ndarray, dict[str, dict[str, dict[int, np.ndarray]]], float]:
    """``(t_grid, series[construction][quantity][seed] -> array, terminal_margin)``."""
    series: dict = {}
    t_grid = None
    for seed_dir in sorted(p for p in selection_dir.glob("seed*") if p.is_dir()):
        path = seed_dir / "erreur_vs_temps.yaml"
        if not path.exists():
            logger.warning(f"  {seed_dir.name}: no erreur_vs_temps.yaml, skipped")
            continue
        seed = int(seed_dir.name.replace("seed", ""))
        with open(path) as f:
            data = yaml.safe_load(f)
        t_here = np.asarray(data.pop("t_grid"))
        if t_grid is None:
            t_grid = t_here
        elif not np.allclose(t_grid, t_here):
            raise ValueError(f"{seed_dir.name}: t grid differs from the first seed's")
        for construction, quantities in data.items():
            for quantity, values in quantities.items():
                series.setdefault(construction, {}).setdefault(quantity, {})[seed] = np.asarray(values, dtype=float)
        logger.info(f"  {seed_dir.name}: constructions {sorted(data)}")
    if t_grid is None:
        raise SystemExit("no seed directory with erreur_vs_temps.yaml found")
    return t_grid, series, 0.01


def _masked(t_grid: np.ndarray, quantity: str, T: float, margin: float) -> np.ndarray:
    return t_grid <= T - margin if quantity == "gamma" else t_grid < T


def plot_median_over_seeds(t_grid, series, out_path: Path, T: float, margin: float) -> None:
    constructions = [c for c in CONSTRUCTION_LABEL if c in series]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))
    handles = []
    for ax, quantity in zip(axes, ("price", "delta", "gamma")):
        mask = _masked(t_grid, quantity, T, margin)
        for construction in constructions:
            per_seed = series[construction][quantity]
            color = CONSTRUCTION_COLOR.get(construction)
            for values in per_seed.values():
                ax.plot(t_grid[mask], values[mask], "-", color=color, alpha=0.25, linewidth=0.8)
            stacked = np.stack([per_seed[s] for s in sorted(per_seed)])
            median = np.median(stacked, axis=0)
            (line,) = ax.plot(t_grid[mask], median[mask], "-", color=color, linewidth=2.0,
                              label=f"{CONSTRUCTION_LABEL[construction]} (médiane, $n={len(per_seed)}$)")
            if ax is axes[0]:
                handles.append(line)
        ax.set_yscale("log")
        ax.set_xlabel("Temps calendaire $t$")
        ax.set_title(QUANTITY_TITLE[quantity], fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel(r"Erreur relative $\ell^2$ sur $s$ (échelle log)")
    legend = fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.2), ncol=1, fontsize=8,
                        title="Construction de $g_2$ (assemblage du résidu)")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.42, wspace=0.25)
    finalize_figure(fig, out_path, legends=[legend], formula=FORMULA_TEXT, axes=list(axes), formula_fontsize=6.5)


def plot_per_construction(t_grid, series, out_path: Path, T: float, margin: float) -> None:
    constructions = [c for c in CONSTRUCTION_LABEL if c in series]
    fig, axes = plt.subplots(len(constructions), 3, figsize=(15.5, 3.4 * len(constructions) + 1.2), sharex=True)
    axes = np.atleast_2d(axes)
    seed_colors = plt.cm.viridis(np.linspace(0.1, 0.9, 5))
    handles = {}
    for row, construction in enumerate(constructions):
        for col, quantity in enumerate(("price", "delta", "gamma")):
            ax = axes[row, col]
            mask = _masked(t_grid, quantity, T, margin)
            per_seed = series[construction][quantity]
            for seed in sorted(per_seed):
                (line,) = ax.plot(t_grid[mask], per_seed[seed][mask], "-", color=seed_colors[seed % 5],
                                  linewidth=1.0, label=f"graine {seed}")
                handles[seed] = line
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3)
            if row == 0:
                ax.set_title(QUANTITY_TITLE[quantity], fontsize=9)
            if col == 0:
                ax.set_ylabel(CONSTRUCTION_LABEL[construction] + "\n" + r"erreur relative $\ell^2$", fontsize=8)
            if row == len(constructions) - 1:
                ax.set_xlabel("Temps calendaire $t$")
    # Same y-limits per column so the constructions are comparable by eye.
    for col in range(3):
        lo = min(axes[r, col].get_ylim()[0] for r in range(len(constructions)))
        hi = max(axes[r, col].get_ylim()[1] for r in range(len(constructions)))
        for r in range(len(constructions)):
            axes[r, col].set_ylim(lo, hi)
    legend = fig.legend(handles=[handles[s] for s in sorted(handles)], loc="center", bbox_to_anchor=(0.5, 0.06),
                        ncol=len(handles), fontsize=8, title="Graine maîtresse")
    fig.subplots_adjust(left=0.09, right=0.99, top=0.95, bottom=0.13, hspace=0.15, wspace=0.25)
    finalize_figure(fig, out_path, legends=[legend], formula=FORMULA_TEXT.replace(
        "Trait plein : médiane sur les graines ; traits fins : graines individuelles.",
        "Une ligne par construction ; les cinq graines superposées ; mêmes ordonnées par colonne."),
        axes=list(axes.ravel()), formula_fontsize=6.5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selection-dir", type=str, required=True,
                        help="Directory holding one seed<k>/ sub-directory per master seed, each written by "
                             "select_terminal_function_at_strike.py --out-dir.")
    parser.add_argument("--maturity", type=float, default=1.0, help="T of the contract (for the Gamma margin).")
    parser.add_argument("--out-dir", type=str, default=None, help="Default: <selection-dir>/across_seeds.")
    args = parser.parse_args()
    selection_dir = Path(args.selection_dir)
    out_dir = Path(args.out_dir) if args.out_dir else selection_dir / "across_seeds"
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "across_seeds.log")])
    logger.info(f"  Command: {' '.join(sys.argv)}")
    t_grid, series, margin = read_series(selection_dir)
    summary = {}
    for construction, quantities in series.items():
        summary[construction] = {}
        for quantity, per_seed in quantities.items():
            mask = _masked(t_grid, quantity, args.maturity, margin)
            stacked = np.stack([per_seed[s][mask] for s in sorted(per_seed)])
            summary[construction][quantity] = {
                "n_seeds": len(per_seed),
                "time_mean_of_median_curve": float(np.median(stacked, axis=0).mean()),
                "time_mean_per_seed": {int(s): float(per_seed[s][mask].mean()) for s in sorted(per_seed)},
            }
            logger.info(f"  {construction:<24s} {quantity:<6s} time-mean of the median curve "
                        f"{summary[construction][quantity]['time_mean_of_median_curve']:.4e}  per seed "
                        + ", ".join(f"{v:.3e}" for v in summary[construction][quantity]["time_mean_per_seed"].values()))
    with open(out_dir / "erreur_vs_temps_across_seeds.yaml", "w") as f:
        yaml.dump({"t_grid": t_grid.tolist(), "gamma_margin": margin, "summary": summary,
                   "series": {c: {q: {int(s): v.tolist() for s, v in ps.items()} for q, ps in qs.items()}
                              for c, qs in series.items()}}, f, default_flow_style=False, sort_keys=False)
    plot_median_over_seeds(t_grid, series, out_dir / "erreur_vs_temps_median_over_seeds.png", args.maturity, margin)
    plot_per_construction(t_grid, series, out_dir / "erreur_vs_temps_per_construction.png", args.maturity, margin)
    logger.info(f"  Saved -> {out_dir}")


if __name__ == "__main__":
    main()
