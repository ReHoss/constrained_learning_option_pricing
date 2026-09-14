r"""Training budget against error floor: best interior loss and relative error at two
iteration budgets, per construction of g_2 and per master seed.

Reads two aggregation directories written by ``aggregate_terminal_function_comparison.py``
(``summary.yaml`` holds, per configuration and per seed, ``best_loss`` and
``rel_l2_outside_corner``) and draws one panel per metric: abscissa = iteration budget,
one colour per construction, faint points per seed, solid line through the across-seed
medians. Optionally overlays, on the error panel, the floor predicted from the
terminal-trace defect (``predicted_vs_measured.yaml`` of ``predict_corner_layer_floor.py``,
key ``predicted_floor["outside corner"]``) as a dashed horizontal line.

The figure shows whether a longer budget lowers the residual without lowering the error,
i.e. whether the error is a floor of the ansatz rather than a lack of training. Saved
artefacts only; no model evaluation.

Usage:
    python3 experiments/python_scripts/exp_barrier_option/diagnostic_scripts/plot_budget_floor.py \
        --aggregations data/aggregate_terminal_function_comparison/<20k dir> \
                       data/aggregate_terminal_function_comparison/<50k dir> \
        --predicted-floor data/predict_corner_layer_floor/<dir>/predicted_vs_measured.yaml
"""
from __future__ import annotations

import argparse
import logging
import statistics
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from learning_option_pricing.utils.figure_layout import finalize_figure  # noqa: E402
from learning_option_pricing.utils.run_context import script_data_dir  # noqa: E402

logger = logging.getLogger("plot_budget_floor")

LABEL = {"blackscholes": r"Black-Scholes $V^e$, assemblage ordinaire",
         "blackscholes_analyticres": r"Black-Scholes $V^e$, assemblage à deux termes",
         "split": r"Split-semigroupe $\pi$, assemblage à deux termes"}
COLOR = {"blackscholes": "tab:blue", "blackscholes_analyticres": "tab:purple", "split": "tab:red"}
FORMULA_TEXT = (
    r"Gauche : meilleur loss $\min_k\,\mathrm{mean}_{\mathrm{batch}_k}(\mathcal{L}^{BS}\Phi_\theta)^2$ sur les itérations. "
    r"Droite : $\|\Phi_\theta-V_{DO}\|_{L^2(\Omega\setminus N_{0.1})}/\|V_{DO}\|_{L^2(\Omega\setminus N_{0.1})}$, "
    r"$\Omega=(B,s_\infty)\times(0,T)$, $N_{0.1}$ la fenêtre du coin."
    "\n"
    r"Points : graines ; trait plein : médiane sur les graines ; tiretés : plancher prédit par le défaut de trace terminale "
    r"$-(1-\zeta)(K-s)^+$ (predict_corner_layer_floor.py), indépendant de $\theta$ et de la construction."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aggregations", nargs="+", type=str, required=True,
                        help="Aggregation directories (one per budget), each holding summary.yaml.")
    parser.add_argument("--predicted-floor", type=str, default=None,
                        help="predicted_vs_measured.yaml of predict_corner_layer_floor.py (dashed line).")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else script_data_dir(__file__) / datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "plot_budget_floor.log")])
    logger.info(f"  Command: {' '.join(sys.argv)}")

    budgets = []
    for path in args.aggregations:
        with open(Path(path) / "summary.yaml") as f:
            summary = yaml.safe_load(f)
        budgets.append((summary["iters"], summary["configurations"], path))
        logger.info(f"  {path}: iters={summary['iters']}, configurations={list(summary['configurations'])}")
    budgets.sort(key=lambda b: b[0])
    floor = None
    if args.predicted_floor:
        with open(args.predicted_floor) as f:
            floor = yaml.safe_load(f)["predicted_floor"]["outside corner"]
        logger.info(f"  predicted floor outside the corner window: {floor:.4e}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    handles = []
    summary_out = {}
    for ax, metric, title in zip(axes, ("best_loss", "rel_l2_outside_corner"),
                                 ("Meilleur loss intérieur", r"Erreur relative $L^2$ hors fenêtre du coin")):
        for configuration in LABEL:
            xs, medians = [], []
            for iters, configurations, _ in budgets:
                entry = configurations.get(configuration)
                if entry is None:
                    continue
                per_seed = [v for v in entry["metrics"][metric]["per_seed"].values() if v is not None]
                ax.scatter([iters] * len(per_seed), per_seed, s=18, alpha=0.35, color=COLOR[configuration], zorder=2)
                xs.append(iters); medians.append(statistics.median(per_seed))
                summary_out.setdefault(configuration, {}).setdefault(metric, {})[iters] = {
                    "median": medians[-1], "min": min(per_seed), "max": max(per_seed)}
            (line,) = ax.plot(xs, medians, "-o", color=COLOR[configuration], markeredgecolor="black", zorder=3,
                              label=LABEL[configuration])
            if ax is axes[0]:
                handles.append(line)
        if metric == "rel_l2_outside_corner" and floor is not None:
            (fl,) = ax.plot([budgets[0][0] * 0.9, budgets[-1][0] * 1.1], [floor, floor], "--", color="black",
                            label=f"Plancher prédit (défaut de trace terminale) = {floor:.3f}")
            handles.append(fl)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks([b[0] for b in budgets]); ax.set_xticklabels([f"{b[0]:,}".replace(",", " ") for b in budgets])
        ax.set_xlabel("Itérations d'entraînement"); ax.set_title(title, fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
    legend = fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.17), ncol=2, fontsize=8)
    fig.suptitle("Budget d'entraînement contre plancher d'erreur — down-and-out put, coin exclu, 5 graines par construction", fontsize=10)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.36, wspace=0.25)
    finalize_figure(fig, out_dir / "budget_vs_floor.png", legends=[legend], formula=FORMULA_TEXT, axes=list(axes), formula_fontsize=6.5)
    with open(out_dir / "budget_vs_floor.yaml", "w") as f:
        yaml.dump({"aggregations": args.aggregations, "predicted_floor_outside_corner": floor, "per_configuration": summary_out},
                  f, default_flow_style=False, sort_keys=False)
    for configuration, metrics in summary_out.items():
        for metric, per_budget in metrics.items():
            its = sorted(per_budget)
            ratio = per_budget[its[-1]]["median"] / per_budget[its[0]]["median"]
            logger.info(f"  {configuration:<26s} {metric:<22s} " + "  ".join(f"{i}: {per_budget[i]['median']:.3e}" for i in its) + f"  ratio {ratio:.2f}")
    logger.info(f"  Saved -> {out_dir / 'budget_vs_floor.png'}")


if __name__ == "__main__":
    main()
