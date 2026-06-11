r"""Two-factor comparison: terminal-extension smoothing x ansatz mode.

Reads the already-trained ``call`` (softplus-smoothed payoff extension) and
``call_cm`` (Chen--Mangasarian-smoothed extension) ablation runs and isolates
two factors on the European-call terminal-value problem:

* **factor 1 -- terminal extension** :math:`g`:  softplus smoothing of the
  payoff (``call``) vs Chen--Mangasarian smoothing (``call_cm``).  This sets the
  theta-independent extension-forcing floor :math:`\mathbb{E}[(\mathcal{P}\Psi)^2]`:
  a rougher :math:`g` has larger :math:`g''`, hence a larger floor.
* **factor 2 -- ansatz mode**:  additive extension ``hard_constant``
  :math:`\hat u=(1-\lambda)\Phi+g` vs convex combination ``hard_convex``
  :math:`\hat u=(1-\lambda)\Phi+\lambda g` (the convex form damps the diffusion
  forcing by :math:`\lambda(t)\le 1`).  ``soft_pinn`` (terminal datum as a
  penalty, not enforced) is shown as the unconstrained reference.

The figure is a 1x2 grouped-bar chart (rel L2 over the space-time eval window;
rel L2 on the back-propagated t=0 inception slice), bars grouped by mode and
coloured by extension, error bars = std over seeds, log axis.  A machine-readable
``terminal_extension_vs_mode.yaml`` table (mean +/- std per cell) is also written.

Torch-free: reads only the saved ``summary.yaml`` artefacts; no retraining.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ansatz_forms_catalogue as cat  # noqa: E402
from _figure_layout import finalize_figure  # noqa: E402

from learning_option_pricing.utils.run_context import (  # noqa: E402
    script_data_dir,
    utc_timestamp,
)

# Terminal-extension factor: IC name -> (display, colour).  Solid bars (these
# are trained models); colour encodes the extension smoothing, per the repo
# convention that the swept axis goes in colour, not stroke.
EXTENSIONS = {
    "call": ("softplus extension", "#d1495b"),       # rougher g (sharp g'')
    "call_cm": ("Chen--Mangasarian extension", "#1b6ca8"),  # smoother g
}
# Ansatz-mode factor: the x-axis groups.  Linear interpolation coefficient is
# the canonical one; the exponential variants go into the table only.
MODES = [
    ("hard_constant_linear", "hard_constant\n$(1-\\lambda)\\Phi+g$"),
    ("hard_convex_linear", "hard_convex\n$(1-\\lambda)\\Phi+\\lambda g$"),
    ("soft_pinn", "soft_pinn\n(penalty)"),
]
# Variants tabulated (incl. exponential interpolation) but not all plotted.
TABLE_VARIANTS = [
    "hard_constant_linear", "hard_constant_exp",
    "hard_convex_linear", "hard_convex_exp", "soft_pinn",
]
METRICS = ("rel_l2", "rel_l2_t0", "tc_l2")


def _mean_std(vals):
    a = np.asarray(vals, dtype=float)
    return (float(a.mean()), float(a.std()), int(a.size)) if a.size else (float("nan"), 0.0, 0)


def gather(data_root: Path, ic: str) -> dict:
    """Return ``{variant: {metric: (mean, std, n_seeds)}}`` for one IC."""
    runs = sorted(glob.glob(str(data_root / f"*_{ic}_iters*_seed*")))
    runs = [d for d in runs if "_debug_" not in os.path.basename(d)]
    acc: dict = {v: {m: [] for m in METRICS} for v in TABLE_VARIANTS}
    for d in runs:
        sp = os.path.join(d, "summary.yaml")
        if not os.path.exists(sp):
            continue
        summ = yaml.safe_load(open(sp)) or {}
        for v in TABLE_VARIANTS:
            if v in summ:
                for m in METRICS:
                    acc[v][m].append(float(summ[v][m]))
    return {v: {m: _mean_std(acc[v][m]) for m in METRICS} for v in TABLE_VARIANTS}


def make_figure(data: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    panels = [("rel_l2", r"relative $L^2$ over $\mathcal{X}_{\rm eval}\times[0,T]$"),
              ("rel_l2_t0", r"relative $L^2$ on the $t=0$ inception slice")]
    n_ext = len(EXTENSIONS)
    width = 0.8 / n_ext
    x = np.arange(len(MODES))
    for ax, (metric, title) in zip(axes, panels):
        for j, (ic, (disp, color)) in enumerate(EXTENSIONS.items()):
            means, stds = [], []
            for vname, _ in MODES:
                mu, sd, _ = data[ic][vname][metric]
                means.append(mu)
                stds.append(sd)
            ax.bar(x + j * width, np.clip(means, 1e-30, None), width,
                   yerr=stds, color=color, label=disp, capsize=3,
                   edgecolor="black", linewidth=0.4)
        ax.set_yscale("log")
        ax.set_xticks(x + 0.4 - width / 2)
        ax.set_xticklabels([lbl for _, lbl in MODES], fontsize=8)
        ax.set_ylabel(title, fontsize=9)
        ax.grid(True, axis="y", which="both", alpha=0.3)
    axes[0].set_title("Accuracy over the full space-time window", fontsize=10)
    axes[1].set_title("Accuracy at the inception price ($t=0$)", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    # legend sits above the bottom formula box (which finalize_figure anchors at
    # y~0.012); keep them at distinct heights so they never overlap.
    leg = fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9,
                     frameon=True, bbox_to_anchor=(0.5, 0.10))
    fig.suptitle("Terminal extension $g$ (colour) vs ansatz mode (x-axis) on the "
                 "European call\nmean $\\pm$ std over 3 seeds; lower is better",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0.22, 1, 0.93])
    finalize_figure(
        fig, out_dir / "terminal_extension_vs_mode.png", legends=[leg],
        axes=list(axes),
        formula=(r"hard forms enforce $\hat u(\cdot,T)=g$ exactly (tc $=0$); "
                 r"soft_pinn penalises it (tc $\approx 7\times10^{-3}$). "
                 r"Dominant lever is the smoothness of $g$ (the floor "
                 r"$\mathbb{E}[(\mathcal{P}\Psi)^2]$), not the additive-vs-convex algebra."),
        dpi=140, formula_fontsize=8,
    )


def write_table(data: dict, out_dir: Path) -> None:
    table = {}
    for ic in EXTENSIONS:
        table[ic] = {}
        for v in TABLE_VARIANTS:
            table[ic][v] = {
                m: {"mean": data[ic][v][m][0], "std": data[ic][v][m][1],
                    "n_seeds": data[ic][v][m][2]}
                for m in METRICS
            }
    with open(out_dir / "terminal_extension_vs_mode.yaml", "w") as fh:
        yaml.dump(table, fh, default_flow_style=False, sort_keys=False)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=None,
                   help="dir holding the *_call*_iters*_seed* runs "
                        "(default: ablation_ansatz_forms data dir)")
    args = p.parse_args(argv)

    data_root = (Path(args.data_root) if args.data_root
                 else script_data_dir(Path(__file__).parent / "ablation_ansatz_forms.py"))
    out_dir = script_data_dir(__file__) / utc_timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = {ic: gather(data_root, ic) for ic in EXTENSIONS}
    make_figure(data, out_dir)
    write_table(data, out_dir)

    # console table
    print(f"\nwrote figure + table to {out_dir}\n")
    for ic, (disp, _) in EXTENSIONS.items():
        print(f"=== {ic}  ({disp}) ===")
        for v in TABLE_VARIANTS:
            d = data[ic][v]
            print(f"  {v:24s} relL2={d['rel_l2'][0]:.2e}+-{d['rel_l2'][1]:.0e}  "
                  f"relL2_t0={d['rel_l2_t0'][0]:.2e}+-{d['rel_l2_t0'][1]:.0e}  "
                  f"tc={d['tc_l2'][0]:.1e}  (n={d['rel_l2'][2]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
