r"""Cross-seed / cross-IC aggregation for the ansatz-form ablation.

Scans the per-(IC, seed) ablation directories produced by
``ablation_ansatz_forms.py``, groups them by initial condition and method
variant, and produces summary figures with seed variance:

* ``rel_l2_by_ic.png`` -- grouped bar chart of the relative L2 error against the
  exact solution, one group per IC, error bars = std over seeds (log scale);
* ``terminal_mismatch.png`` -- terminal-condition L2 mismatch at t = T per
  variant/IC (hard forms are exactly zero; the soft PINN is small; the pure-NN
  control fails);
* ``floor_vs_accuracy.png`` -- the theta-independent extension-forcing floor
  E[(P Psi)^2] (median over training) against the achieved rel L2, for the hard
  forms only, illustrating that the monitored floor channel predicts the
  hard-form ranking.

A combined ``summary_across_seeds.yaml`` (mean +/- std per IC/variant) is also
written. Torch-free; reads only the saved .npz / .yaml artefacts.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

import _ansatz_forms_catalogue as cat  # noqa: E402

_RUN_RE = re.compile(r"Z_(?P<ic>[a-z0-9]+)_iters\d+$")


def discover_runs(data_root: Path) -> dict:
    """Return ``{ic: {variant: {metric: [values over seeds]}}}`` plus floors."""
    agg: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    run_dirs = sorted(
        d for d in glob.glob(str(data_root / "*_iters*"))
        if os.path.isdir(d) and "_debug_" not in os.path.basename(d)
    )
    for d in run_dirs:
        m = _RUN_RE.search(os.path.basename(d))
        if m is None:
            continue
        ic = m.group("ic")
        summ_path = os.path.join(d, "summary.yaml")
        if not os.path.exists(summ_path):
            continue
        summ = yaml.safe_load(open(summ_path)) or {}
        for v in cat.variant_names():
            if v not in summ:
                continue
            for key in ("rel_l2", "rel_l2_t0", "tc_l2"):
                agg[ic][v][key].append(float(summ[v][key]))
            hist_path = os.path.join(d, f"variant_{v}", "hist.npz")
            if os.path.exists(hist_path):
                h = np.load(hist_path)
                if "forcing_floor" in h:
                    agg[ic][v]["forcing_floor"].append(float(np.median(h["forcing_floor"])))
    return agg, run_dirs


def _mean_std(vals):
    a = np.asarray(vals, dtype=float)
    return (float(a.mean()), float(a.std())) if a.size else (float("nan"), 0.0)


def _save(fig, path):
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_rel_l2_by_ic(agg, out_dir):
    ics = [ic for ic in cat.ic_names() if ic in agg]
    variants = cat.variant_names()
    colors = {v["name"]: v["color"] for v in cat.METHOD_VARIANTS}
    labels = {v["name"]: v["label"] for v in cat.METHOD_VARIANTS}
    fig, ax = plt.subplots(figsize=(11, 5.5))
    n_v = len(variants)
    width = 0.8 / n_v
    x = np.arange(len(ics))
    for j, v in enumerate(variants):
        means, stds = [], []
        for ic in ics:
            mu, sd = _mean_std(agg[ic][v].get("rel_l2", []))
            means.append(mu)
            stds.append(sd)
        ax.bar(x + j * width, np.clip(means, 1e-30, None), width,
               yerr=stds, color=colors[v], label=labels[v], capsize=2)
    ax.set_yscale("log")
    ax.set_xticks(x + 0.4 - width / 2)
    ax.set_xticklabels(ics)
    ax.set_ylabel(r"relative $L^2$ error vs exact (lower is better)")
    ax.set_title("Accuracy per ansatz form and initial condition (mean ± std over seeds)")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7, frameon=True)
    fig.tight_layout(rect=[0, 0, 0.80, 1])
    _save(fig, out_dir / "rel_l2_by_ic.png")


def plot_terminal_mismatch(agg, out_dir):
    ics = [ic for ic in cat.ic_names() if ic in agg]
    variants = cat.variant_names()
    colors = {v["name"]: v["color"] for v in cat.METHOD_VARIANTS}
    labels = {v["name"]: v["label"] for v in cat.METHOD_VARIANTS}
    fig, ax = plt.subplots(figsize=(11, 5.5))
    n_v = len(variants)
    width = 0.8 / n_v
    x = np.arange(len(ics))
    for j, v in enumerate(variants):
        means = [_mean_std(agg[ic][v].get("tc_l2", []))[0] for ic in ics]
        # floor tiny positive so log-scale shows the exact-zero hard forms
        ax.bar(x + j * width, np.clip(means, 1e-6, None), width,
               color=colors[v], label=labels[v])
    ax.set_yscale("log")
    ax.set_xticks(x + 0.4 - width / 2)
    ax.set_xticklabels(ics)
    ax.set_ylabel(r"terminal mismatch $\|\hat u(\cdot,T)-g\|/\|g\|$ (floored at $10^{-6}$)")
    ax.set_title("Terminal-condition enforcement (hard forms are exactly 0)")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7, frameon=True)
    fig.tight_layout(rect=[0, 0, 0.80, 1])
    _save(fig, out_dir / "terminal_mismatch.png")


def plot_floor_vs_accuracy(agg, out_dir):
    hard = [v["name"] for v in cat.METHOD_VARIANTS if v["form"].startswith("hard")]
    markers = {"sine": "o", "theta3": "s", "call": "^"}
    colors = {v["name"]: v["color"] for v in cat.METHOD_VARIANTS}
    fig, ax = plt.subplots(figsize=(8, 6))
    for ic in cat.ic_names():
        if ic not in agg:
            continue
        for v in hard:
            floor, _ = _mean_std(agg[ic][v].get("forcing_floor", []))
            rl, _ = _mean_std(agg[ic][v].get("rel_l2", []))
            if not (np.isfinite(floor) and np.isfinite(rl)):
                continue
            ax.scatter(floor, rl, c=colors[v], marker=markers.get(ic, "o"),
                       s=90, edgecolors="black", linewidths=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"extension-forcing floor $\mathbb{E}[(\mathcal{P}\Psi)^2]$ (median over training)")
    ax.set_ylabel(r"relative $L^2$ error vs exact")
    ax.set_title("Forcing floor predicts hard-form accuracy")
    ax.grid(True, which="both", alpha=0.3)
    # legends: colour = variant, marker = IC
    from matplotlib.lines import Line2D
    var_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[v],
                          markeredgecolor="black", markersize=9,
                          label=next(d["label"] for d in cat.METHOD_VARIANTS if d["name"] == v))
                   for v in hard]
    ic_handles = [Line2D([0], [0], marker=markers[ic], color="w", markerfacecolor="grey",
                         markeredgecolor="black", markersize=9, label=ic)
                  for ic in markers if ic in agg]
    leg1 = ax.legend(handles=var_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                     fontsize=7, frameon=True, title="form")
    ax.add_artist(leg1)
    ax.legend(handles=ic_handles, loc="lower left", bbox_to_anchor=(1.02, 0.0),
              fontsize=7, frameon=True, title="IC")
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    _save(fig, out_dir / "floor_vs_accuracy.png")


def write_summary(agg, path):
    out: dict = {}
    for ic in agg:
        out[ic] = {}
        for v in agg[ic]:
            entry = {}
            for key in ("rel_l2", "rel_l2_t0", "tc_l2", "forcing_floor"):
                mu, sd = _mean_std(agg[ic][v].get(key, []))
                entry[f"{key}_mean"] = mu
                entry[f"{key}_std"] = sd
            entry["n_seeds"] = len(agg[ic][v].get("rel_l2", []))
            out[ic][v] = entry
    with open(path, "w") as f:
        yaml.dump(out, f, default_flow_style=False, sort_keys=False, width=float("inf"))


def main(argv=None) -> int:
    from learning_option_pricing.utils.run_context import script_data_dir

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=str, default=None,
                        help="Folder holding the per-(IC,seed) ablation dirs "
                             "(default: data/ablation_ansatz_forms).")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Where to write the aggregate figures "
                             "(default: <data-root>/aggregate_seeds).")
    args = parser.parse_args(argv)

    data_root = Path(args.data_root) if args.data_root else (
        script_data_dir(Path(__file__).parent / "ablation_ansatz_forms.py"))
    out_dir = Path(args.out_dir) if args.out_dir else data_root / "aggregate_seeds"
    out_dir.mkdir(parents=True, exist_ok=True)

    agg, run_dirs = discover_runs(data_root)
    if not agg:
        raise SystemExit(f"No (non-debug) ablation runs found under {data_root}")
    print(f"Aggregated {len(run_dirs)} run dirs over ICs {sorted(agg)} into {out_dir}")

    plot_rel_l2_by_ic(agg, out_dir)
    plot_terminal_mismatch(agg, out_dir)
    plot_floor_vs_accuracy(agg, out_dir)
    write_summary(agg, out_dir / "summary_across_seeds.yaml")
    print("wrote: rel_l2_by_ic.png, terminal_mismatch.png, floor_vs_accuracy.png, "
          "summary_across_seeds.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
