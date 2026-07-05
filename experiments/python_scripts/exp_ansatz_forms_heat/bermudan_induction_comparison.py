r"""Cross-run comparison figures for the learned Bermudan backward induction.

Reads the ``validation.npz`` artefacts written by ``bermudan_backward_induction.py``
for several runs (different ``m`` / ansatz form) and produces:

* ``error_propagation_comparison.png`` -- relative L2 error against the exact
  multi-stage reference at each global exercise date, one curve per run; shows how
  the per-stage learning error propagates down the induction (right = maturity,
  left = inception) and contrasts the hard convex form with the soft PINN.
* ``inception_comparison.png`` -- the learned inception price V(.,0) for each run
  overlaid on the exact Bermudan value (dashed) and the European put / payoff,
  confirming the learned chains reproduce the price.

Torch-free; reads only the saved arrays. Pass run dirs as arguments, or rely on
the default glob over the ablation data directory.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _figure_layout import finalize_figure  # noqa: E402
from learning_option_pricing.utils.run_context import (  # noqa: E402
    script_data_dir,
    utc_timestamp,
)


def _load(run_dir: Path) -> dict:
    z = np.load(run_dir / "validation.npz")
    n_stages = sum(1 for k in z.files if k.endswith("_rel_l2"))
    stages = [{
        "k": k,
        "t_global": float(z[f"stage{k}_t_global"][0]),
        "v_net": z[f"stage{k}_v_net"],
        "v_exact": z[f"stage{k}_v_exact"],
        "rel_l2": float(z[f"stage{k}_rel_l2"][0]),
    } for k in range(n_stages)]
    # parse "m" and form from the run-dir name ..._m<M>_<form>_iters...
    name = run_dir.name
    m = int(name.split("_m")[1].split("_")[0])
    form = name.split(f"_m{m}_")[1].split("_iters")[0]
    return {"S": z["S"], "payoff": z["payoff"], "european0": z["european0"],
            "stages": stages, "m": m, "form": form, "name": name,
            "inception_rel_l2": stages[0]["rel_l2"]}


def _aggregate(runs, out) -> int:
    """Group runs by ``(m, form)`` and plot the per-stage relative-error seed mean
    with a min--max band, colour-coded by the number of exercise dates. Writes the
    figure and a ``seed_statistics.yaml`` (inception and peak-per-date error, seed
    mean and standard deviation per group). Returns ``0``."""
    import yaml
    from collections import defaultdict

    formdisp = {"hard_convex_linear": "hard convex", "soft_pinn": "soft"}
    groups = defaultdict(list)
    for r in runs:
        groups[(r["m"], r["form"])].append(r)
    keys = sorted(groups, key=lambda k: (k[0], 0 if k[1].startswith("hard") else 1))
    ms = sorted({m for m, _ in keys})
    cmap = plt.cm.viridis(np.linspace(0.12, 0.85, len(ms)))
    mcol = {m: c for m, c in zip(ms, cmap)}

    fig, ax = plt.subplots(figsize=(8.5, 5))
    summary = []
    for (m, form) in keys:
        grp = groups[(m, form)]
        n_stages = len(grp[0]["stages"])
        ts = np.array([grp[0]["stages"][k]["t_global"] for k in range(n_stages)])
        # (n_seeds, n_stages) relative errors, stages aligned by index (shared t_global)
        errs = np.array([[run["stages"][k]["rel_l2"] for k in range(n_stages)]
                         for run in grp])
        mean, lo, hi = errs.mean(0), errs.min(0), errs.max(0)
        col, ls = mcol[m], ("--" if form == "soft_pinn" else "-")
        ax.fill_between(ts, lo, hi, color=col, alpha=0.18, lw=0)
        ax.semilogy(ts, mean, marker="o", color=col, ls=ls,
                    label=f"$m={m}$ {formdisp.get(form, form)} ($n={len(grp)}$)")
        inc, peak = errs[:, 0], errs.max(1)  # stage 0 is inception
        summary.append({"m": int(m), "form": form, "n_seeds": len(grp),
                        "inception_mean": float(inc.mean()), "inception_std": float(inc.std()),
                        "peak_mean": float(peak.mean()), "peak_std": float(peak.std())})

    ax.set_xlabel("global time $t_k$ (0 = inception, $T$ = maturity)")
    ax.set_ylabel(r"relative $L^2$ error vs exact (per exercise date)")
    ax.set_title(r"Error propagation: seed mean $\pm$ min--max band", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.invert_xaxis()  # induction runs maturity (right) -> inception (left)
    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=True)
    fig.tight_layout(rect=[0, 0.1, 0.80, 1])
    finalize_figure(
        fig, out / "error_propagation_seeds.png", legends=[leg], axes=[ax],
        formula=(r"seed mean of $\mathrm{rel}\,L^2(t_k)$ over $n=3$ seeds; band = "
                 r"min--max (narrower than the markers, so seed spread $<0.5\%$). "
                 r"Colour by $m$; hard convex solid, soft dashed."),
        dpi=140, formula_fontsize=8)

    with open(out / "seed_statistics.yaml", "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    print(f"\nwrote aggregated figure + seed_statistics.yaml to {out}\n")
    for s in summary:
        print(f"m={s['m']:>2} {s['form']:18s} n={s['n_seeds']}  "
              f"inception={s['inception_mean']:.3e} +/- {s['inception_std']:.1e}  "
              f"peak={s['peak_mean']:.3e} +/- {s['peak_std']:.1e}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", nargs="*", help="bermudan_backward_induction run dirs")
    p.add_argument("--aggregate-seeds", action="store_true",
                   help="group runs by (m, form); plot per-stage seed mean with a min--max band")
    args = p.parse_args(argv)

    if args.run_dirs:
        run_dirs = [Path(d) for d in args.run_dirs]
    else:
        root = script_data_dir(Path(__file__).parent / "bermudan_backward_induction.py")
        run_dirs = [Path(d) for d in sorted(glob.glob(str(root / "*_m*_iters*_seed*")))
                    if "_debug_" not in d and Path(d, "validation.npz").exists()]
    runs = [_load(d) for d in run_dirs]
    if not runs:
        raise SystemExit("no runs with validation.npz found")

    out = script_data_dir(__file__) / utc_timestamp()
    out.mkdir(parents=True, exist_ok=True)

    if args.aggregate_seeds:
        return _aggregate(runs, out)

    # colour by run; hard forms solid, soft_pinn dashed (it does not enforce the
    # terminal condition, so it is the odd one out — a reference, not the method).
    palette = plt.cm.viridis(np.linspace(0.1, 0.8, len(runs)))
    def style(r):
        return {"ls": "--" if r["form"] == "soft_pinn" else "-"}

    def label(r):
        return f"m={r['m']} {r['form']}"

    # ---- Figure 1: error propagation ----
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for r, col in zip(runs, palette):
        ts = [s["t_global"] for s in r["stages"]]
        es = [s["rel_l2"] for s in r["stages"]]
        ax.semilogy(ts, es, marker="o", color=col, label=label(r), **style(r))
    ax.set_xlabel("global time $t_k$ (0 = inception, $T$ = maturity)")
    ax.set_ylabel(r"relative $L^2$ error vs exact (per exercise date)")
    ax.set_title("Error propagation through the learned backward induction", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.invert_xaxis()  # induction runs maturity (right) -> inception (left)
    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=True)
    fig.tight_layout(rect=[0, 0.1, 0.80, 1])
    finalize_figure(
        fig, out / "error_propagation_comparison.png", legends=[leg], axes=[ax],
        formula=(r"$\mathrm{rel}\,L^2(t_k)=\|\hat V(\cdot,t_k)-V^\star(\cdot,t_k)\|/"
                 r"\|V^\star(\cdot,t_k)\|$ on the eval window; $V^\star$ exact chained-"
                 r"convolution Bermudan value, $\hat V$ learned. Hard forms solid, "
                 r"soft_pinn dashed."), dpi=140, formula_fontsize=8)

    # ---- Figure 2: inception price overlay ----
    fig, ax = plt.subplots(figsize=(8.5, 5))
    S = runs[0]["S"]
    ax.plot(S, runs[0]["payoff"], ":", color="#d1495b", lw=1.4, label=r"payoff $(K-S)^+$")
    ax.plot(S, runs[0]["european0"], "--", color="#888888", lw=1.2, label="European put")
    # exact Bermudan inception value differs by m; draw each run's exact + learned
    for r, col in zip(runs, palette):
        ax.plot(S, r["stages"][0]["v_exact"], "--", color=col, lw=1.0, alpha=0.7)
        ax.plot(S, r["stages"][0]["v_net"], color=col, lw=1.6, label=label(r), **style(r))
    ax.set_xlabel("spot $S$"); ax.set_ylabel("value $V(\\cdot,0)$")
    ax.set_title("Learned inception price vs exact (dashed, per run)", fontsize=10)
    ax.grid(True, alpha=0.3)
    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=True)
    fig.tight_layout(rect=[0, 0.1, 0.80, 1])
    finalize_figure(
        fig, out / "inception_comparison.png", legends=[leg], axes=[ax],
        formula=(r"solid = learned $\hat V(\cdot,0)$; dashed (same colour) = exact "
                 r"Bermudan $V^\star(\cdot,0)$ for that $m$; more exercise dates "
                 r"raise the value toward the American limit"), dpi=140, formula_fontsize=8)

    # console summary
    print(f"\nwrote comparison figures to {out}\n")
    for r in runs:
        chain = "  ".join(f"t={s['t_global']:.3f}:{s['rel_l2']:.2e}" for s in r["stages"])
        print(f"m={r['m']:>2} {r['form']:18s} inception={r['inception_rel_l2']:.2e}  | {chain}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
