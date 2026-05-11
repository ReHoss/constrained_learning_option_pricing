"""Temporary plot: gamma slices for vpinn_lbfgs vs vpinn_lbfgs_is_tau.

Question: does the importance-biased time sampling (q(tau) ~ tau^(alpha-1))
help recover the gamma near tau=0, where the analytical solution is singular?

The global eps_Gamma metric in summary.yaml is computed at tau=T/2 and is
NOT representative of the near-maturity behavior — it only tells us about
the middle of the time domain.  Here we compare the predicted gamma at all
five GT slices (tau = 0.02T, T/4, T/2, 3T/4, T) and zoom in on the
sharpest one (tau = 0.02T) to assess the actual goal.

Usage (from repo root):
    python experiments/python_scripts/exp_singularity_european_call/scratch/\
tmp_gamma_near_maturity_is_tau.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUN_DIR = Path(
    "data/exp_singularity_european_call/"
    "20260509_154015_compare-boundary-singularity-european-call_logS_iters20000"
)
OUT_DIR = RUN_DIR / "comparison" / "_scratch"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Strike used in the experiment (constant K from learning_option_pricing.p3 — hardcoded
# here to avoid the package import which pulls torch and a lot of state).
K_STRIKE = 100.0


def load_gt(name: str) -> dict:
    return dict(np.load(RUN_DIR / f"variant_{name}" / "gt_comparison.npz"))


def main() -> None:
    runs = {
        "vpinn":               dict(color="tab:red",    linestyle=":",                lw=2.0),
        "vpinn_50k":           dict(color="darkred",    linestyle=(0, (5, 1, 1, 1, 1, 1)), lw=2.0),
        "vpinn_lbfgs":         dict(color="tab:pink",   linestyle=(0, (1, 1)),       lw=2.0),
        "vpinn_lbfgs_is_tau":  dict(color="tab:cyan",   linestyle=(0, (3, 1, 1, 1)), lw=2.0),
    }
    data = {name: load_gt(name) for name in runs}

    # Shared metadata (same _GT_TAU_SLICES across variants):
    tau_slices = data["vpinn_lbfgs"]["tau_slices"]       # (5,)
    S_vals     = np.exp(data["vpinn_lbfgs"]["x_greek"])  # (120,) — log-S grid back to S
    n_tau      = len(tau_slices)

    # ---- Figure 1: all five tau slices side by side ----
    fig, axes = plt.subplots(1, n_tau, figsize=(4.5 * n_tau, 5), sharey=False)
    for j, tau_val in enumerate(tau_slices):
        ax = axes[j]
        # Black-Scholes reference (same in both data files; take from the first one)
        ax.plot(S_vals, data["vpinn_lbfgs"]["gamma_ref_slices"][j],
                "k--", linewidth=1.5, label=r"$\Gamma^{\rm BS}$", zorder=10)
        for name, style in runs.items():
            ax.plot(S_vals, data[name]["gamma_pred_slices"][j],
                    color=style["color"], linestyle=style["linestyle"],
                    linewidth=style["lw"], label=name)
        ax.axvline(K_STRIKE, color="gray", linestyle=":", linewidth=0.8)
        ax.set_title(rf"$\tau = {tau_val:.3f}$", fontsize=11)
        ax.set_xlabel("$S$")
        if j == 0:
            ax.set_ylabel(r"$\Gamma = \partial^2 V / \partial S^2$")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Gamma per $\\tau$ — uniform vs IS-biased time sampling  "
        "(K=100, log-S coordinates)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT_DIR / "tmp_gamma_per_tau_lbfgs_vs_is_tau.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved → {out}")

    # ---- Figure 2: zoom on the near-maturity slice (tau = 0.02T) ----
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(13, 5))
    j = 0   # tau = 0.02 * T — closest to singularity
    tau_val = tau_slices[j]
    for ax in (ax_full, ax_zoom):
        ax.plot(S_vals, data["vpinn_lbfgs"]["gamma_ref_slices"][j],
                "k--", linewidth=1.8, label=r"$\Gamma^{\rm BS}$", zorder=10)
        for name, style in runs.items():
            ax.plot(S_vals, data[name]["gamma_pred_slices"][j],
                    color=style["color"], linestyle=style["linestyle"],
                    linewidth=style["lw"] + 0.5, label=name)
        ax.axvline(K_STRIKE, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel("$S$")
        ax.set_ylabel(r"$\Gamma$")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    ax_full.set_title(rf"Full eval range — $\tau = {tau_val:.3f}$ (near maturity)")
    # Zoom around the strike (where the BS gamma has its narrow peak)
    ax_zoom.set_xlim(0.85 * K_STRIKE, 1.15 * K_STRIKE)
    ax_zoom.set_title(rf"Zoom around $S = K = {K_STRIKE:.0f}$")

    # Compute peak heights for quick numeric comparison
    bs_peak = data["vpinn_lbfgs"]["gamma_ref_slices"][j].max()
    msgs = [f"BS peak: {bs_peak:.4f}"]
    for name in runs:
        peak = data[name]["gamma_pred_slices"][j].max()
        msgs.append(f"{name}: peak={peak:.4f}  ({100 * peak / bs_peak:.1f}% of BS)")
    fig.suptitle(
        "Gamma near maturity (tau=0.02T) — does IS-biased sampling sharpen the peak?\n"
        + "   |   ".join(msgs),
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT_DIR / "tmp_gamma_zoom_tau002_lbfgs_vs_is_tau.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved → {out}")

    # ---- Also print numeric comparison ----
    print("\nGamma peak ratio (predicted / Black-Scholes) at each tau slice:")
    header = f"{'tau':>10}  " + "  ".join(f"{name:>20}" for name in runs)
    print(header)
    for j, tau_val in enumerate(tau_slices):
        bs_peak = data["vpinn_lbfgs"]["gamma_ref_slices"][j].max()
        ratios = {name: data[name]["gamma_pred_slices"][j].max() / bs_peak for name in runs}
        row = f"{tau_val:>10.3f}  " + "  ".join(f"{ratios[name]:>19.1%}" for name in runs)
        print(row)


if __name__ == "__main__":
    main()
