"""Comparison plots for the ansatz-form ablation (torch-free replot).

Every figure is rebuilt from the per-variant ``hist.npz`` / ``metrics.npz`` /
``slices.npz`` artefacts written by the runner; no model is reloaded and nothing
is recomputed, so ``--replot`` is cheap and login-node safe.

Plot conventions (see ``~/.claude/CLAUDE.md``):
    * trained models -> solid stroke, distinguished by colour;
    * analytical reference (exact solution / terminal datum) -> dashed;
    * legends placed outside the axes;
    * a formula textbox is added below each figure.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

import _ansatz_forms_catalogue as cat  # noqa: E402
from _figure_layout import finalize_figure  # noqa: E402

HARD_VARIANTS = ("hard_constant_linear", "hard_constant_exp",
                 "hard_blended_linear", "hard_blended_exp")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_run(ablation_dir: Path):
    """Return (metadata, {variant_name: {hist, metrics, slices}})."""
    meta = {}
    meta_path = ablation_dir / "metadata.yaml"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = yaml.safe_load(f) or {}

    runs: dict[str, dict] = {}
    for v in cat.METHOD_VARIANTS:
        vdir = ablation_dir / f"variant_{v['name']}"
        if not vdir.exists():
            continue
        entry = {"variant": v}
        for key in ("hist", "metrics", "slices"):
            f = vdir / f"{key}.npz"
            if f.exists():
                entry[key] = dict(np.load(f))
        runs[v["name"]] = entry
    return meta, runs


def _assemble_summary(ablation_dir: Path, runs: dict) -> None:
    """Merge per-variant ``summary_<name>.yaml`` (array mode) into ``summary.yaml``."""
    merged: dict = {}
    for part in sorted(ablation_dir.glob("summary_*.yaml")):
        with open(part) as f:
            merged.update(yaml.safe_load(f) or {})
    if merged:
        with open(ablation_dir / "summary.yaml", "w") as f:
            yaml.dump(merged, f, default_flow_style=False, sort_keys=False, width=float("inf"))


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _legend_right(ax):
    """Place the legend in the right gutter, outside the data area; return it."""
    return ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                     fontsize=7, frameon=True)


def _mark_eval_window(ax, runs):
    """Draw dotted vertical markers at the inner evaluation-window edges."""
    for entry in runs.values():
        s = entry.get("slices")
        if s is not None and "x_eval_lo" in s:
            lo, hi = float(s["x_eval_lo"][0]), float(s["x_eval_hi"][0])
            x = next(iter(runs.values()))["slices"]["x"]
            if lo > x.min() or hi < x.max():  # only when there is a buffer
                ax.axvline(lo, ls=":", color="grey", lw=1.0)
                ax.axvline(hi, ls=":", color="grey", lw=1.0,
                           label="eval window")
            return


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_loss_components(out_dir, runs, label):
    panels = [("loss", "total loss"), ("loss_pde", r"PDE residual $\mathcal{L}_{\rm pde}$"),
              ("loss_tc", r"terminal mismatch (diagnostic)"),
              ("boundary_error", r"spatial-boundary drift (diagnostic)")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (key, title) in zip(axes.flat, panels):
        for name, entry in runs.items():
            if "hist" not in entry:
                continue
            h = entry["hist"]
            if key not in h:
                continue
            v = entry["variant"]
            y = np.clip(h[key], 1e-30, None)
            ax.loglog(h["iter"], y, "-", marker=".", ms=3,
                      color=v["color"], label=v["label"], lw=1.3)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("iteration (log scale; iter 1 at left)")
        ax.grid(True, which="both", alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7,
                     frameon=True, bbox_to_anchor=(0.5, 0.14))
    fig.suptitle("Loss components per ansatz form", fontsize=11)
    fig.tight_layout(rect=[0, 0.30, 1, 0.97])
    components_formula = (
        label + "\n"
        r"optimised: $\mathcal{L}_{\rm pde}=\mathbb{E}_\mu[(\mathcal{P}\hat u)^2]$"
        r" (soft form also penalises terminal mismatch $\mathcal{L}_{\rm tc}$)" + "\n"
        r"diagnostics (not in loss): terminal $\mathbb{E}_x[(\hat u(x,T)-g)^2]$,  "
        r"boundary drift $\mathbb{E}_{t\sim U[0,T]}[(\hat u(x_b,t)-u^\star(x_b,t))^2]$, "
        r"$x_b\in\{x_{\rm lo},x_{\rm hi}\}$"
    )
    finalize_figure(fig, out_dir / "loss_components.png", legends=[leg],
                    axes=list(axes.flat), formula=components_formula, dpi=130,
                    formula_fontsize=7)


def _plot_loss_decomposition(out_dir, runs, label):
    """Decomposition channels for the hard forms, with the forcing floor split
    by mechanism (blending-velocity vs damped-diffusion)."""
    channels = [
        ("network_energy", r"$\mathbb{E}[R_\theta^2]$ (network energy)"),
        ("cross_term", r"$|2\,\mathbb{E}[R_\theta\,\mathcal{P}\Psi]|$ (cross term)"),
        ("forcing_floor", r"$\mathbb{E}[(\mathcal{P}\Psi)^2]$ (floor)"),
        ("forcing_velocity",
         r"$\mathbb{E}[(\partial_t\Psi)^2]$ (blending-velocity $\lambda'g$)"),
        ("forcing_diffusion",
         r"$\mathbb{E}[(\frac{\sigma^2}{2}\partial_{xx}\Psi)^2]$ (damped diffusion $\lambda\frac{\sigma^2}{2}g''$)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    flat = axes.flat
    any_data = False
    for ax, (key, title) in zip(flat, channels):
        for name in HARD_VARIANTS:
            entry = runs.get(name)
            if entry is None or "hist" not in entry or key not in entry["hist"]:
                continue
            any_data = True
            h = entry["hist"]
            v = entry["variant"]
            y = np.abs(h[key]) if key == "cross_term" else h[key]
            ax.loglog(h["iter"], np.clip(y, 1e-30, None), "-", marker=".", ms=3,
                      color=v["color"], label=v["label"], lw=1.3)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("iteration (log scale; iter 1 at left)")
        ax.grid(True, which="both", alpha=0.3)
    axes[1, 2].axis("off")  # 6th slot: only 5 channels
    if not any_data:
        plt.close(fig)
        return
    handles, labels = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7,
                     frameon=True, bbox_to_anchor=(0.5, 0.13))
    fig.suptitle(r"Stage-residual decomposition (hard forms): "
                 r"$\mathcal{L}=\mathbb{E}[R_\theta^2]+2\mathbb{E}[R_\theta\mathcal{P}\Psi]"
                 r"+\mathbb{E}[(\mathcal{P}\Psi)^2]$;  "
                 r"floor $=\mathbb{E}[(\partial_t\Psi+\frac{\sigma^2}{2}\partial_{xx}\Psi)^2]$",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0.16, 1, 0.96])
    decomposition_formula = (
        label + "\n"
        r"hard ansatz $\hat u=(1-\lambda)\Phi_\theta+\Psi$; floor split by operator part: "
        r"blending-velocity $\partial_t\Psi$ vs damped-diffusion $\frac{\sigma^2}{2}\partial_{xx}\Psi$." + "\n"
        r"For $\Psi=\lambda g$: $\partial_t\Psi=\lambda'g$ (large when $g$ has nonzero mean), "
        r"$\frac{\sigma^2}{2}\partial_{xx}\Psi=\lambda\frac{\sigma^2}{2}g''$ (large when $g''$ is sharp)"
    )
    finalize_figure(fig, out_dir / "loss_decomposition.png", legends=[leg],
                    axes=list(flat), formula=decomposition_formula, dpi=130,
                    formula_fontsize=7)


def _plot_solution_slice(out_dir, runs, label, tag, fname, title, ref_key,
                         ref_label, desc=""):
    fig, ax = plt.subplots(figsize=(8, 5))
    ref_drawn = False
    for name, entry in runs.items():
        if "slices" not in entry:
            continue
        s = entry["slices"]
        v = entry["variant"]
        ax.plot(s["x"], s[f"u_pred_{tag}"], "-", color=v["color"], label=v["label"], lw=1.4)
        if not ref_drawn and ref_key in s:
            ax.plot(s["x"], s[ref_key], "--", color="black", label=ref_label, lw=1.6)
            ref_drawn = True
    _mark_eval_window(ax, runs)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.grid(True, alpha=0.3)
    leg = _legend_right(ax)
    fig.tight_layout(rect=[0, 0.12, 0.80, 1])
    slice_formula = label + ("\n" + desc if desc else "")
    finalize_figure(fig, out_dir / fname, legends=[leg], axes=[ax],
                    formula=slice_formula, dpi=130, formula_fontsize=7)


def _plot_error_t0(out_dir, runs, label):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, entry in runs.items():
        if "slices" not in entry:
            continue
        s = entry["slices"]
        v = entry["variant"]
        err = np.abs(s["u_pred_t0"] - s["u_ref_t0"])
        ax.semilogy(s["x"], np.clip(err, 1e-30, None), "-",
                    color=v["color"], label=v["label"], lw=1.4)
    _mark_eval_window(ax, runs)
    ax.set_title(r"Absolute error vs exact at $t=0$", fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$|\hat u - u^\star|$")
    ax.grid(True, which="both", alpha=0.3)
    leg = _legend_right(ax)
    fig.tight_layout(rect=[0, 0.13, 0.80, 1])
    error_formula = (
        label + "\n"
        r"pointwise absolute error $|\hat u(x,0)-u^\star(x,0)|$ on the back-propagated "
        r"$t=0$ slice (log axis); solid: trained forms; dotted: evaluation-window edges"
    )
    finalize_figure(fig, out_dir / "error_t0.png", legends=[leg], axes=[ax],
                    formula=error_formula, dpi=130, formula_fontsize=7)


def _plot_summary_metrics(out_dir, runs, label):
    keys = [("rel_l2", r"rel. $L^2$ (space-time)"),
            ("rel_l2_t0", r"rel. $L^2$ at $t=0$"),
            ("tc_l2", r"terminal rel. $L^2$ at $t=T$")]
    names = [n for n in runs if "metrics" in runs[n]]
    if not names:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (key, title) in zip(axes, keys):
        vals = [float(runs[n]["metrics"][key][0]) for n in names]
        colors = [runs[n]["variant"]["color"] for n in names]
        ax.bar(range(len(names)), np.clip(vals, 1e-30, None), color=colors)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=9)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=6)
        ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.suptitle("Summary metrics per form (lower is better)", fontsize=11)
    fig.tight_layout(rect=[0, 0.20, 1, 0.95])
    summary_formula = (
        label + "\n"
        r"per-form relative $L^2$ error vs exact (log axis, one bar per ansatz form):" + "\n"
        r"space-time $\|\hat u-u^\star\|_{L^2(\mathcal{X}_{\rm eval}\times[0,T])}/\|u^\star\|_{L^2}$;  "
        r"the $t=0$ slice $\|\hat u(\cdot,0)-u^\star(\cdot,0)\|/\|u^\star(\cdot,0)\|$;  "
        r"terminal $\|\hat u(\cdot,T)-g\|/\|g\|$ (hard forms $=0$ by construction)"
    )
    finalize_figure(fig, out_dir / "summary_metrics.png", legends=[],
                    axes=list(axes), formula=summary_formula, dpi=130,
                    formula_fontsize=7)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def replot(ablation_dir: Path) -> None:
    """Regenerate every comparison figure from saved artefacts."""
    ablation_dir = Path(ablation_dir)
    meta, runs = _load_run(ablation_dir)
    _assemble_summary(ablation_dir, runs)
    if not runs:
        raise FileNotFoundError(f"No variant_* artefacts found under {ablation_dir}")

    out_dir = ablation_dir / "comparison"
    out_dir.mkdir(exist_ok=True)
    label = meta.get("label", "")

    _plot_loss_components(out_dir, runs, label)
    _plot_loss_decomposition(out_dir, runs, label)
    _plot_solution_slice(
        out_dir, runs, label, tag="t0", fname="solution_t0.png",
        title=r"Trained solution vs exact at $t=0$ (propagated-back slice)",
        ref_key="u_ref_t0", ref_label=r"exact $u^\star(\cdot,0)$",
        desc=(r"solid: trained $\hat u(\cdot,0)$ per ansatz form;  "
              r"dashed: exact $u^\star(\cdot,0)$;  dotted: evaluation-window edges"),
    )
    _plot_solution_slice(
        out_dir, runs, label, tag="tT", fname="terminal_tT.png",
        title=r"Trained solution vs terminal datum at $t=T$",
        ref_key="g", ref_label=r"terminal datum $g$",
        desc=(r"solid: trained $\hat u(\cdot,T)$ per ansatz form;  "
              r"dashed: terminal datum $g$ ($\hat u(\cdot,T)=g$ exactly for the hard forms)"),
    )
    _plot_error_t0(out_dir, runs, label)
    _plot_summary_metrics(out_dir, runs, label)
    _plot_greeks(out_dir, runs, label)


def _plot_greeks(out_dir, runs, label):
    """Spot Delta and Gamma at t=0 vs the reference (option ICs only)."""
    has = any("slices" in e and "nn_delta" in e["slices"] for e in runs.values())
    if not has:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ref_drawn = False
    for name, entry in runs.items():
        s = entry.get("slices")
        if s is None or "nn_delta" not in s:
            continue
        v = entry["variant"]
        axes[0].plot(s["spot"], s["nn_delta"], "-", color=v["color"], lw=1.3, label=v["label"])
        axes[1].plot(s["spot"], s["nn_gamma"], "-", color=v["color"], lw=1.3, label=v["label"])
        if not ref_drawn:
            axes[0].plot(s["spot"], s["ref_delta"], "--", color="black", lw=1.4, label="exact")
            axes[1].plot(s["spot"], s["ref_gamma"], "--", color="black", lw=1.4, label="exact")
            ref_drawn = True
    axes[0].set_title(r"$\Delta = \partial V/\partial S$ at $t=0$", fontsize=10)
    axes[1].set_title(r"$\Gamma = \partial^2 V/\partial S^2$ at $t=0$", fontsize=10)
    for ax in axes:
        ax.set_xlabel("S"); ax.grid(True, alpha=0.3)
    handles, labs = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labs, loc="lower center", ncol=4, fontsize=7,
                     frameon=True, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Spot Greeks vs reference (trained solid, exact dashed)", fontsize=11)
    fig.tight_layout(rect=[0, 0.14, 1, 0.95])
    finalize_figure(fig, out_dir / "greeks.png", legends=[leg], axes=list(axes),
                    formula=(label + "\n"
                             r"$\Delta_S=e^{-x}\partial_x u$,  "
                             r"$\Gamma_S=e^{-2x}(\partial_{xx}u-\partial_x u)$, $x=\ln S$; "
                             r"autograd through the trained field vs the exact reference"),
                    formula_fontsize=7)
