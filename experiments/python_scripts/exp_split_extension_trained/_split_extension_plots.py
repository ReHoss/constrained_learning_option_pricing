"""Comparison plots for the stage-2 split-extension ablation (torch-free).

Every figure is rebuilt from the per-variant ``hist.npz`` / ``metrics.npz`` /
``slices.npz`` / ``spectra.npz`` artefacts written by the runner; no model is
reloaded and nothing is recomputed, so ``--replot`` is cheap and login-node
safe.

Plot conventions (repository-wide):
    * trained variants -> solid stroke, distinguished by colour;
    * analytical reference (exact solution, terminal target, exact forcing
      spectrum) -> dashed;
    * auxiliary annotation (cutoff markers, thresholds) -> dotted;
    * legends placed outside the axes; a formula textbox below each figure.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

import _split_extension_catalogue as catalogue  # noqa: E402
from _figure_layout import finalize_figure  # noqa: E402

# Fallback for runs whose metadata.yaml is absent: the cell name is embedded
# in the directory name per the folder convention of the specification.
_DIRECTORY_NAME_CELL_PATTERN = re.compile(
    r"Z_(?P<cell>[a-z0-9_]+)_iters\d+(?:_seed\d+)?$"
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _positive_or_masked(values):
    """Values for a logarithmic axis, with the non-positive ones omitted.

    A logarithmic axis cannot represent zero. Flooring a zero to 1e-30 draws it
    as a tiny non-zero value, which misreports the datum (the exact-solution
    variant has an identically zero forcing, and it must not appear as a small
    positive one). Masking omits the point: the curve is simply absent there.
    """
    import numpy as np

    array = np.asarray(values, dtype=float)
    return np.ma.masked_where(~(array > 0.0), array)


def _resolve_cell_name(ablation_dir: Path, metadata: dict) -> str:
    if "cell" in metadata:
        return str(metadata["cell"])
    match = _DIRECTORY_NAME_CELL_PATTERN.search(ablation_dir.name)
    if match is None:
        raise ValueError(
            f"Cannot determine the cell of {ablation_dir}: metadata.yaml has "
            "no 'cell' key and the directory name does not follow the "
            "'Z_<cell>_iters<N>[_seed<S>]' convention."
        )
    return match.group("cell")


def _load_run(ablation_dir: Path):
    """Return ``(metadata, cell_name, {variant_name: entry})``."""
    metadata = {}
    metadata_path = ablation_dir / "metadata.yaml"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = yaml.safe_load(f) or {}
    cell_name = _resolve_cell_name(ablation_dir, metadata)

    runs: dict[str, dict] = {}
    for variant in catalogue.variants_for_cell(cell_name):
        variant_dir = ablation_dir / f"variant_{variant['name']}"
        if not variant_dir.exists():
            continue
        entry = {"variant": variant}
        for key in ("hist", "metrics", "slices", "spectra"):
            artefact = variant_dir / f"{key}.npz"
            if artefact.exists():
                entry[key] = dict(np.load(artefact))
        runs[variant["name"]] = entry
    return metadata, cell_name, runs


def _assemble_summary(ablation_dir: Path) -> None:
    """Merge per-variant ``summary_<name>.yaml`` into ``summary.yaml``."""
    merged: dict = {}
    for part in sorted(ablation_dir.glob("summary_*.yaml")):
        with open(part) as f:
            merged.update(yaml.safe_load(f) or {})
    if merged:
        with open(ablation_dir / "summary.yaml", "w") as f:
            yaml.dump(
                merged, f, default_flow_style=False, sort_keys=False,
                width=float("inf"),
            )


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _legend_below(fig, source_ax, *, ncol=3, anchor=(0.5, 0.02)):
    handles, labels = source_ax.get_legend_handles_labels()
    if not handles:
        return None
    return fig.legend(handles, labels, loc="lower center", ncol=ncol,
                      fontsize=7, frameon=True, bbox_to_anchor=anchor)


def _legend_right(ax):
    return ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                     fontsize=7, frameon=True)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_loss_components(out_dir, runs, label):
    panels = [
        ("loss", r"Total loss $\mathbb{E}[(P\hat u)^2]$"),
        ("loss_tc", r"Terminal mismatch (diagnostic, $=0$ for hard forms)"),
        ("forcing_floor", r"Forcing floor $\mathbb{E}[(P\Psi)^2]$"),
        ("grad_norm", r"Gradient norm (probe)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (key, title) in zip(axes.flat, panels):
        for entry in runs.values():
            if "hist" not in entry or key not in entry["hist"]:
                continue
            h = entry["hist"]
            v = entry["variant"]
            ax.loglog(h["iter"], _positive_or_masked(h[key]), "-",
                      marker=".", ms=3, color=v["color"], label=v["label"],
                      lw=1.3)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Iteration (log scale)")
        ax.grid(True, which="both", alpha=0.3)
    leg = _legend_below(fig, axes[0, 0], ncol=2, anchor=(0.5, 0.13))
    fig.suptitle("Loss components per extension variant", fontsize=11)
    fig.tight_layout(rect=[0, 0.30, 1, 0.97])
    components_formula = (
        label + "\n"
        r"optimised: $\mathbb{E}_\mu[(P\hat u)^2]$ over the interior batch; "
        r"$P\hat u=R_\theta+P\Psi$, $R_\theta=(1-\lambda)P\Phi_\theta-\lambda'\Phi_\theta$;" + "\n"
        r"diagnostics (not in the loss): terminal "
        r"$\mathbb{E}_x[(\hat u(x,T)-g)^2]$ and the gradient-norm probe"
    )
    finalize_figure(fig, out_dir / "loss_components.png", legends=[leg],
                    axes=list(axes.flat), formula=components_formula,
                    dpi=130, formula_fontsize=7)


def _plot_loss_decomposition(out_dir, runs, label):
    """Six residual-decomposition channels including advection/reaction."""
    channels = [
        ("network_energy", r"$\mathbb{E}[R_\theta^2]$ (network energy)"),
        ("cross_term", r"$|2\,\mathbb{E}[R_\theta\,P\Psi]|$ (cross term)"),
        ("forcing_velocity", r"$\mathbb{E}[(\partial_t\Psi)^2]$ (velocity)"),
        ("forcing_diffusion", r"$\mathbb{E}[(\nu\,\partial_{xx}\Psi)^2]$ (diffusion)"),
        ("forcing_advection", r"$\mathbb{E}[(\mu\,\partial_x\Psi)^2]$ (advection)"),
        ("forcing_reaction", r"$\mathbb{E}[(r_0\Psi)^2]$ (reaction)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    any_data = False
    for ax, (key, title) in zip(axes.flat, channels):
        for entry in runs.values():
            if "hist" not in entry or key not in entry["hist"]:
                continue
            any_data = True
            h = entry["hist"]
            v = entry["variant"]
            values = np.abs(h[key]) if key == "cross_term" else h[key]
            ax.loglog(h["iter"], _positive_or_masked(values), "-",
                      marker=".", ms=3, color=v["color"], label=v["label"],
                      lw=1.3)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Iteration (log scale)")
        ax.grid(True, which="both", alpha=0.3)
    if not any_data:
        plt.close(fig)
        return
    leg = _legend_below(fig, axes[0, 0], ncol=3, anchor=(0.5, 0.12))
    fig.suptitle(
        r"Residual decomposition: "
        r"$\mathcal{L}=\mathbb{E}[R_\theta^2]+2\,\mathbb{E}[R_\theta\,P\Psi]"
        r"+\mathbb{E}[(P\Psi)^2]$; floor split by operator channel",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.26, 1, 0.96])
    decomposition_formula = (
        label + "\n"
        r"$P\Psi=\partial_t\Psi+\nu\,\partial_{xx}\Psi+\mu\,\partial_x\Psi"
        r"+r_0\Psi$; the four floor channels attribute the forcing by "
        r"mechanism (velocity / diffusion / advection / reaction)"
    )
    finalize_figure(fig, out_dir / "loss_decomposition.png", legends=[leg],
                    axes=list(axes.flat), formula=decomposition_formula,
                    dpi=130, formula_fontsize=7)


def _plot_solution_t0(out_dir, runs, label):
    fig, ax = plt.subplots(figsize=(8, 5))
    reference_drawn = False
    for entry in runs.values():
        if "slices" not in entry:
            continue
        s = entry["slices"]
        v = entry["variant"]
        ax.plot(s["x"], s["u_pred_t0"], "-", color=v["color"],
                label=v["label"], lw=1.4)
        if not reference_drawn:
            ax.plot(s["x"], s["u_ref_t0"], "--", color="black",
                    label=r"exact $u^\star(\cdot,0)$", lw=1.6)
            reference_drawn = True
    ax.set_title(r"Trained solution vs exact at $t=0$", fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.grid(True, alpha=0.3)
    leg = _legend_right(ax)
    fig.tight_layout(rect=[0, 0.12, 0.72, 1])
    finalize_figure(
        fig, out_dir / "solution_t0.png", legends=[leg], axes=[ax],
        formula=(label + "\n"
                 r"solid: trained $\hat u(\cdot,0)$ per extension variant; "
                 r"dashed: exact $u^\star(\cdot,0)$ (finite component sum)"),
        dpi=130, formula_fontsize=7)


def _plot_error_t0(out_dir, runs, label):
    fig, ax = plt.subplots(figsize=(8, 5))
    for entry in runs.values():
        if "slices" not in entry:
            continue
        s = entry["slices"]
        v = entry["variant"]
        pointwise_error = np.abs(s["u_pred_t0"] - s["u_ref_t0"])
        ax.semilogy(s["x"], _positive_or_masked(pointwise_error), "-",
                    color=v["color"], label=v["label"], lw=1.4)
    ax.set_title(r"Absolute error vs exact at $t=0$", fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$|\hat u - u^\star|$")
    ax.grid(True, which="both", alpha=0.3)
    leg = _legend_right(ax)
    fig.tight_layout(rect=[0, 0.13, 0.72, 1])
    finalize_figure(
        fig, out_dir / "error_t0.png", legends=[leg], axes=[ax],
        formula=(label + "\n"
                 r"pointwise absolute error $|\hat u(x,0)-u^\star(x,0)|$ on "
                 r"the back-propagated $t=0$ slice (log axis)"),
        dpi=130, formula_fontsize=7)


def _plot_terminal_network_target(out_dir, runs, label):
    """Trained bare network at t=T (solid) vs its exact target (dashed)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for entry in runs.values():
        if "slices" not in entry or "phi_theta_tT" not in entry["slices"]:
            continue
        s = entry["slices"]
        v = entry["variant"]
        ax.plot(s["x"], s["phi_theta_tT"], "-", color=v["color"],
                label=v["label"], lw=1.4)
        ax.plot(s["x"], s["phi_star_tT"], "--", color=v["color"], lw=1.1)
    ax.set_title(
        r"Terminal network profile vs exact target at $t=T$", fontsize=10
    )
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\Phi$")
    ax.grid(True, alpha=0.3)
    leg = _legend_right(ax)
    fig.tight_layout(rect=[0, 0.15, 0.72, 0.95])
    finalize_figure(
        fig, out_dir / "terminal_network_target.png", legends=[leg], axes=[ax],
        formula=(label + "\n"
                 r"solid: trained $\Phi_\theta(\cdot,T)$; dashed (same "
                 r"colour): exact target "
                 r"$\Phi^\star(\cdot,T)=-(P\Psi)(\cdot,T)/d_T'(T)"
                 r"=T\,(P\Psi)(\cdot,T)$ for the linear factor;" + "\n"
                 r"zero-forcing variants have $\Phi^\star=0$"),
        dpi=130, formula_fontsize=7)


def _plot_residual_spectra(out_dir, runs, label):
    """Cancellation ratios (left) and residual/forcing power spectra (right)."""
    fig, (ax_ratio, ax_power) = plt.subplots(1, 2, figsize=(13, 5.2))
    any_ratio = False
    for entry in runs.values():
        if "spectra" not in entry:
            continue
        sp = entry["spectra"]
        v = entry["variant"]
        k = sp["wavenumber_bins"]
        forcing_defined = bool(sp["forcing_defined"][0])
        if forcing_defined:
            any_ratio = True
            mask = sp["in_band_mask"].astype(bool) & (k > 0)
            ax_ratio.semilogx(
                k[mask],
                sp["cancellation_ratio_running_mean"][mask],
                "-", color=v["color"], lw=1.5, label=v["label"],
            )
            if bool(sp["k_star_defined"][0]):
                ax_ratio.axvline(float(sp["k_star"][0]), ls=":",
                                 color=v["color"], lw=1.0)
            ax_power.loglog(
                k[1:], _positive_or_masked(sp["forcing_power"][1:]), "--",
                color=v["color"], lw=1.0,
            )
        ax_power.loglog(
            k[1:], _positive_or_masked(sp["residual_power"][1:]), "-",
            color=v["color"], lw=1.4, label=v["label"],
        )
    ax_ratio.axhline(1.0, ls=":", color="grey", lw=1.0)
    # Bound the VIEW, never the data: a ratio above the frame runs off it.
    ax_ratio.set_ylim(0.0, 1.5)
    ax_ratio.axhline(0.5, ls=":", color="grey", lw=0.8)
    ax_ratio.set_xlabel("Spatial wavenumber $k$")
    ax_ratio.set_ylabel(r"$|\hat r_k|^2/|\widehat{Lh}(k)|^2$ (running mean)")
    ax_ratio.set_title(
        "Per-spectral-component cancellation (dotted: measured $k_\\star$)",
        fontsize=10,
    )
    ax_ratio.grid(True, which="both", alpha=0.3)
    if not any_ratio:
        ax_ratio.text(0.5, 0.5, "no non-zero forcing in this cell",
                      ha="center", va="center",
                      transform=ax_ratio.transAxes, fontsize=9)
    ax_power.set_xlabel("Spatial wavenumber $k$")
    ax_power.set_ylabel(r"Power $|\hat{\cdot}_k|^2$")
    ax_power.set_title(
        "Residual power (solid) vs exact forcing power (dashed)",
        fontsize=10,
    )
    ax_power.grid(True, which="both", alpha=0.3)
    leg = _legend_below(fig, ax_power, ncol=3, anchor=(0.5, 0.16))
    fig.tight_layout(rect=[0, 0.36, 1, 0.95])
    finalize_figure(
        fig, out_dir / "residual_spectra.png", legends=[leg],
        axes=[ax_ratio, ax_power],
        formula=(label + "\n"
                 r"residual side: slice-averaged FFT power of "
                 r"$r(x,t_s)=(P\hat u)(x,t_s)$, $t_s/T\in\{0.1,\dots,0.9\}$; "
                 r"forcing side: exact per-wavenumber" + "\n"
                 r"coefficient $\widehat{Lh}(k,t_s)$ (closed form, never an "
                 r"FFT); $k_\star$: first in-band $k$ with running-mean "
                 r"ratio $\geq 1/2$"),
        dpi=130, formula_fontsize=7)


def _plot_summary_metrics(out_dir, runs, label):
    keys = [
        ("rel_l2", r"Relative $L^2$ (space-time strip)"),
        ("rel_l2_t0", r"Relative $L^2$ at $t=0$"),
        ("rel_l2_corner_t0", r"Corner-window relative $L^2$ at $t=0$"),
        ("best_loss", r"Best training loss"),
    ]
    names = [n for n in runs if "metrics" in runs[n]]
    if not names:
        return
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, (key, title) in zip(axes, keys):
        values = [float(runs[n]["metrics"][key][0]) for n in names]
        colors = [runs[n]["variant"]["color"] for n in names]
        ax.bar(range(len(names)), _positive_or_masked(values), color=colors)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=9)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=6)
        ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.suptitle(
        "Summary metrics per extension variant (lower is better)", fontsize=11
    )
    fig.tight_layout(rect=[0, 0.22, 1, 0.94])
    summary_formula = (
        label + "\n"
        r"relative $L^2$ errors against the exact component sum on the "
        r"$1024\times 11$ evaluation grid; corner window: "
        r"$\mathrm{dist}(x,x^\star)\leq\pi/16$ on the circle; best loss: "
        r"restored best $\mathbb{E}[(P\hat u)^2]$"
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
    metadata, cell_name, runs = _load_run(ablation_dir)
    _assemble_summary(ablation_dir)
    if not runs:
        raise FileNotFoundError(
            f"No variant_* artefacts found under {ablation_dir}"
        )

    out_dir = ablation_dir / "comparison"
    out_dir.mkdir(exist_ok=True)
    label = metadata.get("label", catalogue.cell_by_name(cell_name)["label"])

    _plot_loss_components(out_dir, runs, label)
    _plot_loss_decomposition(out_dir, runs, label)
    _plot_solution_t0(out_dir, runs, label)
    _plot_error_t0(out_dir, runs, label)
    _plot_terminal_network_target(out_dir, runs, label)
    _plot_residual_spectra(out_dir, runs, label)
    _plot_summary_metrics(out_dir, runs, label)
