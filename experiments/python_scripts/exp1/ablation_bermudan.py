"""Ablation study — Bermuda put option pricing.

Fixed settings:
    g2      = "bs"   (European Black-Scholes anchor for Stage A)
    tc_enforced = True  (hard-enforced terminal condition via ETCNN ansatz)

Ablation axes (Stage B only, on [0, t1]):
    put_ansatz         singularity extraction ansatz (U_B = v + ũ_θ)
    bypass_v           operator bypass: drop fictitious put v from PDE loss
    use_spatial_weight inverted-Gaussian weighting of PDE loss near s*

Five variants:
    baseline       put_ansatz=False, bypass_v=False, spatial_weight=False
    +put-ansatz    put_ansatz=True,  bypass_v=False, spatial_weight=False
    +bypass        put_ansatz=True,  bypass_v=True,  spatial_weight=False
    +spatial_wt    put_ansatz=True,  bypass_v=False, spatial_weight=True
    full           put_ansatz=True,  bypass_v=True,  spatial_weight=True

Stage A (ETCNN on [t1, T]) is trained once and shared across all Stage B variants
so that only Stage B design choices are compared.

Usage (from repo root):
    # Smoke test — 50 iters each stage:
    python3 experiments/python_scripts/exp1/ablation_bermudan.py

    # Full run — 500 iters each stage:
    python3 experiments/python_scripts/exp1/ablation_bermudan.py --iters-a 500 --iters-b 500

    # Regenerate comparison plots from a saved run (no retraining):
    python3 experiments/python_scripts/exp1/ablation_bermudan.py \\
        --replot data/ablation_bermudan/20260422_132317_itersA500_itersB500
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

# Make learning_option_pricing importable
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
# Make phase3_training importable as a module
sys.path.insert(0, str(Path(__file__).resolve().parent))

import phase3_training as p3
from phase3_training import bermudan_problem

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ablation variant definitions
# ---------------------------------------------------------------------------
VARIANTS: list[dict] = [
    {
        "name": "baseline",
        "label": "Baseline (PCHIP)",
        "put_ansatz": False,
        "bypass_v": False,
        "use_spatial_weight": False,
        "interp": "pchip",
        "color": "tab:blue",
        "linestyle": "-",
        "linewidth": 2.0,
    },
    {
        "name": "put-ansatz",
        "label": r"$+$put-ansatz",
        "put_ansatz": True,
        "bypass_v": False,
        "use_spatial_weight": False,
        "interp": "pchip",
        "color": "tab:orange",
        "linestyle": "--",
        "linewidth": 2.0,
    },
    {
        "name": "bypass",
        "label": r"$+$put-ansatz $+$bypass$_v$",
        "put_ansatz": True,
        "bypass_v": True,
        "use_spatial_weight": False,
        "interp": "pchip",
        "color": "tab:green",
        "linestyle": "-.",
        "linewidth": 2.0,
    },
    {
        "name": "spatial_wt",
        "label": r"$+$put-ansatz $+$spatial weight",
        "put_ansatz": True,
        "bypass_v": False,
        "use_spatial_weight": True,
        "interp": "pchip",
        "color": "tab:red",
        "linestyle": ":",
        "linewidth": 2.0,
    },
    {
        "name": "full",
        "label": r"Full (ext $+$ bypass$_v$ $+$ sw)",
        "put_ansatz": True,
        "bypass_v": True,
        "use_spatial_weight": True,
        "interp": "pchip",
        "color": "tab:purple",
        "linestyle": "-",
        "linewidth": 2.5,
    },
]

_SUPTITLE_PARAMS = (
    r"Bermudan put — $g_2^{(A)}=V^e$, "
    r"$K=100$, $r=0.02$, $\sigma=0.25$, $T=1$, $t_1=0.5$, "
    r"$\lambda_f=20$, $\lambda_{tc}=1$"
)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _save_variant_results(res: dict, vdir: Path) -> None:
    """Persist the numeric arrays and loss history needed for --replot."""
    # Loss history arrays
    hist = res["hist_b"]
    np.savez_compressed(
        vdir / "hist_b.npz",
        iter=np.array(hist["iter"]),
        loss=np.array(hist["loss"]),
        loss_f=np.array(hist["loss_f"]),
        loss_tc=np.array(hist["loss_tc"]),
        grad_norm=np.array(hist["grad_norm"]),
        lr=np.array(hist["lr"]),
        tc_enforced=np.array([hist.get("tc_enforced", True)]),
    )
    # Price arrays
    np.savez_compressed(
        vdir / "prices.npz",
        etcnn_b_prices=np.array(res["etcnn_b_prices"]),
        bt_prices=np.array(res["bt_prices"]),
        s_eval_arr=np.array(res["s_eval_arr"]),
    )


def _load_variant_results(vdir: Path, summary_entry: dict) -> dict:
    """Reconstruct a results dict from saved .npz files and summary metrics."""
    hist_npz   = np.load(vdir / "hist_b.npz")
    prices_npz = np.load(vdir / "prices.npz")
    hist = {
        "iter":        hist_npz["iter"].tolist(),
        "loss":        hist_npz["loss"].tolist(),
        "loss_f":      hist_npz["loss_f"].tolist(),
        "loss_tc":     hist_npz["loss_tc"].tolist(),
        "grad_norm":   hist_npz["grad_norm"].tolist(),
        "lr":          hist_npz["lr"].tolist(),
        "tc_enforced": bool(hist_npz["tc_enforced"][0]),
    }
    return {
        **summary_entry,
        "hist_b":         hist,
        "etcnn_b_prices": prices_npz["etcnn_b_prices"],
        "bt_prices":      prices_npz["bt_prices"],
        "s_eval_arr":     prices_npz["s_eval_arr"],
    }


# ---------------------------------------------------------------------------
# Aggregation plots
# ---------------------------------------------------------------------------

def _plot_comparison(results: list[dict], ablation_dir: Path, iters_b: int) -> None:
    """Generate the five cross-variant comparison plots with unified LaTeX notation."""
    comp_dir = ablation_dir / "comparison"
    comp_dir.mkdir(exist_ok=True)

    colors     = [v["color"]     for v in VARIANTS]
    linestyles = [v["linestyle"] for v in VARIANTS]
    labels     = [v["label"]     for v in VARIANTS]
    linewidths = [v["linewidth"] for v in VARIANTS]

    # ------------------------------------------------------------------
    # Plot 1 — Stage B PDE residual loss  L_f^(B)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 7.5))
    for i, res in enumerate(results):
        hist = res.get("hist_b")
        if hist is None:
            continue
        ax.semilogy(
            hist["iter"], hist["loss_f"],
            label=labels[i], color=colors[i],
            linestyle=linestyles[i], linewidth=linewidths[i],
        )
    ax.set_xlabel("Iteration (Stage B)")
    ax.set_ylabel(r"$\mathcal{L}_f^{(B)}$")
    ax.set_title(
        r"$\mathcal{L}_f^{(B)} = \frac{1}{N_f}\sum_i"
        r"\,|\mathcal{F}[U_{\mathrm{pde}}](s_i,t_i)|^2$"
        f"   ({iters_b} iters,  $N_f={p3.N_F}$)"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.suptitle(f"Ablation — Stage B PDE residual loss\n{_SUPTITLE_PARAMS}", fontsize=11)
    
    caption_text = (
        "Mathematical formulations of the plotted PDE residual loss $\\mathcal{L}_f^{(B)}$:\n\n"
        r"• Baseline:  $\frac{1}{N_f} \sum_i \left| \mathcal{L}[ (t_1-t)u_\theta(s_i,t_i) + V^{\mathrm{Berm}}_{\bar{\theta}}(s_i,t_1) ] \right|^2$" "\n"
        r"• +put-ansatz:  $\frac{1}{N_f} \sum_i \left| \mathcal{L}[ v(s_i,t_i) + (t_1-t)u_\theta(s_i,t_i) + g_2(s_i) ] \right|^2$" "\n"
        r"• +bypass$_v$:  $\frac{1}{N_f} \sum_i \left| \mathcal{L}[ (t_1-t)u_\theta(s_i,t_i) + g_2(s_i) ] \right|^2$" "\n"
        r"• +spatial weight / Full:  $\frac{1}{N_f} \sum_i W(s_i) \left| \mathcal{L}[ \dots ] \right|^2$"
    )
    fig.tight_layout(rect=[0, 0.22, 1, 1])
    fig.text(0.5, 0.02, caption_text, ha='center', va='bottom', fontsize=9,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray', boxstyle='round,pad=0.5'))
    
    fig.savefig(comp_dir / "abl1_loss_pde.png", dpi=150)
    plt.close(fig)
    logger.info("[OK] abl1_loss_pde.png")

    # ------------------------------------------------------------------
    # Plot 2 — Stage B total loss  L^(B) = lam_f L_f^(B) + lam_tc L_tc^(B)
    # tc is hard-enforced so L_tc^(B) ~ 0  =>  L^(B) ~ 20 L_f^(B)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, res in enumerate(results):
        hist = res.get("hist_b")
        if hist is None:
            continue
        ax.semilogy(
            hist["iter"], hist["loss"],
            label=labels[i], color=colors[i],
            linestyle=linestyles[i], linewidth=linewidths[i],
        )
    ax.set_xlabel("Iteration (Stage B)")
    ax.set_ylabel(r"$\mathcal{L}^{(B)}$")
    ax.set_title(
        r"$\mathcal{L}^{(B)} = \lambda_f\,\mathcal{L}_f^{(B)} + \lambda_{tc}\,\mathcal{L}_{tc}^{(B)}"
        r"\;\approx\; 20\,\mathcal{L}_f^{(B)}$"
        "\n"
        r"(tc hard-enforced by ansatz: $g_1^{(B)}(s,t_1)=0$, so $\mathcal{L}_{tc}^{(B)}\approx 0$)"
        f"   ({iters_b} iters)"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.suptitle(f"Ablation — Stage B total loss\n{_SUPTITLE_PARAMS}", fontsize=11)
    fig.tight_layout()
    fig.savefig(comp_dir / "abl2_loss_total.png", dpi=150)
    plt.close(fig)
    logger.info("[OK] abl2_loss_total.png")

    # ------------------------------------------------------------------
    # Plot 3 — Pricing curves at t=0
    # ------------------------------------------------------------------
    bt_prices  = results[0].get("bt_prices")
    s_eval_arr = results[0].get("s_eval_arr")

    fig, ax = plt.subplots(figsize=(10, 6))
    if bt_prices is not None and s_eval_arr is not None:
        ax.plot(s_eval_arr, bt_prices,
                label=r"$V^{\mathrm{BT}}(s,0)$  (binomial tree, $N=2000$)",
                color="black", linewidth=2.5, zorder=10)
    for i, res in enumerate(results):
        prices = res.get("etcnn_b_prices")
        s_arr  = res.get("s_eval_arr")
        if prices is None or s_arr is None:
            continue
        ax.plot(s_arr, prices,
                label=r"$\tilde{u}^{(B)}_\theta(s,0)$ — " + labels[i],
                color=colors[i], linestyle=linestyles[i], linewidth=linewidths[i])
    ax.set_xlabel("Asset price $s$")
    ax.set_ylabel(r"Price at $t=0$")
    ax.set_title(
        r"$\tilde{u}^{(B)}_\theta(s,0)$ vs $V^{\mathrm{BT}}(s,0)$  —  all variants"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.suptitle(f"Ablation — Pricing comparison at $t=0$\n{_SUPTITLE_PARAMS}", fontsize=11)
    fig.tight_layout()
    fig.savefig(comp_dir / "abl3_prices.png", dpi=150)
    plt.close(fig)
    logger.info("[OK] abl3_prices.png")

    # ------------------------------------------------------------------
    # Plot 4 — Pointwise absolute error vs BT at t=0
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 7.5))
    for i, res in enumerate(results):
        prices = res.get("etcnn_b_prices")
        bt     = res.get("bt_prices")
        s_arr  = res.get("s_eval_arr")
        if prices is None or bt is None or s_arr is None:
            continue
        err = np.abs(prices - bt)
        mae = np.mean(err)
        ax.plot(s_arr, err,
                label=rf"{labels[i]}  ($\mathrm{{MAE}}={mae:.2e}$)",
                color=colors[i], linestyle=linestyles[i], linewidth=linewidths[i])
    ax.set_xlabel("Asset price $s$")
    ax.set_ylabel(
        r"$|\tilde{u}^{(B)}_\theta(s,0) - V^{\mathrm{BT}}(s,0)|$"
    )
    ax.set_title(
        r"Pointwise error $|\tilde{u}^{(B)}_\theta(s,0) - V^{\mathrm{BT}}(s,0)|$"
        "   at $t=0$"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"Ablation — Pointwise error vs $V^{{\\mathrm{{BT}}}}$ at $t=0$\n{_SUPTITLE_PARAMS}",
        fontsize=11,
    )
    
    caption_text = (
        "Mathematical formulations of the trial solution $\\tilde{u}^{(B)}_\\theta(s, t)$:\n\n"
        r"• Baseline (no put-ansatz):  $\tilde{u}^{(B)}_\theta(s, t) = (t_1 - t)u_\theta(s, t) + V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_1)$" "\n\n"
        r"• With put-ansatz (+put-ansatz, +bypass$_v$, etc.):  $\tilde{u}^{(B)}_\theta(s, t) = v(s, t) + (t_1 - t)u_\theta(s, t) + g_2(s)$" "\n"
        r"   (where $v$ is the fictitious European put, and $g_2(s) = V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_1) - v(s, t_1)$ is the strictly $C^1$ residual)"
    )
    fig.tight_layout(rect=[0, 0.22, 1, 1])
    fig.text(0.5, 0.02, caption_text, ha='center', va='bottom', fontsize=9,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray', boxstyle='round,pad=0.5'))
    
    fig.savefig(comp_dir / "abl4_error_vs_bt.png", dpi=150)
    plt.close(fig)
    logger.info("[OK] abl4_error_vs_bt.png")

    # ------------------------------------------------------------------
    # Plot 5 — Summary bar chart (MAE and relative L2)
    # ------------------------------------------------------------------
    maes    = [res["mae_bt"]    for res in results]
    rel_l2s = [res["rel_l2_bt"] for res in results]
    x       = np.arange(len(VARIANTS))
    xlabels = [v["name"] for v in VARIANTS]   # plain names — no LaTeX in bar tick

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bars0 = axes[0].bar(x, maes, color=colors, edgecolor="black", linewidth=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(xlabels, rotation=20, ha="right", fontsize=9)
    axes[0].set_ylabel(
        r"$\mathrm{MAE} = \frac{1}{N}\sum_i"
        r"\,|\tilde{u}^{(B)}_\theta(s_i,0)-V^{\mathrm{BT}}(s_i,0)|$"
    )
    axes[0].set_title("Mean Absolute Error vs $V^{\\mathrm{BT}}$")
    axes[0].grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars0, maes):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:.2e}", ha="center", va="bottom", fontsize=8,
        )

    bars1 = axes[1].bar(x, rel_l2s, color=colors, edgecolor="black", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(xlabels, rotation=20, ha="right", fontsize=9)
    axes[1].set_ylabel(
        r"$\|\tilde{u}^{(B)}_\theta(\cdot,0)-V^{\mathrm{BT}}(\cdot,0)\|_2"
        r"\;/\;\|V^{\mathrm{BT}}(\cdot,0)\|_2$"
    )
    axes[1].set_title("Relative $L^2$ error vs $V^{\\mathrm{BT}}$")
    axes[1].grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars1, rel_l2s):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:.2e}", ha="center", va="bottom", fontsize=8,
        )

    fig.suptitle(f"Ablation — Summary metrics\n{_SUPTITLE_PARAMS}", fontsize=11)
    fig.tight_layout()
    fig.savefig(comp_dir / "abl5_summary_metrics.png", dpi=150)
    plt.close(fig)
    logger.info("[OK] abl5_summary_metrics.png")


# ---------------------------------------------------------------------------
# Replot mode — regenerate comparison plots from a saved run directory
# ---------------------------------------------------------------------------

def _replot(ablation_dir: Path) -> None:
    """Load saved variant data and regenerate comparison plots."""
    summary_path = ablation_dir / "summary.yaml"
    meta_path    = ablation_dir / "metadata.yaml"

    if not summary_path.exists():
        raise FileNotFoundError(f"summary.yaml not found in {ablation_dir}")

    with open(summary_path) as f:
        summary = yaml.safe_load(f)
    iters_b = 0
    if meta_path.exists():
        with open(meta_path) as f:
            meta = yaml.safe_load(f)
        iters_b = meta.get("iters_b", 0)

    results = []
    for v in VARIANTS:
        vname = v["name"]
        vdir  = ablation_dir / f"variant_{vname}"
        hist_path   = vdir / "hist_b.npz"
        prices_path = vdir / "prices.npz"
        if not hist_path.exists() or not prices_path.exists():
            raise FileNotFoundError(
                f"Missing hist_b.npz or prices.npz in {vdir}.\n"
                "This run predates --replot support. Re-run the ablation to regenerate."
            )
        results.append(_load_variant_results(vdir, summary.get(vname, {})))

    logger.info(f"Loaded {len(results)} variants from {ablation_dir}")
    _plot_comparison(results, ablation_dir, iters_b)
    logger.info(f"Plots written to {ablation_dir / 'comparison'}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation study — Bermuda put, g2=bs, tc_enforced=True"
    )
    parser.add_argument(
        "--iters-a", type=int, default=50,
        help="Stage A iterations (default 50 — smoke test)",
    )
    parser.add_argument(
        "--iters-b", type=int, default=50,
        help="Stage B iterations per variant (default 50 — smoke test)",
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--sigma-w", type=float, default=1.0,
        help="Bandwidth for inverted-Gaussian spatial weight (default 1.0)",
    )
    parser.add_argument(
        "--eps-w", type=float, default=1e-3,
        help="Floor of spatial weight at s* (default 1e-3)",
    )
    parser.add_argument("--n-tc", type=int, default=None, help="Override N_TC")
    parser.add_argument("--n-f",  type=int, default=None, help="Override N_F")
    parser.add_argument(
        "--load-stage-a", type=str, default=None, metavar="PATH",
        help="Path to a pre-trained etcnn_a.pt (or a run directory containing "
             "models/etcnn_a.pt) to skip Stage A training for all variants.",
    )
    parser.add_argument(
        "--replot", type=str, default=None, metavar="DIR",
        help="Regenerate comparison plots from an existing ablation directory "
             "(no retraining). Requires hist_b.npz and prices.npz in each variant subdir.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Replot mode: just regenerate plots, no training
    # ------------------------------------------------------------------
    if args.replot is not None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(message)s",
            datefmt="%H:%M:%S",
            handlers=[logging.StreamHandler()],
        )
        logging.getLogger("matplotlib.mathtext").setLevel(logging.WARNING)
        _replot(Path(args.replot))
        return

    # ------------------------------------------------------------------
    # Apply settings to phase3_training globals
    # ------------------------------------------------------------------
    p3._apply_device_arg(args.device)
    if args.n_tc is not None:
        p3.N_TC = args.n_tc
    if args.n_f is not None:
        p3.N_F = args.n_f

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    ablation_dir = (
        Path("data/ablation_bermudan")
        / f"{timestamp}_itersA{args.iters_a}_itersB{args.iters_b}"
    )
    ablation_dir.mkdir(parents=True, exist_ok=True)
    (ablation_dir / "comparison").mkdir(exist_ok=True)

    for v in VARIANTS:
        vdir = ablation_dir / f"variant_{v['name']}"
        for sub in ("training_metrics", "pricing", "greeks", "diagnostics", "models"):
            (vdir / sub).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(ablation_dir / "ablation.log"),
        ],
    )
    logging.getLogger("matplotlib.mathtext").setLevel(logging.WARNING)

    logger.info("=" * 70)
    logger.info("ABLATION STUDY — Bermuda put  (g2=bs, tc_enforced=True)")
    logger.info("=" * 70)
    logger.info(f"  iters_a={args.iters_a}  iters_b={args.iters_b}")
    logger.info(f"  variants: {[v['name'] for v in VARIANTS]}")
    logger.info(f"  output:   {ablation_dir}")

    # ------------------------------------------------------------------
    # Save metadata
    # ------------------------------------------------------------------
    metadata = {
        "command": " ".join(sys.argv),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fixed": {"g2_type": "bs", "tc_enforced": True},
        "ablation_axes": ["put_ansatz", "bypass_v", "use_spatial_weight"],
        "variants": [
            {k: v for k, v in var.items()
             if k not in ("color", "linestyle", "linewidth")}
            for var in VARIANTS
        ],
        "iters_a": args.iters_a,
        "iters_b": args.iters_b,
        "sigma_w": args.sigma_w,
        "eps_w": args.eps_w,
        "weight_decay": args.weight_decay,
    }
    with open(ablation_dir / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False,
                  width=float("inf"))

    # ------------------------------------------------------------------
    # Run variants
    # ------------------------------------------------------------------
    results: list[dict] = []

    # Resolve pre-trained Stage A path if provided
    load_etcnn_a_path: Path | None = None
    if args.load_stage_a is not None:
        requested = Path(args.load_stage_a)
        if requested.is_dir():
            candidate = requested / "models" / "etcnn_a.pt"
            load_etcnn_a_path = candidate if candidate.exists() else requested / "etcnn_a.pt"
        else:
            load_etcnn_a_path = requested
        if not load_etcnn_a_path.exists():
            raise FileNotFoundError(f"--load-stage-a: file not found: {load_etcnn_a_path}")
        logger.info(f"Stage A: reusing pre-trained model from {load_etcnn_a_path}")

    for idx, variant in enumerate(VARIANTS):
        vname = variant["name"]
        vdir  = ablation_dir / f"variant_{vname}"

        logger.info("")
        logger.info("=" * 70)
        logger.info(f"VARIANT {idx + 1}/{len(VARIANTS)}: {vname}")
        logger.info(f"  put_ansatz={variant['put_ansatz']}"
                    f"  bypass_v={variant['bypass_v']}"
                    f"  spatial_weight={variant['use_spatial_weight']}")
        logger.info("=" * 70)

        res = bermudan_problem(
            out_dir=vdir,
            total_iters=[args.iters_a, args.iters_b],
            interp_method=variant["interp"],
            put_ansatz=variant["put_ansatz"],
            weight_decay=args.weight_decay,
            load_etcnn_a=load_etcnn_a_path,
            g2_type="bs",
            bypass_v=variant["bypass_v"],
            sigma_w=args.sigma_w,
            eps_w=args.eps_w,
            use_spatial_weight=variant["use_spatial_weight"],
        )
        results.append(res)

        # Persist arrays for --replot
        _save_variant_results(res, vdir)

        logger.info(
            f"  [{vname}] done — "
            f"MAE={res['mae_bt']:.4e}  rel_L2={res['rel_l2_bt']:.4e}"
            f"  jump@t1={res['jump_at_t1']:.4e}"
        )

        if load_etcnn_a_path is None:
            load_etcnn_a_path = vdir / "models" / "etcnn_a.pt"
            logger.info(f"  Stage A saved at: {load_etcnn_a_path}")

    # ------------------------------------------------------------------
    # Save aggregated metrics
    # ------------------------------------------------------------------
    summary = {
        v["name"]: {
            "mae_bt":       float(res["mae_bt"]),
            "rel_l2_bt":    float(res["rel_l2_bt"]),
            "jump_at_t1":   float(res["jump_at_t1"]),
            "etcnn_b_at_K": float(res["etcnn_b_at_K"]),
            "s_star": (
                float(res["s_star"])
                if not (isinstance(res["s_star"], float) and np.isnan(res["s_star"]))
                else "nan"
            ),
        }
        for v, res in zip(VARIANTS, results)
    }
    with open(ablation_dir / "summary.yaml", "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False,
                  width=float("inf"))

    # ------------------------------------------------------------------
    # Aggregation comparison plots
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("Generating comparison plots ...")
    _plot_comparison(results, ablation_dir, args.iters_b)

    # ------------------------------------------------------------------
    # Final summary table
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 70)
    logger.info("ABLATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Variant':<22} {'MAE':>12} {'rel_L2':>12} {'jump@t1':>12}")
    logger.info("-" * 60)
    for v, res in zip(VARIANTS, results):
        logger.info(
            f"{v['name']:<22} {res['mae_bt']:>12.4e}"
            f" {res['rel_l2_bt']:>12.4e} {res['jump_at_t1']:>12.4e}"
        )
    logger.info("=" * 70)
    logger.info(f"All outputs saved to: {ablation_dir}")


if __name__ == "__main__":
    main()
