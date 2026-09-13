r"""Predict the error floor caused by the corner-layer terminal-trace defect, and compare it
with the measured error of trained models, band by band.

For every terminal-function mode of the pilot, the trial solution is
Phi_theta = g1 u_theta + g2 with g1(s, T) = 0 and g2(s, t) = zeta((s-B)/eps) h(s, t), where
h(s, T) = (K - s)^+ (raw payoff, Black-Scholes put price, split-semigroup profile alike).
Its terminal trace is therefore

    Phi_theta(s, T) = zeta((s-B)/eps) (K - s)^+,

which differs from the payoff by the DEFECT delta(s) = -(1 - zeta((s-B)/eps)) (K - s)^+,
supported on the corner layer [B, B + eps] and of size K - B at the barrier. This defect is
forced by the exactness of the barrier condition (zeta(0) = 0) and cannot be removed by
training: g1 vanishes on t = T, so the network has no influence on the terminal trace.

If the trained solution had zero interior residual and exact barrier and far-field data,
its error w = Phi_theta - V_DO would solve L^BS w = 0 in (B, s_inf) x (0, T) with
w(., T) = delta, w(B, .) = 0, w(s_inf, .) ~ 0, i.e. w would be the price of a down-and-out
claim with payoff delta. By the reflection principle for the log-price diffusion absorbed
at b = ln B (drift nu = r - sigma^2/2), with x0 = ln s,

    w(s, tau) = e^{-r tau} int_b^{ln(B+eps)} [ p(tau; x0, x) - e^{2 nu (b - x0)/sigma^2} p(tau; 2b - x0, x) ]
                                              delta(e^x) dx,

p the Gaussian density of mean x0 + nu tau and variance sigma^2 tau. This PREDICTED floor is
evaluated on the pilot's evaluation grid by quadrature (the same formula with the full payoff
in place of delta reproduces the Reiner-Rubinstein closed form, which is checked and logged),
and compared with the MEASURED error of the trained models saved by
``aggregate_terminal_function_comparison.py`` under ``model_based_diagnostics/evaluation_grids/``
(``learned`` and ``reference`` on the same grid): relative L2 error per band of s (corner window
removed, as in the aggregator's --s-band-edges diagnostic), for the predicted floor alone and
for the measured error, plus the relative L2 norm of (measured error - predicted floor), which
is what remains once the terminal-trace defect is accounted for.

No training, no model evaluation: closed-form quadrature and saved grids only.

Usage:
    python3 experiments/python_scripts/exp_barrier_option/diagnostic_scripts/predict_corner_layer_floor.py \
        --aggregation-dir data/aggregate_terminal_function_comparison/<...>_iters50000_eps0.1_nocorner
"""
from __future__ import annotations

import argparse
import logging
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from learning_option_pricing.pricing.barrier import (  # noqa: E402
    _smoothstep01,
    reiner_rubinstein_down_and_out_put,
)
from learning_option_pricing.utils.figure_layout import finalize_figure  # noqa: E402
from learning_option_pricing.utils.run_context import script_data_dir  # noqa: E402

logger = logging.getLogger("predict_corner_layer_floor")

FORMULA_TEXT = (
    r"Terminal trace of the ansatz: $\Phi_\theta(s,T)=\zeta((s-B)/\varepsilon)(K-s)^+$;  defect "
    r"$\delta(s)=-(1-\zeta((s-B)/\varepsilon))(K-s)^+$ on $[B,B+\varepsilon]$."
    "\n"
    r"Predicted floor $w$: $\mathcal{L}^{BS}w=0$, $w(\cdot,T)=\delta$, $w(B,\cdot)=0$, by the reflection principle "
    r"$w(s,\tau)=e^{-r\tau}\int_b^{\ln(B+\varepsilon)}[p(\tau;x_0,x)-e^{2\nu(b-x_0)/\sigma^2}p(\tau;2b-x_0,x)]\,\delta(e^x)\,dx$, "
    r"$x_0=\ln s$, $b=\ln B$, $\nu=r-\sigma^2/2$."
    "\n"
    r"Measured: $\Phi_\theta-V_{DO}$ of the trained models on the same grid (corner window removed). "
    "Dashed: predicted floor; solid: measured (median over seeds; faint points: seeds)."
)


def absorbed_transition_density(x0: torch.Tensor, x: torch.Tensor, tau: torch.Tensor, b: float,
                                nu: float, sigma: float) -> torch.Tensor:
    """Density at x (> b) of the drifted log-price started at x0 (> b) and killed at b,
    method of images; shapes broadcast."""
    variance = sigma**2 * tau
    def gaussian(mean):
        return torch.exp(-(x - mean) ** 2 / (2 * variance)) / torch.sqrt(2 * torch.pi * variance)
    return gaussian(x0 + nu * tau) - torch.exp(2 * nu * (b - x0) / sigma**2) * gaussian(2 * b - x0 + nu * tau)


def down_and_out_price_by_quadrature(payoff, s_grid: torch.Tensor, tau_grid: torch.Tensor, B: float,
                                     r: float, sigma: float, x_lo: float, x_hi: float, n_quad: int) -> torch.Tensor:
    """e^{-r tau} E[payoff(S_tau) 1_{not knocked out}] on the (s, tau) grid, quadrature over x in [x_lo, x_hi]."""
    b, nu = float(torch.log(torch.tensor(B, dtype=torch.float64))), r - 0.5 * sigma**2
    x = torch.linspace(x_lo, x_hi, n_quad, dtype=torch.float64)
    dx = float(x[1] - x[0])
    payoff_values = payoff(torch.exp(x))
    ss, tt = torch.meshgrid(s_grid, tau_grid, indexing="ij")
    result = torch.zeros_like(ss)
    x0 = torch.log(ss)
    for i in range(ss.shape[0]):  # one price at a time: (n_tau, n_quad) kernels
        kernel = absorbed_transition_density(x0[i, :, None], x[None, :], tt[i, :, None], b, nu, sigma)
        result[i, :] = torch.exp(-r * tt[i, :]) * (kernel * payoff_values[None, :]).sum(dim=1) * dx
    return result


def _grid_l2_norm(values: torch.Tensor, mask: torch.Tensor, cell_area: float) -> float:
    return float(torch.sqrt((values[mask] ** 2).sum() * cell_area))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aggregation-dir", type=str, required=True,
                        help="An aggregation directory whose model_based_diagnostics/evaluation_grids/ holds the "
                             "saved (learned, reference) grids of the trained models.")
    parser.add_argument("--s-band-edges", nargs="+", type=float, default=[0.6, 0.7, 1.0, 2.0, 3.0])
    parser.add_argument("--n-quad", type=int, default=20001, help="Quadrature nodes in log-price.")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    aggregation_dir = Path(args.aggregation_dir)
    grids_dir = aggregation_dir / "model_based_diagnostics" / "evaluation_grids"
    out_dir = Path(args.out_dir) if args.out_dir else script_data_dir(__file__) / (
        f"{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}_{aggregation_dir.name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "predict_corner_layer_floor.log")])
    logger.info("Predicted corner-layer floor vs measured error")
    logger.info(f"  Command: {' '.join(sys.argv)}")
    logger.info(f"  Saved evaluation grids read from: {grids_dir}")
    logger.info(f"  Output directory: {out_dir}")

    grid_files = sorted(grids_dir.glob("*.pt"))
    if not grid_files:
        logger.error("No saved evaluation grid found.")
        sys.exit(1)
    # Contract and epsilon from the run directories' metadata (all runs share them).
    first = torch.load(grid_files[0], weights_only=False)
    run_dir = Path(yaml.safe_load(open(aggregation_dir / "summary.yaml"))["base_dir"]) / grid_files[0].stem
    meta = yaml.safe_load(open(run_dir / "metadata.yaml"))
    K, B, r, sigma, T = (meta["contract"][k] for k in ("K", "B", "r", "sigma", "T"))
    epsilon = float(re.search(r"_eps([0-9.]+)_seed", grid_files[0].stem).group(1))
    s_grid, t_grid = first["s_grid"].double(), first["t_grid"].double()
    tau_grid = T - t_grid
    corner_window = first["corner_window"]
    cell_area = first["cell_area"]
    logger.info(f"  Contract K={K:g} B={B:g} r={r:g} sigma={sigma:g} T={T:g}; epsilon={epsilon:g}; "
                f"grid {len(s_grid)}x{len(t_grid)}; corner window {corner_window:g}")

    # ---- closed-form check of the quadrature: full payoff must give V_DO ----
    x_lo, x_hi = float(torch.log(torch.tensor(B, dtype=torch.float64))), float(torch.log(torch.tensor(K, dtype=torch.float64)))
    price_quadrature = down_and_out_price_by_quadrature(lambda s: (K - s).clamp(min=0.0), s_grid, tau_grid,
                                                        B, r, sigma, x_lo, x_hi, args.n_quad)
    ss, tt = torch.meshgrid(s_grid, t_grid, indexing="ij")
    reference = reiner_rubinstein_down_and_out_put(ss, K, B, r, sigma, T - tt)
    check_mask = (T - tt) >= 0.05  # the quadrature grid cannot resolve the kernel as tau -> 0
    check = float((price_quadrature - reference)[check_mask].abs().max() / reference[check_mask].abs().max())
    logger.info(f"  Quadrature check (full payoff vs Reiner-Rubinstein, tau >= 0.05): max rel discrepancy {check:.2e}")

    # ---- predicted floor from the terminal-trace defect ----
    def defect(s):
        return -(1.0 - _smoothstep01((s - B) / epsilon)) * (K - s).clamp(min=0.0)
    floor = down_and_out_price_by_quadrature(defect, s_grid, tau_grid, B, r, sigma,
                                             x_lo, float(torch.log(torch.tensor(B + epsilon, dtype=torch.float64))),
                                             args.n_quad)
    torch.save({"s_grid": s_grid, "t_grid": t_grid, "predicted_floor": floor, "reference": reference,
                "epsilon": epsilon, "contract": meta["contract"]}, out_dir / "predicted_floor_grid.pt")

    corner = (ss - B).abs() + (T - tt) <= corner_window
    edges = list(args.s_band_edges)
    edges[-1] = max(edges[-1], float(s_grid.max()))
    bands = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = (ss >= lo) & ((ss < hi) if hi < edges[-1] else (ss <= hi)) & ~corner
        bands[f"[{lo:g}, {hi:g}]"] = inside
    whole = ~corner

    def rel(values, mask):
        d = _grid_l2_norm(reference, mask, cell_area)
        return _grid_l2_norm(values, mask, cell_area) / d if d > 0 else float("nan")

    predicted = {name: rel(floor, mask) for name, mask in bands.items()}
    predicted["outside corner"] = rel(floor, whole)
    logger.info("  Predicted floor (terminal-trace defect alone), relative L2 per band: "
                + ", ".join(f"{k}: {v:.3e}" for k, v in predicted.items()))

    # ---- measured errors and remainder per configuration ----
    results: dict = {}
    for path in grid_files:
        g = torch.load(path, weights_only=False)
        m = re.search(r"_seed(\d+)(_[a-z0-9._]+?)?_nocorner", path.stem)
        seed, tag = int(m.group(1)), (m.group(2) or "")
        configuration = {"": "raw", "_blackscholes": "blackscholes", "_blackscholes_analyticres": "blackscholes_analyticres"}.get(
            tag, "split" if tag.startswith("_split") else tag)
        error = g["learned"].double() - g["reference"].double()
        remainder = error - floor
        entry = {"measured": {}, "remainder": {}, "run": path.stem}
        for name, mask in list(bands.items()) + [("outside corner", whole)]:
            entry["measured"][name] = rel(error, mask)
            entry["remainder"][name] = rel(remainder, mask)
        results.setdefault(configuration, {})[seed] = entry

    lines = [f"# Predicted corner-layer floor vs measured error — {aggregation_dir.name}", "",
             f"Predicted floor: propagation of the terminal-trace defect delta = -(1 - zeta)(K-s)^+ on [B, B+eps], eps = {epsilon:g}, "
             f"by the reflection-principle quadrature (check against Reiner-Rubinstein with the full payoff: {check:.1e}).", "",
             "Relative L2 error per band of s (all t, corner window removed). Measured: median over seeds [min, max]; "
             "remainder = measured error minus predicted floor.", "",
             "| Quantity | " + " | ".join(list(bands) + ["outside corner"]) + " |",
             "|---|" + "|".join("---" for _ in range(len(bands) + 1)) + "|",
             "| Predicted floor (no network needed) | " + " | ".join(f"{predicted[k]:.3e}" for k in list(bands) + ["outside corner"]) + " |"]
    for configuration, per_seed in results.items():
        for key, label in (("measured", "measured"), ("remainder", "measured − predicted")):
            cells = []
            for name in list(bands) + ["outside corner"]:
                v = [e[key][name] for e in per_seed.values()]
                cells.append(f"{statistics.median(v):.3e} [{min(v):.3e}, {max(v):.3e}]")
            lines.append(f"| {configuration}, {label} | " + " | ".join(cells) + " |")
    (out_dir / "predicted_vs_measured.md").write_text("\n".join(lines) + "\n")
    with open(out_dir / "predicted_vs_measured.yaml", "w") as f:
        yaml.dump({"aggregation_dir": str(aggregation_dir), "epsilon": epsilon, "contract": meta["contract"],
                   "quadrature_check_max_rel": check, "predicted_floor": predicted, "per_configuration_per_seed": results},
                  f, default_flow_style=False, sort_keys=False)
    for line in lines[6:]:
        logger.info("  " + line)

    # ---- figure: relative error per band, predicted (dashed) vs measured (solid) ----
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    names = list(bands)
    positions = list(range(len(names)))
    ax.plot(positions, [predicted[n] for n in names], "--", color="black", marker="x", label="Predicted floor (terminal-trace defect alone)")
    colors = {"blackscholes": "tab:blue", "blackscholes_analyticres": "tab:orange", "split": "tab:green", "raw": "tab:grey"}
    for configuration, per_seed in results.items():
        for e in per_seed.values():
            ax.scatter(positions, [e["measured"][n] for n in names], s=14, alpha=0.3, color=colors.get(configuration))
        ax.plot(positions, [statistics.median(e["measured"][n] for e in per_seed.values()) for n in names], "-",
                marker="o", color=colors.get(configuration), label=f"Measured, {configuration} (median over seeds)")
    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"$s\\in{n}$" for n in names])
    ax.set_ylabel("Relative $L^2$ error on the band (corner window removed)")
    ax.set_title(f"Predicted corner-layer floor vs measured error\n{aggregation_dir.name}", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    legend = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.subplots_adjust(left=0.13, right=0.6, top=0.88, bottom=0.3)
    finalize_figure(fig, out_dir / "figures" / "predicted_vs_measured_per_band.png", legends=[legend],
                    formula=FORMULA_TEXT, axes=[ax], formula_fontsize=6.0)
    logger.info(f"  Saved -> {out_dir}")


if __name__ == "__main__":
    main()
