r"""Empirical confirmation of the forcing-floor scaling law (Proposition 8.5).

The operator-channel floor of a mollified first-derivative discontinuity obeys, for
a second-order generator ($\mathcal{L}^X=\tfrac{\sigma^2}{2}\partial_{xx}$, order
$2p=2$) acting on a first-derivative discontinuity (regularity index $r=1$),

    floor(eps) = || (sigma^2/2) g_eps'' ||^2  ~  eps^{-(4p-2r-1)} = eps^{-1}.

Both the softplus (effective width ~ 1/beta) and the Chen--Mangasarian (width eps)
smoothings of the call first-derivative discontinuity are swept; the operator-channel energy is computed by
exact second differentiation on a grid refined to resolve the spike
(dx <= eps/8), and the log-log slope is fitted. The prediction is a slope of -1.

Torch-free apart from autograd for the second derivative. This backs the analytic
Proposition with a measurement, kept separate from it (methodology: split the
provable law from the measured constant).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _figure_layout import finalize_figure  # noqa: E402
from learning_option_pricing.utils.run_context import (  # noqa: E402
    script_data_dir,
    utc_timestamp,
)

K = 100.0
SIGMA = 0.25
X_STRIKE = math.log(K)


def _operator_channel_energy(g_of_x, eps: float) -> float:
    """Return the integral of ((sigma^2/2) g'')^2 over a window around the strike,
    isolating the singular (spike) part of the operator channel.

    The mollified first-derivative discontinuity is a spike in g'' whose width in the log-price coordinate
    is ~ eps/K (the smoother acts on z = e^x - K, and z ~ K (x - x_strike) near the
    strike, so a width-eps feature in z is width-eps/K in x). The window is taken
    proportional to eps/K so the integral is spike-dominated rather than
    contaminated by the O(1), eps-independent curvature of the payoff ramp; the
    grid resolves the spike with dx ~ (eps/K)/25.
    """
    spike_x = eps / K                    # spike width in the log-price coordinate
    half_window = 40.0 * spike_x         # cover ~40 spike widths (captures the tail)
    n = 4000                             # dx = 80 spike_x / 4000 = spike_x/50
    x = torch.linspace(X_STRIKE - half_window, X_STRIKE + half_window, n,
                       dtype=torch.float64, requires_grad=True)
    g = g_of_x(x)
    (gx,) = torch.autograd.grad(g.sum(), x, create_graph=True)
    (gxx,) = torch.autograd.grad(gx.sum(), x)
    op = 0.5 * SIGMA**2 * gxx.detach()
    dx = (2 * half_window) / (n - 1)
    return float((op**2).sum() * dx)  # trapezoidal integral of the squared channel


def _softplus_datum(eps: float):
    beta = 1.0 / eps  # effective smoothing width ~ 1/beta
    return lambda x: F.softplus(torch.exp(x) - K, beta=beta) - math.log(2.0) / beta


def _cm_datum(eps: float):
    # Chen--Mangasarian one-sided smoothing of (e^x - K)^+ at bandwidth eps
    return lambda x: 0.5 * ((torch.exp(x) - K) + torch.sqrt((torch.exp(x) - K) ** 2 + eps**2))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eps-min", type=float, default=1e-3)
    p.add_argument("--eps-max", type=float, default=1e-1)
    p.add_argument("--n-eps", type=int, default=13)
    args = p.parse_args(argv)

    epsilons = np.logspace(math.log10(args.eps_max), math.log10(args.eps_min), args.n_eps)
    out = script_data_dir(__file__) / utc_timestamp()
    out.mkdir(parents=True, exist_ok=True)

    curves = {"softplus": _softplus_datum, "Chen--Mangasarian": _cm_datum}
    colors = {"softplus": "#d1495b", "Chen--Mangasarian": "#1b6ca8"}
    floors = {}
    print(f"{'smoother':18s} {'slope (fit)':>12s}   [prediction: -1]")
    for name, maker in curves.items():
        fl = np.array([_operator_channel_energy(maker(e), e) for e in epsilons])
        floors[name] = fl
        # fit log floor vs log eps
        slope, intercept = np.polyfit(np.log(epsilons), np.log(fl), 1)
        print(f"{name:18s} {slope:>12.3f}")
        for e, f in zip(epsilons, fl):
            print(f"    eps={e:.2e}  floor={f:.4e}")

    # ---- figure: floor vs eps (log-log) with the eps^{-1} reference ----
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for name, fl in floors.items():
        slope, _ = np.polyfit(np.log(epsilons), np.log(fl), 1)
        ax.loglog(epsilons, fl, "-o", color=colors[name], lw=1.6, ms=4,
                  label=f"{name} (slope ${slope:.2f}$)")
    # eps^{-1} reference (dashed), anchored to the softplus curve
    ref = floors["softplus"][0] * (epsilons / epsilons[0]) ** (-1.0)
    ax.loglog(epsilons, ref, "--", color="black", lw=1.2, label=r"$\varepsilon^{-1}$ (predicted)")
    ax.set_xlabel(r"smoothing bandwidth $\varepsilon$")
    ax.set_ylabel(r"operator-channel floor $\|\frac{\sigma^2}{2}g_\varepsilon''\|^2$")
    ax.set_title(r"Forcing-floor scaling: $\|\mathcal{L}^X g_\varepsilon\|^2\sim\varepsilon^{-1}$",
                 fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    leg = ax.legend(loc="upper right", fontsize=9, frameon=True)
    fig.tight_layout()
    finalize_figure(
        fig, out / "forcing_floor_scaling.png", legends=[leg], axes=[ax],
        formula=(r"first-derivative discontinuity ($r=1$) under a second-order operator ($p=1$): "
                 r"$\|\mathcal{L}^X g_\varepsilon\|^2\sim\varepsilon^{-(4p-2r-1)}=\varepsilon^{-1}$; "
                 r"both smoothers measured on grids refined to resolve the spike"),
        formula_fontsize=8)
    print(f"\nwrote figure to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
