r"""Measured damping operator of the Bermudan error recursion.

The analytical model of the induction error is the recursion
    e_{j-1} = S_{dtau} Pi_j e_j + eps^solve_{j-1},
with S_{dtau} = exp(dtau L^X) the backward-heat propagation semigroup (an L^2
contraction that damps mode k by exp(-(sigma^2/2) k^2 dtau)) and Pi_j the
exercise (max-gluing, 1-Lipschitz). This script measures the composite damping
operator S Pi directly, with no learning: a known perturbation delta is injected
into the continuation value at one exercise date, the exact induction is continued
to inception, and the resulting perturbation at inception is measured. It confirms
that inherited error is damped, not amplified, and quantifies the frequency- and
distance-dependence the recursion predicts.

Two measurements:
  * frequency damping -- inject a single spatial mode delta = eps0 cos(k (x-x0))
    at the first intermediate date and measure the inception gain vs k; the
    prediction is the heat factor exp(-(sigma^2/2) k^2 t_inject) up to the
    exercise mask;
  * distance damping -- inject a fixed smooth perturbation at each date and measure
    the inception gain vs the injection time (propagation distance).

Exact throughout (Gaussian-convolution propagation via heat_propagate); torch on
CPU. Backs the analytical recursion with a measurement, kept separate from it.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _figure_layout import finalize_figure  # noqa: E402
from learning_option_pricing.pde import heat_propagate, heat_put_payoff  # noqa: E402
from learning_option_pricing.utils.run_context import (  # noqa: E402
    script_data_dir,
    utc_timestamp,
)

K = 100.0
SIGMA = 0.25
T = 1.0
Y_LO, Y_HI = math.log(5.0), math.log(600.0)      # propagation support
X_LO, X_HI = math.log(60.0), math.log(140.0)     # evaluation window
N_QUAD = 4000


def _interp(y_nodes, v_nodes):
    """Linear-interpolant callable of the grid function (v_nodes on y_nodes)."""
    def f(q):
        qc = q.clamp(float(y_nodes[0]), float(y_nodes[-1]))
        idx = torch.searchsorted(y_nodes, qc).clamp(1, len(y_nodes) - 1)
        y0, y1 = y_nodes[idx - 1], y_nodes[idx]
        v0, v1 = v_nodes[idx - 1], v_nodes[idx]
        w = (qc - y0) / (y1 - y0)
        return v0 + w * (v1 - v0)
    return f


def _induction_inception(y, exercise_times, perturb_stage=None, perturb_fn=None):
    """Exact backward induction on the grid y; optionally add perturb_fn to the
    continuation at exercise date index perturb_stage (1-based into exercise_times
    excluding maturity). Returns the inception value on y."""
    tau = [0.0] + list(exercise_times)          # [0, t_1, ..., t_m=T]
    m = len(exercise_times)
    payoff = heat_put_payoff(y, K)
    V = payoff.clone()                          # value at maturity t_m
    for k in range(m - 1, 0, -1):               # exercise dates t_{m-1} ... t_1
        cont = heat_propagate(_interp(y, V), y, torch.full_like(y, tau[k]),
                              t_terminal=tau[k + 1], sigma=SIGMA,
                              y_lo=Y_LO, y_hi=Y_HI, n_quad=N_QUAD)
        if perturb_stage is not None and k == perturb_stage:
            cont = cont + perturb_fn(y)
        V = torch.maximum(payoff, cont)
    # inception: propagate value at t_1 to t=0 (no exercise at 0)
    return heat_propagate(_interp(y, V), y, torch.zeros_like(y),
                          t_terminal=tau[1], sigma=SIGMA, y_lo=Y_LO, y_hi=Y_HI, n_quad=N_QUAD)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--m", type=int, default=10, help="number of exercise dates")
    p.add_argument("--amp", type=float, default=1e-3, help="perturbation amplitude (linear regime)")
    args = p.parse_args(argv)

    exercise_times = [T * j / args.m for j in range(1, args.m + 1)]
    n_grid = 1400
    y = torch.linspace(Y_LO, Y_HI, n_grid, dtype=torch.float64)
    win = (y >= X_LO) & (y <= X_HI)
    x_mid = 0.5 * (X_LO + X_HI)
    x_half = 0.5 * (X_HI - X_LO)

    base = _induction_inception(y, exercise_times)

    # --- frequency damping: single-mode injection at the first intermediate date ---
    inject_stage = 1                     # t_1 (nearest inception among exercise dates)
    t_inject = exercise_times[inject_stage - 1]
    # low-to-moderate wavenumbers, where the heat damping is above numerical noise
    wavenumbers = np.arange(1, 13)
    gains, predicted = [], []
    for kk in wavenumbers:
        def delta(q, kk=kk):
            # cosine under a smooth Gaussian window -> narrow spectrum around the
            # central wavenumber (a hard window would inject broadband edge content
            # that survives the propagation and floors the gain).
            xw = (q - x_mid) / x_half
            window = torch.exp(-(xw / 0.45) ** 2)
            return args.amp * torch.cos(kk * math.pi * xw) * window
        pert = _induction_inception(y, exercise_times, perturb_stage=inject_stage, perturb_fn=delta)
        dv = (pert - base)[win]
        d0 = delta(y)[win]
        gains.append(float(dv.norm() / d0.norm()))
        # predicted heat damping over the elapsed variance from t_inject to 0
        phys_k = kk * math.pi / x_half    # physical wavenumber of the windowed mode
        predicted.append(math.exp(-0.5 * SIGMA**2 * phys_k**2 * t_inject))

    # --- distance damping: fixed smooth perturbation injected at each date ---
    def smooth_delta(q):
        xw = (q - x_mid) / x_half
        window = torch.exp(-(xw / 0.45) ** 2)
        return args.amp * torch.cos(math.pi * xw) * window   # lowest mode, smoothly windowed
    inject_times, dist_gains = [], []
    for js in range(1, args.m):
        pert = _induction_inception(y, exercise_times, perturb_stage=js, perturb_fn=smooth_delta)
        dv = (pert - base)[win]; d0 = smooth_delta(y)[win]
        inject_times.append(exercise_times[js - 1])
        dist_gains.append(float(dv.norm() / d0.norm()))

    out = script_data_dir(__file__) / f"{utc_timestamp()}_m{args.m}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"m={args.m}, injection at t_1={t_inject:.3f}")
    for kk, g, pr in zip(wavenumbers, gains, predicted):
        print(f"  k={kk:2d}  gain={g:.3e}  predicted(heat)={pr:.3e}")

    # persist the measured gains so the figure (and any report table) can be
    # regenerated without recomputing the exact induction
    import yaml
    with open(out / "gains.yaml", "w") as f:
        yaml.safe_dump({
            "m": args.m, "amp": args.amp, "t_inject": float(t_inject),
            "frequency_gains": [
                {"k": int(kk), "gain": float(g), "predicted_heat": float(pr)}
                for kk, g, pr in zip(wavenumbers, gains, predicted)],
            "distance_gains": [
                {"t_inject": float(t), "gain": float(g)}
                for t, g in zip(inject_times, dist_gains)],
        }, f, sort_keys=False)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))
    axL.semilogy(wavenumbers, np.clip(gains, 1e-30, None), "-o", color="#1b6ca8",
                 lw=1.6, ms=4, label="measured gain (exact induction)")
    axL.semilogy(wavenumbers, np.clip(predicted, 1e-30, None), "--", color="black",
                 lw=1.3, label=r"$e^{-\frac{\sigma^2}{2}k^2 t_{\rm inject}}$ (heat damping)")
    axL.set_xlabel("perturbation wavenumber $k$")
    axL.set_ylabel(r"inception gain $\|\Delta V(\cdot,0)\|/\|\delta\|$")
    axL.set_title(f"Frequency damping (injected at $t_1={t_inject:.2f}$)", fontsize=10)
    axL.grid(True, which="both", alpha=0.3)
    legL = axL.legend(loc="lower left", fontsize=8, frameon=True)

    axR.plot(inject_times, dist_gains, "-o", color="#d1495b", lw=1.6, ms=4)
    axR.set_xlabel("injection time $t_{\\rm inject}$ (0 = inception, $T$ = maturity)")
    axR.set_ylabel(r"inception gain (low-mode $\delta$)")
    axR.set_title("Distance damping: a fixed smooth perturbation", fontsize=10)
    axR.grid(True, alpha=0.3)
    axR.set_ylim(0, 1.05)

    fig.suptitle(r"Measured damping operator $S\,\Pi$: inherited error is damped, not amplified",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    finalize_figure(
        fig, out / "perturbation_propagation.png", legends=[legL], axes=[axL, axR],
        formula=(r"a known $\delta$ injected into one continuation, read at inception through the "
                 r"exact induction: in the measurable range the gain tracks the heat factor "
                 r"$e^{-\frac{\sigma^2}{2}k^2 t}$ (semigroup $S$), plateauing at the probe's spectral "
                 r"floor where the true damping is below precision; a smooth perturbation is damped "
                 r"monotonically with distance --- every gain $<1$, inherited error never grows"),
        formula_fontsize=8)
    print(f"\nwrote figure to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
