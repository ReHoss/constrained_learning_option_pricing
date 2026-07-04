r"""Measured neural-network confirmation of the ideal-filter model.

The analytic toy (spectral_toy_operator_channel.py) models the network as an ideal
low-pass filter with a HARD cutoff. This script replaces the ideal filter by an
actual trained network on the same periodic backward-heat problem and measures its
achieved residual spectrum, to show the cutoff is realised as a SOFT roll-off ---
the network's spectral bias --- rather than a step. This turns the conjectured
neural-tangent-kernel identification (report Remark) into a measurement, kept
separate from the proven ideal-filter model.

Setup. On the circle x in [0, 2 pi), P u = d_t u + (sigma^2/2) d_xx u = 0,
terminal datum g. Hard-constrained ansatz u_hat = (1 - lambda(t)) Phi_theta + lambda(t) g,
lambda(t) = t/T, so u_hat(.,T) = g exactly. Phi_theta is a tanh MLP on the periodic
features [cos x, sin x, t/T]; its spectral bias (low wavenumbers learned first) is
what we measure. Training minimises the mean-squared PDE residual.

Measurement. On a fine grid at several time slices, the forcing f = P(lambda g) =
lambda' g + lambda (sigma^2/2) g'' and the achieved residual r = P u_hat are
FFT'd; the per-mode cancellation ratio |r_k|^2 / |f_k|^2 rises from ~0 (low k,
cancelled) to ~1 (high k, uncancelled), a soft transition whose knee is the
effective cutoff. Contrasted with the ideal filter's hard step.
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
import torch.nn as nn  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _figure_layout import finalize_figure  # noqa: E402
from learning_option_pricing.utils.run_context import (  # noqa: E402
    script_data_dir,
    utc_timestamp,
)

SIGMA = 0.25
T = 1.0
N_GRID = 1024
T_SLICES = (0.1, 0.3, 0.5, 0.7, 0.9)


def _terminal_data(name, n_modes=20):
    """Return a smooth, trainable periodic datum. The 'kink' datum is a triangle
    wave truncated to its first ``n_modes`` odd harmonics: it carries the kink's
    k^{-2} amplitude envelope (so its operator channel k^2 * ghat is flat up to the
    truncation, a white forcing to measure the network cutoff against) while being
    C^infinity, so g'' is bounded and the hard-ansatz residual is well posed. The
    raw triangle wave has a singular (Dirac) g'' and cannot be trained on directly."""
    if name == "kink":
        # seeded rough field with a kink envelope |ghat_k| ~ k^{-2} at EVERY
        # wavenumber up to k_max (not the odd-only triangle comb): its operator
        # channel k^2 * ghat is white up to k_max, defined at every k, so the
        # network's cancellation ratio can be read cleanly and probed past its reach.
        k_max = 150
        gg = torch.Generator().manual_seed(0)
        ks = torch.arange(1, k_max + 1, dtype=torch.float64)
        amp = ks ** (-2.0)
        phase = 2 * math.pi * torch.rand(k_max, generator=gg, dtype=torch.float64)

        def g(x):
            return (amp.to(x) * torch.cos(ks.to(x) * x.unsqueeze(-1) + phase.to(x))).sum(-1)
        return g
    if name == "smooth":
        return lambda x: torch.exp(torch.cos(x)) - float(np.i0(1.0))
    raise ValueError(name)


class PeriodicField(nn.Module):
    """tanh MLP on periodic features [cos x, sin x, t/T] -> scalar."""

    def __init__(self, width=64, depth=3):
        super().__init__()
        layers, d = [], 3
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]
            d = width
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        feat = torch.stack([torch.cos(x), torch.sin(x), t / T], dim=-1)
        return self.net(feat).squeeze(-1)


def _ansatz(field, g_fn, x, t):
    lam = t / T
    return (1.0 - lam) * field(x, t) + lam * g_fn(x)


def _residual(field, g_fn, x, t):
    """P u_hat = d_t u_hat + (sigma^2/2) d_xx u_hat via autograd."""
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    u = _ansatz(field, g_fn, x, t)
    (ux,) = torch.autograd.grad(u.sum(), x, create_graph=True)
    (uxx,) = torch.autograd.grad(ux.sum(), x, create_graph=True)
    (ut,) = torch.autograd.grad(u.sum(), t, create_graph=True)
    return ut + 0.5 * SIGMA**2 * uxx


def train(field, g_fn, iters, seed, device):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    opt = torch.optim.Adam(field.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)
    for it in range(iters):
        x = (2 * math.pi * torch.rand(2048, generator=gen)).to(device)
        t = (T * torch.rand(2048, generator=gen)).to(device)
        loss = (_residual(field, g_fn, x, t) ** 2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(field.parameters(), max_norm=1.0)
        if torch.isfinite(loss):
            opt.step()
        sched.step()
        if it % max(1, iters // 8) == 0 or it == iters - 1:
            print(f"  it={it:5d} loss={loss.item():.4e}")
    return field


def _spectra(field, g_fn, device):
    """Slice-averaged power spectra of the forcing and the achieved residual, and
    the per-mode cancellation ratio."""
    x = torch.linspace(0.0, 2 * math.pi, N_GRID + 1, device=device)[:-1]
    k = np.fft.rfftfreq(N_GRID, d=1.0 / N_GRID)
    fpow = np.zeros(len(k)); rpow = np.zeros(len(k))
    for frac in T_SLICES:
        t = torch.full_like(x, frac * T)
        lam = frac; lamp = 1.0 / T
        xx = x.clone().requires_grad_(True)
        g = g_fn(xx)
        (gx,) = torch.autograd.grad(g.sum(), xx, create_graph=True)
        (gxx,) = torch.autograd.grad(gx.sum(), xx)
        f = (lamp * g + lam * 0.5 * SIGMA**2 * gxx).detach().cpu().numpy()
        r = _residual(field, g_fn, x, t).detach().cpu().numpy()
        fpow += np.abs(np.fft.rfft(f - f.mean())) ** 2
        rpow += np.abs(np.fft.rfft(r - r.mean())) ** 2
    fpow /= len(T_SLICES); rpow /= len(T_SLICES)
    ratio = rpow / np.clip(fpow, 1e-30, None)
    return k, fpow, rpow, ratio


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datum", default="kink", choices=["kink", "smooth"])
    p.add_argument("--iters", type=int, default=8000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    g_fn = _terminal_data(args.datum)
    field = PeriodicField().to(device)
    print(f"training periodic field on the {args.datum} datum ({args.iters} iters)")
    train(field, g_fn, args.iters, args.seed, device)

    field.eval()
    k, fpow, rpow, ratio = _spectra(field, g_fn, device)
    # restrict to wavenumbers carrying forcing (the datum band); smooth the ratio
    # with a short running mean, then read the effective soft cutoff as the first
    # in-band wavenumber where the smoothed cancellation ratio reaches 1/2.
    band = fpow > 1e-5 * fpow.max()
    kmax_band = int(k[band].max()) if band.any() else int(k[-1])
    w = 7
    rsm = np.convolve(np.clip(ratio, 0.0, 1.5), np.ones(w) / w, mode="same")
    kstar = next((int(k[i]) for i in range(1, len(k)) if band[i] and rsm[i] >= 0.5), kmax_band)
    print(f"effective soft cutoff k* (smoothed ratio crosses 1/2) = {kstar} "
          f"(datum band up to k={kmax_band})")

    out = script_data_dir(__file__) / f"{utc_timestamp()}_{args.datum}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))
    axL.loglog(k[1:], np.clip(fpow[1:], 1e-30, None), "-", color="#d1495b", lw=1.5,
               label=r"forcing $\mathcal{P}\Psi$")
    axL.loglog(k[1:], np.clip(rpow[1:], 1e-30, None), "--", color="black", lw=1.4,
               label=r"achieved residual $\mathcal{P}\hat u$ (trained net)")
    axL.set_xlabel("spatial wavenumber $k$"); axL.set_ylabel(r"power $|\widehat{\cdot}_k|^2$")
    axL.set_title(f"Forcing vs achieved residual ({args.datum} datum)", fontsize=10)
    axL.grid(True, which="both", alpha=0.3)
    legL = axL.legend(loc="lower left", fontsize=8, frameon=True)

    bmask = band & (k > 0)
    axR.semilogx(k[bmask], np.clip(ratio[bmask], 0, 1.3), color="#9ecae1", lw=0.8, alpha=0.7,
                 label="measured (raw)")
    axR.semilogx(k[bmask], np.clip(rsm[bmask], 0, 1.3), "-", color="#1b6ca8", lw=1.8,
                 label="measured (smoothed)")
    axR.axhline(1.0, ls=":", color="grey", lw=1)
    axR.axvline(kstar, ls="--", color="#1b6ca8", lw=1, label=f"soft cutoff $k^\\star={kstar}$")
    # ideal hard filter for contrast: step from 0 to 1 at k*
    kb = k[bmask]
    axR.step(kb, (kb > kstar).astype(float), where="mid", color="black", lw=1.2,
             ls="--", label="ideal filter (hard step)")
    axR.set_xlim(1, kmax_band)
    axR.set_xlabel("spatial wavenumber $k$")
    axR.set_ylabel(r"cancellation ratio $|\widehat{\mathcal{P}\hat u}_k|^2/|\widehat{\mathcal{P}\Psi}_k|^2$")
    axR.set_title("Spectral bias: a soft cutoff, not a step", fontsize=10)
    axR.grid(True, which="both", alpha=0.3)
    legR = axR.legend(loc="lower right", fontsize=8, frameon=True)

    fig.suptitle("Trained network realises the ideal-filter cutoff as a soft roll-off",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    finalize_figure(
        fig, out / "spectral_bias_periodic_nn.png", legends=[legL, legR], axes=[axL, axR],
        formula=(r"trained hard-constrained network on the periodic backward heat: the "
                 r"achieved residual tracks the forcing above a soft cutoff (spectral bias), "
                 r"and lies far below it at low $k$ (cancelled) --- the ideal filter's hard "
                 r"step, softened"),
        formula_fontsize=8)
    print(f"\nwrote figure to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
