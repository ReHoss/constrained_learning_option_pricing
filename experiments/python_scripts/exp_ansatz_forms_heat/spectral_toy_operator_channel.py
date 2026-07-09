r"""Analytic toy illustrating the operator channel / spectral-gap mechanism.

This is the fully solvable companion to the empirical spectral analysis of
``spectral_gap_analysis.py``.  It strips the problem down to a periodic
backward-heat equation on the circle, where every Fourier mode decouples and the
network is replaced by its *idealised* spectral behaviour (an ideal low-pass
filter).  No training and no neural network is involved; every quantity below is
closed form.

Setup.  On the circle :math:`x\in[0,2\pi)`, with :math:`\mathcal{L}=\tfrac{\sigma^2}{2}\partial_{xx}`,

.. math::

    \mathcal{P}u=\partial_t u+\mathcal{L}u=0,\qquad u(\cdot,T)=g=\sum_k\hat g_k e^{ikx}.

The exact solution is mode-wise decay in :math:`\tau=T-t`,
:math:`u^\star_k(t)=\hat g_k\,e^{-\frac{\sigma^2}{2}k^2(T-t)}`.  The hard-constrained
trial solution is :math:`\hat u=(1-\lambda(t))\Phi+\lambda(t)g` with
:math:`\lambda(t)=t/T` (so :math:`\lambda(T)=1`), and the network is modelled as an
**ideal low-pass filter**: it can represent (and therefore cancel) any spatial
content with wavenumber :math:`|k|\le k^\star`, and nothing above.

Consequences, mode by mode (derived in the report, checked numerically here):

* :math:`|k|\le k^\star`:  the filter reproduces the exact mode, residual :math:`=0`.
* :math:`|k|>k^\star`:  :math:`\Phi_k=0`, so :math:`\hat u_k(t)=\lambda(t)\hat g_k` and

  .. math::

     \text{residual}_k(t)=\Bigl(\underbrace{\lambda'(t)}_{\text{velocity}}
        -\underbrace{\tfrac{\sigma^2}{2}k^2\lambda(t)}_{\text{operator channel}}\Bigr)\hat g_k,
     \qquad
     \text{error}_k(t)=\bigl(\lambda(t)-e^{-\frac{\sigma^2}{2}k^2(T-t)}\bigr)\hat g_k.

Three terminal data with controlled spectra are compared:

* ``band_limited``  -- a few low modes; :math:`\hat g_k=0` above a finite wavenumber
  (an *exact* spectral gap);
* ``smooth``        -- :math:`g=e^{\cos x}` (analytic; :math:`\hat g_k` decays
  super-exponentially);
* ``nonsmooth``     -- a triangle wave (:math:`|\hat g_k|^2\sim k^{-4}`; the
  regularised-payoff analogue, a first-derivative discontinuity).

Figures (saved under ``data/spectral_toy_operator_channel/<timestamp>/``):

* ``gap_vs_cutoff.png``   -- residual energy and solution error as a function of the
  filter cutoff :math:`k^\star`: performance is set by the forcing mass above
  :math:`k^\star`.  The band-limited datum drops sharply, the smooth datum decays
  fast, the non-smooth datum has a slow power-law tail.
* ``channel_amplification.png`` -- the datum spectrum :math:`|\hat g_k|^2` and the
  operator-channel spectrum :math:`(\tfrac{\sigma^2}{2}k^2)^2|\hat g_k|^2`: the
  :math:`k^4` amplification turns the non-smooth datum's decaying spectrum into a
  flat white plateau, and leaves the smooth datum decaying.
* ``residual_vs_error.png`` -- (left) a single high mode's residual and error over
  time: the residual maps to the error through the operator inverse (heat
  propagation), which damps high :math:`k`, so the error peaks in the interior and
  vanishes at :math:`t=T`; (right) aggregate error vs residual traced over time is
  a curve, not a line -- the residual-to-error constant depends on :math:`\tau`,
  which is why the empirical error predictor did not collapse across problems.
"""
from __future__ import annotations

import argparse
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

N = 4096          # spatial grid on the circle (power of two for the FFT)
SIGMA = 0.25      # diffusion scale (matches the empirical study); operator (sigma^2/2) d_xx
T = 1.0           # terminal time
KSTAR_DEFAULT = 12  # ideal low-pass cutoff (network reach), in wavenumbers


# ---------------------------------------------------------------------------
# Terminal data with controlled spectra
# ---------------------------------------------------------------------------

def _terminal_data(x: np.ndarray) -> dict:
    """Return {name: g(x)} for the three controlled terminal data."""
    # band-limited: finite Fourier support (exact spectral gap above k=5)
    band = np.cos(3 * x) + 0.5 * np.cos(5 * x)
    # smooth analytic: Fourier coefficients are modified Bessel I_k(1) ~ 1/(2^k k!)
    smooth = np.exp(np.cos(x)) - np.i0(1.0)  # subtract mean (I_0(1)) so hat g_0 = 0
    # non-smooth: triangle wave, C^0 with slope jumps -> |hat g_k|^2 ~ k^{-4}
    nonsmooth = (2.0 / np.pi) * np.arcsin(np.sin(x))
    return {"band_limited": band, "smooth": smooth, "nonsmooth": nonsmooth}


# ---------------------------------------------------------------------------
# Closed-form spectral quantities
# ---------------------------------------------------------------------------

def _modes(N: int) -> np.ndarray:
    """Integer wavenumbers for a length-N FFT on [0, 2*pi)."""
    return np.fft.fftfreq(N, d=1.0 / N).astype(float)  # 0, 1, ..., -1


def _lam(t: float) -> float:
    return t / T


def _lam_dot(t: float) -> float:
    return 1.0 / T


def _residual_hat(ghat, k, t, kstar):
    """Per-mode residual coefficient of the trial solution (Fourier)."""
    hi = np.abs(k) > kstar
    f = (_lam_dot(t) - 0.5 * SIGMA**2 * k**2 * _lam(t)) * ghat
    return np.where(hi, f, 0.0)


def _error_hat(ghat, k, t, kstar):
    """Per-mode solution-error coefficient (trial minus exact), Fourier."""
    hi = np.abs(k) > kstar
    decay = np.exp(-0.5 * SIGMA**2 * k**2 * (T - t))
    e = (_lam(t) - decay) * ghat
    return np.where(hi, e, 0.0)


def _energy(coeff_hat) -> float:
    """Spatial-mean L2 energy from (unnormalised numpy FFT) Fourier coefficients.

    Parseval for numpy's forward FFT: ``sum_n |f_n|^2 = (1/N) sum_k |fhat_k|^2``,
    so the spatial mean ``(1/N) sum_n |f_n|^2 = (1/N^2) sum_k |fhat_k|^2``.
    """
    return float(np.sum(np.abs(coeff_hat) ** 2) / coeff_hat.size**2)


def _exact_hat(ghat, k, t):
    return ghat * np.exp(-0.5 * SIGMA**2 * k**2 * (T - t))


# ---------------------------------------------------------------------------
# Numerical self-check of the closed-form residual
# ---------------------------------------------------------------------------

def _self_check(ghat, k, kstar) -> float:
    """Compare the closed-form residual against a direct finite-difference-in-time
    + spectral-second-derivative evaluation of P applied to the trial solution.

    Returns the max relative discrepancy over a few interior times (should be ~0).

    NOTE: this exercises the operator-channel term of ``_residual_hat`` only for
    modes above the cutoff.  Data with no spectral mass above the cutoff (the
    band-limited and smooth cases) therefore test essentially only the low-mode
    branch; the operator-channel coefficient and sign are tested for every run by
    the dedicated :func:`_operator_channel_probe`.
    """
    def u_hat_of_t(t):  # trial solution coefficients: exact for low k, lambda*g for high k
        hi = np.abs(k) > kstar
        return np.where(hi, _lam(t) * ghat, _exact_hat(ghat, k, t))

    worst = 0.0
    dt = 1e-4
    for t in (0.2 * T, 0.5 * T, 0.8 * T):
        uh = u_hat_of_t(t)
        # d/dt via central difference; d^2/dx^2 via spectral (-k^2)
        dudt = (u_hat_of_t(t + dt) - u_hat_of_t(t - dt)) / (2 * dt)
        lap = -(k**2) * uh
        p_u_numeric = dudt + 0.5 * SIGMA**2 * lap
        p_u_closed = _residual_hat(ghat, k, t, kstar)
        # Normalise the discrepancy by a fixed problem scale (the datum coefficient
        # magnitude), not by the residual itself: when the extension is fully
        # cancellable (band-limited / smooth) the residual is genuinely ~0, and a
        # residual-relative norm would spuriously amplify a harmless finite-difference error.
        scale = max(np.max(np.abs(p_u_closed)), np.max(np.abs(ghat)), 1e-12)
        worst = max(worst, float(np.max(np.abs(p_u_numeric - p_u_closed)) / scale))
    return worst


def _operator_channel_probe(k, kstar, k_probe=25) -> float:
    """Always-on test of the operator-channel coefficient and sign.

    Places unit synthetic mass at a single wavenumber ``k_probe > kstar`` (so the
    high-mode branch of :func:`_residual_hat` is genuinely exercised regardless of
    the terminal datum), and checks the closed-form residual there against the same
    finite-difference-in-time + spectral-Laplacian evaluation.  A sign or factor
    error in the operator-channel term ``(sigma^2/2) k^2 lambda`` makes this fail.
    """
    assert k_probe > kstar, "probe wavenumber must lie above the cutoff"
    ghat = np.zeros_like(k, dtype=complex)
    ghat[np.abs(k) == k_probe] = 1.0  # +k_probe and -k_probe
    return _self_check(ghat, k, kstar)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

COLORS = {"band_limited": "#2e7d32", "smooth": "#1b6ca8", "nonsmooth": "#d1495b"}
DISP = {"band_limited": "band-limited $g$ (exact gap)",
        "smooth": r"smooth $g=e^{\cos x}$",
        "nonsmooth": "non-smooth $g$ (triangle wave)"}
T_SLICES = np.array([0.1, 0.3, 0.5, 0.7, 0.9]) * T  # for time-averaged residual


def _fig_gap_vs_cutoff(ghats, k, out):
    """Residual energy and solution error vs the low-pass cutoff k*."""
    kstars = np.arange(1, 60)
    ts_err = np.linspace(0.05, 0.95, 19) * T  # time integration for the space-time error
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))
    for name, ghat in ghats.items():
        res, err = [], []
        u_norm2 = float(np.sum([_energy(_exact_hat(ghat, k, t)) for t in ts_err]))
        for ks in kstars:
            # left: time-averaged residual energy (the quantity the training objective penalises)
            r = float(np.mean([_energy(_residual_hat(ghat, k, t, ks)) for t in T_SLICES]))
            # right: relative L2 solution error over the space-time window
            e2 = float(np.sum([_energy(_error_hat(ghat, k, t, ks)) for t in ts_err]))
            res.append(r)
            err.append(np.sqrt(e2 / max(u_norm2, 1e-30)))
        axL.semilogy(kstars, np.clip(res, 1e-30, None), "-", color=COLORS[name],
                     lw=1.6, label=DISP[name])
        axR.semilogy(kstars, np.clip(err, 1e-30, None), "-", color=COLORS[name], lw=1.6)
    for ax, ttl in ((axL, "Uncancellable residual (training objective)"),
                    (axR, "Solution error over the space-time window")):
        ax.set_xlabel(r"low-pass cutoff $k^\star$ (network reach)")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_title(ttl, fontsize=10)
    axL.set_ylabel(r"$\|\Pi_{\mathcal{S}^\perp}\mathcal{P}\Psi\|^2$ (time-averaged)")
    axR.set_ylabel(r"relative $L^2$ error $\|\hat u-u^\star\|/\|u^\star\|$")
    handles, labels = axL.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
                     frameon=True, bbox_to_anchor=(0.5, 0.11))
    fig.suptitle("Performance is set by the forcing mass above the cutoff", fontsize=11)
    fig.tight_layout(rect=[0, 0.22, 1, 0.94])
    finalize_figure(
        fig, out / "gap_vs_cutoff.png", legends=[leg], axes=[axL, axR],
        formula=(r"band-limited $g$: residual and error drop sharply once $k^\star$ passes "
                 r"its top mode (exact gap). smooth $g$: both fall fast." "\n"
                 r"non-smooth $g$: the residual is a near-flat white plateau (operator channel "
                 r"$\sim k^4|\hat g_k|^2=$const), yet the error still decays — the operator "
                 r"inverse damps the uncancelled high-$k$ modes"),
        formula_fontsize=8)


def _fig_channel_amplification(ghats, k, out, kstar):
    """Datum spectrum vs operator-channel spectrum: the k^4 amplification."""
    order = np.argsort(k)
    kk = k[order]
    pos = kk > 0
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, name in zip(axes, ("nonsmooth", "smooth")):
        ghat = ghats[name][order]
        datum = np.abs(ghat) ** 2
        op = (0.5 * SIGMA**2 * kk**2) ** 2 * datum  # |operator channel|^2 ~ k^4 |ghat|^2
        # keep only modes that carry real content: the triangle wave has odd
        # harmonics only, so plotting the full comb would draw the even modes as
        # spikes down to the floor.  Masking to the nonzero modes shows the clean
        # envelope (the plateau) rather than a picket fence.
        sig = pos & (datum > 1e-13 * datum.max())
        ax.loglog(kk[sig], datum[sig], "--", color="#888888",
                  lw=1.4, label=r"datum $|\hat g_k|^2$")
        ax.loglog(kk[sig], op[sig], "-", color=COLORS[name],
                  lw=1.6, label=r"operator channel $|\widehat{\mathcal{L}g}_k|^2$")
        ax.axvline(kstar, ls=":", color="black", lw=1.0)
        ax.set_title(DISP[name], fontsize=10)
        ax.set_xlabel("wavenumber $k$")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_ylim(1e-12, 1e6)
    axes[0].set_ylabel("power")
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9,
                     frameon=True, bbox_to_anchor=(0.5, 0.12))
    fig.suptitle(r"The operator multiplies component $k$ by $-\frac{\sigma^2}{2}k^2$ "
                 r"(power $\times k^4$)", fontsize=11)
    fig.tight_layout(rect=[0, 0.16, 1, 0.94])
    finalize_figure(
        fig, out / "channel_amplification.png", legends=[leg], axes=list(axes),
        formula=(r"non-smooth: $|\hat g_k|^2\sim k^{-4}\Rightarrow$ operator channel "
                 r"$\sim k^4 k^{-4}=$ const to leading order (near-flat plateau, overhangs "
                 r"the cutoff $k^\star$, dotted); smooth: $|\hat g_k|^2$ decays fast enough "
                 r"that $k^4|\hat g_k|^2$ still decays (fits below $k^\star$)"),
        formula_fontsize=8)


def _fig_residual_vs_error(ghats, k, out, kstar):
    """Time structure: residual->error is the operator inverse (damps high k)."""
    ts = np.linspace(0.0, T, 200)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))

    # left: a single high mode's |residual| and |error| over time (non-smooth datum)
    ghat = ghats["nonsmooth"]
    k0 = 20  # a wavenumber above the cutoff
    idx = int(np.argmin(np.abs(k - k0)))
    g0 = ghat[idx]
    res_t = np.abs([(_lam_dot(t) - 0.5 * SIGMA**2 * k0**2 * _lam(t)) * g0 for t in ts])
    err_t = np.abs([(_lam(t) - np.exp(-0.5 * SIGMA**2 * k0**2 * (T - t))) * g0 for t in ts])
    # both curves are analytic (there is no trained model in this panel), so the
    # solid/dashed strokes here merely separate two analytic quantities rather than
    # marking model vs reference; magnitudes are plotted (the residual dip at
    # t~1/(sigma^2 k0^2/2) is a sign change, not a vanishing).
    axL.plot(ts, res_t / res_t.max(), "-", color="#d1495b", lw=1.6,
             label=r"$|$residual$|$ $|\widehat{\mathcal{P}\hat u}_{k_0}|$ (normalised)")
    axL.plot(ts, err_t / err_t.max(), "--", color="black", lw=1.6,
             label=r"$|$error$|$ $|\hat u_{k_0}-u^\star_{k_0}|$ (normalised)")
    axL.axvline(T, ls=":", color="grey", lw=1.0)
    axL.set_xlabel("time $t$ (terminal $T=1$ on the right)")
    axL.set_ylabel("normalised magnitude")
    axL.set_title(f"One high component $k_0={k0}$: error peaks in the interior, "
                  f"$=0$ at $t=T$", fontsize=9)
    axL.grid(True, alpha=0.3)
    legL = axL.legend(loc="upper left", bbox_to_anchor=(0.0, -0.16), fontsize=8, frameon=True)

    # right: aggregate error vs residual traced over time -> a curve, not a line
    ghat = ghats["nonsmooth"]
    res_agg, err_agg = [], []
    for t in ts:
        res_agg.append(np.sqrt(_energy(_residual_hat(ghat, k, t, kstar))))
        e_hat = _error_hat(ghat, k, t, kstar)
        u_hat = _exact_hat(ghat, k, t)
        err_agg.append(np.sqrt(_energy(e_hat) / max(_energy(u_hat), 1e-30)))
    sc = axR.scatter(res_agg, err_agg, c=ts, cmap="viridis", s=14)
    axR.set_xlabel(r"achieved residual $\|\Pi_{\mathcal{S}^\perp}\mathcal{P}\hat u\|$"
                   r" (training objective)")
    axR.set_ylabel(r"solution error $\|\hat u-u^\star\|/\|u^\star\|$")
    axR.set_title("Error vs residual over time: a curve, not a line", fontsize=9)
    axR.grid(True, alpha=0.3)
    cb = fig.colorbar(sc, ax=axR, fraction=0.046, pad=0.04)
    cb.set_label("time $t$")
    fig.suptitle(r"Residual $\to$ error map is the operator inverse "
                 r"$\mathcal{P}^{-1}$ (heat propagation, damps high $k$)", fontsize=11)
    fig.tight_layout(rect=[0, 0.14, 1, 0.94])
    finalize_figure(
        fig, out / "residual_vs_error.png", legends=[legL], axes=[axL, axR],
        formula=(r"error$_k(t)=(\lambda(t)-e^{-\frac{\sigma^2}{2}k^2(T-t)})\hat g_k$: at "
                 r"$t=T$ both terms are $\hat g_k$ (error $0$); the constant linking "
                 r"error to residual depends on $\tau=T-t$, so no single predictor holds"),
        formula_fontsize=8)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kstar", type=int, default=KSTAR_DEFAULT,
                   help="ideal low-pass cutoff (network reach) for the fixed-cutoff figures")
    args = p.parse_args(argv)

    x = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    k = _modes(N)
    g = _terminal_data(x)
    ghats = {name: np.fft.fft(gi) for name, gi in g.items()}

    out = script_data_dir(__file__) / utc_timestamp()
    out.mkdir(parents=True, exist_ok=True)

    # numerical self-check of the closed-form residual (must be ~0)
    print("numerical self-check (max rel. discrepancy of closed-form residual):")
    for name, ghat in ghats.items():
        disc = _self_check(ghat, k, args.kstar)
        status = "OK" if disc < 1e-4 else "FAIL"
        print(f"  {name:16s} {disc:.2e}  [{status}]")
    # the per-datum checks above only exercise the operator channel where the datum
    # has mass above the cutoff; this probe tests its coefficient/sign for every run
    probe = _operator_channel_probe(k, args.kstar)
    print(f"  {'operator-probe':16s} {probe:.2e}  [{'OK' if probe < 1e-4 else 'FAIL'}]  "
          f"(synthetic mass above the cutoff)")

    _fig_gap_vs_cutoff(ghats, k, out)
    _fig_channel_amplification(ghats, k, out, args.kstar)
    _fig_residual_vs_error(ghats, k, out, args.kstar)

    print(f"\nwrote figures to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
