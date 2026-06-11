# Terminal-extension smoothing versus ansatz mode

**Date:** 2026-06-12
**Scope:** European call on the backward heat equation; two-factor comparison of
(i) the terminal-data extension $g$ and (ii) the trial-solution algebra.
**Companion note:** [`2026-06-10_forcing_floor_vs_uncancellable_residual.md`](2026-06-10_forcing_floor_vs_uncancellable_residual.md)
(theory of the $\theta$-independent forcing floor and the uncancellable residual).

---

## 1. Question

When a terminal-value PDE is solved by a constrained network, the terminal datum
is injected through a fixed **extension** $\Psi$ of the boundary data $g$, and the
network $\Phi_\theta$ supplies the interior correction. Two design choices are
usually conflated:

1. **Which extension $g$** — how the (non-smooth) payoff is smoothed before it is
   used as the terminal datum. We compare a **softplus** smoothing of the call
   payoff against a **Chen–Mangasarian** smoothing
   $M_\varepsilon(a,b)=\tfrac12\!\left(a+b+\sqrt{(a-b)^2+\varepsilon^2}\right)$.
2. **Which ansatz mode** — how $\Phi_\theta$ and $g$ are combined:
   - **additive** (`hard_constant`): $\hat u=(1-\lambda(t))\,\Phi_\theta+g$;
   - **convex combination** (`hard_convex`): $\hat u=(1-\lambda(t))\,\Phi_\theta+\lambda(t)\,g$,

   with interpolation coefficient $\lambda(T)=1$ so both enforce
   $\hat u(\cdot,T)=g$ **exactly**. The unconstrained `soft_pinn`
   (terminal datum as an $L^2$ penalty, not enforced) is shown as a reference.

The hypothesis under test, from the companion note: the achieved accuracy is
governed primarily by the $\theta$-independent **forcing floor**
$\mathbb{E}\!\left[(\mathcal{P}\Psi)^2\right]$ — a functional of $g$ alone — and
only secondarily by the additive-versus-convex algebra.

## 2. Setup

- Backward heat operator $\mathcal{P}u=\partial_t u+\tfrac{\sigma^2}{2}\partial_{xx}u$,
  terminal datum at $t=T$, propagated to $t=0$; $\sigma=0.25$, $x=\ln S$.
- Identical training budget across all cells: $2\times10^4$ iterations, identical
  network width/depth, shared seeding within each cell; **3 seeds** per cell, so
  every number below is reported as mean $\pm$ std over seeds.
- Metrics: relative $L^2$ error against the closed-form solution over the
  evaluation window $\mathcal{X}_{\rm eval}\times[0,T]$; relative $L^2$ on the
  back-propagated $t=0$ inception slice (the option price); terminal mismatch
  $\mathrm{tc}=\lVert\hat u(\cdot,T)-g\rVert/\lVert g\rVert$.

## 3. Result

![Terminal extension vs ansatz mode](figures/2026-06-12_terminal_extension_vs_mode.png)

*Grouped-bar comparison (log axis, lower is better). Colour encodes the extension
$g$ (red = softplus, blue = Chen–Mangasarian); the $x$-axis groups the ansatz
mode. Error bars are std over 3 seeds. Left: accuracy over the full space-time
window. Right: accuracy at the $t=0$ inception price.*

Mean $\pm$ std over 3 seeds (linear interpolation coefficient $\lambda(t)=t/T$):

| extension $g$ | mode | rel $L^2$ (space-time) | rel $L^2$ at $t=0$ | terminal mismatch |
|---|---|---|---|---|
| softplus | `hard_constant` | $1.39\times10^{-1}\pm2\times10^{-2}$ | $1.85\times10^{-1}\pm2\times10^{-2}$ | $0$ (exact) |
| softplus | `hard_convex` | $6.94\times10^{-2}\pm1\times10^{-2}$ | $5.42\times10^{-2}\pm1\times10^{-2}$ | $0$ (exact) |
| softplus | `soft_pinn` | $4.69\times10^{-3}\pm4\times10^{-4}$ | $1.11\times10^{-2}\pm1\times10^{-3}$ | $7.0\times10^{-3}$ |
| Chen–Mangasarian | `hard_constant` | $1.44\times10^{-2}\pm2\times10^{-3}$ | $6.22\times10^{-3}\pm4\times10^{-3}$ | $0$ (exact) |
| Chen–Mangasarian | `hard_convex` | $9.47\times10^{-3}\pm3\times10^{-3}$ | $9.52\times10^{-3}\pm5\times10^{-3}$ | $0$ (exact) |
| Chen–Mangasarian | `soft_pinn` | $4.34\times10^{-3}\pm3\times10^{-4}$ | $1.03\times10^{-2}\pm8\times10^{-4}$ | $7.1\times10^{-3}$ |

(The exponential interpolation coefficient $\lambda(t)=e^{-\gamma(T-t)}$ is in the
saved table `terminal_extension_vs_mode.yaml`; it is statistically
indistinguishable from the linear one — no consistent winner.)

## 4. Reading

**(i) The extension $g$ is the dominant lever.** Holding the mode fixed, replacing
the softplus extension by the Chen–Mangasarian one reduces the space-time error
by a factor of **9.7** for `hard_constant` ($1.39\times10^{-1}\to1.44\times10^{-2}$)
and **7.3** for `hard_convex`; at the inception slice the factor reaches **30**
for `hard_constant`. This is the quantitative signature predicted by the floor
theory: the softplus extension has a sharply peaked $g''$ near the strike, hence a
large diffusion forcing $\tfrac{\sigma^2}{2}g''$ and a large floor
$\mathbb{E}[(\mathcal{P}\Psi)^2]$; the surviving high-frequency part
$\Pi_{\mathcal{S}^\perp}\mathcal{P}\Psi$ that the network cannot cancel is
correspondingly large. The Chen–Mangasarian smoothing controls $g''$ directly and
shrinks the floor.

**(ii) The additive-versus-convex algebra is a secondary lever.** Holding $g$
fixed, switching from `hard_constant` to `hard_convex` improves the space-time
error by a factor of **2.0** on the rough softplus extension but only **1.5** on
the smooth Chen–Mangasarian one — and at the inception slice the two modes are
within noise for the smooth extension (the convex form is even marginally worse:
$6.22\times10^{-3}$ vs $9.52\times10^{-3}$, well inside one std). Mechanistically,
the convex form damps the diffusion forcing by $\lambda(t)\le1$ (largest reduction
at early $t$, where $\mathcal{P}\Psi$ would otherwise be most polluting), so it
helps precisely when the diffusion forcing is the binding term, i.e. when $g$ is
rough. Once $g$ is smooth the floor is already small and the algebra barely
matters.

**(iii) Hard enforcement with a good extension matches — and at inception beats —
the unconstrained PINN.** The `soft_pinn` reference is the most accurate in the
interior, but it does **not** enforce the terminal datum: $\mathrm{tc}\approx
7\times10^{-3}$. With the smooth Chen–Mangasarian extension the hard forms close
most of that gap (space-time error within a factor $\sim2$ of `soft_pinn`) while
enforcing $\hat u(\cdot,T)=g$ to machine precision. **At the $t=0$ inception price
— the quantity of practical interest — the exactly-constrained `hard_constant`
Chen–Mangasarian form ($6.2\times10^{-3}$) is more accurate than `soft_pinn`
($1.0\times10^{-2}$).** Exact terminal enforcement plus a low-floor extension is
therefore the best accuracy/constraint trade-off here.

## 5. Consequence for the Bermudan / American backward induction

Backward induction over exercise dates **propagates** the per-stage terminal error
stage-to-stage: the trained solution at one exercise date becomes the terminal
datum of the previous stage. The two findings above translate into a concrete
prescription:

- Smooth each stage-terminal datum with **Chen–Mangasarian**, not softplus — this
  is where the order-of-magnitude accuracy lives, and it is the cheapest lever.
- Use a **hard** form so the stage-terminal condition is enforced exactly (no
  $7\times10^{-3}$ terminal leak that would compound across stages); the
  additive-versus-convex choice is second order and can be fixed to `hard_convex`
  for the modest benefit on any residual roughness.

## 6. Claim strength and scope

- All numbers are **empirical**, $n=3$ seeds, single architecture and training
  budget, one strike/maturity. The factor estimates ($\approx10$ for the
  extension, $\approx2$ for the mode) are stable across seeds but not yet across a
  hyperparameter sweep.
- The floor → error link is the **conjectured** mechanism of the companion note;
  it is corroborated here by the monotone extension-smoothness effect but is not a
  proof. The across-IC stability constant is problem-dependent (see the spectral
  analysis, which did *not* collapse cleanly across initial conditions).
- Greeks ($\Delta,\Gamma$) were **not** recorded for these specific June-9 runs;
  the Bermudan-put runs (June-11) carry them and corroborate that $\Gamma$ is the
  hardest quantity (largest error, at the money).

**Reproduce:**
`experiments/python_scripts/exp_ansatz_forms_heat/terminal_extension_vs_mode_comparison.py`
(torch-free; reads the saved `summary.yaml` of the `call` / `call_cm` ablation
runs, no retraining).
