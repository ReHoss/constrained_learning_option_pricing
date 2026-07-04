# Regularity, spectral decay, and the forcing floor under smoothing

**Date:** 2026-06-13
**Scope:** the spectral mechanism that links the *regularity of the terminal
datum* to the *magnitude and frequency content of the operator forcing*, and to
its $\varepsilon$-dependence under smoothing.
**Companion notes:**
[`2026-06-10_forcing_floor_vs_uncancellable_residual.md`](2026-06-10_forcing_floor_vs_uncancellable_residual.md)
(forcing floor / uncancellable residual) and
[`2026-06-12_terminal_extension_smoothing_vs_ansatz_mode.md`](2026-06-12_terminal_extension_smoothing_vs_ansatz_mode.md)
(softplus vs Chen–Mangasarian extension ablation). This note supplies the
decay-rate derivation that §2.5 of the first and §3.1/§5 of the second invoke
without proof.

> Math rendering: open in a Markdown + KaTeX/MathJax viewer.

---

## Part I — Abstract theory

The statements of Part I are classical Fourier analysis (proven); the link from
the resulting frequency content to the trainability of a neural field is the
**conjectured** NTK spectral-bias mechanism of the companion note, flagged where
it is used.

### I.1 The differentiation rule and the Dirac baseline

Fix the Fourier convention $\hat g(k)=\int_{\mathbb R} g(x)\,e^{-ikx}\,dx$.
Differentiation is multiplication by $ik$:

$$
\widehat{g^{(n)}}(k) = (ik)^n\,\hat g(k).
$$

The Dirac delta has a **flat** ("white") spectrum, $\hat\delta(k)=1$, i.e.
$|\hat\delta(k)|^2\sim k^0$ — equal power at every frequency. It is the
canonical maximally-unsmooth object and the reference against which decay rates
are measured.

### I.2 Regularity $\Rightarrow$ spectral decay

Let the **regularity index** $r$ be the order of the lowest discontinuous
derivative: $g\in C^{r-1}$ but $g^{(r)}$ has a jump (so $g^{(r+1)}$ contains a
Dirac at the singular point). Since a delta sits in $g^{(r+1)}$,
$\widehat{g^{(r+1)}}(k)\to\text{const}\neq0$ as $k\to\infty$; combining with the
differentiation rule $\widehat{g^{(r+1)}}(k)=(ik)^{r+1}\hat g(k)$ gives the tail

$$
\hat g(k)\sim \frac{\text{const}}{k^{\,r+1}},
\qquad\boxed{\,|\hat g(k)|^2\sim k^{-2(r+1)}.\,}
$$

| singularity | $r$ | $|\hat g(k)|^2$ |
|---|---|---|
| jump in $g$ (step) | $0$ | $k^{-2}$ |
| **kink** (jump in $g'$) | $1$ | $k^{-4}$ |
| jump in $g''$ | $2$ | $k^{-6}$ |
| $C^\infty$ / analytic | — | faster than any power (exponential) |

*Integration-by-parts reading.* Each smooth integration by parts in
$\hat g(k)=\int g\,e^{-ikx}\,dx$ buys one factor $1/(ik)$; the decay **stalls** at
the order $r+1$ where the first non-integrable derivative (the delta) appears,
fixing the amplitude tail at $k^{-(r+1)}$.

### I.3 Elliptic amplification

Let $\mathcal{L}$ be a constant-coefficient elliptic operator of order $2p$. Its
Fourier symbol grows like $|k|^{2p}$ (heat: $\mathcal{L}=\tfrac{\sigma^2}{2}\partial_{xx}$,
$2p=2$, symbol $\tfrac{\sigma^2}{2}k^2$). Hence

$$
|\widehat{\mathcal{L}g}(k)|^2 \sim |k|^{4p}\,|\hat g(k)|^2 \sim k^{\,4p-2(r+1)}.
$$

Applying $\mathcal{L}$ **reddens nothing and whitens everything**: it tilts the
spectrum up by $|k|^{4p}$, moving energy toward high frequencies. The net
exponent $4p-2r-2$ decides whether $\mathcal{L}g$ is square-integrable.

### I.4 Smoothing is a spectral cutoff at $k_{\max}\sim1/\varepsilon$

Replacing the singularity by a transition of spatial width $\varepsilon$ (a
mollifier at scale $\varepsilon$) leaves the spectrum essentially unchanged below
$k\sim1/\varepsilon$ and suppresses it rapidly above:

$$
|\hat g_\varepsilon(k)|^2 \sim
\begin{cases}
|\hat g(k)|^2, & k\lesssim 1/\varepsilon,\\[2pt]
\text{rapidly}\to0, & k\gtrsim 1/\varepsilon,
\end{cases}
$$

by the Fourier scaling/uncertainty principle (a feature of width $\varepsilon$
concentrates its content below $1/\varepsilon$). The cutoff $k_{\max}\sim
1/\varepsilon$ is the only role $\varepsilon$ plays in the scaling.

### I.5 The forcing-floor scaling law

The operator-channel "floor" is $\lVert\mathcal{L}g_\varepsilon\rVert^2$. By
Parseval and §I.2–I.4,

$$
\lVert\mathcal{L}g_\varepsilon\rVert^2
\;\propto\;\int_0^{\infty}|\widehat{\mathcal{L}g_\varepsilon}(k)|^2\,dk
\;\sim\;\int_0^{1/\varepsilon} k^{\,4p-2r-2}\,dk .
$$

If the exponent exceeds $-1$ the integral is **upper-cutoff dominated**, giving

$$
\boxed{\;\lVert\mathcal{L}g_\varepsilon\rVert^2\;\sim\;\varepsilon^{-(4p-2r-1)}\;}
\qquad\text{(when }4p-2r-1>0\text{).}
$$

- **Divergent / $\varepsilon$-sensitive regime** ($r<2p-\tfrac12$): the raw
  singularity gives an infinite floor; smoothing renders it finite but growing as
  $\varepsilon\to0$. The amplitude prefactor is set by the **jump magnitude** of
  $g^{(r)}$ and an $O(1)$ shape constant of the specific smoother.
- **Convergent regime** ($r>2p-\tfrac12$): the floor is finite even for the raw
  singularity and essentially $\varepsilon$-independent — the datum is smooth
  enough relative to the operator order.

*Real-space cross-check.* For a kink ($r=1$) under an order-$2p=2$ operator, the
smoothed $\mathcal{L}g$ is a regularised delta of height $\sim 1/\varepsilon$ and
width $\sim\varepsilon$ carrying fixed area (the jump); its squared norm is
height$^2\times$width $\sim\varepsilon^{-2}\cdot\varepsilon=\varepsilon^{-1}$,
matching the Fourier law with $4p-2r-1=1$.

### I.6 The trainability link (conjectured, NTK spectral bias)

Under the NTK/linearised-training approximation (companion note §2.3), the set of
network-reachable forcings is a subspace $\mathcal{S}$; gradient descent fills it
in **decreasing NTK-eigenvalue order**, which on a bounded domain corresponds to
**increasing frequency**. Within a finite training budget,
$\mathcal{S}\approx\{\text{low-}k\}$ and $\mathcal{S}^\perp\approx\{\text{high-}k\}$.
The achievable error is governed by the **uncancellable** projection
$\Pi_{\mathcal{S}^\perp}f$ (companion note §2.4). Consequently a forcing whose
spectrum is **tilted to high $k$** — exactly what §I.3 says $\mathcal{L}g$ is —
lands predominantly in $\mathcal{S}^\perp$ and dominates the error, whereas a
**low-$k$** forcing is cancellable.

---

## Part II — Specialisation to the backward heat equation and the call

### II.1 Problem and the payoff kink

Backward heat operator $\mathcal{P}=\partial_t+\mathcal{L}$,
$\mathcal{L}=\tfrac{\sigma^2}{2}\partial_{xx}$, so $2p=2$ ($p=1$); coordinate
$x=\ln S$. The terminal datum is the call payoff $g(x)=(e^x-K)^+$. Near the strike
$x^\star=\ln K$, $e^x-K\simeq K\,(x-x^\star)$, so

$$
g(x)\simeq K\,(x-x^\star)^+ \quad\text{near }x^\star,
$$

a **kink**: $g$ is continuous, $g'(x)=e^x\mathbf 1_{x>x^\star}$ jumps by
$e^{x^\star}=K$ at the strike, and

$$
g''(x)=\underbrace{e^x\mathbf 1_{x>x^\star}}_{\text{bounded}}+\underbrace{K\,\delta(x-x^\star)}_{\text{singular}} .
$$

Hence regularity index $r=1$ and $|\hat g(k)|^2\sim K^2\,k^{-4}$: the **amplitude
of the tail is set by the slope jump $K$** (this is the spectral origin of the
$O(K)$ price scale that makes the absolute floor magnitudes large; cf. the
loss-magnitude discussion of the ablation note).

### II.2 The operator channel is a white spectrum (the Dirac $g''$)

With $p=1$, $r=1$ the net exponent of §I.3 is $4p-2r-2=0$:

$$
|\widehat{\mathcal{L}g}(k)|^2\sim k^4\cdot K^2k^{-4}=K^2\,k^{0}\quad(\text{flat}).
$$

So $\mathcal{L}g=\tfrac{\sigma^2}{2}g''$ of the **raw** payoff is white — it *is*
the Dirac $\tfrac{\sigma^2}{2}K\,\delta(x-x^\star)$ — with infinite $L^2$ norm.
This is the diffusion/operator forcing channel of the ablation note (§3.1); its
flat spectrum places almost all of its energy in $\mathcal{S}^\perp$ (§I.6),
identifying it as the **high-frequency, uncancellable** part of the forcing and
hence the binding term for the error.

### II.3 Smoothing: the $\varepsilon^{-1}$ floor and softplus vs Chen–Mangasarian

The two extensions are mollifiers of the kink at scale $\varepsilon$:

- **softplus**, $\tfrac1\beta\log(1+e^{\beta z})$ smoothing of $z^+$, effective
  $\varepsilon\sim1/\beta$;
- **Chen–Mangasarian**,
  $M_\varepsilon(a,b)=\tfrac12\!\big(a+b+\sqrt{(a-b)^2+\varepsilon^2}\big)$, with
  $M_\varepsilon(z,0)=\tfrac12\!\big(z+\sqrt{z^2+\varepsilon^2}\big)$ and
  $M_\varepsilon''(z,0)=\tfrac12\,\varepsilon^2(z^2+\varepsilon^2)^{-3/2}$ — an
  explicit regularised delta of height $\tfrac1{2\varepsilon}$, width
  $\sim\varepsilon$, unit area.

Both obey the §I.5 law with $4p-2r-1=1$:

$$
\text{floor}=\Big\lVert\tfrac{\sigma^2}{2}g''_\varepsilon\Big\rVert^2
\sim c_{\rm shape}\,(\tfrac{\sigma^2}{2})^2\,(\text{slope jump})^2\,\varepsilon^{-1},
$$

finite but **growing as the smoothing sharpens** ($\varepsilon\to0$). The slope
jump ($\propto K$), $\sigma$, the operator, and the domain are common to both
extensions; only the $O(1)$ shape constant $c_{\rm shape}$ and the effective
$\varepsilon$ differ. Therefore the **measured floor ratio is, to leading order,
the ratio of effective spectral cutoffs**:

$$
\frac{\text{floor}_{\rm softplus}}{\text{floor}_{\rm CM}}
\;\approx\;\frac{c_{\rm shape}^{\rm sp}}{c_{\rm shape}^{\rm CM}}\cdot
\frac{\varepsilon_{\rm CM}}{\varepsilon_{\rm softplus}} .
$$

The ablation measures this ratio at $\approx79$ for the additive form
($7200$ vs $91$): with comparable shape constants this says the
Chen–Mangasarian smoothing presents an effective cutoff roughly $79\times$
coarser (larger effective $\varepsilon$) than the softplus one at the chosen
parameters — it keeps far less white high-frequency energy.

> *Caveat (measurement to do).* The exact prefactor — including the power of $K$,
> which depends on whether $\varepsilon$ is taken in price or in $\log$-spot units
> and where the $x=\ln S$ Jacobian falls — requires the precise smoother
> definitions and parameters used in the runs. The robust statements are (i) the
> $\varepsilon^{-1}$ scaling and (ii) the floor *ratio* $\approx$ effective-cutoff
> ratio at fixed payoff/operator. Pinning the constant down (closed-form
> $\lVert g''_\varepsilon\rVert^2$ for each smoother at its actual parameter)
> would convert the $\approx79\times$ into a statement about the two effective
> $\varepsilon$.

### II.4 Velocity versus operator channel, spectrally

For the convex extension $\Psi=\lambda(t)g$ the forcing splits as
$f=\underbrace{\lambda'g}_{\text{velocity}}+\underbrace{\lambda\,\mathcal{L}g}_{\text{operator}}$
(ablation note §3.1). Spectrally the two channels are opposites:

- **velocity** $\lambda'g\propto g$: spectrum $K^2k^{-4}$, **low-$k$ concentrated**
  — it carries the (smooth, large-amplitude) profile of the datum itself. By §I.6
  it lies in $\mathcal{S}$ and is **cancellable**, which is why a large velocity
  floor coexists with a small achieved residual (the measured inversion).
- **operator** $\lambda\,\mathcal{L}g$: spectrum $K^2k^{0}$ up to $1/\varepsilon$,
  **high-$k$/white** — it lies in $\mathcal{S}^\perp$ and is **uncancellable**.

So the *same* total floor decomposes into a cancellable low-frequency part and an
uncancellable white part; only the latter sets the error, and only the latter is
controlled by the smoothing.

### II.5 Consequence: the cancellability–bias trade-off

The floor law $\sim\varepsilon^{-1}$ and the spectral-bias link give a single
prescription with a built-in tension:

- **Larger $\varepsilon$** (smoother datum) lowers the white operator floor as
  $\varepsilon^{-1}$, shrinks $\lVert\Pi_{\mathcal{S}^\perp}\mathcal{L}g\rVert$, and
  lowers the PDE-residual-limited error — this is the order-of-magnitude lever
  measured in the ablation (softplus $\to$ Chen–Mangasarian).
- **But** a larger $\varepsilon$ moves the smoothed terminal datum away from the
  true payoff, introducing an interior **bias** $\lVert g_\varepsilon-g\rVert$ in
  the target itself. The optimal $\varepsilon$ balances the (decreasing in
  $\varepsilon$) uncancellable forcing against the (increasing in $\varepsilon$)
  payoff bias — the open trade-off flagged as question 1 of the companion note
  ($\varepsilon_0\to0$).

---

## Claim strength

- **Proven (classical):** the differentiation rule, the Dirac flat spectrum, the
  decay law $|\hat g_k|^2\sim k^{-2(r+1)}$, the amplification $|\widehat{\mathcal{L}g}_k|^2\sim k^{4p}|\hat g_k|^2$,
  and the floor scaling $\sim\varepsilon^{-(4p-2r-1)}$ (Parseval $+$ cutoff).
- **Measured:** the $\approx79\times$ additive-form floor ratio between softplus
  and Chen–Mangasarian (ablation note §3.1); the white vs low-$k$ character of the
  two channels is consistent with the measured floor-vs-uncancellable inversion.
- **Conjectured:** the identification $\mathcal{S}\approx\{\text{low-}k\}$,
  $\mathcal{S}^\perp\approx\{\text{high-}k\}$ (NTK spectral bias) and hence the
  step from "white forcing" to "uncancellable, error-dominating" — corroborated
  but not proven here.
