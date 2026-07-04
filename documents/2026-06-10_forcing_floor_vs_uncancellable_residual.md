# Why the forcing floor predicts accuracy across problems but not across ansatz forms

**Date:** 2026-06-10
**Context:** boundary-constrained learning of PDEs — ansatz-form ablation
(`experiments/python_scripts/exp_ansatz_forms_heat/`). Four terminal-condition
enforcement forms (two hard boundary-constrained ansatze, a soft-penalty PINN,
a pure-network control) on backward parabolic problems (heat equation with three
terminal data: two-mode sine, Jacobi $\vartheta_3$, smoothed call payoff; plus a
time-dependent Chen–Mangasarian variant `call_cm`).

> Math rendering: open in a Markdown + KaTeX/MathJax viewer.

---

## 1. The empirical observation

For the **hard** forms the stage loss is the PDE residual, which splits (report
`rem:residual-decomposition`) into a $\theta$-dependent network contribution and
a $\theta$-independent **floor** $\mathbb{E}_\mu[(\mathcal{P}\Psi)^2]$ set by the
terminal-data extension $\Psi$ alone.

Two facts, both measured (mean over 3 seeds, $20000$ iterations):

**(a) Across problems, at fixed form — the floor predicts accuracy.** Tracking
`hard_constant_linear` across the four ICs:

| IC | floor $\mathbb{E}[(\mathcal{P}\Psi)^2]$ | rel $L^2$ |
|----|----|----|
| theta3 | 6.9 | 0.0015 |
| call_cm | 43 | 0.0144 |
| sine | 791 | 0.068 |
| call | 2870 | 0.139 |

Monotone, and roughly $\text{rel }L^2 \sim \sqrt{\text{floor}}$ (log–log slope
$\approx 0.5\text{–}0.7$ over a $420\times$ range in the floor).

**(b) Across forms, at fixed problem — the floor does *not* predict accuracy.**
Tracking the four hard forms within `call_cm`:

| form | floor | rel $L^2$ |
|----|----|----|
| hard_constant_linear | 43 | 0.0144 |
| hard_constant_exp | 43 | 0.0150 |
| hard_convex_exp | 127 | 0.0174 |
| **hard_convex_linear** | **943** | **0.0095** |

The form with the **largest** floor (`hard_convex_linear`, $943$) achieves the
**smallest** error ($0.0095$). The same inversion appears for `sine`.

The rest of this note explains the asymmetry: **the floor is not the right
predictor — the *uncancellable* part of it is.**

---

## 2. General formulation

Let $\mathcal{P}$ be **any linear** PDE operator, written $\mathcal{P} = \partial_t + \mathcal{L}$,
with $\mathcal{L}$ the spatial part — an elliptic operator of order $2m$ (the
heat case is $\mathcal{L} = \tfrac{\sigma^2}{2}\partial_{xx}$, $2m=2$; the
construction below is independent of the specific $\mathcal{L}$). The problem is

$$\mathcal{P}u = 0,\qquad u(\cdot,T) = g.$$

The hard boundary-constrained ansatz is, with interpolation coefficient
$\lambda(t)$ ($\lambda(T)=1$) and terminal-data extension $\Psi$ ($\Psi(\cdot,T)=g$),

$$\hat u_\theta = (1-\lambda)\,\Phi_\theta + \Psi,$$

where $\Phi_\theta$ is the free neural field. The terminal (and lateral
boundary) data are satisfied **exactly** by construction.

### 2.1 Linearity splits the residual

Because $\mathcal{P}$ is linear,

$$
\mathcal{P}\hat u_\theta
= \underbrace{\mathcal{P}\big[(1-\lambda)\Phi_\theta\big]}_{R_\theta\ \text{(network contribution)}}
+ \underbrace{\mathcal{P}\Psi}_{f\ \text{(forcing; $\theta$-independent)}}.
$$

### 2.2 The floor is the "do-nothing" residual

With $\|\cdot\|$ the $L^2(\mu)$ norm of the collocation measure $\mu$, the stage
loss is

$$
\mathcal{L}(\theta) = \|R_\theta + f\|^2
= \|R_\theta\|^2 + 2\langle R_\theta, f\rangle + \|f\|^2,
\qquad
\boxed{\ \text{floor} = \|f\|^2 = \|\mathcal{P}\Psi\|^2.\ }
$$

The floor is exactly $\mathcal{L}$ evaluated at $R_\theta = 0$ — i.e. the loss of
the **untrained** ansatz. It ignores that the optimiser *varies* $R_\theta$.

### 2.3 What the optimiser can actually achieve

Let $\mathcal{R} = \{\mathcal{P}[(1-\lambda)\Phi_\theta] : \theta\in\Theta\}$ be
the set of network-reachable forcings — the residual fields the network can
*attain* by varying its weights $\theta$. The trained loss is the squared
$L^2(\mu)$-distance from the target $-f$ to that set,

$$
\mathcal{L}^\star = \min_\theta \|R_\theta + f\|^2 = \operatorname{dist}^2(-f,\ \mathcal{R}).
$$

In general this is intractable: the map $\theta \mapsto \Phi_\theta$ is nonlinear,
so $\mathcal{R}$ is a **curved, non-convex** manifold in $L^2(\mu)$ and the
distance has no closed form.

**The NTK / linearised-training approximation removes the curvature.** In the
wide-network (lazy-training) regime $\Phi_\theta$ stays close to its
initialisation $\Phi_{\theta_0}$ throughout training, so one linearises in the
weights,

$$
\Phi_\theta \approx \Phi_{\theta_0} + \nabla_\theta\Phi_{\theta_0}\,(\theta-\theta_0).
$$

The map $\theta \mapsto \Phi_\theta$ is then **affine** in $\theta$, and applying
the linear operator $\mathcal{P}[(1-\lambda)\,\cdot\,]$ preserves affineness, so
$\theta \mapsto R_\theta$ is affine too. The reachable set $\mathcal{R}$ becomes
an affine subspace; absorbing the constant initialisation offset, it is
effectively the **linear subspace**

$$
\mathcal{S}
= \operatorname{span}\big\{\,\mathcal{P}\big[(1-\lambda)\,\partial_{\theta_i}\Phi_{\theta_0}\big] : i\,\big\}
\subset L^2(\mu),
$$

spanned by the operator-transformed Jacobian (neural-tangent) features
$\partial_{\theta_i}\Phi_{\theta_0}$, frozen at initialisation. The effect of the
approximation is to replace a non-convex *distance-to-a-manifold* problem by a
**linear least-squares projection** onto a fixed subspace — which has a closed
form.

Minimising $\|R_\theta + f\|^2$ over $R_\theta\in\mathcal{S}$ is then least
squares: the optimiser cancels exactly the part of $f$ lying **inside**
$\mathcal{S}$, and the irreducible remainder is the projection of $f$ onto the
**orthogonal complement** $\mathcal{S}^\perp$,

$$
\boxed{\ \mathcal{L}^\star = \big\|\Pi_{\mathcal{S}^\perp} f\big\|^2,\qquad
\underbrace{\|f\|^2}_{\text{floor}}
= \underbrace{\|\Pi_{\mathcal{S}} f\|^2}_{\text{cancellable}}
+ \underbrace{\|\Pi_{\mathcal{S}^\perp} f\|^2}_{\text{uncancellable}\,=\,\mathcal{L}^\star}.\ }
$$

The second identity is the Pythagorean split of the floor along
$\mathcal{S}\oplus\mathcal{S}^\perp$. The floor over-counts by the
**cancellable** part $\Pi_{\mathcal{S}}f$ — the component of the forcing the
network can synthesise and subtract — and only the **uncancellable** part
$\Pi_{\mathcal{S}^\perp}f$ survives at the optimum. ($\Pi_{\mathcal{S}^\perp}$
denotes the $L^2(\mu)$-orthogonal projector onto $\mathcal{S}^\perp$.)

> **Scope of the approximation.** Finite-width networks and finite training
> budgets satisfy the linearisation only approximately, hence the "(effectively)"
> hedge: the claim is that $\mathcal{R}$ behaves *like* the subspace $\mathcal{S}$
> closely enough to predict the **ordering** of errors across forms, not that it
> literally is one.

### 2.4 Error is set by the uncancellable residual, not the floor

Two distinct quantities must be related here, and they lie on different sides of
the PDE:

- $\text{rel }L^2 = \|\hat u_\theta - u^\star\|/\|u^\star\|$ — the relative error
  of the **solution** ($u$);
- $\mathcal{L}^\star = \|\mathcal{P}\hat u_\theta\|^2$ — the trained **residual**
  loss ($\mathcal{P}u$), shown in §2.3 to equal $\|\Pi_{\mathcal{S}^\perp}f\|^2$.

The bridge between them is the well-posedness (stability) estimate of
$\mathcal{P}$. The problem posed for $\mathcal{P} = \partial_t + \mathcal{L}$ on
the space–time cylinder $\Omega\times[0,T]$ has **three** data: the interior
forcing, the **terminal** condition at $t=T$, and the **lateral** boundary
condition on $\partial\Omega\times[0,T]$. Set $e = \hat u_\theta - u^\star$ and
apply $\mathcal{P}$: since $u^\star$ solves the PDE exactly
($\mathcal{P}u^\star = 0$, $u^\star(\cdot,T)=g$, $u^\star|_{\partial\Omega}=b$),
the error $e$ solves the *same* operator equation with data

$$
\underbrace{\mathcal{P}e = \mathcal{P}\hat u_\theta}_{\text{interior residual}},\qquad
\underbrace{e(\cdot,T) = \hat u_\theta(\cdot,T) - g}_{\text{terminal trace}},\qquad
\underbrace{e|_{\partial\Omega} = \hat u_\theta|_{\partial\Omega} - b}_{\text{lateral trace}}.
$$

The well-posedness estimate for the backward parabolic problem bounds the
solution norm by exactly these three data, each in its own trace norm,

$$
\|e\|_{L^2(\Omega\times[0,T])} \ \le\
C_{\text{int}}\,\|\mathcal{P}e\|
+ C_{\text{term}}\,\|e(\cdot,T)\|_{L^2(\Omega)}
+ C_{\text{lat}}\,\|e|_{\partial\Omega}\|_{L^2(\partial\Omega\times[0,T])}.
$$

The hard ansatz $\hat u_\theta = (1-\lambda)\Phi_\theta + \Psi$ matches the
terminal and lateral data **exactly** by construction
($\hat u_\theta(\cdot,T) = g$ and $\hat u_\theta|_{\partial\Omega} = b$), so the
terminal trace and the lateral trace of $e$ are **identically zero**. The last
two terms vanish, leaving

$$
\|\hat u_\theta - u^\star\| \ =\ \|e\| \ \le\ C\,\|\mathcal{P}e\|
\ =\ C\,\|\mathcal{P}\hat u_\theta\|
\ =\ C\sqrt{\mathcal{L}^\star}\ =\ C\,\big\|\Pi_{\mathcal{S}^\perp} f\big\|,
$$

where $C = C_{\text{interior}}$ is the **stability constant** of $\mathcal{P}$ on
the collocation domain — a fixed property of the problem, independent of
$\theta$.

**This is a one-sided inequality (proved), not yet the equivalence (empirical).**
Small residual $\Rightarrow$ small error, rigorously. Upgrading $\le$ to the
scaling relation

$$
\boxed{\ \text{rel }L^2 \ \sim\ \sqrt{\mathcal{L}^\star}
\ =\ \big\|\Pi_{\mathcal{S}^\perp}\,\mathcal{P}\Psi\big\|\ }
\qquad\text{— governed by } \Pi_{\mathcal{S}^\perp}f,\ \text{not by }\|f\|,
$$

requires two further, *non-proven* assumptions:

1. **Tightness.** The stability bound is not excessively loose, i.e. $\|e\| \approx
   C\sqrt{\mathcal{L}^\star}$ rather than $\|e\| \ll C\sqrt{\mathcal{L}^\star}$. A
   residual concentrated in the near-null-space of $\mathcal{P}$ could make the
   error substantially larger than the bound's converse suggests; the assumption
   made here is that this regime does not hold.
2. **Common constant.** Across the objects being *compared* — the four forms at
   fixed problem (§1b) — the operator $\mathcal{P}$, the domain, and the
   collocation measure $\mu$ are shared, so the stability constant $C$ and the
   normaliser $\|u^\star\|$ are essentially common. Then
   $\text{rel }L^2 \approx (C/\|u^\star\|)\sqrt{\mathcal{L}^\star} \propto
   \sqrt{\mathcal{L}^\star}$, and the common factor cancels out of any ranking.

Hence the `$\sim$` is a statement about **ordering**, not absolute value: the
ranking of forms by accuracy is set by $\sqrt{\mathcal{L}^\star} =
\|\Pi_{\mathcal{S}^\perp}\mathcal{P}\Psi\|$ alone. Across *problems* (§1a) the
constant $C/\|u^\star\|$ drifts (different $u^\star$, different effective
stability constant), so the collapse is expected to be a tight band rather than a
single exact line — which is precisely what §5 step 1 proposes to test by
re-plotting rel $L^2$ against $\sqrt{\mathcal{L}^\star}$.

The achievable error is therefore controlled **not** by the size of the forcing
$\|\mathcal{P}\Psi\|$ (the floor) but by the size of the component the network
**cannot** synthesise and subtract, $\Pi_{\mathcal{S}^\perp}\mathcal{P}\Psi$.

### 2.5 Which subspace is $\mathcal{S}$? (spectral bias)

Gradient descent on a neural field learns target components in **decreasing order
of NTK eigenvalue**: large-eigenvalue = **low spatial frequency** modes are
learned fast and accurately; high-frequency modes are learned slowly/poorly
within a finite training budget. To good approximation,

$$\mathcal{S} \approx \{\text{low-frequency forcings}\},\qquad
\mathcal{S}^\perp \approx \{\text{high-frequency / sharply-localised forcings}\}.$$

---

## 3. Applying it to the two forcing mechanisms

Write the convex-combination extension as $\Psi = \lambda(t)\,g(x)$.
Since $\mathcal{P} = \partial_t + \mathcal{L}$,

$$
f = \mathcal{P}\Psi
= \underbrace{\lambda'(t)\,g(x)}_{f_{\rm vel}\ \text{(interpolation-velocity)}}
+ \underbrace{\lambda(t)\,\mathcal{L}g(x)}_{f_{\rm op}\ \text{(operator / damped-diffusion)}}.
$$

(The constant form $\Psi=g$ is the special case $f = \mathcal{L}g$, i.e. $f_{\rm vel}=0$.)

- $f_{\rm vel} \propto g$ carries the spatial profile of the **datum itself** —
  as smooth/low-frequency as $g$. Its *magnitude* can be large (for the call,
  $g$ is $O(100)$), so it inflates the **floor**; but being low-frequency it lies
  (mostly) in $\mathcal{S}$ ⟹ **cancellable** ⟹ small contribution to
  $\mathcal{L}^\star$.
- $f_{\rm op} \propto \mathcal{L}g$ carries the profile of the **operator applied
  to $g$**. A degree-$2m$ elliptic $\mathcal{L}$ multiplies a Fourier mode of
  wavenumber $k$ by $\sim |k|^{2m}$ — it **amplifies high frequencies**. For a
  $g$ whose first derivative is nearly discontinuous (the call payoff),
  $\mathcal{L}g$ is a sharp spike ⟹
  high-frequency ⟹ lies in $\mathcal{S}^\perp$ ⟹ **uncancellable** ⟹ dominates
  $\mathcal{L}^\star$.

This is corroborated by the monitored sub-channels
$\mathbb{E}[(\partial_t\Psi)^2]$ (velocity) and
$\mathbb{E}[(\tfrac{\sigma^2}{2}\partial_{xx}\Psi)^2]$ (diffusion): for the
convex-combination call the floor is dominated by the velocity term
($\mathbb{E}[(\lambda'g)^2]\approx 840$), while for the constant call it is pure
diffusion (the spike, $\approx 2870$).

---

## 4. Conclusion

- The **constant** form ($\Psi=g$) has $f=\mathcal{L}g$: the high-frequency spike,
  mostly in $\mathcal{S}^\perp$ — a floor that is small-to-moderate but
  **uncancellable**.
- The **convex-combination** form adds the large but low-frequency $f_{\rm vel}=\lambda'g$:
  a **bigger floor that is mostly cancellable**.

Because the error depends on $\|\Pi_{\mathcal{S}^\perp} f\|$ and not on $\|f\|$, a
form with a **larger floor but a more cancellable (lower-frequency) forcing** can
achieve a **smaller error** than a form with a lower but spiky floor. This is
exactly the `call_cm` inversion (`hard_convex_linear`: floor $943$, error
$0.0095$, vs `hard_constant`: floor $43$, error $0.0144$).

**Precise statement.** The floor is $\|\mathcal{P}\Psi\|^2$; the achievable error is
governed by $\|\Pi_{\mathcal{S}^\perp}\mathcal{P}\Psi\|^2$ — the high-frequency,
spectrally **uncancellable** part of the forcing. The floor predicts accuracy
*across problems at fixed form* (where the cancellable fraction is roughly
constant) but breaks *across forms at fixed problem* (where the cancellable
fraction changes with the extension).

---

## 5. Next steps

1. **Re-plot against the achieved residual (cheap, no new runs).** Replace the
   floor on the $x$-axis of `floor_vs_accuracy.png` by the trained
   $\mathcal{L}^\star$ (the best-state final $\mathcal{L}_{\rm pde}$, already
   logged). Hypothesis: $\text{rel }L^2 \sim \sqrt{\mathcal{L}^\star}$ collapses
   **both** across forms and across ICs onto one line.
2. **Spectral decomposition (some compute).** Fourier-decompose $\mathcal{P}\Psi$
   and the achieved residual in $x$; verify the network cancels low modes and
   leaves high ones — a direct measurement of $\Pi_{\mathcal{S}^\perp}$.
3. **$\varepsilon_0$ sweep for `call_cm` (new runs).** Larger $\varepsilon_0$
   pushes the forcing to lower frequency (more cancellable, lower error) but
   raises the interior bias from the true payoff — mapping the
   cancellability ↔ bias trade-off (report open question 1, $\varepsilon_0\to0$).

---

*Artifacts:* cross-seed summary (4 ICs, 3 seeds)
`data/ansatz_forms_cross_seed_summary/2026-06-09-22-43-44-724075Z_4ic_3seed/`;
per-run decomposition figures (with the velocity/diffusion sub-channels)
under `data/ablation_ansatz_forms/*_iters20000_seed*/comparison/loss_decomposition.png`.
