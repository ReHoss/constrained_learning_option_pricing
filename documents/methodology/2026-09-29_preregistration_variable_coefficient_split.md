# Pre-registration — split extension for variable-coefficient generators (orders 2 and 4)

**Written 2026-09-29, before any implementation or run.** This document fixes the question, the
cells, the compared extensions, the amplitude sweep, the analytical predictions and the falsification conditions ahead
of the campaign, so that the outcome cannot steer the design. The outcome is to be reported
whatever it is. Status: **decided, revision 2 (2026-09-29), before any implementation or run.** The
open decisions of revision 1 were delegated by the author ("prends la meilleure décision selon les
coefficients ... la cohérence, la facilité de lecture, pas trop d'artifices") and are recorded in §8;
the changes of revision 2 are listed in §9.

## 1. Question

Every trained cell so far has constant coefficients, so the full semigroup $e^{sL}$ is a Fourier
multiplier and the requirement C4 (computable extension) never selects anything: the exact
solution is available and the split extension is a closed-form special case of it. This study
places the split extension where C4 binds: a generator with **variable coefficients**, whose full
semigroup has no closed form, while the semigroup of a constant-coefficient operator does.

The question is then which constant-coefficient operator $A$ to retain. Two candidates are
compared, both computable:

- **frozen at the singular point**: the principal coefficient evaluated at the break point $x^\star$
  of the datum (the only point where the datum is not $C^\infty$);
- **frozen at the mean**: the principal coefficient replaced by its spatial average.

## 2. Setting

Circle $\Omega=\mathbb T=\mathbb R/(2\pi\mathbb Z)$, horizon $T=1$, the band-limited Bernoulli datum
$g_{K}(x)=\sum_{k=1}^{K}\cos(kx)/(\pi^2k^2)$ with $K=128$, break point $x^\star=0$, jump of the first
derivative $[g']_{x^\star}\neq0$. Same network, budget, sampler and master seeds $\{0,1,2\}$ as the
stage-2 cells.

Two cells, each the variable-coefficient version of a trained constant-coefficient cell:

- **LV2 (order 2, local volatility in log-price).**
  $L^X u = a(x)\,\partial_{xx}u + (r-a(x))\,\partial_x u - r\,u$, with
  $a(x)=\nu_0\bigl(1+\varepsilon\cos(x-\varphi)\bigr)$, $\nu_0=0.125$, $\varepsilon=0.5$, $\varphi=\pi/4$, $r=0.03$.
  At $a\equiv\nu_0$ it is the Black–Scholes cell $G_2$. $a(x)=\sigma(x)^2/2$ with
  $\sigma(x)\in[0.35,0.61]$. The phase $\varphi=\pi/4$ makes $a'(x^\star)\neq0$ (the generic case) and
  $a(x^\star)=\nu_0(1+\varepsilon/\sqrt2)\neq\bar a=\nu_0$, so the two freezings differ.
- **LV4 (order 4).** $L^X u = -\beta(x)\,\partial_x^4u + 1.3\,\partial_x u - 0.4\,u$, with
  $\beta(x)=\beta_0\bigl(1+\varepsilon\cos(x-\varphi)\bigr)$, $\beta_0=0.05$, same $\varepsilon$, $\varphi$. At
  $\beta\equiv\beta_0$ it is the cell $G_3$.

The periodic setting is an idealisation of a local-volatility model (the log-price domain is not a
circle); it is chosen because it keeps the evaluation exact up to a controlled truncation.

### 2.1 Provenance of the base values (recorded 2026-09-29, at the author's question)

The base values are **inherited** from the constant-coefficient cells, so that each variable-
coefficient cell differs from an already measured cell by the spatial variation alone, and reduces
to it at $arepsilon=0$. Their own provenance, as found in the repository:

- **$G_2$** ($
u_0=\sigma^2/2=0.125$, advection $r-\sigma^2/2=-0.095$, reaction $-r=-0.03$): the
  Black–Scholes generator in log-price with $\sigma=0.5$, $r=0.03$. The structure is dictated by the
  model; the two numerical values are conventional and **no documented protocol selects them**.
- **$G_1$** ($0.7\,\partial_{xx}+1.3\,\partial_x-0.4$) and **$G_3$** ($-0.05\,\partial_x^4+1.3\,\partial_x-0.4$):
  introduced with the periodic spectral toolbox (commit `356417c`, 2026-07-10) and in the report,
  **without any recorded justification or selection protocol**. The only properties that can be read
  off them are structural: every lower-order channel is present (non-zero advection and reaction,
  so the remainder $B$ of each split is non-trivial), the principal term is dissipative, and $G_3$
  shares the lower-order terms of $G_1$ so that $G_1$ and $G_3$ differ by the principal part only.
  Whether these properties motivated the values is not documented; it is an inference.

Consequence for the claims: no result of this study (nor of the constant-coefficient cells) is
claimed to hold uniformly over the coefficients. Dependence on the variation amplitude is measured
by the sweep of §3.1; dependence on the base values is not studied and is stated as a limitation.

## 3. Compared extensions (per cell)

| Name | Extension $h$ | Computable? |
|---|---|---|
| `convex_raw` | $\lambda(t)\,g$ | yes |
| `constant_in_time` | $g$ | yes |
| `split_frozen_singular` | $e^{(T-t)A_\star}g$, $A_\star=c_{2p}(x^\star)\,\partial_x^{2p}$ | yes (Fourier multiplier) |
| `split_frozen_mean` | $e^{(T-t)\bar A}g$, $\bar A=\bar c_{2p}\,\partial_x^{2p}$ | yes (Fourier multiplier) |

The two splits retain the principal term only, so their remainders have the same lower-order
structure and the comparison isolates the point at which the principal coefficient is frozen.
`split_frozen_mean` is the extension of the constant-coefficient split already trained on the
reference cell ($\{\partial_{xx}\}$ on $G_2$, $\{\partial_x^4\}$ on $G_3$), now applied to the
variable-coefficient problem. No exact-solution control is trained: the exact solution has no closed
form in these cells, which is the situation C4 describes; the network's own error level on this datum
is read from the zero-forcing controls of $G_2$ and $G_3$ (same network, datum and budget).

### 3.1 Amplitude sweep (dose–response)

The amplitude $arepsilon$ is swept instead of fixed. The attribution of any difference between the two
freezings to the coefficient mismatch at the singular point predicts a **dose–response**: the gap
vanishes at $arepsilon=0$, where both freezings coincide with the constant-coefficient split already
measured on $G_2$/$G_3$ (the anchor of the curve), and grows with $arepsilon$. The constraint
$0<arepsilon<1$ keeps the principal coefficient of strict sign (ellipticity), and
$a_{\max}/a_{\min}=(1+arepsilon)/(1-arepsilon)$.

- Forcing energies (no training): $arepsilon\in\{0.1,\,0.25,\,0.5,\,0.75\}$.
- Training: $arepsilon\in\{0.25,\,0.75\}$ (ratios $a_{\max}/a_{\min}=5/3$ and $7$).

The phase stays $arphi=\pi/4$: the two non-degeneracy conditions are $\cosarphi
eq0$ (otherwise
$c(x^\star)=ar c$ and the two freezings coincide) and $\sinarphi
eq0$ (otherwise $x^\star$ is a critical
point of the coefficient, $
u=2$), and $arphi=\pi/4$ maximises $\min(|\cosarphi|,|\sinarphi|)$.

For LV2, `split_frozen_mean` coincides with a graded Gaussian at $\nu_c=\bar a$, so no graded
Gaussian arm is run; no Chen–Mangasarian arm is run either, its role (the mollification objection)
being settled on $G_1$–$G_3$. For each split the forcing is
$\partial_t h + L h = \bigl(c_{2p}(x)-c_{2p}^{A}\bigr)\partial_x^{2p}h + (\text{lower-order terms})\,h$,
with $c^{A}_{2p}$ the frozen value: the principal part is cancelled only where
$c_{2p}(x)=c_{2p}^{A}$.

## 4. Analytical predictions (heuristic scaling; to be turned into proofs)

Near the break point, $h(\cdot,T-s)$ is the datum convolved with the kernel of $A$, of width
$w=(|c^{A}_{2p}|\,s)^{1/(2p)}$, and $\partial_x^{2p}h$ contains $[g']_{x^\star}\,\partial_x^{2p-2}K_w(x-x^\star)$,
of height $w^{-(2p-1)}$ and width $w$. A coefficient difference vanishing to order $\nu$ at $x^\star$
multiplies it by $|x-x^\star|^{\nu}\sim w^{\nu}$. Hence the squared spatial $L^2$ norm of the principal
forcing scales as $w^{\,2\nu-4p+3}$, and its time integral near $s=0$ converges iff
$2\nu-4p+3>-2p$, that is
$$p<\tfrac32+\nu .$$
Freezing at the mean gives $\nu=0$; freezing at the singular point gives $\nu\ge1$ ($\nu=1$ generically,
here since $c_{2p}'(x^\star)\neq0$). This is a heuristic derivation from the kernel scaling, not a proof;
it is consistent with the criterion $m_B<\rho+\kappa$ of the constant-coefficient analysis, with
Sobolev index $\rho=3/2^-$ for the kinked datum.

| Prediction | LV2 ($p=1$) | LV4 ($p=2$) |
|---|---|---|
| **P1** forcing of `split_frozen_mean` in $L^2(Q)$ as $K\to\infty$ | yes ($1<3/2$) | **no** ($2\not<3/2$): energy grows linearly in $K$ |
| **P2** forcing of `split_frozen_singular` in $L^2(Q)$ as $K\to\infty$ | yes, and pointwise bounded | yes ($2<5/2$), pointwise unbounded |
| **P3** `constant_in_time`, `convex_raw` | energy grows like $K$ | energy grows like $K^{5}$ |

Consequence: at order 2 both freezings are admissible and differ by the constant
$(a(x^\star)-\bar a)^2$; at order 4 **only freezing at the singular point** keeps the forcing
square-integrable. For order 6 ($p=3$) even the singular-point freezing fails generically
($3\not<5/2$) and would require $\nu\ge2$ (freezing at a point where the coefficient is also
stationary) or a smoother datum — outside this study.

**Conjectured trained outcome (not a prediction of the analysis; to be tested):** on LV4,
`split_frozen_singular` reaches an error comparable to the $G_3$ splits, while `split_frozen_mean`
fails like the non-split extensions of $G_3$; on LV2 both freezings succeed, with a smaller error for
the singular-point freezing.

## 5. Measurements

- **Forcing energy** $\lVert\partial_t h+Lh\rVert^2_{L^2(Q)}$ in closed form. With coefficients
  $c_j(x)=c_{j,0}+c_{j,1}\cos(x-\varphi)$, the Fourier coefficient of the forcing at wavenumber $k$
  is a combination of the three terms $e^{s\lambda_m}\hat g_m$, $m\in\{k-1,k,k+1\}$, where
  $\lambda_m$ is the symbol of the retained operator; the time integral of its squared modulus is
  then a finite sum of $\int_0^Te^{zs}\,ds$, exact. Evaluated at $K\in\{32,64,128,256,512,1024\}$,
  at $\varepsilon\in\{0,0.1,0.25,0.5,0.75\}$ ($\varepsilon=0$ reproducing the constant-coefficient
  closed forms), to test P1–P3 by the growth in $K$ (CPU, `prepost`, no training).
- **Trained relative $L^2$ error** $\delta_\Gamma$ on the evaluation grid (1024 points, 11 time slices),
  median and quartiles over 3 seeds; also at $t=0$ and near $(x^\star,T)$.
- **Reference solution.** Fourier–Galerkin discretisation of $L^X$ on $|k|\le N$ (multiplication by
  $\cos(x-\varphi)$ couples $k$ to $k\pm1$), $u^\star(\cdot,t)=\exp\bigl((T-t)L^X_N\bigr)g$ by matrix
  exponential, with $N=512$ and a convergence check against $N=1024$ (tolerance stated in the
  report of the run).

## 6. Falsification conditions

- P1 is falsified if the energy of `split_frozen_mean` on LV4 stays bounded as $K$ grows from 32 to
  1024; P2 if that of `split_frozen_singular` grows without bound on LV4 over the same range.
- If `split_frozen_mean` and `split_frozen_singular` reach the same trained error on LV4 (overlapping
  quartiles), the conjectured trained consequence is not supported, whatever the energies do.
- A reference solution failing its $N=512$ versus $N=1024$ check invalidates the error measurements
  of the cell.
- **Dose–response.** If the trained-error gap between the two freezings on LV4 does not increase from
  $\varepsilon=0.25$ to $\varepsilon=0.75$ (medians, with the quartiles reported), its attribution to the
  coefficient mismatch at the singular point is not supported. Two trained amplitudes establish a
  direction, not a functional form; no fit of the gap against $\varepsilon$ is made.

## 7. Cost and implementation

Implementation: $x$-dependent coefficients in the autograd operator and in the analytic bypass;
a Fourier–Galerkin variable-coefficient module (matrix, reference solution, derivatives on demand);
split fields as constant-coefficient semigroups of the frozen operator; new cells and variants;
tests (Galerkin convergence, forcing identity, bypass against autograd, reduction to the
constant-coefficient closed forms at $\varepsilon=0$). Training: 2 orders × 2 amplitudes × 4 variants ×
3 seeds = 48 tasks, about 7 min each (LV2) and 30 min each (LV4), about 15 GPU-hours on V100
(`akz`), on the same hardware as every earlier trained cell, so that the comparison with $G_2$/$G_3$
and between variants of one seed is made on identical arithmetic. Energy study on `prepost`
(non-billed).

## 8. Decisions

- **Amplitude.** Swept (§3.1), by the author's decision of 2026-09-29.
- **Base values: inherited** (§2.1). Each variable-coefficient cell then reduces to a measured cell
  at $\varepsilon=0$, which anchors the dose–response, and no new selection protocol is introduced — a
  protocol fixing dimensionless ratios would itself be one choice among many, and would break the
  pairing with the measured cells. The absence of a documented protocol for $G_1$/$G_3$ is stated as
  a limitation (§2.1).
- **Phase $\varphi=0$ (critical point, $\nu=2$): not run.** At orders 2 and 4 the analysis predicts the
  same qualitative outcome for $\nu=1$ and $\nu=2$ (square-integrable forcing in both cases); the
  refinement concerns the scaling law itself and belongs to the later mathematical paper.
- **Order 6: not run.** It has no motivation in the setting of this paper, and the boundary
  $p<3/2+\nu$ that it would test belongs to the mathematical programme, which the author has placed
  after this framework paper.
- **Domain: the circle is kept.** It keeps the reference solution exact up to a controlled
  truncation and keeps the construction identical to $G_1$–$G_3$; a bounded interval would add
  boundary conditions and a second distance factor.

## 9. Revision log

- **Revision 1** (commit `000261b`, 2026-09-29): design with seven variants per cell.
- **Revision 1b** (commit `c609ce8`): provenance of the base values (§2.1) and amplitude sweep (§3.1).
- **Revision 2** (this version, before any implementation or run): decisions of §8 taken under the
  author's delegation; variant set reduced to four (removed: the singular-point split with frozen
  advection, which changes the lower-order remainder and so no longer isolates the freezing point;
  Chen–Mangasarian, whose role is settled on $G_1$–$G_3$; the exact-solution control, which has no
  closed form here); forcing energies computed in closed form rather than by quadrature, with
  $\varepsilon=0$ added as the constant-coefficient anchor and $K$ extended to 1024.
