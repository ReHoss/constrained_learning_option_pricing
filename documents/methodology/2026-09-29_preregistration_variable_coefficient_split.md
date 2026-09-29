# Pre-registration — split extension for variable-coefficient generators (orders 2 and 4)

**Written 2026-09-29, before any implementation or run.** This document fixes the question, the
cells, the compared extensions, the analytical predictions and the falsification conditions ahead
of the campaign, so that the outcome cannot steer the design. The outcome is to be reported
whatever it is. Status: **DRAFT for the author's confirmation**; decisions reserved to the author
are flagged **[AUTHOR]**.

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

## 3. Compared extensions (per cell)

| Name | Extension $h$ | Computable? |
|---|---|---|
| `convex_raw` | $\lambda(t)\,g$ | yes |
| `constant_in_time` | $g$ | yes |
| `split_frozen_singular` | $e^{(T-t)A_\star}g$, $A_\star=c_{2p}(x^\star)\,\partial_x^{2p}$ | yes (Fourier multiplier) |
| `split_frozen_singular_advection` | as above, plus the advection coefficient frozen at $x^\star$ (LV2 only) | yes |
| `split_frozen_mean` | $e^{(T-t)\bar A}g$, $\bar A=\bar c_{2p}\,\partial_x^{2p}$ | yes |
| `graded_chen_mangasarian` | CM kernel, $\varepsilon_0=\sqrt{2\nu_{\mathrm{ref}}T}$ | yes |
| `exact_solution` (control) | numerical reference $u^\star$ | **no closed form**: Fourier–Galerkin reference (§5) |

For LV2, `split_frozen_mean` coincides with a graded Gaussian at $\nu_c=\bar a$; no separate graded
Gaussian arm is run. For each split the forcing is
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

- **Forcing energy** $\lVert\partial_t h+Lh\rVert^2_{L^2(Q)}$ by quadrature on a tensor grid (the spectral
  closed forms of the constant-coefficient cells do not apply), at $K\in\{32,64,128,256,512\}$, to test
  P1–P3 by the growth in $K$ (a CPU computation on `prepost`, no training).
- **Trained relative $L^2$ error** $\delta_\Gamma$ on the evaluation grid (1024 points, 11 time slices),
  median and quartiles over 3 seeds; also at $t=0$ and near $(x^\star,T)$.
- **Reference solution.** Fourier–Galerkin discretisation of $L^X$ on $|k|\le N$ (multiplication by
  $\cos(x-\varphi)$ couples $k$ to $k\pm1$), $u^\star(\cdot,t)=\exp\bigl((T-t)L^X_N\bigr)g$ by matrix
  exponential, with $N=512$ and a convergence check against $N=1024$ (tolerance stated in the
  report of the run). The exact-solution control uses the same reference as its extension.

## 6. Falsification conditions

- P1 is falsified if the quadrature energy of `split_frozen_mean` on LV4 stays bounded as $K$ grows
  from 32 to 512; P2 if that of `split_frozen_singular` grows without bound on LV4.
- If `split_frozen_mean` and `split_frozen_singular` reach the same trained error on LV4 (overlapping
  quartiles), the conjectured trained consequence is not supported, whatever the energies do.
- A reference solution failing its $N=512$ versus $N=1024$ check invalidates the error measurements
  of the cell.

## 7. Cost and implementation

Implementation: $x$-dependent coefficients in the autograd operator and in the analytic bypass;
a Fourier–Galerkin variable-coefficient module (matrix, reference solution, derivatives on demand);
split fields as constant-coefficient semigroups of the frozen operator; new cells and variants;
tests (Galerkin convergence, forcing identity, bypass against autograd). Training: 2 cells × 7
variants × 3 seeds = 42 tasks, about 7 min each (LV2) and 30 min each (LV4), about 13 GPU-hours on
V100 (`akz`). Energy study on `prepost` (non-billed).

## 8. Open decisions

- **[AUTHOR]** Coefficient amplitude $\varepsilon=0.5$ and phase $\varphi=\pi/4$ (generic case), or also a
  case $\varphi=0$ where $x^\star$ is a critical point of the coefficient ($\nu=2$), which the analysis
  predicts to be more favourable still.
- **[AUTHOR]** Whether an order-6 cell is added to test the predicted failure of the singular-point
  freezing at $p=3$.
- **[AUTHOR]** Whether to keep the periodic idealisation or move to a bounded interval with boundary
  conditions for the local-volatility cell.
