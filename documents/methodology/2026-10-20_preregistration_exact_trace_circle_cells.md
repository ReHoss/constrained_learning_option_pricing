# Pre-registration — the circle comparisons on the datum g itself (exact-trace cells)

**Written 2026-10-20, before any training or evaluation of this study.** The author asked for the circle comparisons on the untruncated datum (« je voulais les résultats mais pour la donnée non tronquée », relayed by Astra, 2026-10-20), then chose the retraining option (« go pour (B) »).

## 1. What changes

The datum is the periodised Bernoulli polynomial $g(x)=y^2-y+\tfrac16$, $y=(x\bmod2\pi)/(2\pi)$, with cosine series $\sum_{k\ge1}\cos(kx)/(\pi^2k^2)$ and a first-derivative discontinuity at $x=0$.

On the full-datum cells `g2_bernoulli_full` and `g3_bernoulli_full` (series `full_datum_sampling_2026-10-08`, `full_datum_remaining_choices_2026-10-12`, `soft_constraint_comparison_2026-10-18`), the data-path variants already use $g$:
- the raw constant extension;
- the transported datum;
- the soft constraint.

The spectral fields are cosine sums over $|k|\le4096$:
- the highest-order split;
- the split of orders $2p$ and $1$;
- the full-operator control;
- the reference $u^\ast$.

Their trace is therefore $g_{4096}$. For these variants the initial condition was replaced by $g_{4096}$, and every variant was evaluated against the solution of datum $g_{4096}$. The size of this replacement is $\|g-g_{4096}\|_{L^2}/\|g\|_{L^2}\simeq2.1\times10^{-6}$ (zero mode excluded) and $\|g-g_{4096}\|_\infty\simeq2.47\times10^{-5}$.

The exact-trace cells `g2_bernoulli_exact_trace` and `g3_bernoulli_exact_trace` (code revisions d9c0849, f9c1a8d, 193f56d) evaluate every spectral field and the reference as `ExactTraceAdaptiveBandField`:
- at the datum slice, the closed form of $g$;
- at time $s>0$ since the datum, the cosine series over the smallest power-of-two band $K\ge4096$ with $s\,(cK^{r}-b)\ge40$, up to $2^{21}$.

Beyond $2^{21}$, a reported (never silent) truncation applies, with the bound of the omitted part.

**Amendment, 2026-10-20, before any run.** The author asked to include the convolution method, the square-root smoothing, in the comparisons (« je pense qu'il va falloir évaluer la méthode par convolution aussi »).
- Its components carry the multiplier $zK_1(z)$, $z=karepsilon(s)$, $arepsilon(s)=arepsilon_0 s/T$, which decays like $e^{-z}$. Its band therefore grows like $40/(arepsilon_0 s)$ and exceeds $2^{21}$ for $s<40/(arepsilon_0 2^{21})$, about $3.8	imes10^{-5}$ on `g2` with $arepsilon_0=0.5$.
- At those points the maximum band applies, and the truncation is reported with its bound.
- Its forcing energy on $g$ is infinite (Proposition 4.2), so its sampled objective has no finite mean.

## 2. Design

**Part A — retraining (Jean Zay, V100 `gpu_p13`, `akz@v100`, `qos_gpu-t3`).**
- Variants: the four spectral variants on each exact-trace cell. On `g2`: `split_diffusion`, `split_diffusion_advection`, `graded_chen_mangasarian`, `exact_solution`. On `g3`: `split_principal`, `split_principal_advection`, `graded_chen_mangasarian`, `exact_solution`.
- Seeds 0, 1, 2; 24 tasks.
- Hyperparameters identical to the 2026-10-08 and 2026-10-12 series (20000 iterations, validation selection), checked at `--init-only`.
- A `--debug` smoke test on `qos_gpu-dev` comes first. Series folder: `exact_trace_circle_2026-10-20`.

**Part B — re-evaluation (Jean Zay `prepost`, no training).** The saved models of the data-path variants are re-evaluated against the exact-trace reference of the corresponding exact-trace cell:
- the raw constant extension;
- the transported datum;
- the three soft-constraint weights.

These cover 3 seeds on each cell, 30 models. Their training used $g$, and their selection uses the validation residual, which does not involve the reference, so retraining them would reproduce the same models.

## 3. Predictions

- **E1 (trace).** Every retrained run passes the initial-value check against $g$ itself (`tc_l2` $=0$).
- **E2 (re-evaluation).** For each re-evaluated model, $|\delta^{\mathrm{new}}-\delta^{\mathrm{old}}|\le10^{-3}\,\delta^{\mathrm{old}}$, where $\delta$ is the relative error on the evaluation grid.
- **E3 (retrained medians).** For each spectral variant except the square-root smoothing, and on each cell, the median $\delta$ over the three seeds lies within a factor $1.25$ of the median of the corresponding $K=4096$ variant.
- **E4 (ordering).** On each cell, the medians satisfy: raw constant and transported datum $>$ soft constraint at its best weight $>$ highest-order split; and full-operator control $<$ highest-order split.
- **E5 (bands).** For the variants other than the square-root smoothing, no training point needs a band above $2^{21}$. On `g3`, no such point needs a band above $4096$ for $s\ge3\times10^{-12}$.
- **E6 (square-root smoothing).** On each cell, the median $\delta$ of the square-root smoothing is at least $0.8$ times its median at $K=4096$: $6.59\times10^{-3}$ on `g2` and $33.9$ on `g3`. It also exceeds the median of the highest-order split on the same cell. The number of training points evaluated on the maximum band is reported.

Admissibility follows the earlier series: finite metrics, no gradient-norm safeguard activation, and the initial-value check passed.

## 4. Outcome

Written 2026-10-20 after the runs.

**Runs.**
- Part A: Jean Zay array 1352862, 24 tasks on V100 (`gpu_p13`, `akz@v100`), code revision 893a300, no tracked modification. It was preceded by the `--debug` smoke test 1352861: analytic-versus-autograd cross-checks at most $3\times10^{-8}$, values not reported.
- Part B: re-evaluation job 1352863 (`prepost`). It wrote one `reevaluation_<cell>.yaml` beside the saved models of each of the 18 full-datum runs.
- Band counts: job 1355845 (`prepost`, revision 6c475f3). It replays the seeded samplers of each finished run, without training (`training_sampler_band_counts.py`). A test checks that the replay equals the times drawn by `train_variant`.

All 24 retrained runs are admissible: finite metrics, no gradient-norm safeguard activation, no non-finite gradient norm, and `tc_l2` $=0$.

**Medians of the relative error $\delta$ over the three seeds.** The $K=4096$ column is the median of the same variant on `g2_bernoulli_full` / `g3_bernoulli_full`. Rows marked (re-evaluated) are the saved full-datum models evaluated against the exact-trace reference.

| Cell | Variant | Median $\delta$, exact trace | Median $\delta$, $K=4096$ | Ratio |
|---|---|---|---|---|
| `g2` | highest-order split (`split_diffusion`) | $4.938\times10^{-4}$ | $4.934\times10^{-4}$ | 1.001 |
| `g2` | split of orders $2p$ and $1$ (`split_diffusion_advection`) | $1.068\times10^{-4}$ | $1.070\times10^{-4}$ | 0.997 |
| `g2` | full-operator control (`exact_solution`) | $7.696\times10^{-5}$ | $7.747\times10^{-5}$ | 0.993 |
| `g2` | square-root smoothing (`graded_chen_mangasarian`) | $2.111\times10^{-3}$ | $6.592\times10^{-3}$ | 0.320 |
| `g2` | raw constant (re-evaluated) | $1.383\times10^{-1}$ | $1.383\times10^{-1}$ | 1.000 |
| `g2` | transported datum (re-evaluated) | $1.368\times10^{-1}$ | $1.368\times10^{-1}$ | 1.000 |
| `g2` | soft constraint, weight 1 / 10 / 100 (re-evaluated) | $3.961$ / $2.969$ / $4.658\times10^{-3}$ | $3.960$ / $2.968$ / $4.657\times10^{-3}$ | 1.000 |
| `g3` | highest-order split (`split_principal`) | $4.904\times10^{-4}$ | $4.904\times10^{-4}$ | 1.000 |
| `g3` | split of orders $2p$ and $1$ (`split_principal_advection`) | $4.451\times10^{-4}$ | $4.451\times10^{-4}$ | 1.000 |
| `g3` | full-operator control (`exact_solution`) | $4.403\times10^{-5}$ | $4.403\times10^{-5}$ | 1.000 |
| `g3` | square-root smoothing (`graded_chen_mangasarian`) | $5.213$ | $33.94$ | 0.154 |
| `g3` | raw constant (re-evaluated) | $1.510\times10^{-1}$ | $1.510\times10^{-1}$ | 1.000 |
| `g3` | transported datum (re-evaluated) | $1.408\times10^{-1}$ | $1.408\times10^{-1}$ | 1.000 |
| `g3` | soft constraint, weight 1 / 10 / 100 (re-evaluated) | $17.71$ / $9.746$ / $8.171\times10^{-3}$ | $17.71$ / $9.746$ / $8.170\times10^{-3}$ | 1.000 |

**E1 holds.** `tc_l2` $=0$ in all 24 runs.

**E2 holds.** The largest relative change of $\delta$ among the 30 re-evaluated models is $3.7\times10^{-4}$ (soft constraint, weight 10, on `g2`). For the raw constant extension and the transported datum, it is at most $1.8\times10^{-7}$.

**E3 holds.** The six ratios lie between 0.993 and 1.001.

**E4 holds.** On `g2`, $1.38\times10^{-1}$ and $1.37\times10^{-1}$ exceed $2.97\times10^{-3}$ (soft constraint, weight 10). That value exceeds $4.94\times10^{-4}$ (highest-order split), which exceeds $7.70\times10^{-5}$ (control). On `g3`, the corresponding chain is $1.51\times10^{-1}$ and $1.41\times10^{-1}$, then $8.17\times10^{-3}$ (weight 100), then $4.90\times10^{-4}$, then $4.40\times10^{-5}$.

**E5 holds.** These counts come from the sampler replay. In each of the six runs with the square-root smoothing, the replay finds the first training point above the maximum band at the iteration and the time since the datum of the warning logged during training.
- *Variants other than the square-root smoothing* (18 field–run pairs): no training or validation point lies above the maximum band.
  - On `g2`, the largest band used is $2^{17}=131072$, by 2 to 6 of the $8.192\times10^{7}$ training points of a run.
  - On `g3`, every training point lies on the base band $4096$. The band required at $s=3\times10^{-12}$ is $4096$ for the three fields, and the required band is non-increasing in $s$.
- *Resolution of the sampled times.* The training times are drawn in single precision, so the smallest positive time since the datum is $2^{-24}\simeq5.96\times10^{-8}$ in every run. No field is evaluated during training at a smaller time.

**E6 fails in its first part, and holds in its second.**
- The median $\delta$ of the square-root smoothing is 0.320 times its $K=4096$ median on `g2` and 0.154 times on `g3`, below the predicted lower bound 0.8.
- The decrease holds for each of the six seed pairs:

  | Cell | Seed | $\delta$, $K=4096$ | $\delta$, exact trace | Ratio |
  |---|---|---|---|---|
  | `g2` | 0 | $6.59\times10^{-3}$ | $1.54\times10^{-3}$ | 0.23 |
  | `g2` | 1 | $3.10\times10^{-3}$ | $2.29\times10^{-3}$ | 0.74 |
  | `g2` | 2 | $7.59\times10^{-3}$ | $2.11\times10^{-3}$ | 0.28 |
  | `g3` | 0 | $33.9$ | $3.31$ | 0.098 |
  | `g3` | 1 | $48.9$ | $19.5$ | 0.40 |
  | `g3` | 2 | $13.4$ | $5.21$ | 0.39 |

- The median of the square-root smoothing exceeds that of the highest-order split, by a factor of $4.3$ on `g2` and $1.06\times10^{4}$ on `g3`.
- Points above the maximum band $2^{21}$, out of $8.192\times10^{7}$ training points per run: 3107, 3107 and 3113 on `g2`, and 2293, 2327 and 2317 on `g3` (seeds 0, 1, 2). One validation point, in seed 1 on `g2`, also lies above it. The first of these points lies at a time since the datum between $7.5\times10^{-6}$ and $2.9\times10^{-5}$. There, the bound on the omitted part of the field lies between $2.6\times10^{-20}$ and $6.8\times10^{-11}$. The warnings logged during the runs gave values 3.7 to 7.0 times smaller, $3.7\times10^{-21}$ to $1.8\times10^{-11}$, because they bounded the multiplier $zK_1(z)$ by $e^{-z}$ (corrected 2026-10-21; pre-registration of 2026-10-21, Section 5). At the smallest sampled time, $2^{-24}$, the same bound is $A\,e^{-s c K}K^{1-q}/(q-1)\le A/K=1/(\pi^2 2^{21})\simeq4.8\times10^{-8}$.

**Forcing of the square-root smoothing at identical points (measured).** The two series share the master seeds, hence the sampler draws. At iteration 1 the zero- and first-order components of the logged forcing floor coincide to four digits in every seed. The components of the order of the operator differ as follows.
- On `g3`, the fourth-order component of the batch mean of the squared forcing at iteration 1 of seed 0 is $90.4$ on the exact trace and $2.41\times10^{5}$ at $K=4096$. Its median over the 150 logged iterations is $567$, $444$ and $717$ on the exact trace (seeds 0, 1, 2), against $1.29$, $1.43$ and $1.49\times10^{6}$ at $K=4096$: ratios $2.3$, $3.2$ and $2.1\times10^{3}$.
- On `g2`, the second-order component at iteration 1 coincides to four digits in seeds 0 and 1, and is $2.022\times10^{-3}$ against $2.163\times10^{-3}$ in seed 2. Its medians over the logged iterations differ by at most 3 %.

**Origin of the fourth-order term at $K=4096$ (proved).** Write $h_K(x,s)=\sum_{k=1}^{K}m(k\varepsilon(s))\,\hat g_k\cos(kx)$, with $\hat g_k=1/(\pi^2k^2)$, $m(z)=zK_1(z)$ and $\varepsilon(s)=\varepsilon_0s/T$. Then $\gamma_4\partial_x^4h_K(x,s)=(\gamma_4/\pi^2)\sum_{k=1}^{K}k^2m(k\varepsilon(s))\cos(kx)$.
- *Truncated sum.* Since $m(0)=1$, this tends to $(\gamma_4/\pi^2)\sum_{k=1}^{K}k^2\cos(kx)$ as $K\varepsilon(s)\to0$. With the Dirichlet kernel $D_K(x)=\tfrac12+\sum_{k=1}^{K}\cos(kx)=\sin((K+\tfrac12)x)/(2\sin(x/2))$, for $x\in(0,2\pi)$,
  $$\sum_{k=1}^{K}k^2\cos(kx)=-\partial_{xx}D_K(x)=\frac{(K+\tfrac12)^2\sin((K+\tfrac12)x)}{2\sin(x/2)}+R_K(x),\qquad |R_K(x)|\le\frac{K+\tfrac12}{2\sin^2(x/2)}+\frac{1}{4|\sin(x/2)|^3}.$$
  At every $x$ away from the first-derivative discontinuity, the leading part of the term has amplitude $|\gamma_4|(K+\tfrac12)^2/(2\pi^2|\sin(x/2)|)$, which is $\simeq4.25\times10^{4}/|\sin(x/2)|$ for $\gamma_4=-0.05$ and $K=4096$.
- *Untruncated sum.* The kernel $\kappa_\varepsilon(y)=\varepsilon^2/(2(y^2+\varepsilon^2)^{3/2})$ has Fourier transform $m(|\xi|\varepsilon)$. By the Poisson summation formula, $\sum_{k\ge1}k^2m(k\varepsilon)\cos(kx)=-\pi\sum_{n\in\mathbb Z}\partial_{yy}\kappa_\varepsilon(x+2\pi n)$. Since $|\partial_{yy}\kappa_\varepsilon(y)|\le6\varepsilon^2/|y|^5$, for every $\eta\in(0,\pi)$,
  $$\sup_{x\in[\eta,2\pi-\eta]}\Bigl|\sum_{k\ge1}k^2m(k\varepsilon)\cos(kx)\Bigr|=O(\varepsilon^2)\quad\text{as }\varepsilon\to0.$$

The truncation at $K$ thus adds, wherever $K\varepsilon(s)$ is small, that is for $s$ small compared with $T/(\varepsilon_0K)\simeq3.7\times10^{-4}$ on `g3`, a fourth-order forcing whose amplitude grows like $K^2$ at every point of the circle; the field on $g$ has no such term. At order two, the same computation gives the bounded amplitude $|\gamma_2|/(2\pi^2|\sin(x/2)|)$, with no factor $K^2$. This is consistent with the measured ratios: about $2\times10^{3}$ on `g3` and at most 7 % on `g2`.

**Not established.**
- That the change of the training objective on `g3` causes the decrease of $\delta$: these runs measure both and do not isolate the link.
- The cause of the decrease on `g2`: a factor 3.1 in median, observed in all three seeds, while the training forcing changes by at most 7 %. The spread of $\delta$ over the seeds within the $K=4096$ series is a factor 2.4.

**Consequence for reporting.** For statements about the datum $g$, the exact-trace values of the square-root smoothing ($2.11\times10^{-3}$ on `g2`, $5.21$ on `g3`) supersede the $K=4096$ values ($6.59\times10^{-3}$, $33.9$). On `g3` the $K=4096$ value contains the truncation term above. The other spectral variants change by at most 0.7 %. Fields at $K=4096$ whose decay order is below the order of the operator, such as the graded Gaussian of `full_datum_extra_variants_2026-10-13` on `g3`, are exposed to the same truncation term for $s$ below about $1/(cK^{r})$; this was not checked.

**Reports during the runs.** After training, the cancellation-cutoff diagnostic evaluates the forcing on the datum slice $s=0$ (1024 points). Since that forcing is not defined for $g$, the field evaluated it on the base band $K=4096$ and reported it once at WARNING. This affects the diagnostic quantities `k_star` and `forcing_floor_closed_form`, and none of the errors above.
