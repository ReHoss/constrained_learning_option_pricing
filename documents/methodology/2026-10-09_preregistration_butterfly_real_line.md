# Pre-registration — the butterfly datum on the real line, without lateral condition

**Written 2026-10-09, before any training run of this study other than a local code-path check.** The author asked on 2026-10-09 for training runs on the butterfly case on $\mathbb R$, « pas de condi laterale mais fait les choses bien », then « lance sur jean zay ». The local check ran 200 iterations per variant on CPU with `--debug`. Its errors are not results of this study and are not reported.

## 1. Question

The boundary paper (Example 3.1, Example 4.1, Example 4.2) treats the butterfly datum
$$g(x)=(1-|x|)^+=(x+1)^+-2x^++(x-1)^+,\qquad x\in\mathbb R,$$
whose first derivative has the jumps $(w_k)_{1\le k\le3}=(1,-2,1)$ at the kink points $(a_k)_{1\le k\le3}=(-1,0,1)$. It is not band-limited, and no Fourier truncation is involved: every extension below is a closed form in $x$ and $t$.

Every earlier trained comparison is on the circle. This study asks two questions on $\mathbb R$.
- **Is the split extension better than the competitors?** The split extension has finite forcing energy (Proposition 4.2 of the paper). The raw extensions, the transported datum and the graded algebraic kernel have infinite forcing energy. The graded Gaussian at half the diffusivity has finite forcing energy.
- **Do the extensions with singular forcing reach the limits the theory predicts?** For each of them, a model whose pointwise residual vanishes almost everywhere is not the solution. Its limit is a closed form (Section 2).

## 2. Design

**Cell `butterfly_real_line`.** The runner works in the backward convention:
$$Pu=\partial_tu+Au=0\ \text{on}\ \mathbb R\times(0,T),\qquad u(\cdot,T)=g,\qquad A=0.125\,\partial_x^2+0.6\,\partial_x-0.3,\qquad T=1.$$
This is Example 3.1 of the paper with $\nu=0.125$, $\mu=0.6$, $\varrho=0.3$, $\ell=1$, $x^\star=0$, after the change of variable $t\mapsto T-t$. These are the values of Figures 1 and 2 of the paper. With $s=T-t$ and $V$ the heat evolution of $g$ at diffusivity $\nu$, the exact solution is
$$u^\star(x,t)=e^{-\varrho s}\,V(x+\mu s,s),\qquad V(y,s)=\sum_kw_k\Bigl[(y-a_k)\,\Phi\Bigl(\tfrac{y-a_k}{\sqrt{2\nu s}}\Bigr)+\sqrt{2\nu s}\,\varphi\Bigl(\tfrac{y-a_k}{\sqrt{2\nu s}}\Bigr)\Bigr].$$

**No lateral condition.** No condition is imposed at the edges of the sampled region.
- Training, validation and evaluation-batch points are uniform on $W\times(0,T)$, with the window $W=[-4.8,4.2]$.
- Measured on a grid of 2001 times, $|u^\star|\le4.3\times10^{-12}$ at both edges of $W$, while $\max|u^\star|\simeq0.996$.
- Without a lateral condition, the residual on $W\times(0,T)$ with the terminal datum does not determine the solution in $W$. Errors are therefore also reported on the interior window $W_{\mathrm{int}}=[-3.6,3.0]$, at distance $1.2$ from the edges, where $|u^\star|\le2.7\times10^{-6}$ at the edges.

**Network.** The network is the residual network of the circle series (width 64, four blocks of two layers). Its input is the affine image of $(x,t)$ in $[-1,1]^2$, in place of the periodic features of the circle, so it has $33537$ parameters.

**Extensions.** All use the linear factor $\lambda(t)=t/T$. The forcing column gives the pointwise forcing. Every field is implemented in `learning_option_pricing/pde/real_line_butterfly_fields.py`.

| Variant | Extension $\Psi$ | Retained operator | Pointwise forcing $P\Psi$ | Forcing energy |
|---|---|---|---|---|
| `convex_raw` | $(t/T)\,g$ | — | $g/T+(t/T)(\mu g'+r_0g)$, plus $(t/T)\,\nu\sum_kw_k\delta_{a_k}$ unseen | infinite |
| `constant_in_time` | $g$ | $0$ (case (i)) | $\mu g'+r_0g$, plus $\nu\sum_kw_k\delta_{a_k}$ unseen | infinite |
| `transported_datum` | $e^{r_0s}g(x+\mu s)$ | $\mu\partial_x+r_0$ (case (ii)) | $0$ off the lines $x+\mu s=a_k$, Dirac masses on them unseen | infinite |
| `split_diffusion` | $V(x,s)$ | $\nu\partial_x^2$ (case (iii)) | $\mu\partial_x\Psi+r_0\Psi$ | finite |
| `split_diffusion_advection` | $V(x+\mu s,s)$ | $\nu\partial_x^2+\mu\partial_x$ | $r_0\Psi$ | finite |
| `graded_gaussian_mismatched` | heat evolution at $\nu/2$ | $\tfrac\nu2\partial_x^2$ | $\tfrac\nu2\partial_x^2\Psi+\mu\partial_x\Psi+r_0\Psi$ | finite |
| `graded_chen_mangasarian` | $g$ convolved with $\tfrac1{2\varepsilon}(1+(x/\varepsilon)^2)^{-3/2}$, $\varepsilon=\varepsilon_0s/T$, $\varepsilon_0=\sqrt{2\nu T}=0.5$ | none | finite at each point | infinite |
| `graded_chen_mangasarian_narrow` | as above with $\varepsilon_0=0.25$ | none | finite at each point | infinite |
| `exact_solution` | $u^\star$ | $A$ (case (iv)) | $0$ | $0$ |

Here $r_0=-\varrho$. For the algebraic kernel, the squared spatial norm of $\partial_x^2\Psi$ is proportional to $1/\varepsilon(t)$, whose time integral diverges logarithmically at $t=T$.

**Limits of the extensions with singular forcing.** Suppose a field $u=\Psi+(1-\lambda)\Phi$ has a pointwise residual that vanishes almost everywhere. Then $Pu$ equals the unseen singular part $f$ of $P\Psi$ in the sense of distributions, with $u(\cdot,T)=g$.
- For `transported_datum`, $f=P\Psi$, so $\Phi=0$ is a minimiser and the limit is $\Psi=h_A$ itself (Proposition 4.4 of the paper).
- For the datum path, $f=c(t)\,\nu\sum_kw_k\delta_{a_k}$, with $c=1$ for `constant_in_time` and $c(t)=t/T$ for `convex_raw`. The limit is $u^\star+w$, where $Pw=f$ and $w(\cdot,T)=0$. By Duhamel's formula,
$$w(x,t)=-\nu\sum_kw_k\int_0^{T-t}c(t+\sigma)\,e^{r_0\sigma}\,G_{2\nu\sigma}(x+\mu\sigma-a_k)\,\mathrm d\sigma,$$
with $G_v$ the centred normal density of variance $v$. It is computed by Gauss–Legendre quadrature and is tested against adaptive quadrature and against its equation.

**Everything else** is unchanged from the validation-selected series:
- $20000$ iterations with Adam and cosine annealing;
- batches of $4096$ points;
- selection by the residual on a fixed validation set of $16384$ points, evaluated every $100$ iterations;
- an evaluation grid of $1024$ equally spaced points of $W$, edges included, times $11$ time slices.

**Seeds and tasks.** The seeds are $\{0,1,2\}$, giving $9\times3=27$ tasks.
- The tasks run on Jean Zay V100: `gpu_p13`, account `akz@v100`, `qos_gpu-t3`.
- One array of 9 tasks is launched per seed. A smoke test runs first on `qos_gpu-dev` with `--debug`.
- The run directories go under `data/ablation_split_extension_trained/butterfly_real_line_2026-10-09/`.
- The aggregation is `split_extension_cross_seed_summary.py --statistics-only` on `prepost`.

**Recorded quantities.**
- **For every run:**
  - $\delta$ (`rel_l2`) on the grid of $W$, and on the grid of $W_{\mathrm{int}}$ (`rel_l2_interior`);
  - the corner errors on the union of the windows $|x-a_k|\le\pi/16$;
  - the validation residual at the retained state and the residual on the evaluation batch;
  - the median of the training forcing channel, and its closed-form value: the mean square of the pointwise forcing on $W\times(0,T)$, recorded as NaN for the algebraic kernel.
- **For the three extensions with singular forcing:**
  - `line_source_correction_relative_l2`, the relative distance $\lVert v-u^\star\rVert/\lVert u^\star\rVert$ of the limit $v$ to $u^\star$. It is a closed form, independent of training.
  - `relative_l2_to_line_source_limit`, the measured relative distance $\lVert\hat u-v\rVert/\lVert v\rVert$ of the trained field to the limit.
- **Not recorded:** the periodic spectra are not defined on $\mathbb R$ and are recorded as absent.

## 3. Predictions (stated before the training runs)

Write $\bar\delta_\Psi$ for the median of $\delta$ over the three seeds.
- **B1 (zero pointwise forcing).** For `transported_datum`:
  - $\bar\delta$ is at least half of its closed-form `line_source_correction_relative_l2`;
  - the median `relative_l2_to_line_source_limit` is smaller than $\bar\delta$.
- **B2 (the datum path reaches $u^\star+w$).** The same two inequalities hold for `constant_in_time` and for `convex_raw`.
- **B3 (split against raw).** $\bar\delta_{\texttt{split\_diffusion}}$ is smaller than half of $\bar\delta$ for each of `convex_raw`, `constant_in_time` and `transported_datum`.
- **B4 (split against kernels).** $\bar\delta_{\texttt{split\_diffusion}}$ is smaller than $\bar\delta$ for each of `graded_gaussian_mismatched`, `graded_chen_mangasarian` and `graded_chen_mangasarian_narrow`.

A prediction is falsified if one of its inequalities fails on the medians. The same inequalities are also evaluated on `rel_l2_interior` and reported, without being predictions.

The predictions rest on different grounds:
- **B1 and B2 assume that training nearly cancels the regular part of the forcing.** The theory gives the limit of a field with zero pointwise residual. It does not predict that training reaches it.
- **B3 and B4 are beliefs.** No proposition implies them. Proposition 4.2 bounds the forcing energy, not the trained error.
- **No prediction** is made for `split_diffusion_advection` or for `exact_solution`. The latter is the control: zero correction gives the exact solution, so its error measures the optimiser-noise floor of the pipeline.

## 4. What is reported

- **All 27 runs.** The medians and quartiles of every recorded quantity. Nothing is selected on the evaluation grid or on the evaluation batch.
- **The closed-form values.** The values of `line_source_correction_relative_l2` and of the closed-form forcing floors are read from the smoke-test log and added to Section 5 before the arrays start.

## 5. Closed-form values from the smoke test

Filled in after the smoke test (Jean Zay job 804755, revision efadce1, 2026-10-09) and before the arrays. These values are closed forms, independent of training. The smoke log prints three digits. The digits below are recomputed at the same revision with the same functions.

**Limits of the extensions with singular forcing.** Relative distance $\lVert v-u^\star\rVert/\lVert u^\star\rVert$ on the evaluation grid:

| Variant | Limit $v$ | `line_source_correction_relative_l2` | Threshold of B1 and B2 (half of it) |
|---|---|---|---|
| `convex_raw` | $u^\star+w$, $c(t)=t/T$ | $0.139849$ | $0.069924$ |
| `constant_in_time` | $u^\star+w$, $c=1$ | $0.234605$ | $0.117303$ |
| `transported_datum` | $h_A$ | $0.220874$ | $0.110437$ |

**Closed-form forcing floors.** Mean square of the pointwise forcing on $W\times(0,T)$:

| Variant | Closed-form floor |
|---|---|
| `convex_raw` | $8.074\times10^{-2}$ |
| `constant_in_time` | $8.667\times10^{-2}$ (exact value $0.78/9$) |
| `transported_datum` | $5\times10^{-34}$ (round-off of zero) |
| `split_diffusion` | $4.246\times10^{-2}$ (times $\lvert W\rvert T=9$: $0.3821$, the energy $E_V$ of Figure 1 of the paper) |
| `split_diffusion_advection` | $5.264\times10^{-3}$ |
| `graded_gaussian_mismatched` | $6.327\times10^{-2}$ |
| `graded_chen_mangasarian`, `graded_chen_mangasarian_narrow` | not defined (infinite) |
| `exact_solution` | $6\times10^{-34}$ (round-off of zero) |

**Smoke-test checks.**
- On the compute node, 98 tests pass (`test/pde/test_real_line_butterfly_fields.py` and `test/experiments`).
- Every analytic-versus-autograd deviation is at most $1.6\times10^{-15}$.
- The gradient-norm safeguard does not activate in any run.
- The cost is about $0.024$ s per iteration on a V100.
- The smoke runs (300 iterations, `--debug`) also measured errors. They are not results of this study and are not reported.

## 6. Outcome (added 2026-10-09, after the runs)

**Runs.** All 27 tasks of Jean Zay job 804861 completed at revision 8e397c8. No task had a non-finite gradient norm or a safeguard activation.

**Aggregation.** The first aggregation (job 804862) failed before reading any value. The closed-form floor had been written to 21 of the 27 summaries as a numpy scalar, which the aggregator's safe loader rejects. The fix is revision 2273e1e. The 21 summaries were rewritten with plain floats, every value was checked unchanged, and the originals were kept as `summary_*.yaml.orig`. The aggregation then ran on `prepost` (job 812533, statistics only): `data/split_extension_cross_seed_summary/2026-10-09-07-33-45Z_butterfly_real_line_statistics/`.

Medians over the seeds $\{0,1,2\}$, with quartiles for $\delta$. All values are measured, except the closed-form column.

| Variant | $\delta$ (`rel_l2`) | $\delta$ on $W_{\mathrm{int}}$ | Validation residual at the retained state | Distance of the limit to $u^\star$ (closed form) | Distance of the trained field to the limit |
|---|---|---|---|---|---|
| `convex_raw` | $0.1399$ $[0.1397,0.1401]$ | $0.1399$ | $1.43\times10^{-4}$ | $0.1398$ | $1.18\times10^{-3}$ |
| `constant_in_time` | $0.2347$ $[0.2337,0.2347]$ | $0.2347$ | $3.14\times10^{-4}$ | $0.2346$ | $1.61\times10^{-3}$ |
| `transported_datum` | $0.2209$ $[0.2209,0.2209]$ | $0.2209$ | $7.80\times10^{-10}$ | $0.2209$ | $2.98\times10^{-5}$ |
| `split_diffusion` | $5.72\times10^{-4}$ $[4.82,5.84]\times10^{-4}$ | $4.78\times10^{-4}$ | $5.34\times10^{-6}$ | — | — |
| `split_diffusion_advection` | $4.30\times10^{-4}$ $[3.67,4.37]\times10^{-4}$ | $3.77\times10^{-4}$ | $8.67\times10^{-7}$ | — | — |
| `graded_gaussian_mismatched` | $8.52\times10^{-4}$ $[7.45,9.13]\times10^{-4}$ | $7.88\times10^{-4}$ | $4.26\times10^{-5}$ | — | — |
| `graded_chen_mangasarian` | $5.09\times10^{-3}$ $[4.47,8.23]\times10^{-3}$ | $4.63\times10^{-3}$ | $4.38\times10^{-4}$ | — | — |
| `graded_chen_mangasarian_narrow` | $8.59\times10^{-3}$ $[7.06,9.84]\times10^{-3}$ | $8.41\times10^{-3}$ | $1.37\times10^{-3}$ | — | — |
| `exact_solution` | $3.25\times10^{-5}$ $[2.41,4.14]\times10^{-5}$ | $2.50\times10^{-5}$ | $7.60\times10^{-10}$ | — | — |

The test of each prediction:
- **B1 holds.** For `transported_datum`, $\delta\simeq0.2209$, above the threshold $0.1104$. The trained field lies at relative distance $3.0\times10^{-5}$ from $h_A$. Its validation residual, $7.80\times10^{-10}$, is that of the exact-solution control, $7.60\times10^{-10}$, with an error larger by a factor of about $6.8\times10^{3}$.
- **B2 holds.**
  - For `constant_in_time`, $\delta\simeq0.2347$, above $0.1173$. For `convex_raw`, $\delta\simeq0.1399$, above $0.0699$.
  - The measured $\delta$ coincides with the closed-form distance of the limit to $u^\star$ to three digits for both.
  - The trained fields lie at $1.6\times10^{-3}$ and $1.2\times10^{-3}$ from $u^\star+w$.
- **B3 holds.** $\delta$ of `split_diffusion` is smaller by factors of about $2.4\times10^{2}$ (`convex_raw`), $4.1\times10^{2}$ (`constant_in_time`) and $3.9\times10^{2}$ (`transported_datum`).
- **B4 holds.** $\delta$ of `split_diffusion` is smaller by factors of about $1.5$ (`graded_gaussian_mismatched`), $8.9$ (`graded_chen_mangasarian`) and $15$ (`graded_chen_mangasarian_narrow`). Against the graded Gaussian, the interquartile ranges over the three seeds are disjoint.

**On the interior window.** Every inequality of B1–B4 also holds with $\delta$ evaluated on $W_{\mathrm{int}}$, which was reported without being a prediction.

**Without a prediction.** `split_diffusion_advection` (remainder of order zero) has a smaller $\delta$ than `split_diffusion`, $4.30\times10^{-4}$ against $5.72\times10^{-4}$. The exact-solution control gives the optimiser-noise floor $\delta\simeq3.2\times10^{-5}$.
