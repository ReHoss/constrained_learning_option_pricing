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

To be written after the runs.
