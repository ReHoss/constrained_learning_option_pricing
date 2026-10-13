# Pre-registration — variable-coefficient generators on the full datum

**Written 2026-10-13, before any training run of this study.** The author asked on 2026-10-13 to train, on the full datum $g$ (no Fourier projection), the variable-coefficient versions of the two circle equations of the CMAP talk (« peux tu faire les entrainements sur les memes EDP que les slides avec la donnée g non tronquée ? »). The runner change and its tests (catalogue cells `lv*_bernoulli_full_*`, `FullDatumGalerkinReference`) are not results.

## 1. Question

The variable-coefficient campaign of 2026-09-29 (`2026-09-29_preregistration_variable_coefficient_split.md`) trained the cells LV2 and LV4 on the band-limited datum $g_{128}$ only, with the selection protocol of that date (weights retained on the training loss). Both cells are the circle equations of the talk with one coefficient made variable:
- **LV2**, Black–Scholes in log-price with local volatility: $\mathcal L^X u=a(x)\,\partial_{xx}u+(r-a(x))\,\partial_xu-r\,u$, $a(x)=0.125\,(1+\varepsilon\cos(x-\pi/4))$, $r=0.03$;
- **LV4**, order four: $\mathcal L^X u=-\beta(x)\,\partial_{xxxx}u+1.3\,\partial_xu-0.4\,u$, $\beta(x)=0.05\,(1+\varepsilon\cos(x-\pi/4))$.

With variable coefficients, no constant-coefficient retained operator $A$ reproduces the term of order $2p$, so the remainder $B=\mathcal L^X-A$ has order $m_B=2p$ and Proposition 4.2 asks for $g\in H^{p}$. The full datum lies in $H^1$ and not in $H^2$. The study measures the trained error on $g$ and compares it with $g_{128}$ at the same revision.

## 2. Design

**Cells.** For $\varepsilon\in\{0.25,0.75\}$:
- full datum: `lv2_bernoulli_full_eps0p25`, `lv2_bernoulli_full_eps0p75`, `lv4_bernoulli_full_eps0p25`, `lv4_bernoulli_full_eps0p75`;
- band-limited baseline, re-run at the same revision and on the same hardware: `lv2_bernoulli_bandlimited_eps0p25`, `lv2_bernoulli_bandlimited_eps0p75`, `lv4_bernoulli_bandlimited_eps0p25`, `lv4_bernoulli_bandlimited_eps0p75`.

**Choices of $h$** (the same three in every cell):
- `constant_in_time`, $A=0$: on the full-datum cells, the unprojected datum with autograd derivatives (datum path), as in the full-datum study of 2026-10-08;
- `split_frozen_singular`: $A$ retains the term of order $2p$ with its coefficient frozen at the break point $x^\star=0$;
- `split_frozen_mean`: the same with the coefficient frozen at its mean.

On the full-datum cells, the two frozen splits are Fourier sums truncated at $4096$ components.

**Reference.** Fourier–Galerkin solution with truncation $N=512$ (order 2) and $N=256$ (order 4), as in the 2026-09-29 campaign. On the full-datum cells it is built on $g_{N-1}$ and returns the full datum at $t=T$, where $u^\star(\cdot,T)=g$. For $t<T$ the components above $N$ are damped by at least $e^{-cN^{2p}(T-t)}$. The smoke test reports the deviation of the reference between $N$ and $\tfrac32N$ at the evaluation times $t<T$.

**Protocol.** That of the paper's validation-selected series, unchanged: network, 20000 Adam updates, batches of 4096 points, selection on the fixed validation sample, evaluation grid $1024\times11$. Seeds $\{0,1,2\}$. This gives $8$ cells $\times$ $3$ choices $\times$ $3$ seeds $=72$ tasks.
- Jean Zay V100 (`gpu_p13`), account `akz@v100`, `qos_gpu-t3`.
- A smoke test on `qos_gpu-dev` with `--debug` precedes the array. Its values are not results.
- The closed-form forcing floor of each frozen split is recorded in the run summary, at the band of its cell ($128$ or $4096$).

## 3. Predictions

Every prediction concerns the median over the three seeds of $\delta_\Gamma$, the relative error on the evaluation grid, at the same revision. The heuristic of the 2026-09-29 pre-registration (§4: the forcing is square-integrable if $p<\tfrac32+\nu$, with $\nu$ the order of vanishing of the coefficient difference at $x^\star$) is not a proof.

- **V1 (order 2, both freezings).** On LV2, for both $\varepsilon$ and both freezings, the median on the full datum lies within a relative $10\%$ of the median on $g_{128}$. Motivation: Proposition 4.2 with $p=1$, $m_B=2$ bounds $E_h$ by $\lVert g\rVert^2_{H^1}$, uniformly in the truncation.
- **V2 (order 4, frozen at the mean).** On LV4, for both $\varepsilon$, the median of `split_frozen_mean` on the full datum exceeds its median on $g_{128}$ by more than $10\%$. Motivation: the heuristic gives $\nu=0$ and $p=2\not<\tfrac32$, a forcing energy growing linearly with the band.
- **V3 (order 4, frozen at the singular point).** On LV4, for both $\varepsilon$, the median of `split_frozen_singular` on the full datum lies within a relative $10\%$ of its median on $g_{128}$. Motivation: the heuristic gives $\nu=1$ and $p=2<\tfrac52$.
- **V4 (order 4, comparison of the freezings).** On the full-datum LV4 cells, for both $\varepsilon$, `split_frozen_singular` has a smaller median than `split_frozen_mean`.
- **V5 ($A=0$).** On the four full-datum cells, `constant_in_time` has a median larger than $10$ times that of `split_frozen_singular`.

Admissibility follows the paper: finite validation and evaluation quantities, exact initial-value check, gradient-norm safeguard reported. A run that is not admissible is reported with its seed and excluded from the tests.

## 4. Outcome

Written 2026-10-13 after Jean Zay array 1017682 (smoke test 1017681) completed at revision `a252181`. The values are read from the `summary_*.yaml` files of the 24 run directories (`rel_l2` is $\delta_\Gamma$), copied to `data/ablation_split_extension_trained/variable_coefficient_full_datum_2026-10-13/`. No checkpoint files exist.

**Smoke test** (not results). The Galerkin reference changes by a relative $9\times10^{-13}$ to $2\times10^{-12}$ (order 2) and $1.5\times10^{-8}$ to $3.0\times10^{-8}$ (order 4) between $N$ and $\tfrac32N$, at the evaluation times $t<T$.

**Admissibility.** The 72 runs are admissible:
- finite validation and evaluation quantities;
- exact initial-value check ($\mathrm{tc}_{l^2}=0$);
- no gradient-norm safeguard activation and no non-finite norm over the 20000 recorded updates.

Wall times range from $408$ to $702$ s at order 2 and from $1647$ to $2116$ s at order 4.

**Median relative error $\delta_\Gamma$** (measured, 3 seeds; same revision and hardware for both data):

| Cell | Choice of $h$ | $g_{128}$ | $g$ | Change from $g_{128}$ to $g$ |
|---|---|---|---|---|
| LV2, $\varepsilon=0.25$ | $A=0$ | $\simeq1.23\times10^{-3}$ | $\simeq0.157$ | factor $\simeq1.3\times10^{2}$ |
| LV2, $\varepsilon=0.25$ | frozen at $x^\star$ | $\simeq4.520\times10^{-4}$ | $\simeq4.528\times10^{-4}$ | $+0.2\%$ |
| LV2, $\varepsilon=0.25$ | frozen at the mean | $\simeq5.698\times10^{-4}$ | $\simeq5.733\times10^{-4}$ | $+0.6\%$ |
| LV2, $\varepsilon=0.75$ | $A=0$ | $\simeq1.87\times10^{-3}$ | $\simeq0.193$ | factor $\simeq1.0\times10^{2}$ |
| LV2, $\varepsilon=0.75$ | frozen at $x^\star$ | $\simeq4.027\times10^{-4}$ | $\simeq4.034\times10^{-4}$ | $+0.2\%$ |
| LV2, $\varepsilon=0.75$ | frozen at the mean | $\simeq7.168\times10^{-4}$ | $\simeq6.649\times10^{-4}$ | $-7.2\%$ |
| LV4, $\varepsilon=0.25$ | $A=0$ | $\simeq11.1$ | $\simeq0.165$ | factor $\simeq0.015$ |
| LV4, $\varepsilon=0.25$ | frozen at $x^\star$ | $\simeq5.558\times10^{-4}$ | $\simeq5.461\times10^{-4}$ | $-1.7\%$ |
| LV4, $\varepsilon=0.25$ | frozen at the mean | $\simeq1.354\times10^{-3}$ | $\simeq1.389\times10^{-3}$ | $+2.6\%$ |
| LV4, $\varepsilon=0.75$ | $A=0$ | $\simeq5.22$ | $\simeq0.190$ | factor $\simeq0.036$ |
| LV4, $\varepsilon=0.75$ | frozen at $x^\star$ | $\simeq1.108\times10^{-3}$ | $\simeq1.086\times10^{-3}$ | $-2.0\%$ |
| LV4, $\varepsilon=0.75$ | frozen at the mean | $\simeq2.107\times10^{-3}$ | $\simeq2.382\times10^{-3}$ | $+13.0\%$ |

**Closed-form forcing floors of the frozen splits** (band $128$ against band $4096$): at order 2 they change by at most $0.4\%$; at order 4, frozen at $x^\star$, by less than $10^{-4}$ relative; at order 4, frozen at the mean, they grow by a factor $\simeq2.5$ ($\varepsilon=0.25$, $1.07\times10^{-2}$ to $2.66\times10^{-2}$) and $\simeq10.9$ ($\varepsilon=0.75$, $1.45\times10^{-2}$ to $0.158$).

**Predictions.**
- **V1 holds.** On LV2 the four medians on $g$ lie within a relative $10\%$ of those on $g_{128}$ (largest change $-7.2\%$, frozen at the mean, $\varepsilon=0.75$).
- **V2 does not hold.** On LV4 the median of the split frozen at the mean increases by $+13.0\%$ at $\varepsilon=0.75$, beyond the $10\%$ of the prediction, but by $+2.6\%$ only at $\varepsilon=0.25$. The closed-form forcing floor grows with the band, as the heuristic states; the trained error changes by less than the prediction at the smaller amplitude.
- **V3 holds.** On LV4 the median of the split frozen at $x^\star$ changes by $-1.7\%$ and $-2.0\%$.
- **V4 holds.** On the full-datum LV4 cells the split frozen at $x^\star$ has the smaller median: by a factor $\simeq2.5$ ($\varepsilon=0.25$) and $\simeq2.2$ ($\varepsilon=0.75$).
- **V5 holds.** On the four full-datum cells $A=0$ exceeds the split frozen at $x^\star$ by a factor $\simeq3.5\times10^{2}$, $\simeq4.8\times10^{2}$ (LV2) and $\simeq3.0\times10^{2}$, $\simeq1.7\times10^{2}$ (LV4).

**Observations, not predicted.**
- On the full datum the three seeds of $A=0$ give $\delta_\Gamma$ equal to two or three digits ($0.1569$ to $0.1574$, $0.1925$ to $0.1929$, $0.1644$ to $0.1652$, $0.1891$ to $0.1899$), at validation residuals of $10^{-6}$ (LV2) and $10^{-4}$ (LV4), as for the constant-coefficient cells. No closed-form limit is computed with variable coefficients.
- At order 4, $A=0$ has a much smaller error on $g$ than on $g_{128}$ ($0.165$ against $11.1$, $0.190$ against $5.22$), with the retained updates of the $g_{128}$ runs between $100$ and $4500$. The same ordering holds at constant coefficients (order four on $\mathbb T$: $0.151$ on $g$, $9.29$ on $g_{128}$).
- On LV4 the split frozen at the mean has one seed with a larger error on both data (seed 1: $5.2\times10^{-3}$ and $6.0\times10^{-3}$ at $\varepsilon=0.25$, $1.3\times10^{-2}$ and $6.6\times10^{-3}$ at $\varepsilon=0.75$), retained at earlier updates ($8000$ to $14800$).
