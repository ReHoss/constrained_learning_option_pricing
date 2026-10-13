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

To be written after the run.
