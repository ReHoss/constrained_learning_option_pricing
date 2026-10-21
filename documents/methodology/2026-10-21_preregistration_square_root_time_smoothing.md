# Pre-registration — the square-root smoothing with the width law of Zhang, Guo and Lu

**Written 2026-10-21, before any training or evaluation of this study.** The author asked: « tu peux lancer une campagne de résultats pour la convolution utilisée par Zhang stp ? ».

## 1. What is compared

**The field of Zhang, Guo and Lu.** The exact-terminal-condition field of Zhang, Guo and Lu (2026, page 9), written in the time to maturity $\tau$, is
$$z(s,\tau)=\tfrac12\Bigl(s-c(\tau)+\sqrt{(s-c(\tau))^2+\beta(\tau)}\Bigr),\qquad c(\tau)=Ke^{-r\tau},\quad \beta(\tau)=K^2(1-e^{-2r\tau}).$$
Two facts fix its transposition to the paper's cases.
- $\tfrac12(y+\sqrt{y^2+\varepsilon^2})$ is the convolution of $y^+$ with the kernel $\kappa_\varepsilon(y)=\varepsilon^2/(2(y^2+\varepsilon^2)^{3/2})$ of the paper (Appendix B). The width is $\varepsilon(\tau)=\sqrt{\beta(\tau)}=K\sqrt{2r\tau}\,(1+O(\tau))$ as $\tau\to0$, a law in $\sqrt\tau$. The paper's square-root smoothing has the linear law $\varepsilon_0\tau/T$.
- The centre $c(\tau)=Ke^{-r\tau}$ is the strike transported by the terms of orders 1 and 0 of the Black–Scholes operator: the evolution of $(s-K)^+$ under $rs\,\partial_s-r$ is $(s-Ke^{-r\tau})^+$. The field is therefore the square-root smoothing of the transported datum.

**Variants** (catalogue names; $s$ is the time since the datum, $\varepsilon_0=\sqrt{2\nu_{\mathrm{ref}}T}$ is the final width of the paper's rule):
- `square_root_smoothing_sqrt_time`: $h(\cdot,s)=\kappa_{\varepsilon(s)}*g$, with $\varepsilon(s)=\varepsilon_0\sqrt{s/T}$. This variant isolates the width law.
- `square_root_smoothing_sqrt_time_transported`: $h(\cdot,s)=\kappa_{\varepsilon(s)}*h_{\mathrm{tr}}(\cdot,s)$, where $h_{\mathrm{tr}}$ is the transported datum (retained operator of orders 1 and 0). This is the construction of Zhang, Guo and Lu.

The proportionality constant of the width is that of the paper's rule. In Zhang, Guo and Lu it is $K\sqrt{2r}$, which has no counterpart on these data.

**Cases.** All three cases use the datum $g$ itself:
- the butterfly datum on $\mathbb R$ (closed forms);
- $G_1$ and $G_2$ on the circle, on the exact-trace cells `g2_bernoulli_exact_trace` and `g3_bernoulli_exact_trace`. For $s>0$, the band criterion becomes $\varepsilon_0\sqrt{s/T}\,K\ge40$, so the band grows like $s^{-1/2}$, with the maximum $2^{21}$ and a reported truncation.

**Comparators**, already measured and not re-run:
- the linear square-root smoothing (`graded_chen_mangasarian`);
- the highest-order split;
- the full-operator control.

Their medians are on the butterfly series of 2026-10-09 and on the exact-trace series of 2026-10-20.

## 2. Expected forcing energy (scaling argument, not proved here)

Near a first-derivative discontinuity, a smoothing of width $\varepsilon$ gives $\|\partial_x^{j}h(\cdot,s)\|^2_{L^2}$ of order $\varepsilon^{3-2j}$. The time derivative contributes at the same order as $j=2$, since $\varepsilon\varepsilon'=\varepsilon_0^2/(2T)$.
- **Order two (butterfly, $G_1$).** $\|\mathcal Lh(\cdot,s)\|^2$ is of order $\varepsilon^{-1}\propto s^{-1/2}$, which is integrable: the forcing energy is finite. For the linear law, $\varepsilon^{-1}\propto s^{-1}$ gives the logarithmic divergence of Proposition 4.1. For Zhang's own field this is proved in the appendix draft of 2026-10-20 (Proposition 2).
- **Order four ($G_2$).** The fourth-order term gives $\varepsilon^{-5}\propto s^{-5/2}$: the forcing energy is infinite, with or without transport.

## 3. Design

- Jean Zay, V100 (`gpu_p13`, `akz@v100`, `qos_gpu-t3`).
- Two variants on each of the three cells, seeds 0, 1 and 2: 18 tasks.
- Hyperparameters identical to the butterfly series of 2026-10-09 and the exact-trace series of 2026-10-20 (20000 iterations, validation selection).
- A `--debug` smoke test on `qos_gpu-dev` first.
- Series folder: `square_root_time_smoothing_2026-10-21`.

## 4. Predictions

Write $\bar\delta$ for the median relative error over the three seeds. All four predictions are intuitive. Z1–Z3 rest on the scaling of Section 2. No proposition implies a trained error.
- **Z1 (order two: finite energy against infinite).** On the butterfly and on $G_1$, both variants have a smaller $\bar\delta$ than the linear square-root smoothing: $5.09\times10^{-3}$ on the butterfly and $2.11\times10^{-3}$ on $G_1$.
- **Z2 (order two: against the split).** On the butterfly and on $G_1$, both variants have a larger $\bar\delta$ than the highest-order split: $5.72\times10^{-4}$ and $4.94\times10^{-4}$. Motivation: the split's forcing is of lower order and bounded.
- **Z3 (order four).** On $G_2$, both variants have $\bar\delta$ larger than ten times that of the highest-order split, that is larger than $4.9\times10^{-3}$.
- **Z4 (transport).** On each case, the transported variant has a smaller $\bar\delta$ than the non-transported one. Motivation: centring on the transported discontinuity removes the first-order part of the forcing. The transport coefficient is $0.6$ on the butterfly, $-0.095$ on $G_1$ and $1.3$ on $G_2$, so the effect is expected to be smallest on $G_1$.

A prediction is falsified if one of its inequalities fails on the medians.

**Admissibility**, as in the earlier series:
- finite metrics;
- no gradient-norm safeguard activation;
- initial-value check `tc_l2` $=0$;
- analytic-versus-autograd cross-check below $10^{-3}$.

**Also reported.**
- The number of training points above the maximum band, by the sampler replay (`training_sampler_band_counts.py`).
- The terminal-target diagnostic. It is expected to be non-finite, because the time derivative of these fields has no finite value at the datum.

## 5. Correction recorded at the same time

The reported bound of the omitted part of a square-root smoothing used $e^{-z}$. The multiplier is $m(z)=zK_1(z)$, which exceeds $e^{-z}$ by up to $\sqrt{\pi z/2}\,(1+3/(8z))$.
- From revision of this pre-registration on, the bound uses $m(z)$ itself. Since $m$ decreases from 1, $m(z)$ bounds every omitted component.
- The bounds quoted in the outcome of the 2026-10-20 study are corrected there.

## 6. Outcome

Written 2026-10-21 after the runs.

**Runs.**
- Jean Zay array 1371281, 18 tasks on V100 (`gpu_p13`, `akz@v100`), revision 9c3b863, no tracked modification. Hyperparameters coincide with the butterfly series of 2026-10-09 and the exact-trace series of 2026-10-20, checked at `--init-only`.
- Before the array:
  - a first smoke test (1370738) failed before any computation: the variants were missing from the `--variant` choices (fixed in a3fa3d4, with a test);
  - the second smoke test (1370793) passed on the six cases;
  - it showed an infinite closed-form forcing floor on the circle, a rounding artefact of the quadrature (fixed in 9c3b863; CPU check 1370969: $1.09\times10^{-4}$ against a training median of $1.02\times10^{-4}$);
  - 350 tests passed.

**Admissibility.** All 18 runs are admissible:
- `tc_l2` $=0$;
- no gradient-norm safeguard activation and no non-finite norm;
- analytic-versus-autograd cross-check at most $9.0\times10^{-8}$.

**Medians of $\delta$ over the three seeds.** The comparators are read from their saved series.

| Case | Square-root time | Square-root time, transported | Linear square-root smoothing | Highest-order split | Full-operator control |
|---|---|---|---|---|---|
| Butterfly on $\mathbb R$ | $1.198\times10^{-3}$ | $1.172\times10^{-3}$ | $5.087\times10^{-3}$ | $5.719\times10^{-4}$ | $3.247\times10^{-5}$ |
| $G_1$ on $\mathbb T$ | $7.650\times10^{-4}$ | $6.177\times10^{-4}$ | $2.111\times10^{-3}$ | $4.938\times10^{-4}$ | $7.696\times10^{-5}$ |
| $G_2$ on $\mathbb T$ | $0.3979$ | $0.2667$ | $5.213$ | $4.904\times10^{-4}$ | $4.403\times10^{-5}$ |

Individual values:
- butterfly: square-root time 1.198, 1.124 and 1.921, transported 1.172, 1.385 and $0.933\times10^{-3}$;
- $G_1$: 0.765, 1.098 and 0.747, transported 0.480, 1.049 and $0.618\times10^{-3}$;
- $G_2$: 0.398, 0.667 and 0.137, transported 0.267, 0.706 and 0.188 (seeds 0, 1, 2).

**Z1 holds.** At order two, both variants are below the linear smoothing: by factors 0.236 and 0.230 on the butterfly, and 0.362 and 0.293 on $G_1$.

**Z2 holds.** At order two, both variants are above the highest-order split: by factors 2.10 and 2.05 on the butterfly, and 1.55 and 1.25 on $G_1$.

**Z3 holds.** On $G_2$, the medians are 811 and 544 times that of the highest-order split. They are also below the linear smoothing, by factors of about 13 and 20.

**Z4 holds on the medians,** with ratios transported to plain of 0.978 (butterfly), 0.807 ($G_1$) and 0.670 ($G_2$). The three seed ranges of the two variants overlap in every case, so three seeds do not separate them. On the butterfly the difference of the medians is 2.2 %.

**Forcing diagnostics (recorded by the runs).**
- On the butterfly, the median training forcing floor is $4.12\times10^{-2}$ without transport (closed form $4.14\times10^{-2}$) and $1.86\times10^{-3}$ with transport (closed form $2.17\times10^{-3}$). Centring on the transported discontinuities divides the floor by about 22, while the median error changes by 2.2 %.
- On $G_1$, the closed-form floors are $1.09\times10^{-4}$ and $7.40\times10^{-5}$.
- On $G_2$, the closed-form floor on the base band, $8.77\times10^{5}$, is that of the band $K=4096$; Section 2 predicts an infinite value on $g$, which this finite band value does not test.

**Bands.**
- No training point lies above the maximum band.
- With the law in $\sqrt s$, the band $2^{21}$ is reached only for $s<(40/(\varepsilon_0 2^{21}/\sqrt T))^2$: $1.5\times10^{-9}$ on $G_1$ and $8.1\times10^{-10}$ on $G_2$, both below the smallest sampled time $2^{-24}\simeq6.0\times10^{-8}$.

**Terminal-target diagnostic.**
- Non-finite on the circle, as predicted.
- On the butterfly it is finite (0.975 to 0.979), because the evaluation grid avoids the discontinuities, where the almost-everywhere limit $\varepsilon_0^2/(4T|y-a_k|)$ is infinite. The value depends on the grid and is not reported as a result.

**Not established.**
- That the finiteness of the forcing energy at order two causes the gain over the linear smoothing: the runs measure both, not the link.
- The effect of the transport (Z4) is not resolved by three seeds.
