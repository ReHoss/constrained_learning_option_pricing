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

To be written after the runs.
