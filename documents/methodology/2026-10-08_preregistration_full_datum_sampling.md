# Pre-registration — singular forcing that pointwise sampling does not see (full datum)

**Written 2026-10-08, before any training run of this study.** The author decided it on 2026-10-08 with « lance les calculs sur le cluster stp ». This is item (2) of the paper plan: the problem with the uncaptured singularity. Item (1b) is pre-registered in `2026-10-08_preregistration_truncation_study.md`.

## 1. Question

Every earlier run trains on the band-limited datum $g_K$. Here the datum is the full periodised Bernoulli polynomial
$$g(x)=y^2-y+\tfrac16,\qquad y=(x\bmod 2\pi)/2\pi,$$
whose Fourier series is $\sum_{k\ge1}\cos(kx)/(\pi^2k^2)$. In the sense of distributions,
$$\partial_x^2g=\tfrac{1}{2\pi^2}+J\,\delta_0,\qquad J=-\tfrac1\pi .$$

**The raw constant extension $\Psi=g$.** Its forcing $Ag$ has the singular part
$$f=J\sum_{j\ge2}c_j\,\delta_0^{(j-2)},$$
where $A=\sum_jc_j\partial_x^j$ is the generator.
- The sampled residual evaluates $g$ and its derivatives pointwise, with autograd differentiating through $x\bmod2\pi$. These values are the classical ones away from $x=0$, so the sampled residual sees only the regular part of the forcing.
- A field of the model class whose pointwise residual vanishes almost everywhere solves $Pu=f$ in the sense of distributions, with $u(\cdot,T)=g$. Hence $u=u^\star+w$, where $Pw=f$ and $w(\cdot,T)=0$.
- With $a_k=\sum_jc_j(ik)^j$, the correction is, in closed form,
$$\hat w_k(t)=-\hat f_k\,\frac{e^{(T-t)a_k}-1}{a_k}.$$

**The question.** Does training reach this wrong limit, with a small sampled residual and an error close to $\lVert w\rVert/\lVert u^\star\rVert$? And how does the highest-order split on the same datum compare? The split's forcing has no singular part (Proposition 4.2).

## 2. Design

- **Cells.** `g2_bernoulli_full` ($G_2$, Black–Scholes in log-price with $\sigma=0.5$, $r=0.03$) and `g3_bernoulli_full` ($G_3$, order four). Both have the full datum.
- **Variants.**
  - `constant_in_time`, with the full datum: datum path, autograd derivatives.
  - The highest-order split: `split_diffusion` on $G_2$, `split_principal` on $G_3$. It is the Fourier sum truncated at the reference band $K_{\mathrm{ref}}=4096$, whose trace is $g_{4096}$.
- **Exact reference.** $u^\star$ is the Fourier sum truncated at $K_{\mathrm{ref}}=4096$.
  - For $t<T$, the truncated modes are damped by at least $e^{-0.125\cdot4096^2(T-t)}$ on $G_2$, and by a stronger factor on $G_3$.
  - At $t=T$, the truncation error of $g$ is at most $\sum_{k>4096}1/(\pi^2k^2)\le2.5\times10^{-5}$ in absolute value. The raw variant has trace $g$ exactly, so its $\delta_\Gamma$ contains this mismatch at the terminal slice. The split has trace $g_{4096}$ and does not.
- **Seeds.** $\{0,1,2\}$.
- **Everything else** is unchanged from the validation-selected series: network, $20000$ iterations, batch, validation selection, and evaluation grid $1024\times11$.
- **Tasks.** $2\times2\times3=12$ tasks, run on Jean Zay V100. The run directories go under `data/ablation_split_extension_trained/full_datum_sampling_2026-10-08/`.
- **Additional recorded quantities**, for the raw variant only:
  - `line_source_correction_relative_l2` $=\lVert w\rVert/\lVert u^\star\rVert$ on the evaluation grid. This is a closed-form prediction, independent of training.
  - `relative_l2_to_line_source_limit` $=\lVert\hat u-(u^\star+w)\rVert/\lVert u^\star+w\rVert$. This is measured.
  - The closed-form forcing floor of the raw variant is recorded as missing (NaN in the archive, `null` in the YAML summary), because the forcing of the full datum has infinite energy.

## 3. Predictions (stated before the training runs)

- **Q1 (the loss does not see the singularity).** For each cell, the median validation residual at the retained state of the raw constant extension on the full datum is smaller than the median of the same extension on $g_{128}$, read from the validation-selected series. For the band-limited datum the forcing grows with $K$; for the full datum the pointwise forcing is bounded.
- **Q2 (the error stays bounded below).** For each cell, the median $\delta_\Gamma$ of the raw constant extension on the full datum is at least half of `line_source_correction_relative_l2`.
- **Q3 (the trained field approaches the wrong limit).** For each cell, the median `relative_l2_to_line_source_limit` is smaller than the median $\delta_\Gamma$ of the same runs. That is, the trained field is closer to $u^\star+w$ than to $u^\star$.
- **Q4 (the method).** For each cell, the median $\delta_\Gamma$ of the highest-order split on the full datum is smaller than half of the median $\delta_\Gamma$ of the raw constant extension on the full datum.

A prediction is falsified for a cell if its inequality fails on the medians.

Two predictions rest on the theory and one is a belief.
- **Q2 and Q3 assume that training nearly cancels the regular part of the forcing.** The theory predicts the limit $u^\star+w$ of a field with zero pointwise residual. It does not predict that training reaches it.
- **Q4 is not implied by any proposition.**

## 4. What is reported

- **All 12 runs**: median and quartiles of $\delta_\Gamma$, the validation residual at the retained state, the residual on the evaluation batch, and, for the raw variant, the two line-source quantities. Nothing is selected on the evaluation part.
- **The closed-form value of `line_source_correction_relative_l2`** for each cell. It is read from the smoke-test log, before the array, and added to Section 5 of this document before the array starts.

## 5. Closed-form prediction from the smoke test

Filled in after the smoke test (job 754987, revision 25680ed, 2026-10-08) and before the array. These values are closed-form, independent of training:

| Cell | `line_source_correction_relative_l2` | Threshold of Q2 (half of it) |
|---|---|---|
| `g2_bernoulli_full` | $0.1384$ | $0.0692$ |
| `g3_bernoulli_full` | $0.1504$ | $0.0752$ |

The smoke runs (300 iterations, `--debug`) also measured errors. They are not results of this study and are not reported.
