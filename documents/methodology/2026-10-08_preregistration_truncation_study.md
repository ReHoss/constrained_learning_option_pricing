# Pre-registration — trained extensions under refinement of the datum band edge

**Written 2026-10-08, before any run of this study.** The author decided it on 2026-10-08 with « lance les calculs sur le cluster stp ». The decision follows the paper plan of the same day: illustrate (1) the non-admissible extension and its effect on the loss under refinement, (2) the singular forcing that pointwise sampling does not see, and (3) the efficiency of the method. This document covers item (1b), the trained part of (1). Item (2) is pre-registered separately.

## 1. Question

The constant-coefficient cells train on the band-limited datum
$g_K(x)=\sum_{k=1}^{K}\cos(kx)/(\pi^2k^2)$, with $K=128$ in every earlier run. The question is how the trained solution error depends on $K$ for two kinds of extension:
- **The raw extensions**, whose forcing energy grows without bound as $K\to\infty$. For the full datum they are not initial-data extensions (Corollary of the paper).
- **The highest-order split**, whose forcing energy is bounded uniformly in $K$ (Proposition 4.2).

## 2. Forcing energies (already measured, no new computation)

The canonical energy study `data/measure_variable_coefficient_forcing_energy/2026-09-29-11-37-05-597447Z_Kmax1024_N512/` contains the amplitude $\varepsilon=0$. It was computed in closed form at code revision c3e6051, with a passed check against the constant-coefficient closed forms to $10^{-10}$. At $\varepsilon=0$ the two variable-coefficient families coincide with the cells of this study:
- LV2 at $\varepsilon=0$ is $G_2=0.125\,\partial_x^2-0.095\,\partial_x-0.03$, the Black–Scholes operator in log-price with $\sigma=0.5$ and $r=0.03$;
- LV4 at $\varepsilon=0$ is $G_3=-0.05\,\partial_x^4+1.3\,\partial_x-0.4$.

Strip forcing energies $E$, read from `summary.yaml`:

| Cell | Extension | $K=32$ | $K=128$ | $K=512$ |
|---|---|---|---|---|
| $G_2$ | Raw constant | $0.0170$ | $0.0654$ | $0.259$ |
| $G_2$ | Raw linear | $0.0330$ | $0.0491$ | $0.114$ |
| $G_2$ | Highest-order split | $3.52\times10^{-4}$ | $3.52\times10^{-4}$ | $3.52\times10^{-4}$ |
| $G_3$ | Raw constant | $584$ | $5.65\times10^{5}$ | $5.70\times10^{8}$ |
| $G_3$ | Raw linear | $195$ | $1.88\times10^{5}$ | $1.90\times10^{8}$ |
| $G_3$ | Highest-order split | $0.0647$ | $0.0647$ | $0.0647$ |

On $G_2$ the raw-linear energy is dominated at small $K$ by the term $-g_K/T$, whose norm does not grow with $K$. Its local log-log slope is $0.22$ at $K=32$ and $0.81$ at $K=512$. The raw-constant energy grows with slope $0.96$ to $1.00$ over the same range. Both raw extensions are therefore trained: the raw-constant one isolates the growth at order two.

## 3. Design

- **Cells.** $G_2$ and $G_3$. $G_1$ is not included: the author was offered its removal from the paper on 2026-10-08, because $G_2$ has the Black–Scholes interpretation and is the constant-coefficient case of the LV2 family.
- **Variants.** `constant_in_time` (raw constant), `convex_raw` (raw linear), and the highest-order split, which is the comparator pre-registered for stage 2 (`split_diffusion` on $G_2$, `split_principal` on $G_3$).
- **Band edges.** $K\in\{32,128,512\}$, through the runner option `--truncation-wavenumber`.
- **Seeds.** $\{0,1,2\}$.
- **Everything else** is unchanged from the validation-selected series (`2026-09-29_preregistration_validation_selected_series.md`):
  - network and optimiser;
  - $20000$ iterations, with fresh training batches of $4096$ points;
  - retention of the state of lowest residual on a fixed validation set of $16384$ points, evaluated every $100$ iterations;
  - evaluation of $\delta_\Gamma$ on the $1024\times11$ grid against the exact solution for the datum $g_K$.
- **Tasks.** $2\times3\times3\times3=54$ tasks, run on Jean Zay V100 (`gpu_p13`, `qos_gpu-t3`, `akz@v100`). The run directories go under `data/ablation_split_extension_trained/truncation_study_2026-10-08/K<K>/`. Each band edge is aggregated separately, with `--data-root` and `--band-edge <K>`; the aggregator raises if a run's recorded band edge differs.

## 4. Predictions (stated before the runs)

- **P1 (raw extensions).** For each cell and each raw variant, the median $\delta_\Gamma$ over the three seeds increases from $K=32$ to $K=512$.
- **P2 (highest-order split).** For each cell, the medians of $\delta_\Gamma$ at $K=32$, $128$ and $512$ lie within a factor $2$ of one another.
- **P3 (gap).** For each cell and each raw variant, the ratio of the raw median to the split median is larger at $K=512$ than at $K=32$.

A prediction is falsified for a (cell, variant) pair if its inequality fails on the medians. No prediction concerns the size of the effect beyond these inequalities.

P1 and P3 do not follow from the forcing energies. The energies are not error bounds: the cross term in the expansion of the objective can be negative, and a lower forcing energy was measured not to imply a lower error (Chen–Mangasarian against the raw-linear extension on $G_1$ and $G_2$). The predictions express the expectation that a network of fixed size and budget fits a forcing with more high-frequency content less accurately.

## 5. What is reported

- **All 54 runs.** Median and quartiles of $\delta_\Gamma$ per (cell, variant, $K$); the residual on the evaluation batch; the retained iteration. Nothing is selected on the evaluation part.
- **The forcing energies of Section 2** for the same cells and band edges.
- **A reproducibility check at $K=128$.** These runs repeat the configuration of the validation-selected series at a later code revision. Their difference from that series is reported as a check, and the two are not pooled.
- **Failures.** A run that produces non-finite values is reported as failed, with its iteration and the gradient-norm safeguard record. It is never dropped silently. The safeguard (threshold $10^{12}$) records every activation. The raw forcing on $G_3$ at $K=512$ has energy of order $10^{8}$, so activations are possible there.

## 6. Smoke test

Before the array, one `qos_gpu-dev` job runs:
- the full test suite;
- two `--debug` runs of $300$ iterations at $K=512$: raw constant on $G_3$, which has the largest forcing, and the highest-order split on $G_2$.

The job checks that the code runs and gives the time per iteration, from which the array time limit is set.

## 7. Outcome (added 2026-10-08, after the runs)

**Runs.** All 54 tasks of Jean Zay job 754879 completed, at revision 258695d. No task had a non-finite gradient norm or a safeguard activation.

**Aggregates.** They were computed on `prepost` by job 757898, one per band edge (`--band-edge K`):
- `data/split_extension_cross_seed_summary/2026-10-08-03-33-29Z_truncation_study_K{32,128,512}_statistics/` (statistics);
- `data/split_extension_cross_seed_summary/2026-10-08-03-33-29Z_truncation_study_K{32,128,512}/` (closed forms, tables, figures).

Median $\delta_\Gamma$ over seeds $\{0,1,2\}$ (measured):

| Cell | Extension | $K=32$ | $K=128$ | $K=512$ |
|---|---|---|---|---|
| $G_2$ | Raw constant | $1.32\times10^{-3}$ | $1.29\times10^{-3}$ | $2.76\times10^{-3}$ |
| $G_2$ | Raw linear | $1.04\times10^{-3}$ | $1.21\times10^{-3}$ | $2.53\times10^{-3}$ |
| $G_2$ | Highest-order split | $4.94\times10^{-4}$ | $4.93\times10^{-4}$ | $4.94\times10^{-4}$ |
| $G_3$ | Raw constant | $0.129$ | $9.29$ | $48.2$ |
| $G_3$ | Raw linear | $0.114$ | $3.80$ | $11.3$ |
| $G_3$ | Highest-order split | $6.24\times10^{-4}$ | $4.91\times10^{-4}$ | $4.90\times10^{-4}$ |

The test of each prediction:
- **P1 holds** for the four (cell, raw variant) pairs. From $K=32$ to $K=512$ the median grows by a factor $2.1$ (raw constant, $G_2$), $2.4$ (raw linear, $G_2$), $372$ (raw constant, $G_3$) and $99$ (raw linear, $G_3$). On $G_2$ the raw-constant median is not monotone: at $K=128$ it is slightly below its value at $K=32$, and P1 compares only $K=32$ with $K=512$.
- **P2 holds.** The largest-to-smallest ratio of the split medians is $1.003$ on $G_2$ and $1.27$ on $G_3$.
- **P3 holds** for the four pairs. The ratio of the raw median to the split median goes:
  - from $2.7$ to $5.6$ (raw constant, $G_2$);
  - from $2.1$ to $5.1$ (raw linear, $G_2$);
  - from $2.1\times10^{2}$ to $9.8\times10^{4}$ (raw constant, $G_3$);
  - from $1.8\times10^{2}$ to $2.3\times10^{4}$ (raw linear, $G_3$).

Further observations:
- **Retained states of the raw extensions on $G_3$.** At $K=128$ and $512$, the retained iterations are early (medians $400$ to $5700$). The validation residual at the retained state is between $2.9\times10^{4}$ and $8.2\times10^{7}$. The network does not cancel the forcing.
- **Reproducibility check.** At $K=128$, the medians of $\delta_\Gamma$ and of the validation residual are equal, to the seven digits printed, to those of the validation-selected series (`2026-09-29-14-47-30Z_series_validation_selected_8cell_3seed`). This holds for all six (cell, variant) pairs. As stated in Section 5, the two are not pooled.
