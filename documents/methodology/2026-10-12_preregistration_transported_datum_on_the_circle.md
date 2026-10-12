# Pre-registration — transported datum on the circle

**Written 2026-10-12, before any training run of this study.** The author asked on 2026-10-12 to run the retained operator of orders $1$ and $0$ (« Run it ! ») on the circle cells, whose row was empty in the median table of the CMAP talk. No computation of this study has been made before this document. The library tests of commit `856deff` check the new field and are not results.

## 1. Question

In the paper, the transported datum, the evolution of the retained operator $A=\gamma_{\mathcal L^X,1}\partial_x+\gamma_{\mathcal L^X,0}$, is trained on the butterfly datum on $\mathbb R$ only. There it serves Proposition 4.4. Its remainder $B=\mathcal L^X-A$ has order $m_B=2p>p$. The study measures its solution error on the two band-limited circle cells of the talk, Black–Scholes (`g2_bernoulli_bandlimited`, the paper's $G_2$) and order four (`g3_bernoulli_bandlimited`, the paper's $G_3$), under the protocol of the paper's validation-selected series.

## 2. Design

**Cells** (unchanged, runner's backward convention $Pu=\partial_tu+Au$, $T=1$, datum $g_{128}$):
- `g2_bernoulli_bandlimited`: $A=0.125\,\partial^2_{xx}-0.095\,\partial_x-0.03$ (Black–Scholes in log-price, $\sigma=0.5$, $r=0.03$);
- `g3_bernoulli_bandlimited`: $A=-0.05\,\partial^4_{xxxx}+1.3\,\partial_x-0.4$.

**Extension under study.** `transported_datum`: $h(x,t)=e^{r_0(T-t)}\,g_{128}(x+\mu(T-t))$ with $\mu=\gamma_1$ and $r_0=\gamma_0$, so $m_B=2p$.
- Library: `PeriodicExtensionField` kind `transported_datum`, commit `856deff`.
- Closed-form counterpart in the runner: `SplitSemigroupExtension` with the subset of orders $\{0,1\}$.
- The runner's first-batch cross-check of the analytic and autograd forcings (abort above $10^{-3}$) applies.

**Reproducibility control.** The highest-order split, `split_diffusion` on `g2` and `split_principal` on `g3`, is re-run at the same revision. Its medians are compared with the paper's values ($4.93\times10^{-4}$ and $4.91\times10^{-4}$). This shows whether the transported runs can be compared with the paper's series.

**Everything else** is unchanged from the paper's validation-selected series: network, 20000 Adam updates, batches of 4096 points, selection on the fixed validation sample, evaluation grid. The runner defaults of these cells are used, and the hyperparameters recorded in `metadata.yaml` at `--init-only` are compared with those of the series before launch.

**Seeds and tasks.** Seeds $\{0,1,2\}$ give $2$ cells $\times$ $2$ extensions $\times$ $3$ seeds $=12$ tasks.
- Jean Zay V100 (`gpu_p13`), account `akz@v100`, `qos_gpu-t3`, one GPU per task.
- A smoke test on `qos_gpu-dev` with `--debug` precedes the array. Its values are not results.

## 3. Predictions

Every prediction concerns the median over the three seeds of $\delta_\Gamma$, the relative error on the evaluation grid.

- **T1.** On both cells, the transported datum has a larger median $\delta_\Gamma$ than the highest-order split.
- **T2.** On both cells, the median $\delta_\Gamma$ of the transported datum lies within a factor $10$ of the median of $A=0$ (raw constant) in the paper: $1.29\times10^{-3}$ on `g2`, $9.29$ on `g3`.
  - Motivation: the forcing of the transported datum is $c_{2p}\,\partial^{2p}_xh$, whose energy is $\lVert c_{2p}\partial^{2p}_xg_{128}\rVert^2\int_0^Te^{2r_0s}\,\mathrm ds$.
  - At high wavenumbers, this is the energy of $A=0$ multiplied by $(1-e^{2r_0T})/(-2r_0T)$, that is $\simeq0.97$ on `g2` and $\simeq0.69$ on `g3`.
  - The closed-form floors logged by the smoke test are recorded in the outcome.
- **R0 (reproducibility).** The re-run medians of the highest-order split lie within a relative $10^{-2}$ of the paper's values.

Admissibility follows the paper: finite validation and evaluation quantities, exact initial-value check, gradient-norm safeguard reported. A run that is not admissible is reported with its seed and excluded from the tests.

## 4. Outcome

Written 2026-10-12 after Jean Zay array 961770 (smoke test 961769) completed at revision `7a237a7`. The values are read from the `summary_*.yaml` files of the six run directories (`rel_l2` is $\delta_\Gamma$), copied to `data/ablation_split_extension_trained/transported_circle_study_2026-10-12/`. No checkpoint files exist.

**Smoke test** (values not results):
- analytic and autograd forcings coincide to $0$ (`g2`) and $3.4\times10^{-10}$ (`g3`);
- the training floors $9.67\times10^{-3}$ and $6.24\times10^{4}$ match the closed forms $9.96\times10^{-3}$ and $6.19\times10^{4}$.

**Admissibility.** The 12 runs are admissible:
- finite validation and evaluation quantities;
- exact initial-value check ($\mathrm{tc}_{l^2}=0$);
- no gradient-norm safeguard activation and no non-finite norm over the 20000 recorded updates.

Wall times range from $396$ to $433$ s on `g2` and from $1662$ to $2220$ s on `g3`.

**Relative error $\delta_\Gamma$** (measured, 3 seeds):

| Cell | Extension | Seed 0 | Seed 1 | Seed 2 | Median |
|---|---|---|---|---|---|
| `g2` | `transported_datum` | $\simeq1.538\times10^{-3}$ | $\simeq1.526\times10^{-3}$ | $\simeq2.081\times10^{-3}$ | $\simeq1.54\times10^{-3}$ |
| `g2` | `split_diffusion` (control) | $\simeq4.596\times10^{-4}$ | $\simeq4.925\times10^{-4}$ | $\simeq5.657\times10^{-4}$ | $\simeq4.93\times10^{-4}$ |
| `g3` | `transported_datum` | $\simeq4.565$ | $\simeq4.339$ | $\simeq1.803$ | $\simeq4.34$ |
| `g3` | `split_principal` (control) | $\simeq4.915\times10^{-4}$ | $\simeq4.381\times10^{-4}$ | $\simeq9.053\times10^{-4}$ | $\simeq4.91\times10^{-4}$ |

**Predictions.**
- **T1 holds.** The transported datum has a larger median than the highest-order split: $1.54\times10^{-3}$ against $4.93\times10^{-4}$ on `g2`, a factor $\simeq3.1$; $4.34$ against $4.91\times10^{-4}$ on `g3`, a factor $\simeq8.8\times10^{3}$.
- **T2 holds.** The ratio to the median of $A=0$ in the paper is $\simeq1.19$ on `g2` ($1.54\times10^{-3}$ against $1.29\times10^{-3}$) and $\simeq0.47$ on `g3` ($4.34$ against $9.29$), both within a factor $10$.
- **R0 holds.** The re-run controls reproduce the paper's values seed by seed to the seven digits recorded in `generated/individual_results.csv`, hence the medians $4.925126\times10^{-4}$ and $4.914822\times10^{-4}$.

**Observation, not predicted.** On `g3` the transported runs retain early iterates for two seeds: retained updates $1500$ and $1900$, against $12000$ for the third. The evaluation residuals ($\simeq6.5\times10^{4}$ to $6.9\times10^{4}$) remain of the order of the closed-form forcing floor.
