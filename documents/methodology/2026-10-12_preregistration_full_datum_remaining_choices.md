# Pre-registration — the remaining choices of $h$ on the full datum

**Written 2026-10-12, before any training run of this study.** The author asked on 2026-10-12 to train, on the full datum $g$ (no Fourier projection), the four choices of $h$ of the CMAP median table that the full-datum study of 2026-10-08 did not run (« yeah run the other four rows on full datum ! »). No computation of this study has been made before this document. The tests of the runner change are not results.

## 1. Question

The study of 2026-10-08 (`2026-10-08_preregistration_full_datum_sampling.md`) trained two choices on the full datum $g(x)=\sum_{k\ge1}\cos(kx)/(\pi^2k^2)$:
- $A=0$ (raw constant): $\delta_\Gamma\simeq0.138$ on Black–Scholes and $0.151$ at order four, equal to the closed-form limit;
- the highest-order split: $\delta_\Gamma\simeq4.93\times10^{-4}$ and $4.90\times10^{-4}$, equal to its value on $g_{128}$.

This study completes the table on $g$. It runs the four remaining choices on the same cells, `g2_bernoulli_full` (Black–Scholes, the paper's $G_2$) and `g3_bernoulli_full` (order four, the paper's $G_3$):
- the split retaining the orders $2p$ and $1$ ($m_B=0$);
- Chen–Mangasarian with $\alpha=1$;
- the control $A=\mathcal L^X$;
- the transported datum ($A$ of orders $1,0$, $m_B=2p$).

## 2. Design

**Cells.** `g2_bernoulli_full` and `g3_bernoulli_full`, unchanged: runner's backward convention, $T=1$, reference band $4096$.

**How each extension uses the datum.**
- The split $\{2p,1\}$, Chen–Mangasarian and the control are Fourier sums truncated at $4096$ modes. This is how the split of the 2026-10-08 study was built, and how the exact reference $u^*$ is built.
- The transported datum is built on the full datum: `FullDatumTransportedField` gives $h(x,t)=e^{r_0(T-t)}g(x+\mu(T-t))$, differentiated by autograd. This mirrors the raw constant extension of the 2026-10-08 study. Its pointwise residual contains the classical part of its forcing, $c_2g''=c_2/(2\pi^2)$ on Black–Scholes and $0$ at order four, and misses the Dirac line on $x+\mu(T-t)\equiv0$.
  - Its value at $t=T$ is the full datum, so the initial-value check is exact.
  - Its closed-form forcing floor is recorded as NaN, since its energy is infinite.

**Everything else** is unchanged from the paper's validation-selected series: network, 20000 Adam updates, batches of 4096 points, selection on the fixed validation sample, evaluation grid. The hyperparameters recorded at `--init-only` are compared with those of the series before launch.

**Seeds and tasks.** Seeds $\{0,1,2\}$, giving $2$ cells $\times$ $4$ choices $\times$ $3$ seeds $=24$ tasks.
- Jean Zay V100 (`gpu_p13`), account `akz@v100`, `qos_gpu-t3`.
- A smoke test on `qos_gpu-dev` with `--debug` comes first. Its values are not results.

## 3. Predictions

Every prediction concerns the median over the three seeds of $\delta_\Gamma$, the relative error on the evaluation grid.

- **F1.** The split $\{2p,1\}$ on the full datum lies within a relative $10\%$ of its value on $g_{128}$ in the paper: $1.07\times10^{-4}$ on Black–Scholes and $4.45\times10^{-4}$ at order four. Motivation: the highest-order split coincided to three digits between the two data.
- **F2.** On both cells, the control has a smaller median than the highest-order split on the full datum ($4.93\times10^{-4}$ and $4.90\times10^{-4}$).
- **F3.** On both cells, the transported datum has a median larger than $10$ times the highest-order split on the full datum, that is larger than $4.9\times10^{-3}$. Motivation: like $A=0$, its forcing has a singular part that sampling does not see ($m_B=2p$).
- **F4.** On both cells, Chen–Mangasarian has a larger median than the highest-order split on the full datum.

Admissibility follows the paper: finite validation and evaluation quantities, exact initial-value check, gradient-norm safeguard reported. A run that is not admissible is reported with its seed and excluded from the tests.

## 4. Outcome

Written 2026-10-12 after Jean Zay array 991840 (smoke test 991839) completed at revision `1d5e982`. The values are read from the `summary_*.yaml` files of the six run directories (`rel_l2` is $\delta_\Gamma$), copied to `data/ablation_split_extension_trained/full_datum_remaining_choices_2026-10-12/`. No checkpoint files exist. The reference values on $g_{128}$ are read from `RHB_notes/paper/generated/individual_results.csv`.

**Naming.** The variant designated "the split $\{2p,1\}$" in Sections 1 and 3 retains every term of $\mathcal L^X$ except the term of order zero: $A=\mathcal L^X-\gamma_{\mathcal L^X,0}$, hence $m_B=0$. On the two cells $\mathcal L^X$ has the orders $2p$, $1$ and $0$ only, so this is the paper's highest- and first-order split. Since $\gamma_{\mathcal L^X,0}$ is a constant, $A$ and $\gamma_{\mathcal L^X,0}$ commute and the retained evolution is $h=e^{-\gamma_{\mathcal L^X,0}t}u^\star$ in the forward time of the paper. The label $\{2p,1\}$ is withdrawn (author ruling, 2026-10-12). Sections 1 to 3 are kept as written before the run.

**Admissibility.** The 24 runs are admissible:
- finite validation and evaluation quantities;
- exact initial-value check ($\mathrm{tc}_{l^2}=0$);
- no gradient-norm safeguard activation and no non-finite norm over the 20000 recorded updates.

Wall times range from $417$ to $901$ s on `g2` and from $1674$ to $2176$ s on `g3`.

**Relative error $\delta_\Gamma$ on the full datum** (measured, 3 seeds):

| Cell | Choice of $h$ | Seed 0 | Seed 1 | Seed 2 | Median | Median validation residual |
|---|---|---|---|---|---|---|
| `g2` | $A=\mathcal L^X-\gamma_{\mathcal L^X,0}$ | $\simeq1.070\times10^{-4}$ | $\simeq1.449\times10^{-4}$ | $\simeq9.194\times10^{-5}$ | $\simeq1.070\times10^{-4}$ | $\simeq2.81\times10^{-9}$ |
| `g2` | Chen–Mangasarian, $\alpha=1$ | $\simeq6.592\times10^{-3}$ | $\simeq3.104\times10^{-3}$ | $\simeq7.587\times10^{-3}$ | $\simeq6.59\times10^{-3}$ | $\simeq8.28\times10^{-6}$ |
| `g2` | Control $A=\mathcal L^X$ | $\simeq7.706\times10^{-5}$ | $\simeq7.942\times10^{-5}$ | $\simeq7.747\times10^{-5}$ | $\simeq7.75\times10^{-5}$ | $\simeq1.72\times10^{-9}$ |
| `g2` | Transported datum | $\simeq0.13684$ | $\simeq0.13685$ | $\simeq0.13683$ | $\simeq0.1368$ | $\simeq1.73\times10^{-9}$ |
| `g3` | $A=\mathcal L^X-\gamma_{\mathcal L^X,0}$ | $\simeq2.696\times10^{-4}$ | $\simeq4.659\times10^{-4}$ | $\simeq4.451\times10^{-4}$ | $\simeq4.451\times10^{-4}$ | $\simeq2.06\times10^{-7}$ |
| `g3` | Chen–Mangasarian, $\alpha=1$ | $\simeq33.94$ | $\simeq48.94$ | $\simeq13.38$ | $\simeq33.9$ | $\simeq1.03\times10^{7}$ |
| `g3` | Control $A=\mathcal L^X$ | $\simeq3.481\times10^{-5}$ | $\simeq4.403\times10^{-5}$ | $\simeq5.650\times10^{-5}$ | $\simeq4.40\times10^{-5}$ | $\simeq2.33\times10^{-9}$ |
| `g3` | Transported datum | $\simeq0.140781$ | $\simeq0.140780$ | $\simeq0.140778$ | $\simeq0.1408$ | $\simeq2.34\times10^{-9}$ |

**Predictions.**
- **F1 holds.** The median of $A=\mathcal L^X-\gamma_{\mathcal L^X,0}$ on the full datum differs from its median on $g_{128}$ by a relative $7.9\times10^{-4}$ on `g2` ($1.070445\times10^{-4}$ against $1.071288\times10^{-4}$) and $4.1\times10^{-4}$ on `g3` ($4.451310\times10^{-4}$ against $4.449494\times10^{-4}$), both within the $10\%$ of the prediction.
- **F2 holds.** The control is below the highest-order split on the full datum by a factor $\simeq6.4$ on `g2` ($7.75\times10^{-5}$ against $4.93\times10^{-4}$) and $\simeq11$ on `g3` ($4.40\times10^{-5}$ against $4.90\times10^{-4}$).
- **F3 holds.** The transported datum exceeds the threshold $4.9\times10^{-3}$: its median is $\simeq2.8\times10^{2}$ times that of the highest-order split on `g2` ($0.1368$) and $\simeq2.9\times10^{2}$ times on `g3` ($0.1408$).
- **F4 holds.** Chen–Mangasarian exceeds the highest-order split on the full datum by a factor $\simeq13$ on `g2` ($6.59\times10^{-3}$) and $\simeq6.9\times10^{4}$ on `g3` ($33.9$).

**Observations, not predicted.**
- The three seeds of the transported datum give $\delta_\Gamma$ equal to four digits on `g2` and to five digits on `g3`, at validation residuals equal, to within $1\%$, to those of the control ($1.73\times10^{-9}$ against $1.72\times10^{-9}$ on `g2`, $2.34\times10^{-9}$ against $2.33\times10^{-9}$ on `g3`). The medians $0.1368$ and $0.1408$ differ by a relative $1.1\%$ and $6.8\%$ from those of $A=0$ on the full datum in the study of 2026-10-08 ($0.1383$ and $0.1510$). No closed-form limit was computed for the transported datum.
- The control on the full datum coincides with its value on $g_{128}$ to a relative $0.7\%$ on `g2` ($7.75\times10^{-5}$ against $7.69\times10^{-5}$) and to a relative $5\times10^{-5}$ on `g3` ($4.4028\times10^{-5}$ against $4.4026\times10^{-5}$).
- Chen–Mangasarian is larger on the full datum than on $g_{128}$: $6.59\times10^{-3}$ against $1.15\times10^{-3}$ on `g2`, $33.9$ against $2.87$ on `g3`. On `g3` the retained updates are $500$, $1100$ and $5200$, and the median training floor $1.43\times10^{6}$ is smaller than the closed-form floor $1.25\times10^{9}$ (reference band $4096$) by a factor $\simeq8.7\times10^{2}$.
