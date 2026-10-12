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

To be written after the run.
