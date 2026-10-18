# Pre-registration — the soft constraint against the hard-constrained choices of $h$

**Written 2026-10-18, before any training run of this study.** After the meeting with Charles-Albert Lehalle, the author asked to set the hard-constrained approach of the paper against the soft constraint (« il faut d'une manière ou d'une autre mettre l'approche en opposition avec l'approche "soft constraints" »), and then to run the soft constraint in the experiments (« Go ahead », 2026-10-18). No computation of this study has been made before this document. The tests of the runner change are not results.

## 1. Question

In the paper, the trial field $\Phi^h_\theta=h+d_0\Psi_\theta$ satisfies the initial condition exactly, and the objective is the residual of the equation alone. In the physics-informed formulation of Raissi et al. (2019, preprint arXiv 1711.10561, equation (4)), the trial field is the network $\Psi_\theta$ itself, and the misfit to the initial data is added to the residual of the equation in the objective. This is the soft constraint of Lu et al. (2021). This study measures, on the full datum and on the butterfly datum, the solution error of the soft constraint against the hard-constrained choices of $h$ already measured under the same protocol.

## 2. Design

**Cells.** The three cells of the paper's results, unchanged:
- `g2_bernoulli_full` (Black–Scholes generator on $\mathbb T$, full datum $g$);
- `g3_bernoulli_full` (order four on $\mathbb T$, full datum $g$);
- `butterfly_real_line` (butterfly datum on $\mathbb R$, window $[-4.8,4.2]$).

**Soft-constraint variants.** In the runner's backward convention the datum is imposed at $t=T$. The trial field is the network $\Psi_\theta$ (form `soft_pinn`, no extension). The training objective at each update is
$$
L_{\mathrm{pde}}+w\,L_{\mathrm{tc}},
$$
where $L_{\mathrm{pde}}$ is the mean squared residual on the interior batch of $4096$ points, and $L_{\mathrm{tc}}$ is the mean squared mismatch $(\Psi_\theta(x,T)-g(x))^2$ on $1024$ fresh uniform points of the spatial domain at $t=T$. The datum $g$ is the full datum on the circle cells and the butterfly on the line.
- The weight $w$ has no canonical value. Raissi et al. use $w=1$. Three weights are run, $w\in\{1,10,100\}$ (variants `soft_constraint_weight_1`, `_10`, `_100`), and all three are reported. None is selected on the evaluation error.
- The terminal points are drawn from a generator of their own (role `terminal_penalty_sampler` of the master seed). The interior sampler therefore draws the same batches as for the hard variants of the same seed.

**Selection.** The retained state minimises the training objective on fixed validation sets:
- the residual on the fixed validation sample of $16384$ points of the cylinder, as for the hard variants;
- plus $w$ times the terminal mismatch on a fixed set of $16384$ terminal points (role `terminal_validation`).
The evaluation sample and the evaluation grid select nothing.

**Everything else** is unchanged from the paper's validation-selected series: network, $20000$ Adam updates with cosine annealing, learning rate $10^{-3}$, validation every $100$ updates, evaluation grid $\Gamma$ and metric $\delta_\Gamma$.

**Seeds and tasks.** Seeds $\{0,1,2\}$, giving $3$ cells $\times$ $3$ weights $\times$ $3$ seeds $=27$ tasks.
- Jean Zay V100 (`gpu_p13`), account `akz@v100`, `qos_gpu-t3`.
- A smoke test with `--debug` on `qos_gpu-dev` comes first. Its values are not results.
- Series subfolder: `soft_constraint_comparison_2026-10-18`.

## 3. Predictions

Every prediction concerns the median over the three seeds of $\delta_\Gamma$, the relative error on the evaluation grid. The reference medians are those of the paper's series under the same protocol:

| Cell | $A=0$ (raw constant) | Highest-order split |
|---|---|---|
| `g2_bernoulli_full` | $0.138$ | $4.93\times10^{-4}$ |
| `g3_bernoulli_full` | $0.151$ | $4.90\times10^{-4}$ |
| `butterfly_real_line` | $0.2347$ | $5.72\times10^{-4}$ |

- **S1.** On every cell and for every weight, the soft constraint has a smaller median than $A=0$.
  - Motivation: the trained soft field is a $C^\infty$ network, whose residual has no singular part, and the penalty acts on its terminal profile. The error of $A=0$, in contrast, equals the limit left by the singular part of its forcing, which sampling does not see.
- **S2.** On every cell and for every weight, the soft constraint has a larger median than the highest-order split.
  - Motivation: the soft field approximates the datum, which has first-derivative discontinuities, by a $C^\infty$ network at $t=T$, and the penalty leaves a mismatch there. The split matches the datum exactly and has finite forcing energy.
- **S3.** On every cell, the median of the terminal mismatch `tc_l2` decreases strictly from $w=1$ to $w=10$ to $w=100$.
  - Motivation: the weight multiplies the penalised mismatch.

No prediction is made on which weight gives the smallest $\delta_\Gamma$.

**Admissibility.** A run is admissible when its validation and evaluation quantities are finite, and the gradient-norm safeguard is reported. The exact initial-value check of the hard variants does not apply. The terminal mismatch `tc_l2` is a measured quantity of the soft variants. A run that is not admissible is reported with its seed and excluded from the tests.

## 4. Outcome

Written 2026-10-18 after Jean Zay array 1269248 completed (smoke test 1269151, tests 1269150: 53 passed) at revision `1660618`. The values are read from the `summary_*.yaml` files of the nine run directories (`rel_l2` is $\delta_\Gamma$, `tc_l2` the relative $L^2$ terminal mismatch on the evaluation grid), copied to `data/ablation_split_extension_trained/soft_constraint_comparison_2026-10-18/`. The penalty weight and the validation terminal mismatch are read from `metrics.npz`: the runner did not write them to the summary (fixed in `25898a5`, after this series). No checkpoint files exist.

**Admissibility.** The 27 runs are admissible: finite validation and evaluation quantities, a positive and finite terminal mismatch, no gradient-norm safeguard activation and no non-finite norm over the 20000 recorded updates. Wall times range from $389$ to $474$ s on `g2` and on the butterfly cell, and from $1689$ to $2161$ s on `g3`.

**Relative error $\delta_\Gamma$** (measured, 3 seeds):

| Cell | Weight $\chi$ | Seed 0 | Seed 1 | Seed 2 | Median | Median `tc_l2` |
|---|---|---|---|---|---|---|
| `g2` | $1$ | $\simeq4.975\times10^{-3}$ | $\simeq3.960\times10^{-3}$ | $\simeq3.697\times10^{-3}$ | $\simeq3.96\times10^{-3}$ | $\simeq1.18\times10^{-2}$ |
| `g2` | $10$ | $\simeq2.912\times10^{-3}$ | $\simeq2.968\times10^{-3}$ | $\simeq3.099\times10^{-3}$ | $\simeq2.97\times10^{-3}$ | $\simeq8.07\times10^{-3}$ |
| `g2` | $100$ | $\simeq4.657\times10^{-3}$ | $\simeq4.229\times10^{-3}$ | $\simeq5.572\times10^{-3}$ | $\simeq4.66\times10^{-3}$ | $\simeq6.32\times10^{-3}$ |
| `g3` | $1$ | $\simeq1.920\times10^{-2}$ | $\simeq1.771\times10^{-2}$ | $\simeq1.545\times10^{-2}$ | $\simeq1.77\times10^{-2}$ | $\simeq4.68\times10^{-2}$ |
| `g3` | $10$ | $\simeq9.330\times10^{-3}$ | $\simeq1.014\times10^{-2}$ | $\simeq9.746\times10^{-3}$ | $\simeq9.75\times10^{-3}$ | $\simeq2.57\times10^{-2}$ |
| `g3` | $100$ | $\simeq8.170\times10^{-3}$ | $\simeq8.818\times10^{-3}$ | $\simeq7.775\times10^{-3}$ | $\simeq8.17\times10^{-3}$ | $\simeq1.87\times10^{-2}$ |
| butterfly | $1$ | $\simeq4.557\times10^{-3}$ | $\simeq5.805\times10^{-3}$ | $\simeq3.365\times10^{-3}$ | $\simeq4.56\times10^{-3}$ | $\simeq1.15\times10^{-2}$ |
| butterfly | $10$ | $\simeq3.034\times10^{-3}$ | $\simeq2.808\times10^{-3}$ | $\simeq2.308\times10^{-3}$ | $\simeq2.81\times10^{-3}$ | $\simeq6.33\times10^{-3}$ |
| butterfly | $100$ | $\simeq3.319\times10^{-3}$ | $\simeq3.972\times10^{-3}$ | $\simeq2.413\times10^{-3}$ | $\simeq3.32\times10^{-3}$ | $\simeq4.36\times10^{-3}$ |

**Predictions.**
- **S1 holds.** Every median lies below that of $A=0$: at most $4.66\times10^{-3}$ against $0.138$ on `g2`, $1.77\times10^{-2}$ against $0.151$ on `g3`, and $4.56\times10^{-3}$ against $0.2347$ on the butterfly cell.
- **S2 holds.** Every median exceeds that of the highest-order split, by a factor between $\simeq6.0$ and $\simeq9.4$ on `g2` ($4.93\times10^{-4}$), between $\simeq17$ and $\simeq36$ on `g3` ($4.90\times10^{-4}$), and between $\simeq4.9$ and $\simeq8.0$ on the butterfly cell ($5.72\times10^{-4}$).
- **S3 holds.** On every cell the median of `tc_l2` decreases strictly from $\chi=1$ to $\chi=10$ to $\chi=100$.

**Observations, not predicted.**
- The weight with the smallest median $\delta_\Gamma$ is $\chi=10$ on `g2` and on the butterfly cell, and $\chi=100$ on `g3`. No weight is selected; all three are reported.
- The soft constraint has a smaller median than the algebraic kernel ($6.59\times10^{-3}$, $33.9$ and $5.09\times10^{-3}$) at every weight on `g3` and on the butterfly cell, and at every weight on `g2`.
- The median validation residual of the soft constraint (the residual part of the selection criterion) ranges from $3.07\times10^{-7}$ to $6.95\times10^{-5}$ and increases with $\chi$ on every cell.
