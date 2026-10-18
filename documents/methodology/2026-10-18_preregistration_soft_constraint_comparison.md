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

To be written after the runs, from the `summary_*.yaml` files of the run directories.
