# Down-and-out put — corner-regularised ETCNN ansatz

> Math rendering: open in a Markdown+KaTeX/MathJax renderer for rendered equations.

This document describes the pilot implementation of a knock-out barrier option
under the ETCNN framework. It follows the same conventions as
[`architecture.md`](architecture.md) (problem formulation, exact terminal /
boundary functions, network architecture, math → code mapping) but for a
structurally different problem: a *second* hard constraint on a spatial
boundary, rather than the temporal-only terminal condition of the American /
Bermudan case.

Key reference (Doc A):

* S. Ouaissi — *A rigorous statement of exact-constraint learning at a
  conflicting constraint corner: the knock-out barrier option*, internship
  working note, 2026-06-24. A structural comparison of the code against this
  note is kept in [`docs_travail/barrier_start_map.md`](../../docs_travail/barrier_start_map.md).

---

## 1. Problem formulation

A down-and-out put with strike $K$ and knock-out barrier $B$, $0 < B < K$
(Assumption 1 of Doc A: a *reverse* knock-out, $g(B) = K - B > 0$), solves the
boundary-value problem

$$
\mathcal{L}^{BS}V_{DO} = 0 \text{ on } Q = (B, +\infty) \times (0, T),
\qquad V_{DO} = g \text{ on } \Sigma_T, \qquad V_{DO} = 0 \text{ on } \Sigma_B,
$$

where $\Sigma_T$ is the terminal lid, $\Sigma_B$ the barrier face, and
$\mathcal{L}^{BS}V = \partial_t V + \tfrac12\sigma^2 s^2 \partial_{ss}V + rs\partial_s V - rV$
(code: `learning_option_pricing.pricing.terminal.bsm_operator`). The two data
$g$ (on $\Sigma_T$) and $0$ (on $\Sigma_B$) are mutually inconsistent at the
corner $\mathfrak{c} = (B, T)$ (Doc A, Proposition 1): no continuous function
carries both traces exactly, since $g(B) = K - B \neq 0$.

## 2. Corner-regularised ansatz

The trial solution is the ordinary ETCNN form
$U_\theta(s,t) = g_1(s,t)\,u_\theta(s,t) + g_2(s,t)$, code:
`learning_option_pricing.models.etcnn.ETCNN` (unmodified base class — see §4
for why this class, not `AmericanPutETCNN` or `TerminalAnsatz`).

**Composite distance** (Doc A, Definition 4):

$$
g_1(s,t) = d_{\partial_p Q}(s,t) = (T-t)(s-B),
$$

vanishing exactly on $\Sigma_T \cup \Sigma_B$, including at $\mathfrak{c}$
itself. No regularisation is needed for this factor.
Code: `learning_option_pricing.pricing.barrier.barrier_composite_distance`.

**Corner-regularised extension** (Doc A, Definition 5):

$$
g_2(s,t) = h_\varepsilon(s,t) = \zeta\!\left(\frac{s-B}{\varepsilon}\right)(K-s)^+,
$$

with $\zeta:\mathbb{R}\to[0,1]$ the standard $C^\infty$ compactly-supported
transition ($\zeta(r)=0$ for $r\le 0$, $\zeta(r)=1$ for $r\ge 1$, built from
$f(r) = e^{-1/r}\mathbb{1}_{r>0}$ as $\zeta(r) = f(r)/(f(r)+f(1-r))$).
Code: `learning_option_pricing.pricing.barrier.make_corner_regularised_extension`.

Because the training loss below has no separate boundary-condition term (§3),
the two hard constraints hold **exactly** wherever $g_1$ and $g_2$ deliver
them — training reduces to the interior PDE residual alone, exactly Doc A
Section 4's stated goal.

## 3. Loss — interior residual only

Unlike the American / Bermudan pipeline (`pricing.loss`, three complementarity
terms) or the plain European case ($\mathcal{L}_f + \mathcal{L}_{tc}$), this
ansatz needs a **single** term:

$$
\mathcal{L}(\theta) = \frac{1}{N_f}\sum_{i=1}^{N_f}\bigl[\mathcal{L}^{BS}U_\theta(s_i,t_i)\bigr]^2,
$$

evaluated on interior collocation points $(s_i, t_i)$ sampled uniformly on
$(B, s_\infty) \times (0, T)$. There is no $\mathcal{L}_{tc}$ term: unlike
`AmericanPutETCNN`'s Taylor-expansion $g_2$, $h_\varepsilon$ matches the
terminal payoff *exactly* (not merely approximately) outside the corner
layer, so a terminal-condition penalty would be redundant by construction.
Code: `experiments.python_scripts.exp_barrier_option.pilot_down_and_out_put.compute_loss`.

## 4. Why the base `ETCNN` class, not `TerminalAnsatz` or `AmericanPutETCNN`

`ETCNN.__init__` already accepts `g1`/`g2` as arbitrary callables of
$(s,t)$ ([`learning_option_pricing/models/etcnn.py`](../../learning_option_pricing/models/etcnn.py)),
so `ETCNN.forward` — literally `g1_val * u_nn + g2_val` — matches Doc A's
eq. (12), $\Phi_\theta = h_\varepsilon + d_{\partial_p Q}\,\Psi_\theta$,
without any change to the class. `AmericanPutETCNN` was not used because it
hard-codes $g_1(s,t)=T-t$ (a single, temporal-only, distance factor) and a
put-payoff-specific $g_2$; extending it would have meant editing shared
package code rather than passing different callables at construction.
`TerminalAnsatz` (the other candidate identified in
[`docs_travail/barrier_start_map.md`](../../docs_travail/barrier_start_map.md) §2)
was set aside for this pilot in favour of `ETCNN`: a deliberate scope
decision, not a claim that `TerminalAnsatz` is unsuitable — see the open
question in that document.

## 5. Deliberate scope decisions of this pilot

These are documented explicitly, per the "no silent approximation" convention
of this repository — none of them is a bug or an oversight.

- **$h_\varepsilon$ is time-independent.** Doc A's Definition 5 describes a
  regularisation over an $\ell^1$-ball $\mathcal{N}_\varepsilon$ centred on
  the corner (both $s$ and $t$). The construction used here,
  $\zeta((s-B)/\varepsilon)\cdot g(s)$, depends only on $s$. It is still
  rigorously admissible (it satisfies conditions (11) of Doc A — see the
  docstring of `make_corner_regularised_extension` for the proof sketch —
  and is in fact *stronger* than required on $\Sigma_B$, since the barrier
  datum is identically zero there), but it is a simpler shape than the
  literal corner ball.
- **No hard far-field condition at $s_\infty$.** Doc A's Remark 2 notes the
  domain truncation $(B, s_\infty)$ should carry $V_{DO}(s_\infty,t)=0$
  approximately. This pilot truncates the domain but does not add a loss
  term enforcing that condition — consistent with how the existing
  ETCNN pipeline (`experiments/python_scripts/exp1/phase3_training.py`)
  also does not enforce its outer spatial boundary `S_TRAIN_HI`.
- **`ETCNN` base class in native $(s,t)$ coordinates**, not `TerminalAnsatz`
  in the log-price coordinate $x=\ln s$ (which would have reused
  `TerminalAnsatz.residual_decomposition`'s analytic-derivative
  cross-check machinery, at the cost of a coordinate change). See §4.
- **No cross-check optimiser** (ENGD): plain Adam + the repo's two-stage
  exponential LR decay (`build_lr_lambda`, identical schedule to
  `phase3_training.py`).

## 6. Reference closed form

$V_{DO}$ is available in closed form via the method of images (Doc A, Remark
6): reflecting the log-price transition density across the barrier gives

$$
V_{DO}(s,\tau) = \mathrm{TP}(s,K,B,\tau) - \left(\frac{B}{s}\right)^{2r/\sigma^2-1}\mathrm{TP}\!\left(\frac{B^2}{s},K,B,\tau\right),
$$

where $\mathrm{TP}(s,K,B,\tau) = e^{-r\tau}\mathbb{E}[(K-S_\tau)^+\mathbb{1}_{S_\tau>B}]$
is a truncated European put expectation (both $S_\tau > B$ and $S_\tau < K$).
Code: `learning_option_pricing.pricing.barrier.reiner_rubinstein_down_and_out_put`.
See that function's docstring for how the reflection-prefactor sign was
validated against an independent discretely-monitored Monte-Carlo simulation
(the deviation from Monte-Carlo shrinks like $O(1/\sqrt{N})$ in the monitoring
frequency $N$, the expected discretisation-bias signature of a correct
continuous-barrier formula).

## 7. Math → code mapping

| Symbol | Description | Code location |
|--------|-------------|----------------|
| $\mathcal{L}^{BS}(V)$ | Black–Scholes operator | `pricing.terminal.bsm_operator` |
| $g(s)=(K-s)^+$ | Put payoff | `pricing.terminal.payoff_put` |
| $g_1(s,t)=d_{\partial_p Q}$ | Composite distance (eq. 9) | `pricing.barrier.barrier_composite_distance` |
| $g_2(s,t)=h_\varepsilon(s,t)$ | Corner-regularised extension (Def. 5) | `pricing.barrier.make_corner_regularised_extension` |
| $\zeta(r)$ | $C^\infty$ compactly-supported transition | `pricing.barrier._smoothstep01` |
| $V_{DO}(s,t)$ | Exact closed-form reference | `pricing.barrier.reiner_rubinstein_down_and_out_put` |
| $U_\theta(s,t)=g_1 u_\theta+g_2$ | Trial solution | `models.etcnn.ETCNN.forward` |
| $\mathcal{L}(\theta)$ | Interior-residual-only loss | `exp_barrier_option.pilot_down_and_out_put.compute_loss` |
| $\varepsilon$ sweep, corner-window error | Multi-$\varepsilon$ training + evaluation | `exp_barrier_option.pilot_down_and_out_put.{train_one_epsilon,evaluate_against_closed_form}` |

## 8. Reproducing a run

```
python3 experiments/python_scripts/exp_barrier_option/pilot_down_and_out_put.py \
    --epsilons 0.2 0.1 0.05 0.02 0.01 --iters 20000
```

Regenerate figures from a previous run without retraining:

```
python3 experiments/python_scripts/exp_barrier_option/pilot_down_and_out_put.py \
    --replot data/pilot_down_and_out_put/<run_dir>
```

Every run's `metadata.yaml`, per-$\varepsilon$ `summary_eps<value>.yaml`, and
`models/model_eps<value>.pt` are sufficient to reproduce every figure via
`--replot` without access to a GPU or re-running training.
