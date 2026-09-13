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

## 9. Diagnostic — curvature profiles in $s$ at fixed $t$

`experiments/python_scripts/exp_barrier_option/diagnostic_scripts/compare_payoff_modes_at_strike.py`
compares four already-trained runs sharing a corner bandwidth $\varepsilon$ and a
master seed, one per terminal-function mode of $g_2$. Its analyses 3, 5 and 6
hold the underlying price fixed (at or near the strike) and sweep the calendar
time. Analysis 8 is their transpose: the calendar time is held fixed at the
values passed to `--gamma-profile-times`, and the whole $s$-profile of the
second price derivative is drawn over $s\in[s_{\min},s_{\max}]$ (default
$[B,2K]$), a range spanning both the corner layer
$\mathcal N_\varepsilon\cap\{t=\text{const}\}=\{B\le s\le B+\varepsilon\}$ and
the strike.

The quantity drawn is the second price derivative of the **full** trial
solution,

$$
\partial_{ss}\Phi_\theta(s,t) = \partial_{ss}\big(g_1(s,t)\,u_\theta(s,t)\big)
    + \partial_{ss}h_\varepsilon(s,t),
$$

not of the corner-regularised extension $h_\varepsilon=g_2$ alone. Two figures
are produced from one evaluation:

| Figure | Content |
|--------|---------|
| `figures/gamma_profiles_vs_price.png` | one panel per fixed $t$; $\partial_{ss}\Phi_\theta(\cdot,t)$ for the four modes, against the exact $\partial_{ss}V_{DO}(\cdot,t)$ (dashed) |
| `figures/gamma_profiles_vs_price_decomposition.png` | $3\times n_t$ grid: the same profiles split into $\partial_{ss}\Phi_\theta$, the network term $\partial_{ss}(g_1u_\theta)$, and the extension term $\partial_{ss}h_\varepsilon$ |

$\partial_{ss}\Phi_\theta$ and $\partial_{ss}(g_1u_\theta)$ are obtained by two
nested `torch.autograd.grad` passes on the trained model's own forward pass
(`ETCNN.forward` and `ETCNN.forward_neural_manifold` respectively);
$\partial_{ss}h_\varepsilon$ follows by subtraction, exactly, by linearity of
$\partial_{ss}$. The reference is
`pricing.barrier.reiner_rubinstein_down_and_out_put_gamma`, closed form, with
no network involved. The $y$-axis is symmetric-logarithmic (linear below
`--gamma-profile-linear-threshold`, default $10^{-1}$) because the profiles
change sign inside the corner layer, which a logarithmic axis cannot
represent. Both figures are redrawn from the saved
`gamma_profiles_vs_price.npz` without re-running any autograd pass.

Caveat for the `raw` mode, restated from the script: $(K-s)^+$ is affine on
each side of $s=K$, so $\partial_{ss}g_2=0$ at every grid point away from that
single point of Lebesgue measure zero. The `raw` curve is therefore the
compensating curvature the network has learned, never the (distributional)
curvature of the first-derivative discontinuity itself; the finite-difference
step sweep of analysis 5 is the tool that exhibits the latter.

## 10. Choice of the terminal-function mode for the strike singularity

The payoff's first-derivative discontinuity at $s=K$ and the conflicting
corner $(B,T)$ are two distinct singularities of this problem, regularised by
two distinct bandwidths ($\varepsilon_0$ and $\varepsilon$ respectively).
This section records the choice of $g_2$ made for the first of them; the
corner is treated separately and afterwards.

**Selection protocol.** Every metric is restricted to a strike band
$|s-K|\le\delta$ with the $\ell^1$ corner window $(s-B)+(T-t)\le w$ removed,
because the signed-error fields measured that the corner carries 47 to 63 per
cent of the squared error of every smoothed mode and masks the strike ranking
entirely. The corner is excluded from the metrics only: the training that
produced these runs sampled the whole domain uniformly, so about 8.5 of 4096
collocation points per iteration fall inside the corner layer at
$\varepsilon=0.1$. Each mode is compared at its own best $\varepsilon_0$
(envelope against envelope), since two modes carry a free bandwidth and three
carry none. Measurements come from
`diagnostic_scripts/select_terminal_function_at_strike.py`; a single master
seed (0) was used, a deliberate limitation discussed under "Strength of the
evidence" below.

Two clamps were found to drive metrics silently and had to be neutralised
before any ranking could be read:

- The exact $\Gamma$ is unbounded as $t\to T$; the closed form returns a
  finite value there only through its own floor $\tau\ge10^{-8}$
  (`_TAU_EPS`), giving $\partial_{ss}V_{DO}(K,T)=1.330\times10^{4}$. That
  single time slice dominated any $L^2$ norm over the band and drove the
  relative $\Gamma$ error of every mode to $1.000$. Metrics are therefore
  evaluated on $t\le T-\text{margin}$, reported for three margins.
- The strike half-width $\delta$ is arbitrary and the ranking of the two best
  modes reverses at $\delta=0.2$, where the band $[0.8,1.2]$ ceases to be a
  neighbourhood of the strike. The sensitivity is reported with the result.

**Result** ($\varepsilon=0.1$, 20 000 iterations, seed 0, $\delta=0.05$,
maturity margin $0.01$, corner excluded):

| Mode | $\varepsilon_0$ | Price rel. $L^2$ | $\Gamma$ rel. $L^2$ | Min price |
|---|---:|---:|---:|---:|
| Black-Scholes $V^e$ | -- | $4.837\times10^{-2}$ | $1.233\times10^{-1}$ | $8.9\times10^{-4}$ |
| Chen-Mangasarian, constant | $0.02$ | $6.783\times10^{-2}$ | $4.365\times10^{-1}$ | $5.1\times10^{-3}$ |
| Chen-Mangasarian, constant | $0.05$ | $7.721\times10^{-2}$ | $2.723\times10^{-1}$ | $1.2\times10^{-2}$ |
| Chen-Mangasarian, time-graded | $0.05$ | $1.027\times10^{-1}$ | $7.503$ | $7.9\times10^{-4}$ |
| Split semigroup (generic) | -- | $1.322\times10^{-1}$ | $\mathbf{7.344\times10^{-2}}$ | $5.5\times10^{-4}$ |
| Raw payoff $(K-s)^+$ | -- | $1.287$ | $1.183$ | $\mathbf{-6.53\times10^{-2}}$ |

The split-semigroup extension, measured after the table above was first
written, **reproduces the curvature better than the Black-Scholes oracle** ---
by a factor $1.68$, $1.68$ and $1.64$ at maturity margins $0.01$, $0.05$ and
$0.2$ respectively --- and attains the lowest training loss of the thirteen
runs ($7.31\times10^{-4}$), while carrying no closed-form requirement. Its
price error is $2.73$ times that of the oracle, of which $78.3$ per cent is a
constant offset of $-7.46\times10^{-3}$ (an additive bias contributes nothing
to $\partial_{ss}$, which is consistent with its curvature being the best of
the set). Under the primary metric fixed before the measurement --- the price
--- the Black-Scholes extension remains the selection for this pilot, where a
closed form exists; the split extension is the viable generic substitute where
none does. See `rapports/selection_g2_strike/` for the full argument.

**Decision: the Black-Scholes extension
(`make_corner_regularised_extension_with_black_scholes_payoff`) is adopted as
the terminal-function mode for the strike.** It is best on both metrics at
every setting of $\delta$ up to $0.1$ and at every maturity margin, by a
factor $1.40$ on the price and $2.2$ to $2.9$ on the $\Gamma$.

Two competing modes are eliminated on grounds that do not depend on the
measurement precision:

- The raw payoff is **disqualified**, not merely last: it is the only mode
  producing a negative price ($-6.53\times10^{-2}$), an arbitrage violation.
  Its second price derivative also has no $h\to0$ limit --- the central second
  difference of $(K-s)^+$ at $s=K$ is exactly $1/h$, measured as
  $1.0\times10^{2},10^{3},10^{4},10^{5}$ for $h=10^{-2}\ldots10^{-5}$ ---
  so the curvature of the trial solution is undefined at the strike.
- The time-graded bandwidth is dominated by the constant one. Its
  $\Gamma$ error grows from $8.43\times10^{-1}$ at maturity margin $0.2$ to
  $7.503$ at margin $0.01$, i.e. it diverges as maturity is approached,
  reproducing the analytic prediction $\partial_{ss}g_{\varepsilon_0}(K,t)=
  1/(2\varepsilon_0(t))\to\infty$: it acquires an exact terminal trace at the
  cost of a curvature that diverges faster than the true $\Gamma$.

**Scope of the claim.** The Black-Scholes extension inserts the closed-form
European price of the operator being solved. It is therefore an upper bound on
what an extension can supply for this contract, not a construction available
for a general one; the split-semigroup extension
(`make_corner_regularised_extension_split`) is its generic counterpart and
carries no such requirement. Within this pilot the requirement is met, and
the choice has a further property that the corner study needs: $\mathcal
L^{BS}h_\varepsilon^{BS}=0$ wherever the cutoff $\zeta$ is constant (verified
in `diagnostic_scripts/check_bs_g2_residual.py`), so with this $g_2$ the only
interior forcing the network must cancel is supported on the corner layer
itself. The strike ceases to be a source of forcing, which is exactly the
isolation required to study $(B,T)$ next.

**Strength of the evidence.** One master seed. The inter-seed dispersion
measured on the five available raw-mode runs at $\varepsilon=0.1$ is a
coefficient of variation of $4.9$ per cent on this same metric (values
$1.287$, $1.243$, $1.217$, $1.143$, $1.160$; standard deviation
$5.94\times10^{-2}$). Against that scale, the Black-Scholes advantage over the
best Chen-Mangasarian run ($40$ per cent) is $8.2$ standard deviations and the
constant-over-time-graded advantage ($51$ per cent) is $10.4$; both are
decided without replication. The choice of $\varepsilon_0$ *within* the
Chen-Mangasarian family ($13.8$ per cent, $2.8$ standard deviations) is
**not** decided by these runs --- it does not need to be, that family having
been set aside. The dispersion estimate comes from the raw mode, the only
multi-seed series available, and calibrates an order of magnitude rather than
substituting for replication of the modes actually compared.

## 11. Evaluation metrics and model-based diagnostics for the corner-excluded comparison

Let $\Omega = (B, s_\infty) \times (0, T)$ be the evaluation domain, discretised by the pilot on a
uniform $300 \times 100$ grid, and let $V_{DO}$ denote the Reiner-Rubinstein closed form. For a
region $R \subset \Omega$ the relative $L^2$ error of the trained trial solution $\Phi_\theta = g_1 u_\theta + g_2$ is

$$
\mathrm{rel}_{L^2}(R) = \frac{\|\Phi_\theta - V_{DO}\|_{L^2(R)}}{\|V_{DO}\|_{L^2(R)}} .
$$

Three regions are reported by `evaluate_against_closed_form` (`summary_eps<EPSILON>.yaml` keys in
parentheses), with $N_w = \{(s,t) : |s-B| + (T-t) \le w\}$ the $\ell^1$ corner window and $w$ the
value of `--corner-window` (default: the largest $\varepsilon$ of the sweep):

- $R = \Omega$ (`rel_l2_global`) --- the corner window **included**. Inside $N_w$ the closed form
  is of order $K - B$ while the barrier factor $g_1$ forces $\Phi_\theta = 0$ on $s = B$, so this
  metric is dominated by the corner discontinuity whenever $N_w$ is not negligible. Kept for
  continuity with earlier runs only.
- $R = N_w$ (`rel_l2_corner`) --- the window alone. With `--exclude-corner-from-collocation` the
  residual is never enforced there; the value ($0.55$--$0.62$ on every run of the 20000-iteration
  comparison) is a diagnostic of the corner treatment, not of the training.
- $R = \Omega \setminus N_w$ (`rel_l2_outside_corner`) --- the complement, i.e. exactly the region
  where the residual is enforced when the corner is excluded. This is the comparison metric.
  It is recomputed from the saved model by `--replot` (the evaluation grid is deterministic:
  existing values are reproduced to $10^{-7}$ relative across machines).

Two diagnostics are computed by `aggregate_terminal_function_comparison.py` from the saved
models (no retraining), with $\tau = T - t$:

**Window-shape sweep.** $\mathrm{rel}_{L^2}(\Omega \setminus N)$ for three families of excluded
window $N$, plotted against the excluded area fraction $|N \cap \Omega| / |\Omega|$ rather than
against the family parameter, so that the three shapes share one abscissa:

$$
N_w = \{ |s-B| + \tau \le w \}, \quad
N_c = \{ |s-B| \le c\, B \sigma \sqrt{\tau} \}, \quad
N_d = \{ \tau\,(s-B) \le d \},
$$

with $w \in \{0.1, 0.2, 0.3, 0.5\}$, $c \in \{0, 1, 2, 3\}$, $d \in \{0.005, 0.02, 0.05, 0.1\}$.
If the error were concentrated at the corner, the curves would decrease with the excluded
area; measured on the 20000-iteration comparison (5 seeds per configuration, `data/aggregate_terminal_function_comparison/20260912_124805_iters20000_eps0.1_nocorner/model_based_diagnostics/`),
the across-seed medians are flat or **increasing** with the excluded area for all three shapes
and all three configurations (lozenge, Black-Scholes ordinary route: $0.169, 0.156, 0.158, 0.172$
for $w = 0.1, 0.2, 0.3, 0.5$; parabola: $0.220, 0.207, 0.202, 0.249$ for $c = 0, 1, 2, 3$;
hyperbola: $0.164, 0.158, 0.169, 0.197$ for $d = 0.005, 0.02, 0.05, 0.1$). The error of these runs
is therefore not localised at the corner: it is spread over the interior, and the relative
metric rises as the regions where $\|V_{DO}\|$ is concentrated are removed from the denominator.

**Network contribution in a band.** In $\mathcal{B} = \{ b_{lo} < |s-B| < b_{hi} \}$ (all $t$;
default $b_{lo} = 0.1$, $b_{hi} = 0.3$), the discrete norms
$\|\Phi_\theta - V_{DO}\|_{L^2(\mathcal{B})}$ and $\|h_\varepsilon - V_{DO}\|_{L^2(\mathcal{B})}$
are compared, where $h_\varepsilon = g_2$ is the terminal-function extension alone. A ratio close
to $1$ would mean the network adds nothing in the band. Measured (median over 5 seeds):
$\|h_\varepsilon - V_{DO}\| = 4.37 \times 10^{-2}$ (Black-Scholes modes) and $4.31 \times 10^{-2}$
(split), $\|\Phi_\theta - V_{DO}\| = 4.8$--$5.8 \times 10^{-3}$, ratio $0.11$--$0.13$ (range over
seeds $0.097$--$0.181$), against $\|V_{DO}\|_{L^2(\mathcal{B})} = 6.17 \times 10^{-2}$. The network
reduces the extension's error in the band by a factor of $5.5$ to $10$; the remaining error is
$7$--$13$ per cent of $\|V_{DO}\|$ in the band.

### 11.1 Measured at 50000 iterations (canonical runs on `republique`, 5 seeds per configuration)

Aggregation: `data/aggregate_terminal_function_comparison/20260913_172225_iters50000_eps0.1_nocorner/` (`table.md`, `budget_comparison.md`, `model_based_diagnostics/`).
The 50000-iteration runs were trained on AVX2 hosts only (`porte-d-orleans` up to iteration
26000 for the split runs, `republique` afterwards; a 300-iteration control gives bit-identical
weights on the two hosts, whereas the Sandy Bridge hosts differ by .6 \times 10^{-5}$ after 300
iterations), with 4 threads per run, as recorded in each `metadata.yaml`.

| Configuration | best loss,  \to 50000$ | $\mathrm{rel}_{L^2}(\Omega\setminus N_{0.1})$,  \to 50000$ | ratio |
|---|---|---|---|
| Black-Scholes, ordinary route | .0 \times 10^{-4} \to 3.7 \times 10^{-5}$ | /bin/zsh.169 \to 0.122$ | /bin/zsh.72$ |
| Black-Scholes, two-term route | .1 \times 10^{-4} \to 1.7 \times 10^{-5}$ | /bin/zsh.148 \to 0.124$ | /bin/zsh.83$ |
| Split-semigroup | .3 \times 10^{-4} \to 1.3 \times 10^{-5}$ | /bin/zsh.134 \to 0.127$ | /bin/zsh.95$ |

The interior residual decreases by a factor of $ to $ while the error on the training domain
decreases by at most a factor of .4$ and settles at /bin/zsh.12569X-/bin/zsh.13$ for the three configurations
(across-seed ranges /bin/zsh.083569X-/bin/zsh.311$, overlapping). The best loss is again attained in the last
$ per cent of the iterations (569X-$). The Greeks at the strike
(`data/evaluate_greeks_no_corner/`, `--iters 50000 --hosts republique`) improve only near
maturity: at  = 0.9$ the relative Gamma error falls from /bin/zsh.8569X-.0 \times 10^{-2}$ to
.1569X-.2 \times 10^{-3}$, while at  = 0.5$ it stays at /bin/zsh.24569X-/bin/zsh.25$ and the Delta error at
/bin/zsh.08$ (unchanged from 20000 iterations). The residual error is therefore a floor that the
interior residual does not penalise, not a lack of iterations; the far-field boundary
 = s_\infty$, where no condition is imposed (Remark 2), is the candidate to examine next
(empirical observation; the attribution is a conjecture).

Hardware replicate: the ten Black-Scholes runs launched in parallel on the non-AVX2 hosts
(`porte-d-auteuil`, `porte-de-la-chapelle`, `porte-pouchet`; two of them died at start-up with
`Illegal instruction` in `libtorch_cpu.so`) give, on the 8 surviving runs, medians of
$\mathrm{rel}_{L^2}(\Omega\setminus N_{0.1})$ of /bin/zsh.117$ (ordinary route,  = 4$) and /bin/zsh.151$
(two-term route,  = 4$) against /bin/zsh.122$ and /bin/zsh.124$ on `republique`: the host effect is of the
same order as the across-seed dispersion and does not change the conclusion.
