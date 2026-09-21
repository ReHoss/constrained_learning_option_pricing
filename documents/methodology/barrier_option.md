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
| $\pi(x,t) = e^{(T-t)\nu_c\partial_{xx}}(K-e^{x})^+$ | Split-semigroup profile, closed form (section 14) | `pde.real_line_extension_fields.PutPayoffGaussianSemigroupExtensionField` |
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

Aggregation: `data/aggregate_terminal_function_comparison/20260913_172225_iters50000_eps0.1_nocorner/`
(`table.md`, `budget_comparison.md`, `model_based_diagnostics/`). The 50000-iteration runs were
trained on AVX2 hosts only (`porte-d-orleans` up to iteration 26000 for the split runs,
`republique` afterwards; a 300-iteration control gives bit-identical weights on the two hosts,
whereas the Sandy Bridge hosts differ by $8.6 \times 10^{-5}$ after 300 iterations), with 4 threads
per run, as recorded in each `metadata.yaml`.

| Configuration | best loss, $20000 \to 50000$ | $\mathrm{rel}_{L^2}(\Omega\setminus N_{0.1})$, $20000 \to 50000$ | ratio |
|---|---|---|---|
| Black-Scholes, ordinary route | $4.0 \times 10^{-4} \to 3.7 \times 10^{-5}$ | $0.169 \to 0.122$ | $0.72$ |
| Black-Scholes, two-term route | $5.1 \times 10^{-4} \to 1.7 \times 10^{-5}$ | $0.148 \to 0.124$ | $0.83$ |
| Split-semigroup | $1.3 \times 10^{-4} \to 1.3 \times 10^{-5}$ | $0.134 \to 0.127$ | $0.95$ |

The interior residual decreases by a factor of $10$ to $30$ while the error on the training domain
decreases by at most a factor of $1.4$ and settles at $0.12$--$0.13$ for the three configurations
(across-seed ranges $0.083$--$0.311$, overlapping). The best loss is again attained in the last
$10$ per cent of the iterations ($46551$--$49923$). The Greeks at the strike
(`data/evaluate_greeks_no_corner/`, `--iters 50000 --hosts republique`) improve only near
maturity: at $t = 0.9$ the relative Gamma error falls from $0.8$--$1.0 \times 10^{-2}$ to
$1.1$--$4.2 \times 10^{-3}$, while at $t = 0.5$ it stays at $0.24$--$0.25$ and the Delta error at
$0.08$ (unchanged from 20000 iterations). The residual error is therefore a floor that the
interior residual does not penalise, not a lack of iterations; section 11.2 localises it.

Hardware replicate: the ten Black-Scholes runs launched in parallel on the non-AVX2 hosts
(`porte-d-auteuil`, `porte-de-la-chapelle`, `porte-pouchet`; two of them died at start-up with
`Illegal instruction` in `libtorch_cpu.so`) give, on the 8 surviving runs, medians of
$\mathrm{rel}_{L^2}(\Omega\setminus N_{0.1})$ of $0.117$ (ordinary route, $n = 4$) and $0.151$
(two-term route, $n = 4$) against $0.122$ and $0.124$ on `republique`: the host effect is of the
same order as the across-seed dispersion and does not change the conclusion.

### 11.2 Error per band of the underlying price (50000 iterations, corner window removed)

`data/aggregate_terminal_function_comparison/20260913_172225_iters50000_eps0.1_nocorner/model_based_diagnostics/s_band_errors.md` (`--s-band-edges 0.6 0.7 1 2 3`). Relative $L^2$
error per band of $s$ (all $t$, $N_{0.1}$ removed), median over 5 seeds: on $[B, B+\varepsilon] = [0.6, 0.7]$,
the transition band of the switching factor $\zeta((s-B)/\varepsilon)$, $0.20$--$0.22$ for the three
configurations; on $[0.7, 1]$, $0.075$--$0.083$; on $[1, 2]$, $0.05$--$0.06$; on $[2, s_\infty]$, $13$ to $57$
(the closed form is of order $10^{-4}$ there while the trained field is of order $10^{-3}$--$10^{-2}$,
with a factor-30 dispersion across seeds). Share of the total error energy $\sum \|\Phi_\theta - V_{DO}\|^2$
per band (median over seeds): $[0.6, 0.7]$ $34$--$54$ per cent, $[0.7, 1]$ $24$--$40$, $[1, 2]$ $1$,
$[2, s_\infty]$ $2$ (ordinary route) to $42$ (split; seed-dependent, $1$--$75$). The transition band
holds the largest share of the error in $4$ per cent of the price range (error-energy density
$4$ times that of $[0.7, 1]$); between 20000 and 50000 iterations its relative error moved from
$0.20$--$0.25$ to $0.20$--$0.22$ only. Two components are therefore present: a floor localised in
the $\zeta$ transition band (the same for the three terminal functions, hence attributable to the
switching factor rather than to the datum it weights), and an unconstrained far-field component
of seed-dependent size. Attribution of the transition-band floor to the switching factor is the
reading of these measurements, not yet a controlled test (a sweep of $\varepsilon$ at fixed corner
exclusion, or a different $\zeta$, would be the test).

## 12. Far-field condition on the truncated domain

Let $\mathcal L^{BS} V = \partial_t V + \tfrac12\sigma^2 s^2\partial_{ss}V + r s\,\partial_s V - rV$
and let $Q_{s_\infty} = (B, s_\infty)\times(0,T)$ be the truncated training domain, with
$B < s_\infty < \infty$.

**On the natural domain** $(B,\infty)\times(0,T)$ no boundary value is required at infinity: the
exact price $V_{DO}$ is bounded ($0 \le V_{DO} \le K$) and uniqueness of the terminal-boundary
problem holds in the class of bounded solutions (in the log-price variable $x = \ln s$ the operator
is a constant-coefficient heat operator with drift on a half-line, for which uniqueness holds
under the Tychonoff growth condition $|V| \le C e^{a x^2}$).

**On the truncated domain**, the far segment $\Sigma_\infty = \{s_\infty\}\times(0,T)$ belongs to the
parabolic boundary, and a condition on it is necessary for uniqueness: for any
$\varphi\in C([0,T])$ with $\varphi(T)=0$ there is a solution $w$ of $\mathcal L^{BS}w = 0$ in
$Q_{s_\infty}$ with $w = 0$ on $\{s=B\}$ and $\{t=T\}$ and $w = \varphi$ on $\Sigma_\infty$; every
such $w$ has zero interior residual, so the interior loss of the pilot cannot distinguish
$V_{DO}$ from $V_{DO} + w$. This is the unconstrained far-field component measured in section 11.2
(seed-dependent, $2$ to $42$ per cent of the error energy).

The pilot's `--far-field-dirichlet` imposes the Dirichlet condition
$\Phi_\theta(s_\infty, t) = g_2(s_\infty, t)$ in hard form, through the factor
$g_1(s,t) = (T-t)(s-B)\,\dfrac{s_\infty - s}{s_\infty - B}$
(`barrier_composite_distance_with_far_field`), which vanishes on the three faces
$\{s = B\}$, $\{t = T\}$, $\{s = s_\infty\}$. The error committed by the truncation is controlled as
follows.

**Proposition (truncation error).** Let $V$ be the solution of $\mathcal L^{BS} V = 0$ in
$Q_{s_\infty}$ with $V(B,t) = 0$, $V(s,T) = (K-s)^+$ and $V(s_\infty, t) = \varphi(t)$. Then

$$
\sup_{Q_{s_\infty}} |V - V_{DO}| \;\le\; \sup_{t\in(0,T)} \big|\varphi(t) - V_{DO}(s_\infty, t)\big| .
$$

*Proof.* Set $w = V - V_{DO}$. Then $\mathcal L^{BS} w = 0$ in $Q_{s_\infty}$, $w = 0$ on
$\{s = B\}$ and $\{t = T\}$, and $w = \varphi - V_{DO}(s_\infty,\cdot)$ on $\Sigma_\infty$. In the
variable $\tau = T - t$ the equation reads $\partial_\tau w + L w = 0$ with
$L w = -\tfrac12\sigma^2 s^2 \partial_{ss} w - r s\,\partial_s w + r w$, which is uniformly
parabolic on $(B, s_\infty)$ (since $B > 0$) with zeroth-order coefficient $r \ge 0$. The weak
maximum principle for such operators (Evans, *Partial Differential Equations*, 2nd ed.,
§7.1.4, Theorem 9, applied to $w$ and to $-w$) gives $\sup |w| \le \sup_{\partial_p Q}|w|$, and
$w$ vanishes on the parabolic boundary except on $\Sigma_\infty$. $\square$

With $\varphi = g_2(s_\infty,\cdot)$ the pilot evaluates the right-hand side in float64 from the
closed form (`far_field_truncation_error_bound`, logged and stored as
`far_field_truncation_error_bound` in the summary). For the contract $K = 1$, $B = 0.6$,
$r = 0.03$, $\sigma = 0.3$, $T = 1$ and $s_\infty = 3$: $\sup_t |V_{DO}(s_\infty, t)| = 1.0\times10^{-5}$,
and the bound equals $4.1\times10^{-8}$ for the Black-Scholes terminal function and
$1.6\times10^{-6}$ for the split-semigroup profile -- four to six orders of magnitude below the
measured error floor of section 11, so the far-field condition removes the far-field
component at no measurable cost in truncation error.

### 12.1 Measured with the far-field condition (50000 iterations, `republique`, 5 seeds per construction)

Aggregation: `data/aggregate_terminal_function_comparison/20260916_111337_iters50000_eps0.1_nocorner_farfield/` (runs tagged `_farfield`; `--far-field yes --hosts republique`,
`budget_comparison.md` against the canonical batch of section 11.1). Recorded truncation-error
bounds: $4.09 \times 10^{-8}$ for the ten Black-Scholes runs, $1.58 \times 10^{-6}$ for the five split runs.

| Construction | $\|\Phi_\theta - V_{DO}\|_{L^2([2,s_\infty]\times(0,T))}$, without $\to$ with | $\mathrm{rel}_{L^2}(\Omega\setminus N_{0.1})$, without $\to$ with | remainder after the predicted floor |
|---|---|---|---|
| Black-Scholes, ordinary route | $1.5\times10^{-3} \to 3.0\times10^{-4}$ | $0.122 \to 0.111$ | $0.021$ |
| Black-Scholes, two-term route | $3.7\times10^{-3} \to 1.5\times10^{-4}$ | $0.124 \to 0.115$ | $0.026$ |
| Split-semigroup | $6.4\times10^{-3} \to 2.4\times10^{-4}$ | $0.127 \to 0.099$ | $0.018$ |

(medians over seeds). The far-field component of section 11.2 is reduced by a factor of $5$ to $26$
and its across-seed dispersion from a factor $30$ to a factor $10$; what remains on $[2, s_\infty]$ is
of the order of $\|V_{DO}\|$ there ($1.1\times10^{-4}$). On the training domain the error settles on the
floor predicted in section 13 ($0.112$): per band, $0.186$--$0.215$ on $[0.6,0.7]$ (predicted $0.210$),
$0.074$--$0.082$ on $[0.7,1]$ ($0.085$), $0.042$--$0.059$ on $[1,2]$ ($0.047$); the remainder
$\|(\Phi_\theta - V_{DO}) - w\| / \|V_{DO}\|$ is $0.018$--$0.026$. The three constructions remain
indistinguishable at $n = 5$ (overlapping ranges). The best interior loss is $1.0$ to $1.6$ times higher
with the condition (the factor $(s_\infty - s)/(s_\infty - B)$ changes the function class), without
effect on the error.

## 13. The corner-layer floor is the terminal-trace defect of the ansatz (predicted, then measured)

For every terminal-function mode, $g_1(s,T) = 0$ and $g_2(s,T) = \zeta\big(\tfrac{s-B}{\varepsilon}\big)(K-s)^+$
(the Black-Scholes put price, the split profile and the raw payoff all equal $(K-s)^+$ at $t = T$).
The terminal trace of the trial solution is therefore

$$
\Phi_\theta(s,T) = \zeta\!\left(\frac{s-B}{\varepsilon}\right)(K-s)^+ ,
\qquad
\delta(s) = \Phi_\theta(s,T) - (K-s)^+ = -\Big(1-\zeta\!\left(\tfrac{s-B}{\varepsilon}\right)\Big)(K-s)^+ ,
$$

a defect supported on $[B, B+\varepsilon]$, equal to $-(K-B)$ at $s = B^+$, and independent of
$\theta$: the network cannot change it. It is forced by the exactness of the barrier condition
($\zeta(0) = 0$) together with the incompatibility of the data at the corner
($V_{DO}(B,t) = 0$ for $t < T$ while $V_{DO}(s,T) \to K-B \neq 0$ as $s \to B^+$).

**Proposition (predicted floor).** Let $w$ solve $\mathcal L^{BS}w = 0$ in $(B,\infty)\times(0,T)$,
$w(B,\cdot) = 0$, $w(\cdot,T) = \delta$, i.e. $w$ is the price of the down-and-out claim with
payoff $\delta$. If $\Phi_\theta$ had zero interior residual, exact barrier and far-field data,
then $\Phi_\theta - V_{DO} = w$ (uniqueness in the class of bounded solutions, section 12). By the
reflection principle for the log-price diffusion absorbed at $b = \ln B$,

$$
w(s,\tau) = e^{-r\tau}\int_b^{\ln(B+\varepsilon)}
\Big[p(\tau;x_0,x) - e^{2\nu(b-x_0)/\sigma^2}\,p(\tau;2b-x_0,x)\Big]\,\delta(e^x)\,dx ,
\qquad x_0 = \ln s,\ \nu = r - \tfrac{\sigma^2}{2},
$$

with $p(\tau;x_0,\cdot)$ the Gaussian density of mean $x_0+\nu\tau$ and variance $\sigma^2\tau$.
*Proof.* The integrand is the transition density of the killed process (the same identity, with
the payoff $(K-s)^+$ in place of $\delta$, is the Reiner-Rubinstein formula of
`reiner_rubinstein_down_and_out_put`; the script checks it numerically to $8\times10^{-9}$). $\square$

**Measured** (`experiments/python_scripts/exp_barrier_option/diagnostic_scripts/predict_corner_layer_floor.py`,
output `data/predict_corner_layer_floor/20260913_214751_20260913_172225_iters50000_eps0.1_nocorner/`): relative $L^2$ error per band of $s$, corner window removed, 50000 iterations, medians over 5 seeds.

| Band of $s$ | Predicted floor $w$ alone | Measured, Black-Scholes ordinary | Measured, Black-Scholes two-term | Measured, split |
|---|---|---|---|---|
| $[0.6, 0.7]$ | $0.210$ | $0.204$ | $0.216$ | $0.205$ |
| $[0.7, 1]$ | $0.085$ | $0.077$ | $0.083$ | $0.075$ |
| $[1, 2]$ | $0.047$ | $0.059$ | $0.051$ | $0.049$ |
| $[2, s_\infty]$ | $0.013$ | $13$ | $33$ | $57$ |
| outside corner | $0.112$ | $0.122$ | $0.124$ | $0.127$ |

On the three bands $s \le 2$ the predicted floor accounts for the measured error to within
$10$ per cent, for the three terminal functions alike, and the remainder
$\|(\Phi_\theta - V_{DO}) - w\|/\|V_{DO}\|$ is $0.03$--$0.05$ on $[0.6, 0.7]$ and $0.007$--$0.03$
elsewhere (at 20000 iterations the remainder was $0.04$--$0.07$ and $0.02$--$0.13$). The far
band $[2, s_\infty]$ is the separate, unconstrained far-field component of section 12, not
explained by $\delta$. Conclusion (measured): the error floor of the corner-excluded runs on
the training domain is the propagated terminal-trace defect $-(1-\zeta)(K-s)^+$ of the
$\zeta$-switched ansatz -- a property of the ansatz, identical for every terminal function
that equals the payoff at $t = T$, and unreachable by training. Its size is set by $\varepsilon$
(the support of $\delta$) and by $K - B$ (its amplitude); the corner exclusion window does not
change it, since $w$ is determined by the data, not by where the residual is enforced.

## 14. Cost of the split-semigroup route: closed-form evaluation of the profile

**Observation (measured).** The five split-semigroup runs of section 12.1 took $1.06$--$1.07$ s
per iteration on `republique` (4 threads, `n_f = 4096`, float32) against $0.083$--$0.085$ s for the
two-term Black-Scholes route and $0.111$--$0.119$ s for the ordinary route: a factor $12.6$. The
whole excess is in the evaluation of $\mathcal L^{BS} h_\varepsilon^{\mathrm{split}}$, whose
profile $\pi$ was obtained by `GaussianSemigroupExtensionField` as a fixed-grid trapezoidal
convolution on $n_{\mathrm{quad}} = 8000$ log-price nodes, recomputed at every iteration for every
collocation point ($O(n_f \times n_{\mathrm{quad}})$ exponentials per call).

**Closed form.** For the put datum the convolution is an explicit integral. Let $k = \ln K$,
$x = \ln s$, $m = \sigma_c\sqrt{T-t}$, $d_1 = (k - x)/m$ and $d_2 = d_1 - m$, with $\Phi$ and
$\varphi$ the standard normal distribution and density. Completing the square in
$\int_{-\infty}^{k} (K - e^{y})\,\varphi_m(x - y)\,dy$ gives, for $t < T$,

$$
\pi(x, t) = K\,\Phi(d_1) - e^{x + m^2/2}\,\Phi(d_2),
\qquad
\partial_x \pi = -e^{x + m^2/2}\,\Phi(d_2),
\qquad
\partial_{xx} \pi = -e^{x + m^2/2}\,\Phi(d_2) + \frac{K\,\varphi(d_1)}{m},
$$

and $\partial_t \pi = -\nu_c\,\partial_{xx}\pi$ by the heat equation the profile satisfies
($\nu_c = \sigma_c^2/2$). The first-derivative simplification uses the identity
$e^{x + m^2/2}\varphi(d_2) = K\varphi(d_1)$. The term $K\varphi(d_1)/m$ is the Dirac mass of weight
$K$ (the jump of $\partial_y (K-e^y)^+$ at $k$) smoothed by the kernel; it diverges as
$(T-t)^{-1/2}$ at $x = k$, which is the behaviour the quadrature class documents. At $t = T$ the
datum, its one-sided derivative and a zero second derivative are returned, matching the
quadrature class's terminal-slice convention.

**Rigour.** The closed form evaluates the *same* mathematical object as the quadrature, exactly:
there is no support truncation ($y_{\mathrm{lo}}, y_{\mathrm{hi}}$), no resolution error, and no
unresolved band of width $\sim (\Delta y/\sigma_c)^2$ near the terminal slice (the
`time_to_terminal_floor` of the quadrature class is zero here and its floor report never counts
an activation). The two-term assembly of the interior residual,
$\mathcal L^{BS}(g_1 u_\theta + h_\varepsilon^{\mathrm{split}}) = \mathcal L^{BS}(g_1 u_\theta) + \mathcal L^{BS} h_\varepsilon^{\mathrm{split}}$,
is unchanged; only the evaluation of the second, parameter-independent term differs. The
change therefore removes a numerical approximation rather than introducing one. The closed
form applies to the put datum only; a network-valued or glued datum (the Bermudan stage datum
$\max(g, C)$) has no closed form and keeps the quadrature route.

**Measured agreement and cost** (laptop, 4 threads, `n_f = 4096`, float32, one training
iteration including backward and optimiser step, $\sigma_c = \sigma = 0.3$, $\varepsilon = 0.1$,
far-field Dirichlet, same collocation batch):

| Route | Iteration | $\mathcal L^{BS} h_\varepsilon$ alone |
|---|---|---|
| Black-Scholes, two-term analytic | $44.9$ ms | $1.35$ ms |
| Split, quadrature $n_{\mathrm{quad}} = 8000$ | $163.1$ ms | $119.8$ ms |
| Split, closed form | $44.6$ ms | $1.25$ ms |

Agreement between the two split routes on the batch: $\max|h_{\mathrm{quad}} - h_{\mathrm{closed}}| = 4.2\times10^{-5}$
on values up to $20.6$, and $\max|\mathcal L^{BS}h_{\mathrm{quad}} - \mathcal L^{BS}h_{\mathrm{closed}}| = 4.5$ on
values up to $5.2\times10^{6}$ (the $(T-t)^{-1/2}$ divergence at collocation points close to the
slice), i.e. a relative discrepancy of $10^{-6}$ that is the quadrature's own error.
`test/pricing/test_barrier.py` pins both routes against an independent closed form written in
the test (a different code path from the production class) on value, second price derivative
and Black-Scholes residual. The split route is thereby brought to the per-iteration cost of the
Black-Scholes two-term route; on `republique` the expected gain is the measured factor $12.6$
(to be confirmed by a run tagged `_closedform`).

**Code.** `pde.real_line_extension_fields.PutPayoffGaussianSemigroupExtensionField` (the
profile and its analytic derivatives, one shared evaluation for $\pi$, $\partial_x\pi$,
$\partial_{xx}\pi$); `pricing.barrier.make_corner_regularised_extension_split(profile=...)`
with `SPLIT_PROFILE_ROUTES = ("closed_form", "quadrature")`, closed form by default;
`pilot_down_and_out_put.py --split-profile {closed_form,quadrature}` (default `closed_form`,
run tag `_closedform` or `_nquad<n>`). `load_trained_model` falls back to the quadrature route
for metadata that predates the `split_profile` key, so the runs of sections 10--13 replot
faithfully; a resume keeps the on-disk route and warns if the command line asks for the other.

## 15. Exact singular subtraction at the corner (Method 1, Section 5.1 of Doc A)

The smoothing construction of sections 2--14 spreads the corner jump $\Delta = K - B$ over the
layer of bandwidth $\varepsilon$ by the cutoff $\zeta((s-B)/\varepsilon)$; the interior residual
of the extension is of order $\Delta\varepsilon^{-1}$ in that layer, and section 13 identified
the transition band of $\zeta$ as the localised floor of the error. Method 1 of Doc A replaces
the smoothing by a closed-form, $\mathcal L^{BS}$-exact function reproducing the jump, so that
the singular part contributes nothing to the residual and no bandwidth remains. This section
records the construction implemented, the one choice it required beyond Doc A, its
verification, and the batch launched to measure it.

### 15.1 Construction

Let $V_{DOD}:\overline Q\to[0,1]$ be the price of the down-and-out cash-or-nothing claim of
unit payoff $\mathbf 1_{s>B}$ knocked out at $B$ (equation (15) of Doc A),

$$
V_{DOD}(s,t) = e^{-r(T-t)}\Big[N\big(d_-(s,t)\big) - (s/B)^{1-2r/\sigma^2}\,N\big(d_-(B^2/s,\,t)\big)\Big],
\qquad
d_-(s,t) = \frac{\ln(s/B) + (r-\tfrac12\sigma^2)(T-t)}{\sigma\sqrt{T-t}} .
$$

It solves $\mathcal L^{BS}V_{DOD} = 0$ on $Q$, vanishes on $\Sigma_B$ (the two terms coincide at
$s = B$) and equals $\mathbf 1_{s>B}$ on $\Sigma_T$. The subtracted estimator of Definition 7 is

$$
\Phi_\theta = \Delta\,V_{DOD} + h + d_{\partial_pQ}\,\Psi_\theta,
\qquad d_{\partial_pQ}(s,t) = (T-t)(s-B),
$$

with $h\in C^{2,1}(Q)\cap C^0(\overline Q)$ an extension of the data of the subtracted price
$\widetilde V_{DO} = V_{DO} - \Delta V_{DOD}$: terminal datum $g - \Delta\mathbf 1_{s>B}$, barrier
datum $0$, which coincide at the corner (Proposition 3 of Doc A).

**Choice of $h$.** Doc A leaves $h$ free. Let $\pi:\overline Q\to\mathbb R$ be a terminal
profile, a function with $\pi(s,T) = (K-s)^+$, taken among the three functions already used as
terminal functions of the smoothing ansatz: the raw payoff $\pi = (K-s)^+$, the European put
$\pi = V^e$, and the split-semigroup profile of section 14. The extension used is

$$
h(s,t) = \pi(s,t) - \pi(B,t).
$$

On $\Sigma_T$, $h(s,T) = (K-s)^+ - (K-B) = g(s) - \Delta$ for $s > B$, the subtracted terminal
datum; on $\Sigma_B$, $h(B,t) = 0$ for every $t$, the subtracted barrier datum; at the corner
both traces are $0$. The subtraction of $\pi(B,t)$ rather than of the constant $\Delta$ is what
makes the barrier trace hold for the time-dependent profiles ($V^e(B,t)\neq K-B$ for $t<T$);
for the raw payoff $\pi(B,t) = \Delta$ and $h = (K-s)^+ - \Delta$ literally. The price of this
choice is that $h$ is not annihilated by the operator even when $\pi$ is: since $\pi(B,\cdot)$
depends on $t$ alone,

$$
\mathcal L^{BS}h = \mathcal L^{BS}\pi + \partial_t\pi(B,t) - r\,\pi(B,t),
$$

a bounded function up to the terminal face for the three profiles (for $V^e$ and the split
profile, $\partial_t\pi(B,t)$ is bounded as $t\to T$ because $B\neq K$), which the free network
absorbs. The trial solution is $g_1 u_\theta + g_2$ with $g_2 = \Delta V_{DOD} + h$ and the
unchanged $g_1 = d_{\partial_pQ}$ (or its far-field variant of section 12).

**Interior residual.** By linearity and Proposition 4 of Doc A,
$\mathcal L^{BS}\Phi_\theta = \mathcal L^{BS}(g_1 u_\theta) + \mathcal L^{BS}h$, the digital term
contributing exactly zero. The residual is assembled through the two-term route of section 7:
$\mathcal L^{BS}(g_1u_\theta)$ by autograd on the network manifold, $\mathcal L^{BS}h$ in closed
form from the profile's derivatives ($\partial_s V^e = -N(-\tilde d_1)$,
$\partial_{ss}V^e = \varphi(\tilde d_1)/(s\sigma\sqrt{T-t})$, $\partial_t V^e$ from
$\mathcal L^{BS}V^e = 0$; the split profile's derivatives from section 14). Autograd through the
full trial solution is not used: $\partial_{ss}V_{DOD}$ is unbounded at the corner (it grows like
$(T-t)^{-1}$ along a path of fixed similarity variable), and autograd would evaluate the zero
residual of the digital as the difference of large terms in float32 at every collocation point
close to $(B,T)$. The omitted digital residual is exactly zero for the constant coefficients
$(r,\sigma)$ the digital is built with only (Remark 8 of Doc A); the implementation refuses
other coefficients rather than returning a residual missing the term (19).

**No corner layer.** The collocation sampler covers the whole domain, corner included: there is
nothing to exclude, the residual of $g_2$ is bounded up to $(B,T)$, and both hard constraints
hold exactly on the whole parabolic boundary. The evaluation metrics keep the $\ell^1$ window
$N_{0.1}$ of section 11 so that $\mathrm{rel}_{L^2}(\Omega\setminus N_{0.1})$ is computed on the
same region as for the smoothing runs; $\mathrm{rel}_{L^2}(\Omega)$ (window included) is now a
meaningful metric as well.

**Chen--Mangasarian profile.** Not offered in this mode: it reintroduces a smoothing bandwidth
$\varepsilon_0$ at the strike, which is the class of calibration Method 1 removes at the corner,
and the family was set aside by the selection of section 10 ($8.2$ standard deviations behind
the Black-Scholes profile at its best $\varepsilon_0$).

### 15.2 Verification (`test/pricing/test_barrier.py`, float64)

- $\mathcal L^{BS}V_{DOD} = 0$ by autograd on a $600\times600$ grid of $(B+0.005, 3)\times(0, T-10^{-3})$:
  maximum $|\mathcal L^{BS}V_{DOD}| < 10^{-10}$ (measured $1.9\times10^{-16}$ on the development grid).
- The closed-form $\partial_sV_{DOD}$, $\partial_{ss}V_{DOD}$, $\partial_tV_{DOD}$ against autograd
  of the price: discrepancies $< 10^{-10}$, $< 10^{-8}$, $< 10^{-10}$.
- Traces: $V_{DOD}(B,t) = 0$ and $V_{DOD}(s,T) = \mathbf 1_{s>B}$ exactly; $g_2(s,T) = (K-s)^+$ down
  to $s = B + 10^{-9}$ and $g_2(B,t) = 0$ for every $t$, for the three profiles.
- $\mathcal L^{BS}g_2$ in closed form against autograd through the whole $g_2$ (digital included):
  discrepancy $< 10^{-9}$ off the strike; the residual stays below $1$ on a sequence of points
  approaching the corner ($s - B = 10\tau$, $\tau\in\{10^{-2},10^{-4},10^{-6}\}$).
- The subtracted split extension equals the smoothing split extension shifted by
  $\Delta V_{DOD} - \pi(B,t)$ where $\zeta = 1$; the quadrature route of the split profile agrees
  with the closed form to $10^{-6}$.

### 15.3 Math → code mapping

| Symbol | Code |
|---|---|
| $V_{DOD}$, $\partial_sV_{DOD}$, $\partial_{ss}V_{DOD}$, $\partial_tV_{DOD}$ | `down_and_out_digital_price`, `down_and_out_digital_price_and_derivatives` (`learning_option_pricing/pricing/barrier.py`) |
| $\pi$ and its derivatives | `RawPutPayoffTerminalProfile`, `BlackScholesPutTerminalProfile`, `SplitSemigroupPutTerminalProfile` (`value_and_derivatives`) |
| $g_2 = \Delta V_{DOD} + \pi - \pi(B,\cdot)$, $\mathcal L^{BS}h$, $\partial_s g_2$, $\partial_{ss}g_2$ | `SubtractedDigitalCornerExtension` (`__call__`, `black_scholes_residual`, `first_price_derivative`, `second_price_derivative`), built by `make_subtracted_digital_extension` |
| Corner treatment of a run | `pilot_down_and_out_put.py --corner-treatment {smoothing,subtraction}`; profile from the payoff flags (`--black-scholes-payoff`, `--split-payoff`, none = raw); metadata keys `corner_treatment`, `subtraction_terminal_profile`; directory tag `_subtraction_<profile>`, placeholder `eps0` in file names |
| Decomposition figure | `figures/subtraction_decomposition.png` ($\Delta V_{DOD}$, $h$, $g_1u_\theta$, $\Phi_\theta$, $V_{DO}$ at $t\in\{0,0.5,0.9\}$) |
| Aggregation and Greeks | `aggregate_terminal_function_comparison.py`, `evaluate_greeks_no_corner.py`: configurations `subtraction_{raw,blackscholes,split}`, collected regardless of `--epsilon` and of the corner-exclusion filter, labelled "[corner included in collocation]" in the figures |

### 15.4 Measured (50000 iterations, 5 seeds per configuration, corner included in collocation)

Batch: `bash_scripts/cluster/cmap/joblist_50k_subtraction_republique.txt` (Black-Scholes and
split profiles, `republique`, 4 jobs $\times$ 4 threads) and `joblist_50k_subtraction_orleans.txt`
(raw-payoff baseline, `porte-d-orleans`, 5 jobs $\times$ 4 threads), launched 2026-09-20 20:17,
finished 22:18; $n_f = 4096$, float32, no far-field condition, seeds 0--4, the same seeds and
thread count as the canonical smoothing batch of section 11.1. Both hosts are AVX2 and
bit-identical for this pilot. Cost: $0.050$--$0.051$ s per iteration for the Black-Scholes and
split profiles and $0.057$--$0.063$ s for the raw profile (measured; $2510$--$3130$ s per run),
against $0.083$--$0.119$ s for the smoothing runs of section 14.

Aggregation: `data/aggregate_terminal_function_comparison/20260921_subtraction_vs_smoothing_iters50000/`
(`--hosts republique porte-d-orleans --s-band-edges 0.6 0.7 1 2 3`, `table.md`,
`model_based_diagnostics/s_band_errors.md`); Greeks:
`data/evaluate_greeks_no_corner/20260921_subtraction_vs_smoothing_iters50000/`. The smoothing
rows below are the canonical runs of section 11.1 (corner excluded from collocation,
$\varepsilon = 0.1$), re-read by the same aggregation; the subtraction rows include the corner.
Medians over 5 seeds, [min, max] in `table.md`.

| Configuration | $\mathrm{rel}_{L^2}(\Omega\setminus N_{0.1})$ | $\mathrm{rel}_{L^2}(\Omega)$ | $\mathrm{rel}_{L^2}(N_{0.1})$ | best loss |
|---|---|---|---|---|
| Smoothing, Black-Scholes, ordinary route | $1.22\times10^{-1}$ | $1.86\times10^{-1}$ | $5.84\times10^{-1}$ | $3.7\times10^{-5}$ |
| Smoothing, Black-Scholes, two-term route | $1.24\times10^{-1}$ | $1.83\times10^{-1}$ | $5.96\times10^{-1}$ | $1.7\times10^{-5}$ |
| Smoothing, split-semigroup | $1.27\times10^{-1}$ | $1.91\times10^{-1}$ | $5.86\times10^{-1}$ | $1.3\times10^{-5}$ |
| Exact subtraction, raw payoff | $4.55\times10^{-1}$ | $4.42\times10^{-1}$ | $5.4\times10^{-5}$ | $2.6\times10^{-7}$ |
| Exact subtraction, Black-Scholes | $4.69\times10^{-3}$ [$2.1\times10^{-3}$, $1.4\times10^{-2}$] | $4.55\times10^{-3}$ | $4.3\times10^{-5}$ | $1.7\times10^{-8}$ |
| Exact subtraction, split-semigroup | $3.97\times10^{-3}$ [$9.4\times10^{-4}$, $5.2\times10^{-3}$] | $3.85\times10^{-3}$ | $4.1\times10^{-5}$ | $1.1\times10^{-8}$ |

Per band of $s$ (all $t$, $N_{0.1}$ removed, medians; `s_band_errors.md`):

| Configuration | $[0.6, 0.7]$ | $[0.7, 1]$ | $[1, 2]$ | $[2, s_\infty]$ (abs. $L^2$) |
|---|---|---|---|---|
| Smoothing, three configurations | $0.20$--$0.22$ | $0.075$--$0.083$ | $0.049$--$0.059$ | $1.5$--$6.4\times10^{-3}$ |
| Exact subtraction, raw payoff | $4.0\times10^{-2}$ | $0.34$ | $1.29$ | $7.8\times10^{-4}$ |
| Exact subtraction, Black-Scholes | $1.4\times10^{-4}$ | $1.8\times10^{-4}$ | $1.1\times10^{-3}$ | $3.5\times10^{-4}$ |
| Exact subtraction, split-semigroup | $1.0\times10^{-4}$ | $1.2\times10^{-4}$ | $5.2\times10^{-4}$ | $3.0\times10^{-4}$ |

Greeks at the strike (`greeks_no_corner_table.md`, medians over 5 seeds of the relative error;
the smoothing values are those of section 11.1):

| $t$ | $\mathrm{err}_{\mathrm{rel}}\,\Delta$, smoothing $\to$ subtraction (Black-Scholes / split) | $\mathrm{err}_{\mathrm{rel}}\,\Gamma$, smoothing $\to$ subtraction |
|---|---|---|
| $0$ | $0.10$--$0.14 \to 1.5\times10^{-4}$ / $3.0\times10^{-4}$ | $0.017$--$0.067 \to 1.5\times10^{-3}$ / $3.0\times10^{-3}$ |
| $0.5$ | $0.079$--$0.084 \to 2.4\times10^{-4}$ / $2.1\times10^{-4}$ | $0.24$--$0.25 \to 1.1\times10^{-3}$ / $9.7\times10^{-4}$ |
| $0.9$ | $4\times10^{-4}$--$2.6\times10^{-3} \to 7.9\times10^{-5}$ / $1.9\times10^{-4}$ | $1.1$--$4.2\times10^{-3} \to 8.3\times10^{-5}$ / $5.9\times10^{-4}$ |

**Reading.** With the Black-Scholes or the split profile, the exact subtraction lowers the
comparison metric by a factor of $26$ to $32$ (medians; the across-seed ranges,
$[2.1\times10^{-3}, 1.4\times10^{-2}]$ and $[9.4\times10^{-4}, 5.2\times10^{-3}]$ against
$[0.083, 0.31]$, do not overlap), the error in the former transition band $[0.6, 0.7]$ by a
factor of about $1400$, and the best interior loss by a factor of about $1000$. The corner
window, on which nothing was enforced before, now has a relative error of $4\times10^{-5}$ with
the corner included in collocation. The floor of section 13 is therefore removed with the
smoothing scale, which is the prediction of Doc A's Proposition 4 and of section 13's
attribution of the floor to the switching factor. What remains is the far-field component of
section 12: on $[2, s_\infty]$ the absolute error ($3.0$--$3.5\times10^{-4}$) is $15$ to $30$ times
the one on the three inner bands, and it is the largest share of the residual error; the
far-field Dirichlet condition of section 12 (`--far-field-dirichlet`, compatible with this
ansatz) is the next step and has not been measured with the subtraction yet. The Delta at the
strike is reproduced to $1$--$3\times10^{-4}$ at every time and the Gamma to $10^{-3}$ or better
away from maturity, against $0.08$ and $0.24$ at $t = 0.5$ for the smoothing runs.

The raw-payoff profile is the negative control: with the corner resolved exactly, the error is
the strike singularity's alone, $0.34$ on $[0.7, 1]$ and $1.29$ on $[1, 2]$ (the extension
$h = (K-s)^+ - \Delta$ has a first-derivative discontinuity at $s = K$ and its residual
$-rs\mathbf 1_{s<K} - rh$ a jump there), while the corner band $[0.6, 0.7]$ is at $4\times10^{-2}$;
its Greeks at the strike are meaningless (the profile's one-sided derivative is evaluated exactly
at the kink). The two constructions that treat the strike singularity remain indistinguishable
at $n = 5$ (overlapping ranges), as in section 11.

**Comparability caveat.** The smoothing runs exclude the $\ell^1$ corner window from collocation
and have $\varepsilon = 0.1$; the subtraction runs include the corner and have no bandwidth. The
comparison is between the two treatments as each is meant to be run, not a one-factor ablation;
the figure labels the difference.

## 16. Corner enrichment (Method 2, Section 5.2 of Doc A)

Method 2 is the short-time limit of Method 1: when no closed-form digital exists, only the
leading similarity profile of the corner is retained. It is implemented on the same structure
as section 15 so that the two can be compared at equal budget.

### 16.1 Construction

Let $\xi(s,t) = \ln(s/B)/(\sigma\sqrt{2(T-t)})$ be the similarity variable of Lemma 1 and
$\Lambda = \mathrm{erf}$ the similarity profile, the bounded self-similar solution of
$\partial_\tau u = \tfrac12\sigma^2\partial_{yy}u$ on $y = \ln(s/B) > 0$ with unit inner datum and
null datum at $y = 0$. Let $\chi:[B,\infty)\to[0,1]$ be the fixed $C^\infty$ cutoff

$$
\chi(s) = 1 - \zeta\!\left(\frac{s - B - \delta_0}{\delta_1 - \delta_0}\right),
\qquad \chi \equiv 1 \text{ on } \{s - B \le \delta_0\},\quad \chi \equiv 0 \text{ on } \{s - B \ge \delta_1\},
$$

with $\zeta$ the smooth step of section 2 and radii $0 < \delta_0 < \delta_1$ (defaults
$\delta_0 = 0.1$, the canonical evaluation window, and $\delta_1 = 0.3$, the outer edge of the
band of section 11). The enriched estimator (22) is
$\Phi_\theta = E + h + d_{\partial_pQ}\Psi_\theta$ with

$$
E(s,t) = \chi(s)\,\Delta\,\Lambda\big(\xi(s,t)\big),
\qquad
h(s,t) = \pi(s,t) - \chi(s)\,\pi(B,t),
$$

for the same three terminal profiles $\pi$ as section 15. Traces: on $\Sigma_T$, $\Lambda\to1$
so $E(s,T) = \chi(s)\Delta$ and $h(s,T) = (K-s)^+ - \chi(s)\Delta = g - E(\cdot,T)$, the residual
terminal datum of Definition 8; on $\Sigma_B$, $\xi = 0$ gives $E = 0$ and $\chi(B) = 1$ gives
$h = 0$. The cutoff depends on $s$ only — the same simplification of the $\ell^1$ ball as the
time-independent $h_\varepsilon$ of section 5 — which is what lets the barrier trace of $h$ hold
for every $t$; its transition strip $\{\delta_0 < s - B < \delta_1\}$ is at distance $\delta_0 > 0$
from the corner, as the commutator estimate of Proposition 5 requires. Section 15's choice
$h = \pi - \pi(B,\cdot)$ is the case $\chi\equiv1$ of this formula, and both extensions are
instances of one class (`_AnalyticallyResolvedCornerExtension`, $g_2 = S + \pi - \chi\,\pi(B,\cdot)$
with $S$ the singular part).

**Interior residual.** Unlike the digital, $\Lambda(\xi)$ is not annihilated by
$\mathcal L^{BS}$: in the variable $y$ the operator reads
$\partial_t + \tfrac12\sigma^2\partial_{yy} + (r - \tfrac12\sigma^2)\partial_y - r$, the first two
terms cancel on $\Lambda$, and

$$
\mathcal L^{BS}\Lambda = \big(r - \tfrac12\sigma^2\big)\,\partial_y\Lambda - r\Lambda,
\qquad \partial_y\Lambda = \frac{\Lambda'(\xi)}{\sigma\sqrt{2(T-t)}} = \frac{2}{\sqrt\pi}\,\frac{e^{-\xi^2}}{\sigma\sqrt{2(T-t)}},
$$

of order $(T-t)^{-1/2}e^{-\xi^2}$: unbounded at the corner along paths of fixed $\xi$, but
square-integrable on $Q$ (Proposition 5) and with no small parameter (Remark 10). The full
residual $\mathcal L^{BS}(E + h)$, commutator terms of $\chi$ included, is assembled in closed
form from the profile's derivatives ($\partial_s\Lambda = \Lambda'\,\partial_s\xi$,
$\partial_{ss}\Lambda = \Lambda''(\partial_s\xi)^2 + \Lambda'\partial_{ss}\xi$,
$\partial_t\Lambda = \Lambda'\,\xi/(2(T-t))$) and enters the two-term route as in section 15.
The training loss samples a residual that is not zero at the corner, in contrast with Method 1;
the comparison between the two is precisely the exact-versus-leading-order row of Table 1 of
Doc A.

### 16.2 Verification (`test/pricing/test_barrier.py`, float64)

- $\Lambda$ solves the heat equation in $y$ ($\partial_t\Lambda + \tfrac12\sigma^2(s^2\partial_{ss} + s\partial_s)\Lambda = 0$)
  to $10^{-12}$ on a $400\times400$ grid; closed-form derivatives against autograd to
  $10^{-10}$, $10^{-8}$, $10^{-10}$.
- $V_{DOD}\to\Lambda(\xi)$ at fixed $\xi$ as $\tau\to0$ (the enrichment is the short-time limit
  of the digital): the gap decreases monotonically over $\tau\in\{10^{-2},10^{-3},10^{-4}\}$ and
  is below $10^{-3}$ at the last.
- Traces exact for the three profiles; $\chi = 1$ at $s = B + \delta_0$ and $0$ at
  $s = B + \delta_1$; $E$ vanishes beyond $\delta_1$ where $g_2$ reduces to $\pi$.
- $\mathcal L^{BS}g_2$, $\partial_sg_2$, $\partial_{ss}g_2$, $\partial_tg_2$ in closed form
  against autograd through the whole $g_2$: discrepancies $< 10^{-9}$, $10^{-10}$, $10^{-7}$,
  $10^{-9}$ off the strike; where $\chi = 1$ the residual of $E$ equals
  $\Delta[(r - \tfrac12\sigma^2)\partial_y\Lambda - r\Lambda]$ to $10^{-12}$.

### 16.3 Math → code mapping

| Symbol | Code |
|---|---|
| $\Lambda(\xi)$ and derivatives | `corner_similarity_profile_value_and_derivatives` |
| $g_2 = \chi\Delta\Lambda + \pi - \chi\,\pi(B,\cdot)$, $\mathcal L^{BS}g_2$, $\partial_sg_2$, $\partial_{ss}g_2$ | `CornerEnrichedExtension` (`enrichment`, `regular_part`, `black_scholes_residual`, `first_price_derivative`, `second_price_derivative`), built by `make_corner_enriched_extension` |
| $\delta_0$, $\delta_1$ | `pilot_down_and_out_put.py --corner-treatment enrichment --enrichment-delta0 --enrichment-delta1`; directory tag `_enrichment_<profile>_d0<δ0>_d1<δ1>`; metadata keys `enrichment_delta0`, `enrichment_delta1` |
| Aggregation and Greeks | configurations `enrichment_{raw,blackscholes,split}`, same conventions as section 15 |

### 16.4 Measured (50000 iterations, 5 seeds per configuration, corner included in collocation)

Batch: `bash_scripts/cluster/cmap/joblist_50k_enrichment_republique.txt` (Black-Scholes and split
profiles, `republique`) and `joblist_50k_enrichment_orleans.txt` (raw profile, `porte-d-orleans`),
$\delta_0 = 0.1$, $\delta_1 = 0.3$, launched 2026-09-21 00:30/00:33, finished 01:24/02:40;
otherwise identical to the subtraction batch of section 15.4 ($0.053$ s per iteration measured).
Aggregation of the nine configurations:
`data/aggregate_terminal_function_comparison/20260921_corner_treatments_iters50000/`; Greeks:
`data/evaluate_greeks_no_corner/20260921_corner_treatments_iters50000/`; figure report:
`rapports/corner_treatments_20260921/corner_treatment_figures.pdf`
(`build_corner_treatment_figure_report.py`). Medians over 5 seeds, [min, max] in `table.md`.

| Configuration | $\mathrm{rel}_{L^2}(\Omega\setminus N_{0.1})$ | $\mathrm{rel}_{L^2}(N_{0.1})$ | $[0.6,0.7]$ | $[0.7,1]$ | $[1,2]$ | best loss |
|---|---|---|---|---|---|---|
| Smoothing, Black-Scholes / split (section 11.1) | $0.12$--$0.13$ | $0.58$--$0.59$ | $0.20$--$0.22$ | $0.075$--$0.083$ | $0.049$--$0.059$ | $1$--$4\times10^{-5}$ |
| Exact subtraction, Black-Scholes | $4.7\times10^{-3}$ [$2.1\times10^{-3}$, $1.4\times10^{-2}$] | $4.3\times10^{-5}$ | $1.4\times10^{-4}$ | $1.8\times10^{-4}$ | $1.1\times10^{-3}$ | $1.7\times10^{-8}$ |
| Exact subtraction, split | $4.0\times10^{-3}$ [$9.4\times10^{-4}$, $5.2\times10^{-3}$] | $4.1\times10^{-5}$ | $1.0\times10^{-4}$ | $1.2\times10^{-4}$ | $5.2\times10^{-4}$ | $1.1\times10^{-8}$ |
| Corner enrichment, Black-Scholes | $7.3\times10^{-3}$ [$2.2\times10^{-3}$, $1.8\times10^{-2}$] | $7.9\times10^{-4}$ | $4.4\times10^{-4}$ | $5.8\times10^{-4}$ | $1.7\times10^{-3}$ | $2.8\times10^{-7}$ |
| Corner enrichment, split | $6.9\times10^{-3}$ [$2.9\times10^{-3}$, $1.5\times10^{-2}$] | $8.7\times10^{-4}$ | $5.3\times10^{-4}$ | $8.4\times10^{-4}$ | $3.0\times10^{-3}$ | $4.4\times10^{-7}$ |
| Corner enrichment, raw payoff | $0.456$ | $6.4\times10^{-4}$ | $3.9\times10^{-2}$ | $0.34$ | $1.30$ | $6.2\times10^{-7}$ |

Greeks at the strike (medians over 5 seeds of the relative error):

| $t$ | $\mathrm{err}_{\mathrm{rel}}\,\Delta$: subtraction / enrichment (Black-Scholes profile) | $\mathrm{err}_{\mathrm{rel}}\,\Gamma$: subtraction / enrichment |
|---|---|---|
| $0$ | $1.5\times10^{-4}$ / $1.4\times10^{-3}$ | $1.5\times10^{-3}$ / $4.7\times10^{-2}$ |
| $0.5$ | $2.4\times10^{-4}$ / $1.2\times10^{-3}$ | $1.1\times10^{-3}$ / $4.9\times10^{-3}$ |
| $0.9$ | $7.9\times10^{-5}$ / $6.0\times10^{-4}$ | $8.3\times10^{-5}$ / $9.6\times10^{-4}$ |

**Reading.** The enrichment removes the smoothing floor as the subtraction does: the comparison
metric drops from $0.12$--$0.13$ to $7\times10^{-3}$ (a factor of $17$ to $18$; the across-seed
ranges $[2.2\times10^{-3}, 1.8\times10^{-2}]$ do not overlap with the smoothing ones), the former
transition band from $0.20$ to $5\times10^{-4}$, and the corner window to $8\times10^{-4}$. It is
however measurably behind the exact subtraction on every quantity that is sensitive to the
corner: corner-window error $8\times10^{-4}$ against $4\times10^{-5}$ (a factor of $19$),
band $[0.6, 0.7]$ $4.4$--$5.3\times10^{-4}$ against $1.0$--$1.4\times10^{-4}$ (a factor of $3$ to $5$),
best interior loss $3$--$4\times10^{-7}$ against $1$--$2\times10^{-8}$ (a factor of $20$ to $40$),
Gamma at the strike $5\times10^{-3}$ against $1\times10^{-3}$ at $t = 0.5$ and $5\times10^{-2}$
against $1.5\times10^{-3}$ at $t = 0$. On the comparison metric itself the two treatments are close
(medians $7\times10^{-3}$ against $4$--$5\times10^{-3}$, ranges overlapping at $n = 5$), because that
metric is dominated by the far-field component of section 12 in both cases (absolute error
$5.2$--$5.5\times10^{-4}$ on $[2, s_\infty]$ for the enrichment, $3.0$--$3.5\times10^{-4}$ for the
subtraction, against $10^{-5}$ on the inner bands). The ordering of the losses is the one predicted
by Table 1 of Doc A: the digital contributes exactly zero residual (Proposition 4), the enrichment
a square-integrable but nonzero residual $\Delta[(r-\tfrac12\sigma^2)\partial_y\Lambda - r\Lambda]$
plus the commutator of the cutoff (Proposition 5), which the interior loss must absorb through the
network. Under constant coefficients, where the closed form exists, Method 1 is therefore the one to
use; Method 2 is the fallback for variable coefficients, and these measurements bound the price of
the fallback for this contract at a factor of $3$ to $20$ on the corner quantities and none on the
far-field-dominated global metric. The raw-payoff profile again fails at the strike ($0.34$ on
$[0.7, 1]$), identically for the two treatments, confirming that this failure is the strike's and
not the corner's. The cutoff radii $(\delta_0, \delta_1) = (0.1, 0.3)$ were not swept; whether a
wider $\delta_0$ narrows the gap to Method 1 is not measured.
