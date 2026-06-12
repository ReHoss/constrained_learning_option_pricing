# Terminal-condition enforcement forms on the backward heat equation

> Math rendering: open in a Markdown+KaTeX/MathJax renderer for rendered equations.

This document describes the ansatz-form ablation requested for the report
*On boundary-constrained learning of partial differential equations*
(`latex_documents/reports/2026_01_29_constrained_learning_pde_lehalle_hosseinkhan`).
It compares how the **terminal (initial) condition** of a backward parabolic PDE is
enforced inside a physics-informed neural network, and monitors each component of
the training objective separately.

Library code: `learning_option_pricing.pde` (operator + references),
`learning_option_pricing.models.terminal_ansatz` (the four forms + decomposition).
Experiment: `experiments/python_scripts/exp_ansatz_forms_heat/`.

---

## 1. Problem

The backward heat (terminal-value) equation is solved on a bounded spatial window
$\mathcal{X} = [x_{\mathrm{lo}}, x_{\mathrm{hi}}]$ and $t \in [0, T]$:

$$
\mathcal{P} u := \partial_t u + \tfrac{\sigma^2}{2}\,\partial_{xx} u = 0,
\qquad u(x, T) = g(x).
$$

This is the parabolic counterpart of the Black–Scholes operator
(`pricing.terminal.bsm_operator`) in the same terminal-value convention; under the
substitution $\tau = T - t$ it is a *forward* heat equation, hence well posed
(the mode amplitudes decay as $t$ decreases from $T$).

The PDE is posed on all of $\mathbb{R}$; the bounded window only supports the
Monte-Carlo sampling measure for the residual and terminal terms. **No lateral
boundary condition is imposed.** Accuracy is reported on an inner evaluation
window $[x_{\mathrm{eval,lo}}, x_{\mathrm{eval,hi}}]$ leaving a buffer of a few
diffusion lengths $\sigma\sqrt{T}$ from the sampling edges, so the edge
under-determination (the Gaussian domain of dependence reaches outside the
sampled window) contributes only exponentially-small error to the metrics. The
boundary mismatch is retained as a *diagnostic* only.

### Initial (terminal) conditions

| id | terminal datum $g(x)$ | window | exact solution |
|----|----|----|----|
| `sine` | $\sin(\pi x) + c\,\sin(f\pi x)$ | $[0,1]$ | $e^{\frac{\sigma^2\pi^2}{2}(t-T)}\sin(\pi x) + c\,e^{\frac{\sigma^2 f^2\pi^2}{2}(t-T)}\sin(f\pi x)$ |
| `theta3` | $1 + 2\sum_{n\ge1} e^{-n^2}\cos(\pi n x)$ | $[-1,1]$ | $1 + 2\sum_{n\ge1} e^{-n^2+\frac{\sigma^2\pi^2 n^2}{2}(t-T)}\cos(\pi n x)$ |
| `call` | softplus$_\beta(e^x - K) - \tfrac{\log 2}{\beta}$ | $[\ln 20, \ln 200]$, eval $[\ln 60, \ln 140]$ | $e^{x+\sigma^2\tau/2}N(d_1) - K\,N(d_2)$, $\tau = T-t$ |

The `sine` and `theta3` references derive from `cal_notes/example_heat.tex`. The
`call` datum is the smoothed payoff of the singularity study, here placed under the
*pure* heat operator (drift and reaction terms dropped — see §5).

**Sign-convention note (theta_3).** The source note writes the $\vartheta_3$
solution with growing mode amplitudes $e^{+\frac{\sigma^2\pi^2 n^2}{2}(T-t)}$,
which actually solves $\partial_t u - \tfrac{\sigma^2}{2}\partial_{xx}u = 0$ and is
inconsistent with its stated operator. We implement the operator-consistent
**decaying** solution ($e^{(t-T)}$ exponent), matching the `sine` reference and the
Black–Scholes convention. The problem is then well posed for any $T$ and no
divergence horizon is needed.

---

## 2. The four forms

Let $\Phi_\theta(x,t)$ be the free network and $\lambda(t)$ the interpolation
coefficient with $\lambda(T)=1$, so the network prefactor $1-\lambda(t)$ vanishes
at the terminal time.

| form | trial solution $\hat u(x,t)$ | terminal enforcement |
|----|----|----|
| `hard_constant` | $(1-\lambda)\,\Phi_\theta + g$ | exact (report `eq:bermudan-ansatz`) |
| `hard_convex` | $(1-\lambda)\,\Phi_\theta + \lambda\,g$ | exact (report `eq:bermudan-ansatz-alt`) |
| `soft_pinn` | $\Phi_\theta$ | penalty in loss |
| `pure_nn` | $\Phi_\theta$ | none — non-identifiable control |

The two hard forms differ only in the terminal-data extension $\Psi$:
`hard_constant` uses the time-constant $\Psi = g$, `hard_convex` the damped
$\Psi = \lambda g$. They are formally equivalent under the relabelling
$g \mapsto \lambda g$, but differ off the terminal slice and hence in training
dynamics — the object of study.

### Interpolation coefficient

$$
\lambda_{\text{linear}}(t) = \frac{t - t_{\text{start}}}{T - t_{\text{start}}},
\qquad
\lambda_{\text{exp}}(t) = e^{-\gamma (T - t)}.
$$

Both satisfy $\lambda(T)=1$. The `linear` form is the report's
$\lambda_k(t)=t/t_k$; its $b'/(1-b)=1/(T-t)$ is singular at $T$ (strong
enforcement). The `exponential` form preserves hard enforcement (unlike the
source note's $b=1-e^{-(T-t)}$, which gives $b(T)=0$ and does **not** enforce the
terminal condition); at the eigenvalue-matched rate $\gamma=\sigma^2\pi^2/2$ it
reproduces the note's "ideal" single-mode interpolation coefficient. The interpolation-coefficient sweep applies to
the two hard forms only.

---

## 3. Loss components (separately monitored)

The total objective per form:

$$
\text{hard / pure: } \mathcal{L} = \mathcal{L}_{\rm pde},
\qquad
\text{soft: } \mathcal{L} = a\,\mathcal{L}_{\rm pde} + (1-a)\,\mathcal{L}_{\rm tc},
$$

with $\mathcal{L}_{\rm pde} = \mathbb{E}_{\mu}[(\mathcal{P}\hat u)^2]$ and
$\mathcal{L}_{\rm tc} = \mathbb{E}_{\mu_{|T}}[(\Phi_\theta(\cdot,T)-g)^2]$. The
`pure_nn` control omits $\mathcal{L}_{\rm tc}$ entirely (and is therefore
under-determined). No spatial-boundary term enters the loss.

### Stage-residual decomposition (`rem:residual-decomposition`)

For the hard forms the residual splits, in the report's notation, as

$$
\mathcal{P}\,\hat u =
\underbrace{(1-\lambda)\,\mathcal{P}\Phi_\theta - \lambda'\,\Phi_\theta}_{R_\theta\ \text{(network contribution)}}
+ \underbrace{\mathcal{P}\Psi}_{\text{extension forcing}},
$$

and the loss into three monitored channels

$$
\mathcal{L} =
\underbrace{\mathbb{E}_\mu[R_\theta^2]}_{\text{network energy}}
+ \underbrace{2\,\mathbb{E}_\mu[R_\theta\,\mathcal{P}\Psi]}_{\text{cross term}}
+ \underbrace{\mathbb{E}_\mu[(\mathcal{P}\Psi)^2]}_{\theta\text{-independent floor}}.
$$

The floor cannot be removed by optimising $\theta$; only the choice of extension
controls it. For `hard_constant` ($\Psi=g$) it is
$\mathbb{E}_\mu[(\tfrac{\sigma^2}{2}g'')^2]$; for `hard_convex` ($\Psi=\lambda g$)
it is $\mathbb{E}_\mu[(\lambda' g + \tfrac{\sigma^2}{2}\lambda g'')^2]$. Contrasting
these floors across the two hard forms is the quantitative core of the study. The
decomposition identity $\mathcal{P}\hat u = R_\theta + \mathcal{P}\Psi$ is verified
numerically in `test/models/test_terminal_ansatz.py`.

### Forcing split by mechanism (interpolation-velocity vs damped-diffusion)

The forcing $\mathcal{P}\Psi$ splits into its two operator parts
$\mathcal{P}\Psi = \partial_t\Psi + \tfrac{\sigma^2}{2}\partial_{xx}\Psi$, and
the floor is monitored separately along each:

$$
\underbrace{\mathbb{E}_\mu[(\partial_t\Psi)^2]}_{\text{interpolation-velocity}},
\qquad
\underbrace{\mathbb{E}_\mu[(\tfrac{\sigma^2}{2}\partial_{xx}\Psi)^2]}_{\text{damped diffusion}}.
$$

For the convex-combination form $\Psi=\lambda(t)\,g(x)$ these are
$\mathbb{E}_\mu[(\lambda' g)^2]$ and $\mathbb{E}_\mu[(\lambda\tfrac{\sigma^2}{2}g'')^2]$;
the first is large when $g$ has a **nonzero mean** (e.g. theta_3, where
$\lambda' g \approx g/T$ injects a spurious forcing the constant form avoids),
the second is large when $g''$ is **sharp** (e.g. the call payoff). The split is
the channels `forcing_velocity` / `forcing_diffusion`, plotted in the bottom row
of `loss_decomposition.png`; for a general (e.g. time-dependent CM) extension the
same operator-part split applies via
`learning_option_pricing.pde.heat_operator_parts`.

Per-iteration history records: total loss, $\mathcal{L}_{\rm pde}$,
$\mathcal{L}_{\rm tc}$ (diagnostic), boundary drift (diagnostic), the three
decomposition channels, the two forcing sub-channels, gradient norm and learning
rate.

---

## 4. Experiment structure

* **Catalogue** `_ansatz_forms_catalogue.py` — torch-free; defines the six method
  variants (4 hard = 2 forms × 2 interpolation coefficients, plus soft and pure) and the three IC
  configs. Importable on a login node for the `--init-only` phase.
* **Runner** `ablation_ansatz_forms.py` — Adam, cosine LR schedule, best-state
  restoration, deterministic master-seed → role-tagged per-role seeds
  (`derive_seed` via blake2b), self-contained run log. Modes: single variant
  (`--variant`, array task), all variants for one IC (local), `--replot`,
  `--init-only`. Smoke-test guard at `SMOKE_TEST_NUM_ITERATIONS_THRESHOLD = 1000`;
  short runs require `--debug` (and are prefixed `_debug_`).
* **Plots** `_ansatz_forms_plots.py` — torch-free replot from saved
  `hist/metrics/slices.npz`; loss-component panels, decomposition channels, exact
  vs trained slices at $t=0$ and $t=T$, error at $t=0$, summary bars. Trained
  curves solid, references dashed, eval-window markers dotted, formula textbox
  below each figure (per the repository plot conventions).

Parallelism: one array task per `(ic, variant, seed)` triple writing to a shared
`--ablation-dir`, finalised by a `--replot` aggregation step.

---

## 5. Results (3 seeds per IC, 20000 iterations)

Aggregated by `ansatz_forms_cross_seed_summary.py` (figures under
`data/ansatz_forms_cross_seed_summary/<timestamp>_<n_ics>ic_<n_seeds>seed/`). Relative $L^2$ error against the
exact solution, mean over seeds $\{0,1,2\}$:

| | hard $\Psi{=}g$ lin | hard $\Psi{=}g$ exp | hard $\Psi{=}\lambda g$ lin | hard $\Psi{=}\lambda g$ exp | soft PINN | pure NN |
|---|---|---|---|---|---|---|
| sine | 6.7e-2 | 1.1e-1 | 1.0e-1 | **6.5e-2** | 1.9e-1 | 9.3e-1 |
| theta3 | **1.5e-3** | 1.5e-3 | 6.7e-3 | 1.9e-3 | 2.9e-2 | 9.6e-1 |
| call | 1.5e-1 | 1.7e-1 | **8.0e-2** | 9.0e-2 | 4.8e-3 | 1.0e0 |

Stylised facts:

1. **Terminal enforcement.** All four hard forms achieve $\mathcal{L}_{\rm tc}=0$
   exactly (by construction); the soft PINN leaves a small residual
   ($5\times10^{-3}$–$2\times10^{-2}$); the `pure_nn` control fails
   ($\approx 1$) — the residual-only inverse problem is non-identifiable
   without terminal data.
2. **Hard $\gg$ soft on smooth/periodic data** (sine, theta3), by 3–20×.
3. **Soft wins on the sharp call payoff** (4.8e-3 vs $\geq 8$e-2 for the hard
   forms): the smoothed-payoff extension has a large $g''$, so the hard ansatz
   carries a large extension-forcing floor $\mathbb{E}[(\mathcal{P}\Psi)^2]$ the
   network must cancel — a quantified instance of Remark *the extension is not
   innocuous*.
4. **The forcing floor predicts the hard-form ranking** (`floor_vs_accuracy.png`):
   across every IC the lower-floor extension is the more accurate one. The
   `theta3` reversal — where the *constant* extension $\Psi=g$ beats the convex-combination
   $\Psi=\lambda g$ — follows because $\vartheta_3$ has a nonzero mean, so the
   convex-combination form injects a spurious $\lambda'(t)\,g \approx g/T$ forcing that the
   constant form avoids; for the zero-mean sine and the sharp-$g''$ call the
   constant form's $\tfrac{\sigma^2}{2}g''$ dominates instead and the convex combination lowers
   the floor.

These observations are empirical (3 seeds, fixed network capacity and 20000
Adam iterations); seed variance is shown as error bars in `rel_l2_by_ic.png`.

## 6. `pure heat` vs the already-implemented `call` (singularity study)

The singularity experiment (`exp_singularity_european_call`) solves the
**Black–Scholes operator in log-price** coordinates,

$$
\mathcal{F}[V] = \partial_t V + \tfrac{\sigma^2}{2}\partial_{xx}V
  + \big(r - \tfrac{\sigma^2}{2}\big)\partial_x V - r V,
$$

i.e. heat **plus** a constant drift $(r-\sigma^2/2)\partial_x V$ and a reaction
$-rV$. The `call` IC here uses the **pure heat** operator
$\partial_t u + \tfrac{\sigma^2}{2}\partial_{xx}u$ (drift and reaction dropped,
equivalently $r=0$), so that all three ICs share one operator and the comparison
isolates the terminal-condition enforcement rather than confounding it with the
drift/discount terms. The terminal datum (smoothed payoff) and the inner-window
methodology are identical; only the operator differs.
