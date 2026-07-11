# Stage 2 — trained split-extension ablation: implementation specification

This document is the complete implementation specification for the trained
stage-2 ablation of the split-extension study.  It is written so that an
implementing agent can produce the code, tests, and cluster launches without
further questions.  Stage 1 (closed-form measurements, no training) is
documented in `documents/methodology/spectral_forcing_floor_and_split_extensions.md`
(referred to below as *the stage-1 note*); its library modules are
`learning_option_pricing/pde/periodic_spectral_toolbox.py` and
`learning_option_pricing/pde/terminal_data_extensions.py`.  The trained
infrastructure to be reused is the `exp_ansatz_forms_heat` experiment family
(`experiments/python_scripts/exp_ansatz_forms_heat/`), whose variant schema,
training loop, array-launcher contract, and reference hyperparameters are
pinned throughout.

Reviewer inputs.  `documents/draft.md` (read-only) contains two directives
that are honoured here: (i) the derivative bypass for the analytic terminal
function — since the extension satisfies known closed-form derivatives, the
automatic-differentiation engine must not differentiate the extension during
back-propagation (the substitution described for Zhang et al.'s $g_2$ term);
(ii) the exactness policy — analytical formulas are preferred to numerical
recomputation wherever a closed form exists.  The scientific hypothesis under
test is the stage-2 sentence of the stage-1 note, Section 5.5: *at equal
network capacity, the split extensions leave a residual floor smaller by the
measured unreachable-mass ratio*.

---

## 0. Setting and standing notation

All training cells are posed on the circle $[0, 2\pi)$ over the strip
$(0, T) \times [0, 2\pi)$ with terminal time $T = 1$.  Let
$A = \nu\,\partial_{xx} + \mu\,\partial_x + r_0$ be a constant-coefficient
spatial generator, with diffusivity $\nu \in (0, +\infty)$, advection
coefficient $\mu \in \mathbb{R}$, and reaction coefficient
$r_0 \in \mathbb{R}$.  The backward evolution problem is: find
$u : [0, T] \times [0, 2\pi) \to \mathbb{R}$ with

$$
\partial_t u + A u = 0 \ \text{ on } (0, T) \times [0, 2\pi),
\qquad
u(\cdot, T) = g,
$$

where $g \in L^2(0, 2\pi)$ is the terminal datum.  The Fourier symbol of $A$
is $a(k) = -\nu k^2 + i \mu k + r_0$ for $k \in \mathbb{Z}$ (the convention
of the stage-1 note, Section 3).  The residual operator is denoted
$P u = \partial_t u + A u$.

The trial solution is the hard-constrained form of the
`exp_ansatz_forms_heat` family: let
$\Phi_\theta : \mathbb{R}^2 \to \mathbb{R}$ be the free network and let
$\lambda : [0, T] \to \mathbb{R}$ be the interpolation coefficient with
$\lambda(T) = 1$; the trial solution is

$$
\hat u(x, t) = \bigl(1 - \lambda(t)\bigr)\, \Phi_\theta(x, t) + \Psi(x, t),
$$

where $\Psi$ is the terminal-data extension.  The terminal-distance factor is
$d_T(t) = 1 - \lambda(t)$; every stage-2 variant except the
single-spectral-component control uses the linear coefficient
$\lambda(t) = t / T$, so $d_T(t) = 1 - t/T$ and $d_T'(t) = -1/T$.  The
residual decomposes as
$P \hat u = R_\theta + P\Psi$ with network contribution
$R_\theta = (1 - \lambda) P\Phi_\theta - \lambda' \Phi_\theta$; the
$\theta$-independent forcing $P\Psi$ is the object the extension catalogue
controls.

Terminal datum of the generator cells.  The datum is the band-limited
truncation of the regularity-index-$1$ periodised Bernoulli datum of the
stage-1 note (Section 2.1), with truncation wavenumber $K_g = 128$:

$$
g(x) = \sum_{k=1}^{K_g} \frac{\cos(k x)}{\pi^2 k^2},
\qquad
c_k = \frac{1}{2 \pi^2 k^2} \ \ (1 \le |k| \le K_g),
\qquad
c_0 = 0,
$$

so that $|c_k| = 2/(2\pi k)^2$ holds exactly on the retained band, matching
the stage-1 datum.  The truncation makes the exact solution a finite sum of
decaying spectral components (Section 3.2 below), which is what permits every
error metric to be evaluated against an exact reference.  The full datum's
break point at $x^\star = 0$ becomes, after truncation, the point of maximal
oscillation concentration; the corner-window metrics of Section 3 are centred
there.  Rationale for $K_g = 128$: it is the largest bandwidth demand of any
convergent extension in the stage-1 P5 table ($\gamma = 10^{-6}$ for the
split $\{\partial_{xx}\}$), it is expected to lie well above the network's
reachable cutoff, and it keeps every extension evaluation a cheap finite sum.

---

## 1. Variant set (specification item 1)

### 1.1 Cells

Three cells, each a (generator, datum) pair.  A cell plays the role of the
`--ic` axis of `ablation_ansatz_forms.py`; the CLI flag is named `--cell`.

| Cell name | Generator (real space) | Symbol $a(k)$ | Datum |
|---|---|---|---|
| `g1_bernoulli_bandlimited` | $A = 0.7\,\partial_{xx} + 1.3\,\partial_x - 0.4$ (stage-1 G1) | $-0.7 k^2 + 1.3\,ik - 0.4$ | $g$ above, $K_g = 128$ |
| `g2_bernoulli_bandlimited` | $A = 0.125\,\partial_{xx} - 0.095\,\partial_x - 0.03$ (stage-1 G2, Black–Scholes log-price, $\sigma = 0.5$, $r = 0.03$) | $-0.125 k^2 - 0.095\,ik - 0.03$ | $g$ above, $K_g = 128$ |
| `heat_sine_single_component` | $A = 0.125\,\partial_{xx}$ (pure heat at the G2 diffusivity) | $-0.125 k^2$ | $g(x) = \sin x$ |

The pure-heat generator is confined to the control cell because, under pure
heat, the split $\{\partial_{xx}\}$ extension **is** the exact solution (the
defect symbol vanishes identically): a pure-heat generator admits no
non-trivial split comparison.  The trained comparison therefore runs on G1
and G2, both of which have a non-trivial defect (advection and reaction for
the split $\{\partial_{xx}\}$; reaction alone for the split
$\{\partial_{xx}, \partial_x\}$), exactly as in stage 1.

### 1.2 Variants of the generator cells (7 per cell)

Every variant below uses the linear interpolation coefficient
$\lambda(t) = t/T$ (distance factor $d_T(t) = 1 - t/T$), the same network,
the same sampler, and the same seeds; the intervention is the extension
alone.  Notation for the closed forms: for the split and graded classes let
$E_k(t) = e^{-(T - t)\,\nu_{\mathrm{eff}} k^2}$ denote the per-component
decay factor, where $\nu_{\mathrm{eff}}$ is the diffusivity named in each
row, and let $\theta_k(x, t) = k x + (T - t)\,\mu k$ denote the
phase-advected argument.

**V1 `convex_raw`** — existing baseline (`hard_convex` + linear).
Extension $\Psi(x, t) = \lambda(t)\, g(x) = (t/T)\, g(x)$; this is the
`ConvexRawExtension` of the stage-1 catalogue.  Analytic derivatives:

$$
\partial_t \Psi = \frac{g(x)}{T},
\qquad
\partial_x \Psi = -\frac{t}{T} \sum_{k=1}^{K_g} \frac{\sin(k x)}{\pi^2 k},
\qquad
\partial_{xx} \Psi = -\frac{t}{T\,\pi^2} \sum_{k=1}^{K_g} \cos(k x).
$$

Schema mapping: `form="hard_convex"`, `interpolation="linear"`,
`extension=None` (the datum path of `TerminalAnsatz.extension`).  The
analytic-derivative bypass is **not** applied to this variant (it remains on
the existing autograd route, both for backward compatibility and because its
extension is the cheap product $\lambda(t) g(x)$).

**V2 `constant_in_time`** — existing baseline (`hard_constant` + linear,
default datum extension).  Extension $\Psi(x, t) = g(x)$; the
`ConstantInTimeExtension` of the stage-1 catalogue.  $\partial_t \Psi = 0$;
spatial derivatives as for V1 without the $t/T$ factor.  Schema mapping:
`form="hard_constant"`, `interpolation="linear"`, `extension=None`.

**V3 `split_diffusion`** — split semigroup extension, subset
$\{\partial_{xx}\}$, defect $B = \mu\,\partial_x + r_0$
(defect order $1 \le \rho = 1$).  With $\nu_{\mathrm{eff}} = \nu$:

$$
h(x, t) = \sum_{k=1}^{K_g} \frac{E_k(t)}{\pi^2 k^2} \cos(k x),
\qquad
\partial_t h = \frac{\nu}{\pi^2} \sum_{k=1}^{K_g} E_k(t) \cos(k x),
$$

$$
\partial_x h = -\sum_{k=1}^{K_g} \frac{E_k(t)}{\pi^2 k} \sin(k x),
\qquad
\partial_{xx} h = -\frac{1}{\pi^2} \sum_{k=1}^{K_g} E_k(t) \cos(k x).
$$

The forcing satisfies the split identity $P h = B h = \mu\,\partial_x h +
r_0 h$ (stage-1 P1).  Schema mapping: `form="hard_constant"`,
`interpolation="linear"`, `extension="split_diffusion"`.

**V4 `split_diffusion_advection`** — split semigroup extension, subset
$\{\partial_{xx}, \partial_x\}$, defect $B = r_0$ (defect order $0$).
With $\nu_{\mathrm{eff}} = \nu$:

$$
h(x, t) = \sum_{k=1}^{K_g} \frac{E_k(t)}{\pi^2 k^2} \cos\theta_k(x, t),
$$

$$
\partial_t h = \sum_{k=1}^{K_g} \frac{E_k(t)}{\pi^2 k^2}
\bigl[ \nu k^2 \cos\theta_k(x, t) + \mu k \sin\theta_k(x, t) \bigr],
$$

$$
\partial_x h = -\sum_{k=1}^{K_g} \frac{E_k(t)}{\pi^2 k} \sin\theta_k(x, t),
\qquad
\partial_{xx} h = -\frac{1}{\pi^2} \sum_{k=1}^{K_g} E_k(t) \cos\theta_k(x, t).
$$

The forcing is $P h = r_0\, h$.  Schema mapping: `form="hard_constant"`,
`interpolation="linear"`, `extension="split_diffusion_advection"`.

**V5 `graded_gaussian_matched`** — graded Gaussian extension with
comparison diffusivity $\nu_c = \nu$.  Mathematically identical to V3
(stage-1 measured coincidence: maximal absolute spectra difference
$1.7 \times 10^{-18}$ on G1, $1.4 \times 10^{-20}$ on G2).  It is retained
as a **plumbing-consistency control**: its extension is evaluated through
the graded code path (explicit $\nu_c$) rather than the subset-symbol path,
and the two trained runs must agree within seed-level scatter (Section 6,
risk R3).  Closed forms: those of V3 with $\nu_{\mathrm{eff}} = \nu_c = \nu$
and no phase advection.  Schema mapping: `form="hard_constant"`,
`interpolation="linear"`, `extension="graded_gaussian"`,
`comparison_diffusivity_ratio=1.0`.

**V6 `graded_gaussian_mismatched`** — graded Gaussian extension with
$\nu_c = \nu / 2$ (`comparison_diffusivity_ratio=0.5`).  Closed forms:
those of V3 with $\nu_{\mathrm{eff}} = \nu_c$ and no phase advection; the
forcing is $P h = \partial_t h + \nu\,\partial_{xx} h + \mu\,\partial_x h +
r_0 h$ with per-wavenumber coefficient $(a(k) + \nu_c k^2)\,\hat h(k, t)$.
Schema mapping: `form="hard_constant"`, `interpolation="linear"`,
`extension="graded_gaussian"`, `comparison_diffusivity_ratio=0.5`.

**V7 `exact_solution`** — the exact solution as extension (zero-forcing
control).  With $F_k(t) = e^{-(T - t)(\nu k^2 - r_0)}$:

$$
h(x, t) = u^\star(x, t)
= \sum_{k=1}^{K_g} \frac{F_k(t)}{\pi^2 k^2} \cos\theta_k(x, t),
$$

$$
\partial_t h = \sum_{k=1}^{K_g} \frac{F_k(t)}{\pi^2 k^2}
\bigl[ (\nu k^2 - r_0) \cos\theta_k(x, t) + \mu k \sin\theta_k(x, t) \bigr],
$$

with $\partial_x h$ and $\partial_{xx} h$ as in V4 with $E_k$ replaced by
$F_k$.  The forcing vanishes identically (per spectral component:
$(\nu k^2 - r_0)\cos\theta_k + \mu k \sin\theta_k - \nu k^2 \cos\theta_k -
\mu k \sin\theta_k + r_0 \cos\theta_k = 0$).  The trained loss then measures
the optimiser-noise floor of the pipeline: the residual reduces to
$R_\theta$ alone, whose exact minimiser is $\Phi_\theta \equiv 0$.  Schema
mapping: `form="hard_constant"`, `interpolation="linear"`,
`extension="exact_solution"`.

Soft forms (`soft_pinn`, `pure_nn`) are **excluded** from stage 2: the
stage-2 question concerns the $\theta$-independent forcing of hard
extensions, on which the soft forms are silent.  Decision recorded in
Section 6 (D2).

### 1.3 Variants of the control cell (2)

The single-spectral-component sine cell instantiates Proposition 1 of the
methodology report (matched exponential factor, zero forcing).  Datum
$g(x) = \sin x$ (one spectral component, $k_0 = 1$); pure-heat generator
$A = \nu\,\partial_{xx}$ with $\nu = 0.125$; exact solution
$u^\star(x, t) = e^{-\nu (T - t)} \sin x$.

**C1 `matched_exponential_factor`** — `form="hard_convex"`,
`interpolation="exponential"` with the rate **explicitly** set to
$\gamma = \nu k_0^2 = 0.125$ (catalogue field `exponential_rate_gamma:
0.125`).  The extension is $\Psi(x, t) = \lambda(t)\, g(x) =
e^{-\nu (T - t)} \sin x = u^\star(x, t)$, so the forcing vanishes
identically.  Caution for the implementer: the default rate of
`make_interpolation_coefficient` is the eigenvalue-matched value of the
**unit-interval** family, $\sigma^2 \pi^2 / 2$; on the circle the matched
rate for $k_0 = 1$ is $\nu k_0^2$, so `gamma` must be passed explicitly —
relying on the default silently mismatches the factor by $\pi^2$.

**C2 `convex_raw`** — `form="hard_convex"`, `interpolation="linear"`,
`extension=None`: the non-zero-forcing contrast within the cell, with
forcing $P\Psi = \bigl(\tfrac{1}{T} - \tfrac{t}{T}\nu\bigr)\sin x$.

### 1.4 Schema extension (exact statement of what is added)

The recon finding stands: the variant schema is string-valued, extensions
enter as callables only through the problem side, and no slot exists for
analytic derivatives.  The following extensions are specified.

1. **Variant schema** (torch-free catalogue): each entry of
   `METHOD_VARIANTS` in the new
   `experiments/python_scripts/exp_split_extension_trained/_split_extension_catalogue.py`
   has the fields
   `name: str`, `form: str` (one of the existing `FORMS`),
   `interpolation: str | None`, `extension: str | None` (a **registry key**,
   resolved to torch callables at build time inside the runner — the
   `call_cm` pattern generalised), `comparison_diffusivity_ratio: float |
   None` (graded variants only), `exponential_rate_gamma: float | None`
   (control cell only), `color: str`, `label: str`.  The catalogue remains
   importable without torch.

2. **Torch extension fields** (new library module
   `learning_option_pricing/pde/periodic_extension_fields.py`): a class
   `PeriodicExtensionField` built from the generator coefficients
   $(\nu, \mu, r_0)$, the real datum coefficients $(c_k)_{k=1}^{K_g}$ (or
   $k_0$ and amplitude for the sine cell), the extension kind, the
   comparison diffusivity, and $T$.  It exposes the torch callables
   `field(x, t)`, `time_derivative(x, t)`, `space_derivative(x, t)`,
   `second_space_derivative(x, t)` implementing the closed forms of
   Sections 1.2–1.3 as vectorised finite sums (coefficient tensors
   precomputed at construction, moved to the training device), plus
   `terminal_forcing_profile(x)` (the profile $(P h)(\cdot, T)$, Section
   3.5) and a module-level `exact_solution_field(...)` for $u^\star$.  The
   registry `EXTENSION_FIELD_REGISTRY` maps the schema keys
   `"split_diffusion"`, `"split_diffusion_advection"`, `"graded_gaussian"`,
   `"exact_solution"` to constructors.  Numerical policy: `float64`
   coefficient synthesis, cast to the training dtype at the end; the
   terminal identity `field(x, T) == datum(x)` must hold exactly in
   floating point (the decay factor at $t = T$ is $e^0 = 1$ exactly, and
   the summation order is fixed), and is asserted by a unit test.

3. **Analytic-derivative bypass** (`learning_option_pricing/models/terminal_ansatz.py`):
   `TerminalAnsatz.__init__` gains the keyword
   `extension_derivative_fns: dict[str, Callable[[Tensor, Tensor], Tensor]] | None = None`
   with the exact keys `"dt"`, `"dx"`, `"dxx"`; supplying it with any form
   other than `"hard_constant"` raises `ValueError` (the `hard_convex`
   product rule with $\lambda$ is not implemented — not needed by any
   stage-2 variant).  `residual_decomposition` gains the keyword-only
   parameter `generator_coefficients: dict[int, float] | None = None`
   (mapping differential order to coefficient, `{2: nu, 1: mu, 0: r0}`);
   exactly one of `sigma` and `generator_coefficients` must be supplied,
   and `sigma` alone reproduces the current behaviour through
   `{2: 0.5 * sigma**2, 1: 0.0, 0: 0.0}` (backward compatible — all
   existing call sites unchanged).  When the derivative callables are
   present, the forcing is assembled **analytically and outside the
   autograd graph** (`torch.no_grad()`), as
   $P\Psi = \partial_t \Psi + \nu\,\partial_{xx}\Psi + \mu\,\partial_x\Psi
   + r_0 \Psi$ — this is the derivative bypass: $P\Psi$ is
   $\theta$-independent, so the loss gradient needs its values, never its
   graph.  When absent, the existing autograd route through
   `constant_coefficient_operator_parts` is used.  The returned channel
   dictionary gains `forcing_advection`
   ($\mathbb{E}[(\mu\,\partial_x \Psi)^2]$) and `forcing_reaction`
   ($\mathbb{E}[(r_0 \Psi)^2]$) beside the existing `forcing_velocity` and
   `forcing_diffusion`.

4. **Operator generalisation** (`learning_option_pricing/pde/operators.py`):
   new functions `constant_coefficient_operator(field, coord, t,
   coefficients)` and `constant_coefficient_operator_parts(...)` (returning
   the dict of channels `velocity`, `diffusion`, `advection`, `reaction`),
   with `heat_operator` re-expressed as the special case `{2: 0.5 *
   sigma**2}`.  Autograd with `create_graph=True` and `allow_unused=True`
   as today.

5. **Startup cross-check** (runner obligation): on the first interior
   batch, the analytic $P\Psi$ and the autograd $P\Psi$ are both evaluated
   and the relative $L^2$ deviation is logged; a deviation above $10^{-3}$
   (float32 double-backward noise allowance) aborts the run.  This is the
   guard that the closed forms of Section 1.2 are implemented correctly on
   the training device.

---

## 2. Network, sampler, loss

* **Network**: the reference ResNet backbone (`learning_option_pricing/models/resnet.py`)
  with `net_width=64`, `net_blocks=4`, `net_layers_per_block=2`, but
  `d_in=3`: the `normalizer` slot of `TerminalAnsatz` is used as a periodic
  feature map $(x, t) \mapsto (\cos x, \sin x, 2t/T - 1)$, so periodicity of
  $\Phi_\theta$ holds by construction.  Predicted parameter count: $33601$
  (the reference $33537$ plus $64$ input weights) — a prediction, to be
  confirmed against the run log's measured count.
* **Sampler**: uniform Monte-Carlo, resampled every iteration from a CPU
  generator seeded with the sampler seed: interior `n_interior=4096` points
  with $x \sim \mathcal{U}(0, 2\pi)$, $t \sim \mathcal{U}(0, T)$; terminal
  `n_terminal=1024` at $t = T$ (diagnostic only for hard forms);
  `n_boundary=0` — the periodic embedding makes the lateral identification
  exact, so the boundary sampler and the boundary-drift diagnostic are
  removed (decision D4).
* **Loss**: hard forms only, $\mathbb{E}[(P\hat u)^2]$ over the interior
  batch, computed through `residual_decomposition` with
  `generator_coefficients` of the cell; no terminal or boundary term.
* **Optimiser and schedule**: Adam, learning rate $10^{-3}$;
  `CosineAnnealingLR` with `T_max=num_iterations`; gradient-norm probe
  `clip_grad_norm_(max_norm=1e12)` retained as in the reference (a probe,
  not a clip); best-state tracking with CPU deep copy and end-of-training
  restore, exactly as `ablation_ansatz_forms.py` lines 395–437.  No
  mid-training checkpointing (decision D6: runs last minutes; an
  interruption costs one run).

---

## 3. Observables and stage-1 comparison quantities (specification item 2)

### 3.1 Stage-1 measured anchors

The trained runs are compared against the following stage-1 measurements
(read off the production artefacts; provenance:
`data/measure_forcing_spectra/2026-07-10-18-51-41-123149Z_rho1_specK4096_stripK1048576/summary.yaml`
and
`data/measure_spectral_gap/2026-07-11-02-04-03-743828Z_rho1_log2K20/summary.yaml`,
Jean Zay prepost jobs 1715125 and 1721863).

Squared strip forcing $2\pi \sum_{0<|k|\le K} I_k$ at band edge $K = 2^{20}$,
$\rho = 1$ untruncated datum:

| Extension | G1 | G2 |
|---|---|---|
| Convex raw | $5.523662 \times 10^{3}$ | $1.761639 \times 10^{2}$ |
| Constant-in-time | $1.657104 \times 10^{4}$ | $5.284100 \times 10^{2}$ |
| Split $\{\partial_{xx}\}$ | $3.536834 \times 10^{-2}$ | $3.520249 \times 10^{-4}$ |
| Split $\{\partial_{xx}, \partial_x\}$ | $2.840672 \times 10^{-3}$ | $2.701137 \times 10^{-5}$ |
| Graded matched | $3.536834 \times 10^{-2}$ | $3.520249 \times 10^{-4}$ |
| Graded mismatched | $6.304372 \times 10^{-2}$ | $1.116074 \times 10^{-3}$ |
| Exact solution | $0$ | $0$ |

Flat (divergent) forms grow by a measured factor $15.96$–$16.00$ per
$16$-fold band increase; the G2 convex-to-split ratio at $2^{20}$ is a
factor $\approx 5.0 \times 10^{5}$.  P5 fitted tail exponents (fit window
$[2^{14}, 2^{17}]$, 14/14 agreements within $0.1$): $-3.0008$ for the split
$\{\partial_{xx}\}$ and the matched graded extension (predicted $-3$),
$-5.0000$ for the split $\{\partial_{xx}, \partial_x\}$ (predicted $-5$),
$-1.0557$ for the mismatched graded extension (predicted $-1$), $-0.0557$
for the two flat forms (band-limited reference-curve slope).  Bandwidth
demand $K_\star(\gamma)$ on G2: split $\{\partial_{xx}\}$ —
$8, 8, 16, 128$ at $\gamma = 10^{-1}, 10^{-2}, 10^{-3}, 10^{-6}$; split
$\{\partial_{xx}, \partial_x\}$ — $8, 8, 8, 16$; graded mismatched —
$16, 128, 1024, 524288$; flat forms — not reached within the band at any
$\gamma$.  Mismatch power ratio: predicted $0.25$, measured
$0.2500000000005$ (G2).

**Band-edge caution.**  The stage-1 anchors above are evaluated at
$K = 2^{20}$ on the *untruncated* datum.  The trained datum is truncated at
$K_g = 128$, so every closed-form quantity used for a *trained-run
comparison* (floor values, unreachable fractions, per-wavenumber integrals
$I_k$) is **recomputed at band edge $K_g$** by the aggregation script,
through the same library calls (`squared_forcing_time_integral`,
`total_strip_forcing_squared`).  The $2^{20}$ values are quoted only as the
stage-1 reference; for the flat forms the $K_g$ values are smaller by
construction (linear growth in the band edge), for the convergent forms
they are unchanged to within their already-converged tails.

Reachable band: **no saved measurement of a network cutoff exists** (the
`spectral_bias_periodic_nn` run directory holds a figure only, and its
network differs from the stage-2 backbone).  The cutoff $k_\star$ is
therefore measured *within* stage 2, per trained run (Section 3.4); every
slot below that references $k_\star$ remains empty until that measurement
exists.

### 3.2 Exact solution for the error metrics

For the generator cells, the exact solution is the finite spectral-component
sum

$$
u^\star(x, t)
= \sum_{k=1}^{K_g} \frac{1}{\pi^2 k^2}\,
  e^{-(T - t)(\nu k^2 - r_0)}\,
  \cos\bigl(k x + (T - t)\, \mu k\bigr),
$$

and for the control cell $u^\star(x, t) = e^{-\nu(T - t)} \sin x$.  All
error metrics are evaluated on the tensor grid
$G_{\mathrm{eval}}$ of $n_x = 1024$ uniform points on $[0, 2\pi)$ times the
eleven time slices $t \in \{0, 0.1T, \ldots, T\}$, in `float64` for the
reference side.

### 3.3 Per-run trained observables

Each `variant_<name>/` directory saves `models/model.pt`, `hist.npz`,
`metrics.npz`, `slices.npz`, `spectra.npz`; each array task writes
`summary_<name>.yaml`.  History channels: `iter, loss, loss_pde, loss_tc,
network_energy, cross_term, forcing_floor, forcing_velocity,
forcing_diffusion, forcing_advection, forcing_reaction, grad_norm, lr`.
Metrics:

1. **Final residual**: best (restored) training loss
   $\mathbb{E}[(P\hat u)^2]$ with its iteration index; also the
   `network_energy` at the best state.
2. **Solution error**: relative $L^2$ error of $\hat u$ against $u^\star$
   over $G_{\mathrm{eval}}$ (strip error), and restricted to the $t = 0$
   slice (`rel_l2_t0`).
3. **Error near the corner**: relative $L^2$ error restricted to the corner
   window $W = \{ x \in [0, 2\pi) : \operatorname{dist}(x, x^\star) \le
   \pi/16 \}$ with $x^\star = 0$ and distance taken on the circle (the
   window spans $\pm 4$ periods of the highest retained component); reported
   at $t = 0$ and as the maximum over the eleven slices.
4. **Terminal-condition check**: `tc_l2`, the $L^2$ mismatch of
   $\hat u(\cdot, T)$ against $g$ — exactly zero for every hard form by
   construction, asserted.
5. **Forcing-floor consistency**: the median over training of the
   `forcing_floor` channel, against the closed-form value
   $\tfrac{1}{T} \sum_{0<|k|\le K_g} I_k$ (the Monte-Carlo expectation over
   the uniform sampling measure equals the squared strip norm divided by
   $2\pi T$; with the Parseval prefactor,
   $\mathbb{E}[(P\Psi)^2] = \|Lh\|^2_{\mathrm{strip}} / (2\pi T)$).  Both
   numbers are recorded; agreement within Monte-Carlo scatter (a few per
   cent) validates the plumbing.

### 3.4 Residual frequency decomposition (reused method)

The measurement reuses the three-step recipe of
`spectral_bias_periodic_nn.py`: (i) evaluate the residual field
$r(x, t_s) = (P\hat u)(x, t_s)$ by autograd on the uniform $1024$-point grid
at the five time slices $t_s / T \in \{0.1, 0.3, 0.5, 0.7, 0.9\}$; (ii) real
FFT with mean removal, normalised so that the bin at wavenumber $k$ estimates
the Fourier coefficient of the sampled field, powers averaged over the
slices; (iii) ratio against the forcing power — with one stage-2 upgrade
taken from the library policy: the forcing side is **not** an FFT of samples
but the exact per-wavenumber coefficient
$\widehat{Lh}(k, t_s)$ from `TerminalDataExtension.forcing_coefficient`,
averaged in power over the same slices.  The per-spectral-component
cancellation ratio is $|\hat r_k|^2 / |\widehat{Lh}(k)|^2$; the in-band mask
is (forcing power) $> 10^{-5} \times$ its maximum; the estimated reachable
cutoff $k_\star$ is the first in-band wavenumber at which the $7$-point
running mean of the ratio reaches $1/2$.  All arrays (residual power per
slice, exact forcing power, ratio, running mean, mask, $k_\star$) are saved
in `spectra.npz`.  For the two zero-forcing variants (V7, C1) the ratio is
undefined; the absolute residual power spectrum is saved instead and
$k_\star$ is recorded as absent.

### 3.5 Terminal-target check

At $t = T$ the network prefactor vanishes and the residual reduces to
$-\lambda'(T)\,\Phi_\theta(\cdot, T) + (P\Psi)(\cdot, T)$, so the exact
minimiser's terminal profile is
$\Phi^\star(\cdot, T) = -\,(P\Psi)(\cdot, T) / d_T'(T) = T\,(P\Psi)(\cdot, T)$
for the linear factor.  Closed forms per variant (generator cells):

* `split_diffusion`:
  $\Phi^\star(x, T) = T \sum_{k=1}^{K_g} \frac{1}{\pi^2 k^2}
  \bigl[ r_0 \cos(k x) - \mu k \sin(k x) \bigr]$
  (the profile $T\,(Bg)(x)$ with $B = \mu\,\partial_x + r_0$ — one
  derivative order smoother than the datum, per the stage-1 note Section 6);
* `split_diffusion_advection`: $\Phi^\star(x, T) = T\, r_0\, g(x)$;
* `constant_in_time`: $\Phi^\star(x, T) = T \sum_{k=1}^{K_g}
  \frac{1}{\pi^2 k^2} \bigl[ (r_0 - \nu k^2) \cos(k x) - \mu k \sin(k x)
  \bigr] = T\,(Ag)(x)$;
* `graded_gaussian` ($\nu_c$): $\Phi^\star(x, T) = T \sum_{k=1}^{K_g}
  \frac{1}{\pi^2 k^2} \bigl[ (r_0 - (\nu - \nu_c) k^2) \cos(k x) -
  \mu k \sin(k x) \bigr]$;
* `exact_solution` and `matched_exponential_factor`:
  $\Phi^\star(\cdot, T) = 0$;
* `convex_raw` (`hard_convex`): $\Phi^\star(x, T) = g(x) + T\,(Ag)(x)$.

The observable is the relative $L^2$ distance of the trained
$\Phi_\theta(\cdot, T)$ (the bare network, `free_network`, on the
$1024$-point grid) to $\Phi^\star(\cdot, T)$; for the zero-target variants
the absolute $L^2$ norm is reported instead.  The profile is supplied by
`PeriodicExtensionField.terminal_forcing_profile`.

### 3.6 Stage-2 hypotheses (predictions, to be confronted with measurement)

Stated as hypotheses; none is a proven statement, and no measured slot is
filled before the runs exist.

* **H1 (unreachable-mass floor)**: the best trained loss of each variant is
  bounded below by, and approximately attains, the unreachable forcing mass
  at the measured cutoff,
  $\mathcal{F}(k_\star) = \tfrac{1}{T} \sum_{|k| > k_\star,\ |k| \le K_g}
  I_k$, with $I_k$ the closed-form per-wavenumber time integrals at the
  cell's generator.  Consequently the ratio of best losses between
  `convex_raw` and `split_diffusion` is predicted to be of the order of the
  ratio of their unreachable masses at the common $k_\star$.
* **H2 (cutoff invariance)**: $k_\star$ is a property of the network and
  optimisation, not of the extension: the measured $k_\star$ agrees across
  variants of a cell within the dyadic resolution of the running-mean
  estimator.
* **H3 (terminal-target attainment)**: the trained $\Phi_\theta(\cdot, T)$
  approaches the profile of Section 3.5, with the split variants' target
  one derivative order smoother than the convex baselines' target.
* **H4 (controls)**: the zero-forcing variants (V7, C1) train to a loss at
  the optimiser-noise level, orders of magnitude below every non-zero-floor
  variant in the same cell; V5 and V3 agree within seed-level scatter.

---

## 4. Budget, seeds, folders (specification item 3)

Pinned to the `ansatz_forms_cross_seed_summary` reference
(`_ansatz_forms_catalogue.py::DEFAULT_HPARAMS` and the per-run
`metadata.yaml` of the 2026-06-09 runs):

* `num_iterations = 20000`; Adam `learning_rate = 1e-3`; cosine schedule.
* Network `net_width = 64`, `net_blocks = 4`, `net_layers_per_block = 2`
  (with `d_in = 3` per Section 2).
* Sampler `n_interior = 4096`, `n_terminal = 1024`, `n_boundary = 0`
  (deviation from the reference value $256$ recorded as decision D4).
* Smoke guard `SMOKE_TEST_NUM_ITERATIONS_THRESHOLD = 1000` with
  `SystemExit` below it without `--debug`.

Seeds: master seed from `--seed`, per-role seeds via the existing
`derive_seed(master, role)` (blake2b, roles `"model_init"` and
`"sampler"`).  **Shared-seed policy**: all variants of a cell at a given
master seed receive identical initial weights and sampler trajectories.
The explicit seed axis is $\{0, 1, 2\}$, layered as separate run
directories.  Folder-name convention (seed suffix mandatory):

```
data/ablation_split_extension_trained/<debug_prefix><UTC-timestamp>Z_<cell>_iters<N>_seed<S>/
```

with per-variant subdirectories `variant_<name>/` and per-task
`summary_<name>.yaml`, exactly as the reference layout.  The cross-seed
aggregator's discovery regex must accept the suffix:
`Z_(?P<cell>[a-z0-9_]+)_iters\d+(?:_seed\d+)?$`, `_debug_` directories
excluded.

Run-log contract: full command line, Python/torch/CUDA versions and device
name/memory, every hyperparameter after YAML translation, master and derived
seeds, measured parameter count, per-iteration cadence as in the reference
(every iteration up to 100, then `num_iterations // 50`), final wall-clock,
seconds per iteration, best loss and its iteration.

---

## 5. Bermudan extension — block 2 (specification item 4; SPEC ONLY)

Block 2 is implemented **after** block 1 and reuses its extension registry.
Decision D7 (Section 6) resolves the setting: block 2 remains on the circle
with the G2 generator, so that every exact reference stays a finite (or
closed-form-coefficient) spectral-component sum; the "reference settings"
adopted are the training hyperparameters of Section 4 and the two-date
geometry of the real-line Bermudan reference (`bermudan_backward_induction.py`,
$m = 2$).

**Geometry.**  Exercise dates $t_1 = T/2$ and $t_2 = T = 1$ (one
intermediate date; an $m = 3$ cell with dates $\{T/3, 2T/3\}$ is contingent
on the $m = 2$ outcome).  The exercise payoff on the circle is the datum
itself, $p = g$ (band-limited, $K_g = 128$) — the Bermudan-analogue in which
the same payoff is compared at every date.

**Exact reference.**  Upper stage on $[t_1, T]$: $u^\star$ of Section 3.2.
Glued datum at $t_1$: $g_1 = \max(p, u^\star(\cdot, t_1))$ — a piecewise
trigonometric polynomial whose crossing set
$X^\star = \{ x \in [0, 2\pi) : p(x) = u^\star(x, t_1) \}$ is located by
dense-grid sign-change bracketing plus bisection to absolute tolerance
$10^{-12}$; the Fourier coefficients of $g_1$ are then **closed-form
piecewise integrals** (each piece is a finite cosine sum; the integrals
$\int_{x_i}^{x_{i+1}} \cos(m x + \varphi)\, e^{-i k x}\, dx$ are
elementary).  The reference band is truncated at
$K_{\mathrm{ref}} = 4096$; since $g_1$ is continuous and piecewise smooth
with first-derivative discontinuities at $X^\star$, its coefficients satisfy
$|c_k(g_1)| = O(k^{-2})$ as $|k| \to \infty$, and the truncation is
validated by recomputing at $K_{\mathrm{ref}}/2$ and reporting the relative
difference of every derived quantity.  Lower stage on $[0, t_1]$: exact
component sum driven by $c_k(g_1)$.

**Trained pipeline.**  Per extension variant $E \in
\{\texttt{split\_diffusion}\ (\text{correctly specified}),\
\texttt{graded\_gaussian\_mismatched}\ (\text{mis-specified})\}$: stage A
trains on $[t_1, T]$ (local time, terminal datum $g$) with extension $E$;
the frozen continuation is $\hat C_1(x) = \hat u_A(x, t_1)$.  The lower
stage's datum is $\hat g_1 = \max(p, \hat C_1 + \delta)$ with the injected
perturbation $\delta$ (below).  Because $\hat C_1$ has no closed-form
coefficients, the stage-B extension is built from **projected**
coefficients: trapezoidal projection of $\hat g_1$ onto
$\{e^{ikx}\}_{|k| \le K_{\mathrm{stage}}}$, $K_{\mathrm{stage}} = 256$, on
an $8192$-point uniform grid, with the projection error quantified by
re-projection on a $16384$-point grid (this is datum *construction*, not a
measurement; the stage-1 exact-coefficient policy is not violated).  Stage
B then trains on $[0, t_1]$ with extension $E$ applied to the projected
coefficients.

**Perturbation injection** (the `bermudan_perturbation_propagation.py`
pattern transplanted to the circle): the perturbation is added to the
continuation *before* the max-gluing,
$\delta(x) = \varepsilon_0 \cos\bigl(k_{\mathrm{inj}} (x - x_0)\bigr)$ with
amplitude $\varepsilon_0 = 10^{-3}$ (linear regime, as in the reference
script), centre $x_0$ the first element of $X^\star$, and
$k_{\mathrm{inj}} \in \{1, 2, 4, 8, 16, 32\}$.  Since $\delta$ is a single
spectral component, the perturbed glued datum remains piecewise
trigonometric and the **exact** perturbed induction is computable by the
same piecewise-coefficient machinery — giving the no-learning reference
gain.

**Block-2 observables.**

1. Inception gain
   $\|\hat u_{B,\delta}(\cdot, 0) - \hat u_{B,0}(\cdot, 0)\|_{L^2} /
   \|\delta\|_{L^2}$ versus $k_{\mathrm{inj}}$, overlaid on (i) the exact
   induction gain and (ii) the semigroup damping envelope
   $e^{-t_1 (\nu k_{\mathrm{inj}}^2 - r_0)}$ (the exercise mask makes the
   envelope an upper bound, not an equality).  Hypothesis: under the
   correctly-specified extension the trained gain follows the exact gain;
   under the mis-specified extension the additional forcing floor
   contributes a $k_{\mathrm{inj}}$-independent error term that dominates
   once the damped exact gain falls below it.
2. Greeks diagnostics: $\partial_x \hat u$ and $\partial_{xx} \hat u$ by
   autograd at $t = 0$ and $t = t_1^-$, against the exact component sums
   $\partial_x u^\star_1, \partial_{xx} u^\star_1$ (differentiating the
   finite sums term by term); reported as relative $L^2$ over the full
   circle and over the corner windows
   $\{ \operatorname{dist}(x, x^\star_j) \le \pi/16 \}$ around each
   crossing $x^\star_j \in X^\star$ — the corner of block 2 **is** the
   gluing set $X^\star$, where the datum's first derivative jumps and the
   Gamma error is predicted to concentrate.  A figure overlays trained
   (solid) and exact (dashed) Gamma near each $x^\star_j$, with the
   crossing marked by a dotted vertical line.

**Block-2 run grid** (all at the Section-4 hyperparameters): stage A —
2 extensions $\times$ 3 seeds; stage B baseline (no injection) — 2 $\times$
3; stage B injected — 2 extensions $\times$ 6 wavenumbers at seed 0 only.
Total 24 trainings, predicted $\le 700$ s each (Section 6, wall-clock
basis), hence $\le 4.7$ V100-hours.  File: a new runner
`experiments/python_scripts/exp_split_extension_trained/bermudan_split_extension_circle.py`
following the same array contract, plus catalogue entries in the same
`_split_extension_catalogue.py`; artefacts mirror
`bermudan_backward_induction.py` (`--replot`, `--revalidate` from saved
models).

---

## 6. Execution plan (specification item 5)

### 6.1 Files created

| Path | Content |
|---|---|
| `learning_option_pricing/pde/periodic_extension_fields.py` | Torch extension fields, registry, exact-solution field, terminal forcing profiles (Section 1.4 item 2) |
| `test/pde/test_periodic_extension_fields.py` | Tests T1–T5 below |
| `experiments/python_scripts/exp_split_extension_trained/_split_extension_catalogue.py` | Torch-free cells + variants + `DEFAULT_HPARAMS` + `RUNNER_SCRIPT_STEM = "ablation_split_extension_trained"` (asserted by the runner) |
| `experiments/python_scripts/exp_split_extension_trained/ablation_split_extension_trained.py` | Runner: CLI `--cell`, `--variant`, `--seed`, `--num-iterations`, `--device`, `--debug`, `--replot DIR`, `--init-only`, `--ablation-dir`, `--config-dir`/`--config-name` — the exact array-worker contract of `ablation_ansatz_forms.py` |
| `experiments/python_scripts/exp_split_extension_trained/_split_extension_plots.py` | `replot`: merge `summary_*.yaml`, rebuild all figures from `npz` artefacts alone; figure list: `loss_components.png`, `loss_decomposition.png` (six channels incl. advection/reaction), `solution_t0.png`, `error_t0.png`, `terminal_network_target.png` ($\Phi_\theta(\cdot,T)$ solid vs $\Phi^\star(\cdot,T)$ dashed), `residual_spectra.png` (cancellation ratios, colour per variant, $k_\star$ marked dotted), `summary_metrics.png`; every figure with the formula textbox below the axes via `learning_option_pricing/utils/figure_layout.py`, legends outside the axes |
| `experiments/python_scripts/exp_split_extension_trained/split_extension_cross_seed_summary.py` | Cross-seed aggregation: discovery per Section 4 regex; recomputation of every closed-form comparison quantity at band edge $K_g$; outputs `summary_across_seeds.yaml`, `rel_l2_by_cell.png`, `floor_vs_accuracy.png` (best loss vs closed-form floor and vs unreachable mass $\mathcal{F}(k_\star)$ — H1), `cutoff_by_variant.png` (H2), `terminal_target.png` (H3) |
| `experiments/python_scripts/exp_split_extension_trained/bermudan_split_extension_circle.py` | Block 2 (after block 1) |

### 6.2 Files modified

| Path | Change |
|---|---|
| `learning_option_pricing/pde/operators.py` | `constant_coefficient_operator(_parts)`; `heat_operator` delegates |
| `learning_option_pricing/models/terminal_ansatz.py` | `extension_derivative_fns` keyword; `generator_coefficients` keyword in `residual_decomposition`; `forcing_advection`/`forcing_reaction` channels; backward compatible |
| `learning_option_pricing/pde/__init__.py` | Export the new operator functions and module |
| `test/pde/test_operators.py` (or new file) | Parts-sum identity; heat special case bitwise equality |
| `test/models/test_terminal_ansatz.py` (extend) | Analytic path equals autograd path within tolerance; `ValueError` on non-`hard_constant` derivative fns |
| `documents/methodology/spectral_forcing_floor_and_split_extensions.md` | After the runs: a stage-2 results section (not before) |
| `CONTRIBUTING.md` / `pyproject.toml` | No new dependency is introduced (numpy, torch, matplotlib, pyyaml already present); update only if the implementer adds an entry point |

Required library tests (all `float64`): **T1** terminal exactness —
`field(x, T)` equals the datum sum exactly in floating point; **T2**
analytic derivatives against `torch.autograd` on random points, relative
deviation $\le 10^{-10}$; **T3** real-space split identity — for V3/V4,
$P h - B h$ relative deviation $\le 10^{-12}$; **T4** exact-solution forcing
$\le 10^{-12}$ of the field scale; **T5** quadrature consistency — the
trapezoidal strip integral of $(P h)^2$ on a fine tensor grid against
$2\pi \sum_k I_k$ from `terminal_data_extensions`, relative deviation
$\le 10^{-6}$.

### 6.3 Smoke path

Commit and push first (`git push github master`; the cluster pulls from its
`origin`).  Smoke launch on Jean Zay via the existing launcher (flag names
per the launcher's usage header, lines 40–68 of
`bash_scripts/cluster/jeanzay/python/experiment_array_launcher.sh`):

* one cell, one seed, `--init-args "--cell g2_bernoulli_bandlimited --seed 0
  --num-iterations 300 --debug --init-only --device cpu"`;
* TRAIN override `--qos qos_gpu-dev --time 00:20:00` (dev QoS: fast
  scheduling, max 2 h, billed) on the default V100 partition `gpu_p13`,
  account `akz@v100`, 1 GPU per task;
* FINALIZE `--finalize-args "--replot {EXPDIR} --device cpu"` on
  `--partition=prepost` (non-billed), account `akz@cpu` — the launcher's
  defaults;
* acceptance: the startup cross-check of Section 1.4 item 5 passes and is
  logged; `tc_l2 = 0` for every variant; all figures and
  `summary_<variant>.yaml` present; then wipe with
  `find data -type d -name '_debug_*' -prune -exec rm -rf {} +`.

### 6.4 Production array

Nine launcher invocations — the loop over cell $\times$ seed:

```
for cell in g1_bernoulli_bandlimited g2_bernoulli_bandlimited heat_sine_single_component; do
  for s in 0 1 2; do
    bash bash_scripts/cluster/jeanzay/python/experiment_array_launcher.sh \
      <script path> \
      --init-args "--cell $cell --seed $s --init-only --device cpu" \
      --finalize-args "--replot {EXPDIR} --device cpu" \
      --time 01:00:00
  done
done
```

Array widths: 7 tasks for each generator cell, 2 for the control cell —
16 tasks per seed, 48 trainings total.  Partition/QoS routing (per the
routing table): TRAIN on `gpu_p13` (default V100), `qos_gpu-t3`,
`--account=akz@v100`, `--gres=gpu:1`, `--cpus-per-task=10`; FINALIZE on
`prepost` (non-billed), `--account=akz@cpu`, CPU only.  SLURM logs go to
`$EXPDIR/slurm/` (launcher default).  After all nine FINALIZE jobs
complete, the cross-seed aggregation runs once on `prepost`
(`sbatch --partition=prepost --account=akz@cpu --time=00:30:00` wrapping
`python split_extension_cross_seed_summary.py`).

Note on the task-to-variant mapping: the worker indexes an **unsorted**
`find` enumeration of the config YAMLs; task index is not variant identity.
Variant identity is always read from the config file name and content,
never from the array index.

### 6.5 Wall-clock estimate

Measured reference (V100, torch 2.6.0+cu124, 20000 iterations): 342.7 s to
370.0 s per variant on the `sine` cell, 458.3 s on the heaviest
(`bermudan_put`) cell.  Stage-2 deltas: `d_in=3` is negligible; the
$K_g = 128$-component extension sums are forward-only under the derivative
bypass ($\sim 4096 \times 128$ evaluations per iteration, negligible on
V100), and the bypass removes the double backward through the extension.
Predicted per-task time: 400–700 s (a prediction — the smoke run measures
it before production).  Budget `--time 01:00:00` per task ($\ge 5\times$
margin).  Predicted total: 48 tasks $\times \le 0.2$ h $\le 10$ V100-hours
billed; wall-clock per seed of the order of the queue latency plus
$\sim 12$ minutes.

---

## 7. Risks and resolved decisions (specification item 6)

* **D1 — reviewer directives.**  `documents/draft.md` contains working
  notes, not a structured directive list; the directives honoured are the
  derivative bypass and the exactness preference (its Zhang-et-al.
  discussion), and the stage-2 hypothesis sentence of the stage-1 note
  Section 5.5.  No other directive is inferred from the draft.
* **D2 — soft forms excluded.**  `soft_pinn` and `pure_nn` are omitted:
  the stage-2 axis is the extension forcing, which the soft forms do not
  possess.  The convex/constant hard baselines provide the comparison.
* **D3 — matched graded kept as a duplicate.**  V5 duplicates V3
  mathematically; it is retained as a consistency control of the graded
  code path.  Expectation: extension fields agree to $\le 10^{-6}$
  relative (asserted at build time), but trained trajectories may diverge
  through floating-point non-associativity over 20000 iterations; the
  acceptance criterion is agreement of final metrics within seed-level
  scatter, not bitwise identity.
* **D4 — no boundary sampler.**  The periodic feature map makes the
  lateral identification exact; `n_boundary = 0` deviates from the
  reference value 256, recorded here and in the catalogue comment.
* **D5 — architecture deviation.**  The reference network takes
  $(x, t) \in \mathbb{R}^2$; stage 2 requires periodicity, so the input is
  the three-dimensional periodic embedding and the parameter count rises
  from 33537 to a predicted 33601.  Cross-variant comparisons inside stage
  2 are unaffected (shared architecture and seeds); any comparison against
  stage-1 trained numbers (`exp_ansatz_forms_heat`) is cross-architecture
  and must be labelled as such in figures.
* **D6 — no mid-training checkpointing.**  Runs last minutes; the
  checkpoint/resume machinery is not added.  Best-state tracking and
  end-restore are kept.
* **D7 — block 2 stays on the circle.**  "Exact component sums" for the
  Greeks fixes the periodic setting; the real-line Bermudan machinery
  (`heat_propagate` quadrature) is not reused for block 2.  The real-line
  Gaussian-smoothed put (P1b) remains the bridge result for the eventual
  option-pricing transfer, outside the scope of this specification.
* **D8 — projected coefficients in block 2 are construction, not
  measurement.**  The trained continuation has no closed form; trapezoidal
  projection with a quantified re-projection error is the stated mechanism,
  and every block-2 *measurement* against a reference uses the closed-form
  piecewise coefficients of the exact induction.
* **D9 — $k_\star$ slots stay empty.**  No saved network-cutoff
  measurement exists; every quantity depending on $k_\star$ (unreachable
  fractions of H1, the H2 comparison) is computed only after the stage-2
  spectra are measured.  No placeholder value is written anywhere.
* **D10 — exponential rate on the circle.**  The control cell's matched
  rate is $\gamma = \nu k_0^2 = 0.125$, passed explicitly; the library
  default $\sigma^2 \pi^2 / 2$ belongs to the unit-interval family and
  would silently mismatch (Section 1.3).
* **R1 — float32 forcing evaluation.**  The closed-form sums are evaluated
  in the training dtype; the startup cross-check (Section 1.4 item 5)
  bounds the analytic-versus-autograd deviation at $10^{-3}$ and logs the
  measured value, so a dtype-induced discrepancy cannot pass silently.
* **R2 — spectral leakage in the residual FFT.**  The residual field is
  periodic by construction (periodic network, periodic extension), so no
  window is needed; mean removal handles the $k = 0$ bin.  The
  $1024$-point grid resolves wavenumbers to $512$, four times the datum
  band — aliasing from network content above $512$ is possible in
  principle and is checked by recomputing one variant's spectra on a
  $2048$-point grid at aggregation time.
