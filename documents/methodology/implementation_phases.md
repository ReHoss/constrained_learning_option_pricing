# Implementation Phases & Training Procedure

> Math rendering: open in a Markdown renderer with KaTeX/MathJax support.

This document serves as the implementation reference for the core components and the training procedure.

For neural network architecture and loss functions, see [`architecture.md`](architecture.md).
For reference numerical methods, see [`solvers_and_benchmarks.md`](solvers_and_benchmarks.md).

**Parameters used throughout Phase 1 & 3 (Section 4.1.2 of the paper):**

$$K = 100,\quad r = 0.02,\quad \sigma = 0.25,\quad T = 1,\quad q = 0$$

---

## Phase 1 — BSM Mathematical Components

This phase implements the core mathematical components introduced in the paper. The mathematical formulations for the Exact Terminal Functions ($g_1$, $g_2$) and the BSM PDE operator are detailed in [`architecture.md`](architecture.md).

### 1.1 Payoff functions

**Code:** `learning_option_pricing/pricing/terminal.py`

```python
payoff_put(s, K)   # -> torch.Tensor, same shape as s
payoff_call(s, K)  # -> torch.Tensor, same shape as s
```

### 1.2 BSM PDE & Time Value operators

**Implementation note:** derivatives are computed via `torch.autograd.grad` so that the operator is differentiable and can be embedded in the training loss.

**Code:**
- `terminal.bsm_operator(V, s, t, r, q, sigma)` — requires `s` and `t` to be leaf tensors with `requires_grad=True`.
- `terminal.time_value(V, s, K, option_type)`

### 1.3 European put price & Taylor expansion

**Singular limit:** as $\tau \to 0$, the log-moneyness terms diverge. A floor $\tau_{\varepsilon} = 10^{-8}$ is applied before all divisions: $\tau_{\text{safe}} = \max(\tau,\; 10^{-8})$.

**Code:**
```python
black_scholes_put(s, K, r, sigma, tau)   # V^e(s, t)
_d_tilde_1(s, tau, K, r, sigma)
_d_tilde_2(s, tau, K, r, sigma)
european_put_ve1(s, K, r, sigma, tau)   # V_1^e
european_put_ve2(s, K, r, sigma, tau)   # V_2^e
g2_american_put(s, K, r, sigma, tau)    # g_2 = V_1^e + V_2^e
```

### 1.4 Phase 1 scalar validation results

Run: `experiments/python_scripts/exp1/phase1_bsm_validation.py`

| Check | Value | Expected |
|-------|-------|----------|
| $\max_s \lvert V^e(s,T) - \Phi(s)\rvert$ | $0$ | $< 10^{-6}$ |
| $\max_s \lvert g_2(s,T) - \Phi(s)\rvert$ | $0$ | $< 10^{-6}$ |
| $V^e(K{=}100,\; t{=}0.5)$ | $6.5218$ | $\approx 6$–$7$ |
| $g_2(K{=}100,\; t{=}0.5)$ | $6.5310$ | close to above |
| NaN/Inf at $\tau = 10^{-8}$ | None | None |

---

## Phase 3 — Training and Validation

This section describes the training procedure for the ETCNN on two toy problems: a **European put** (analytical ground truth) and a **Bermudan put** with one intermediate exercise date (binomial tree ground truth).

### 3.0 European vs Bermudan — a unified view

The European option is the **zero-exercise-date special case** of the Bermudan option:

| Option type | Exercise dates | Sub-intervals | Terminal condition source |
|-------------|---------------|--------------|--------------------------|
| European | none (only $T$) | $[0, T]$ (one block) | Analytical payoff $\Phi(s) = (K-s)^+$ |
| Bermudan $(m=1)$ | $\{t_1, T\}$ | $[t_1, T]$ then $[0, t_1]$ | Analytical at $T$; interpolated $V^{\mathrm{Berm}}_{\bar\theta}(s,t_1)$ at $t_1$ |
| Bermudan $(m)$ | $\{t_1,\ldots,t_m,T\}$ | $m+1$ sub-intervals | Backward induction, one stage per sub-interval |

On each sub-interval the option satisfies the **European BSM equation** (no continuous free boundary), so the loss function is always the European-type loss $\lambda_f \mathcal{L}_f + \lambda_{tc} \mathcal{L}_{tc}$. The Bermudan early-exercise condition enters only at the discrete exercise dates when constructing the terminal condition for the next backward stage.

This means the European problem code (`european_problem`) exercises the same ETCNN / loss stack as a single Stage A of the Bermudan code, and serves as the primary validation baseline before adding the inter-stage coupling.

### 3.1 Optimiser and learning rate schedule

- **Optimiser:** Adam with $\beta_1 = 0.9$, $\beta_2 = 0.999$
- **Initial learning rate:** $\mathrm{lr}_0 = 0.01$
- **Two-stage exponential decay** (Section 3.4):
  - First 10,000 iterations: decay by $\gamma = 0.85$ every 2,000 steps
  - Remaining iterations: decay by $\gamma = 0.85$ every 5,000 steps
- **Total iterations:** 50,000

Code: `phase3_training.py` — `build_lr_lambda`.

### 3.2 Problem 1 — European put

The European put has an analytical solution $V^e(s, t)$ (Black–Scholes, Eq. 16), enabling exact error measurement.

Both ETCNN and PINN are trained on the same domain with the same hyperparameters. The PINN uses a standard output layer (no $g_1/g_2$ modification), so it must learn the terminal condition from the $\mathcal{L}_{tc}$ penalty alone.

**Sampling:** $N_{tc} = 1024$ terminal points, $N_f = 4 N_{tc} = 4096$ interior points, resampled uniformly each iteration.
**Training domain:** $s \in [20, 160]$, $t \in [0, T]$.  
**Evaluation domain:** $s \in [60, 120]$, $t \in [0, T]$.

**Expected results:**
- ETCNN's $\mathcal{L}_{tc}$ should be at or near machine zero throughout training (enforced by architecture).
- ETCNN relative $L^2$ error should be $< 5 \times 10^{-4}$.
- PINN relative $L^2$ error should be roughly one order of magnitude larger.
- Largest pointwise errors concentrate near the kink at $(s, t) = (K, T)$.

**Plots produced:**
| Plot | File | Description |
|------|------|-------------|
| E1 | `training_metrics/plotE1_loss_curves.png` | Loss curves ($\mathcal{L}_f$, $\mathcal{L}_{tc}$) for ETCNN and PINN vs iteration |
| E2 | `pricing/plotE2_surface_comparison.png` | Predicted vs analytical price surface heatmaps over $s \in [60,120]$, $t \in [0,T]$ |
| E3 | `diagnostics/plotE3_errors.png` | Side-by-side absolute error heatmaps: ETCNN $\lvert \tilde{u}_\theta - V^e\rvert$ and PINN $\lvert u_\theta - V^e\rvert$ |
| E4 | `pricing/plotE5_slices.png` | Slice comparison $\tilde{u}(s, t_k)$ vs $V^e(s, t_k)$ at $t_k \in \{0.25, 0.5, 0.75\}$ |
| E5 | `greeks/plotE6_greeks.png` | Greeks $\Delta, \Gamma, \Theta$ at $t = 0$ via autograd vs analytical Black–Scholes |

**Plot E1 — Loss curves:**
Tracks the two training objectives per iteration for both models:
$$\mathcal{L}_f = \frac{1}{N_f}\sum_i \lvert \mathcal{F}[\tilde{u}](s_i, t_i)\rvert^2, \qquad \mathcal{L}_{tc} = \frac{1}{N_{tc}}\sum_j \lvert \tilde{u}(s_j, T) - \Phi(s_j)\rvert^2$$
For the ETCNN, $\mathcal{L}_{tc}$ is at or near machine zero throughout (enforced by architecture); for the PINN it is a learned soft penalty.

**Plot E2 — Price surface:**
Side-by-side heatmaps of the predicted price $\tilde{u}(s,t)$ and the analytical Black–Scholes price $V^e(s,t)$ over the evaluation grid. Visual agreement should be near-perfect; residual differences appear mainly near the kink at $(s,t) = (K, T)$.

**Plot E3 — Pointwise error comparison (ETCNN vs PINN):**
A side-by-side 2-panel figure over the evaluation grid $s \in [60, 120]$, $t \in [0, T]$ showing the absolute price error for both models against the analytical Black–Scholes solution:
$$\varepsilon_{\text{ETCNN}}(s,t) = \left\lvert \tilde{u}_\theta(s,t) - V^e(s,t) \right\rvert, \qquad \varepsilon_{\text{PINN}}(s,t) = \left\lvert u_\theta(s,t) - V^e(s,t) \right\rvert$$
Both panels share the same colour scale so differences in accuracy are immediately visible. Because the ETCNN encodes the terminal condition exactly via $g_1$ and $g_2$, the terminal layer $t = T$ is near machine zero; for the PINN it is only a soft penalty. Residual ETCNN error concentrates near the kink at $(s,t) = (K, T)$.

**Plot E4 — Time slices:**
Overlays $\tilde{u}(s, t_k)$, PINN, and $V^e(s, t_k)$ for fixed times $t_k \in \{0.25, 0.5, 0.75\}$. Deviations from the analytical curve quantify the space-only error profile; the kink at $s = K$ is the most demanding region.

**Plot E5 — Greeks:**
Computes the three first-order sensitivities of $\tilde{u}$ at $t = 0$ via automatic differentiation and compares with the Black–Scholes closed-form:
$$\Delta = \frac{\partial \tilde{u}}{\partial s}, \qquad \Gamma = \frac{\partial^2 \tilde{u}}{\partial s^2}, \qquad \Theta = \frac{\partial \tilde{u}}{\partial t}$$
Large $\Gamma$ errors near $s = K$ indicate how well the network resolves the non-differentiable kink in the payoff.

### 3.3 Problem 2 — Bermudan put (one intermediate exercise date)

A Bermudan put with one intermediate exercise date $t_1 = 0.5$ is solved by piecewise backward induction over two European-type sub-problems.

**Two-stage backward solving:**

| Stage | Domain | Terminal condition | Network |
|-------|--------|--------------------|---------|
| A | $[t_1, T]$ | $\Phi(s) = (K - s)^+$ at $t = T$ | ${ETCNN}_A$ |
| B (intermediate) | — | $V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_1) = \max\!\bigl(\Phi(s),\;\tilde{u}^{(A)}_{\bar{\theta}}(s, t_1)\bigr)$ | — |
| C+D | $[0, t_1]$ | $V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_1)$ at $t = t_1$ | ${ETCNN}_B$ |

**Stage B — intermediate condition:** At $t = t_1$, the Bermudan holder compares immediate exercise $\Phi(s)$ with the continuation value $\tilde{u}^{(A)}_{\bar{\theta}}(s, t_1)$ (the Stage A ETCNN output). The maximum defines the terminal condition for Stage D.

**Interpolation:** $V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_1)$ is evaluated on a dense grid of 2,000 points and stored as a look-up table. As detailed in [`architecture.md`](architecture.md), this is interpolated using either a natural cubic spline ($C^2$) or a PCHIP interpolant ($C^1$, shape-preserving) to preserve the diffusion term $\frac{\partial^2 g_2}{\partial s^2}$. PCHIP is recommended near kinks (exercise boundaries) where the global cubic spline can overshoot.

**Terminal functions for Stage D:**
$$
g_1(s, t) = t_1 - t, \qquad g_2(s, t_1) = V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_1)
$$

Note: $g_2$ is constant in $t$ and does not capture $\sqrt{\tau}$ singularities. This is acceptable because $V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_1)$ is a network output evaluated at a fixed time, not a Black–Scholes closed form — it carries no $\sqrt{\tau}$ singular structure by construction.

**Full piecewise neural network approximation:**

The complete Bermudan approximation stitches both ETCNNs across their respective sub-intervals. For any $(s, t)$:

*Without singularity extraction* (`--put-ansatz` off, default):

$$
V_{\theta}(s, t) = \begin{cases}
(T - t)\, u_{\theta_A}(s, t) + g_2^{(A)}(s, t), & t \in [t_1, T] \\[4pt]
(t_1 - t)\, u_{\theta_B}(s, t) + V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_1), & t \in [0, t_1]
\end{cases}
$$

where $g_2^{(A)}$ is the Stage A anchor (Taylor, BS, or BS2002) and $V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_1)$ is the interpolated intermediate terminal condition.

*With singularity extraction* (`--put-ansatz`):

$$
V_{\theta}(s, t) = \begin{cases}
(T - t)\, u_{\theta_A}(s, t) + g_2^{(A)}(s, t), & t \in [t_1, T] \\[4pt]
v(s, t) + (t_1 - t)\, u_{\theta_B}(s, t) + g_2^{(B)}(s), & t \in [0, t_1]
\end{cases}
$$

where $v(s, t)$ is the fictitious European put that absorbs the $C^0$ kink at $s^*$ (see [`architecture.md §2.2`](architecture.md)), and $g_2^{(B)}(s) = V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_1) - v(s, t_1)$ is the smooth $C^1$ PCHIP residual. At $t = t_1$, $g_1^{(B)} = 0$ so the neural manifold vanishes and the formula collapses to $v(s, t_1) + g_2^{(B)}(s) = V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_1)$, matching the terminal condition exactly.

**Exercise boundary:**
The exercise boundary $s^*$ at $t_1$ is the asset price where $\Phi(s) = \tilde{u}^{(A)}_{\bar{\theta}}(s, t_1)$, found by sign-change detection. For these parameters, $s^* < K = 100$.

**Spatial weighting** (`--spatial-weight`, requires `--put-ansatz`)**:**

Even after extracting the $C^0$ kink into $v(s, t)$, the PCHIP interpolant of the residual $g_2^{(B)}$ still has a $C^1$ knot at $s^*$: the second derivative $\partial_{ss} g_2^{(B)}$ is bounded but has a jump there. Differentiating through this knot during PDE-loss backpropagation produces a localized gradient spike that can destabilize training.

To suppress it, an inverted-Gaussian spatial weight $w(s)$ is applied to the PDE loss collocation points:

$$
w(s) = 1 - (1 - \epsilon_w)\exp\!\left(-\frac{(s - s^*)^2}{2\sigma_w^2}\right)
$$

The weighted PDE loss replaces the plain MSE:

$$
\mathcal{L}_f = \mathrm{mean}\bigl(W \cdot \mathcal{F}(U_{\mathrm{pde}})^2\bigr), \qquad W = \mathrm{detach}(w(s))
$$

- At $s^*$: $w(s^*) = \epsilon_w \approx 10^{-3}$ — near-complete suppression of the gradient spike.
- Far from $s^*$: $w(s) \to 1$ — loss is unaffected.
- $\sigma_w$ (default `1.0`) controls the width of the suppression window.
- $W$ is strictly detached from the computational graph: it acts as a static scalar, not a learnable weight.

Disabled by default; activate with `--spatial-weight`.

**Plots produced:**
| Plot | File | Description |
|------|------|-------------|
| B1 | `pricing/plotB1_intermediate.png` | Intermediate terminal condition: hold $\tilde{u}^{(A)}_{\bar{\theta}}(s, t_1)$, exercise $\Phi(s)$, Bermudan value $V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_1)$, and annotated $s^*$ |
| B1b | `diagnostics/plotB1b_interp_diagnostic.png` | Interpolation diagnostic: $V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_1)$, $\partial_s V^{\mathrm{Berm}}_{\bar{\theta}}$, $\partial_s^2 V^{\mathrm{Berm}}_{\bar{\theta}}$, $\mathcal{F}[V^{\mathrm{Berm}}_{\bar{\theta}}(\cdot, t_1)]$ for cubic vs linear |
| B1c | `pricing/plotB1c_*.png` | Interpolated $V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_1)$ with $s^*$; or singularity decomposition $V^{\mathrm{Berm}}_{\bar{\theta}} = v + g_2$ (extraction mode) |
| B1d | `diagnostics/plotB1d_*.png` | Curvature ($\Gamma$) of interpolant near $s^*$; or curvature comparison: $\partial_s^2 V^{\mathrm{Berm}}_{\bar{\theta}}$ vs $\partial_s^2 g_2$ (extraction mode) |
| B3 | `training_metrics/plotB3_stageA_loss.png` | Stage A loss curves ($\mathcal{L}_f$, $\mathcal{L}_{tc}$) for ETCNN$_A$ on $[t_1, T]$ |
| B4 | `training_metrics/plotB4_stageD_loss.png` | Stage D loss curves ($\mathcal{L}_f$, $\mathcal{L}_{tc}$) for ETCNN$_B$ on $[0, t_1]$ |
| B5 | `pricing/plotB5_price_comparison.png` | Price slice at $t = 0$: ETCNN$_B$ vs binomial tree vs European $V^e$ |
| B6 | `pricing/plotB6_bermudan_surface.png` | Full piecewise Bermudan price surface $\tilde{u}(s, t)$ over $[0, T]$ with $t_1$ boundary marked |
| B7 | `diagnostics/plotB7_error_vs_bt.png` | Pointwise error $\lvert \tilde{u}^{(B)}(s, 0) - V^{BT}(s, 0)\rvert$ at $t = 0$ |
| B8 | `greeks/plotB8_greeks.png` | Greeks $\Delta, \Gamma, \Theta$ at $t = 0$: Bermudan ETCNN$_B$ vs European analytical |
| B9 | `diagnostics/plotB9_test2_pde_residual.png` | PDE residual $\lvert \mathcal{F}[\tilde{u}^{(B)}](s, t_1^-)\rvert^2$ across asset prices — spike near $s^*$ indicates kink effect |
| B9b | `diagnostics/plotB9b_pde_residual_heatmap.png` | Spatio-temporal heatmap of $\lvert \mathcal{F}[\tilde{u}](s,t)\rvert$ over the full $(s,t)$ domain, split at $t_1$: left panel ETCNN$_B$ on $[0, t_1^-]$, right panel ETCNN$_A$ on $[t_1, T]$; shared log colour scale |
| B10 | `diagnostics/plotB10_test3_weight_distribution.png` | Neuron weight magnitude distribution in ETCNN$_A$ (violinplot by layer) |

**Plot B1 — Intermediate terminal condition:**
Shows the three curves that define the Stage B decision at $t = t_1$:
$$V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_1) = \max\!\bigl(\Phi(s),\; \tilde{u}^{(A)}_{\bar{\theta}}(s, t_1)\bigr)$$
where $\Phi(s) = (K-s)^+$ is the immediate exercise value and $\tilde{u}^{(A)}_{\bar{\theta}}(s, t_1)$ is the Stage A ETCNN continuation value. The crossover point is the exercise boundary $s^*$.

**Plot B1b — Interpolation diagnostic:**
A 2×2 diagnostic showing how well the cubic ($C^2$) vs linear ($C^0$) interpolant of $V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_1)$ supports the diffusion term in the PDE loss. The four panels are:

| Panel | Quantity |
|-------|---------|
| (a) | $V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_1)$ — the interpolated terminal value (visually identical for both) |
| (b) | $\partial V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_1)/\partial s$ — Delta of the terminal condition |
| (c) | $\partial^2 V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_1)/\partial s^2$ — Gamma; identically zero for the linear interpolant |
| (d) | $\mathcal{F}[V^{\mathrm{Berm}}_{\bar{\theta}}(\cdot, t_1)] = \tfrac{1}{2}\sigma^2 s^2\, \partial_s^2 V^{\mathrm{Berm}}_{\bar{\theta}} + r s\, \partial_s V^{\mathrm{Berm}}_{\bar{\theta}} - r\, V^{\mathrm{Berm}}_{\bar{\theta}}$ — the BSM operator applied to $V^{\mathrm{Berm}}_{\bar{\theta}}(\cdot, t_1)$ alone |

Panel (c) is the key diagnostic: a vanishing $\Gamma$ means the diffusion term $\frac{1}{2}\sigma^2 s^2 \frac{\partial^2 V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_1)}{\partial s^2}$ contributes nothing to the PDE residual, which can leave the network under-constrained near the kink.

**Plot B1c — Interpolated function / Singularity extraction:**
In standard mode: plots the chosen interpolant of $V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_1) = \max\!\bigl(\Phi(s),\, \tilde{u}^{(A)}_{\bar{\theta}}(s, t_1)\bigr)$ with the exercise boundary $s^*$ marked.
In extraction mode: shows the singularity decomposition at $t = t_1$ (the terminal date of the fictitious put, where $v(s, t_1) = c\cdot(s^* - s)^+$):
$$V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_1) = v(s, t_1) + g_2(s, t_1), \qquad v(s, t_1) = c\cdot(s^* - s)^+$$
Here $v(s, t)$ is the full fictitious European put (architecture.md §2.2); this plot evaluates it at its own maturity $t_k = t_1$, where it reduces to the scaled payoff above. $g_2(s, t_1)$ is the smooth $C^1$ residual interpolated by PCHIP.

**Plot B1d — Curvature diagnostic:**
In standard mode: plots the Gamma $\partial^2 g_2(s, t_1)/\partial s^2$ of the selected interpolant near $s^*$ on both a wide and fine grid, verifying whether the second derivative remains bounded.
In extraction mode: compares the Gamma of $V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_1)$ (which has a kink at $s^*$) against the smooth residual $g_2(s, t_1)$, confirming that the singularity has been successfully absorbed into $v(s, t_1)$.

**Plot B5 — Price at $t = 0$:**
Compares the Bermudan price $\tilde{u}^{(B)}(s, 0)$ to the binomial-tree reference $V^{BT}(s, 0)$ and the European lower bound $V^e(s, 0)$. The Bermudan price should lie above the European price for $s < s^*$.

**Plot B7 — Error at $t = 0$:**
Pointwise absolute error against the binomial tree:
$$\varepsilon(s) = \left\lvert \tilde{u}^{(B)}(s, 0) - V^{BT}(s, 0) \right\rvert$$
Largest errors typically occur near $s^*$ where the terminal condition $g_2$ has a kink.

**Plot B9 — PDE residual at $t_1^-$:**
The BSM residual of ETCNN$_B$ evaluated just before the exercise date:
$$\mathcal{F}[\tilde{u}^{(B)}](s, t_1^-) = \frac{\partial \tilde{u}}{\partial t} + \tfrac{1}{2}\sigma^2 s^2 \frac{\partial^2 \tilde{u}}{\partial s^2} + rs\frac{\partial \tilde{u}}{\partial s} - r\tilde{u}$$
A spike near $s^*$ indicates that the interpolant's smoothness directly limits how well the PDE can be satisfied close to the exercise boundary.

**Plot B9b — Spatio-temporal PDE residual heatmap:**
A two-panel heatmap of $|\mathcal{F}[\tilde{u}](s,t)|$ over the full evaluation domain $s \in [S_{\min}, S_{\max}]$, covering both sub-intervals:

| Panel | Model | Domain |
|-------|-------|--------|
| Left | ETCNN$_B$ | $t \in [0,\, t_1^-]$ (includes the point just before the exercise date) |
| Right | ETCNN$_A$ | $t \in [t_1,\, T]$ |

Both panels share a single log-scale colour axis so relative magnitudes are directly comparable. The horizontal dashed line marks the exercise boundary $s^*$; cyan vertical lines mark $t_1$ and $T$.

The BSM operator is:
$$\mathcal{F}[\tilde{u}](s,t) = \frac{\partial \tilde{u}}{\partial t} + \tfrac{1}{2}\sigma^2 s^2 \frac{\partial^2 \tilde{u}}{\partial s^2} + rs\frac{\partial \tilde{u}}{\partial s} - r\tilde{u}$$

**Validity conditions checked at runtime:**

1. **Structural** — two distinct sub-intervals must exist: $t_1 > 0$ and $T > t_1$. If not, the plot is skipped.
2. **Spatial dimension** — the BSM PDE is 1D in $s$ (single-asset), so a $(s,t)$ heatmap is always well-defined. For a multi-factor model one would need to fix the extra state dimensions; not applicable here.
3. **Rate sign** — when $r \leq 0$ it may be optimal never to exercise a put early ($s^* \to 0$), so no kink region appears in the heatmap. A warning is logged but the plot is still generated.
4. **Domain coverage** — if $s^* \notin [S_{\min}, S_{\max}]$ the kink is off-screen; a warning is logged. We also flag the financially inconsistent case $s^* \geq K$ (for a put with $r > 0$, $s^* < K$ is expected).

---

## 4. Running experiments

### 4.1 Full training run (GPU recommended)

```bash
python3 experiments/python_scripts/exp1/phase3_training.py \
    --iters 50000 \
    --device auto
```

### 4.2 Fast diagnostic run (CPU, ~20 min)

Reduced batch sizes and iterations for quick debugging:

```bash
python3 experiments/python_scripts/exp1/phase3_training.py \
    --device cpu \
    --iters 2000 \
    --n-f 1024 \
    --n-tc 256
```

### 4.3 Bermudan-only with PCHIP interpolation

Skip the European problem and use shape-preserving PCHIP interpolation to avoid
Gamma explosions from the global cubic spline near the exercise boundary:

```bash
python3 experiments/python_scripts/exp1/phase3_training.py \
    --bermudan-only \
    --interp pchip \
    --iters 20000
```

### 4.4 Two-stage iterations with weight decay

Use different iteration counts for Stage A and Stage B, with L2 regularisation:

```bash
python3 experiments/python_scripts/exp1/phase3_training.py \
    --iters 20000 5000 \
    --weight-decay 0.01 \
    --interp pchip
```

### 4.5 Resume from a pre-trained Stage A model

Skip Stage A training by loading a saved `etcnn_a.pt`:

```bash
python3 experiments/python_scripts/exp1/phase3_training.py \
    --bermudan-only \
    --load-etcnn-a data/phase3_training/<run_folder>/etcnn_a.pt \
    --iters 10000 \
    --interp pchip
```

### 4.6 BS-2002 American put anchor

Use the Bjerksund–Stensland (2002) approximation as the $g_2$ anchor, which captures
the early-exercise boundary kink analytically:

```bash
python3 experiments/python_scripts/exp1/phase3_training.py \
    --g2 bs2002 \
    --iters 50000 \
    --device auto
```

Bermudan with BS-2002 anchor, singularity extraction, and Operator Bypass:

```bash
python3 experiments/python_scripts/exp1/phase3_training.py \
    --g2 bs2002 \
    --put-ansatz \
    --bypass-v \
    --bermudan-only \
    --iters 20000 5000
```

### 4.7 CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--iters N [M]` | `50000` | Training iterations. One value = same for all stages; two values = Stage A, Stage B. |
| `--g2 {taylor,bs,bs2002}` | `taylor` | Terminal function $g_2$ for ETCNN: `taylor` ($V_1^e + V_2^e$), `bs` (exact BS European put), `bs2002` (Bjerksund–Stensland 2002 American put approximation). |
| `--put-ansatz` | off | Enable singularity extraction ansatz for Stage B. |
| `--bypass-v` | off | Operator Bypass (**Bermudan/Stage B+ only**): skip differentiating the fictitious put $v(s,t)$ in the PDE loss to prevent catastrophic cancellation of its diverging derivatives near $s^*$. Not applicable to the European or Stage A problems: `AmericanPutETCNN` contains no singular $v$ component, so no bypass is needed or defined there. |
| `--spatial-weight` | off | Enable inverted-Gaussian spatial weighting of the Stage B PDE loss to suppress the gradient spike from the $C^1$ PCHIP knot at $s^*$. Requires `--put-ansatz`. |
| `--sigma-w` | `1.0` | Bandwidth $\sigma_w$ of the spatial suppression window (requires `--spatial-weight`). |
| `--eps-w` | `1e-3` | Floor weight $\epsilon_w$ at $s^*$; prevents complete nullification of the loss (requires `--spatial-weight`). |
| `--interp {cubic,pchip,linear}` | `cubic` | Interpolation method for $V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_1)$. `pchip` is shape-preserving ($C^1$). Ignored when `--put-ansatz` is set. |
| `--device {auto,cuda,cpu}` | `auto` | Compute device. |
| `--bermudan-only` | off | Skip European problem, run only Bermudan stages. |
| `--european-only` | off | Skip Bermudan problem, run only European stages. |
| `--weight-decay` | `0.0` | L2 regularisation penalty for Adam. |
| `--load-etcnn-a PATH` | — | Path to pre-trained `etcnn_a.pt` to skip Stage A. |
| `--n-tc N` | `1024` | Number of terminal condition boundary points. |
| `--n-f N` | `4096` | Number of interior PDE collocation points. |
| `--log-every N` | `1000` | Logging interval (iterations). |

---

## 5. Math → code mapping (Phase 3 additions)

| Symbol | Description | Code location |
|--------|-------------|---------------|
| $\mathcal{L}_f$ | PDE residual loss | `phase3_training.compute_losses` |
| $\mathcal{L}_{tc}$ | Terminal condition loss | `phase3_training.compute_losses` |
| $V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_1)$ | Intermediate terminal condition $\max(\Phi(s), \tilde{u}^{(A)}_{\bar{\theta}}(s, t_1))$ | `phase3_training.bermudan_problem` — `v_interp_t1` |
| $\Delta, \Gamma, \Theta$ | Option Greeks computation | `phase3_training.compute_greeks_nn`, `compute_greeks_analytical` |
| BT reference | Bermudan binomial tree | `solvers.bermuda_put_binomial_tree` |
| $s^*$ | Exercise boundary at $t_1$ | Sign-change detection in `bermudan_problem` |
| $w(s)$ | Spatial weight function | `phase3_training.compute_losses` |
| $W$ | Detached spatial weight tensor | `phase3_training.compute_losses` |
