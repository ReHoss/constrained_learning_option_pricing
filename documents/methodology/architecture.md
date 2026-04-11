# ETCNN Architecture & Loss Functions

> Math rendering: open in a Markdown+KaTeX/MathJax renderer for rendered equations.

This document describes the neural network architecture and loss design for extending
ETCNN to various option types, including European, American, and Bermudan options.

For the training procedure and implementation phases, see [`implementation_phases.md`](implementation_phases.md).
For reference numerical methods, see [`solvers_and_benchmarks.md`](solvers_and_benchmarks.md).

Key reference:

* W. Zhang, Y. Guo, B. Lu — *Exact Terminal Condition Neural Network for American
  Option Pricing Based on the Black–Scholes–Merton Equations*, J. Comput. Appl. Math.
  480 (2026) 117253.

---

## 1. Problem formulation

### 1.1 Black–Scholes–Merton equation (single asset)

The option price $V(s, t)$ satisfies the BSM PDE operator:

$$
\mathcal{F}(V) = \frac{\partial V}{\partial t}
  + \tfrac{1}{2}\sigma^2 s^2 \frac{\partial^2 V}{\partial s^2}
  + (r - q)\,s \frac{\partial V}{\partial s} - r V
$$

Code: `pricing.terminal.bsm_operator`.

### 1.2 European option

$$
\mathcal{F}(V) = 0, \quad (s, t) \in \mathbb{R}_{>0} \times [0, T)
\qquad V(s, T) = \Phi(s)
$$

For a put: $\Phi(s) = (K - s)^+$.  Analytical solution: $V^e(s, t)$ (Black–Scholes, Eq. 16).
Code: `pricing.terminal.black_scholes_put`.

> **Relation to Bermudan:** the European option is the **zero-exercise-date** special case of the Bermudan option (§1.4). With no intermediate exercise dates the backward induction collapses to a single stage on $[0, T]$, and the loss reduces to $\lambda_f \mathcal{L}_f + \lambda_{tc} \mathcal{L}_{tc}$ with the analytical payoff $\Phi(s)$ as the terminal condition. The European problem therefore serves as the primary validation baseline: if the single-stage ETCNN fails on the European put, the multi-stage Bermudan will fail too.

### 1.3 American option (linear complementarity)

$$
\mathcal{F}(V) \le 0, \quad V \ge \Phi, \quad \mathcal{F}(V)\cdot(V - \Phi) = 0,
\quad V(s, T) = \Phi(s)
$$

Three penalty terms $\mathcal{L}_{bs}$, $\mathcal{L}_{tv}$, $\mathcal{L}_{eq}$ enforce this.
Code: `pricing.loss`.

### 1.4 Bermudan option (discrete exercise dates)

A Bermudan option may be exercised at a finite set of dates $\{t_1, \ldots, t_m, T\}$.
Between exercise dates the option satisfies the European BSM equation. At each exercise
date $t_k$ the holder compares immediate exercise against continuation:

$$
V(s, t_k) = \max\!\bigl(\Phi(s),\; V^{\mathrm{hold}}(s, t_k)\bigr)
$$

This is solved by **piecewise backward induction**: train one ETCNN per sub-interval $[t_{k-1}, t_k]$
backward in time, using the intermediate value $V(s, t_k)$ at $t_k$ as the terminal condition for the
next sub-problem.

---

## 2. Exact terminal / exercise-date functions

### 2.1 American & European options

The trial solution is $\tilde{u}_\theta(s,t) = g_1(s,t)\,u_\theta(s,t) + g_2(s,t)$, where:

- $g_1(s,t) = T - t$ — vanishes at the terminal date, forcing exact satisfaction of the terminal condition.
- $g_2(s,t) = V_1^e(s,t) + V_2^e(s,t)$ — first-order Taylor expansion of the European put price around $\tilde{d}_0$:

$$
V_1^e(s,t) = N(\tilde{d}_0)\,(K e^{-r\tau} - s)
$$
$$
V_2^e(s,t) = \frac{\sigma\sqrt{\tau}}{2\sqrt{2\pi}}\,e^{-\tilde{d}_0^2/2}\,(K e^{-r\tau} + s)
$$

where $\tilde{d}_0 = -\frac{1}{\sigma\sqrt{\tau}}\!\left(\ln\frac{s}{K} + r\tau\right)$ and $\tau = T - t$.

This choice (i) satisfies the terminal condition exactly, (ii) captures the $\sqrt{\tau}$ singularity near expiry at-the-money, and (iii) preserves the non-differentiability at $s = K, t = T$.

An alternative is $g_2(s,t) = P^{\text{BS}}(s, K, r, \sigma, \tau)$, the **exact** Black-Scholes European put price. This is smoother (fully analytic) but does not explicitly decompose the $\sqrt{\tau}$ singular behaviour. Both variants are available via the `g2_type` constructor argument of `AmericanPutETCNN` (`"taylor"` — default, or `"bs"`) and via the `--g2 {taylor,bs}` CLI flag of `phase3_training.py`.

Code: `learning_option_pricing/pricing/terminal.py` — `g1_linear`, `g2_american_put`, `european_put_ve1`, `european_put_ve2`, `black_scholes_put`.

### 2.2 Bermuda options — singularity extraction ansatz

**Notation convention:** We use a bar over the network parameters, $\bar{\theta}$, to denote weights that are frozen (i.e., already trained in a previous stage and no longer updated during the current optimization step).

For the sub-interval $[t_{k-1}, t_k]$ the terminal condition is

$$
V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_k) = \max\!\bigl(\Phi(s),\; V^{\text{hold}}_{\bar{\theta}}(s, t_k)\bigr)
$$

**The pathology.** At the optimal exercise boundary $s^*$ (where $\Phi(s^*) = V^{\text{hold}}_{\bar{\theta}}(s^*, t_k)$), $V^{\mathrm{Berm}}_{\bar{\theta}}$ is only $C^0$:

- For $s < s^*$ (exercise region, put): $\partial V^{\mathrm{Berm}}_{\bar{\theta}}/\partial s = -1$ (constant)
- For $s > s^*$ (hold region): $\partial V^{\mathrm{Berm}}_{\bar{\theta}}/\partial s = \Delta_A(s)$, the hold delta — a smooth but non-constant function of $s$

The first derivative has a jump discontinuity at $s^*$ of magnitude $[\![\partial_s V^{\mathrm{Berm}}_{\bar{\theta}}]\!]_{s^*} = \Delta_A(s^*) + 1$. Consequently, $\partial^2 V^{\mathrm{Berm}}_{\bar{\theta}}/\partial s^2$ contains a **Dirac distribution** at $s^*$. If a neural network is forced to learn this boundary condition, the infinite curvature causes the BSM PDE residual to explode (stiffness), destroying the optimisation.

**Solution: singularity extraction ansatz.**
Instead of smoothing $V^{\mathrm{Berm}}_{\bar{\theta}}$ with an interpolant (which either introduces oscillations or drops the diffusion term), we analytically extract the singularity.

The Stage B network must learn the **full price function** $V_\theta(s, t)$ on the sub-interval $[t_{k-1}, t_k]$, subject to the terminal condition $V_\theta(s, t_k) = V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_k)$. Note that $V_\theta(s, t)$ is a function of both $s$ and $t$, whereas $V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_k)$ is only defined at the single time $t_k$. The kink in $V^{\mathrm{Berm}}_{\bar{\theta}}$ propagates into $V_\theta$ through the terminal condition, causing the BSM PDE residual $\mathcal{F}[V_\theta]$ to explode near $s^*$.

This full price $V_\theta(s, t)$ is decomposed as

$$
V_\theta(s, t) = v(s, t) + \tilde{u}_\theta(s, t)
$$

where $v(s, t)$ is a **fictitious European put** designed to perfectly absorb the $C^0$ kink:

$$
v(s, t) = c \cdot P^{\text{BS}}(s,\; s^*,\; r,\; \sigma,\; t_k - t)
$$

with:

- **Fictitious strike** $= s^*$ (the exercise boundary, not the contract strike $K$).
- **Fictitious maturity** $= t_k$ (the exercise date).
- **Scaling constant** $c = \Delta_A(s^*) + 1 = [\![\partial_s V^{\mathrm{Berm}}_{\bar{\theta}}]\!]_{s^*}$ (matches the jump discontinuity of $\partial_s V^{\mathrm{Berm}}_{\bar{\theta}}$ at $s^*$).

**Why this works:** At $t = t_k$, the fictitious put reduces to $v(s, t_k) = c \cdot (s^* - s)^+$. Its derivative w.r.t. $s$ is $-c$ for $s < s^*$ and $0$ for $s > s^*$, so $[\![\partial_s v]\!]_{s^*} = 0 - (-c) = c$. Since $c = [\![\partial_s V^{\mathrm{Berm}}_{\bar{\theta}}]\!]_{s^*}$, the residual

$$
\tilde{u}_{{\bar{\theta}}}(s, t_k) = V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_k) - v(s, t_k)
$$

is **strictly $C^1$** at $s^*$. The Dirac distribution in the second derivative is removed entirely.

This smooth residual serves as the exact terminal function $g_2(s, t)$ for the ETCNN trial solution in this sub-interval. Since the terminal condition is evaluated at a fixed time $t_k$, $g_2$ is constant with respect to time:

$$
g_2(s, t) = V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_k) - v(s, t_k)
$$

Note (Remy): Here it is enough to have a function that maps the derivative for the case t=t_k.

**PDE structure.** Because the BSM operator $\mathcal{L}$ is linear and $v$ is an exact Black-Scholes solution ($\mathcal{L}v = 0$), we have

$$
\mathcal{L}(V_\theta) = \mathcal{L}(v) + \mathcal{L}(\tilde{u}_\theta) = 0 + \mathcal{L}(\tilde{u}_\theta) = \mathcal{L}(\tilde{u}_\theta)
$$

The network only needs to minimise $\mathcal{L}(\tilde{u}_\theta) = 0$, and the training loop requires no special handling — the PDE residual on the full output $V_\theta$ equals the ETCNN residual.

**Computing $c$ analytically.** The scaling constant is determined by the hold-value delta at the exercise boundary:

$$
c = \frac{\partial V_\theta^A}{\partial s}\bigg|_{s = s^*} + 1
$$

where $\partial V_\theta^A/\partial s$ is computed via `torch.autograd.grad` on the trained Stage A network, evaluated at $s = s^*$. For a well-trained put, $-1 < \Delta_A(s^*) < 0$, so $0 < c < 1$.

**Residual interpolation.** The $C^1$ residual $g_2(s, t) = V^{\mathrm{Berm}}_{\bar{\theta}} - v|_{t_k}$ is interpolated using **PCHIP** ($C^1$, shape-preserving), which exactly matches the regularity of the residual (no attempt to enforce spurious $C^2$ smoothness at the remaining discontinuity of $\partial_s^2 \tilde{u}_\theta$).

**Backward-compatible opt-in.** The singularity extraction ansatz is activated by the `--extraction` flag in `experiments/python_scripts/exp1/phase3_training.py`. When the flag is *not* set (default), Stage B falls back to the classical approach of directly interpolating $V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_k)$ without extraction, and the interpolator is controlled by `--interp` (`cubic`, `pchip`, or `linear`).

Code:
- `learning_option_pricing/pricing/singularity.py` — `find_exercise_boundary`, `compute_scaling_constant`, `FictitiousEuropeanPut`, `build_singularity_extraction`
- `learning_option_pricing/models/etcnn.py` — `BermudaETCNN`
- `learning_option_pricing/pricing/interpolation.py` — `PchipInterpolator` (for the $C^1$ residual)

#### 2.2.1 Interpolation utilities (reference)

The interpolation module provides three interpolators for general use, though the singularity extraction ansatz now uses PCHIP exclusively for the extracted residual:

| Interpolator | Regularity | Scope | Shape-preserving | Use case |
|-------------|-----------|-------|-----------------|----------|
| `CubicSplineInterpolator` | $C^2$ | Global | No | Smooth data without kinks |
| `PchipInterpolator` | $C^1$ | Local | Yes | Extracted residual, data with kinks |
| `PiecewiseLinearInterpolator` | $C^0$ | Local | Yes | Benchmarking only (drops diffusion) |

Code: `learning_option_pricing/pricing/interpolation.py`.

#### 2.2.2 CLI flags for `phase3_training.py`

The Phase 3 experiment exposes the design choices of §2.1–§2.2 as command-line flags so that different configurations can be benchmarked without touching the code:

| Flag | Default | Scope | Effect |
|------|---------|-------|--------|
| `--g2 {taylor,bs}` | `taylor` | Stage A + European | Sets $g_2$ in `AmericanPutETCNN`: `taylor` uses $V_1^e + V_2^e$ (§2.1); `bs` uses the exact BS European put price. |
| `--extraction` | *off* | Stage B+ | If set, decomposes $V_\theta = v + \tilde{u}_\theta$ via the singularity extraction ansatz (§2.2). If unset, Stage B directly interpolates $V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_1)$ (legacy path). |
| `--interp {cubic,pchip,linear}` | `cubic` | Stage B+ (only when `--extraction` is NOT set) | Interpolator used for $V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_1)$ in the legacy path. Ignored under `--extraction` (which always uses PCHIP on the $C^1$ residual). |

The output directory is tagged with the active mode and `g2` variant, e.g. `20260410_120000_iters50000_K100_extraction_g2-taylor` or `20260410_120000_iters50000_K100_interp-cubic_g2-bs`, so that benchmarks over the combinatorial grid do not collide.

---

## 3. Network architecture

### 3.1 ResNet backbone

$M$ residual blocks, each consisting of $L$ fully-connected layers of width $n$ with tanh activations and skip connections:

$$
g^{(m+1,0)}(x) = f_\theta^{(m,L)}(x) + g^{(m,0)}(x)
$$

Baseline configuration (Section 4, paper): $M=4$, $L=2$, $n=50$.

Code: `learning_option_pricing/models/resnet.py` — `ResidualBlock`, `ResNet`.

Default configuration: `ResNet(d_in=2, d_out=1, n=50, M=4, L=2)` — **20,601 trainable parameters**.

### 3.2 Input normalization (Section 3.3)

Asset prices $s$ are divided by $K$ (moneyness $s/K$) before entering the network, while $t$ passes through unchanged. This preserves the homogeneity property $\alpha V(s,K,t) = V(\alpha s, \alpha K, t)$.

Code: `learning_option_pricing/models/etcnn.py` — `InputNormalization`.

### 3.3 Output modification for exact terminal conditions (Fig. 3)

The last linear layer is replaced by:

$$
\tilde{u}_\theta(s,t) = g_1(s,t)\cdot u_\theta(s/K, t) + g_2(s,t)
$$

where $u_\theta(s/K, t) = W^{out}\cdot g^{(M+1,0)}(s/K, t) + b^{out}$ is the raw ResNet output. At $t = T$, $g_1(s, T) = 0$ exactly, so $\tilde{u}_\theta(s, T) = g_2(s, T) = \Phi(s)$ regardless of the network weights.

Code: `learning_option_pricing/models/etcnn.py` — `ETCNN`, `AmericanPutETCNN`.

### 3.4 PINN baseline

A plain physics-informed neural network with the same ResNet backbone but without the $g_1/g_2$ output modification. Used as a control to demonstrate the advantage of exact terminal condition enforcement.

Code: `learning_option_pricing/models/etcnn.py` — `PINN`.

---

## 4. Loss Functions

The loss function varies depending on the type of option being priced.

### 4.1 European Option Loss

For a European option, the option satisfies the BSM PDE exactly ($\mathcal{F}(V) = 0$). The loss is simply the mean squared PDE residual:

$$
\mathcal{L}_{\text{European}}(\theta) = \lambda_f \, \mathcal{L}_f(\theta) + \lambda_{tc} \, \mathcal{L}_{tc}(\theta)
$$

| Term | Formula | Description |
|------|---------|-------------|
| $\mathcal{L}_f$ | $\frac{1}{N_f} \sum_{i=1}^{N_f} \left[\mathcal{F}(\tilde{u}_\theta)(s_i, t_i)\right]^2$ | Mean squared PDE residual at interior collocation points |
| $\mathcal{L}_{tc}$ | $\frac{1}{N_{tc}} \sum_{j=1}^{N_{tc}} \left[\tilde{u}_\theta(s_j, T) - \Phi(s_j)\right]^2$ | Mean squared terminal error (identically zero for ETCNN) |

Weights: $\lambda_f = 20$, $\lambda_{tc} = 1$ (Section 3.4).

### 4.2 American Option Loss (Linear Complementarity)

The composite loss for American options enforces the continuous early-exercise boundary via three penalty terms (ETCNN naturally eliminates the terminal condition term):

$$
\mathcal{L}_{\text{American}}(\theta) = \lambda_{bs}\,\mathcal{L}_{bs} + \lambda_{tv}\,\mathcal{L}_{tv} + \lambda_{eq}\,\mathcal{L}_{eq}
$$

| Term | Formula | Enforces |
|------|---------|---------|
| $\mathcal{L}_{bs}$ | $\text{mean}(\max(\mathcal{F}(\tilde{u}_\theta), 0)^2)$ | BSM operator $\mathcal{F}(V) \le 0$ |
| $\mathcal{L}_{tv}$ | $\text{mean}(\max(-\mathcal{TV}(\tilde{u}_\theta), 0)^2)$ | Time value $V - \Phi \ge 0$ |
| $\mathcal{L}_{eq}$ | $\text{mean}((\mathcal{F}(\tilde{u}_\theta)\cdot\mathcal{TV}(\tilde{u}_\theta))^2)$ | Complementarity $\mathcal{F}\cdot\mathcal{TV} = 0$ |

Default weights: $\lambda_{bs} = \lambda_{tv} = \lambda_{eq} = 1$.
Code: `learning_option_pricing/pricing/loss.py`.

### 4.3 Bermudan Option Loss (Piecewise European)

Unlike American options, Bermudan options cannot be exercised continuously. Between any two discrete exercise dates $t_{k-1}$ and $t_k$, the option behaves exactly like a European option. There is no early exercise opportunity within the open interval $(t_{k-1}, t_k)$.

Therefore, **the Bermudan loss function is identical to the European loss function** applied piecewise to each sub-interval:

$$
\mathcal{L}_{\text{Bermudan}}^{(k)}(\theta) = \lambda_f \, \mathcal{L}_f^{(k)}(\theta) + \lambda_{tc} \, \mathcal{L}_{tc}^{(k)}(\theta)
$$

Where:
- $\mathcal{L}_f^{(k)}$ is the mean squared PDE residual $\mathcal{F}(\tilde{u}_\theta) = 0$ evaluated on collocation points sampled within $t \in [t_{k-1}, t_k]$.
- $\mathcal{L}_{tc}^{(k)}$ is the terminal condition loss at $t = t_k$.

**Degree of exactness of the terminal condition.** The answer depends on the option type:

| Option type | $g_2$ at terminal date | TC satisfied |
|-------------|------------------------|--------------|
| European / American | Closed-form analytic ($V_1^e + V_2^e$ or $P^{\text{BS}}$) | **Exactly everywhere** — $g_1(s,T)=0$ and $g_2(s,T) = \Phi(s)$ at all $s$, regardless of network weights. $\mathcal{L}_{tc} \equiv 0$. |
| Bermudan (each sub-interval) | PCHIP interpolation of the tabulated residual $V^{\mathrm{Berm}}_{\bar{\theta}} - v\vert_{t_k}$ on $n_{\text{grid}}$ nodes | **Exactly at the $n_{\text{grid}}$ interpolation nodes** (PCHIP is a node-interpolating scheme); **approximately between nodes** (PCHIP error $O(h^2)$ near the kink at $s^*$, $O(h^4)$ on $C^2$ segments). |

More precisely, at $t = t_k$ the Bermudan trial solution evaluates to

$$
V_\theta(s, t_k)
  = \underbrace{v(s, t_k)}_{\text{fictitious put at maturity}}
  + \underbrace{0}_{g_1 = 0} \cdot u_\theta
  + \underbrace{g_2(s, t_k)}_{\text{PCHIP}(V^{\mathrm{Berm}}_{\bar{\theta}} - v|_{t_k})(s)}
$$

At a node $s_i$ this equals $V^{\mathrm{Berm}}_{\bar{\theta}}(s_i, t_k)$ exactly; at a non-node point it equals $V^{\mathrm{Berm}}_{\bar{\theta}}$ only up to the interpolation error. Therefore $\mathcal{L}_{tc}^{(k)}$ is not identically zero — it is zero to **interpolation accuracy**. With the default $n_{\text{grid}} = 2000$ this error is negligible relative to the PDE residual, but it is not machine-precision exact.

**Crucial Distinction:** The complementarity terms ($\mathcal{L}_{bs}$, $\mathcal{L}_{tv}$, $\mathcal{L}_{eq}$) are **not** used during the training of a Bermudan sub-interval. The early exercise condition is instead enforced discretely at the boundaries $t_k$ when constructing the intermediate terminal condition $V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_k) = \max(\Phi(s), \tilde{u}^{(A)}_{\bar{\theta}}(s, t_k))$ for the subsequent backward step.

---

## 5. Math → code mapping

| Symbol | Description | Code location |
|--------|-------------|---------------|
| $\mathcal{F}(V)$ | BSM PDE operator (Eq. 2) | `terminal.bsm_operator` |
| $\mathcal{TV}(V)$ | Time value $V - \Phi$ | `terminal.time_value` |
| $\Phi(s)$ | Payoff function | `terminal.payoff_put` / `payoff_call` |
| $V^e(s,t)$ | European put price (Eq. 16) | `terminal.black_scholes_put` |
| $\tilde{d}_1, \tilde{d}_2$ | log-moneyness terms for European put | `terminal._d_tilde_1` / `_d_tilde_2` |
| $\tilde{d}_0$ | midpoint of $\tilde{d}_1, \tilde{d}_2$ | `terminal._d_tilde_0` |
| $V_1^e$ | zeroth-order Taylor term | `terminal.european_put_ve1` |
| $V_2^e$ | first-order Taylor term (√τ singularity) | `terminal.european_put_ve2` |
| $g_1(s,t)$ | terminal vanishing factor | `terminal.g1_linear` |
| $g_2(s,t)$ | exact terminal function (American) | `terminal.g2_american_put` |
| $g_2(s,t)$ | interpolated residual (Bermudan) | `interpolation.PchipInterpolator` (of $V^{\mathrm{Berm}}_{\bar{\theta}} - v$) |
| $v(s,t)$ | fictitious European put | `singularity.FictitiousEuropeanPut` |
| $s^*$ | exercise boundary | `singularity.find_exercise_boundary` |
| $c$ | scaling constant $\Delta_A(s^*) + 1$ | `singularity.compute_scaling_constant` |
| $V_\theta^A(s,t)$ | Stage A parameterized price (American put ETCNN) | `etcnn.AmericanPutETCNN.forward` |
| $V_\theta(s,t)$ | Bermudan solution $v + \tilde{u}_\theta$ | `etcnn.BermudaETCNN.forward` |
| $\mathcal{L}_{bs}$ | BSM penalty loss | `loss.loss_bs` |
| $\mathcal{L}_{tv}$ | time-value penalty loss | `loss.loss_tv` |
| $\mathcal{L}_{eq}$ | complementarity loss | `loss.loss_eq` |
| $u_\theta$ | raw ResNet output | `resnet.ResNet` |
| $\tilde{u}_\theta$ | ETCNN trial solution ($g_1 u_\theta + g_2$) | `etcnn.ETCNN.forward` |
| $s/K$ | input normalisation | `etcnn.InputNormalization` |
