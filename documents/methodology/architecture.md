# ETCNN Architecture

> Math rendering: open in a Markdown+KaTeX/MathJax renderer for rendered equations.

This document describes the neural network architecture and loss design for extending
ETCNN to Bermuda options.

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
next sub-problem. See `implementation_phases.md` for the full two-stage procedure.

---

## 2. Exact terminal / exercise-date functions

### American options (implemented — Phase 1)

The trial solution is $\tilde{u}_{NN}(s,t) = g_1(s,t)\,u_{NN}(s,t) + g_2(s,t)$, where:

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

Code: `learning_option_pricing/pricing/terminal.py` — `g1_linear`, `g2_american_put`, `european_put_ve1`, `european_put_ve2`.

### Bermuda options — interpolation of $g_2$

For the sub-interval $[t_{k-1}, t_k]$ the terminal condition is the tabulated continuation
value $V(s, t_k) = \max(\Phi(s), V^{\text{hold}}(s, t_k))$. Since no closed-form expression
exists, $g_2(s, t) = V_{\text{interp}}(s)$ is constructed by interpolating the tabulated
values.

**Regularity requirement.** Applying the BSM operator to the trial solution
$\tilde{u}_{NN} = g_1 u_{NN} + g_2$ yields (by linearity):

$$
\mathcal{F}(\tilde{u}_{NN}) = \mathcal{F}(g_1 u_{NN}) + \mathcal{F}(g_2)
$$

Since $g_2$ is independent of $t$, the operator reduces to:

$$
\mathcal{F}(g_2) = \tfrac{1}{2}\sigma^2 s^2 \frac{\partial^2 g_2}{\partial s^2}
  + r\,s \frac{\partial g_2}{\partial s} - r\,g_2
$$

If $g_2$ is only $C^0$ (piecewise linear), then $\frac{\partial^2 g_2}{\partial s^2} = 0$
almost everywhere, and the **diffusion term** $\tfrac{1}{2}\sigma^2 s^2 g_2''$ is entirely
lost. This means the volatility of the underlying does not propagate through the boundary
condition — a structural defect.

**Solution: natural cubic spline ($C^2$).** A cubic spline $S(x)$ is piecewise cubic on each
sub-interval, with $S, S', S'' \in C^0$. The second derivative $S''(x)$ is piecewise-linear
(not identically zero), so the full BSM operator applies correctly.

The spline coefficients are computed once from the tabulated nodes via the Thomas algorithm
(tridiagonal solve), and evaluation uses polynomial torch ops so that `torch.autograd.grad`
can compute $S'$ and $S''$.

Code: `learning_option_pricing/pricing/interpolation.py` — `CubicSplineInterpolator` (default),
`PiecewiseLinearInterpolator` (for benchmarking).

CLI flag: `--interp cubic` (default) or `--interp linear`.

**Diagnostic plot B1b** compares both interpolants: values, first derivatives, second
derivatives, and $\mathcal{F}(g_2)$ evaluated on $g_2$ alone.

See also: `BermudaETCNN` stub in `learning_option_pricing/models/etcnn.py`.

---

## 3. Network architecture

### ResNet backbone

$M$ residual blocks, each consisting of $L$ fully-connected layers of width $n$ with tanh activations and skip connections:

$$
g^{(m+1,0)}(x) = f_\theta^{(m,L)}(x) + g^{(m,0)}(x)
$$

Baseline configuration (Section 4, paper): $M=4$, $L=2$, $n=50$.

Code: `learning_option_pricing/models/resnet.py` — `ResidualBlock`, `ResNet`.

Default configuration: `ResNet(d_in=2, d_out=1, n=50, M=4, L=2)` — **20,601 trainable parameters**.

### Input normalization (Section 3.3)

Asset prices $s$ are divided by $K$ (moneyness $s/K$) before entering the network, while $t$ passes through unchanged. This preserves the homogeneity property $\alpha V(s,K,t) = V(\alpha s, \alpha K, t)$.

Code: `learning_option_pricing/models/etcnn.py` — `InputNormalization`.

### Output modification for exact terminal conditions (Fig. 3)

The last linear layer is replaced by:

$$
\tilde{u}_{NN}(s,t) = g_1(s,t)\cdot u_{NN}(s/K, t) + g_2(s,t)
$$

where $u_{NN}(s/K, t) = W^{out}\cdot g^{(M+1,0)}(s/K, t) + b^{out}$ is the raw ResNet output. At $t = T$, $g_1(s, T) = 0$ exactly, so $\tilde{u}_{NN}(s, T) = g_2(s, T) = \Phi(s)$ regardless of the network weights.

Code: `learning_option_pricing/models/etcnn.py` — `ETCNN`, `AmericanPutETCNN`.

### PINN baseline

A plain physics-informed neural network with the same ResNet backbone but without the $g_1/g_2$ output modification. Used as a control to demonstrate the advantage of exact terminal condition enforcement.

Code: `learning_option_pricing/models/etcnn.py` — `PINN`.

---

## 4. Loss function

The composite loss for American options (ETCNN eliminates the terminal condition term):

$$
\mathcal{L}(\theta) = \lambda_{bs}\,\mathcal{L}_{bs} + \lambda_{tv}\,\mathcal{L}_{tv} + \lambda_{eq}\,\mathcal{L}_{eq}
$$

| Term | Formula | Enforces |
|------|---------|---------|
| $\mathcal{L}_{bs}$ | $\text{mean}(\max(\mathcal{F}(u_\theta), 0)^2)$ | BSM operator $\mathcal{F}(V) \le 0$ |
| $\mathcal{L}_{tv}$ | $\text{mean}(\max(-\mathcal{TV}(u_\theta), 0)^2)$ | Time value $V - \Phi \ge 0$ |
| $\mathcal{L}_{eq}$ | $\text{mean}((\mathcal{F}(u_\theta)\cdot\mathcal{TV}(u_\theta))^2)$ | Complementarity $\mathcal{F}\cdot\mathcal{TV} = 0$ |

Default weights: $\lambda_{bs} = \lambda_{tv} = \lambda_{eq} = 1$.

Code: `learning_option_pricing/pricing/loss.py`.

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
| $g_2(s,t)$ | interpolated terminal (Bermudan) | `interpolation.CubicSplineInterpolator` |
| $\mathcal{L}_{bs}$ | BSM penalty loss | `loss.loss_bs` |
| $\mathcal{L}_{tv}$ | time-value penalty loss | `loss.loss_tv` |
| $\mathcal{L}_{eq}$ | complementarity loss | `loss.loss_eq` |
| $u_{NN}$ | raw ResNet output | `resnet.ResNet` |
| $\tilde{u}_{NN}$ | ETCNN trial solution | `etcnn.ETCNN.forward` |
| $s/K$ | input normalisation | `etcnn.InputNormalization` |

---

## 6. Phase 2 validation results

Validation script: `experiments/python_scripts/exp1/phase2_etcnn_architecture.py`.

### Scalar checks

| Check | Value | Criterion |
|-------|-------|-----------|
| Total trainable parameters | 20,601 | Matches manual calculation for $M{=}4, L{=}2, n{=}50$ |
| $\max_s |\tilde{u}_{NN}(s, T) - \Phi(s)|$ | $0.00$ | $< 10^{-7}$ (exact terminal condition) |
| $\max_s |g_1(s, T)|$ | $0.00$ | $= 0$ (network output suppressed at $t{=}T$) |
| Gradient norms at init | $[0.12, 2.11]$ | All $> 10^{-8}$ (no vanishing) and $< 100$ (no exploding) |
| PINN terminal error | $40.05$ | $\gg 0$ (confirms PINN does NOT satisfy terminal condition) |

### Plots produced

| Plot | Description |
|------|-------------|
| 1 | Untrained ETCNN surface vs true $V^e(s,t)$ — terminal edge matches |
| 2 | Terminal edge overlay: $\tilde{u}_{NN}(s,T)$, $\Phi(s)$, $V^e(s,T)$ overlap exactly |
| 3 | PINN vs ETCNN at $t{=}T$ — PINN fails, ETCNN matches payoff |
| 4 | Gradient norms per layer — healthy flow, no vanishing/exploding |
| 5 | Input normalisation effect — confirms $s/K$ scaling is active |
| 6 | Architecture summary — full layer-by-layer description |



# Numerical Solvers and Benchmarks

> Math rendering: open in a Markdown+KaTeX/MathJax renderer for rendered equations.

This document describes the reference numerical methods used to generate benchmark
solutions for option pricing, and their role in the validation pipeline.

Key reference:

* Cox, J.C., Ross, S.A., Rubinstein, M. — *Option Pricing: A Simplified Approach*,
  J. Financ. Econ. **7**(3) (1979) 229–263.

---

## 1. Cox-Ross-Rubinstein (CRR) Binomial Tree

Code: `learning_option_pricing/solvers/binomial_tree.py`

### 1.1 Algorithm

The CRR binomial tree discretises the underlying asset price over $N$ equally spaced
time steps $\Delta t = T / N$.

**Up/down factors and risk-neutral probability:**

$$
u = e^{\sigma\sqrt{\Delta t}}, \qquad d = \frac{1}{u}, \qquad
p = \frac{e^{(r-q)\Delta t} - d}{u - d}
$$

**Terminal asset prices** at step $N$:

$$
S_{N,j} = S_0 \, u^{N-j} \, d^j, \qquad j = 0, 1, \ldots, N
$$

**Backward induction:**

$$
V_{i,j} = e^{-r\Delta t} \bigl[ p \, V_{i+1,j} + (1-p) \, V_{i+1,j+1} \bigr]
$$

with early exercise applied at each node where applicable:

$$
V_{i,j} = \max\!\bigl(\text{hold},\; \Phi(S_{i,j})\bigr)
$$

### 1.2 Implemented variants

| Function | Option type | Early exercise |
|----------|-------------|----------------|
| `european_put_binomial_tree` | European put | None — hold value only |
| `american_put_binomial_tree` | American put | At every node |
| `american_call_binomial_tree` | American call (with dividends) | At every node |
| `bermuda_put_binomial_tree` | Bermuda put | Only at prescribed exercise dates |

### 1.3 Bermuda put — exercise date handling

Allowed exercise dates are mapped to the nearest time-step index:

$$
k_j = \mathrm{round}(t_j / \Delta t), \quad k_j \in \{0, \ldots, N\}
$$

At step $i$: early exercise is applied if $i \in \{k_1, \ldots, k_m, N\}$, otherwise only the
hold value is kept. $t = T$ (step $N$) is always an exercise date.

### 1.4 Configuration

| Parameter | European/American | Bermudan |
|-----------|-------------------|---------|
| $N$ | 4000 (default) | 2000 |
| Rationale | High accuracy reference | One exercise date; 2000 is sufficient |

---

## 2. Data flow

```
phase3_training.py
  ├── bermuda_put_binomial_tree(S, K, r, sigma, T, [t1], N=2000)
  │     → reference price at each s in eval grid
  └── european_put_binomial_tree(S, K, r, sigma, T, N=2000)
        → cross-check against analytical Ve(s, 0)
```

BT prices are computed on a grid of 81 asset prices $s \in [60, 140]$ at $t = 0$ and
stored as NumPy arrays. They are not persisted to disk — recomputed each run (takes
$\sim 5$ s for 81 × $N = 2000$ evaluations).

---

## 3. Validation protocol

ETCNN predictions are evaluated against the BT reference using:

**Relative $L^2$ error:**

$$
\varepsilon_{L^2} = \frac{\sqrt{\frac{1}{n}\sum_{i=1}^n (V_{\theta}(s_i, 0) - V_{BT}(s_i))^2}}
                         {\sqrt{\frac{1}{n}\sum_{i=1}^n V_{BT}(s_i)^2}}
$$

**Mean absolute error:**

$$
\varepsilon_{MAE} = \frac{1}{n}\sum_{i=1}^n \lvert V_\theta(s_i, 0) - V_{BT}(s_i) \rvert
$$

For the European put, the analytical Black–Scholes price $V^e(s, t)$ is used instead of
the binomial tree, giving exact error measurement.

### Financial consistency check

The Bermudan price must satisfy:

$$
V^{\text{European}}(s, 0) \;\le\; V^{\text{Bermudan}}(s, 0) \;\le\; V^{\text{American}}(s, 0)
$$

This ordering is verified at $s = K = 100$ in the training script and reported in the
joint summary.

---

## 4. File layout

```
learning_option_pricing/solvers/
├── __init__.py                  # exports all four BT functions
└── binomial_tree.py             # CRR implementation
        european_put_binomial_tree(S, K, r, sigma, T, N, q)
        american_put_binomial_tree(S, K, r, sigma, T, N, q)
        american_call_binomial_tree(S, K, r, sigma, T, N, q)
        bermuda_put_binomial_tree(S, K, r, sigma, T, exercise_dates, N)

data/phase3_training/<timestamp>/
├── training.log                 # full run log with BT prices
├── plotB5_price_comparison.png  # ETCNN_B vs BT vs European at t=0
└── plotB7_error_vs_bt.png       # pointwise error vs BT
```
