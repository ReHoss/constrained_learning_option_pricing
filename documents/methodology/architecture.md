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

### 2.2 Bermuda options — interpolation of $g_2$

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

See also: `BermudaETCNN` stub in `learning_option_pricing/models/etcnn.py`.

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
\tilde{u}_{NN}(s,t) = g_1(s,t)\cdot u_{NN}(s/K, t) + g_2(s,t)
$$

where $u_{NN}(s/K, t) = W^{out}\cdot g^{(M+1,0)}(s/K, t) + b^{out}$ is the raw ResNet output. At $t = T$, $g_1(s, T) = 0$ exactly, so $\tilde{u}_{NN}(s, T) = g_2(s, T) = \Phi(s)$ regardless of the network weights.

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
| $\mathcal{L}_f$ | $\frac{1}{N_f} \sum_{i=1}^{N_f} \left[\mathcal{F}(\tilde{u}_{NN})(s_i, t_i)\right]^2$ | Mean squared PDE residual at interior collocation points |
| $\mathcal{L}_{tc}$ | $\frac{1}{N_{tc}} \sum_{j=1}^{N_{tc}} \left[\tilde{u}_{NN}(s_j, T) - \Phi(s_j)\right]^2$ | Mean squared terminal error (identically zero for ETCNN) |

Weights: $\lambda_f = 20$, $\lambda_{tc} = 1$ (Section 3.4).

### 4.2 American Option Loss (Linear Complementarity)

The composite loss for American options enforces the continuous early-exercise boundary via three penalty terms (ETCNN naturally eliminates the terminal condition term):

$$
\mathcal{L}_{\text{American}}(\theta) = \lambda_{bs}\,\mathcal{L}_{bs} + \lambda_{tv}\,\mathcal{L}_{tv} + \lambda_{eq}\,\mathcal{L}_{eq}
$$

| Term | Formula | Enforces |
|------|---------|---------|
| $\mathcal{L}_{bs}$ | $\text{mean}(\max(\mathcal{F}(u_\theta), 0)^2)$ | BSM operator $\mathcal{F}(V) \le 0$ |
| $\mathcal{L}_{tv}$ | $\text{mean}(\max(-\mathcal{TV}(u_\theta), 0)^2)$ | Time value $V - \Phi \ge 0$ |
| $\mathcal{L}_{eq}$ | $\text{mean}((\mathcal{F}(u_\theta)\cdot\mathcal{TV}(u_\theta))^2)$ | Complementarity $\mathcal{F}\cdot\mathcal{TV} = 0$ |

Default weights: $\lambda_{bs} = \lambda_{tv} = \lambda_{eq} = 1$.
Code: `learning_option_pricing/pricing/loss.py`.

### 4.3 Bermudan Option Loss (Piecewise European)

Unlike American options, Bermudan options cannot be exercised continuously. Between any two discrete exercise dates $t_{k-1}$ and $t_k$, the option behaves exactly like a European option. There is no early exercise opportunity within the open interval $(t_{k-1}, t_k)$.

Therefore, **the Bermudan loss function is identical to the European loss function** applied piecewise to each sub-interval:

$$
\mathcal{L}_{\text{Bermudan}}^{(k)}(\theta) = \lambda_f \, \mathcal{L}_f^{(k)}(\theta) + \lambda_{tc} \, \mathcal{L}_{tc}^{(k)}(\theta)
$$

Where:
- $\mathcal{L}_f^{(k)}$ is the mean squared PDE residual $\mathcal{F}(\tilde{u}_{NN}) = 0$ evaluated on collocation points sampled within $t \in [t_{k-1}, t_k]$.
- $\mathcal{L}_{tc}^{(k)}$ is the terminal condition loss at $t = t_k$. For the ETCNN, this is identically zero because the network architecture enforces $\tilde{u}_{NN}(s, t_k) = V(s, t_k)$ via the interpolated $g_2$ function.

**Crucial Distinction:** The complementarity terms ($\mathcal{L}_{bs}$, $\mathcal{L}_{tv}$, $\mathcal{L}_{eq}$) are **not** used during the training of a Bermudan sub-interval. The early exercise condition is instead enforced discretely at the boundaries $t_k$ when constructing the intermediate terminal condition $V(s, t_k) = \max(\Phi(s), V^{\text{hold}}(s, t_k))$ for the subsequent backward step.

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
