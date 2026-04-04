# ETCNN Architecture for Bermuda Option Pricing

> Math rendering: open in a Markdown+KaTeX/MathJax renderer for rendered equations.

This document describes the neural network architecture and loss design for extending
ETCNN to Bermuda options.

Key reference:

* W. Zhang, Y. Guo, B. Lu — *Exact Terminal Condition Neural Network for American
  Option Pricing Based on the Black–Scholes–Merton Equations*, J. Comput. Appl. Math.
  480 (2026) 117253.

---

## 1. Problem formulation

<!-- Describe the BSM complementarity conditions for Bermuda options. -->

**TODO**

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

### Bermuda options (TODO)

Piecewise construction needed: at each exercise date $t_k$ the function must vanish, and between dates $g_2$ should use the relevant intrinsic value. See `BermudaETCNN` stub in `learning_option_pricing/models/etcnn.py`.

---

## 3. Network architecture

### ResNet backbone

$M$ residual blocks, each consisting of $L$ fully-connected layers of width $n$ with tanh activations and skip connections:

$$
g^{(m+1,0)}(x) = f_\theta^{(m,L)}(x) + g^{(m,0)}(x)
$$

Baseline configuration (Section 4, paper): $M=4$, $L=2$, $n=50$.

Code: `learning_option_pricing/models/resnet.py`.

### Input normalization

Asset prices $s$ are divided by $K$ (moneyness $s/K$) before entering the network. This preserves the homogeneity property $\alpha V(s,K,t) = V(\alpha s, \alpha K, t)$.

### Output modification for exact terminal conditions

The last linear layer is replaced by:

$$
f_\theta(x) = g_1(x)\cdot(W^{out}\cdot g^{(M+1,0)}(x) + b^{out}) + g_2(x)
$$

Code: `learning_option_pricing/models/etcnn.py` — `ETCNN`.

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
| $g_2(s,t)$ | exact terminal function | `terminal.g2_american_put` |
| $\mathcal{L}_{bs}$ | BSM penalty loss | `loss.loss_bs` |
| $\mathcal{L}_{tv}$ | time-value penalty loss | `loss.loss_tv` |
| $\mathcal{L}_{eq}$ | complementarity loss | `loss.loss_eq` |
