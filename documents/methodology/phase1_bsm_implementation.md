# Phase 1 — BSM Mathematical Components: Implementation Manual

> **Math rendering:** open in a Markdown renderer with KaTeX/MathJax support
> (e.g. VSCode + Markdown Preview Enhanced, Obsidian, or any GitHub-flavoured renderer).

This document is the implementation reference for the core BSM mathematical components
introduced in Phase 1.  It maps every symbol in Zhang, Guo, Lu (2026) to the exact
function and file in this codebase.

**Key reference:**
Zhang, W., Guo, Y., Lu, B. — *Exact Terminal Condition Neural Network for American
Option Pricing Based on the Black–Scholes–Merton Equations*,
J. Comput. Appl. Math. **480** (2026) 117253.
DOI: [10.1016/j.cam.2025.117253](https://doi.org/10.1016/j.cam.2025.117253)

**Parameters used throughout Phase 1 (Section 4.1.2 of the paper):**

$$K = 100,\quad r = 0.02,\quad \sigma = 0.25,\quad T = 1,\quad q = 0$$

---

## 1. Payoff functions

For a put option with strike $K$:

$$\Phi_{\text{put}}(s) = (K - s)^+ = \max(K - s,\, 0)$$

For a call option with strike $K$:

$$\Phi_{\text{call}}(s) = (s - K)^+ = \max(s - K,\, 0)$$

Both are vectorised over batches of underlying prices $s$.

**Code:** `learning_option_pricing/pricing/terminal.py`

```python
payoff_put(s, K)   # -> torch.Tensor, same shape as s
payoff_call(s, K)  # -> torch.Tensor, same shape as s
```

---

## 2. BSM PDE operator

For a single-asset option with continuous dividend yield $q$ (Eq. 2 of the paper):

$$\mathcal{F}(V)(s,t)
  = \frac{\partial V}{\partial t}
  + \frac{1}{2}\sigma^2 s^2 \frac{\partial^2 V}{\partial s^2}
  + (r - q)\,s\,\frac{\partial V}{\partial s}
  - r V$$

The American BSM complementarity conditions require $\mathcal{F}(V) \le 0$ everywhere,
with equality in the continuation region.

**Implementation note:** derivatives are computed via `torch.autograd.grad` so that the
operator is differentiable and can be embedded in the training loss.

**Code:** `terminal.bsm_operator(V, s, t, r, q, sigma)` — requires `s` and `t` to be
leaf tensors with `requires_grad=True`.

---

## 3. Time value operator

$$\mathcal{TV}(V)(s,t) = V(s,t) - \Phi(s)$$

The time value of an American option must satisfy $\mathcal{TV}(V) \ge 0$ at all times
(otherwise immediate exercise is profitable — arbitrage).

**Code:** `terminal.time_value(V, s, K, option_type)`

---

## 4. European put price — Black-Scholes formula (Eq. 16)

Let $\tau = T - t$ denote time to maturity.  Define the log-moneyness terms:

$$\tilde{d}_1(s,\tau,K)
  = -\frac{1}{\sigma\sqrt{\tau}}\!\left(\ln\frac{s}{K} + \left(r + \frac{\sigma^2}{2}\right)\tau\right)$$

$$\tilde{d}_2(s,\tau,K)
  = -\frac{1}{\sigma\sqrt{\tau}}\!\left(\ln\frac{s}{K} + \left(r - \frac{\sigma^2}{2}\right)\tau\right)$$

The European put price is then:

$$V^e(s,t) = K e^{-r\tau}\,N\!\left(\tilde{d}_2\right) - s\,N\!\left(\tilde{d}_1\right)$$

where $N(\cdot)$ is the standard normal CDF.

**Singular limit:** as $\tau \to 0$, the terms $\tilde{d}_1$ and $\tilde{d}_2$ diverge.
A floor $\tau_{\varepsilon} = 10^{-8}$ is applied before all divisions:

$$\tau_{\text{safe}} = \max(\tau,\; 10^{-8})$$

This is validated in Phase 1 Plot 8 — no NaN or Inf was observed for $\tau$ as small as $10^{-8}$.

**Code:**
```python
black_scholes_put(s, K, r, sigma, tau)   # V^e(s, t)
_d_tilde_1(s, tau, K, r, sigma)
_d_tilde_2(s, tau, K, r, sigma)
```

**Verified:** $V^e(s, T) = \Phi(s)$ to machine precision (max error = 0) — Plot 3.

---

## 5. Taylor expansion of $V^e$ — exact terminal function $g_2$

### 5.1 Midpoint term

$$\tilde{d}_0(s,\tau,K)
  = \frac{1}{2}\!\left(\tilde{d}_1 + \tilde{d}_2\right)
  = -\frac{1}{\sigma\sqrt{\tau}}\!\left(\ln\frac{s}{K} + r\tau\right)$$

### 5.2 Zeroth-order term $V_1^e$

$$V_1^e(s,t) = N\!\left(\tilde{d}_0\right)\cdot\left(K e^{-r\tau} - s\right)$$

This satisfies the terminal condition but **does not** capture the $\sqrt{\tau}$
singularity near expiry.  At $s = K$ as $\tau \to 0$:
$V_1^e(K, T) \to N(0)\cdot 0 = 0$, so it collapses to zero instead of tracking
$V^e \sim K\sigma\sqrt{\tau}/\sqrt{2\pi}$.

### 5.3 First-order term $V_2^e$ — singularity correction

$$V_2^e(s,t)
  = \frac{\sigma\sqrt{\tau}}{2\sqrt{2\pi}}\,
    e^{-\tilde{d}_0^2/2}\,
    \left(K e^{-r\tau} + s\right)$$

This term captures the at-the-money approximation.  At $s = K$, $\tau \to 0$:

$$V_2^e(K, t) \;\to\; \frac{1}{\sqrt{2\pi}}\,K\sigma\sqrt{\tau}$$

which matches the known at-the-money limit $V^e(K,t) \approx K\sigma\sqrt{\tau}/\sqrt{2\pi}$
(Brenner & Subrahmanyan 1988, cited in the paper as [47]).

### 5.4 Combined exact terminal function

$$g_2(s,t) = V_1^e(s,t) + V_2^e(s,t)$$

**Properties of $g_2$:**

| Property | Satisfied? |
|----------|-----------|
| $g_2(s, T) = \Phi(s)$ | Yes — verified to machine precision |
| Non-differentiable at $(K, T)$, smooth elsewhere | Yes |
| Captures $\sqrt{\tau}$ singularity near $t=T$ | Yes (via $V_2^e$) |
| Less expensive than full $V^e$ | Yes — avoids evaluating $\text{erfc}$ twice |

**Verified:** $g_2(s, T) = \Phi(s)$ to machine precision — Plot 3.
Gap $\max|V^e - g_2| \approx 0.057$ (residual for NN to learn) — Plot 7.

**Code:**
```python
european_put_ve1(s, K, r, sigma, tau)   # V_1^e
european_put_ve2(s, K, r, sigma, tau)   # V_2^e
g2_american_put(s, K, r, sigma, tau)    # g_2 = V_1^e + V_2^e
```

---

## 6. Terminal vanishing factor $g_1$

$$g_1(s,t) = T - t$$

This satisfies $g_1(s, T) = 0$, ensuring the trial solution $\tilde{u}_{NN}(s,t) = g_1 u_{NN} + g_2$ automatically satisfies the terminal condition regardless of the network output $u_{NN}$.

**Code:** `terminal.g1_linear(T, t)`

---

## 7. Trial solution (ETCNN output)

$$\tilde{u}_{NN}(s,t) = g_1(s,t)\cdot u_{NN}(s,t) + g_2(s,t)$$

where $u_{NN}$ is the raw ResNet output.  By construction:

$$\tilde{u}_{NN}(s,T) = 0 \cdot u_{NN}(s,T) + g_2(s,T) = \Phi(s)$$

so the terminal condition is satisfied exactly for any network weights.

---

## 8. Loss function (Section 2.2.2)

$$\mathcal{L}(\theta)
  = \lambda_{bs}\,\mathcal{L}_{bs}(\theta)
  + \lambda_{tv}\,\mathcal{L}_{tv}(\theta)
  + \lambda_{eq}\,\mathcal{L}_{eq}(\theta)$$

### BSM constraint loss

$$\mathcal{L}_{bs}(\theta)
  = \frac{1}{N_{bs}}\sum_{i=1}^{N_{bs}}
    \left[\max\!\left(\mathcal{F}(u_\theta(s^i, t^i)),\; 0\right)\right]^2$$

Penalises any violation of $\mathcal{F}(V) \le 0$.

### Time-value constraint loss

$$\mathcal{L}_{tv}(\theta)
  = \frac{1}{N_{tv}}\sum_{i=1}^{N_{tv}}
    \left[\max\!\left(-\mathcal{TV}(u_\theta(s^i, t^i)),\; 0\right)\right]^2$$

Penalises any violation of $V \ge \Phi$ (non-negative time value).

### Complementarity equality loss

$$\mathcal{L}_{eq}(\theta)
  = \frac{1}{N_{eq}}\sum_{i=1}^{N_{eq}}
    \left[\mathcal{F}(u_\theta(s^i, t^i))\cdot\mathcal{TV}(u_\theta(s^i, t^i))\right]^2$$

Enforces that at least one inequality holds with equality at every point.

**Default weights:** $\lambda_{bs} = \lambda_{tv} = \lambda_{eq} = 1$.

**Code:**
```python
loss_bs(F_u)           # learning_option_pricing/pricing/loss.py
loss_tv(TV_u)
loss_eq(F_u, TV_u)
composite_loss(F_u, TV_u, lam_bs=1., lam_tv=1., lam_eq=1.)
```

---

## 9. Phase 1 scalar validation results

Run: `experiments/python_scripts/exp1/phase1_bsm_validation.py`

| Check | Value | Expected |
|-------|-------|----------|
| $\max_s \lvert V^e(s,T) - \Phi(s)\rvert$ | $0$ | $< 10^{-6}$ |
| $\max_s \lvert g_2(s,T) - \Phi(s)\rvert$ | $0$ | $< 10^{-6}$ |
| $V^e(K{=}100,\; t{=}0.5)$ | $6.5218$ | $\approx 6$–$7$ |
| $g_2(K{=}100,\; t{=}0.5)$ | $6.5310$ | close to above |
| NaN/Inf at $\tau = 10^{-8}$ | None | None |

---

## 10. Symbol table

| Symbol | Description | Code |
|--------|-------------|------|
| $s$ | Underlying asset price | `s: torch.Tensor` |
| $t$ | Current time | `t: torch.Tensor` |
| $\tau = T - t$ | Time to maturity | `tau: torch.Tensor` |
| $K$ | Strike price | `K: float` |
| $r$ | Risk-free rate | `r: float` |
| $\sigma$ | Volatility | `sigma: float` |
| $q$ | Continuous dividend yield | `q: float` |
| $\Phi(s)$ | Payoff function | `payoff_put` / `payoff_call` |
| $\mathcal{F}(V)$ | BSM PDE operator (Eq. 2) | `bsm_operator` |
| $\mathcal{TV}(V)$ | Time value $V - \Phi$ | `time_value` |
| $\tilde{d}_1, \tilde{d}_2$ | Log-moneyness terms | `_d_tilde_1`, `_d_tilde_2` |
| $\tilde{d}_0$ | Midpoint of $\tilde{d}_1, \tilde{d}_2$ | `_d_tilde_0` |
| $V^e(s,t)$ | European put price (Eq. 16) | `black_scholes_put` |
| $V_1^e(s,t)$ | Zeroth-order Taylor term | `european_put_ve1` |
| $V_2^e(s,t)$ | First-order Taylor term ($\sqrt{\tau}$ singularity) | `european_put_ve2` |
| $g_1(s,t)$ | Terminal vanishing factor | `g1_linear` |
| $g_2(s,t)$ | Exact terminal function | `g2_american_put` |
| $\tilde{u}_{NN}$ | Trial solution (ETCNN output) | `ETCNN.forward` |
| $\mathcal{L}_{bs}$ | BSM penalty loss | `loss_bs` |
| $\mathcal{L}_{tv}$ | Time-value penalty loss | `loss_tv` |
| $\mathcal{L}_{eq}$ | Complementarity loss | `loss_eq` |
