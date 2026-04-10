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

This section describes the training procedure for the ETCNN on two toy problems: a European put (analytical ground truth) and a Bermudan put with one intermediate exercise date (binomial tree ground truth).

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
| Plot | Description |
|------|-------------|
| E1 | Loss curves ($\mathcal{L}_f$, $\mathcal{L}_{tc}$) for ETCNN and PINN |
| E2 | Predicted vs analytical price surface heatmaps |
| E3 | ETCNN pointwise error heatmap |
| E4 | PINN pointwise error heatmap |
| E5 | Slice comparison at $t = 0.25, 0.5, 0.75$ |
| E6 | Greeks ($\Delta, \Gamma, \Theta$) at $t = 0$ compared to analytical |

### 3.3 Problem 2 — Bermudan put (one intermediate exercise date)

A Bermudan put with one intermediate exercise date $t_1 = 0.5$ is solved by piecewise backward induction over two European-type sub-problems.

**Two-stage backward solving:**

| Stage | Domain | Terminal condition | Network |
|-------|--------|--------------------|---------|
| A | $[t_1, T]$ | $\Phi(s) = (K - s)^+$ at $t = T$ | ETCNN$_A$ |
| B (intermediate) | — | $V(s, t_1) = \max\!\bigl(\Phi(s),\;\mathrm{ETCNN}_A(s, t_1)\bigr)$ | — |
| C+D | $[0, t_1]$ | $V(s, t_1)$ at $t = t_1$ | ETCNN$_B$ |

**Stage B — intermediate condition:** At $t = t_1$, the Bermudan holder compares immediate exercise $\Phi(s)$ with the continuation value $\mathrm{ETCNN}_A(s, t_1)$. The maximum defines the terminal condition for Stage D.

**Interpolation:** $V(s, t_1)$ is evaluated on a dense grid of 2,000 points and stored as a look-up table. As detailed in [`architecture.md`](architecture.md), this is interpolated using either a natural cubic spline ($C^2$) or a PCHIP interpolant ($C^1$, shape-preserving) to preserve the diffusion term $\frac{\partial^2 g_2}{\partial s^2}$. PCHIP is recommended near kinks (exercise boundaries) where the global cubic spline can overshoot.

**Terminal functions for Stage D:**
$$
g_1(s, t) = t_1 - t, \qquad g_2(s, t) = V(s, t_1)
$$

Note: $g_2$ is constant in $t$ and does not capture $\sqrt{\tau}$ singularities. This is acceptable because $V(s, t_1)$ is already a smooth continuation value (not a raw payoff).

**Exercise boundary:**
The exercise boundary $s^*$ at $t_1$ is the asset price where $\Phi(s) = \mathrm{ETCNN}_A(s, t_1)$, found by sign-change detection. For these parameters, $s^* < K = 100$.

**Plots produced:**
| Plot | Description |
|------|-------------|
| B1 | Intermediate terminal condition: hold, exercise, and $V(s, t_1)$ at $t_1$ |
| B2 | Exercise boundary $s^*$ at $t_1$ |
| B3 | Stage A loss curves |
| B4 | Stage D loss curves |
| B5 | Price at $t = 0$: ETCNN$_B$, binomial tree, European |
| B6 | Full piecewise Bermudan price surface |
| B7 | Error vs binomial tree at $t = 0$ |
| B8 | Greeks ($\Delta, \Gamma, \Theta$) at $t = 0$ compared to European analytical |
| B9 | Spatial distribution of PDE residual at $t_1^-$ |
| B10 | Neuron weight magnitude distribution in ETCNN$^{(A)}$ |

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

### 4.6 CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--iters N [M]` | `50000` | Training iterations. One value = same for all stages; two values = Stage A, Stage B. |
| `--interp {cubic,pchip,linear}` | `cubic` | Interpolation method for $V(s, t_1)$. `pchip` is shape-preserving ($C^1$). |
| `--device {auto,cuda,cpu}` | `auto` | Compute device. |
| `--bermudan-only` | off | Skip European problem. |
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
| $V(s, t_1)$ | Intermediate terminal condition | `phase3_training.bermudan_problem` — `v_interp_t1` |
| $\Delta, \Gamma, \Theta$ | Option Greeks computation | `phase3_training.compute_greeks_nn`, `compute_greeks_analytical` |
| BT reference | Bermudan binomial tree | `solvers.bermuda_put_binomial_tree` |
| $s^*$ | Exercise boundary at $t_1$ | Sign-change detection in `bermudan_problem` |
