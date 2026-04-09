# Numerical Solvers and Benchmarks

> Math rendering: open in a Markdown+KaTeX/MathJax renderer for rendered equations.

This document describes the reference numerical methods used to generate benchmark
solutions for option pricing, and their role in the validation pipeline.

For neural network architecture details, see [`architecture.md`](architecture.md).
For the training procedure, see [`implementation_phases.md`](implementation_phases.md).

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