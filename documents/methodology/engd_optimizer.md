# Empirical Natural Gradient Descent (ENGD) for PINNs and VPINNs

This document describes the PyTorch port of the Empirical Natural Gradient
Descent optimizer of Zeinhofer et al. (ICML 2023), with two specialised
variants:

* `ENGDOptimizer` — for **strong-form** PINNs (pointwise PDE residual);
* `VPINNENGDOptimizer` — for **variational / weak-form** PINNs (residuals
  integrated against test functions).

Reference (JAX implementation):
<https://github.com/MariusZeinhofer/Natural-Gradient-PINNs-ICML23>

---

## 1. Mathematical background

### 1.1 General setup

Suppose the loss is

$$
\mathcal{L}(\theta) = \frac{1}{M}\sum_{m=1}^{M} f_m(\theta)^2,
$$

where each *measurement* $f_m(\theta)$ is a scalar quantity that depends on
the parameters $\theta \in \mathbb{R}^{n_\theta}$. The **empirical Gram
matrix** of the measurements is

$$
G(\theta) = \frac{1}{M}\sum_{m=1}^{M} J_m(\theta)\, J_m(\theta)^\top
+ \varepsilon I,
\qquad J_m(\theta) = \nabla_\theta f_m(\theta) \in \mathbb{R}^{n_\theta}.
$$

The **natural gradient direction** $\delta$ solves

$$
G(\theta)\,\delta = \nabla_\theta \mathcal{L}(\theta),
$$

and parameters are updated by $\theta \leftarrow \theta - \alpha\delta$
with $\alpha$ chosen by **grid line search** over
$\{\alpha_0 \cdot 2^{-k}\}_{k=0}^{K-1}$.

When $\varepsilon = 0$, $\delta$ is exactly **twice the Gauss–Newton step**
(the factor 2 is absorbed by the line search).

### 1.2 Strong-form PINN (Black–Scholes)

For the BSM operator
$F[u_\theta](s,t) = \partial_t u + \tfrac{1}{2}\sigma^2 s^2 \partial_s^2 u
+ (r-q)\,s\,\partial_s u - r\,u$ and a terminal residual
$\mathrm{TC}_\theta(s) = u_\theta(s,T) - \varphi(s)$, the measurements are

$$
f_m \in \big\{F[u_\theta](s_i, t_i)\big\}_i \cup \big\{u_\theta(s_j, T) - \varphi(s_j)\big\}_j,
$$

so the Gram matrix splits naturally into two contributions:

$$
G(\theta) = \frac{\lambda_F}{N_F}\sum_i J_F(x_i) J_F(x_i)^\top
           + \frac{\lambda_{TC}}{N_{TC}}\sum_j J_{TC}(x_j) J_{TC}(x_j)^\top
           + \varepsilon I.
$$

### 1.3 Variational PINN (VPINN)

In log-moneyness coordinates $(\tau, x) = (T-t, \ln S/K)$ the BSM PDE
reads $u_\tau = \tfrac{\sigma^2}{2}u_{xx} + \mu u_x - r u$ with
$\mu = r - q - \sigma^2/2$. Multiplying by a test function $\phi_k$
that vanishes at $\pm x_{\max}$ and integrating by parts:

$$
R_{i,k}(\theta) = \int_{-x_{\max}}^{x_{\max}}
\Big[\partial_\tau u_\theta\,\phi_k
+ \tfrac{\sigma^2}{2}\,\partial_x u_\theta\,\phi'_k
- \mu\,\partial_x u_\theta\,\phi_k
+ r\,u_\theta\,\phi_k\Big]\mathrm{d}x.
$$

The measurements are now $\{R_{i,k}(\theta)\}_{i=1..N_\tau,\,k=1..K}$
(a *vector* of length $N_\tau K$ at each parameter setting). The Gram
matrix is

$$
G_\text{VPINN}(\theta) = \frac{1}{N_\tau K}\,J_R^\top J_R + \varepsilon I,
\qquad J_R \in \mathbb{R}^{(N_\tau K)\times n_\theta}.
$$

Boundary contributions vanish thanks to integration by parts, so there
is **no separate terminal-Gram term** in the variational case.

---

## 2. Implementation

### 2.1 Module map

```
learning_option_pricing/optimizers/
├── __init__.py                       # public API
├── natural_gradient.py               # core + strong-form PINN ENGD
└── natural_gradient_vpinn.py         # variational PINN ENGD
```

### 2.2 Core building blocks (in `natural_gradient.py`)

| Symbol | Role |
|---|---|
| `flat_params(model)`, `set_flat_params(model, vec)` | Round-trip parameter vector ↔ model |
| `flat_grad(model)` | Concatenated `p.grad` after `loss.backward()` |
| `_bsm_scalar(params_dict, model, s, t, r, q, sigma)` | Scalar BSM residual at one point — **functional** (uses `torch.func.grad`) so it composes with `jacrev` |
| `_tc_scalar(params_dict, model, s, t)` | Scalar terminal residual at one point |
| `compute_jacobians(model, s_f, t_f, s_tc, t_tc, r, q, sigma)` | Vectorised PDE & TC Jacobians via `vmap(jacrev(...))` |
| `measurement_jacobian(fn, params_dict, *args)` | Generic Jacobian builder for any *vector-valued* measurement function (used by VPINN) |
| `solve_cg(g, J_F, J_TC, lam_f, lam_tc, reg, n_iters, tol)` | Conjugate Gradient solver — implicit Gram-vector products, no $G$ ever materialised |
| `grid_line_search(model, loss_fn, delta, n_steps, step_max)` | Grid search over $\{α_0 \cdot 2^{-k}\}$, restores parameters on exit |
| `ENGDOptimizer` | Strong-form PINN driver: assembles Jacobians, solves CG, runs line search, updates |
| `train_model_engd(...)` *(in `phase3_training.py`)* | Drop-in replacement for `train_model` using ENGD |

### 2.3 VPINN-specific (in `natural_gradient_vpinn.py`)

| Symbol | Role |
|---|---|
| `_vpinn_residuals(params_dict, model, tau_batch, x_nodes, phi_w, dphi_w, sigma, mu, r)` | **Functional** weak residual vector $R_{i,k}$ (composable with `jacrev`); equivalent to `VPINNLoss.forward()` but returns the un-squared residuals |
| `vpinn_jacobian(model, vpinn_loss, tau_batch)` | Standalone Jacobian builder for inspection / unit tests |
| `VPINNENGDOptimizer` | Variational PINN driver (analogous to `ENGDOptimizer`) |

### 2.4 Why functional residuals?

The classical `bsm_operator` and `VPINNLoss.forward` both use
`torch.autograd.grad`, which is *not* composable with the
`torch.func.{vmap, jacrev}` transforms.

The functional re-implementations
(`_bsm_scalar`, `_vpinn_residuals`) replace `torch.autograd.grad` with
`torch.func.grad`, allowing the entire Jacobian to be obtained as

```python
J_F  = vmap(jacrev(_bsm_scalar, argnums=0))(params_dict, ...)   # PINN
J_R  = jacrev(_vpinn_residuals, argnums=0)(params_dict, ...)    # VPINN
```

with no Python-level loops over collocation points.

The unit tests
(`test_bsm_scalar_matches_classical_operator`,
 `test_functional_residuals_match_classical_loss`) verify that the
functional and classical implementations agree to machine precision.

### 2.5 Memory-efficient Conjugate Gradient

For the standard ResNet ($n_\theta \sim 2 \cdot 10^4$), the full Gram
$G \in \mathbb{R}^{n_\theta \times n_\theta}$ would take $\approx 1.6$ GB
in float64. The CG solver evaluates only matrix-vector products

$$
G v = \frac{\lambda_F}{N_F}\,J_F^\top (J_F v) + \frac{\lambda_{TC}}{N_{TC}}\,J_{TC}^\top (J_{TC} v) + \varepsilon v,
$$

each of which costs $O((N_F + N_{TC})\,n_\theta)$.

### 2.6 Defensive guards

* `_gram_matvec` skips the $J_{TC}$ contribution when `N_TC == 0` or
  `lam_tc == 0` (avoids division by zero — important for ETCNN models
  where the terminal Gram vanishes by construction).
* `solve_cg` exits early if $p^\top G p \le 0$ or non-finite (would mean
  $G$ is not SPD — usually caused by `reg` being too small).
* `grid_line_search` skips NaN/Inf candidates and falls back to the
  smallest step if every trial diverges.

---

## 3. Validation

### 3.1 Unit tests (`test/optimizers/`)

Run with `pytest -xvs test/optimizers/` (all 13 tests pass).

| Test | What is checked | Tolerance |
|---|---|---|
| `test_bsm_scalar_matches_classical_operator` | Functional BSM residual ≡ `bsm_operator` | rel `1e-9` |
| `test_jacobian_matches_autograd` | `compute_jacobians` rows ≡ per-point `torch.autograd.grad` | rel `1e-9` |
| `test_cg_solves_linear_system` | `solve_cg` ≡ `torch.linalg.solve` on a small SPD $G$ | rel `1e-8` |
| `test_cg_converges_in_at_most_n_steps` | CG converges in $\le n_\theta$ iters (exact arithmetic property) | rel `1e-6` |
| `test_cg_handles_zero_terminal_jacobian` | CG handles $N_{TC}=0$ without division by zero | finite output |
| `test_line_search_picks_best_step` | Grid line search returns the exact minimiser on a quadratic | rel `1e-9` |
| `test_line_search_restores_params_on_exit` | Parameters restored after line search | exact |
| `test_engd_step_decreases_loss_on_fixed_batch` | ENGD step on a fixed batch does not increase the loss | strict `≤` |
| `test_helpers_round_trip` | `flat_params`/`set_flat_params` are inverses | exact |
| `test_flat_grad_matches_concatenated_grads` | `flat_grad` = concatenated `p.grad` | exact |
| `test_functional_residuals_match_classical_loss` (VPINN) | `_vpinn_residuals.pow(2).mean()` ≡ `VPINNLoss(...)` | rel `1e-9` |
| `test_vpinn_jacobian_matches_autograd` (VPINN) | `vpinn_jacobian` rows ≡ per-residual `torch.autograd.grad` | rel `1e-9` |
| `test_vpinn_engd_step_does_not_increase_loss` | VPINN-ENGD step does not increase the loss | strict `≤` |

### 3.2 Convergence experiments (`experiments/python_scripts/exp_engd/`)

#### European put (strong form)

`convergence_european_put.py` — fixed deterministic grid (15×15 in $(s,t)$,
64 terminal points), normalised input $s \to s/K$, `ResNet(M=2, L=2, n=16)`,
$n_\theta = 1153$.

Result of a long run (`--iters-engd 100 --iters-adam 30000`):

| Optimizer | iters | wall time | loss | $L^2$ abs error |
|---|---:|---:|---:|---:|
| Adam (lr 1e-2) | 30 000 | 247 s | $1.7 \cdot 10^{-2}$ | $0.34$ |
| ENGD (reg $10^{-3}$, cg 200) | 100 | **16 s** | $\mathbf{4.1 \cdot 10^{-5}}$ | $\mathbf{0.13}$ |

ENGD reaches a **400× lower loss** in **15× less wall time** and a
**2.5× better $L^2$ error**. By iteration count, ENGD converges in
~22 steps to a level Adam reaches around iteration 3000 — i.e. roughly
**100×** speedup *per iteration*.

#### VPINN (weak form)

`convergence_vpinn.py` — `TanhMLP(hidden=16, depth=3)`, $n_\theta = 609$,
$N_\tau=10$ time points, $K=6$ test functions, $n_\text{quad}=20$. The
loss is the bare VPINN weak residual (no IC term — the trivial solution
$u\equiv 0$ satisfies the homogeneous PDE on the interior).

| Optimizer | iters | wall time | final loss |
|---|---:|---:|---:|
| Adam (lr 1e-2) | 3 000 | 4.3 s | $6.5 \cdot 10^{-10}$ |
| ENGD (reg $10^{-3}$, cg 100) | 40 | 4.2 s | $4.8 \cdot 10^{-8}$ |

Per iteration, the VPINN ENGD descends from $6.5 \cdot 10^{-3}$ to
$3.6 \cdot 10^{-5}$ in **one** step (180× reduction), then continues
geometrically with $\alpha = 1.0$ accepted by the line search. CG
converges to machine precision (`cg_residual_norm ~ 1e-8`).

---

## 4. Practical guidance

### 4.1 Hyperparameter recipe (validated on European put)

| Parameter | Recommended | Notes |
|---|---|---|
| `n_gram` | match the loss batch size | A *too small* Gram batch poisons the preconditioner and slows convergence dramatically (observed: 16× more wall time for the same loss when `n_gram` was 64 vs 225). |
| `reg` ($\varepsilon$) | $10^{-3}$ | Below $10^{-4}$ the CG often diverges (residual ~10²); above $10^{-2}$ the natural-gradient approximation is over-damped and behaves like vanilla SGD. |
| `cg_iters` | $\sim n_\theta / 5$ | CG needs many iterations when the Gram is ill-conditioned; cheap to do because each iter is just two matrix-vector products. |
| `ls_step_max` | 1.0 | If the line search constantly accepts the largest step, increase to e.g. 2.0; if it always picks tiny steps, decrease the gradient scale (rescale `lam_f`/`lam_tc`). |
| `ls_steps` | 30 | Geometric grid $\alpha_0 \cdot 2^{-k}$ down to $\sim 10^{-9}$. |

### 4.2 Deterministic vs stochastic collocation

The original JAX paper uses **deterministic** quadrature points: the
Gram and the loss are evaluated on the *same fixed grid* throughout
training. Resampling at each step injects variance into both the
gradient and the Gram, and the natural-gradient direction loses meaning.

The convergence scripts in this repo follow the deterministic-grid
convention. If stochastic sampling is unavoidable (very large domains,
adaptive sampling), use a much larger `n_gram` and increase `reg` to
compensate.

### 4.3 ETCNN models and `lam_tc = 0`

For an ETCNN ansatz that *exactly* enforces the terminal condition
($g_1(s,T) = 0$ and $g_2(s,T) = \varphi(s)$), the terminal Jacobian
$J_{TC}$ vanishes identically. Set `lam_tc = 0` (or `tc_enforced=True`
in `train_model_engd`) so the CG solver skips the $J_{TC}$ contribution.

### 4.4 Limitations

* `BermudaETCNN.forward_pde()` (the operator-bypass forward used to avoid
  catastrophic cancellation in singularity extraction) is not yet wired
  into the Gram computation; the standard `forward()` is used.
* Models with stochastic layers (Dropout, BatchNorm with running stats)
  must be put in `eval()` before calling `step()`. The optimizer does
  this automatically inside the line search but not during the Jacobian
  build.
* Compute cost per ENGD step scales as $O((N_F + N_{TC})\,n_\theta)$ for
  the Jacobians, dominated by second-order differentiation through the
  network. On CPU, expect a few hundred milliseconds per step for
  $n_\theta \sim 10^3$ and tens of seconds for $n_\theta \sim 10^4$. GPU
  acceleration helps but the implementation is currently CPU-tested.

---

## 5. Usage examples

### 5.1 Strong-form PINN

```python
from learning_option_pricing.optimizers import ENGDOptimizer, flat_grad

engd = ENGDOptimizer(
    model, r=0.02, q=0.0, sigma=0.25,
    lam_f=1.0, lam_tc=10.0,
    reg=1e-3, cg_iters=200, ls_steps=30,
)

for it in range(n_iters):
    # standard backward
    model.zero_grad()
    loss, _, _ = compute_losses(model, s_f, t_f, s_tc, t_tc, ...)
    loss.backward()
    g = flat_grad(model)

    def loss_fn():
        s = s_f.detach().clone().requires_grad_(True)
        t = t_f.detach().clone().requires_grad_(True)
        l, _, _ = compute_losses(model, s, t, s_tc, t_tc, ...)
        return l

    info = engd.step(g, s_f.detach(), t_f.detach(),
                     s_tc, t_tc, loss_fn)
```

A complete training loop is provided by `train_model_engd()` in
`experiments/python_scripts/exp1/phase3_training.py`.

### 5.2 Variational PINN

```python
from learning_option_pricing.optimizers import VPINNENGDOptimizer, flat_grad
from learning_option_pricing.vpinn import VPINNLoss

vpinn_loss = VPINNLoss(sigma=0.25, r=0.02, q=0.0,
                       x_max=1.5, K_test=6, n_quad=20)
tau_batch = torch.linspace(0.05, 0.95, 10)

engd = VPINNENGDOptimizer(
    model, vpinn_loss,
    reg=1e-3, cg_iters=100, ls_steps=30,
)

for it in range(n_iters):
    model.zero_grad()
    L = vpinn_loss(model, tau_batch)
    L.backward()
    g = flat_grad(model)

    info = engd.step(g, tau_batch,
                     loss_fn=lambda: vpinn_loss(model, tau_batch))
```

A complete script is `experiments/python_scripts/exp_engd/convergence_vpinn.py`.

---

## 6. Files reference

| File | Purpose |
|---|---|
| `learning_option_pricing/optimizers/__init__.py` | Public API |
| `learning_option_pricing/optimizers/natural_gradient.py` | Strong-form ENGD + generic primitives |
| `learning_option_pricing/optimizers/natural_gradient_vpinn.py` | VPINN ENGD |
| `experiments/python_scripts/exp1/phase3_training.py::train_model_engd` | Strong-form ENGD training loop |
| `experiments/python_scripts/exp_engd/convergence_european_put.py` | Convergence study, European put |
| `experiments/python_scripts/exp_engd/convergence_vpinn.py` | Convergence study, VPINN sanity check |
| `test/optimizers/test_natural_gradient.py` | 10 unit tests for strong-form ENGD |
| `test/optimizers/test_natural_gradient_vpinn.py` | 3 unit tests for VPINN ENGD |
| `documents/methodology/engd_optimizer.md` | This document |
