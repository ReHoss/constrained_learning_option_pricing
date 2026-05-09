# Empirical Natural Gradient Descent (ENGD) for PINNs

## Overview

The ENGD optimizer replaces the Adam update in the PINN training loop with a
second-order preconditioned step based on the **natural gradient** of the loss.
The implementation is a PyTorch port of the JAX code from:

> Zeinhofer, M. et al. *Natural Gradient PINNs*, ICML 2023.  
> <https://github.com/MariusZeinhofer/Natural-Gradient-PINNs-ICML23>

## Mathematical background

### Gram matrix

For a PINN with Black-Scholes PDE residual $F[u_\theta](s,t)$ and terminal
condition $u_\theta(s,T) - \varphi(s)$, the **Gram matrix** is

$$
G(\theta)
= \frac{\lambda_F}{N_F} \sum_{i=1}^{N_F} J_F(x_i)\, J_F(x_i)^\top
+ \frac{\lambda_{TC}}{N_{TC}} \sum_{j=1}^{N_{TC}} J_{TC}(x_j)\, J_{TC}(x_j)^\top
+ \varepsilon\, I
$$

where

- $J_F(x) = \partial_\theta F[u_\theta](x) \in \mathbb{R}^{n_\theta}$ is the
  Jacobian of the **raw PDE residual** (not the squared loss) w.r.t. parameters,
- $J_{TC}(x) = \partial_\theta u_\theta(x) \in \mathbb{R}^{n_\theta}$ is the
  Jacobian of the model output,
- $\varepsilon > 0$ is a Tikhonov regularisation term for numerical stability.

$G(\theta)$ defines a Riemannian metric on parameter space induced by the
function-space geometry of the PDE problem.

### Natural gradient and update

The **natural gradient direction** $\delta$ solves the linear system

$$
G(\theta)\,\delta = \nabla_\theta L(\theta),
$$

where $\nabla_\theta L$ is the standard (Euclidean) gradient.  The step size
$\alpha$ is chosen by a **grid line search** over the geometric sequence
$\{\alpha_0 \cdot 2^{-k}\}_{k=0}^{K}$, and the parameters are updated as

$$
\theta \;\leftarrow\; \theta - \alpha\,\delta.
$$

### Relationship to Gauss-Newton

When $\lambda_F = \lambda_{TC} = 1$ and $\varepsilon = 0$, the natural gradient
$\delta$ is exactly **twice the Gauss-Newton step**:

$$
\delta = G^{-1} \nabla L
= \left(\frac{1}{N} J^\top J\right)^{-1} \frac{2}{N} J^\top \mathbf{f}
= 2\,(J^\top J)^{-1} J^\top \mathbf{f},
$$

where $\mathbf{f}$ is the vector of residuals.  The factor of 2 is absorbed by
the line search.

## Implementation

### Module location

```
learning_option_pricing/optimizers/natural_gradient.py
```

### Key functions

| Symbol | Role |
|---|---|
| `_bsm_scalar(params_dict, model, s, t, r, q, sigma)` | Functional BSM residual $F[u_\theta](s,t)$ at a single point, using `torch.func.grad` for composability with `jacrev` |
| `compute_jacobians(model, s_f, t_f, s_tc, t_tc, r, q, sigma)` | Returns $J_F \in \mathbb{R}^{N_\text{gram} \times n_\theta}$ and $J_{TC}$ via `vmap(jacrev(...))` |
| `solve_cg(g, J_F, J_TC, lam_f, lam_tc, reg, ...)` | Conjugate Gradient solver for $G\delta = g$ using implicit matrix-vector products |
| `grid_line_search(model, loss_fn, delta, ...)` | Grid search over $\{\alpha_0 \cdot 2^{-k}\}$ |
| `ENGDOptimizer` | High-level class wrapping the above |
| `train_model_engd(...)` | Training loop (in `experiments/.../phase3_training.py`) |

### Why `torch.func.grad` instead of `torch.autograd.grad`

The existing `bsm_operator` uses `torch.autograd.grad` with `create_graph=True`
to compute $\partial V / \partial s$, $\partial^2 V / \partial s^2$,
$\partial V / \partial t$.  This is incompatible with `torch.func.jacrev`
because `jacrev` uses functional transforms that do not intercept
`torch.autograd.grad` calls.

The functional version `_bsm_scalar` recomputes the same derivatives using
`torch.func.grad`, which composes correctly with `jacrev` and `vmap`:

$$
\frac{\partial V}{\partial t} \;=\; \texttt{func\_grad}(\lambda\, t':\; u_\theta(s, t'))(t),
\quad
\frac{\partial^2 V}{\partial s^2} \;=\; \texttt{func\_grad}(\texttt{func\_grad}(\lambda\, s':\; u_\theta(s', t)))(s).
$$

The composition `vmap(jacrev(_bsm_scalar, argnums=0))` then computes the
per-sample Jacobians $J_F$ efficiently for a batch of collocation points.

### Memory-efficient Gram via CG

For the standard ResNet ($n_\theta \approx 20{,}600$), the full Gram matrix
$G \in \mathbb{R}^{n_\theta \times n_\theta}$ would require $\approx 1.7$ GB
in single precision.  The CG solver avoids this by only evaluating
**Gram-vector products**:

$$
G v = \frac{\lambda_F}{N_F} J_F^\top (J_F v)
    + \frac{\lambda_{TC}}{N_{TC}} J_{TC}^\top (J_{TC} v)
    + \varepsilon v,
$$

each of which costs $O(N_\text{gram} \cdot n_\theta)$.

### Grid line search

The step-size grid follows the JAX implementation: $\alpha_k = \alpha_0 \cdot 2^{-k}$
for $k = 0, \ldots, K{-}1$.  Default values are $\alpha_0 = 1.0$, $K = 30$,
giving a minimum step of $\approx 10^{-9}$.  The `loss_fn` callable passed to
`grid_line_search` must evaluate the PINN loss at the current parameter setting
without calling `.backward()`.

**Note:** `bsm_operator` uses `torch.autograd.grad` internally, so the
`loss_fn` must be evaluated under `torch.enable_grad()` (handled automatically
by `grid_line_search`).

## Usage

```python
from learning_option_pricing.optimizers import ENGDOptimizer, flat_grad

engd = ENGDOptimizer(
    model,
    r=0.02, q=0.0, sigma=0.25,
    lam_f=20.0, lam_tc=1.0,
    reg=1e-4,       # Tikhonov regularisation
    cg_iters=50,    # max CG iterations per step
    ls_steps=30,    # line search resolution
)

for it in range(total_iters):
    # --- standard gradient ---
    s_f, t_f, s_tc, t_tc = sample_collocation(...)
    model.zero_grad()
    loss, lf, ltc = compute_losses(model, s_f, t_f, ...)
    loss.backward()
    g = flat_grad(model)

    # --- loss function for line search (same batch) ---
    def loss_fn():
        s = s_f.detach().clone().requires_grad_(True)
        t = t_f.detach().clone().requires_grad_(True)
        l, _, _ = compute_losses(model, s, t, ...)
        return l

    # --- ENGD step ---
    info = engd.step(
        g,
        s_f.detach()[:n_gram],   # interior Gram points
        t_f.detach()[:n_gram],
        s_tc[:n_gram],            # terminal Gram points
        t_tc[:n_gram],
        loss_fn,
    )
    # info: {'step_size': float, 'cg_residual_norm': float, 'J_F_norm': float}
```

A complete training loop is provided by `train_model_engd()` in
`experiments/python_scripts/exp1/phase3_training.py`.

## Hyperparameter guidance

| Parameter | Typical value | Notes |
|---|---|---|
| `n_gram` | 64–256 | Larger = more accurate Gram but slower per step |
| `reg` | $10^{-4}$–$10^{-3}$ | Increase if CG diverges; decrease if update is too damped |
| `cg_iters` | 20–50 | 20 is usually sufficient; check `cg_residual_norm` in logs |
| `ls_steps` | 30 | Rarely needs tuning |
| `ls_step_max` | 1.0 | Reduce if line search frequently accepts the largest step |

## Limitations

- `BermudaETCNN.forward_pde()` (operator bypass) is not yet wired into the
  Gram computation; the standard `forward()` is used instead.
- For ETCNN models with hard-enforced terminal conditions, set `lam_tc=0.0`
  (the terminal Gram vanishes by construction since $J_{TC} \propto g_1(s,T) = 0$).
- Compute cost per step scales as $O(N_\text{gram} \cdot n_\theta)$ for the
  Jacobian (dominated by second-order differentiation through the network).
