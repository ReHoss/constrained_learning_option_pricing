# ENGD on BSM with non-differentiable payoff — diagnostic study

This document reports the diagnostic study carried out when applying
the paper-faithful ENGD optimiser (Zeinhofer et al., ICML 2023) to the
European call BSM problem in $(x = \ln S, t)$ coordinates, and explains
why it stalls at a non-zero loss while the same algorithm converges to
machine epsilon on the Poisson 2D problem in the original paper.

The companion implementation is located in
[`train_variant_engd`](../../experiments/python_scripts/exp_singularity_european_call/ablation_singularity_logS.py)
and the unit-tested optimiser primitives in
[`learning_option_pricing/optimizers/natural_gradient.py`](../../learning_option_pricing/optimizers/natural_gradient.py).

## 1. Setup recap

The setup follows the paper exactly:

- Network: small MLP $[\text{Linear}(2,32) \to \tanh \to \text{Linear}(32,1)]$, $n_\theta = 129$ params.
- Grid: $(N-2)^2 = 784$ interior $(x_i, t_i)$ and $N-1 = 29$ terminal $(x_b, T)$ points, $N=30$.
- Stacked residual

$$
r_{\text{full}}(\theta) =
\begin{bmatrix}
\sqrt{\lambda_f / N_{\text{int}}}\; \mathcal{F}[V_\theta](x_i, t_i) \\
\sqrt{\lambda_{tc} / N_{tc}}\; (V_\theta(x_b, T) - \varphi(x_b))
\end{bmatrix},
\quad
J_{\text{full}}(\theta) = \begin{bmatrix} \sqrt{\lambda_f / N_{\text{int}}}\; J_F \\ \sqrt{\lambda_{tc} / N_{tc}}\; J_{TC} \end{bmatrix}.
$$

- Gram $G(\theta) = J_{\text{full}}^\top J_{\text{full}} = \tfrac{\lambda_f}{N_{\text{int}}} J_F^\top J_F + \tfrac{\lambda_{tc}}{N_{tc}} J_{TC}^\top J_{TC}$.
- Natural gradient $\delta = G^+ \nabla L$ via `torch.linalg.lstsq` with SVD pseudoinverse, plus relative Tikhonov $\varepsilon \lVert G\rVert_2$ to bound amplification of near-zero singular values.
- Halving line search on the same fixed grid.

The unit tests show the algorithm reproduces the paper's Poisson 2D result
($L^2 \approx 1.8\!\times\!10^{-5}$ in 51 steps).

## 2. Observations on BSM

After 1 000 steps the optimiser stagnates at

$$
\text{loss} \approx 10.6,\quad \mathcal{L}_f \approx 0.09,\quad \mathcal{L}_{tc} \approx 8.7,\quad \alpha \to 2^{-20},\quad \lVert \delta \rVert \to 10^{-2}.
$$

Three runtime diagnostics make the failure mode unambiguous.

### 2.1 Spectrum of $G$

The Gram is rank-deficient throughout training: $\kappa(G) \approx 10^{17}\text{–}10^{20}$,
with $\lambda_{\min}(G)$ pinned at machine epsilon. This is **not** the problem on its
own — the Tikhonov term bounds the amplification of small-eigenvalue directions
and improves the rel-$L^2$ error from 0.45 to 0.21.

### 2.2 Angle between $\nabla L$ and $\delta$

$\cos(\nabla L, \delta) \in [0.02,\,0.10]$ throughout training, and crosses
zero at stagnation. The Gram rotates the gradient by ~85°, then ~90°. By itself this
is notable but not conclusive: the natural gradient is a *change of metric*,
not an alignment.

### 2.3 The residual recovery ratio $\rho$

The decisive diagnostic. Define

$$
\rho(\theta) := \frac{\lVert P_{\operatorname{col}(J_{\text{full}})}\, r_{\text{full}}(\theta)\rVert^2}{\lVert r_{\text{full}}(\theta) \rVert^2}
= \frac{r^\top J\, (J^\top J)^+ J^\top r}{\lVert r \rVert^2}\;\in [0,1].
$$

$\rho = 1$ means a single Gauss–Newton step zeros out the residual exactly
(the residual lies entirely in the range of $J$). $\rho = 0$ means
$J^\top r = 0$ — the first-order optimality condition for the least-squares
problem $\min_\theta \lVert r(\theta) \rVert^2$.

Trajectory on BSM (run on commit covered by `engd_singularity_diagnostic.md`):

| iter | loss | $\rho$ | $\cos(\nabla L, \delta)$ |
|------|------|--------|-------|
|    1 | 175  | 0.42   | +0.08 |
|   60 | 111  | 0.70   | +0.11 |
|  100 | 76   | 0.52   | +0.03 |
|  200 | 62   | 0.31   | +0.02 |
|  500 | 30   | 0.05   | +0.01 |
|  900 | 11   | 0.00   | +0.00 |
| 1000 | 11   | **0.0000** | +0.001 |

ENGD did not stall; it **converged to a critical point of**
$\lVert r \rVert^2$. The critical point just happens to have non-zero
loss.

## 3. Why this critical point is non-zero (and why Poisson is different)

For Poisson 2D with $u^\star(x,y) = \sin(\pi x)\sin(\pi y)$, the same
129-parameter tanh network can represent $u^\star$ to high accuracy.
The set $\{r(\theta) = 0\}$ is therefore non-empty, and $\rho(\theta) \to 1$
along the optimisation trajectory because the residual stays in the span
of $J$ as it shrinks. Gauss–Newton converges to $r = 0$.

For BSM with non-differentiable payoff, the network must satisfy **two
constraints simultaneously**:

1. interior PDE: $\partial_t V + \tfrac{\sigma^2}{2}\partial_{xx} V + (r - \tfrac{\sigma^2}{2})\partial_x V - rV = 0$ on $\Omega \times (0, T)$;
2. terminal payoff: $V(x, T) = (e^x - K)^+$ on $\Omega$.

The 129-parameter tanh MLP can fit either constraint well in isolation
(e.g. the $\lambda$-flip experiment in §4 reduces $\mathcal{L}_{tc}$ to
0.4 by sacrificing $\mathcal{L}_f = 25$), but **no $\theta$ satisfies both
to high accuracy** because the joint problem's irreducible residual is
non-zero. The non-zero residual creates many $\theta$ at which
$J^\top r = 0$ — non-zero local minima of $\lVert r \rVert^2$. Gauss–Newton
with line search converges to one of them in finite steps and cannot escape.

This is what the paper's Poisson benchmark does not reveal: when the network is
sufficiently expressive relative to the target, the Gauss–Newton critical
point coincides with the zero of $r$, so the algorithm looks unconditionally
convergent.

## 4. Variants explored (documented dead-ends)

Three variants of the basic ENGD step were tested to confirm the diagnosis:

| Variant            | Final loss | $\rho$ at end | Interpretation |
|--------------------|-----------:|--------------:|------------------|
| `engd` (baseline)  | 10.6       | 0.0000        | Standard joint Gram → exact LSQ critical point. |
| `engd_tc_dense` ($N_{tc} = 200$) | 18.8 | 0.000…  | Increasing terminal points does *not* expand $\operatorname{col}(J_{\text{full}})$ enough: the network capacity is the binding rank limit, not the sample count. |
| `engd_lam_flip` ($\lambda_f=1,\lambda_{tc}=20$) | 33.4 | low | Flipping $\lambda$ swaps which residual is small ($\mathcal{L}_{tc} = 0.4$, $\mathcal{L}_f = 25$). Confirms the trap is a property of the joint problem, not of one residual. |
| `engd_alt` (alternating $G_F, G_{TC}$) | 68.6 | 0.29 | Never reaches a LSQ critical point because the preconditioner doesn't match the gradient — cycles between two non-aligned descent directions. Rules out "$\operatorname{col}(J_F) \perp \operatorname{col}(J_{TC})$" as the primary cause. |

The variants `engd_tc_dense` and `engd_alt` were **removed from the
ablation catalog** ([`_build_variants`](../../experiments/python_scripts/exp_singularity_european_call/ablation_singularity_logS.py)) once
they had served their diagnostic purpose. `engd_lam_flip` is kept as a
useful illustration of the residual trade-off.

## 5. What does work and why

The L-BFGS variant `vpinn_lbfgs_epoch` reaches rel-$L^2 = 1.6 \times 10^{-2}$
on the same problem family. This is order-of-magnitude better than any
ENGD configuration. The mechanistic reasons are independent:

- **Curvature memory**: L-BFGS approximates the inverse Hessian from a
  history of $K$ secant pairs, so it accounts for the fact that nearby $\theta$ have
  different LSQ critical-point structure. Gauss–Newton's local quadratic
  model has no such memory.
- **Wolfe line search**: enforces sufficient decrease *and* a curvature
  condition, which prevents the step-size collapse seen here
  ($\alpha \to 2^{-20}$ with the halving search).
- **Weak-form loss**: integrating by parts eliminates the second-spatial
  derivative term that dominates $J_F$'s norm growth near $t = T$ in the
  strong form.

In short: ENGD's failure mode here is **not** an accident or a tuning issue.
It is the expected behaviour of Gauss–Newton on a problem where the joint
representation power of the network is exhausted before the residual hits
zero.

## 6. Cross-check with ANaGRAM (ICLR 2025)

After finalising the diagnosis above, the same failure mode was found
independently described in Schwencke & Furtlehner, *ANaGRAM: A Natural
Gradient Relative to Adapted Model for efficient PINNs learning*
(ICLR 2025, [arXiv:2412.10782](https://arxiv.org/abs/2412.10782), code
[IloneM/ANaGRAM](https://github.com/IloneM/ANaGRAM/)).

ANaGRAM's vanilla update (Algorithm 2 of the paper) is mathematically the
same Gauss–Newton step used here, recast as a direct SVD of the per-sample
Jacobian $\varphi_\theta = J^\top$ rather than of the Gram $J^\top J$,
with a hard spectral cutoff $\Delta_p^\dagger = 1/\Delta_p$ if $\Delta_p > \epsilon$
else $0$ in place of the additive Tikhonov regularisation. The recast
brings two genuine improvements:

1. **Numerical conditioning** — singular values of $J$ are not squared,
   so $\kappa = \sqrt{\kappa(G)}$ in the implicit operator handled by the solver
   ($10^9$ instead of $10^{18}$ in the present setting).
2. **Computational complexity** $O(\min(P^2 S, S^2 P))$ versus $O(P^3)$ —
   relevant when $P \gg S$, but not in the present case ($P=129$, $S=813$).

Crucially, the paper itself acknowledges the same failure mode
diagnosed here with $\rho$. From §4.2 (after Algorithm 2), characterising the
solution quality:

> "the quality of the solution found by the parametric model $u$
> depends only on:
> - how well $\Gamma = \{D[u_\theta] : \theta \in \mathbb{R}^P\}$ can
>   approximate the source $f$;
> - the curvature of $\Gamma$. More precisely, if its non-linear structure
>   induces convergence to a $D[u_\theta]$ such that $f - D[u_\theta]$ is
>   non-negligible, **while being orthogonal to the tangent space
>   $D[T_\theta M]$**."

The second bullet is exactly $J^\top r = 0$ at $r \ne 0$, i.e. $\rho = 0$,
and the paper notes (footnote 5) that "rigorous proof of this phenomenon
has not yet been provided." So this is a recognised but unaddressed
limitation of the natural-gradient PINN family, not something the
algorithmic refinements of ANaGRAM resolve.

The paper's benchmarks succeed (2 D Laplace, $1+1$ D heat, …, with the
same MLP $[2 \to 32 \to 1]$, $P=129$, that is used here) because their target
$f$ lies inside the closure of $\Gamma$ — the network is expressive
enough that the curvature trap is not encountered. The European-call
problem considered here fails because the joint constraint (smooth interior PDE *and*
$C^0$-only terminal payoff) puts $f$ outside $\overline{\Gamma}$ for this
architecture, so the trap is realised.

The repo also has an experimental `cut_low_signal=False` mode
([`anagram.py::nat_grad_factory`](https://github.com/IloneM/ANaGRAM/blob/main/anagram.py))
that adds back the residual component outside $\operatorname{col}(J)$ as
a vanilla gradient. It is off by default, not analysed in the paper, and
the dimensional consistency of the implementation looks questionable on
inspection — but the *idea* (a small gradient-descent step in the
perpendicular subspace) is essentially what one would have to add to
ENGD/ANaGRAM to escape the $\rho \to 0$ trap.

## 7. Concrete fixes (not implemented)

If one wanted to make ENGD work on this problem, the avenues that would
plausibly help are:

1. **Trust region around the Gauss–Newton step**: bound $\lVert \delta \rVert$
   so the step length doesn't depend on the smallest singular values of $G$.
   Restores progress when $\rho \to 0$ at the cost of converging to a worse
   critical point than the one ENGD finds today.
2. **Levenberg–Marquardt-style damping** with adaptive $\mu$, increased
   when a step fails. Equivalent to a Tikhonov term that grows on stalls.
3. **Larger network** so that the joint problem becomes representable
   ($\rho$ stays close to 1) — but this breaks the $M \gg n_\theta$ regime
   the paper relies on for `lstsq` to be the right solve.
4. **Block-coordinate ENGD**: alternating *gradient and preconditioner*
   together (interior step, then terminal step, each minimising only its
   own loss). Different from `engd_alt`, which alternates only the
   preconditioner.

None of these were necessary for the project (a working
optimiser is available via VPINN+L-BFGS), so they are noted here for future reference.

## 8. Reproducing the diagnostics

The diagnostics ($\rho$, $\cos(\nabla L, \delta)$, $\kappa(G)$, $\lambda_{\min}(G)$,
$\lambda_{\max}(G)$, $\lVert J_F \rVert$, $\lVert J_{TC} \rVert$, $\lVert \delta \rVert$)
are recorded into `hist.npz` at every `log_every` iteration. To reproduce:

```bash
python experiments/python_scripts/exp_singularity_european_call/ablation_singularity_logS.py \
    --add-variant engd:data/exp_singularity_european_call/<run_dir> \
    --device cuda
```

Output is written to `<run_dir>/ablation.log`. The full per-step diagnostic
line has the form

```
[engd] iter N/1000  G=J  loss=… Lf=… Ltc=… alpha=…
       |g|=…  |δ|=…  cos(g,δ)=±0.xxx  ρ=0.xxxx  cond(G)=…
       |J_F|=…  |J_TC|=…  (t s)
```

The Poisson 2D reference run (used to confirm the algorithm itself is
correct) is located in
[`exp_engd/repro_poisson_2d.py`](../../experiments/python_scripts/exp_engd/repro_poisson_2d.py)
and reaches $L^2 = 1.84 \times 10^{-5}$ in 51 iterations as expected.
