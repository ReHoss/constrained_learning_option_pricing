# Pre-registration — sixth-order test case on the circle

**Written 2026-10-11, before any training run of this study.** The author asked on 2026-10-11 to « launch on jeanzay the sixth order test case », the case proposed by the plan editor in message `20261010T211027413110Z_gpt_astra_plan_editor_e65fd540` at the author's request. The only computation done before writing this document is the closed-form evaluation of the forcing floors of Section 3, without training.

## 1. Question

The paper's criterion (Propositions 4.2 and 4.3) depends on the half order $p$ of the generator and on the regularity of the datum through the Sobolev index $s_B=\max(m_B-p,0)$. Every trained comparison so far has $p\in\{1,2\}$. This study asks whether the ordering of the trained errors observed at orders two and four persists at order six, where the remainder of the highest-order split has order $m_B=4>p=3$ and the criterion requires $g\in H^{1}(\mathbb T)$.

## 2. Design

**Cell `sixth_order_bernoulli_bandlimited`.** In the runner's backward convention,
$$Pu=\partial_tu+Au=0\ \text{on}\ \mathbb T\times(0,T),\qquad u(\cdot,T)=g_{K_g},\qquad A=0.01\,\frac{\partial^6}{\partial x^6}-0.05\,\partial^4_{xxxx},\qquad T=1.$$
- The symbol of $A$ is $\lambda_{A,k}=-0.01\,k^6-0.05\,k^4$, dissipative at every wavenumber $k$.
- The datum is the paper's band-limited datum $g_{K_g}(x)=\sum_{k=1}^{K_g}\cos(kx)/(\pi^2k^2)$ with $K_g=128$, the projection of the datum of Example 5.1. The full datum belongs to $H^s(\mathbb T)$ exactly for $s<3/2$ (Corollary A.1), hence to $H^1(\mathbb T)$.
- The coefficient $-0.05$ of the order-4 term is that of $G_3$. The coefficient $0.01$ of the order-6 term gives exact decay rates $0.06$, $1.44$ and $11.34$ at $k=1,2,3$, so at $t=0$ the solution is carried by the first two wavenumbers, a regime comparable to $G_3$.

**Extensions.** All use the linear factor $\lambda(t)=t/T$.

| Variant | Extension | Retained operator | Remainder $B$ | $m_B$ | $s_B$ |
|---|---|---|---|---|---|
| `convex_raw` | $(t/T)\,g_{K_g}$ | — | — | — | — |
| `constant_in_time` | $g_{K_g}$ | $0$ | $A$ | $6$ | $3$ |
| `split_principal` | $e^{(T-t)A_6}g_{K_g}$ | $A_6=0.01\,\partial^6/\partial x^6$ | $-0.05\,\partial^4_{xxxx}$ | $4$ | $1$ |
| `exact_solution` | $u^\star$ | $A$ | $0$ | $0$ | $0$ |

The highest-order split is the paper's comparator. Retaining both terms of $A$ gives the exact solution, so there is no further split variant. The kernel-based extensions are not included: their comparison scales are defined for the second-order kernel and have no order-6 counterpart in the paper.

**Everything else** is unchanged from the validation-selected series of the paper (Appendix C): periodic features $(\cos x,\sin x,1-2t/T)$, residual network of width 64 with four blocks, Adam with cosine annealing over $20000$ updates, batches of $4096$ points, selection by the residual on a fixed validation sample of $16384$ points every $100$ updates, evaluation grid of $1024$ points and $11$ time slices.

**Seeds and tasks.** Seeds $\{0,1,2\}$, giving $4\times3=12$ tasks.
- Jean Zay V100 (`gpu_p13`, account `akz@v100`, `qos_gpu-t3`), one GPU per task, one array of 12 tasks.
- A smoke test runs first on `qos_gpu-dev` with `--debug`, to check the code paths and to measure the time per update. Its values are not results of this study.
- Run directories: `data/ablation_split_extension_trained/sixth_order_study_2026-10-11/`.

**Code.** Order-6 support: revision `ca14a01` (library and tests) and `8a09e8e` (cell). The operator differentiates the network up to order six by repeated automatic differentiation. The split and control extensions use the closed-form derivatives up to order six. The raw extensions use automatic differentiation.

## 3. Closed-form forcing floors

The forcing floor is $\mathbb E[(P\Psi)^2]=E_h/(2\pi T)$, evaluated by the runner's closed-form finite sums over $|k|\le K_g=128$ at revision `8a09e8e`:

| Variant | Forcing floor |
|---|---|
| `convex_raw` | $1.8172\times10^{11}$ |
| `constant_in_time` | $5.4515\times10^{11}$ |
| `split_principal` | $3.7692\times10^{-4}$ |
| `exact_solution` | $0$ |

The raw floors grow like $K_g^{9}$ under refinement: the summands of the paper's raw-energy sums behave like $k^{8}$ at order six, which extends to $p=3$ the argument the paper states for $p\in\{1,2\}$, with growth $K_g^{4p-3}$. The split floor is bounded uniformly in $K_g$ (Proposition 4.2 with $s_B=1$).

## 4. Predictions

Every prediction concerns the median over the three seeds of the relative error $\delta_\Gamma$ on the evaluation grid.

- **S1.** The highest-order split has a smaller median $\delta_\Gamma$ than both raw extensions.
- **S2.** The ratio of the smaller raw median to the split median exceeds $10^{2}$. On $G_3$ at $K_g=128$ the ratio was $3.80/(4.91\times10^{-4})\simeq7.7\times10^{3}$.
- **S3.** The split median is below $10^{-2}$.
- **S4.** The exact-solution control has a smaller median than the split.

A raw extension may produce non-finite values or activate the gradient-norm safeguard, given floors of order $10^{11}$. Every such run is reported with its seed, and no prediction is tested on a run that is not admissible in the sense of the paper (finite validation and evaluation quantities, exact initial-value check).

## 5. Outcome

To be written after the runs.
