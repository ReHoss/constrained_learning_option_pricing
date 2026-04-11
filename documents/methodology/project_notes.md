# Brainstorm: Extending ETCNN to Bermuda Options

## Context

The ETCNN framework (Zhang, Guo, Lu 2026) solves BSM equations for American options
by embedding exact terminal conditions into the neural network output layer. The goal
here is to extend this methodology to **Bermuda options**, which allow early exercise
only at a finite set of prescribed dates.

Key reference:

* W. Zhang, Y. Guo, B. Lu — *Exact Terminal Condition Neural Network for American
  Option Pricing Based on the Black–Scholes–Merton Equations*, J. Comput. Appl. Math.
  480 (2026) 117253.

---

## Open Questions

<!-- Add questions, ideas, and dead ends as the project evolves. -->

- How does the linear complementarity structure change when exercise is only allowed
  at discrete dates $t_1 < t_2 < \ldots < t_M = T$?
- What is the natural generalisation of the exact terminal function $g_2$ to a
  piecewise-in-time setting?
- Can the free boundary at each exercise date be recovered indirectly, as for
  American options?
- What benchmark (binomial tree, finite difference) will serve as the reference
  solution for Bermuda options?
- Multi-asset Bermuda options: does the dimensionality curse affect ETCNN differently
  than for American options?

---

## Directions

<!-- Fill in candidate directions as they crystallise. -->

### Direction A — TODO

### Direction B — TODO

---

## Decision

<!-- Record the chosen direction and rationale. -->

**TODO**



# Decisions

## Project Scope

**Goal:** Extend the ETCNN framework to Bermuda options by adapting the exact terminal
condition design and the linear complementarity loss to the discrete-exercise setting.

---

## Key Decisions

<!-- Record decisions with rationale as they are made. -->

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-04 | Use `g2 = Ve1 + Ve2` (Taylor expansion) not full `Ve` for American put | Same accuracy as full `Ve` with input normalization (Table 3), cheaper to compute — no need to evaluate erfc twice |
| 2026-04-04 | Epsilon floor `tau_eps = 1e-8` on `tau` in all BSM functions | Prevents division by zero as τ→0; verified no NaN at τ=1e-8 in Phase 1 validation |
| 2026-04-04 | Implement BSM operator via `torch.autograd.grad` | Required for loss computation during training; keeps implementation clean and general |
| 2026-04-06 | `InputNormalization` as a separate `nn.Module`, not hardcoded in ResNet | Keeps ResNet reusable for other problems; normalisation is option-pricing–specific |
| 2026-04-06 | PINN baseline shares ResNet architecture, only differs at output | Enables fair comparison — any improvement is attributable to the $g_1/g_2$ modification alone |
| 2026-04-06 | `g1`, `g2` passed as callables to `ETCNN` constructor | Makes the architecture reusable for different option types (call, Bermuda) without subclassing the ResNet |
| 2026-04-07 | European loss uses $\lambda_f = 20$, $\lambda_{tc} = 1$ (no inequality terms) | Matches Section 3.4; European BSM is an equality PDE, no complementarity needed |
| 2026-04-07 | Two-stage LR schedule: decay 0.85 every 2k (first 10k), then every 5k | Follows the paper's training recipe (Section 3.4) |
| 2026-04-07 | Bermudan solved by piecewise backward induction of two European sub-problems | No continuous free boundary — early exercise is a single comparison at $t_1$ |
| 2026-04-07 | Bermudan $g_2$ uses interpolated $V(s, t_1)$, constant in $t$ | $V(s, t_1)$ is already smooth (continuation value, not raw payoff), so $\sqrt{\tau}$ capture is unnecessary |
| 2026-04-07 | Binomial tree with $N = 2000$ for Bermudan reference | Sufficient accuracy for a single exercise date; $N = 4000$ reserved for American |
| 2026-04-07 | Use torch piecewise-linear interpolation for $V(s, t_1)$, not `numpy.interp` | `numpy.interp` breaks the autograd graph — PDE loss then misses $\partial g_2/\partial s$ terms, causing $u_\theta \approx 0$ and the network to ignore time evolution |

---

## Architecture Choices

<!-- Document network architecture decisions here. -->

- Base architecture: ResNet (M=4 blocks, L=2 layers/block, n=50 neurons) — 20,601 parameters
- Input normalization: moneyness $s/K$ via `InputNormalization` module
- Exact terminal function $g_2 = V_1^e + V_2^e$ (American put, validated Phase 1+2)
- PINN baseline: same ResNet, no $g_1/g_2$ — serves as control
- Exercise dates representation for Bermuda: piecewise backward induction with interpolated intermediate conditions

---

## Benchmark Methods

<!-- List the reference methods and their configuration. -->

- Binomial tree (BT) with $N = 4000$ steps — primary American reference
- Binomial tree (BT) with $N = 2000$ steps — Bermudan reference
- European BT for cross-check against analytical solution
- Finite difference (FD) — secondary reference (TODO)

---

## Open Issues

<!-- Issues that are not yet resolved. -->

- TODO
