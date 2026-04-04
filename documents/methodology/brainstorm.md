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
