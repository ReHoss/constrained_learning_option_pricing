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

---

## Architecture Choices

<!-- Document network architecture decisions here. -->

- Base architecture: ResNet (4-block, 2 layers per block, 50 neurons — same as ETCNN baseline)
- Input normalization: moneyness $s/K$ (as in ETCNN)
- Exact terminal function $g_2$: TODO (adapt from American put/call construction)
- Exercise dates representation: TODO

---

## Benchmark Methods

<!-- List the reference methods and their configuration. -->

- Binomial tree (BT) with $N = 4000$ steps — primary reference
- Finite difference (FD) — secondary reference
- TODO: LSM / other methods?

---

## Open Issues

<!-- Issues that are not yet resolved. -->

- TODO
