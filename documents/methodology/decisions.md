# Decisions

## Project Scope

**Goal:** Extend the ETCNN framework to Bermuda options by adapting the exact terminal
condition design and the linear complementarity loss to the discrete-exercise setting.

---

## Key Decisions

<!-- Record decisions with rationale as they are made. -->

| Date | Decision | Rationale |
|------|----------|-----------|
| TODO | TODO | TODO |

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
