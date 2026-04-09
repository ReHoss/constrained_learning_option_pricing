# Reproducibility

## Conventions

<!-- Document any non-obvious conventions (time parameterisation, moneyness
     normalisation, random seeds, etc.) that affect reproducibility. -->

- **Metadata tracking:** All experiment scripts (`phase1_bsm_validation.py`, `phase2_etcnn_architecture.py`, `phase3_training.py`) automatically save a `metadata.yaml` file in their output directory. This file records the exact Python command used, the timestamp, the Black-Scholes parameters ($K, r, \sigma, T, q$), the neural network hyperparameters, and the domain boundaries. This ensures every run is fully reproducible.

---

## Experiments

<!-- One section per experiment. Record commands, key hyperparameters,
     and where outputs are archived. -->

### Exp 1 — TODO

**Goal:** TODO

**Command:**

```bash
# TODO
```

**Outputs:** TODO

---

## Reference solution configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| BT steps $N$ | 4000 | primary reference |
| TODO | TODO | TODO |
