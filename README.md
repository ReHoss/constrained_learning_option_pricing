# learning_option_pricing

ETCNN for option pricing — extending exact terminal condition neural networks
to Bermuda options.

## Installation

```bash
pip install -e .
# with ML dependencies (PyTorch)
pip install -e ".[ml]"
```

## Usage

### Unit tests

```bash
pip install -e ".[dev,ml]"
pytest
```

### Experiments

**Phase 1 — BSM mathematical components validation:**

```bash
python experiments/python_scripts/exp1/phase1_bsm_validation.py
```

## Package structure

```
learning_option_pricing/
    models/         # ResNet backbone + ETCNN wrapper
    pricing/        # BSM loss terms, exact terminal functions
    solvers/        # Reference numerical solvers (binomial tree, FD)
    utils/          # Run context, logging helpers
    visualization/  # Option price surfaces, free boundary plots
```

## References

* W. Zhang, Y. Guo, B. Lu — *Exact Terminal Condition Neural Network for American
  Option Pricing Based on the Black–Scholes–Merton Equations*,
  J. Comput. Appl. Math. 480 (2026) 117253.

