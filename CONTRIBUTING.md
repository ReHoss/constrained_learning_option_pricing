# Contributing

## Setup

Clone and install in development mode with the ML and dev extras:

```bash
git clone <repo-url>
cd constrained_learning_option_pricing
pip install -e ".[ml,dev]"
```

This installs:
- **Core:** `matplotlib`, `numpy`, `scipy`
- **ML:** `torch >= 2.0`
- **Dev:** `pytest`, `ruff`

## Project layout

```
learning_option_pricing/   # core Python package
    pricing/
        terminal.py        # BSM operator, payoffs, Ve, Ve1/Ve2, g1, g2
        loss.py            # complementarity loss terms (L_bs, L_tv, L_eq)
        bjerksund_stensland.py  # BS-2002 American put approximation (g2 anchor)
    models/
        resnet.py          # ResNet backbone
        etcnn.py           # ETCNN wrapper (trial solution = g1*NN + g2)
    solvers/
        binomial_tree.py   # reference CRR solver
    visualization/
        option_plots.py    # price surface, error heatmap, free boundary plots
    utils/
        run_context.py     # experiment metadata and run-directory creation

experiments/
    python_scripts/        # standalone experiment scripts
        exp1/              # single-asset American put (Section 4.1, Zhang et al. 2026)
    notebooks/             # Jupyter notebooks for exploration

data/                      # generated experiment output (not committed)
    <script_name>/
        <timestamp>_<key_params>/   # one folder per run

documents/
    methodology/           # design decisions, architecture notes, math→code mappings
```

## Running an experiment

```bash
python experiments/python_scripts/exp1/phase1_bsm_validation.py
```

Output is saved to `data/phase1_bsm_validation/<timestamp>_<params>/`.

## Tests

```bash
pytest
```

Tests live in `test/` and mirror the package structure.

## Code style

```bash
ruff check .
ruff format .
```

## Adding a new experiment

1. Create `experiments/python_scripts/<exp_name>/<exp_name>.py`.
2. Use `learning_option_pricing.utils.run_context.create_run_dir` to create the output
   directory under `data/<exp_name>/<timestamp>_<key_params>/`.
3. Store all generated plots and CSVs in that directory.
4. Update this file if new dependencies are required.

## Reference

Zhang, W., Guo, Y., Lu, B. — *Exact Terminal Condition Neural Network for American
Option Pricing Based on the Black–Scholes–Merton Equations*,
J. Comput. Appl. Math. **480** (2026) 117253.
https://doi.org/10.1016/j.cam.2025.117253
