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
    optimizers/
        natural_gradient.py        # Empirical Natural Gradient Descent (PINN, strong form)
        natural_gradient_vpinn.py  # ENGD for variational PINNs (Galerkin form)
    vpinn/
        loss.py            # weak-form Black-Scholes residual loss
        quadrature.py      # Gauss-Legendre quadrature
        test_functions.py  # sinusoidal test functions
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

## Running on an HPC cluster

Bash launchers live under `bash_scripts/cluster/{jeanzay,ruche}/python/`.

### One-time setup (on the cluster)

```bash
# 1. Clone the repo. PATH_CONTENT_ROOT in the launchers defaults to
#    "$WORK/git_repositories/constrained_learning_option_pricing" on Jean Zay
#    and "$WORKDIR/git_repositories/constrained_learning_option_pricing" on Ruche.
mkdir -p "$WORK"/git_repositories   # (Jean Zay; use $WORKDIR on Ruche)
cd "$WORK"/git_repositories
git clone git@github.com:ReHoss/constrained_learning_option_pricing.git
cd constrained_learning_option_pricing

# 2. Create the Python venv. Jean Zay: load a recent Python module first.
module load python/3.11.5     # or any >=3.10 available; check `module avail python`
python -m venv venv/venv_learning_option_pricing
source venv/venv_learning_option_pricing/bin/activate

# 3. Install the package + ML extras.
pip install --upgrade pip
pip install -e ".[ml]"

# 4. Smoke-test CUDA on a GPU compute node (Jean Zay: srun, see below).
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Submitting a job

Single GPU job, one variant:

```bash
bash bash_scripts/cluster/jeanzay/python/python_script_launcher_gpu.sh \
  -p experiments/python_scripts/exp_singularity_european_call/ablation_singularity_logS.py \
  -a "--mode compare-boundary-singularity-european-call --add-variant vpinn_lbfgs_full_batch --resume"
```

The launcher creates a timestamped log directory under
`$WORK/logs/constrained_learning_option_pricing/<script>/<timestamp>_<variant>/`
and prints the `tail -f` command for live monitoring. Submit multiple variants
in parallel by calling the launcher once per variant — each becomes an
independent sbatch job.

Resume-friendly: `--requeue` is set, so SLURM resubmits the job on preemption
or time-limit, and the Python script's `--resume` flag picks up from the last
checkpoint.

### Interactive debugging

```bash
bash bash_scripts/cluster/jeanzay/python/run_interactive_job.sh
# (drops you on a compute node with 1 GPU for 30 min, qos_gpu-dev)
```

### Syncing results back to your workstation

```bash
# from your local machine
rsync -avz jeanzay:/path/to/constrained_learning_option_pricing/data/exp_singularity_european_call/ \
           data/exp_singularity_european_call/
```

### Jean Zay account allocations (project 105646)

| Partition | Account string         | Allocation     |
|-----------|------------------------|----------------|
| V100      | `akz@v100`             | 5000 h.gpu     |
| A100      | `akz@a100`             | 5000 h.gpu     |
| H100      | `akz@h100`             | 1250 h.gpu     |
| CPU       | `akz@cpu`              | 26575 h.cpu    |

The Jean Zay SLURM account is `<project>@<partition>` where `<project>` is the
short project name (run `idrproj` to see yours), not the user login. The
`akz` shorthand maps to project 105646.

Defaults in the launchers target V100; edit `S_BATCH_ACCOUNT` and the QoS
(`qos_gpu-t3` → 20h, `qos_gpu-dev` → 2h) at the top of the launcher if you
want a different partition.

## References

* Zhang, W., Guo, Y., Lu, B. — *Exact Terminal Condition Neural Network for
  American Option Pricing Based on the Black–Scholes–Merton Equations*,
  J. Comput. Appl. Math. **480** (2026) 117253.
  <https://doi.org/10.1016/j.cam.2025.117253>

* Zeinhofer, M. et al. — *Natural Gradient PINNs*, ICML 2023.
  <https://github.com/MariusZeinhofer/Natural-Gradient-PINNs-ICML23>
  See `documents/methodology/engd_optimizer.md` for the PyTorch port,
  the strong-form (`ENGDOptimizer`) and variational
  (`VPINNENGDOptimizer`) variants, and the validated hyperparameter
  recipe.
