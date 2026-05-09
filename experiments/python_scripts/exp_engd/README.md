# Empirical Natural Gradient Descent (ENGD) — convergence experiments

Two scripts validate the ENGD implementation on Black-Scholes benchmarks:

| Script | Loss formulation | Reference solution |
|---|---|---|
| `convergence_european_put.py` | strong-form PINN (pointwise residual) | analytic Black-Scholes put |
| `convergence_vpinn.py` | weak-form VPINN (Galerkin against test functions) | $u\equiv 0$ on the homogeneous PDE |

Each script outputs to `data/exp_engd/<timestamp>_<name>/` with:

* `run.log`         — per-iteration log
* `run_info.yaml`   — config + final metrics
* `history_*.yaml`  — full per-iter histories (loss, time, etc.)
* `*.png`           — convergence and diagnostics plots

## Quick start

Light-budget run (a few minutes on CPU):

```bash
# strong-form PINN
python experiments/python_scripts/exp_engd/convergence_european_put.py \
    --iters-engd 60 --iters-adam 3000 \
    --hidden 16 --n-s 15 --n-t 15 \
    --n-gram 225 --n-tc-gram 64 \
    --reg 1e-3 --cg-iters 200

# variational PINN
python experiments/python_scripts/exp_engd/convergence_vpinn.py \
    --iters-engd 40 --iters-adam 3000
```

## Hyperparameter recipe

The defaults in both scripts reflect the validated recipe from the
methodology document:

* `n_gram` ≥ size of the loss collocation set (mismatch poisons the preconditioner)
* `reg = 1e-3` (lower values cause CG divergence on ill-conditioned $G$)
* `cg_iters = 100–200` (many CG iters are cheap; few iters can stall)
* deterministic / fixed-grid collocation (no resampling between steps)

See `documents/methodology/engd_optimizer.md` for the full discussion.

## Reproducibility

All scripts:

* fix the random seed (`--seed`, default 7),
* use `torch.set_default_dtype(torch.float64)`,
* dump the config in `run_info.yaml`.

## Expected results

`convergence_european_put.py` (default light budget) — ENGD reaches the
same loss as Adam in roughly 100× fewer iterations. With sufficient
budget, ENGD is also faster in *wall time* (≈15× on the long run).

`convergence_vpinn.py` — VPINN-ENGD descends from $\sim 10^{-3}$ to
$\sim 10^{-5}$ in a single step (line search accepts $\alpha = 0.5$),
then continues with $\alpha = 1.0$. CG residuals stay at machine
precision throughout.
