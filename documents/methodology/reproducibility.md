# Reproducibility

## Conventions

- **Metadata tracking:** All experiment scripts (`phase1_bsm_validation.py`, `phase2_etcnn_architecture.py`, `phase3_training.py`) automatically save a `metadata.yaml` in their output directory. This file records the exact Python command, timestamp, Black-Scholes parameters ($K, r, \sigma, T, q$), neural-network hyperparameters, domain boundaries, and all active flags.
- **Time convention:** $t$ runs from $0$ (today) to $T$ (expiry). The BSM PDE is solved backward in time via $\tau = T - t$.
- **Random seed:** Fixed globally at `SEED = 42` inside each phase script.
- **Output directories:** Created automatically under `data/<script_name>/` with a timestamp + key config values in the folder name.

---

## Running experiments

### Environment

```bash
source venv/venv_learning_option_pricing/bin/activate
```

---

### Phase 3 — ETCNN training (`phase3_training.py`)

Trains an ETCNN (and PINN baseline) on the European put, then a two-stage Bermudan put via backward induction.

#### Minimal run (all defaults)

```bash
python experiments/python_scripts/exp1/phase3_training.py
```

Runs 50 000 iterations per stage, uses the Taylor $g_2$ form, standard cubic interpolation for $V(s, t_1)$, on GPU if available.

#### Full flag reference

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--iters N [N ...]` | int+ | `50000` | Training iterations per stage. One value → same for all stages. Two values → Stage A then Stage B. Example: `--iters 20000 5000` |
| `--g2 {taylor,bs}` | str | `taylor` | Terminal function $g_2$ for ETCNN. `taylor`: $g_2 = V_1^e + V_2^e$ (Taylor expansion capturing $\sqrt{\tau}$ singularity, §2.1 of architecture.md). `bs`: $g_2 = P^{\text{BS}}$ (exact Black-Scholes European put). Applied to both the European problem and Stage A of the Bermudan. |
| `--extraction` | flag | *off* | Enable singularity extraction ansatz for Stage B (§2.2 of architecture.md). Decomposes $V_\theta = v + \tilde{u}_\theta$, removing the $C^0$ kink at the exercise boundary $s^*$. When *off*, Stage B directly interpolates $V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_1)$. |
| `--interp {cubic,pchip,linear}` | str | `cubic` | Interpolator used for $V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_1)$ in the **non-extraction** path. Ignored when `--extraction` is set. `cubic`: $C^2$ natural spline. `pchip`: $C^1$, shape-preserving. `linear`: $C^0$, drops diffusion (benchmarking only). |
| `--device {auto,cuda,cpu}` | str | `auto` | Compute device. `auto` selects CUDA if available. |
| `--weight-decay W` | float | `0.0` | L2 regularisation weight for Adam. |
| `--log-every N` | int | `1000` | Print/log loss every $N$ iterations. |
| `--n-tc N` | int | `1024` | Number of terminal-condition collocation points per batch. |
| `--n-f N` | int | `4096` | Number of interior PDE collocation points per batch. |
| `--bermudan-only` | flag | *off* | Skip the European problem and go directly to the Bermudan stages. Mutually exclusive with `--european-only`. |
| `--european-only` | flag | *off* | Skip the Bermudan problem and run only the European stages (ETCNN + PINN). Mutually exclusive with `--bermudan-only`. |
| `--load-etcnn-a PATH` | str | `None` | Load a pre-trained Stage A network from file (skips Stage A training). Useful to re-run Stage B with a different `--interp` or `--extraction` setting. |

#### Output directory naming

The output directory encodes the key settings so benchmark runs don't collide:

```
data/phase3_training/<timestamp>_iters<A>_<B>_K<K>_<mode>_g2-<g2>/
```

where `<mode>` is `extraction` or `interp-<method>`. Example:

```
data/phase3_training/20260410_143000_iters20000_5000_K100_extraction_g2-taylor/
```

#### Example commands

```bash
# Default: both problems, cubic interpolation, Taylor g2, 50k iters
python experiments/python_scripts/exp1/phase3_training.py

# European only (ETCNN + PINN baseline, analytical reference)
python experiments/python_scripts/exp1/phase3_training.py \
    --european-only --iters 50000

# Bermudan only, PCHIP interpolation
python experiments/python_scripts/exp1/phase3_training.py \
    --bermudan-only --interp pchip --iters 20000

# Singularity extraction ansatz, BS g2, 20k Stage A + 5k Stage B
python experiments/python_scripts/exp1/phase3_training.py \
    --extraction --g2 bs --iters 20000 5000

# PCHIP interpolation (no extraction), Taylor g2, CPU
python experiments/python_scripts/exp1/phase3_training.py \
    --interp pchip --device cpu --iters 30000 10000

# Skip Stage A using a saved model
python experiments/python_scripts/exp1/phase3_training.py \
    --extraction --load-etcnn-a data/phase3_training/<run>/etcnn_a.pt

# Bermudan only, BS g2, large run
python experiments/python_scripts/exp1/phase3_training.py \
    --bermudan-only --g2 bs --iters 100000 20000 --weight-decay 1e-5
```

#### Outputs

All outputs land in the timestamped output directory:

| File/folder | Description |
|------------|-------------|
| `metadata.yaml` | Full run config (command, parameters, flags) |
| `training.log` | Full training log |
| `models/etcnn_eur.pt`, `models/pinn_eur.pt` | European ETCNN and PINN weights |
| `models/etcnn_a.pt`, `models/etcnn_b.pt` | Bermudan Stage A and Stage B weights |
| `training_metrics/` | Per-model loss and diagnostic curves (E1, B3, B4, per-model metrics) |
| `pricing/` | Price surface and slice plots (E2, E4, B1, B1c, B5, B6) |
| `greeks/` | Delta / Gamma / Theta plots (E5, B8) |
| `diagnostics/` | Error heatmaps, interpolation diagnostics, PDE residuals (E3, B1b, B1d, B7, B9, B10) |

---

## Archived pathological examples

The following run has been archived as a pathological case that was attempted to be solved but did not meet the expected Bermudan-stage quality targets:

| Run ID | Archive location | Status |
|--------|------------------|--------|
| `20260409_154609_iters20000_K100_interpcubic` | `data/working_archives/reproductibility_results/20260409_154609_iters20000_K100_interpcubic_pathological_attempted_solution/` | Pathological attempted-solution example |

---

## Reference solution configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| BT steps $N$ | 4000 | primary binomial-tree reference |
| $K$ | 100 | contract strike |
| $r$ | 0.02 | risk-free rate |
| $\sigma$ | 0.25 | volatility |
| $T$ | 1.0 | expiry (years) |
| $t_1$ | 0.5 | Bermudan exercise date |
