#!/usr/bin/env bash
# =============================================================================
# Jean Zay SLURM-array launcher for ablation_singularity_logS.py
# =============================================================================
#
# Submits a parallel ablation in three phases, all chained via SLURM job
# dependencies — no manual coordination needed:
#
#   1. INIT      — one CPU task: creates the timestamped ablation directory,
#                  metadata.yaml, and an empty summary.yaml.
#   2. TRAIN     — one GPU task per variant (SLURM array): each task calls
#                  --add-variant against the shared directory.  Concurrent
#                  appends to summary.yaml are serialised via fcntl.flock
#                  inside the Python script.
#   3. FINALIZE  — one CPU task: runs --replot against the shared directory
#                  to regenerate every comparison figure once all variants
#                  have finished.
#
# Usage (from the repository root):
#
#     bash experiments/slurm/jeanzay_ablation_array.sh \
#         --mode  hard-ic-ansatz-european-call \
#         --iters 400 \
#         --account <YOUR_IDRIS_ACCOUNT>@gpu \
#         --qos    qos_gpu-t3            # optional, default qos_gpu-t3
#
# The script lists the variants by parsing _build_variants(mode) from the
# Python module and submits an array job of the matching size.  The
# array-task index is used to pick which variant gets trained.
#
# Outputs land under data/<exp_root>/<timestamped_dir>/ where exp_root is
# determined by the mode (see ablation_singularity_logS.py).  Per-job stdout
# goes to <expdir>/slurm-INIT.out, <expdir>/slurm-TRAIN-<task_id>.out, etc.
# =============================================================================
set -euo pipefail

# ── Defaults — override via CLI flags ────────────────────────────────────────
MODE="compare-boundary-singularity-european-call"
ITERS=20000
ACCOUNT=""                     # required (Jean Zay project allocation @ gpu)
QOS="qos_gpu-t3"               # default queue
TIME_TRAIN="04:00:00"          # wall-clock for one variant
TIME_INIT="00:05:00"
TIME_FINALIZE="00:30:00"
PARTITION_TRAIN=""             # leave empty to use account-default partition
EXTRA_PY_ARGS=""               # forwarded as-is to every Python call

# ── Argument parsing (long-only for clarity) ─────────────────────────────────
while (( $# )); do
    case "$1" in
        --mode)          MODE="$2";          shift 2 ;;
        --iters)         ITERS="$2";         shift 2 ;;
        --account)       ACCOUNT="$2";       shift 2 ;;
        --qos)           QOS="$2";           shift 2 ;;
        --time-train)    TIME_TRAIN="$2";    shift 2 ;;
        --time-init)     TIME_INIT="$2";     shift 2 ;;
        --time-finalize) TIME_FINALIZE="$2"; shift 2 ;;
        --partition)     PARTITION_TRAIN="$2"; shift 2 ;;
        --extra)         EXTRA_PY_ARGS="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,40p' "$0"; exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            sed -n '2,40p' "$0"; exit 1 ;;
    esac
done

if [[ -z "$ACCOUNT" ]]; then
    echo "Error: --account is required (e.g. abc@gpu)." >&2
    exit 1
fi

# ── Locate the repository root and the venv ──────────────────────────────────
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
ABLATION_SCRIPT="$REPO_ROOT/experiments/python_scripts/exp_singularity_european_call/ablation_singularity_logS.py"
VENV_DIR="$REPO_ROOT/venv/venv_learning_option_pricing"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Error: venv not found at $VENV_DIR" >&2
    exit 1
fi
if [[ ! -f "$ABLATION_SCRIPT" ]]; then
    echo "Error: ablation script not found at $ABLATION_SCRIPT" >&2
    exit 1
fi

# ── List variant names that the chosen mode advertises ──────────────────────
# We import _build_variants() once on the head node (CPU) to know how big the
# array job needs to be.  This avoids hard-coding variant counts here.
cd "$REPO_ROOT"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

mapfile -t VARIANTS < <(python - <<EOF
import sys
sys.path.insert(0, "experiments/python_scripts/exp_singularity_european_call")
sys.path.insert(0, "experiments/python_scripts/exp1")
import ablation_singularity_logS as als
for v in als._build_variants("$MODE"):
    if v["name"] in als._PLOT_EXCLUDED_VARIANTS:
        continue
    print(v["name"])
EOF
)

N_VARIANTS=${#VARIANTS[@]}
if (( N_VARIANTS == 0 )); then
    echo "Error: mode $MODE returned 0 variants." >&2
    exit 1
fi
echo "Mode $MODE has $N_VARIANTS variant(s) to train:"
printf '  - %s\n' "${VARIANTS[@]}"

# ── Phase 1: INIT — create the shared ablation directory ────────────────────
# We run --init-only on the login/head node so we have the absolute path
# *before* submitting the array.  This keeps the SLURM scripts simple.
echo "Creating shared ablation directory (--init-only on login node)..."
EXPDIR="$(python "$ABLATION_SCRIPT" \
            --mode "$MODE" --iters "$ITERS" --init-only \
            --device cpu \
            ${EXTRA_PY_ARGS:+$EXTRA_PY_ARGS} 2>/dev/null | tail -n1)"

if [[ ! -d "$EXPDIR" ]]; then
    echo "Error: --init-only did not create a usable directory (got: $EXPDIR)" >&2
    exit 1
fi
echo "Ablation directory: $EXPDIR"

# Persist the variant list inside the run directory for traceability
printf '%s\n' "${VARIANTS[@]}" > "$EXPDIR/variants.txt"

# ── Phase 2: TRAIN — one GPU job per variant via SLURM array ────────────────
TRAIN_SBATCH="$EXPDIR/train_variant.sbatch"
cat > "$TRAIN_SBATCH" <<EOF_TRAIN
#!/usr/bin/env bash
#SBATCH --job-name=ablation_train
#SBATCH --account=$ACCOUNT
#SBATCH --qos=$QOS
#SBATCH --time=$TIME_TRAIN
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --array=0-$((N_VARIANTS-1))
#SBATCH --output=$EXPDIR/slurm-TRAIN-%a.out
#SBATCH --error=$EXPDIR/slurm-TRAIN-%a.err
$( [[ -n "$PARTITION_TRAIN" ]] && echo "#SBATCH --partition=$PARTITION_TRAIN" )
set -euo pipefail
cd "$REPO_ROOT"
source "$VENV_DIR/bin/activate"

# Pick the variant for this array task
mapfile -t VARIANTS_ARR < "$EXPDIR/variants.txt"
VARIANT="\${VARIANTS_ARR[\$SLURM_ARRAY_TASK_ID]}"
echo "task \$SLURM_ARRAY_TASK_ID  → variant \$VARIANT"

# --add-variant trains the chosen variant and appends to summary.yaml.
# Concurrent appends are protected by fcntl.flock inside the Python module
# (see _summary_yaml_lock).
python "$ABLATION_SCRIPT" \\
    --add-variant "\$VARIANT:$EXPDIR" \\
    --device cuda \\
    ${EXTRA_PY_ARGS:+$EXTRA_PY_ARGS}
EOF_TRAIN

echo "Submitting TRAIN array job ($N_VARIANTS tasks)..."
TRAIN_JOB_ID=$(sbatch --parsable "$TRAIN_SBATCH")
echo "  → TRAIN job id: $TRAIN_JOB_ID"

# ── Phase 3: FINALIZE — replot once every training task succeeded ───────────
FINALIZE_SBATCH="$EXPDIR/finalize.sbatch"
cat > "$FINALIZE_SBATCH" <<EOF_FINALIZE
#!/usr/bin/env bash
#SBATCH --job-name=ablation_finalize
#SBATCH --account=$ACCOUNT
#SBATCH --qos=qos_cpu-t3
#SBATCH --time=$TIME_FINALIZE
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --output=$EXPDIR/slurm-FINALIZE.out
#SBATCH --error=$EXPDIR/slurm-FINALIZE.err
#SBATCH --dependency=afterok:$TRAIN_JOB_ID
set -euo pipefail
cd "$REPO_ROOT"
source "$VENV_DIR/bin/activate"
python "$ABLATION_SCRIPT" --replot "$EXPDIR" --device cpu
EOF_FINALIZE

echo "Submitting FINALIZE job (dependency: afterok:$TRAIN_JOB_ID)..."
FINALIZE_JOB_ID=$(sbatch --parsable "$FINALIZE_SBATCH")
echo "  → FINALIZE job id: $FINALIZE_JOB_ID"

# ── Summary ─────────────────────────────────────────────────────────────────
cat <<EOF_SUMMARY

──────────────────────────────────────────────────────────────────────
  Ablation submitted in parallel.

  Mode      : $MODE
  Iters     : $ITERS
  Variants  : $N_VARIANTS  ($(IFS=, ; echo "${VARIANTS[*]}"))
  Exp dir   : $EXPDIR

  Jobs:
    TRAIN array : $TRAIN_JOB_ID   (one task per variant)
    FINALIZE    : $FINALIZE_JOB_ID  (afterok)

  Watch progress:
    squeue -u "\$USER"
    tail -f $EXPDIR/slurm-TRAIN-0.out
    tail -f $EXPDIR/slurm-FINALIZE.out

  Final results:
    cat $EXPDIR/summary.yaml
    ls  $EXPDIR/comparison/
──────────────────────────────────────────────────────────────────────
EOF_SUMMARY
