#!/bin/bash
# =============================================================================
# Jean Zay job-array launcher for ablation_singularity_logS.py
# =============================================================================
#
# Builds on the existing pair (this file + slurm_job_array/job_array_batch_xp.slurm)
# so that no new SLURM worker template is needed.  Three phases, all chained
# via SLURM job dependencies:
#
#   1. INIT (login node, no sbatch) — calls --init-only to create the
#      timestamped ablation directory, metadata.yaml, and an empty
#      summary.yaml.  Then enumerates the active variants of the chosen
#      mode (skipping anything in _PLOT_EXCLUDED_VARIANTS) and writes one
#      YAML config per variant under <expdir>/configs/.
#
#   2. TRAIN (sbatch array) — reuses slurm_job_array/job_array_batch_xp.slurm
#      as the per-task worker.  Each task reads its YAML via
#      --config-dir / --config-name, which the Python script translates
#      into an --add-variant call.  Concurrent appends to summary.yaml are
#      serialised by an fcntl.flock inside ablation_singularity_logS.py.
#
#   3. FINALIZE (sbatch, --dependency=afterok, --wrap) — runs --replot to
#      regenerate every comparison figure once every variant has finished.
#
# Usage (from the project root on Jean Zay):
#     bash bash_scripts/cluster/jeanzay/python/ablation_array_launcher.sh \
#         --mode hard-ic-ansatz-european-call
#         # → every variant uses its own catalogue-declared
#         #   default_num_iterations.  Recommended for real ablations.
#
#     # Smoke test → use --debug so the run lands under _debug_<ts>_...
#     bash ... --mode hard-ic-ansatz-european-call --debug
#
# Scope: this launcher owns SLURM-deployment concerns only (account, qos,
# wall-clock, GPU count, project paths, debug marker).  Training
# hyperparameters live in the catalogue — every variant declares its own
# `default_num_iterations`, so no `--num-iterations` flag is exposed here.
# If you need a different iter count on the cluster for one run, either:
#   • tweak the catalogue's `default_num_iterations` on a feature branch
#     (cleanest: the change is reviewable and reproducible), or
#   • edit `<EXPDIR>/configs/*.yaml` between Phase 1 and Phase 2 (set
#     `num_iterations: <N>`).  The launcher pauses between phases just
#     long enough to make this awkward — it is meant for production runs.
# =============================================================================

set -euo pipefail

# ── Defaults — override via CLI flags ────────────────────────────────────────
NAME_PROJECT="constrained_learning_option_pricing"
V_ENV_NAME="venv_learning_option_pricing"
MODE=""

# SLURM resource defaults (match the legacy job_array_launcher_gpu.sh)
S_BATCH_TIME="04:00:00"
S_BATCH_TIME_FINALIZE="00:30:00"
S_BATCH_QOS="qos_gpu-t3"
S_BATCH_ACCOUNT="akz@v100"
S_BATCH_CPU_PER_TASK=10
S_BATCH_GPUS=1                       # one GPU per array task
DEBUG=0                              # mark run as test (folder prefixed `_debug_`)

# ── Argument parsing ─────────────────────────────────────────────────────────
while (( $# )); do
    case "$1" in
        --mode)              MODE="$2";                  shift 2 ;;
        --account)           S_BATCH_ACCOUNT="$2";       shift 2 ;;
        --qos)               S_BATCH_QOS="$2";           shift 2 ;;
        --time)              S_BATCH_TIME="$2";          shift 2 ;;
        --time-finalize)     S_BATCH_TIME_FINALIZE="$2"; shift 2 ;;
        --venv-name)         V_ENV_NAME="$2";            shift 2 ;;
        --name-project)      NAME_PROJECT="$2";          shift 2 ;;
        --debug)             DEBUG=1;                    shift 1 ;;
        -h|--help)
            sed -n '2,32p' "$0"; exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            sed -n '2,32p' "$0"; exit 1 ;;
    esac
done

if [[ -z "$MODE" ]]; then
    echo "Error: --mode is required (e.g. hard-ic-ansatz-european-call)." >&2
    exit 1
fi

# ── Locate the project on Jean Zay's $WORK ──────────────────────────────────
WORKDIR="${WORK:?WORK env var is not set — are you on Jean Zay?}"
PATH_CONTENT_ROOT="$WORKDIR/pycharm_remote_project/$NAME_PROJECT"
PATH_PYTHON_SCRIPT="$PATH_CONTENT_ROOT/experiments/python_scripts/exp_singularity_european_call/ablation_singularity_logS.py"
PATH_VENV_BIN="$PATH_CONTENT_ROOT/venv/$V_ENV_NAME/bin/activate"
PATH_PARENT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PATH_WORKER_SLURM="$PATH_PARENT/slurm_job_array/job_array_batch_xp.slurm"

for path in "$PATH_CONTENT_ROOT" "$PATH_PYTHON_SCRIPT" "$PATH_VENV_BIN" "$PATH_WORKER_SLURM"; do
    if [[ ! -e "$path" ]]; then
        echo "Error: missing path $path" >&2
        exit 1
    fi
done

echo "Project root: $PATH_CONTENT_ROOT"
echo "Python:       $PATH_PYTHON_SCRIPT"
echo "Venv:         $PATH_VENV_BIN"
echo "Worker:       $PATH_WORKER_SLURM"
echo "Mode:         $MODE  (each variant uses its catalogue default_num_iterations)"
echo

# ── Activate venv on the login node (for --init-only and YAML generation) ───
# shellcheck source=/dev/null
source "$PATH_VENV_BIN"
cd "$PATH_CONTENT_ROOT"

# ── Phase 1: create the shared ablation directory ───────────────────────────
DEBUG_FLAG=""
if (( DEBUG == 1 )); then
    DEBUG_FLAG="--debug"
    echo "DEBUG mode: the timestamped folder will be prefixed with '_debug_'"
fi
echo "Phase 1: --init-only (login node)..."
EXPDIR="$(python "$PATH_PYTHON_SCRIPT" \
            --mode "$MODE" --init-only \
            $DEBUG_FLAG \
            --device cpu 2>/dev/null | tail -n1)"

if [[ ! -d "$EXPDIR" ]]; then
    echo "Error: --init-only did not return a usable directory (got: $EXPDIR)" >&2
    exit 1
fi
echo "  ablation dir: $EXPDIR"

# ── Phase 1b: enumerate variants and write one YAML per variant ─────────────
# Imports `_ablation_catalogue` directly (not the main script) so that the
# YAML generation phase does not pay the multi-second torch import on
# Lustre — `_ablation_catalogue` is intentionally torch-free.
PATH_FOLDER_CONFIGS="$EXPDIR/configs"
mkdir -p "$PATH_FOLDER_CONFIGS"
python - <<PY
import sys, yaml
from pathlib import Path
sys.path.insert(0, "experiments/python_scripts/exp_singularity_european_call")
import _ablation_catalogue as ac
mode    = "$MODE"
expdir  = Path("$EXPDIR")
out_dir = Path("$PATH_FOLDER_CONFIGS")
written = []
for v in ac._build_variants(mode):
    if v["name"] in ac._PLOT_EXCLUDED_VARIANTS:
        continue
    # ``num_iterations: null`` instructs the worker to let each variant use
    # its catalogue-declared ``default_num_iterations``.  If you need to
    # override on the cluster, edit this YAML before the array starts.
    cfg = {
        "mode":            mode,
        "variant_name":    v["name"],
        "ablation_dir":    str(expdir),
        "num_iterations":  None,
    }
    (out_dir / f"{v['name']}.yaml").write_text(yaml.safe_dump(cfg))
    written.append(v["name"])
print(f"Wrote {len(written)} YAML configs (torch-free):")
for n in written:
    print(f"  - {n}")
PY

N_CONFIGS=$(find "$PATH_FOLDER_CONFIGS" -name "*.yaml" | wc -l)
if (( N_CONFIGS == 0 )); then
    echo "Error: 0 YAML configs were written — mode $MODE has no active variants?" >&2
    exit 1
fi
N_LAST_ARRAYID=$((N_CONFIGS - 1))
echo

# ── Phase 2: submit the training array (reuses the existing worker) ─────────
echo "Phase 2: sbatch --array=0-$N_LAST_ARRAYID (training, $N_CONFIGS GPU tasks)..."
TRAIN_JOB_ID=$(sbatch --parsable \
    --job-name="ablation_${MODE}" \
    --array=0-${N_LAST_ARRAYID} \
    --output="$EXPDIR/slurm-TRAIN-%A_%a.out" \
    --error="$EXPDIR/slurm-TRAIN-%A_%a.err" \
    --export=NAME_PROJECT="$NAME_PROJECT",PATH_PYTHON_SCRIPT="$PATH_PYTHON_SCRIPT",PATH_FOLDER_CONFIGS="$PATH_FOLDER_CONFIGS",WORKDIR="$WORKDIR",V_ENV_NAME="$V_ENV_NAME" \
    --cpus-per-task="$S_BATCH_CPU_PER_TASK" \
    --time="$S_BATCH_TIME" \
    --qos="$S_BATCH_QOS" \
    --account="$S_BATCH_ACCOUNT" \
    --nodes=1 \
    --ntasks-per-node=1 \
    --gres=gpu:"$S_BATCH_GPUS" \
    --hint=nomultithread \
    "$PATH_WORKER_SLURM")
echo "  TRAIN job id: $TRAIN_JOB_ID"

# ── Phase 3: submit the finalize/replot job (afterok) ───────────────────────
# --wrap keeps the script count at one — no separate finalize.slurm template.
# We request 0 GPUs because --replot runs the BS reference + matplotlib only.
echo "Phase 3: sbatch --dependency=afterok:$TRAIN_JOB_ID (replot)..."
FINALIZE_JOB_ID=$(sbatch --parsable \
    --job-name="ablation_${MODE}_finalize" \
    --dependency="afterok:${TRAIN_JOB_ID}" \
    --output="$EXPDIR/slurm-FINALIZE.out" \
    --error="$EXPDIR/slurm-FINALIZE.err" \
    --cpus-per-task=4 \
    --time="$S_BATCH_TIME_FINALIZE" \
    --qos="$S_BATCH_QOS" \
    --account="$S_BATCH_ACCOUNT" \
    --nodes=1 --ntasks-per-node=1 \
    --gres=gpu:0 \
    --hint=nomultithread \
    --wrap "cd '$PATH_CONTENT_ROOT' && source '$PATH_VENV_BIN' && python '$PATH_PYTHON_SCRIPT' --replot '$EXPDIR' --device cpu")
echo "  FINALIZE job id: $FINALIZE_JOB_ID"

# ── Summary ─────────────────────────────────────────────────────────────────
cat <<EOF_SUMMARY

──────────────────────────────────────────────────────────────────────
  Ablation submitted in parallel.

  Mode      : $MODE  (each variant uses its catalogue default_num_iterations)
  Variants  : $N_CONFIGS  (see $PATH_FOLDER_CONFIGS/*.yaml)
  Exp dir   : $EXPDIR

  Jobs:
    TRAIN array : $TRAIN_JOB_ID   (one GPU task per variant)
    FINALIZE    : $FINALIZE_JOB_ID  (afterok, --replot, CPU)

  Watch progress:
    squeue -u "\$USER"
    tail -f $EXPDIR/slurm-TRAIN-${TRAIN_JOB_ID}_0.out
    tail -f $EXPDIR/slurm-FINALIZE.out

  Results:
    cat $EXPDIR/summary.yaml
    ls  $EXPDIR/comparison/
──────────────────────────────────────────────────────────────────────
EOF_SUMMARY
