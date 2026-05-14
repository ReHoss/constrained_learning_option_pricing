#!/bin/bash
#
# Launch a single Python script on Adastra (CINES) — GPU (MI250X) hardware.
#
# Usage:
#   bash python_script_launcher_gpu.sh \
#       -A <account>                                    \
#       -p <path/to/script.py>                          \
#       -a "<space-separated args for the python script>"
#
# Example (one variant of an ablation, 1 GCD, 4h):
#   bash python_script_launcher_gpu.sh \
#       -A bae1234 \
#       -p experiments/python_scripts/exp_singularity_european_call/ablation_singularity_logS.py \
#       -a "--mode compare-boundary-singularity-european-call --add-variant vpinn_lbfgs_full_batch --resume"
#
# Reuses slurm_script/run_python_script.slurm (same script as the CPU
# launcher) — only the sbatch options differ.
#
# Notes:
#   - Constraint=MI250 → AMD MI250X quadri-GPU nodes.  A node carries 4 cards
#     = 8 GCDs.  We request 1 GCD + 8 cores in shared mode so the smallest
#     billed unit (1 GCD-hour) is what gets charged when the job spans <1
#     full node.  Bump --gpus-per-node and drop --shared for multi-GCD work.
#   - Adastra never accepts ``--qos=`` from the user; QoS is automatic.
#   - PyTorch on Adastra GPUs must be a ROCm build (MI250X = gfx90a).  A
#     CUDA wheel will not work; the diagnostic block in the SLURM script
#     prints the HIP version so this is visible at the top of the log.

NAME_PROJECT="constrained_learning_option_pricing"
NAME_JOB_SCRIPT="run_python_script.slurm"

WORKDIR="${WORKDIR:?WORKDIR env var is not set — are you on an Adastra login node?}"

PATH_PARENT=$(
  cd "$(dirname "${BASH_SOURCE[0]}")" || exit
  pwd -P
)
PATH_PROJECT_PARENT_DIR_REL="${PATH_PROJECT_PARENT_DIR_REL:-pycharm_remote_project}"
PATH_CONTENT_ROOT="${PATH_CONTENT_ROOT:-$WORKDIR/$PATH_PROJECT_PARENT_DIR_REL/$NAME_PROJECT}"

V_ENV_NAME="venv_learning_option_pricing"
PATH_VENV_BIN="$PATH_CONTENT_ROOT"/venv/"$V_ENV_NAME"/bin/activate
echo "PATH_VENV_BIN: $PATH_VENV_BIN"
echo
# shellcheck disable=SC1090
source "$PATH_VENV_BIN" && echo "Activation of virtual environment: $V_ENV_NAME"
echo

# Parse command-line options
S_BATCH_ACCOUNT=""
while getopts 'A:p:a:' flag; do
  case "${flag}" in
  A) S_BATCH_ACCOUNT="${OPTARG}" ;;
  p) PATH_PYTHON_SCRIPT="${OPTARG}" ;;
  a) ARGS_PYTHON_SCRIPT="${OPTARG}" ;;
  *) echo "Unexpected option ${flag}" && exit 1 ;;
  esac
done

if [ -z "${PATH_PYTHON_SCRIPT:-}" ]; then
  echo "Missing -p <path/to/script.py> option."
  exit 1
fi
if [ -z "$S_BATCH_ACCOUNT" ]; then
  echo "Missing -A <account>.  Discover yours with:"
  echo "  sacctmgr -nP list assoc where user=\$USER format=user,account,partition,defaultqos"
  exit 1
fi

BASENAME_SCRIPT=$(basename "$PATH_PYTHON_SCRIPT" .py)
echo "Script basename: $BASENAME_SCRIPT"
echo

# Per-variant log subdirectory so concurrent submissions do not clash.
VARIANT_TAG=$(echo "${ARGS_PYTHON_SCRIPT:-}" | grep -oP '(?<=--(add-)?variant )\S+' | head -n1 | cut -d: -f1)
PATH_LOG_DIR="$WORKDIR"/logs/$NAME_PROJECT/"$BASENAME_SCRIPT"/$(date +"%Y-%m-%d_%H-%M-%S")${VARIANT_TAG:+_$VARIANT_TAG}
echo "Log directory: $PATH_LOG_DIR"
echo
mkdir -p "$PATH_LOG_DIR"

echo "Launching $NAME_JOB_SCRIPT (GPU MI250)"
echo
echo "PATH_PYTHON_SCRIPT: $PATH_PYTHON_SCRIPT"
echo "ARGS_PYTHON_SCRIPT: ${ARGS_PYTHON_SCRIPT:-}"
echo

# --- sbatch options (Adastra MI250, shared mode, 1 GCD) ---
# MI250 has 64 host cores / 8 GCDs = 8 cores per GCD; reserving exactly that
# avoids wasted GPU-hours when the job uses only 1 GCD.
S_BATCH_CPU_PER_TASK=8
S_BATCH_GPUS_PER_NODE=1
S_BATCH_TIME=02:00:00
S_BATCH_CONSTRAINT=MI250
S_BATCH_REQUEUE="--requeue"

# Adastra accounts are per-constraint variants (e.g. cad14975_mi250).
# Auto-derive from the bare -A passed by the user; idempotent if the
# user already passed the suffixed form.
# shellcheck source=_lib/account.sh
source "$PATH_PARENT/_lib/account.sh"
S_BATCH_ACCOUNT_FULL="$(adastra_account_for_constraint "$S_BATCH_ACCOUNT" "$S_BATCH_CONSTRAINT")"

echo "sbatch options:"
echo "  --job-name=$BASENAME_SCRIPT${VARIANT_TAG:+_$VARIANT_TAG}"
echo "  --output=$PATH_LOG_DIR/%j.out"
echo "  --error=$PATH_LOG_DIR/%j.err"
echo "  --constraint=$S_BATCH_CONSTRAINT  # GPU MI250X, shared mode (1 GCD, 8 cores)"
echo "  --account=$S_BATCH_ACCOUNT_FULL  (auto-derived from -A $S_BATCH_ACCOUNT)"
echo "  --cpus-per-task=$S_BATCH_CPU_PER_TASK"
echo "  --gpus-per-node=$S_BATCH_GPUS_PER_NODE"
echo "  --threads-per-core=1"
echo "  --time=$S_BATCH_TIME"
echo "  $S_BATCH_REQUEUE"
echo "  $PATH_PARENT/slurm_script/$NAME_JOB_SCRIPT"
echo
echo "Tail the log in real time after submission:"
echo "  tail -f $PATH_LOG_DIR/<jobid>.out"
echo

# shellcheck disable=SC2086
sbatch \
  --job-name="$BASENAME_SCRIPT${VARIANT_TAG:+_$VARIANT_TAG}" \
  --output="$PATH_LOG_DIR"/%j.out \
  --error="$PATH_LOG_DIR"/%j.err \
  --export=NAME_PROJECT="$NAME_PROJECT",PATH_PYTHON_SCRIPT="$PATH_PYTHON_SCRIPT",ARGS_PYTHON_SCRIPT="${ARGS_PYTHON_SCRIPT:-}",WORKDIR="$WORKDIR",PATH_CONTENT_ROOT="$PATH_CONTENT_ROOT",V_ENV_NAME="$V_ENV_NAME" \
  --constraint="$S_BATCH_CONSTRAINT" \
  --account="$S_BATCH_ACCOUNT_FULL" \
  --nodes=1 \
  --ntasks-per-node=1 \
  --cpus-per-task="$S_BATCH_CPU_PER_TASK" \
  --gpus-per-node="$S_BATCH_GPUS_PER_NODE" \
  --threads-per-core=1 \
  --time="$S_BATCH_TIME" \
  $S_BATCH_REQUEUE \
  "$PATH_PARENT"/slurm_script/"$NAME_JOB_SCRIPT"
