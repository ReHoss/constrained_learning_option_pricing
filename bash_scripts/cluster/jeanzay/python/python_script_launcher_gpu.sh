#!/bin/bash
#
# Launch a single Python script on Jean Zay (IDRIS) — GPU partition.
#
# Usage:
#   bash python_script_launcher_gpu.sh \
#       -p <path/to/script.py> \
#       -a "<space-separated args for the python script>"
#
# Example (one variant of the singularity ablation, 1 GPU, 4h):
#   bash python_script_launcher_gpu.sh \
#       -p experiments/python_scripts/exp_singularity_european_call/ablation_singularity_logS.py \
#       -a "--mode compare-boundary-singularity-european-call --add-variant vpinn_lbfgs_full_batch --resume"
#
# Reuses the slurm script slurm_script/run_python_script.slurm (the same one
# used by the CPU launcher) — only the sbatch options differ here.

NAME_PROJECT="constrained_learning_option_pricing"
NAME_JOB_SCRIPT="run_python_script.slurm"

# Alias for workdir on Jean Zay
WORKDIR=$WORK

PATH_PARENT=$(
  cd "$(dirname "${BASH_SOURCE[0]}")" || exit
  pwd -P
)
PATH_CONTENT_ROOT="${PATH_CONTENT_ROOT:-$WORKDIR/git_repositories/$NAME_PROJECT}"

V_ENV_NAME="venv_learning_option_pricing"
PATH_VENV_BIN="$PATH_CONTENT_ROOT"/venv/"$V_ENV_NAME"/bin/activate
echo "PATH_VENV_BIN: $PATH_VENV_BIN"
echo
# shellcheck disable=SC1090
source "$PATH_VENV_BIN" && echo "Activation of virtual environment: $V_ENV_NAME"
echo

# Parse command-line options
while getopts 'p:a:' flag; do
  case "${flag}" in
  p) PATH_PYTHON_SCRIPT="${OPTARG}" ;;
  a) ARGS_PYTHON_SCRIPT="${OPTARG}" ;;
  *) echo "Unexpected option ${flag}" && exit 1 ;;
  esac
done

if [ -z "$PATH_PYTHON_SCRIPT" ]; then
  echo "Missing -p <path/to/script.py> option."
  exit 1
fi

BASENAME_SCRIPT=$(basename "$PATH_PYTHON_SCRIPT" .py)
echo "Script basename: $BASENAME_SCRIPT"
echo

# Per-variant log subdirectory so concurrent submissions do not clash.
# A short variant tag is extracted from ARGS_PYTHON_SCRIPT (--add-variant <name>).
VARIANT_TAG=$(echo "$ARGS_PYTHON_SCRIPT" | grep -oP '(?<=--add-variant )\S+' | head -n1)
PATH_LOG_DIR="$WORKDIR"/logs/$NAME_PROJECT/"$BASENAME_SCRIPT"/$(date +"%Y-%m-%d_%H-%M-%S")${VARIANT_TAG:+_$VARIANT_TAG}
echo "Log directory: $PATH_LOG_DIR"
echo
mkdir -p "$PATH_LOG_DIR"

echo "Launching $NAME_JOB_SCRIPT (GPU)"
echo
echo "PATH_PYTHON_SCRIPT: $PATH_PYTHON_SCRIPT"
echo "ARGS_PYTHON_SCRIPT: $ARGS_PYTHON_SCRIPT"
echo

# --- sbatch options (Jean Zay GPU partition) ---
# 4 cores is the standard ratio for 1 GPU on the V100 partition (40 cores / 4 GPUs).
S_BATCH_CPU_PER_TASK=4
# 1 GPU per job — variants are submitted independently to run in parallel.
S_BATCH_GPUS=1
# Time: most variants converge in <2h on V100; bump to 19:59:00 for L-BFGS-3000.
S_BATCH_TIME=02:00:00
# QoS: qos_gpu-t3 → 20h max ; qos_gpu-dev → 2h, higher priority for testing.
S_BATCH_QOS=qos_gpu-t3
# Account: depends on your allocation — check `idracct` on Jean Zay.
# Forms: <project>@v100, <project>@a100, <project>@h100
S_BATCH_ACCOUNT=ucd32aq@v100
S_BATCH_NODES=1
S_BATCH_N_TASKS_PER_NODE=1
# Resume-friendly: SLURM will requeue the job on preemption/time-limit; the
# Python script's --resume flag picks up from the last checkpoint.
S_BATCH_REQUEUE="--requeue"

echo "sbatch options:"
echo "  --job-name=$BASENAME_SCRIPT${VARIANT_TAG:+_$VARIANT_TAG}"
echo "  --output=$PATH_LOG_DIR/%j.out"
echo "  --error=$PATH_LOG_DIR/%j.err"
echo "  --cpus-per-task=$S_BATCH_CPU_PER_TASK"
echo "  --gres=gpu:$S_BATCH_GPUS"
echo "  --time=$S_BATCH_TIME"
echo "  --qos=$S_BATCH_QOS"
echo "  --account=$S_BATCH_ACCOUNT"
echo "  --nodes=$S_BATCH_NODES"
echo "  --ntasks-per-node=$S_BATCH_N_TASKS_PER_NODE"
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
  --export=NAME_PROJECT="$NAME_PROJECT",PATH_PYTHON_SCRIPT="$PATH_PYTHON_SCRIPT",ARGS_PYTHON_SCRIPT="$ARGS_PYTHON_SCRIPT",WORKDIR="$WORKDIR",PATH_CONTENT_ROOT="$PATH_CONTENT_ROOT",V_ENV_NAME="$V_ENV_NAME" \
  --cpus-per-task="$S_BATCH_CPU_PER_TASK" \
  --gres=gpu:"$S_BATCH_GPUS" \
  --time="$S_BATCH_TIME" \
  --qos="$S_BATCH_QOS" \
  --account="$S_BATCH_ACCOUNT" \
  --nodes="$S_BATCH_NODES" \
  --ntasks-per-node="$S_BATCH_N_TASKS_PER_NODE" \
  $S_BATCH_REQUEUE \
  "$PATH_PARENT"/slurm_script/"$NAME_JOB_SCRIPT"
