#!/bin/bash
#
# Launch a single Python script on Ruche (LISN/Mésocentre) — GPU partition.
#
# Usage:
#   bash python_script_launcher_gpu.sh \
#       -p <path/to/script.py> \
#       -a "<space-separated args for the python script>"
#
# Reuses the slurm script slurm_script/run_python_script.slurm (the same one
# used by the CPU launcher) — only the sbatch options differ here.

NAME_PROJECT="constrained_learning_option_pricing"
NAME_JOB_SCRIPT="run_python_script.slurm"

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

# --- sbatch options (Ruche GPU partition) ---
# Common Ruche GPU partitions: gpu, gpup100, gpua100 (check `sinfo` on the cluster).
# Default to `gpu` (mixed pool); override below if you need a specific GPU type.
S_BATCH_PARTITION=gpu
S_BATCH_CPU_PER_TASK=4
S_BATCH_GPUS=1
S_BATCH_TIME=03:59:00
S_BATCH_MEM_PER_NODE="16G"
S_BATCH_NODES=1
S_BATCH_N_TASKS_PER_NODE=1
# Resume-friendly: SLURM requeues on preemption/time-limit; Python --resume
# picks up from the last checkpoint.
S_BATCH_REQUEUE="--requeue"

echo "sbatch options:"
echo "  --job-name=$BASENAME_SCRIPT${VARIANT_TAG:+_$VARIANT_TAG}"
echo "  --output=$PATH_LOG_DIR/%j.out"
echo "  --error=$PATH_LOG_DIR/%j.err"
echo "  --cpus-per-task=$S_BATCH_CPU_PER_TASK"
echo "  --gres=gpu:$S_BATCH_GPUS"
echo "  --time=$S_BATCH_TIME"
echo "  --partition=$S_BATCH_PARTITION"
echo "  --mem=$S_BATCH_MEM_PER_NODE"
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
  --partition="$S_BATCH_PARTITION" \
  --mem="$S_BATCH_MEM_PER_NODE" \
  --nodes="$S_BATCH_NODES" \
  --ntasks-per-node="$S_BATCH_N_TASKS_PER_NODE" \
  $S_BATCH_REQUEUE \
  "$PATH_PARENT"/slurm_script/"$NAME_JOB_SCRIPT"
