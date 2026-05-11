#!/bin/bash
#
# Launch a single Python script on Ruche (LISN Mesocentre) — CPU partition.
#
# Usage:
#   bash python_script_launcher.sh \
#       -p <path/to/script.py> \
#       -a "<space-separated args for the python script>"
#
# For GPU work, use python_script_launcher_gpu.sh instead.

NAME_PROJECT="constrained_learning_option_pricing"
NAME_JOB_SCRIPT="run_python_script.slurm"

PATH_PARENT=$(
  cd "$(dirname "${BASH_SOURCE[0]}")" || exit
  pwd -P
)
# Path to the project's content root directory on Ruche.
# Override by exporting PATH_CONTENT_ROOT before calling this script.
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

PATH_LOG_DIR="$WORKDIR"/logs/$NAME_PROJECT/"$BASENAME_SCRIPT"/$(date +"%Y-%m-%d_%H-%M-%S")
echo "Log directory: $PATH_LOG_DIR"
echo
mkdir -p "$PATH_LOG_DIR"

echo "Launching $NAME_JOB_SCRIPT"
echo
echo "PATH_PYTHON_SCRIPT: $PATH_PYTHON_SCRIPT"
echo "ARGS_PYTHON_SCRIPT: $ARGS_PYTHON_SCRIPT"
echo

# --- sbatch options (Ruche CPU partition) ---
S_BATCH_CPU_PER_TASK=40
# Time: cpu_short → 2h, cpu_med → 24h, cpu_long → 5d (with cpu_long partition)
S_BATCH_TIME=3:59:00
S_BATCH_PARTITION=cpu_med
S_BATCH_N_TASKS_PER_NODE=1

echo "sbatch options:"
echo "  --job-name=$BASENAME_SCRIPT"
echo "  --output=$PATH_LOG_DIR/%j.out"
echo "  --error=$PATH_LOG_DIR/%j.err"
echo "  --cpus-per-task=$S_BATCH_CPU_PER_TASK"
echo "  --time=$S_BATCH_TIME"
echo "  --partition=$S_BATCH_PARTITION"
echo "  --ntasks-per-node=$S_BATCH_N_TASKS_PER_NODE"
echo "  $PATH_PARENT/slurm_script/$NAME_JOB_SCRIPT"
echo

sbatch \
  --job-name="$BASENAME_SCRIPT" \
  --output="$PATH_LOG_DIR"/%j.out \
  --error="$PATH_LOG_DIR"/%j.err \
  --export=NAME_PROJECT="$NAME_PROJECT",PATH_PYTHON_SCRIPT="$PATH_PYTHON_SCRIPT",ARGS_PYTHON_SCRIPT="$ARGS_PYTHON_SCRIPT",WORKDIR="$WORKDIR",PATH_CONTENT_ROOT="$PATH_CONTENT_ROOT",V_ENV_NAME="$V_ENV_NAME" \
  --cpus-per-task="$S_BATCH_CPU_PER_TASK" \
  --time="$S_BATCH_TIME" \
  --partition="$S_BATCH_PARTITION" \
  --ntasks-per-node="$S_BATCH_N_TASKS_PER_NODE" \
  "$PATH_PARENT"/slurm_script/"$NAME_JOB_SCRIPT"
