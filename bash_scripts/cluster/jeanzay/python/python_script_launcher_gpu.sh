#!/bin/bash
#
# Launch a single Python script on Jean Zay (IDRIS) — GPU partition.
#
# Usage:
#   bash python_script_launcher_gpu.sh \
#       -p <path/to/script.py> \
#       -a "<space-separated args for the python script>" \
#       [-q <qos>] [-t <HH:MM:SS>] [-A <account>]
#
# Examples:
#   # Default (V100, qos_gpu-t3, 2h)
#   bash python_script_launcher_gpu.sh -p .../train.py -a "..."
#
#   # Smoke test on the dev QoS (higher priority, 2h cap)
#   bash python_script_launcher_gpu.sh -p .../train.py -a "..." \
#       -q qos_gpu-dev -t 00:10:00
#
#   # A100 allocation
#   bash python_script_launcher_gpu.sh -p .../train.py -a "..." -A akz@a100
#
# Reuses the slurm script slurm_script/run_python_script.slurm (the same one
# used by the CPU launcher) — only the sbatch options differ here.

NAME_PROJECT="learning_option_pricing"
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

# --- sbatch defaults (overridable via CLI flags below) ---
# 4 cores is the standard ratio for 1 GPU on the V100 partition (40 cores / 4 GPUs).
S_BATCH_CPU_PER_TASK=4
# 1 GPU per job — variants are submitted independently to run in parallel.
S_BATCH_GPUS=1
# Time: most variants converge in <2h on V100; bump to 19:59:00 for long runs.
S_BATCH_TIME=02:00:00
# QoS: qos_gpu-t3 → 20h max ; qos_gpu-dev → 2h, higher priority for testing.
S_BATCH_QOS=qos_gpu-t3
# Account: depends on your allocation — check `idracct` on Jean Zay.
# Forms: <project>@v100, <project>@a100, <project>@h100
S_BATCH_ACCOUNT=akz@v100

# Parse command-line options
while getopts 'p:a:q:t:A:' flag; do
  case "${flag}" in
  p) PATH_PYTHON_SCRIPT="${OPTARG}" ;;
  a) ARGS_PYTHON_SCRIPT="${OPTARG}" ;;
  q) S_BATCH_QOS="${OPTARG}" ;;
  t) S_BATCH_TIME="${OPTARG}" ;;
  A) S_BATCH_ACCOUNT="${OPTARG}" ;;
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
# A short variant tag is extracted from ARGS_PYTHON_SCRIPT, matching either
#   --variant NAME           (single-variant standalone run)
#   --add-variant NAME:DIR   (append to existing ablation dir)
# In both cases we keep just NAME (the part before the first colon).
# `\K` resets the match-start so the regex stays fixed-length (some PCRE
# builds reject variable-length lookbehind with a warning otherwise).
VARIANT_TAG=$(echo "$ARGS_PYTHON_SCRIPT" | grep -oP -- '--(?:add-)?variant +\K\S+' | head -n1 | cut -d: -f1)
PATH_LOG_DIR="$WORKDIR"/logs/$NAME_PROJECT/"$BASENAME_SCRIPT"/$(date +"%Y-%m-%d_%H-%M-%S")${VARIANT_TAG:+_$VARIANT_TAG}
echo "Log directory: $PATH_LOG_DIR"
echo
mkdir -p "$PATH_LOG_DIR"

echo "Launching $NAME_JOB_SCRIPT (GPU)"
echo
echo "PATH_PYTHON_SCRIPT: $PATH_PYTHON_SCRIPT"
echo "ARGS_PYTHON_SCRIPT: $ARGS_PYTHON_SCRIPT"
echo

S_BATCH_NODES=1
S_BATCH_N_TASKS_PER_NODE=1
# Resume-friendly: SLURM will requeue the job on preemption/time-limit; the
# Python script's --resume flag (if present) picks up from the last checkpoint.
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
