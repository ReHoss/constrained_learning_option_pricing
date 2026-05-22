#!/bin/bash
#
# Launch a single Python script on Jean Zay (IDRIS) — CPU partition.
#
# Usage:
#   bash python_script_launcher.sh \
#       -p <path/to/script.py> \
#       -a "<space-separated args for the python script>" \
#       [-q <qos>] [-t <HH:MM:SS>] [-P <partition>]
#
# Examples:
#   # Default (cpu_p1 + qos_cpu-t3, 5h)
#   bash python_script_launcher.sh -p .../train.py -a "--device cpu"
#
#   # Smoke test on the dev QoS (higher priority, 2h cap)
#   bash python_script_launcher.sh -p .../train.py -a "--debug" \
#       -q qos_cpu-dev -t 00:10:00
#
#   # Non-billed prepost partition for replot / aggregation
#   bash python_script_launcher.sh -p .../replot.py -a "..." -P prepost -t 01:00:00
#
# Notes:
#   - For GPU work, use python_script_launcher_gpu.sh instead.
#   - This launcher expects a Python virtual environment at
#       $PATH_CONTENT_ROOT/venv/$V_ENV_NAME
#     activated via `source bin/activate` (no conda).

# Name of the project — must match the directory name on the cluster
NAME_PROJECT="learning_option_pricing"
# Name of the slurm job script to launch
NAME_JOB_SCRIPT="run_python_script.slurm"

# Alias for workdir on Jean Zay
WORKDIR=$WORK

PATH_PARENT=$(
  cd "$(dirname "${BASH_SOURCE[0]}")" || exit
  pwd -P
)
# Path to the project's content root directory on the cluster.
# Override by exporting PATH_CONTENT_ROOT before calling this script.
PATH_CONTENT_ROOT="${PATH_CONTENT_ROOT:-$WORKDIR/git_repositories/$NAME_PROJECT}"

# Name of the Python virtual environment (relative to PATH_CONTENT_ROOT/venv/).
V_ENV_NAME="venv_learning_option_pricing"
PATH_VENV_BIN="$PATH_CONTENT_ROOT"/venv/"$V_ENV_NAME"/bin/activate
echo "PATH_VENV_BIN: $PATH_VENV_BIN"
echo
# Activate the virtual environment (no conda required)
# shellcheck disable=SC1090
source "$PATH_VENV_BIN" && echo "Activation of virtual environment: $V_ENV_NAME"
echo


# --- sbatch defaults (overridable via CLI flags below) ---
S_BATCH_CPU_PER_TASK=4
S_BATCH_SLURM_NTASKS=1
S_BATCH_TIME=05:00:00
# Partition: cpu_p1 (general) or prepost (4h max, large memory, no GPU)
S_BATCH_PARTITION=cpu_p1
S_BATCH_QOS=qos_cpu-t3
# Account: depends on your allocation — check `idracct` on Jean Zay
# Typical forms: <project>@cpu, <project>@v100, <project>@a100, <project>@h100
S_BATCH_ACCOUNT=akz@cpu

# Parse command-line options
while getopts 'p:a:q:t:P:A:' flag; do
  case "${flag}" in
  p) PATH_PYTHON_SCRIPT="${OPTARG}" ;;
  a) ARGS_PYTHON_SCRIPT="${OPTARG}" ;;
  q) S_BATCH_QOS="${OPTARG}" ;;
  t) S_BATCH_TIME="${OPTARG}" ;;
  P) S_BATCH_PARTITION="${OPTARG}" ;;
  A) S_BATCH_ACCOUNT="${OPTARG}" ;;
  *) echo "Unexpected option ${flag}" && exit 1 ;;
  esac
done

# Check PATH_PYTHON_SCRIPT is not empty
if [ -z "$PATH_PYTHON_SCRIPT" ]; then
  echo "Missing -p <path/to/script.py> option."
  exit 1
fi

# Get the basename of the python script without the extension
BASENAME_SCRIPT=$(basename "$PATH_PYTHON_SCRIPT" .py)
echo "Script basename: $BASENAME_SCRIPT"
echo

# Create the name of the log directory with the current date and time
PATH_LOG_DIR="$WORKDIR"/logs/$NAME_PROJECT/"$BASENAME_SCRIPT"/$(date +"%Y-%m-%d_%H-%M-%S")
echo "Log directory: $PATH_LOG_DIR"
echo
mkdir -p "$PATH_LOG_DIR"

echo "Launching $NAME_JOB_SCRIPT"
echo
echo "PATH_PYTHON_SCRIPT: $PATH_PYTHON_SCRIPT"
echo "ARGS_PYTHON_SCRIPT: $ARGS_PYTHON_SCRIPT"
echo

echo "sbatch options:"
echo "  --job-name=$BASENAME_SCRIPT"
echo "  --output=$PATH_LOG_DIR/%j.out"
echo "  --error=$PATH_LOG_DIR/%j.err"
echo "  --cpus-per-task=$S_BATCH_CPU_PER_TASK"
echo "  --time=$S_BATCH_TIME"
echo "  --partition=$S_BATCH_PARTITION"
echo "  --qos=$S_BATCH_QOS"
echo "  --account=$S_BATCH_ACCOUNT"
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
  --qos="$S_BATCH_QOS" \
  --account="$S_BATCH_ACCOUNT" \
  "$PATH_PARENT"/slurm_script/"$NAME_JOB_SCRIPT"
