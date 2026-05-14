#!/bin/bash
#
# Launch a single Python script on Adastra (CINES) — CPU (GENOA) hardware.
#
# Usage:
#   bash python_script_launcher.sh \
#       -A <account>                                    \
#       -p <path/to/script.py>                          \
#       -a "<space-separated args for the python script>"
#
# Example:
#   bash python_script_launcher.sh \
#       -A bae1234 \
#       -p experiments/python_scripts/exp_singularity_european_call/ablation_singularity_logS.py \
#       -a "--mode compare-boundary-singularity-european-call --add-variant naive --resume"
#
# Notes:
#   - For GPU work use python_script_launcher_gpu.sh (constraint=MI250).
#   - Adastra selects hardware via ``--constraint``; this launcher uses
#     ``GENOA`` (CPU) and never passes ``--qos`` (Adastra picks it
#     automatically — see CLAUDE.md / Adastra section).
#   - ``--account`` is mandatory; Adastra does NOT ship ``myproject`` —
#     discover projects with
#         sacctmgr -nP list assoc where user=$USER format=user,account,partition,defaultqos
#     (the active project is also visible as the third component of
#      $WORKDIR=/lus/work/<group>/<project>/<user>).
#   - Shared mode (omit ``--exclusive``) is enabled by default for CPU work
#     so a 4-core job bills only 8 cores instead of a whole 192-core node.

NAME_PROJECT="constrained_learning_option_pricing"
NAME_JOB_SCRIPT="run_python_script.slurm"

# WORKDIR is set by the CINES login environment to
# /lus/work/<group>/<project>/<user>.  Fail loudly if missing — that
# usually means we're not on a CINES login node.
WORKDIR="${WORKDIR:?WORKDIR env var is not set — are you on an Adastra login node?}"

PATH_PARENT=$(
  cd "$(dirname "${BASH_SOURCE[0]}")" || exit
  pwd -P
)
# Project parent directory under $WORKDIR — same convention as the array
# launcher.  Override by exporting PATH_CONTENT_ROOT before calling this script.
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

PATH_LOG_DIR="$WORKDIR"/logs/$NAME_PROJECT/"$BASENAME_SCRIPT"/$(date +"%Y-%m-%d_%H-%M-%S")
echo "Log directory: $PATH_LOG_DIR"
echo
mkdir -p "$PATH_LOG_DIR"

echo "Launching $NAME_JOB_SCRIPT"
echo
echo "PATH_PYTHON_SCRIPT: $PATH_PYTHON_SCRIPT"
echo "ARGS_PYTHON_SCRIPT: ${ARGS_PYTHON_SCRIPT:-}"
echo

# --- sbatch options (Adastra GENOA, shared mode) ---
# Constraint=GENOA selects the AMD EPYC 9654 scalar partition (192 cores/node).
# Smallest shared-mode billing is 8 cores; we pick exactly 8 here so we do
# not waste CPU-hours on a node we don't need exclusively.
S_BATCH_CPU_PER_TASK=8
S_BATCH_SLURM_NTASKS=1
S_BATCH_TIME=05:00:00
S_BATCH_CONSTRAINT=GENOA

echo "sbatch options:"
echo "  --job-name=$BASENAME_SCRIPT"
echo "  --output=$PATH_LOG_DIR/%j.out"
echo "  --error=$PATH_LOG_DIR/%j.err"
echo "  --constraint=$S_BATCH_CONSTRAINT  # CPU GENOA, shared mode (no --exclusive)"
echo "  --account=$S_BATCH_ACCOUNT"
echo "  --cpus-per-task=$S_BATCH_CPU_PER_TASK"
echo "  --threads-per-core=1"
echo "  --time=$S_BATCH_TIME"
echo "  $PATH_PARENT/slurm_script/$NAME_JOB_SCRIPT"
echo
echo "Tail the log in real time after submission:"
echo "  tail -f $PATH_LOG_DIR/<jobid>.out"
echo

sbatch \
  --job-name="$BASENAME_SCRIPT" \
  --output="$PATH_LOG_DIR"/%j.out \
  --error="$PATH_LOG_DIR"/%j.err \
  --export=NAME_PROJECT="$NAME_PROJECT",PATH_PYTHON_SCRIPT="$PATH_PYTHON_SCRIPT",ARGS_PYTHON_SCRIPT="${ARGS_PYTHON_SCRIPT:-}",WORKDIR="$WORKDIR",PATH_CONTENT_ROOT="$PATH_CONTENT_ROOT",V_ENV_NAME="$V_ENV_NAME" \
  --constraint="$S_BATCH_CONSTRAINT" \
  --account="$S_BATCH_ACCOUNT" \
  --nodes=1 \
  --ntasks-per-node="$S_BATCH_SLURM_NTASKS" \
  --cpus-per-task="$S_BATCH_CPU_PER_TASK" \
  --threads-per-core=1 \
  --time="$S_BATCH_TIME" \
  "$PATH_PARENT"/slurm_script/"$NAME_JOB_SCRIPT"
