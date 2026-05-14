#!/bin/bash
#
# Open an interactive shell on an Adastra (CINES) compute node — MI250 GPU by
# default.  Uses ``salloc`` (the Adastra-documented entry point for
# interactive batch jobs) rather than ``srun --pty``.
#
# Useful for:
#   - testing the venv has the right ROCm-built PyTorch before a long batch
#   - quick numerical sanity checks
#   - debugging an OOM / crash that you cannot reproduce on the login node
#
# Usage:
#   bash run_interactive_job.sh -A <account>          # default: 1 GCD MI250, 30 min, shared
#   bash run_interactive_job.sh -A <account> -c GENOA # CPU instead of GPU
#
# Once the allocation is granted, the launcher prints the salloc command;
# ssh onto the assigned node and source the venv:
#   source $WORKDIR/pycharm_remote_project/constrained_learning_option_pricing/venv/venv_learning_option_pricing/bin/activate
#   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

NAME_PROJECT="constrained_learning_option_pricing"

WORKDIR="${WORKDIR:?WORKDIR env var is not set — are you on an Adastra login node?}"
PATH_PROJECT_PARENT_DIR_REL="${PATH_PROJECT_PARENT_DIR_REL:-pycharm_remote_project}"
PATH_CONTENT_ROOT="${PATH_CONTENT_ROOT:-$WORKDIR/$PATH_PROJECT_PARENT_DIR_REL/$NAME_PROJECT}"

V_ENV_NAME="venv_learning_option_pricing"
PATH_VENV_BIN="$PATH_CONTENT_ROOT"/venv/"$V_ENV_NAME"/bin/activate
if [ -f "$PATH_VENV_BIN" ]; then
  echo "Venv detected at $PATH_VENV_BIN — activate it on the compute node with:"
  echo "  source $PATH_VENV_BIN"
  echo
else
  echo "Note: venv not found at $PATH_VENV_BIN — set it up before running scripts."
  echo
fi

# --- Defaults ---
S_RUN_TIME="00:30:00"
S_RUN_ACCOUNT=""
S_RUN_CONSTRAINT="MI250"   # MI250 (GPU) by default; -c GENOA / MI300 / HPDA
S_RUN_CPU_PER_TASK=8
S_RUN_GPUS_PER_NODE=1
S_RUN_NODES=1

while getopts 'A:c:t:n:g:' flag; do
  case "${flag}" in
  A) S_RUN_ACCOUNT="${OPTARG}" ;;
  c) S_RUN_CONSTRAINT="${OPTARG}" ;;
  t) S_RUN_TIME="${OPTARG}" ;;
  n) S_RUN_NODES="${OPTARG}" ;;
  g) S_RUN_GPUS_PER_NODE="${OPTARG}" ;;
  *) echo "Unexpected option ${flag}" && exit 1 ;;
  esac
done

if [ -z "$S_RUN_ACCOUNT" ]; then
  echo "Missing -A <account>.  Discover yours with:"
  echo "  sacctmgr -nP list assoc where user=\$USER format=user,account,partition,defaultqos"
  exit 1
fi

# On CPU-only / HPDA constraints, omit --gpus-per-node.
SALLOC_GPU_FLAG=()
case "$S_RUN_CONSTRAINT" in
  MI250|MI300)
    SALLOC_GPU_FLAG=(--gpus-per-node="$S_RUN_GPUS_PER_NODE")
    ;;
  GENOA|HPDA)
    : # no GPU
    ;;
  *)
    echo "Warning: unknown constraint '$S_RUN_CONSTRAINT' (expected MI250|MI300|GENOA|HPDA)."
    ;;
esac

# Adastra's SLURM auto-routes the bare project to the per-constraint
# billing pool internally.  Defensively strip a suffix the user may
# have pasted from sacctmgr output — see _lib/account.sh.
PATH_PARENT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=_lib/account.sh
source "$PATH_PARENT/_lib/account.sh"
S_RUN_ACCOUNT="$(adastra_account_bare "$S_RUN_ACCOUNT")"

echo "salloc options (Adastra, shared mode — no --exclusive):"
echo "  --constraint=$S_RUN_CONSTRAINT"
echo "  --account=$S_RUN_ACCOUNT  (bare; SLURM routes to <project>_<lc(constraint)> internally)"
echo "  --nodes=$S_RUN_NODES"
echo "  --ntasks-per-node=1"
echo "  --cpus-per-task=$S_RUN_CPU_PER_TASK"
echo "  --threads-per-core=1"
if (( ${#SALLOC_GPU_FLAG[@]} > 0 )); then
  echo "  --gpus-per-node=$S_RUN_GPUS_PER_NODE"
fi
echo "  --time=$S_RUN_TIME"
echo

salloc \
  --constraint="$S_RUN_CONSTRAINT" \
  --account="$S_RUN_ACCOUNT" \
  --nodes="$S_RUN_NODES" \
  --ntasks-per-node=1 \
  --cpus-per-task="$S_RUN_CPU_PER_TASK" \
  --threads-per-core=1 \
  "${SALLOC_GPU_FLAG[@]+"${SALLOC_GPU_FLAG[@]}"}" \
  --time="$S_RUN_TIME" \
  --job-name="interactive_${USER}"
