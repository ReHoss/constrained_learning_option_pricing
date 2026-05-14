#!/bin/bash
#
# SLURM job script — runs a single Python script inside the project venv,
# on Adastra (CINES).  Invoked by python_script_launcher.sh (CPU/GENOA) or
# python_script_launcher_gpu.sh (MI250).
#
# Resource directives (constraint / account / time / cpus / gpus / exclusive)
# are passed by the launcher's sbatch command — this script intentionally
# carries NO ``#SBATCH --constraint`` / ``--account`` / ``--qos`` lines so
# it can be reused across constraints, projects, and shared/exclusive modes.
# Adastra picks the QoS automatically (see CLAUDE.md / Adastra section).
#
# Required environment variables (broadcast via --export):
#   NAME_PROJECT, PATH_PYTHON_SCRIPT, ARGS_PYTHON_SCRIPT, WORKDIR,
#   PATH_CONTENT_ROOT, V_ENV_NAME

#SBATCH --job-name=batch_experiment_adastra_default
#SBATCH --mail-type=ALL
#SBATCH --mail-user=remy.hosseinkhan@polytechnique.edu

# --- Required env-var checks ------------------------------------------------
for var in NAME_PROJECT PATH_PYTHON_SCRIPT WORKDIR PATH_CONTENT_ROOT V_ENV_NAME; do
  if [ -z "${!var:-}" ]; then
    echo "The environment variable $var is not defined" >&2
    exit 1
  fi
done

PATH_PARENT=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit ; pwd -P )

# --- Adastra module loads — REQUIRED before sourcing the venv --------------
# The sbatch shell is non-interactive and does NOT source /etc/profile.d/,
# so cray-python + the ROCm runtime must be loaded explicitly.  See the
# matching block in slurm_job_array/job_array_batch_xp.slurm for the
# rationale (Python ABI match + torch==2.9.1+rocm6.4 ROCm matching).
# Loading amd-mixed on a GENOA-only run is harmless (unused libs on
# LD_LIBRARY_PATH); this keeps a single worker script reusable across
# constraints.
module load cray-python/3.11.7 amd-mixed/6.4.3
echo "module list (after explicit loads):"
module list 2>&1 || true
echo

# --- Activate the project venv (pure venv, no conda) -----------------------
PATH_VENV_BIN="$PATH_CONTENT_ROOT"/venv/"$V_ENV_NAME"/bin/activate
echo "PATH_VENV_BIN: $PATH_VENV_BIN"
echo
# shellcheck disable=SC1090
source "$PATH_VENV_BIN" && echo "Activation of virtual environment: $V_ENV_NAME"
echo

# --- Diagnostics: self-contained log ----------------------------------------
echo "Starting $SLURM_JOB_NAME on $SLURMD_NODENAME with the following parameters:"
echo "PWD:               $PWD"
echo "WORKDIR:           $WORKDIR"
echo "PATH_PARENT:       $PATH_PARENT"
echo "PATH_CONTENT_ROOT: $PATH_CONTENT_ROOT"
echo
echo "Python version:    $(python --version 2>&1)"
echo "Python executable: $(which python)"
# On Adastra GPUs PyTorch must be a ROCm build; .cuda.is_available() actually
# reflects ROCm/HIP availability when torch is built with ROCm.
python -c "import torch; print(f'PyTorch:          {torch.__version__}'); \
print(f'HIP/CUDA avail.:  {torch.cuda.is_available()}'); \
print(f'HIP version:      {getattr(torch.version, \"hip\", None)}'); \
print(f'CUDA toolkit:     {torch.version.cuda}'); \
print(f'GPU:              {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else None" 2>/dev/null || \
  echo "(torch not importable in this environment)"
echo
echo "SLURM_SUBMIT_HOST:      ${SLURM_SUBMIT_HOST:-}"
echo "SLURM_JOB_QOS:          ${SLURM_JOB_QOS:-}  (assigned by Adastra automatically)"
echo "SLURM_SUBMIT_DIR:       ${SLURM_SUBMIT_DIR:-}"
echo "SLURM_JOB_ID:           ${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NAME:         ${SLURM_JOB_NAME:-}"
echo "SLURM_JOB_ACCOUNT:      ${SLURM_JOB_ACCOUNT:-}"
echo "SLURM_JOB_PARTITION:    ${SLURM_JOB_PARTITION:-}"
echo "SLURM_JOB_CONSTRAINTS:  ${SLURM_JOB_CONSTRAINTS:-}"
echo "SLURM_JOB_NODELIST:     ${SLURM_JOB_NODELIST:-}"
echo "SLURM_JOB_NUM_NODES:    ${SLURM_JOB_NUM_NODES:-}"
echo "SLURM_NTASKS:           ${SLURM_NTASKS:-}"
echo "SLURM_NTASKS_PER_NODE:  ${SLURM_NTASKS_PER_NODE:-}"
echo "SLURM_CPUS_ON_NODE:     ${SLURM_CPUS_ON_NODE:-}"
echo "SLURM_CPUS_PER_TASK:    ${SLURM_CPUS_PER_TASK:-}"
echo "SLURM_GPUS_PER_NODE:    ${SLURM_GPUS_PER_NODE:-}"
echo "SLURM_RESTART_COUNT:    ${SLURM_RESTART_COUNT:-}"
echo

# Add the project root to PYTHONPATH so `learning_option_pricing` is importable
export PYTHONPATH="${PYTHONPATH:-}:$PATH_CONTENT_ROOT"

IFS=" " read -r -a LIST_ARGS_PYTHON_SCRIPT <<< "$ARGS_PYTHON_SCRIPT"

echo "Running: python $PATH_PYTHON_SCRIPT ${LIST_ARGS_PYTHON_SCRIPT[*]}"
echo
python "$PATH_PYTHON_SCRIPT" "${LIST_ARGS_PYTHON_SCRIPT[@]}"
echo
