#!/bin/bash
#
# SLURM job script — runs a single Python script inside the project venv.
# Invoked by python_script_launcher.sh (CPU) or python_script_launcher_gpu.sh.
#
# Required environment variables (broadcast via --export):
#   NAME_PROJECT, PATH_PYTHON_SCRIPT, ARGS_PYTHON_SCRIPT, WORKDIR,
#   PATH_CONTENT_ROOT, V_ENV_NAME

#SBATCH --job-name=batch_experiment_hosseinkhan_default
#SBATCH --mail-type=ALL
#SBATCH --mail-user=remy.hosseinkhan@lisn.fr

# --- Required env-var checks ------------------------------------------------
for var in NAME_PROJECT PATH_PYTHON_SCRIPT WORKDIR PATH_CONTENT_ROOT V_ENV_NAME; do
  if [ -z "${!var}" ]; then
    echo "The environment variable $var is not defined" >&2
    exit 1
  fi
done

PATH_PARENT=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit ; pwd -P )

# --- Activate the project venv (pure venv, no conda) -----------------------
PATH_VENV_BIN="$PATH_CONTENT_ROOT"/venv/"$V_ENV_NAME"/bin/activate
echo "PATH_VENV_BIN: $PATH_VENV_BIN"
echo
# shellcheck disable=SC1090
source "$PATH_VENV_BIN" && echo "Activation of virtual environment: $V_ENV_NAME"
echo

# --- Diagnostics: self-contained log ----------------------------------------
# Aligned with CLAUDE.md: every log file must capture the full environment.
echo "Starting $SLURM_JOB_NAME on $SLURMD_NODENAME with the following parameters:"
echo "PWD:               $PWD"
echo "WORKDIR:           $WORKDIR"
echo "PATH_PARENT:       $PATH_PARENT"
echo "PATH_CONTENT_ROOT: $PATH_CONTENT_ROOT"
echo
echo "Python version:    $(python --version 2>&1)"
echo "Python executable: $(which python)"
python -c "import torch; print(f'PyTorch:          {torch.__version__}'); \
print(f'CUDA available:   {torch.cuda.is_available()}'); \
print(f'CUDA toolkit:     {torch.version.cuda}') if torch.cuda.is_available() else None; \
print(f'GPU:              {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else None" 2>/dev/null || \
  echo "(torch not importable in this environment)"
echo
echo "SLURM_SUBMIT_HOST:      $SLURM_SUBMIT_HOST"
echo "SLURM_JOB_QOS:          $SLURM_JOB_QOS"
echo "SLURM_SUBMIT_DIR:       $SLURM_SUBMIT_DIR"
echo "SLURM_JOB_ID:           $SLURM_JOB_ID"
echo "SLURM_JOB_NAME:         $SLURM_JOB_NAME"
echo "SLURM_JOB_ACCOUNT:      $SLURM_JOB_ACCOUNT"
echo "SLURM_JOB_PARTITION:    $SLURM_JOB_PARTITION"
echo "SLURM_JOB_NODELIST:     $SLURM_JOB_NODELIST"
echo "SLURM_JOB_NUM_NODES:    $SLURM_JOB_NUM_NODES"
echo "SLURM_NTASKS:           $SLURM_NTASKS"
echo "SLURM_NTASKS_PER_NODE:  $SLURM_NTASKS_PER_NODE"
echo "SLURM_MEM_PER_CPU:      $SLURM_MEM_PER_CPU"
echo "SLURM_MEM_PER_NODE:     $SLURM_MEM_PER_NODE"
echo "SLURM_CPUS_ON_NODE:     $SLURM_CPUS_ON_NODE"
echo "SLURM_JOB_CPUS_PER_NODE:$SLURM_JOB_CPUS_PER_NODE"
echo "SLURM_GPUS:             $SLURM_GPUS"
echo "SLURM_JOB_GPUS:         $SLURM_JOB_GPUS"
echo "SLURM_RESTART_COUNT:    $SLURM_RESTART_COUNT"
echo

# Add the project root to PYTHONPATH so `learning_option_pricing` is importable
export PYTHONPATH="${PYTHONPATH}:$PATH_CONTENT_ROOT"

# Split args (ARGS_PYTHON_SCRIPT is broadcast by sbatch --export)
IFS=" " read -r -a LIST_ARGS_PYTHON_SCRIPT <<< "$ARGS_PYTHON_SCRIPT"

echo "Running: python $PATH_PYTHON_SCRIPT ${LIST_ARGS_PYTHON_SCRIPT[*]}"
echo
python "$PATH_PYTHON_SCRIPT" "${LIST_ARGS_PYTHON_SCRIPT[@]}"
echo
