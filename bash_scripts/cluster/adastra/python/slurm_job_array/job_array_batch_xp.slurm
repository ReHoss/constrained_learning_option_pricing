#!/bin/bash
# =============================================================================
# Adastra (CINES) generic per-task worker for sbatch arrays.
# Invoked by ../experiment_array_launcher.sh; reads the YAML config matching
# $SLURM_ARRAY_TASK_ID and runs the target Python script on it.
#
# All SLURM resource directives (constraint / account / time / cpus / gpus /
# exclusive) are passed by the launcher's sbatch command — this script
# intentionally carries NO ``#SBATCH --constraint`` / ``--account`` /
# ``--qos`` lines so it can be reused across constraints and projects.
# Adastra picks the QoS automatically, never set ``--qos=`` here.
# =============================================================================

#SBATCH --job-name=batch_experiment_adastra_default_file
#SBATCH --mail-type=ALL
#SBATCH --mail-user=remy.hosseinkhan@polytechnique.edu

# The launcher exports the following env vars via sbatch --export=...
#   NAME_PROJECT, PATH_PYTHON_SCRIPT, PATH_FOLDER_CONFIGS, WORKDIR,
#   V_ENV_NAME, PATH_PROJECT_PARENT_DIR_REL

for var in NAME_PROJECT PATH_PYTHON_SCRIPT PATH_FOLDER_CONFIGS WORKDIR V_ENV_NAME PATH_PROJECT_PARENT_DIR_REL; do
    if [ -z "${!var:-}" ]; then
        echo "The environment variable $var is not defined" >&2
        echo "Set it in the launcher's --export, e.g. $var=..." >&2
        exit 1
    fi
done

PATH_PARENT=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit ; pwd -P )
PATH_CONTENT_ROOT="$WORKDIR/$PATH_PROJECT_PARENT_DIR_REL/$NAME_PROJECT"
PATH_VENV_BIN="$PATH_CONTENT_ROOT/venv/$V_ENV_NAME/bin/activate"

echo "PATH_VENV_BIN: $PATH_VENV_BIN"
echo
# shellcheck source=/dev/null
source "$PATH_VENV_BIN" && echo "Activation of virtual environment: $V_ENV_NAME"
echo

# ── Diagnostic dump ─────────────────────────────────────────────────────────
echo "Starting $SLURM_JOB_NAME on $SLURMD_NODENAME with the following parameters:"
echo "PWD:               $PWD"
echo "WORKDIR:           $WORKDIR"
echo "PATH_PARENT:       $PATH_PARENT"
echo
echo "SLURM_SUBMIT_HOST:     ${SLURM_SUBMIT_HOST:-}"
echo "SLURM_JOB_QOS:         ${SLURM_JOB_QOS:-}  (assigned by Adastra automatically)"
echo "SLURM_SUBMIT_DIR:      ${SLURM_SUBMIT_DIR:-}"
echo "SLURM_JOB_ID:          ${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_JOB_ID:    ${SLURM_ARRAY_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID:   ${SLURM_ARRAY_TASK_ID:-}"
echo "SLURM_JOB_NAME:        ${SLURM_JOB_NAME:-}"
echo "SLURM_JOB_ACCOUNT:     ${SLURM_JOB_ACCOUNT:-}"
echo "SLURM_JOB_PARTITION:   ${SLURM_JOB_PARTITION:-}"
echo "SLURM_JOB_NODELIST:    ${SLURM_JOB_NODELIST:-}"
echo "SLURM_JOB_NUM_NODES:   ${SLURM_JOB_NUM_NODES:-}"
echo "SLURM_NTASKS:          ${SLURM_NTASKS:-}"
echo "SLURM_NTASKS_PER_NODE: ${SLURM_NTASKS_PER_NODE:-}"
echo "SLURM_CPUS_PER_TASK:   ${SLURM_CPUS_PER_TASK:-}"
echo "SLURM_GPUS_PER_NODE:   ${SLURM_GPUS_PER_NODE:-}"
echo "SLURM_JOB_CONSTRAINTS: ${SLURM_JOB_CONSTRAINTS:-}"
echo

# Add the project root to the PYTHONPATH
export PYTHONPATH="${PYTHONPATH:-}:$PATH_CONTENT_ROOT"

echo "Fetching the config file name from PATH_FOLDER_CONFIGS: $PATH_FOLDER_CONFIGS"
echo
mapfile -t LIST_PATH_CONFIGS < <(find "$PATH_FOLDER_CONFIGS" -type f -name "*.yaml" | sort)
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo

PATH_CONFIG="${LIST_PATH_CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "PATH_CONFIG: $PATH_CONFIG"
echo

# Stagger task starts to avoid races on shared files (summary.yaml,
# gt_comparison.npz, etc.) when many tasks initialise at once.
SLEEP_TIME=30
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    SLEEP_TIME=0
fi
SLEEP_TIME=$((SLEEP_TIME + SLURM_ARRAY_TASK_ID * 5))
echo "Sleeping for $SLEEP_TIME seconds"
echo
sleep "$SLEEP_TIME"
echo "Done sleeping"
echo

echo "Running the main script with the config file: $PATH_CONFIG"
echo
echo "Python version: $(python --version)"
python "$PATH_PYTHON_SCRIPT" --config-dir="$PATH_FOLDER_CONFIGS" --config-name="$(basename "$PATH_CONFIG" .yaml)"
