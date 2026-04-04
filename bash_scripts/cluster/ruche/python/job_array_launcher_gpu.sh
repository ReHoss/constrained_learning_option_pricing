#!/bin/bash

# Load modules
module purge
module load miniconda3/23.5.2/gcc-13.2.0

# Name of the project
NAME_PROJECT="drloc-sb3"
# Name of the job array script
NAME_JOB_ARRAY_SCRIPT="job_array_batch_xp.slurm"

PATH_CONDA_BIN=/gpfs/softs/spack_0.17/opt/spack/linux-centos7-cascadelake/gcc-13.2.0/miniconda3-23.5.2-scvtvts2zr4k27oespcarh43r6zcswmf/bin/conda
PATH_PARENT=$(
  cd "$(dirname "${BASH_SOURCE[0]}")" || exit
  pwd -P
)
# Path to the project's content root directory
PATH_CONTENT_ROOT="$WORKDIR"/pycharm_remote_project/"$NAME_PROJECT"

# Get the name of the conda environment
CONDA_ENV=$(cat "$PATH_CONTENT_ROOT"/bash_scripts/conda_env_name.txt)
echo Conda environment: "$CONDA_ENV"
echo

# Get the folder name from the command line
while getopts 'n:' flag; do
  case "${flag}" in
  n) FOLDER_NAME="${OPTARG}" ;;
  *) error "Unexpected option ${flag}" ;;
  esac
done

# Check that the folder name was provided
if [ -z "$FOLDER_NAME" ]; then
  echo Missing folder name -n option.
  exit
fi

PATH_FOLDER_CONFIGS="$PATH_CONTENT_ROOT"/configs/batch/"$FOLDER_NAME"

echo "Config folder: $PATH_FOLDER_CONFIGS"
echo

# Get the number of yaml files in the folder PATH_FOLDER_CONFIGS
N_CONFIGS=$(find "$PATH_FOLDER_CONFIGS" -name "*.yaml" | wc -l)
echo "Number of configs: $N_CONFIGS"

# Create the name of the log directory with the current date and time
PATH_LOG_DIR="$WORKDIR"/logs/$NAME_PROJECT/"$FOLDER_NAME"/$(date +"%Y-%m-%d_%H-%M-%S")

echo "Log directory: $PATH_LOG_DIR"
echo

# Create the log directory for the current config file
mkdir -p "$PATH_LOG_DIR"/"$CONFIG_FILE_NAME"

# Create MLFlow experiments given the xp_name entry from the first yaml file in PATH_FOLDER_CONFIGS
echo "Creating MLFlow experiments..."
echo

# Get the first yaml file in PATH_FOLDER_CONFIGS with find program
PATH_CONFIG_FILE=$(find "$PATH_FOLDER_CONFIGS" -name "*.yaml" -print -quit)

# Get the xp_name entry from the yaml file
XP_NAME=$(grep -oP '(?<=xp_name: ).*' "$PATH_CONFIG_FILE")

# Set the MLFlow tracking URI
export MLFLOW_TRACKING_URI=file:"$PATH_CONTENT_ROOT"/data/mlruns

# Create the MLFlow experiments
# If last command failed, then the experiment already exists
if ! $PATH_CONDA_BIN run --no-capture-output --name "$CONDA_ENV" mlflow experiments create --experiment-name "$XP_NAME"; then
  echo "Experiment $XP_NAME already exists (or command failed?)."
  echo
else
  echo "Experiment $XP_NAME created."
  echo
fi

# Set defaults values for the sbatch options
# --- Number of CPUs per task ---
S_BATCH_CPU_PER_TASK=4

# --- Time limit ---
#S_BATCH_TIME=19:59:00
S_BATCH_TIME=3:59:00
#S_BATCH_TIME=9:59:00
#S_BATCH_TIME=48:00:00
#S_BATCH_TIME=00:10:00

# --- Partition ---
#S_BATCH_PARTITION=cpu_med
S_BATCH_PARTITION=cpu_long  # (12 cores at 3.2 GHz), namely 48 cores per node

# --- Quality of service ---
#S_BATCH_QOS=qos_cpu-t3
#S_BATCH_QOS=qos_cpu-t4
#S_BATCH_QOS=qos_cpu-dev

# --- Account ---
S_BATCH_ACCOUNT=rl_for_dy+

# Ruche specific options
# --- Number of nodes ---
#S_BATCH_NODES=1

# --- Number of tasks ---
#S_BATCH_N_TASKS=1

# --- Number of tasks per node ---
S_BATCH_N_TASKS_PER_NODE=1

# --- Number of GPUs ---
S_BATCH_GPUS=0

# --- Memory per node ---
#S_BATCH_MEM_PER_NODE



# Get last array ID
N_LAST_ARRAYID=$((N_CONFIGS - 1))

echo "sbatch options:"
echo "  --job-name=$FOLDER_NAME"
echo "  --output=$PATH_LOG_DIR/job_array_launcher_%A_%a.out"
echo "  --error=$PATH_LOG_DIR/job_array_launcher_%A_%a.err"
echo "  --export=NAME_PROJECT,PATH_PYTHON_SCRIPT,PATH_FOLDER_CONFIGS" #TODO: Add the remaining variables here
echo "  --cpus-per-task=$S_BATCH_CPU_PER_TASK"
echo "  --time=$S_BATCH_TIME"
#echo "  --partition=$S_BATCH_PARTITION"
echo "  --qos=$S_BATCH_QOS"
echo "  --account=$S_BATCH_ACCOUNT"
echo "  --array=0-$N_LAST_ARRAYID"
echo "  --nodes=$S_BATCH_NODES"
echo "  --ntasks-per-node=$S_BATCH_N_TASKS_PER_NODE"
echo "  --gres=gpu:$S_BATCH_GPUS"
echo "  $PATH_PARENT/slurm_job_array/$NAME_JOB_ARRAY_SCRIPT"
echo


sbatch \
  --job-name="$FOLDER_NAME" \
  --array=0-"$N_LAST_ARRAYID" \
  --output="$PATH_LOG_DIR"/job_array_launcher_%A_%a.out \
  --error="$PATH_LOG_DIR"/job_array_launcher_%A_%a.err \
  --export=NAME_PROJECT="$NAME_PROJECT",PATH_PYTHON_SCRIPT="$PATH_PYTHON_SCRIPT",PATH_FOLDER_CONFIGS="$PATH_FOLDER_CONFIGS",WORKDIR="$WORKDIR" \
  --cpus-per-task="$S_BATCH_CPU_PER_TASK" \
  --time="$S_BATCH_TIME" \
  --nodes="$S_BATCH_NODES" \
  --ntasks-per-node="$S_BATCH_N_TASKS_PER_NODE" \
  --gres=gpu:"$S_BATCH_GPUS" \
  "$PATH_PARENT"/slurm_job_array/"$NAME_JOB_ARRAY_SCRIPT"

# Note: The following options are not needed with Ruche
#  --qos="$S_BATCH_QOS" \
#  --account="$S_BATCH_ACCOUNT" \
