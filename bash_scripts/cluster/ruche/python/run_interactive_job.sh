#!/bin/bash

# Name of the project
NAME_PROJECT="control_dde"

# Path to the project's content root directory
PATH_CONTENT_ROOT="$WORKDIR"/pycharm_remote_project/"$NAME_PROJECT"

# Get the venv name from the command line
V_ENV_NAME="venv_control_dde"
# Get the path of the virtual environment
PATH_VENV_BIN="$PATH_CONTENT_ROOT"/venv/"$V_ENV_NAME"/bin/activate
echo PATH_VENV_BIN: "$PATH_VENV_BIN"
echo
# Activate the virtual environment, if working echo the name of the venv
# shellcheck source=/home/hosseinkhan/Documents/work/phd/git_repositories/control_dde/venv/venv_control_dde/bin/activate
source "$PATH_VENV_BIN" && echo "Activation of virtual environment: $V_ENV_NAME"
echo
# Echo the python version
echo
echo Python version: "$(python --version)"

# Set defaults values for the sbatch options
# --- Number of CPUs per task ---
#S_RUN_CPU_PER_TASK=1

# --- Time limit ---
S_RUN_TIME=00:30:00

# --- Partition ---
S_RUN_PARTITION=cpu_short
#S_RUN_PARTITION=cpu_med
#S_RUN_PARTITION=cpu_long  # (12 cores at 3.2 GHz), namely 48 cores per node
#S_RUN_PARTITION=gpup100

# --- Quality of service ---
#S_RUN_QOS=qos_cpu-t3
#S_RUN_QOS=qos_cpu-t4
#S_RUN_QOS=qos_cpu-dev

# --- Account ---
S_RUN_ACCOUNT=rl_for_dy+

# Ruche specific options
# --- Number of nodes ---
S_RUN_NODES=1

# --- Number of tasks ---
#S_RUN_N_TASKS=1

# --- Number of tasks per node ---
#S_RUN_N_TASKS_PER_NODE=1

# --- Number of GPUs ---
S_RUN_GPUS=0

# --- Memory per node ---
#S_RUN_MEM_PER_NODE



echo "srun options:"
echo "  --nodes=$S_RUN_NODES"
echo "  --time=$S_RUN_TIME"
echo "  --partition=$S_RUN_PARTITION"
echo "  --gpu=$S_RUN_GPUS"

srun \
  --nodes="$S_RUN_NODES" \
  --time="$S_RUN_TIME" \
  --partition="$S_RUN_PARTITION" \
  --pty \
  /bin/bash