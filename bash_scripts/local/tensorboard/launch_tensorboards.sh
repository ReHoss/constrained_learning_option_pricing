#!/bin/bash

# Path of the parent directory of this script
PATH_PARENT=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit ; pwd -P )
# Path of the root of the project
PATH_CONTENT_ROOT=$(realpath "$PATH_PARENT"/../../..)

V_ENV_NAME="venv_control_dde"
PATH_VENV="$PATH_CONTENT_ROOT"/venv/"$V_ENV_NAME"/bin/activate

# Define the port to use for the MLFlow UI

# Path of the "store", i.e., where the mlflow runs are stored
PATH_BACKEND_STORE="$PATH_CONTENT_ROOT"/data/mlruns

echo "Loading virtual environment"
echo

# Load virtual environment
source "$PATH_VENV"

echo PATH_VENV: "$PATH_VENV"
echo
echo PATH_CONTENT_ROOT: "$PATH_CONTENT_ROOT"
echo

while getopts 'i:' flag; do
  case "${flag}" in
    i) EXPERIMENT_ID="${OPTARG}" ;;
    *) error "Unexpected option ${flag}" ;;
  esac
done

tensorboard --logdir "$PATH_BACKEND_STORE"/"$EXPERIMENT_ID"
