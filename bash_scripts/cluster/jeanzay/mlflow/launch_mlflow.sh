#!/bin/bash

module load anaconda-py3/2024.06

PATH_PARENT=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit ; pwd -P )
PATH_CONTENT_ROOT="$PATH_PARENT/../../../.."

CONDA_ENV=$(cat "$PATH_CONTENT_ROOT"/bash_scripts/conda_env_name.txt)

# Define the port to use for the MLFlow UI
PORT=5001

echo Conda environment: "$CONDA_ENV"
echo
echo PATH_CONTENT_ROOT: "$PATH_CONTENT_ROOT"
echo

echo "Starting MLFlow UI"
echo

conda run --no-capture-output -n "$CONDA_ENV" mlflow ui --backend-store-uri "$PATH_CONTENT_ROOT"/data/mlruns --port "$PORT" --host 0.0.0.0
