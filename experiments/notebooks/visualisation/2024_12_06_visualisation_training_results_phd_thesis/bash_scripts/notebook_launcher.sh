#!/bin/bash

PATH_PARENT=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit ; pwd -P )
PATH_CONTENT_ROOT="$PATH_PARENT/../../../../.."
#CONDA_ENV=$(cat "$PATH_CONTENT_ROOT"/bash_scripts/conda_env_name.txt)
#export PYTHONPATH="${PYTHONPATH}:$PATH_CONTENT_ROOT"

# Add command line arguments to provide the name of the notebook with the extension
while getopts 'n:' flag; do
  case "${flag}" in
  n) NAME_NOTEBOOK="${OPTARG}" ;;
  *) error "Unexpected option ${flag}" ;;
  esac
done

# Check if the name of the notebook is provided
if [ -z "$NAME_NOTEBOOK" ]; then
  echo "Please provide the name of the notebook with the extension using the -n flag"
  exit 1
fi

V_ENV_NAME="venv_control_dde"
PATH_VENV="$PATH_CONTENT_ROOT"/venv/"$V_ENV_NAME"/bin/activate

# Load virtual environment
source "$PATH_VENV"


NAME_OUTPUT_DIR=$( basename "$( cd "$PATH_PARENT/.." || exit; pwd -P )")

PATH_NOTEBOOK="$PATH_PARENT"/../"$NAME_NOTEBOOK"

PATH_OUTPUT_DIR="$PATH_CONTENT_ROOT"/data/notebooks/"$NAME_OUTPUT_DIR"
ARRAY_CONFIG_FILES=("${PATH_PARENT}"/../configs/*.yaml)

echo "PATH_NOTEBOOK: $PATH_NOTEBOOK"
echo "PATH_OUTPUT_DIR: $PATH_OUTPUT_DIR"

for path_yaml_file in "${ARRAY_CONFIG_FILES[@]}"; do
    echo Following config is procesessed: "$path_yaml_file"
    export PATH_YAML_CONFIG="$path_yaml_file"
    BASENAME_YAML_FILE=$(basename "$path_yaml_file" .yaml)
    BASENAME_NOTEBOOK=$(basename "$PATH_NOTEBOOK" .ipynb)
    # Name of the output file notebook_name + config_name + timestamp
    output_file_name="$BASENAME_NOTEBOOK"_"$BASENAME_YAML_FILE"_"$(date +"%Y_%m_%d_%H_%M_%S")"
    echo "BASENAME_YAML_FILE: $BASENAME_YAML_FILE"
    jupyter nbconvert --execute --to html "$PATH_NOTEBOOK" --output-dir="$PATH_OUTPUT_DIR" --output "$output_file_name".html
done
