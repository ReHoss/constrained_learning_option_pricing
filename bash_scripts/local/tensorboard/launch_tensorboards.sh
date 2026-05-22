#!/bin/bash
#
# Launch TensorBoard on a logs directory written by an experiment script.
#
# Usage:
#   bash launch_tensorboards.sh -i <experiment_id_or_subfolder>
#
# Default backend store: <repo>/data/tensorboard_logs/<experiment_id>
# Override the base directory by exporting PATH_BACKEND_STORE before calling.

# Path of the parent directory of this script
PATH_PARENT=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit ; pwd -P )
# Path of the root of the project
PATH_CONTENT_ROOT=$(realpath "$PATH_PARENT"/../../..)

V_ENV_NAME="venv_learning_option_pricing"
PATH_VENV="$PATH_CONTENT_ROOT"/venv/"$V_ENV_NAME"/bin/activate

# Path of the TensorBoard logs root (neutral default — override if needed)
PATH_BACKEND_STORE="${PATH_BACKEND_STORE:-$PATH_CONTENT_ROOT/data/tensorboard_logs}"

echo "Loading virtual environment"
echo
# shellcheck disable=SC1090
source "$PATH_VENV"

echo "PATH_VENV:          $PATH_VENV"
echo "PATH_CONTENT_ROOT:  $PATH_CONTENT_ROOT"
echo "PATH_BACKEND_STORE: $PATH_BACKEND_STORE"
echo

while getopts 'i:' flag; do
  case "${flag}" in
    i) EXPERIMENT_ID="${OPTARG}" ;;
    *) echo "Unexpected option ${flag}" && exit 1 ;;
  esac
done

if [ -z "${EXPERIMENT_ID:-}" ]; then
  tensorboard --logdir "$PATH_BACKEND_STORE"
else
  tensorboard --logdir "$PATH_BACKEND_STORE"/"$EXPERIMENT_ID"
fi
