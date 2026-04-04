PATH_PARENT=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit ; pwd -P )
PATH_CONTENT_ROOT=$(realpath "$PATH_PARENT/../../..")

NAME_SINGULARITY_DEFINITION_FILE="control_dde-firedrakeDEBUG.def"
PATH_SINGULARITY_DEFINITION_FILE_DIR="$PATH_CONTENT_ROOT"/singularity/definition_files
PATH_SINGULARITY_DEFINITION_FILE="$PATH_SINGULARITY_DEFINITION_FILE_DIR"/"$NAME_SINGULARITY_DEFINITION_FILE"

# User and group IDs
USER_ID=$(id -u)
GROUP_ID=$(id -g)

NAME_IMAGE_SIF_FILE="control_dde-firedrake_uid-${USER_ID}_gid-${GROUP_ID}_hostname-$(hostname).sif"
PATH_SIF_FILE_DIR="$PATH_CONTENT_ROOT"/singularity/images
PATH_SIF_FILE="$PATH_SIF_FILE_DIR"/"$NAME_IMAGE_SIF_FILE"

# Create a temporary directory for the definition file and copy it there
#TEMP_DIR=$(mktemp -d)
#cp "$PATH_SINGULARITY_DEFINITION_FILE" "$TEMP_DIR"

# Repla

# Build the Singularity image
singularity build \
--sandbox \
--no-cleanup \
--fakeroot \
--build-arg USER_ID="$USER_ID" \
--build-arg GROUP_ID="$GROUP_ID" \
"$(readlink -f "$PATH_SIF_FILE")" \
"$(readlink -f "$PATH_SINGULARITY_DEFINITION_FILE")"  || rm -rf "$TEMP_DIR"

# Remove the temporary directory
rm -rf "$TEMP_DIR"
