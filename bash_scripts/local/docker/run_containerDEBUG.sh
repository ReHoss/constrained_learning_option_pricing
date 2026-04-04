#!/bin/bash

PATH_PARENT=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit ; pwd -P )
PATH_CONTENT_ROOT=$(realpath "$PATH_PARENT/../../..")

NAME_MOUNT_DIR="mount_dir"
PATH_CONTAINER_CONTENT_ROOT="/home/firedrake/$NAME_MOUNT_DIR/project_root"

NAME_CONTAINER="control_dde-firedrake"

# Run a shell in the container
docker run \
  -it \
  --rm \
  --mount type=bind,source="$PATH_CONTENT_ROOT"/data,target="$PATH_CONTAINER_CONTENT_ROOT"/data \
  --mount type=bind,source="$PATH_CONTENT_ROOT"/config,target="$PATH_CONTAINER_CONTENT_ROOT"/config \
  --mount type=bind,source="$PATH_CONTENT_ROOT"/control_dde,target="$PATH_CONTAINER_CONTENT_ROOT"/control_dde \
  --mount type=bind,source="$PATH_CONTENT_ROOT"/test,target="$PATH_CONTAINER_CONTENT_ROOT"/test \
  --mount type=bind,source="$PATH_CONTENT_ROOT"/torchbnn,target="$PATH_CONTAINER_CONTENT_ROOT"/torchbnn \
  --mount type=bind,source="$PATH_CONTENT_ROOT"/examples,target="$PATH_CONTAINER_CONTENT_ROOT"/examples \
  --mount type=bind,source=/home/hosseinkhan/Documents/work/phd/singularity-hydrogym/singularity_hydrogym,target="$PATH_CONTAINER_CONTENT_ROOT"/singularity_hydrogym \
  "$NAME_CONTAINER"


# Above it what is necessary to run tests!


# --read-only: Mount the container's root filesystem as read-only to check compatibility with Singularity
# Indeed, Singularity containers are read-only by default
# Unfortunately, firedrake writes at least in /home/firedrake/.cache/pytools in the container,
# so we cannot use --read-only. Consequently, the decision is to make the Singularity container writable
