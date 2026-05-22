#!/bin/bash
#
# Open an SSH tunnel from the local machine to a TensorBoard instance
# running on a Ruche login node.  Browse http://localhost:6007 once
# this is up.

HOST_NAME=ruche

PORT_TENSORBOARD_HOST=6006
PORT_TENSORBOARD_LOCAL=6007
REMOTE_HOST_TENSORBOARD="localhost"

echo "Forwarding ports:"
echo
echo "PORT_TENSORBOARD_HOST:   $PORT_TENSORBOARD_HOST"
echo "PORT_TENSORBOARD_LOCAL:  $PORT_TENSORBOARD_LOCAL"
echo "REMOTE_HOST_TENSORBOARD: $REMOTE_HOST_TENSORBOARD"
echo

ssh -N \
 -L "$PORT_TENSORBOARD_LOCAL":$REMOTE_HOST_TENSORBOARD:"$PORT_TENSORBOARD_HOST" \
$HOST_NAME
