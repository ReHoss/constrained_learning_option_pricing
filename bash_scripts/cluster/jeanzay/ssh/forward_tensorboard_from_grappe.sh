#!/bin/bash
#
# Same as forward_tensorboard.sh but to be launched from the Grappe
# bouncer (where `ssh jeanzay` is reachable but the local machine isn't
# directly).  Browse http://localhost:6007 on the box where you run this.

HOST_NAME=jeanzay

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
