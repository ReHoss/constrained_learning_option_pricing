#!/bin/bash
#
# Open an interactive shell on a Jean Zay compute node (GPU by default).
#
# Useful for:
#   - testing that the venv has the right PyTorch/CUDA before submitting a long batch job
#   - quick numerical sanity checks
#   - debugging an OOM / crash that you cannot reproduce on the login node
#
# Usage:
#   bash run_interactive_job.sh           # default: 1 V100, 30 min, qos_gpu-dev
#
# Once the prompt opens on the compute node:
#   source $WORK/git_repositories/constrained_learning_option_pricing/venv/venv_learning_option_pricing/bin/activate
#   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

NAME_PROJECT="constrained_learning_option_pricing"

WORKDIR=$WORK
PATH_CONTENT_ROOT="${PATH_CONTENT_ROOT:-$WORKDIR/git_repositories/$NAME_PROJECT}"

V_ENV_NAME="venv_learning_option_pricing"
PATH_VENV_BIN="$PATH_CONTENT_ROOT"/venv/"$V_ENV_NAME"/bin/activate
if [ -f "$PATH_VENV_BIN" ]; then
  echo "Venv detected at $PATH_VENV_BIN — activate it on the compute node with:"
  echo "  source $PATH_VENV_BIN"
  echo
else
  echo "Note: venv not found at $PATH_VENV_BIN — set it up before running scripts."
  echo
fi

# --- srun options ---
# Time: qos_gpu-dev has a 2h hard cap; keep short for testing.
S_RUN_TIME=00:30:00
# QoS: qos_gpu-dev → 2h, higher priority for testing.
S_RUN_QOS=qos_gpu-dev
# Account: depends on your allocation — check `idracct`.
S_RUN_ACCOUNT=akz@v100
S_RUN_NODES=1
S_RUN_CPU_PER_TASK=4
S_RUN_GPUS=1

echo "srun options:"
echo "  --nodes=$S_RUN_NODES"
echo "  --time=$S_RUN_TIME"
echo "  --qos=$S_RUN_QOS"
echo "  --account=$S_RUN_ACCOUNT"
echo "  --cpus-per-task=$S_RUN_CPU_PER_TASK"
echo "  --gres=gpu:$S_RUN_GPUS"
echo "  --pty bash"
echo

srun \
  --nodes="$S_RUN_NODES" \
  --time="$S_RUN_TIME" \
  --qos="$S_RUN_QOS" \
  --account="$S_RUN_ACCOUNT" \
  --cpus-per-task="$S_RUN_CPU_PER_TASK" \
  --gres=gpu:"$S_RUN_GPUS" \
  --pty \
  /bin/bash
