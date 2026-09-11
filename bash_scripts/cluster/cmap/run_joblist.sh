#!/bin/bash
# Run every non-comment line of a joblist as one pilot invocation, at most
# MAX_PARALLEL at a time, each pinned to THREADS cores and logged to its own
# file. For the CMAP machines (no SLURM): launch detached with
#   nohup bash bash_scripts/cluster/cmap/run_joblist.sh <joblist> < /dev/null > launch.log 2>&1 & disown
# The pilot derives one output directory per run from its own configuration
# tag (mode, seed, corner exclusion), so runs never collide. Each job is
# materialised as its own one-line script so that xargs only ever handles a
# path (BSD xargs caps the -I replacement at 255 bytes; a pilot command line
# is longer than that).
set -euo pipefail
JOBLIST="${1:?usage: run_joblist.sh <joblist> [max_parallel] [threads]}"
MAX_PARALLEL="${2:-6}"
THREADS="${3:-6}"
cd "$(dirname "$0")/../../.."
source venv/venv_learning_option_pricing/bin/activate
PILOT=experiments/python_scripts/exp_barrier_option/pilot_down_and_out_put.py
LOGDIR="data/pilot_down_and_out_put/launch_logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR/jobs"
echo "joblist=$JOBLIST  max_parallel=$MAX_PARALLEL  threads=$THREADS  logs=$LOGDIR"
# The thread count is passed to the pilot explicitly (--num-threads, which
# calls torch.set_num_threads) in addition to the environment variables: in
# float32 two runs that differ only in their thread count are not comparable
# (see the pilot's --num-threads help), so every job of a comparison must be
# pinned to the same value, and that value must appear in the logged command.
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
n=0
grep -vE '^\s*(#|$)' "$JOBLIST" | while read -r args; do
    n=$((n + 1)); tag=$(printf 'job%02d' "$n")
    printf 'python3 %s %s --num-threads %s > %s/%s.log 2>&1\n' "$PILOT" "$args" "$THREADS" "$LOGDIR" "$tag" > "$LOGDIR/jobs/$tag.sh"
done
ls "$LOGDIR"/jobs/*.sh | xargs -P "$MAX_PARALLEL" -n 1 bash
echo "ALL JOBS FINISHED $(date)"
