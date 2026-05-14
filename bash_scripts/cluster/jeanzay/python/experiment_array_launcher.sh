#!/bin/bash
# =============================================================================
# Jean Zay generic experiment-array launcher
# =============================================================================
#
# Orchestrates the standard three-phase experiment pipeline on Jean Zay:
#
#   1. INIT     (login node, no sbatch) — runs the target Python script with
#               the user-supplied ``--init-args`` to create a timestamped
#               experiment directory that contains
#                   <EXPDIR>/configs/*.yaml
#               one YAML per array task.  The script must print the absolute
#               path to <EXPDIR> on the *last* stdout line.
#
#   2. ARRAY    (sbatch --array) — reuses
#               ``slurm_job_array/job_array_batch_xp.slurm`` as a generic
#               per-task worker.  Each task runs:
#                   python <script> --config-dir <EXPDIR>/configs \
#                                   --config-name <basename of yaml>
#               The mapping ``SLURM_ARRAY_TASK_ID -> basename`` is computed
#               on the compute node by ``mapfile`` over the YAML files.
#
#   3. FINALIZE (sbatch, --dependency=afterok, --wrap) — only submitted when
#               ``--finalize-args`` is non-empty.  The placeholder ``{EXPDIR}``
#               in those args is substituted with the absolute experiment
#               directory captured in phase 1.
#
# This launcher is intentionally **agnostic** to the kind of experiment it
# runs: ablations, hyperparameter sweeps, multi-seed benchmarks, anything
# that respects the contract above.  The Python script owns the experiment
# semantics; the launcher owns the SLURM deployment.
#
# Contract the Python script must respect
# ----------------------------------------
# ``python <script> <init-args>``        (login node, called once)
#     - Must create <EXPDIR>/configs/*.yaml (one per task).
#     - Must print the absolute path to <EXPDIR> on the last stdout line.
#     - Must be torch-free (or otherwise cheap) so it can run on the login
#       node — Jean Zay forbids significant computation there.
#
# ``python <script> --config-dir <DIR> --config-name <BASENAME>``  (compute)
#     - Must consume the YAML at <DIR>/<BASENAME>.yaml.
#     - Must be idempotent over <EXPDIR> (multiple array tasks write to the
#       same parent directory under their own subfolder).
#
# ``python <script> <finalize-args with {EXPDIR} substituted>``    (compute)
#     - Optional aggregation / replot pass after every array task succeeded.
#
# Usage examples
# --------------
# Singularity-PINN ablation (the canonical caller):
#
#     bash bash_scripts/cluster/jeanzay/python/experiment_array_launcher.sh \
#         --python-script experiments/python_scripts/exp_singularity_european_call/ablation_singularity_logS.py \
#         --init-args "--mode hard-ic-ansatz-european-call --seed 0 --init-only --device cpu" \
#         --finalize-args "--replot {EXPDIR} --device cpu"
#
# Smoke test (the debug folder prefix is delegated to the script via its own flag):
#
#     bash ... --python-script .../ablation_singularity_logS.py \
#         --init-args "--mode hard-ic-ansatz-european-call --debug --seed 0 --init-only --device cpu" \
#         --finalize-args "--replot {EXPDIR} --device cpu"
#
# Variance estimate — repeat under fresh master seeds:
#
#     for s in 0 1 2; do
#         bash ... --init-args "--mode ... --seed $s --init-only --device cpu" --finalize-args "..."
#     done
# =============================================================================

set -euo pipefail

# ── Defaults — override via CLI flags ────────────────────────────────────────
NAME_PROJECT="constrained_learning_option_pricing"
V_ENV_NAME="venv_learning_option_pricing"
PYTHON_SCRIPT_REL=""    # required: path relative to the project content root
INIT_ARGS=""            # required: passed verbatim to the script for the init phase
FINALIZE_ARGS=""        # optional: '{EXPDIR}' is substituted post-init

# SLURM resource defaults
S_BATCH_TIME="04:00:00"
S_BATCH_TIME_FINALIZE="00:30:00"
S_BATCH_QOS="qos_gpu-t3"
S_BATCH_ACCOUNT="akz@v100"
S_BATCH_CPU_PER_TASK=10
S_BATCH_GPUS=1                       # one GPU per array task

# Finalize defaults — non-billed prepost partition (pure CPU replot pass).
# Override with --finalize-partition / --finalize-qos / --finalize-account /
# --finalize-gpus only when the aggregation step truly needs a GPU or a
# billed allocation; otherwise leave them as-is so the replot stays free.
S_BATCH_FINALIZE_PARTITION="prepost"
S_BATCH_FINALIZE_QOS=""
S_BATCH_FINALIZE_ACCOUNT=""
S_BATCH_FINALIZE_GPUS=0

# ── Argument parsing ─────────────────────────────────────────────────────────
while (( $# )); do
    case "$1" in
        --python-script)     PYTHON_SCRIPT_REL="$2";     shift 2 ;;
        --init-args)         INIT_ARGS="$2";             shift 2 ;;
        --finalize-args)     FINALIZE_ARGS="$2";         shift 2 ;;
        --account)           S_BATCH_ACCOUNT="$2";       shift 2 ;;
        --qos)               S_BATCH_QOS="$2";           shift 2 ;;
        --time)              S_BATCH_TIME="$2";          shift 2 ;;
        --time-finalize)     S_BATCH_TIME_FINALIZE="$2"; shift 2 ;;
        --cpus-per-task)     S_BATCH_CPU_PER_TASK="$2";  shift 2 ;;
        --gpus)              S_BATCH_GPUS="$2";          shift 2 ;;
        --finalize-partition) S_BATCH_FINALIZE_PARTITION="$2"; shift 2 ;;
        --finalize-qos)      S_BATCH_FINALIZE_QOS="$2";  shift 2 ;;
        --finalize-account)  S_BATCH_FINALIZE_ACCOUNT="$2"; shift 2 ;;
        --finalize-gpus)     S_BATCH_FINALIZE_GPUS="$2"; shift 2 ;;
        --venv-name)         V_ENV_NAME="$2";            shift 2 ;;
        --name-project)      NAME_PROJECT="$2";          shift 2 ;;
        -h|--help)
            sed -n '2,68p' "$0"; exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            sed -n '2,68p' "$0"; exit 1 ;;
    esac
done

if [[ -z "$PYTHON_SCRIPT_REL" ]]; then
    echo "Error: --python-script is required (path relative to project root)." >&2
    exit 1
fi
if [[ -z "$INIT_ARGS" ]]; then
    echo "Error: --init-args is required (args for the init phase)." >&2
    exit 1
fi

# ── Locate the project on Jean Zay's $WORK ──────────────────────────────────
WORKDIR="${WORK:?WORK env var is not set — are you on Jean Zay?}"
PATH_CONTENT_ROOT="$WORKDIR/pycharm_remote_project/$NAME_PROJECT"
PATH_PYTHON_SCRIPT="$PATH_CONTENT_ROOT/$PYTHON_SCRIPT_REL"
PATH_VENV_BIN="$PATH_CONTENT_ROOT/venv/$V_ENV_NAME/bin/activate"
PATH_PARENT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PATH_WORKER_SLURM="$PATH_PARENT/slurm_job_array/job_array_batch_xp.slurm"

for path in "$PATH_CONTENT_ROOT" "$PATH_PYTHON_SCRIPT" "$PATH_VENV_BIN" "$PATH_WORKER_SLURM"; do
    if [[ ! -e "$path" ]]; then
        echo "Error: missing path $path" >&2
        exit 1
    fi
done

echo "Project root:   $PATH_CONTENT_ROOT"
echo "Python script:  $PATH_PYTHON_SCRIPT"
echo "Venv:           $PATH_VENV_BIN"
echo "Worker:         $PATH_WORKER_SLURM"
echo "Init args:      $INIT_ARGS"
echo "Finalize args:  ${FINALIZE_ARGS:-<none>}"
echo

# ── Activate venv on the login node (for the init phase) ────────────────────
# shellcheck source=/dev/null
source "$PATH_VENV_BIN"
cd "$PATH_CONTENT_ROOT"

# ── Phase 1: init must produce <EXPDIR>/configs/*.yaml ──────────────────────
echo "Phase 1: init (login node)..."
# shellcheck disable=SC2086  # we *want* word splitting on INIT_ARGS
EXPDIR="$(python "$PATH_PYTHON_SCRIPT" $INIT_ARGS 2>/dev/null | tail -n1)"

if [[ ! -d "$EXPDIR" ]]; then
    echo "Error: init did not return a usable directory (got: $EXPDIR)" >&2
    echo "       The Python script must print the absolute EXPDIR on the last stdout line." >&2
    exit 1
fi
echo "  experiment dir: $EXPDIR"

PATH_FOLDER_CONFIGS="$EXPDIR/configs"
if [[ ! -d "$PATH_FOLDER_CONFIGS" ]]; then
    echo "Error: $PATH_FOLDER_CONFIGS does not exist." >&2
    echo "       The init phase must populate it with one YAML per array task." >&2
    exit 1
fi

N_CONFIGS=$(find "$PATH_FOLDER_CONFIGS" -name "*.yaml" | wc -l)
if (( N_CONFIGS == 0 )); then
    echo "Error: 0 YAML configs found under $PATH_FOLDER_CONFIGS." >&2
    exit 1
fi
N_LAST_ARRAYID=$((N_CONFIGS - 1))
echo "  configs:        $N_CONFIGS YAML files under $PATH_FOLDER_CONFIGS/"
echo

# Tag the SLURM job name with the experiment directory's basename — enough
# for ``squeue`` to disambiguate concurrent runs without re-deriving the
# specific experiment parameters.
EXPDIR_BASENAME="$(basename "$EXPDIR")"

# All slurm-* stdout/stderr files for this run land in a dedicated subfolder
# so the top of the ablation directory only carries canonical artefacts
# (configs/, metadata.yaml, per-variant subdirs, comparison/, ...).
SLURM_LOG_DIR="$EXPDIR/slurm"
mkdir -p "$SLURM_LOG_DIR"

# ── Phase 2: submit the array (reuses the existing generic worker) ──────────
echo "Phase 2: sbatch --array=0-$N_LAST_ARRAYID ($N_CONFIGS GPU tasks)..."
TRAIN_JOB_ID=$(sbatch --parsable \
    --job-name="xp_${EXPDIR_BASENAME}" \
    --array=0-${N_LAST_ARRAYID} \
    --output="$SLURM_LOG_DIR/slurm-TRAIN-%A_%a.out" \
    --error="$SLURM_LOG_DIR/slurm-TRAIN-%A_%a.err" \
    --export=NAME_PROJECT="$NAME_PROJECT",PATH_PYTHON_SCRIPT="$PATH_PYTHON_SCRIPT",PATH_FOLDER_CONFIGS="$PATH_FOLDER_CONFIGS",WORKDIR="$WORKDIR",V_ENV_NAME="$V_ENV_NAME" \
    --cpus-per-task="$S_BATCH_CPU_PER_TASK" \
    --time="$S_BATCH_TIME" \
    --qos="$S_BATCH_QOS" \
    --account="$S_BATCH_ACCOUNT" \
    --nodes=1 \
    --ntasks-per-node=1 \
    --gres=gpu:"$S_BATCH_GPUS" \
    --hint=nomultithread \
    "$PATH_WORKER_SLURM")
echo "  TRAIN job id:   $TRAIN_JOB_ID"

# ── Phase 3: submit the finalize job, only if --finalize-args was supplied ──
FINALIZE_JOB_ID=""
if [[ -n "$FINALIZE_ARGS" ]]; then
    # Substitute the {EXPDIR} placeholder.  Done in bash (not inside the
    # wrapped heredoc) so the substituted value is visible in ``squeue`` /
    # ``scontrol show job``.
    FINALIZE_ARGS_RESOLVED="${FINALIZE_ARGS//\{EXPDIR\}/$EXPDIR}"
    echo "Phase 3: sbatch --dependency=afterok:$TRAIN_JOB_ID (finalize)..."
    echo "  resolved args:  $FINALIZE_ARGS_RESOLVED"
    echo "  partition:      $S_BATCH_FINALIZE_PARTITION${S_BATCH_FINALIZE_QOS:+ / qos=$S_BATCH_FINALIZE_QOS}${S_BATCH_FINALIZE_ACCOUNT:+ / account=$S_BATCH_FINALIZE_ACCOUNT}${S_BATCH_FINALIZE_GPUS:+ / gpus=$S_BATCH_FINALIZE_GPUS}"
    # Optional sbatch flags — only emit when set so the default prepost
    # routing stays free of qos/account/gres lines that would otherwise
    # pin the job back to a billed allocation.
    FINALIZE_OPT_FLAGS=()
    [[ -n "$S_BATCH_FINALIZE_QOS"     ]] && FINALIZE_OPT_FLAGS+=(--qos="$S_BATCH_FINALIZE_QOS")
    [[ -n "$S_BATCH_FINALIZE_ACCOUNT" ]] && FINALIZE_OPT_FLAGS+=(--account="$S_BATCH_FINALIZE_ACCOUNT")
    (( S_BATCH_FINALIZE_GPUS > 0 ))    && FINALIZE_OPT_FLAGS+=(--gres="gpu:$S_BATCH_FINALIZE_GPUS")
    FINALIZE_JOB_ID=$(sbatch --parsable \
        --job-name="xp_${EXPDIR_BASENAME}_finalize" \
        --dependency="afterok:${TRAIN_JOB_ID}" \
        --output="$SLURM_LOG_DIR/slurm-FINALIZE.out" \
        --error="$SLURM_LOG_DIR/slurm-FINALIZE.err" \
        --partition="$S_BATCH_FINALIZE_PARTITION" \
        --cpus-per-task=4 \
        --time="$S_BATCH_TIME_FINALIZE" \
        "${FINALIZE_OPT_FLAGS[@]+"${FINALIZE_OPT_FLAGS[@]}"}" \
        --nodes=1 --ntasks-per-node=1 \
        --hint=nomultithread \
        --wrap "cd '$PATH_CONTENT_ROOT' && source '$PATH_VENV_BIN' && python '$PATH_PYTHON_SCRIPT' $FINALIZE_ARGS_RESOLVED")
    echo "  FINALIZE job id: $FINALIZE_JOB_ID"
else
    echo "Phase 3: skipped (no --finalize-args supplied)."
fi

# ── Summary ─────────────────────────────────────────────────────────────────
cat <<EOF_SUMMARY

──────────────────────────────────────────────────────────────────────
  Experiment submitted.

  Script    : $PATH_PYTHON_SCRIPT
  Init args : $INIT_ARGS
  Exp dir   : $EXPDIR
  Tasks     : $N_CONFIGS  (see $PATH_FOLDER_CONFIGS/*.yaml)

  Jobs:
    TRAIN array : $TRAIN_JOB_ID   (one GPU task per YAML)
EOF_SUMMARY
if [[ -n "$FINALIZE_JOB_ID" ]]; then
    echo "    FINALIZE    : $FINALIZE_JOB_ID  (afterok, CPU)"
fi
cat <<EOF_SUMMARY

  Watch progress:
    squeue -u "\$USER"
    tail -f $SLURM_LOG_DIR/slurm-TRAIN-${TRAIN_JOB_ID}_0.out
EOF_SUMMARY
if [[ -n "$FINALIZE_JOB_ID" ]]; then
    echo "    tail -f $SLURM_LOG_DIR/slurm-FINALIZE.out"
fi
cat <<EOF_SUMMARY
──────────────────────────────────────────────────────────────────────
EOF_SUMMARY
