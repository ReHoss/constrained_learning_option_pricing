#!/bin/bash
# =============================================================================
# Adastra (CINES) generic experiment-array launcher
# =============================================================================
#
# Adastra port of the Jean Zay ``experiment_array_launcher.sh``.  Same
# three-phase contract (INIT on the login node, ARRAY on compute, FINALIZE
# afterok), but adapted to Adastra's SLURM conventions:
#
#   * hardware is selected via ``--constraint=`` (not ``--partition=``):
#       MI250  — AMD MI250X quadri-GPU nodes (4 cards = 8 GCDs/node, billed).
#       MI300  — AMD MI300A APU quadri-node (billed).
#       GENOA  — AMD EPYC 9654 scalar CPU (192 cores/node, billed).
#       HPDA   — NVIDIA L40 preprocessing/postprocessing nodes (NOT billed).
#   * ``--qos=`` is NEVER passed — Adastra's scheduler picks the QoS
#     automatically from the requested wall-time + node count.  The docs
#     explicitly warn against ``--qos=`` (see CLAUDE.md / Adastra section).
#   * ``--account=<project>`` is mandatory.  Run ``myproject -l`` once on
#     the login node to discover available projects; ``myproject <name>``
#     also switches ``$WORKDIR`` / ``$SCRATCHDIR`` / ``$STOREDIR`` /
#     ``$HOMEDIR`` to that project's storage tree.
#   * GPU specification uses ``--gpus-per-node=N`` (the official example
#     spelling), not ``--gres=gpu:N``.
#   * Job arrays are temporarily capped at 11 tasks (CINES bug workaround,
#     see docs entry dated 2025/12/02).  The launcher fails early if the
#     init phase produced more YAMLs than that.
#
# Routing decisions for this launcher (one comment line each, per the
# Adastra documentation requirement):
#   * TRAIN array  : ``--constraint=MI250`` exclusive — full-node MI250X
#                    allocation, 1 GPU asked per task is left to SLURM's
#                    layout when the array is wider than a node would
#                    hold.  Adjust via ``--constraint`` / ``--gpus-per-node``.
#   * FINALIZE step: ``--constraint=HPDA`` shared — non-billed L40 node
#                    for replot / aggregation; no ``--exclusive`` so the
#                    node is shared with other HPDA jobs.
#
# Contract the Python script must respect — identical to Jean Zay:
#
# ``python <script> <init-args>``        (login node, called once)
#     - Must create <EXPDIR>/configs/*.yaml (one per task).
#     - Must print the absolute path to <EXPDIR> on the last stdout line.
#     - Must be torch-free (or otherwise cheap) so it can run on the
#       Adastra login node — same rule as Jean Zay.
#
# ``python <script> --config-dir <DIR> --config-name <BASENAME>``  (compute)
#     - Must consume the YAML at <DIR>/<BASENAME>.yaml.
#     - Must be idempotent over <EXPDIR>.
#
# ``python <script> <finalize-args with {EXPDIR} substituted>``    (compute)
#     - Optional aggregation / replot pass after every array task succeeded.
#
# Usage example (canonical caller):
#
#     bash bash_scripts/cluster/adastra/python/experiment_array_launcher.sh \
#         --account bae1234 \
#         --python-script experiments/python_scripts/exp_singularity_european_call/ablation_singularity_logS.py \
#         --init-args "--mode hard-ic-ansatz-european-call --seed 0 --init-only --device cpu" \
#         --finalize-args "--replot {EXPDIR} --device cpu"
#
# Smoke test (delegate the ``_debug_`` prefix to the script):
#
#     bash ... --account bae1234 \
#         --python-script .../ablation_singularity_logS.py \
#         --init-args "--mode hard-ic-ansatz-european-call --debug --seed 0 --init-only --device cpu" \
#         --finalize-args "--replot {EXPDIR} --device cpu"
# =============================================================================

set -euo pipefail

# ── Defaults — override via CLI flags ────────────────────────────────────────
NAME_PROJECT="constrained_learning_option_pricing"
V_ENV_NAME="venv_learning_option_pricing"
PYTHON_SCRIPT_REL=""    # required: path relative to the project content root
INIT_ARGS=""            # required: passed verbatim to the script for the init phase
FINALIZE_ARGS=""        # optional: '{EXPDIR}' is substituted post-init

# SLURM resource defaults — TRAIN phase (MI250 exclusive).
S_BATCH_TIME="04:00:00"
S_BATCH_TIME_FINALIZE="00:30:00"
S_BATCH_ACCOUNT=""                     # mandatory on Adastra (no sensible default)
S_BATCH_CONSTRAINT="MI250"             # see CLAUDE.md / Adastra cheat-sheet
S_BATCH_CPU_PER_TASK=8                 # MI250X has 64 cores / 8 GCDs = 8 cores per GCD
S_BATCH_GPUS_PER_NODE=1                # one GPU per array task
S_BATCH_EXCLUSIVE=1                    # 1 → pass --exclusive; 0 → shared mode

# SLURM resource defaults — FINALIZE phase (HPDA, non-billed).
S_BATCH_FINALIZE_CONSTRAINT="HPDA"
S_BATCH_FINALIZE_ACCOUNT=""            # may be required even on HPDA when the
                                       # user has multiple eDARI projects;
                                       # defaults to S_BATCH_ACCOUNT if unset.
S_BATCH_FINALIZE_GPUS=0                # HPDA is for CPU replot; bump to 1 for GPU.

# Project location on Adastra's $WORKDIR.  See ``myproject -l`` for the
# active project; ``$WORKDIR`` is auto-set to /lus/work/<group>/<user>.
PATH_PROJECT_PARENT_DIR_REL="pycharm_remote_project"

# ── Argument parsing ─────────────────────────────────────────────────────────
while (( $# )); do
    case "$1" in
        --python-script)        PYTHON_SCRIPT_REL="$2";          shift 2 ;;
        --init-args)            INIT_ARGS="$2";                  shift 2 ;;
        --finalize-args)        FINALIZE_ARGS="$2";              shift 2 ;;
        --account)              S_BATCH_ACCOUNT="$2";            shift 2 ;;
        --constraint)           S_BATCH_CONSTRAINT="$2";         shift 2 ;;
        --time)                 S_BATCH_TIME="$2";               shift 2 ;;
        --time-finalize)        S_BATCH_TIME_FINALIZE="$2";      shift 2 ;;
        --cpus-per-task)        S_BATCH_CPU_PER_TASK="$2";       shift 2 ;;
        --gpus-per-node)        S_BATCH_GPUS_PER_NODE="$2";      shift 2 ;;
        --shared)               S_BATCH_EXCLUSIVE=0;             shift 1 ;;
        --finalize-constraint)  S_BATCH_FINALIZE_CONSTRAINT="$2"; shift 2 ;;
        --finalize-account)     S_BATCH_FINALIZE_ACCOUNT="$2";   shift 2 ;;
        --finalize-gpus)        S_BATCH_FINALIZE_GPUS="$2";      shift 2 ;;
        --venv-name)            V_ENV_NAME="$2";                 shift 2 ;;
        --name-project)         NAME_PROJECT="$2";               shift 2 ;;
        --project-parent-rel)   PATH_PROJECT_PARENT_DIR_REL="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,80p' "$0"; exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            sed -n '2,80p' "$0"; exit 1 ;;
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
if [[ -z "$S_BATCH_ACCOUNT" ]]; then
    echo "Error: --account is required on Adastra (run 'myproject -l' to list)." >&2
    exit 1
fi

# Default finalize account to the main one if not overridden — keeps HPDA
# submissions valid for users with multiple eDARI projects attached.
if [[ -z "$S_BATCH_FINALIZE_ACCOUNT" ]]; then
    S_BATCH_FINALIZE_ACCOUNT="$S_BATCH_ACCOUNT"
fi

# ── Locate the project on Adastra's $WORKDIR ────────────────────────────────
# $WORKDIR is provided by the CINES login environment (set by the active
# myproject); /lus/work/<group>/<user>.  We don't fall back to anything else
# because mis-locating the project would silently submit stale code.
WORKDIR="${WORKDIR:?WORKDIR env var is not set — are you on Adastra and is a project active (myproject -l)?}"
PATH_CONTENT_ROOT="$WORKDIR/$PATH_PROJECT_PARENT_DIR_REL/$NAME_PROJECT"
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
echo "Account:        $S_BATCH_ACCOUNT"
echo "TRAIN constraint:    $S_BATCH_CONSTRAINT  (exclusive=$S_BATCH_EXCLUSIVE)"
echo "FINALIZE constraint: $S_BATCH_FINALIZE_CONSTRAINT  (account=$S_BATCH_FINALIZE_ACCOUNT)"
echo "Init args:      $INIT_ARGS"
echo "Finalize args:  ${FINALIZE_ARGS:-<none>}"
echo

# ── Activate venv on the login node (for the init phase) ────────────────────
# Same rule as Jean Zay: the init phase must be torch-free / cheap; we only
# need the project's Python on PATH to run the script's argparse + config
# generation.
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
# CINES temporary limit on array size — see CLAUDE.md / Adastra section.
# Re-check the docs (https://dci.dci-gitlab.cines.fr/webextranet/user_support/index.html#job-submission)
# before raising; the error otherwise is "AssocMaxSubmitJobLimit".
ADASTRA_MAX_ARRAY_SIZE=11
if (( N_CONFIGS > ADASTRA_MAX_ARRAY_SIZE )); then
    echo "Error: init produced $N_CONFIGS YAML configs but Adastra currently caps job arrays at $ADASTRA_MAX_ARRAY_SIZE tasks." >&2
    echo "       Split the ablation into chunks (chained --dependency=afterok) or use the packing pattern." >&2
    echo "       Re-check the Adastra docs before raising ADASTRA_MAX_ARRAY_SIZE in this launcher." >&2
    exit 1
fi
N_LAST_ARRAYID=$((N_CONFIGS - 1))
echo "  configs:        $N_CONFIGS YAML files under $PATH_FOLDER_CONFIGS/"
echo

EXPDIR_BASENAME="$(basename "$EXPDIR")"

# Dedicated slurm/ subfolder — same global rule as Jean Zay (see CLAUDE.md).
SLURM_LOG_DIR="$EXPDIR/slurm"
mkdir -p "$SLURM_LOG_DIR"

# ── Phase 2: submit the array ───────────────────────────────────────────────
echo "Phase 2: sbatch --array=0-$N_LAST_ARRAYID ($N_CONFIGS GPU tasks)..."
TRAIN_OPT_FLAGS=()
if (( S_BATCH_EXCLUSIVE == 1 )); then
    TRAIN_OPT_FLAGS+=(--exclusive)
fi
TRAIN_JOB_ID=$(sbatch --parsable \
    --job-name="xp_${EXPDIR_BASENAME}" \
    --array=0-${N_LAST_ARRAYID} \
    --output="$SLURM_LOG_DIR/slurm-TRAIN-%A_%a.out" \
    --error="$SLURM_LOG_DIR/slurm-TRAIN-%A_%a.err" \
    --export=NAME_PROJECT="$NAME_PROJECT",PATH_PYTHON_SCRIPT="$PATH_PYTHON_SCRIPT",PATH_FOLDER_CONFIGS="$PATH_FOLDER_CONFIGS",WORKDIR="$WORKDIR",V_ENV_NAME="$V_ENV_NAME",PATH_PROJECT_PARENT_DIR_REL="$PATH_PROJECT_PARENT_DIR_REL" \
    --constraint="$S_BATCH_CONSTRAINT" \
    --account="$S_BATCH_ACCOUNT" \
    --time="$S_BATCH_TIME" \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task="$S_BATCH_CPU_PER_TASK" \
    --gpus-per-node="$S_BATCH_GPUS_PER_NODE" \
    --threads-per-core=1 \
    "${TRAIN_OPT_FLAGS[@]+"${TRAIN_OPT_FLAGS[@]}"}" \
    "$PATH_WORKER_SLURM")
echo "  TRAIN job id:   $TRAIN_JOB_ID"

# ── Phase 3: submit the finalize job, only if --finalize-args was supplied ──
FINALIZE_JOB_ID=""
if [[ -n "$FINALIZE_ARGS" ]]; then
    FINALIZE_ARGS_RESOLVED="${FINALIZE_ARGS//\{EXPDIR\}/$EXPDIR}"
    echo "Phase 3: sbatch --dependency=afterok:$TRAIN_JOB_ID (finalize on $S_BATCH_FINALIZE_CONSTRAINT)..."
    echo "  resolved args:  $FINALIZE_ARGS_RESOLVED"
    FINALIZE_OPT_FLAGS=()
    if (( S_BATCH_FINALIZE_GPUS > 0 )); then
        FINALIZE_OPT_FLAGS+=(--gpus-per-node="$S_BATCH_FINALIZE_GPUS")
    fi
    FINALIZE_JOB_ID=$(sbatch --parsable \
        --job-name="xp_${EXPDIR_BASENAME}_finalize" \
        --dependency="afterok:${TRAIN_JOB_ID}" \
        --output="$SLURM_LOG_DIR/slurm-FINALIZE.out" \
        --error="$SLURM_LOG_DIR/slurm-FINALIZE.err" \
        --constraint="$S_BATCH_FINALIZE_CONSTRAINT" \
        --account="$S_BATCH_FINALIZE_ACCOUNT" \
        --time="$S_BATCH_TIME_FINALIZE" \
        --nodes=1 --ntasks-per-node=1 \
        --cpus-per-task=8 \
        --threads-per-core=1 \
        "${FINALIZE_OPT_FLAGS[@]+"${FINALIZE_OPT_FLAGS[@]}"}" \
        --wrap "cd '$PATH_CONTENT_ROOT' && source '$PATH_VENV_BIN' && python '$PATH_PYTHON_SCRIPT' $FINALIZE_ARGS_RESOLVED")
    echo "  FINALIZE job id: $FINALIZE_JOB_ID"
else
    echo "Phase 3: skipped (no --finalize-args supplied)."
fi

# ── Summary ─────────────────────────────────────────────────────────────────
cat <<EOF_SUMMARY

──────────────────────────────────────────────────────────────────────
  Experiment submitted (Adastra / CINES).

  Script    : $PATH_PYTHON_SCRIPT
  Init args : $INIT_ARGS
  Exp dir   : $EXPDIR
  Tasks     : $N_CONFIGS  (see $PATH_FOLDER_CONFIGS/*.yaml)

  Jobs:
    TRAIN array : $TRAIN_JOB_ID   (constraint=$S_BATCH_CONSTRAINT, one GPU task per YAML)
EOF_SUMMARY
if [[ -n "$FINALIZE_JOB_ID" ]]; then
    echo "    FINALIZE    : $FINALIZE_JOB_ID  (afterok, constraint=$S_BATCH_FINALIZE_CONSTRAINT, non-billed)"
fi
cat <<EOF_SUMMARY

  Watch progress:
    squeue --me
    tail -f $SLURM_LOG_DIR/slurm-TRAIN-${TRAIN_JOB_ID}_0.out
EOF_SUMMARY
if [[ -n "$FINALIZE_JOB_ID" ]]; then
    echo "    tail -f $SLURM_LOG_DIR/slurm-FINALIZE.out"
fi
cat <<EOF_SUMMARY
──────────────────────────────────────────────────────────────────────
EOF_SUMMARY
