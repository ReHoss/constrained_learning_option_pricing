# shellcheck shell=bash
# =============================================================================
# Adastra (CINES) — account normalisation.
#
# Empirical finding from a smoke test on 2026-05-14: Adastra's SLURM
# accepts ONLY the bare project name on ``--account=``.  When you pass
# ``--account=<project> --constraint=GENOA``, the scheduler auto-routes
# the job internally to the ``<project>_genoa`` accounting pool — that
# remapped name is what shows up in ``sacct -P -o JobID,Account``, NOT
# what you submit.  Three test jobs confirmed the mapping for GENOA,
# MI250 and HPDA.
#
# Passing a pre-suffixed variant (e.g. ``c2117856_hpda``) directly to
# sbatch / srun is **rejected**, with a misleading mask:
#
#     "You must specify an account from those listed by command:
#      myproject -l"
#
# (Same error happens for several unrelated causes — e.g. an expired
# project validity window — so don't take it literally.)
#
# So this helper does the OPPOSITE of what its previous incarnation did:
# it strips any known constraint suffix from a value so that callers
# can submit safely regardless of whether the user typed the bare form
# or accidentally pasted a suffixed variant from sacctmgr output.
#
# Helper contract
# ---------------
# adastra_account_bare <input>
#     Echoes the bare project name.
#     - Input ending in ``_genoa`` / ``_mi250`` / ``_mi300`` / ``_hpda``
#       has the suffix stripped.
#     - Any other input is echoed unchanged (no validation — we don't
#       know all project naming conventions, only the suffix list).
#     - Idempotent: bare → bare; suffixed → bare; bare-with-underscore
#       (unlikely but possible) → unchanged.
#
# Usage:
#     # shellcheck source=_lib/account.sh
#     source "$(dirname "$0")/_lib/account.sh"
#     account_for_sbatch="$(adastra_account_bare "$USER_INPUT")"
# =============================================================================

adastra_account_bare() {
    local input="$1"
    if [[ "$input" =~ ^(.+)_(genoa|mi250|mi300|hpda)$ ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
    else
        printf '%s\n' "$input"
    fi
}
