# shellcheck shell=bash
# =============================================================================
# Adastra (CINES) — account variant derivation.
#
# Adastra uses a per-constraint account-variant convention: the base
# project name is suffixed with the lower-cased constraint name when
# submitting to that constraint.  Confirmed on 2026-05-14 via
# ``sacctmgr -nP list assoc where user=$USER``; both projects of this
# user (cad14975, c2117856) carry five association rows each:
#
#     <project>          (bare, no suffix)
#     <project>_genoa    (GENOA constraint)
#     <project>_mi250    (MI250 constraint)
#     <project>_mi300    (MI300 constraint)
#     <project>_hpda     (HPDA constraint, non-billed)
#
# The trap: SLURM's assoc table accepts the BARE account on any
# constraint, so an unsuffixed --account=<project> --constraint=MI250
# is *not* rejected at sbatch time — the job runs and the billing
# silently misses the right pool.  Always pass the suffixed variant.
#
# Helper contract
# ---------------
# adastra_account_for_constraint <base_or_variant> <CONSTRAINT>
#     Echoes the suffixed account name.  Idempotent: a value already
#     ending in any known suffix (_genoa/_mi250/_mi300/_hpda) is
#     returned unchanged, so callers can safely pass either a bare
#     project or a fully-qualified variant.
#
# Usage:
#     # shellcheck source=_lib/account.sh
#     source "$(dirname "$0")/_lib/account.sh"
#     account="$(adastra_account_for_constraint "$BASE" "$CONSTRAINT")"
# =============================================================================

adastra_account_for_constraint() {
    local base="$1" constraint="$2"
    local lc
    lc="$(printf '%s' "$constraint" | tr '[:upper:]' '[:lower:]')"
    # Idempotency: leave a value that already carries a known Adastra
    # constraint suffix unchanged.  Refusing to "fix" a wrong suffix is
    # deliberate — it would mask a user mistake instead of surfacing it.
    if [[ "$base" =~ _(genoa|mi250|mi300|hpda)$ ]]; then
        printf '%s\n' "$base"
    else
        printf '%s_%s\n' "$base" "$lc"
    fi
}
