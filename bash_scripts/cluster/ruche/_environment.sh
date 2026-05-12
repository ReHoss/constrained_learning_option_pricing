#!/bin/bash
# =============================================================================
# Ruche cluster environment bootstrap.
#
# Source this file from any Ruche launcher *before* activating the project
# venv or invoking conda.  It performs the cluster-specific module-system
# setup required to make `conda` / `python` resolvable on the login node and
# on compute nodes:
#
#   - Reset the module environment to a clean state (`module purge`) so the
#     login shell's accumulated modules do not leak into the job.
#   - Load the Spack-managed miniconda3 build that Ruche ships with.  The
#     exact module string is brittle (cluster admins occasionally rotate it)
#     but is the version that works as of the SciML/control_dde era and that
#     no other launcher in this repo records.
#   - Expose the absolute path to the matching `conda` binary so callers can
#     run `conda` reliably regardless of whether the `module load` step also
#     prepended it to PATH.  Useful when the launcher needs to call conda
#     non-interactively (e.g. `$PATH_CONDA_BIN run --no-capture-output ...`).
#
# Usage:
#     source "$(dirname "${BASH_SOURCE[0]}")/../_environment.sh"
#
# This file is intentionally separate from the launchers so the cluster-
# specific recipe survives any future launcher refactor.
# =============================================================================

module purge
module load miniconda3/23.5.2/gcc-13.2.0

# Absolute path to the conda binary corresponding to the module loaded above.
# Resolved via Spack on Ruche; pinned here so launchers do not have to grep
# `which conda` (which races with the module system on cold shells).
export PATH_CONDA_BIN=/gpfs/softs/spack_0.17/opt/spack/linux-centos7-cascadelake/gcc-13.2.0/miniconda3-23.5.2-scvtvts2zr4k27oespcarh43r6zcswmf/bin/conda
