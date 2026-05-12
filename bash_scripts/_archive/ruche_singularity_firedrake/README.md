# Archived: Ruche + Singularity + Firedrake launchers

These two bash scripts are **not active** on
`constrained_learning_option_pricing`. They are kept here as a reference for
container-based runs on the Ruche cluster (Université Paris-Saclay /
CentraleSupélec) — a pattern that did exist in earlier PhD projects
(`control_dde`) and that we may need again if a future experiment relies on
Firedrake or another finite-element stack.

## What they encode

The mechanics that are non-trivial to rediscover, and that are not preserved
anywhere else in the repository:

- **Singularity image routing.**  The host bind-mounts the project's content
  root into the container at `/home/firedrake/mount_dir/project_root`
  (variable `PATH_CONTAINER_CONTENT_ROOT`).  The Python script path passed to
  SLURM is rewritten relative to *that* container path, not the host path,
  so the in-container Python interpreter sees a consistent layout.
- **Cache directory relocation.**  Firedrake writes large amounts of JIT
  output to `~/.cache`; on Ruche the host home directory lives on a quota'd
  filesystem and Firedrake quickly trips
  `OSError: [Errno 28] No space left on device`.  The launcher therefore
  forces `PATH_CACHE_DIR_HPC` to a roomier scratch path and exports it into
  the container.
- **SIF image naming convention.**  Filename pattern
  `<project>-firedrake_nousernamespace_uid-<UID>_gid-<GID>_hostname-<host>.sif`
  — encodes the build host so the image is not mistakenly reused on an
  incompatible login node.
- **Ruche SLURM partitions / QoS.**  Examples of `cpu_short` / `cpu_med` /
  `cpu_long` partition choices, `S_BATCH_MEM_PER_NODE`, and the comment
  noting that `--qos` / `--account` are *not* needed on Ruche (unlike Jean
  Zay).  Useful as a reference when writing any Ruche launcher.

## Why archived rather than deleted

The bind-mount + cache-relocation + SIF-name recipe is documented nowhere
else in this repository and took non-trivial effort to figure out on the
original project.  Deleting it outright would force a future user to
re-derive it from scratch the next time a Singularity workflow is needed.

## Not used by the active project

`constrained_learning_option_pricing` runs natively in a Python venv on Jean
Zay; no container, no Firedrake.  The active Ruche launchers
(`bash_scripts/cluster/ruche/python/python_script_launcher*.sh`) intentionally
do *not* use Singularity.  If you are wiring up a new experiment on this
project, look there first — these files are reference-only.
