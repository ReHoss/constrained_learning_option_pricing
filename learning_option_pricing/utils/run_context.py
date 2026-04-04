from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def configure_cli_script_logging(*, verbose: bool) -> None:
    """Console logging for standalone scripts (no ``run.log``).

    Call once at process start, then use ``logging.getLogger(__name__)`` in the script.

    Args:
        verbose: If True, use DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)sZ %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        force=True,
    )


def log_parsed_args(
    logger: logging.Logger,
    namespace: argparse.Namespace,
    *,
    banner: str = "Parsed arguments",
) -> None:
    """Log sorted CLI arguments as JSON (reproducibility)."""
    payload = dict(sorted(vars(namespace).items()))
    logger.info("%s:\n%s", banner, json.dumps(payload, indent=2, default=str))


def log_runtime_versions(logger: logging.Logger) -> None:
    """Log Python, NumPy, and PyTorch versions."""
    import numpy as np
    import torch

    logger.info(
        "Runtime: Python %s | numpy %s | torch %s",
        sys.version.split()[0],
        np.__version__,
        torch.__version__,
    )


def utc_timestamp() -> str:
    """Return an ISO-like timestamp usable in folder names."""
    # Example: 2026-03-20-14-33-10-123456Z
    #
    # Note: we intentionally avoid ':' for cross-platform folder-name safety.
    return datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S-%fZ")


def _try_run_git(args: list[str], cwd: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def get_git_metadata(repo_root: Path) -> dict[str, Any]:
    """
    Best-effort Git metadata for reproducibility.

    Falls back to empty dict when not available (e.g., no .git folder).
    """
    if not (repo_root / ".git").exists():
        return {}

    commit = _try_run_git(["rev-parse", "HEAD"], cwd=repo_root)
    branch = _try_run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    status = _try_run_git(["status", "--porcelain"], cwd=repo_root) or ""
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status.strip()),
        "status_porcelain": status.splitlines()[:50],  # keep bounded
    }


def create_run_dir(
    *,
    output_root: Path,
    script_name: str,
    timestamp: str | None = None,
) -> Path:
    """
    Create `output_root/<script_name>/<timestamp>/`.

    `exist_ok` is intentionally False to prevent accidental overwrites.
    """
    ts = timestamp or utc_timestamp()
    run_dir = output_root / script_name / ts
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def init_logging(*, run_dir: Path, level: int = logging.INFO) -> logging.Logger:
    """
    Initialize logging into both console and `<run_dir>/run.log`.

    We attach handlers only once to avoid duplicated log lines when imported.
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    # Avoid adding duplicate handlers.
    if any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)sZ %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    file_handler = logging.FileHandler(run_dir / "run.log", mode="w", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _subset_env(keys: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k in keys:
        if k in os.environ:
            out[k] = os.environ[k]
    return out


def collect_run_metadata(
    *,
    run_dir: Path,
    repo_root: Path,
    script_name: str,
    command: list[str],
    params: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ts = run_dir.name
    meta: dict[str, Any] = {
        "timestamp": ts,
        "script_name": script_name,
        "run_dir": str(run_dir),
        "command": command,
        "cwd": str(Path.cwd()),
        "python": sys.version,
        "platform": platform.platform(),
        "env_subset": _subset_env(
            [
                "SLURM_JOB_ID",
                "SLURM_ARRAY_TASK_ID",
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "PYTHONHASHSEED",
            ]
        ),
        "params": params or {},
        "extra": extra or {},
        "git": get_git_metadata(repo_root),
    }
    return meta


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_command_txt(path: Path, command: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use a shell-like representation for readability.
    rendered = " ".join([str(x) for x in command])
    path.write_text(rendered + "\n", encoding="utf-8")


@dataclass(frozen=True)
class RunContext:
    run_dir: Path
    script_name: str

    def metadata_path(self) -> Path:
        return self.run_dir / "run_metadata.json"

    def command_path(self) -> Path:
        return self.run_dir / "command.txt"

    def params_path(self) -> Path:
        return self.run_dir / "params.json"

