"""Tests for the Git metadata recorded by learning_option_pricing.utils.run_context."""
from __future__ import annotations

import shutil
import subprocess

import pytest

from learning_option_pricing.utils.run_context import get_git_metadata

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def _git(repository, *arguments):
    subprocess.run(
        ["git", "-c", "user.email=test@example.org", "-c", "user.name=test", *arguments],
        cwd=repository, check=True, capture_output=True,
    )


def test_tracked_modifications_ignore_untracked_files(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "tracked.txt").write_text("first\n")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "commit", "-q", "-m", "initial")

    clean = get_git_metadata(tmp_path)
    assert clean["commit"] is not None and len(clean["commit"]) == 40
    assert clean["dirty"] is False
    assert clean["tracked_modifications"] == 0

    (tmp_path / "untracked.txt").write_text("scratch\n")
    untracked_only = get_git_metadata(tmp_path)
    assert untracked_only["dirty"] is True
    assert untracked_only["tracked_modifications"] == 0

    (tmp_path / "tracked.txt").write_text("second\n")
    modified = get_git_metadata(tmp_path)
    assert modified["tracked_modifications"] == 1
    assert modified["commit"] == clean["commit"]


def test_no_repository_gives_empty_metadata(tmp_path):
    assert get_git_metadata(tmp_path) == {}
