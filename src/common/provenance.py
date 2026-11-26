"""Provenance helpers: file checksums, code version and environment snapshot (D6, reproducibility)."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from src.common.config import PROJECT_ROOT

TRACKED_PACKAGES = (
    "torch",
    "torchvision",
    "ultralytics",
    "lap",
    "opencv-python-headless",
    "numpy",
    "pandas",
    "scikit-learn",
    "xgboost",
    "fastapi",
    "pydantic",
    "streamlit",
)


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 hex digest of a file, read in chunks so large videos do not load into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: Path = PROJECT_ROOT) -> str | None:
    """Current commit hash with a `-dirty` suffix if the tree has uncommitted changes; None outside git."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True, timeout=30
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return f"{commit}-dirty" if status else commit


def package_versions(packages: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment_snapshot() -> dict[str, Any]:
    """Everything needed to state *where* and *with what* a result was produced."""
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "git_commit": git_commit(),
        "packages": package_versions(),
    }
