"""Assemble the Hugging Face Space bundle: only what the dashboard needs to run.

Usage:  python scripts/build_hf_space.py --out build/hf_space

Standard library only, so it runs in CI without installing the project. The bundle never contains data, run outputs,
videos, tests, documents or the evidence manifests.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE_DIRS = ("src", "dashboard", "configs", ".streamlit")
INCLUDE_FILES = ("pyproject.toml", "uv.lock", ".python-version", "LICENSE")
SPACE_FILES = {"deploy/hf_space/README.md": "README.md", "deploy/hf_space/Dockerfile": "Dockerfile"}
EXCLUDE = shutil.ignore_patterns("__pycache__", "*.pyc", ".gitkeep", "*.mp4", "*.avi", "*.jpg", "*.png", "*.pt")
FORBIDDEN_SUFFIXES = {".mp4", ".avi", ".jpg", ".jpeg", ".png", ".pt", ".onnx", ".env"}
MAX_FILE_BYTES = 5 * 1024 * 1024  # Spaces reject large files without Git LFS; nothing in the bundle should be big


def build(out: Path) -> list[Path]:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for d in INCLUDE_DIRS:
        shutil.copytree(ROOT / d, out / d, ignore=EXCLUDE)
    for f in INCLUDE_FILES:
        shutil.copy2(ROOT / f, out / f)
    for src, dst in SPACE_FILES.items():
        shutil.copy2(ROOT / src, out / dst)
    (out / ".dockerignore").write_text(".git\n**/__pycache__\n", encoding="utf-8")
    files = sorted(p for p in out.rglob("*") if p.is_file())
    for p in files:
        if p.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise SystemExit(f"refusing to bundle {p.relative_to(out)}")
        if p.stat().st_size > MAX_FILE_BYTES:
            raise SystemExit(f"{p.relative_to(out)} is larger than {MAX_FILE_BYTES // 1024**2} MB")
    return files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "hf_space")
    args = parser.parse_args()
    files = build(args.out)
    size = sum(p.stat().st_size for p in files) / 1024
    print(f"bundle: {len(files)} files, {size:.0f} KB -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
