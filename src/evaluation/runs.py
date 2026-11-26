"""Run records: every reported number is written next to the provenance needed to reproduce it."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.common.config import PipelineConfig
from src.common.provenance import environment_snapshot


def run_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"


def write_run_record(
    out_dir: Path,
    name: str,
    config: PipelineConfig,
    inputs: list[dict[str, Any]],
    model: dict[str, Any],
    results: dict[str, Any],
    notes: list[str] | None = None,
) -> Path:
    """Write `<out_dir>/<name>.json` containing config (+hash), inputs (manifest ids/hashes), model, env, results."""
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "run": name,
        "config": config.model_dump(mode="json"),
        "config_hash": config.config_hash(),
        "inputs": inputs,
        "model": model,
        "environment": environment_snapshot(),
        "results": results,
        "notes": notes or [],
    }
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return path
