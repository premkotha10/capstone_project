"""Structured logging: human-readable console output plus JSON-lines file logs (NFR-6)."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

_STANDARD_ATTRS = set(vars(logging.makeLogRecord({})).keys()) | {"message", "asctime"}


class JsonLinesFormatter(logging.Formatter):
    """One JSON object per line; any `extra={...}` fields are included."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", log_dir: Path | None = None, filename: str = "dsci41.jsonl") -> None:
    """Configure the root logger once. Safe to call repeatedly (handlers are replaced, not duplicated)."""
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root.addHandler(console)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / filename, encoding="utf-8")
        file_handler.setFormatter(JsonLinesFormatter())
        root.addHandler(file_handler)
