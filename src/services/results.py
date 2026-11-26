"""Access to pipeline run outputs (DATA_ROOT/runs/pipeline/<name>/) as typed objects and DataFrames."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import pandas as pd

from src.common.config import Settings


@dataclass
class PipelineRun:
    name: str
    path: Path
    record: dict = field(repr=False)

    @property
    def results(self) -> dict:
        return self.record["results"]

    @property
    def video(self) -> Path | None:
        p = self.path / "annotated.mp4"
        return p if p.is_file() else None

    @property
    def snapshots(self) -> list[Path]:
        return sorted((self.path / "snapshots").glob("*.jpg"))

    def _csv(self, name: str) -> pd.DataFrame:
        p = self.path / name
        return pd.read_csv(p) if p.is_file() and p.stat().st_size > 0 else pd.DataFrame()

    @cached_property
    def frames(self) -> pd.DataFrame:
        return self._csv("frames.csv")

    @cached_property
    def detections(self) -> pd.DataFrame:
        return self._csv("detections.csv")

    @cached_property
    def tracks(self) -> pd.DataFrame:
        return self._csv("tracks.csv")

    @cached_property
    def track_points(self) -> pd.DataFrame:
        return self._csv("track_points.csv")

    @cached_property
    def events(self) -> pd.DataFrame:
        return self._csv("events.csv")

    @cached_property
    def intervals(self) -> pd.DataFrame:
        df = self._csv("intervals.csv")
        if not df.empty:
            df["minute"] = (df["interval_start_s"] // 60).astype(int)
            df["movement"] = df["line"] + ":" + df["direction"]
        return df

    def volume_series(self, movement: str | None = None) -> pd.DataFrame:
        """Per-interval total counts (all classes) for one movement, or all movements summed."""
        df = self.intervals
        if df.empty:
            return pd.DataFrame(columns=["minute", "count", "degraded"])
        if movement:
            df = df[df["movement"] == movement]
        out = df.groupby("minute").agg(
            count=("count", "sum"), degraded=("reliability", lambda s: bool((s != "ok").any()))
        )
        return out.reset_index()


class RunStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Settings().data_root / "runs" / "pipeline"

    def list(self) -> list[PipelineRun]:
        runs = []
        if not self.root.is_dir():
            return runs
        for path in sorted(self.root.iterdir()):
            record_path = path / "run.json"
            if record_path.is_file():
                try:
                    runs.append(PipelineRun(path.name, path, json.loads(record_path.read_text(encoding="utf-8"))))
                except json.JSONDecodeError:
                    continue
        return runs

    def get(self, name: str) -> PipelineRun:
        path = self.root / name
        return PipelineRun(name, path, json.loads((path / "run.json").read_text(encoding="utf-8")))
