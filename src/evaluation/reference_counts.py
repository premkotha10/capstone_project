"""Human reference counts: records, storage, annotator agreement and pipeline-vs-reference accuracy.

See docs/COUNTING_PROTOCOL.md. Reference counts are produced by people with the blind counting tool; nothing in this
module creates counts. It only validates, stores, compares and scores them.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.common.config import PROJECT_ROOT, Settings
from src.common.provenance import sha256_file

PROTOCOL_VERSION = "1.0"
REFERENCE_ROOT = PROJECT_ROOT / "data" / "annotations" / "reference_counts"
PLAN_PATH = REFERENCE_ROOT / "plan_v1.csv"
CLASSES = ("car", "bus", "truck", "motorcycle", "bicycle", "other")
Role = Literal["primary", "secondary", "adjudicator"]


class CountCell(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    line: str
    direction: str
    vehicle_class: Literal["car", "bus", "truck", "motorcycle", "bicycle", "other"]
    count: int = Field(ge=0, le=500)


class ReferenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clip_id: str
    clip_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    geometry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    annotator: str = Field(pattern=r"^[A-Za-z0-9_-]{1,32}$")  # initials or roll number, never a full name
    role: Role
    protocol_version: str = PROTOCOL_VERSION
    started_at: datetime
    finished_at: datetime
    cells: list[CountCell]
    uncertain: bool = False
    notes: str = Field(default="", max_length=2000)
    adjudicates: list[str] = Field(default_factory=list)  # annotators whose disagreement this record resolves

    @model_validator(mode="after")
    def _check(self) -> ReferenceRecord:
        keys = [(c.line, c.direction, c.vehicle_class) for c in self.cells]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate count cells")
        if self.finished_at < self.started_at:
            raise ValueError("finished_at precedes started_at")
        if self.role == "adjudicator" and not self.adjudicates:
            raise ValueError("an adjudication must name the annotators it resolves")
        return self

    @property
    def total(self) -> int:
        return sum(c.count for c in self.cells)


@dataclass(frozen=True)
class PlanClip:
    clip_id: str
    path: Path
    camera_id: str
    run: str
    condition_hint: str
    sha256: str

    @property
    def geometry_path(self) -> Path:
        return PROJECT_ROOT / "configs" / "cameras" / f"{self.camera_id}.yaml"


def load_plan(path: Path = PLAN_PATH, settings: Settings | None = None) -> list[PlanClip]:
    settings = settings or Settings()
    clips = []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            raw = row["path"]
            resolved = (
                settings.data_root / raw.removeprefix("DATA_ROOT/")
                if raw.startswith("DATA_ROOT/")
                else PROJECT_ROOT / raw
            )
            clips.append(
                PlanClip(row["clip_id"], resolved, row["camera_id"], row["run"], row["condition_hint"], row["sha256"])
            )
    return clips


class ReferenceStoreError(ValueError):
    pass


def record_path(clip_id: str, annotator: str, root: Path = REFERENCE_ROOT) -> Path:
    return root / annotator / f"{clip_id}.json"


def save_record(record: ReferenceRecord, clip: PlanClip, root: Path = REFERENCE_ROOT, overwrite: bool = False) -> Path:
    """Validate a record against the plan (clip + geometry checksums) and write it."""
    if record.clip_id != clip.clip_id:
        raise ReferenceStoreError("record and plan clip differ")
    if record.clip_sha256 != clip.sha256:
        raise ReferenceStoreError(f"{clip.clip_id}: clip checksum does not match the plan (wrong or modified file)")
    if record.geometry_sha256 != sha256_file(clip.geometry_path):
        raise ReferenceStoreError(f"{clip.clip_id}: counting lines changed since the tool loaded them; reload the page")
    path = record_path(record.clip_id, record.annotator, root)
    if path.exists() and not overwrite:
        raise ReferenceStoreError(f"{path.name} already saved by {record.annotator}; tick 'replace my saved count'")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_records(root: Path = REFERENCE_ROOT) -> list[ReferenceRecord]:
    records = []
    for path in sorted(root.glob("*/*.json")):
        records.append(ReferenceRecord.model_validate(json.loads(path.read_text(encoding="utf-8"))))
    return records


def _cell_table(records: list[ReferenceRecord]) -> pd.DataFrame:
    rows = []
    for r in records:
        for c in r.cells:
            rows.append(
                {
                    "clip_id": r.clip_id,
                    "annotator": r.annotator,
                    "role": r.role,
                    "line": c.line,
                    "direction": c.direction,
                    "vehicle_class": c.vehicle_class,
                    "count": c.count,
                }
            )
    return pd.DataFrame(rows, columns=["clip_id", "annotator", "role", "line", "direction", "vehicle_class", "count"])


CELL_KEY = ["clip_id", "line", "direction", "vehicle_class"]


@dataclass(frozen=True)
class Agreement:
    clips_double_counted: int
    cells: int
    exact_agreement: float
    within_one: float
    mean_abs_diff: float
    icc: float  # ICC(2,1), absolute agreement, single rater


def icc_2_1(matrix: np.ndarray) -> float:
    """Two-way random, absolute agreement, single-rater ICC (Shrout & Fleiss). matrix: n_targets x k_raters."""
    n, k = matrix.shape
    if n < 2 or k < 2:
        return float("nan")
    grand = matrix.mean()
    ms_rows = k * ((matrix.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_cols = n * ((matrix.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    sse = ((matrix - matrix.mean(axis=1, keepdims=True) - matrix.mean(axis=0, keepdims=True) + grand) ** 2).sum()
    ms_err = sse / ((n - 1) * (k - 1))
    denom = ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n
    return float((ms_rows - ms_err) / denom) if denom > 0 else float("nan")


def consolidate(records: list[ReferenceRecord]) -> tuple[pd.DataFrame, Agreement | None]:
    """Combine primary/secondary counts per cell.

    Returns one row per cell with `primary`, `secondary`, `adjudicated` and `reference`:
    the agreed value, the adjudicated value, or NaN when the annotators disagree and no adjudication exists yet.
    Cells counted by only one annotator get status 'single' and no reference (protocol requires two).
    """
    cells = _cell_table(records)
    if cells.empty:
        return pd.DataFrame(columns=[*CELL_KEY, "primary", "secondary", "adjudicated", "reference", "status"]), None
    wide = cells.pivot_table(index=CELL_KEY, columns="role", values="count", aggfunc="first").reset_index()
    for role in ("primary", "secondary", "adjudicator"):
        if role not in wide:
            wide[role] = np.nan
    # a cell missing in one annotator's record but present in the other's means that annotator counted 0 there,
    # provided they saved that clip at all
    saved = defaultdict(set)
    for r in records:
        saved[r.clip_id].add(r.role)
    for role in ("primary", "secondary"):
        has_clip = wide["clip_id"].map(lambda c, role=role: role in saved[c])
        wide.loc[has_clip & wide[role].isna(), role] = 0
    wide = wide.rename(columns={"adjudicator": "adjudicated"})

    def status(row: pd.Series) -> str:
        if not np.isnan(row["adjudicated"]):
            return "adjudicated"
        if np.isnan(row["primary"]) or np.isnan(row["secondary"]):
            return "single"
        return "agreed" if row["primary"] == row["secondary"] else "disagree"

    wide["status"] = wide.apply(status, axis=1)
    wide["reference"] = np.where(
        wide["status"] == "agreed",
        wide["primary"],
        np.where(wide["status"] == "adjudicated", wide["adjudicated"], np.nan),
    )
    both = wide.dropna(subset=["primary", "secondary"])
    agreement = None
    if not both.empty:
        diff = (both["primary"] - both["secondary"]).abs()
        agreement = Agreement(
            clips_double_counted=both["clip_id"].nunique(),
            cells=len(both),
            exact_agreement=float((diff == 0).mean()),
            within_one=float((diff <= 1).mean()),
            mean_abs_diff=float(diff.mean()),
            icc=icc_2_1(both[["primary", "secondary"]].to_numpy(dtype=float)),
        )
    return wide, agreement


def pipeline_clip_counts(run_dir: Path, clip_stem: str) -> pd.DataFrame:
    """Per-clip estimated counts from a pipeline run, attributing each crossing to the clip of its frame."""
    frames = pd.read_csv(run_dir / "frames.csv", usecols=["frame", "clip"])
    events = pd.read_csv(run_dir / "events.csv")
    if events.empty:
        return pd.DataFrame(columns=["line", "direction", "vehicle_class", "estimate"])
    frame_clip = frames.set_index("frame")["clip"]
    events["clip"] = events["frame"].map(frame_clip)
    mine = events[events["clip"] == clip_stem]
    out = mine.groupby(["line", "direction", "class"]).size().rename("estimate").reset_index()
    return out.rename(columns={"class": "vehicle_class"})


@dataclass(frozen=True)
class AccuracySummary:
    clips: int
    reference_total: int
    estimate_total: int
    mae: float
    rmse: float
    wape: float
    accuracy: float  # 1 - WAPE, the proposed KPI-1 operational definition (REQUIREMENTS §4, not yet frozen)
    accuracy_ci95: tuple[float, float]


def score(table: pd.DataFrame, n_boot: int = 2000, seed: int = 41) -> AccuracySummary | None:
    """Score estimates against references on cells with a final reference. Bootstrap CI resamples clips."""
    t = table.dropna(subset=["reference"])
    if t.empty or t["reference"].sum() == 0:
        return None
    err = t["estimate"] - t["reference"]
    rng = np.random.default_rng(seed)
    clips = t["clip_id"].unique()
    by_clip = {c: g for c, g in t.groupby("clip_id")}
    boots = []
    for _ in range(n_boot):
        sample = pd.concat([by_clip[c] for c in rng.choice(clips, size=len(clips), replace=True)])
        denom = sample["reference"].sum()
        if denom > 0:
            boots.append(1 - (sample["estimate"] - sample["reference"]).abs().sum() / denom)
    wape = float(err.abs().sum() / t["reference"].sum())
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return AccuracySummary(
        clips=len(clips),
        reference_total=int(t["reference"].sum()),
        estimate_total=int(t["estimate"].sum()),
        mae=float(err.abs().mean()),
        rmse=float(np.sqrt((err**2).mean())),
        wape=wape,
        accuracy=1 - wape,
        accuracy_ci95=(float(lo), float(hi)),
    )


def compare_with_pipeline(consolidated: pd.DataFrame, plan: list[PlanClip], runs_root: Path) -> pd.DataFrame:
    """Join reference cells with pipeline estimates for every planned clip that has references."""
    frames = []
    for clip in plan:
        ref = consolidated[consolidated["clip_id"] == clip.clip_id]
        if ref.empty:
            continue
        run_dir = runs_root / clip.run
        est = (
            pipeline_clip_counts(run_dir, clip.path.stem)
            if (run_dir / "events.csv").is_file()
            else pd.DataFrame(columns=["line", "direction", "vehicle_class", "estimate"])
        )
        merged = ref.merge(est, on=["line", "direction", "vehicle_class"], how="outer")
        merged["clip_id"] = clip.clip_id
        merged["condition"] = clip.condition_hint
        merged["estimate"] = merged["estimate"].fillna(0)
        frames.append(merged)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
