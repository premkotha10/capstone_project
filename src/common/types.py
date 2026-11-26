"""Core data types shared across the pipeline.

The `CountSource` distinction is deliberate (SYSTEM_ARCHITECTURE §1, principle 4): human reference counts,
pipeline estimates, model predictions and demo samples are different kinds of evidence and must never be
silently mixed in metrics, reports or charts.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CountSource(StrEnum):
    """Origin of a count value."""

    REFERENCE = "reference"  # human ground truth, or derived from human annotations (documented)
    ESTIMATED = "estimated"  # produced by our detection/tracking/counting pipeline
    PREDICTED = "predicted"  # produced by the forecasting model
    DEMO = "demo"  # illustrative sample data — never valid evidence


class ReliabilityFlag(StrEnum):
    """Whether a result was produced inside the validated operating envelope."""

    OK = "ok"
    DEGRADED = "degraded"  # produced, but outside the validated envelope; must be labelled
    ABSTAIN = "abstain"  # system refused to produce a value


class VehicleClass(StrEnum):
    """Project vehicle taxonomy. Only classes supported by the chosen model/dataset may be claimed."""

    CAR = "car"
    MOTORCYCLE = "motorcycle"
    BUS = "bus"
    TRUCK = "truck"
    BICYCLE = "bicycle"
    VAN = "van"
    OTHER = "other"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class BoundingBox(_Frozen):
    """Axis-aligned box in pixel coordinates (x1, y1) top-left, (x2, y2) bottom-right."""

    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def _check_order(self) -> BoundingBox:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(f"degenerate box: ({self.x1}, {self.y1}, {self.x2}, {self.y2})")
        return self

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Ground-contact point used for homography mapping."""
        return ((self.x1 + self.x2) / 2.0, self.y2)


class VideoMeta(_Frozen):
    """Metadata extracted and validated at ingestion."""

    video_id: str
    path: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0)
    frame_count: int = Field(ge=0)
    duration_s: float = Field(ge=0)
    codec: str | None = None
    sha256: str | None = None
    camera_id: str | None = None
    start_time: datetime | None = None  # only when known from real metadata/filenames


class Detection(_Frozen):
    frame_index: int = Field(ge=0)
    bbox: BoundingBox
    confidence: float = Field(ge=0.0, le=1.0)
    vehicle_class: VehicleClass
    source_class: str  # raw class name from the detector, kept for audit


class Track(_Frozen):
    track_id: int = Field(ge=0)
    vehicle_class: VehicleClass
    frame_indices: tuple[int, ...]
    boxes: tuple[BoundingBox, ...]

    @model_validator(mode="after")
    def _check_lengths(self) -> Track:
        if len(self.frame_indices) != len(self.boxes):
            raise ValueError("frame_indices and boxes must have equal length")
        if list(self.frame_indices) != sorted(set(self.frame_indices)):
            raise ValueError("frame_indices must be strictly increasing")
        return self


class IntervalCount(_Frozen):
    """Vehicle count for one camera / line / direction / class over a time interval."""

    camera_id: str
    interval_start_s: float = Field(ge=0)
    interval_end_s: float
    vehicle_class: VehicleClass
    direction: str | None = None
    count: int = Field(ge=0)
    source: CountSource
    reliability: ReliabilityFlag = ReliabilityFlag.OK

    @model_validator(mode="after")
    def _check_interval(self) -> IntervalCount:
        if self.interval_end_s <= self.interval_start_s:
            raise ValueError("interval_end_s must be greater than interval_start_s")
        return self

    @property
    def is_valid_evidence(self) -> bool:
        """Demo samples are never evidence; abstained values carry no measurement."""
        return self.source is not CountSource.DEMO and self.reliability is not ReliabilityFlag.ABSTAIN
