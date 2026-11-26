"""Multi-object tracking with ByteTrack (FR-5).

Implementation: Ultralytics' `BYTETracker` (AGPL-3.0, already a project dependency — ADR-2/ADR-3). `supervision`
0.32 no longer ships a tracker, so it is not used here.

ByteTrack associates high-confidence detections first, then tries to recover lost tracks with low-confidence ones
(between `track_low_thresh` and `track_high_thresh`). That second stage only helps if the detector passes
low-confidence boxes through; with the O2 detector threshold of 0.25 it is effectively unused, which is a candidate
O3 lever (lower detector threshold + low-score association), to be measured, not assumed.

Each track's vehicle class is the confidence-weighted majority of its detections' classes, smoothing single-frame
class flips (e.g. a minibus alternating car/bus).
"""

from __future__ import annotations

from argparse import Namespace
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from src.common.config import TrackingConfig
from src.common.types import BoundingBox, Detection, VehicleClass

_CLASS_INDEX = {c: i for i, c in enumerate(VehicleClass)}
_INDEX_CLASS = {i: c for c, i in _CLASS_INDEX.items()}


class _Detections:
    """Minimal results-like container accepted by Ultralytics' BYTETracker."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xyxy = xyxy.reshape(-1, 4).astype(np.float32)
        self.conf = conf.astype(np.float32)
        self.cls = cls.astype(np.float32)

    @property
    def xywh(self) -> np.ndarray:
        out = self.xyxy.copy()
        out[:, 2] = self.xyxy[:, 2] - self.xyxy[:, 0]
        out[:, 3] = self.xyxy[:, 3] - self.xyxy[:, 1]
        out[:, 0] = self.xyxy[:, 0] + out[:, 2] / 2
        out[:, 1] = self.xyxy[:, 1] + out[:, 3] / 2
        return out

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, index: np.ndarray) -> _Detections:
        return _Detections(self.xyxy[index], self.conf[index], self.cls[index])


@dataclass(frozen=True)
class TrackedObject:
    track_id: int
    frame_index: int
    bbox: BoundingBox
    confidence: float
    detected_class: VehicleClass  # class of this frame's matched detection


@dataclass
class TrackHistory:
    track_id: int
    first_frame: int
    last_frame: int
    points: list[tuple[int, float, float]] = field(default_factory=list)  # (frame, x, y) bottom-centre
    class_votes: dict[VehicleClass, float] = field(default_factory=lambda: defaultdict(float))

    @property
    def vehicle_class(self) -> VehicleClass:
        return max(self.class_votes.items(), key=lambda kv: kv[1])[0] if self.class_votes else VehicleClass.OTHER

    @property
    def length(self) -> int:
        return len(self.points)


class VehicleTracker:
    """Stateful ByteTrack wrapper. Call `update` once per processed frame, in order (also on empty frames)."""

    def __init__(
        self, config: TrackingConfig, frame_rate: float, low_threshold: float = 0.1, max_history_points: int = 400
    ) -> None:
        from ultralytics.trackers.byte_tracker import BYTETracker

        self.config = config
        self.frame_rate = frame_rate
        self._tracker = BYTETracker(
            Namespace(
                track_high_thresh=config.track_activation_threshold,
                track_low_thresh=low_threshold,
                new_track_thresh=config.track_activation_threshold,
                track_buffer=config.lost_track_buffer,
                match_thresh=config.minimum_matching_threshold,
                fuse_score=True,
            )
        )
        self.histories: dict[int, TrackHistory] = {}
        self.max_history_points = max_history_points

    def update(self, frame_index: int, detections: Sequence[Detection]) -> list[TrackedObject]:
        if detections:
            dets = _Detections(
                np.array([[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2] for d in detections]),
                np.array([d.confidence for d in detections]),
                np.array([_CLASS_INDEX[d.vehicle_class] for d in detections]),
            )
        else:
            dets = _Detections(np.zeros((0, 4)), np.zeros(0), np.zeros(0))
        output = self._tracker.update(dets)
        tracked: list[TrackedObject] = []
        for row in output:
            x1, y1, x2, y2, tid, score, cls = (float(v) for v in row[:7])
            if x2 <= x1 or y2 <= y1:
                continue
            obj = TrackedObject(
                int(tid), frame_index, BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2), score, _INDEX_CLASS[int(cls)]
            )
            tracked.append(obj)
            hist = self.histories.get(obj.track_id)
            if hist is None:
                hist = self.histories[obj.track_id] = TrackHistory(obj.track_id, frame_index, frame_index)
            hist.last_frame = frame_index
            hist.points.append((frame_index, (x1 + x2) / 2.0, y2))
            if len(hist.points) > self.max_history_points:
                del hist.points[: len(hist.points) - self.max_history_points]
            hist.class_votes[obj.detected_class] += obj.confidence
        return tracked
