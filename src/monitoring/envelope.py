"""Per-frame operating-condition monitor (FR-7, NT-1/NT-2 detectors, NFR-6 quality indicators).

Cheap image statistics plus detection-overlap statistics classify each frame's condition. Frames outside the
conditions the system is (or will be) validated for are flagged DEGRADED so their counts are labelled, never shown
as normal results.

THRESHOLDS ARE PROVISIONAL. They were chosen by inspecting a handful of Barbados frames and are recorded here and in
every run record. They must be calibrated on development data (never on the AC-3 partition) before O2 is frozen.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from src.common.types import BoundingBox, ReliabilityFlag
from src.evaluation.detection_metrics import iou_matrix


@dataclass(frozen=True)
class EnvelopeThresholds:
    night_luma: float = 70.0  # mean luma (0-255) below this -> night / low light
    low_contrast_std: float = 28.0  # luma standard deviation below this -> fog, glare, heavy rain haze
    blur_laplacian: float = 40.0  # variance of Laplacian below this -> blur / rain on lens
    dense_overlap_iou: float = 0.3  # a detection overlapping another above this IoU is "occluded"
    dense_occluded_fraction: float = 0.35  # share of occluded detections above this -> dense occlusion
    dense_min_detections: int = 8  # occlusion is only judged with at least this many detections


@dataclass(frozen=True)
class FrameCondition:
    luma: float
    contrast: float
    sharpness: float
    detections: int
    occluded_fraction: float
    labels: tuple[str, ...]  # e.g. ("night",) or ("dense_occlusion",); empty = nominal daytime
    reliability: ReliabilityFlag

    def as_row(self) -> dict[str, object]:
        row = asdict(self)
        row["labels"] = "|".join(self.labels) if self.labels else "nominal"
        row["reliability"] = self.reliability.value
        return row


def image_statistics(image: np.ndarray, downscale: int = 4) -> tuple[float, float, float]:
    """(mean luma, luma std, Laplacian variance) on a downscaled grey image for speed."""
    small = image[::downscale, ::downscale]
    grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return float(grey.mean()), float(grey.std()), float(cv2.Laplacian(grey, cv2.CV_64F).var())


def occluded_fraction(boxes: Sequence[BoundingBox], iou_threshold: float) -> float:
    if len(boxes) < 2:
        return 0.0
    arr = np.array([[b.x1, b.y1, b.x2, b.y2] for b in boxes], dtype=float)
    ious = iou_matrix(arr, arr)
    np.fill_diagonal(ious, 0.0)
    return float((ious.max(axis=1) > iou_threshold).mean())


class EnvelopeMonitor:
    def __init__(self, thresholds: EnvelopeThresholds | None = None) -> None:
        self.t = thresholds or EnvelopeThresholds()

    def assess(self, image: np.ndarray, boxes: Sequence[BoundingBox]) -> FrameCondition:
        luma, contrast, sharpness = image_statistics(image)
        occ = occluded_fraction(boxes, self.t.dense_overlap_iou)
        labels: list[str] = []
        if luma < self.t.night_luma:
            labels.append("night")
        if contrast < self.t.low_contrast_std:
            labels.append("low_contrast")
        if sharpness < self.t.blur_laplacian:
            labels.append("blur")
        if len(boxes) >= self.t.dense_min_detections and occ > self.t.dense_occluded_fraction:
            labels.append("dense_occlusion")
        reliability = ReliabilityFlag.DEGRADED if labels else ReliabilityFlag.OK
        return FrameCondition(luma, contrast, sharpness, len(boxes), occ, tuple(labels), reliability)
