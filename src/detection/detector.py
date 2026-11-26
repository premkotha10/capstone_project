"""YOLO-based vehicle detection (FR-5).

* Only COCO classes that are road vehicles are kept, through an explicit class map. COCO has no "van" class, so
  vans are reported as whatever the model predicts (usually car/truck); we do not claim van detection.
* Model weights live under DSCI41_MODELS_ROOT and their SHA-256 is recorded with every run (model bundle provenance).
* Ultralytics usage analytics are disabled (no data leaves the machine except the one-time weight download).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from src.common.config import DetectionConfig
from src.common.provenance import sha256_file
from src.common.types import BoundingBox, Detection, VehicleClass

logger = logging.getLogger(__name__)

# COCO-80 class index -> project vehicle class
COCO_VEHICLE_MAP: dict[int, VehicleClass] = {
    1: VehicleClass.BICYCLE,
    2: VehicleClass.CAR,
    3: VehicleClass.MOTORCYCLE,
    5: VehicleClass.BUS,
    7: VehicleClass.TRUCK,
}


class ModelBundleError(RuntimeError):
    pass


class VehicleDetector(Protocol):
    def detect(self, images: Sequence[np.ndarray], frame_indices: Sequence[int]) -> list[list[Detection]]: ...


@dataclass(frozen=True)
class ModelInfo:
    name: str
    path: str
    sha256: str
    device: str
    image_size: int
    class_map: dict[int, str]


def _disable_ultralytics_telemetry() -> None:
    try:
        from ultralytics import settings

        if settings.get("sync", False):
            settings.update({"sync": False})
    except Exception as exc:  # never fail detection because of a settings file
        logger.warning("could not disable ultralytics sync: %s", exc)


def ensure_weights(model_name: str, models_root: Path) -> Path:
    """Return a local weights path, downloading official Ultralytics release weights on first use."""
    path = Path(model_name)
    if path.is_file():
        return path
    target = models_root / "detection" / model_name
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    from ultralytics.utils.downloads import attempt_download_asset

    downloaded = Path(attempt_download_asset(str(target)))
    if not downloaded.is_file():
        raise ModelBundleError(f"could not obtain weights {model_name!r}")
    return downloaded


class UltralyticsDetector:
    """Thin, typed wrapper around an Ultralytics YOLO model."""

    def __init__(
        self,
        config: DetectionConfig,
        models_root: Path,
        device: str = "cpu",
        class_map: dict[int, VehicleClass] | None = None,
    ) -> None:
        _disable_ultralytics_telemetry()
        from ultralytics import YOLO

        self.config = config
        self.device = device
        self.class_map = class_map or COCO_VEHICLE_MAP
        weights = ensure_weights(config.model, models_root)
        try:
            self.model = YOLO(str(weights))
        except Exception as exc:
            raise ModelBundleError(f"failed to load {weights}: {exc}") from exc
        names = self.model.names
        for index, vehicle_class in self.class_map.items():
            if index not in names:
                raise ModelBundleError(f"class index {index} ({vehicle_class}) not in model classes")
        self.info = ModelInfo(
            name=config.model,
            path=str(weights),
            sha256=sha256_file(weights),
            device=device,
            image_size=config.image_size,
            class_map={i: names[i] for i in self.class_map},
        )
        self.last_inference_ms: float = 0.0

    def detect(self, images: Sequence[np.ndarray], frame_indices: Sequence[int]) -> list[list[Detection]]:
        if len(images) != len(frame_indices):
            raise ValueError("images and frame_indices must have equal length")
        if not images:
            return []
        start = time.perf_counter_ns()
        results = self.model.predict(
            list(images),
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.image_size,
            classes=sorted(self.class_map),
            device=self.device,
            verbose=False,
        )
        self.last_inference_ms = (time.perf_counter_ns() - start) / 1e6
        output: list[list[Detection]] = []
        for frame_index, result in zip(frame_indices, results, strict=True):
            boxes = result.boxes
            xyxy = boxes.xyxy.cpu().numpy() if len(boxes) else np.zeros((0, 4))
            confs = boxes.conf.cpu().numpy() if len(boxes) else np.zeros(0)
            classes = boxes.cls.cpu().numpy().astype(int) if len(boxes) else np.zeros(0, dtype=int)
            frame_dets: list[Detection] = []
            for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, classes, strict=True):
                if cls not in self.class_map or x2 <= x1 or y2 <= y1:
                    continue
                frame_dets.append(
                    Detection(
                        frame_index=frame_index,
                        bbox=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                        confidence=float(conf),
                        vehicle_class=self.class_map[cls],
                        source_class=str(result.names[cls]),
                    )
                )
            output.append(frame_dets)
        return output
