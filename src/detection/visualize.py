"""Drawing helpers for detections (boxes, class labels, confidence)."""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

from src.common.types import Detection, VehicleClass

# BGR colours, colour-blind-friendly (Okabe-Ito)
CLASS_COLOURS: dict[VehicleClass, tuple[int, int, int]] = {
    VehicleClass.CAR: (178, 114, 0),
    VehicleClass.MOTORCYCLE: (0, 159, 230),
    VehicleClass.BUS: (115, 158, 0),
    VehicleClass.TRUCK: (167, 121, 204),
    VehicleClass.BICYCLE: (66, 228, 240),
    VehicleClass.VAN: (0, 94, 213),
    VehicleClass.OTHER: (128, 128, 128),
}


def draw_detections(image: np.ndarray, detections: Iterable[Detection], label_prefix: str = "") -> np.ndarray:
    """Return a copy of `image` with boxes and '<class> <conf>' labels drawn."""
    canvas = image.copy()
    for det in detections:
        colour = CLASS_COLOURS.get(det.vehicle_class, (255, 255, 255))
        p1 = (int(det.bbox.x1), int(det.bbox.y1))
        p2 = (int(det.bbox.x2), int(det.bbox.y2))
        cv2.rectangle(canvas, p1, p2, colour, 2)
        label = f"{label_prefix}{det.vehicle_class.value} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        top = max(p1[1] - th - 4, 0)
        cv2.rectangle(canvas, (p1[0], top), (p1[0] + tw + 4, top + th + 4), colour, -1)
        cv2.putText(canvas, label, (p1[0] + 2, top + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return canvas
