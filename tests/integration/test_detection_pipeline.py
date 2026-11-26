"""Integration: ingestion -> detection -> run outputs on real data (skipped when data/weights are absent)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.common.config import PROJECT_ROOT, DetectionConfig, Settings
from src.common.types import Detection, VehicleClass
from src.ingestion.video import VideoFileSource

pytestmark = [pytest.mark.slow, pytest.mark.needs_data]

SETTINGS = Settings()
WEIGHTS = SETTINGS.models_root / "detection" / "yolo11n.pt"
CLIPS = sorted((SETTINGS.data_root / "raw" / "barbados_traffic").glob("normanniles1/*.mp4"))


@pytest.fixture(scope="module")
def detector():  # noqa: ANN201
    if not WEIGHTS.is_file():
        pytest.skip("detector weights not downloaded")
    from src.detection.detector import UltralyticsDetector

    return UltralyticsDetector(DetectionConfig(model="yolo11n.pt"), SETTINGS.models_root, device="cpu")


@pytest.mark.skipif(not CLIPS, reason="no Barbados clips downloaded")
def test_detector_returns_typed_vehicle_detections(detector) -> None:  # noqa: ANN001
    frames = list(VideoFileSource(CLIPS[-1], stride=100, max_frames=3))
    results = detector.detect([f.image for f in frames], [f.index for f in frames])
    assert len(results) == len(frames)
    allowed = {VehicleClass.CAR, VehicleClass.MOTORCYCLE, VehicleClass.BUS, VehicleClass.TRUCK, VehicleClass.BICYCLE}
    for frame, dets in zip(frames, results, strict=True):
        for det in dets:
            assert isinstance(det, Detection)
            assert det.frame_index == frame.index
            assert det.vehicle_class in allowed
            assert det.confidence >= detector.config.confidence_threshold
            assert 0 <= det.bbox.x1 < det.bbox.x2 <= frame.image.shape[1] + 1
    assert len(detector.info.sha256) == 64


@pytest.mark.skipif(not CLIPS or not WEIGHTS.is_file(), reason="needs clip and weights")
def test_run_detection_script_produces_outputs(tmp_path: Path) -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_detection.py"),
        str(CLIPS[0]),
        "--stride",
        "50",
        "--max-frames",
        "5",
        "--out",
        str(tmp_path),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert completed.returncode == 0, completed.stderr[-2000:]
    lines = (tmp_path / "detections.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    assert json.loads(lines[1])["frame_index"] == 50
    record = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert record["inputs"][0]["sha256"] and record["model"]["sha256"]
    assert (tmp_path / "annotated.mp4").stat().st_size > 0


def test_run_detection_rejects_corrupt_video(tmp_path: Path) -> None:
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"\x00" * 4096)
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "run_detection.py"), str(bad), "--out", str(tmp_path)]
    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert completed.returncode == 2
    assert "video rejected" in completed.stderr + completed.stdout
