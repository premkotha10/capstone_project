"""Run vehicle detection on a video: detections JSONL + annotated MP4 + run record.

Usage:
  python scripts/run_detection.py <video.mp4> [--config configs/base.yaml] [--stride 1] [--max-frames N]
                                  [--out DIR] [--no-video]

Outputs (default DIR = DATA_ROOT/runs/detection/<video_id>/):
  detections.jsonl   one line per processed frame: {frame_index, timestamp_s, detections:[...]}
  annotated.mp4      boxes, class and confidence drawn on each processed frame
  run.json           config + hash, input checksum, model checksum, environment, latency and count summary
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import PROJECT_ROOT, Settings, load_config  # noqa: E402
from src.common.hardware import select_device  # noqa: E402
from src.common.logging_setup import configure_logging  # noqa: E402
from src.common.provenance import sha256_file  # noqa: E402
from src.detection.detector import UltralyticsDetector  # noqa: E402
from src.detection.visualize import draw_detections  # noqa: E402
from src.evaluation.latency import latency_stats  # noqa: E402
from src.evaluation.runs import write_run_record  # noqa: E402
from src.ingestion.video import VideoFileSource, validate_video  # noqa: E402

logger = logging.getLogger("run_detection")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "base.yaml")
    parser.add_argument("--stride", type=int, default=None, help="overrides detection.frame_stride")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    settings = Settings()
    configure_logging(settings.log_level, settings.log_dir, "detection.jsonl")
    config = load_config(args.config)
    stride = args.stride or config.detection.frame_stride

    report = validate_video(args.video)
    for warning in report.warnings:
        logger.warning("%s: %s", args.video.name, warning)
    if not report.ok:
        logger.error("video rejected: %s", "; ".join(report.errors))
        return 2
    meta = report.meta
    assert meta is not None
    out_dir = args.out or settings.data_root / "runs" / "detection" / meta.video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(settings.device)
    detector = UltralyticsDetector(config.detection, settings.models_root, device=device)
    source = VideoFileSource(args.video, stride=stride, max_frames=args.max_frames)
    out_fps = max(meta.fps / stride, 1.0)
    writer = None
    if not args.no_video:
        writer = cv2.VideoWriter(
            str(out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (meta.width, meta.height)
        )

    latencies: list[float] = []
    class_counts: Counter[str] = Counter()
    frames = 0
    started = time.perf_counter()
    with (out_dir / "detections.jsonl").open("w", encoding="utf-8") as handle:
        for frame in source:
            detections = detector.detect([frame.image], [frame.index])[0]
            latencies.append(detector.last_inference_ms)
            class_counts.update(d.vehicle_class.value for d in detections)
            frames += 1
            handle.write(
                json.dumps(
                    {
                        "frame_index": frame.index,
                        "timestamp_s": round(frame.timestamp_s, 4),
                        "detections": [d.model_dump(mode="json") for d in detections],
                    }
                )
                + "\n"
            )
            if writer is not None:
                writer.write(draw_detections(frame.image, detections))
    if writer is not None:
        writer.release()
    wall = time.perf_counter() - started

    warm = latencies[1:] if len(latencies) > 1 else latencies  # first call includes model warm-up
    stats = latency_stats(warm)
    results = {
        "frames_processed": frames,
        "stride": stride,
        "wall_time_s": round(wall, 2),
        "throughput_fps": round(frames / wall, 2) if wall else None,
        "detector_latency_ms": stats.as_dict(),
        "detections_per_class": dict(class_counts),
        "note": "Detections are raw per-frame boxes; the same vehicle appears in many frames. Not vehicle counts.",
    }
    record = write_run_record(
        out_dir,
        "run",
        config,
        inputs=[{**meta.model_dump(mode="json"), "video": str(args.video), "sha256": sha256_file(args.video)}],
        model=detector.info.__dict__,
        results=results,
    )
    logger.info(
        "processed %d frames in %.1fs (%.1f fps); detector p95 %.1f ms; outputs in %s",
        frames,
        wall,
        results["throughput_fps"] or 0,
        stats.p95_ms,
        record.parent,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
