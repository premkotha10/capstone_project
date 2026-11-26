"""Run the full pipeline (detect -> track -> count) on one video or a time-ordered set of clips.

Examples:
  # one file, no counting lines (detection + tracking only)
  python scripts/process_video.py path/to/video.mp4

  # a 23-minute Barbados window from camera 1 with its counting lines
  python scripts/process_video.py --glob "normanniles1_2025-10-21-13-*.mp4" --camera normanniles1 --name cam1_morning

Outputs go to DATA_ROOT/runs/pipeline/<name>/ (see src/pipeline/runner.py for the file list).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import PROJECT_ROOT, Settings, load_config  # noqa: E402
from src.common.hardware import select_device  # noqa: E402
from src.common.logging_setup import configure_logging  # noqa: E402
from src.counting.line_counter import CameraGeometry  # noqa: E402
from src.detection.detector import UltralyticsDetector  # noqa: E402
from src.pipeline.runner import process_clips  # noqa: E402

logger = logging.getLogger("process_video")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="*", type=Path)
    parser.add_argument("--glob", help="pattern matched under DATA_ROOT/raw/barbados_traffic/<camera>/")
    parser.add_argument("--camera", help="camera id; loads configs/cameras/<camera>.yaml if present")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "o2_a.yaml")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    settings = Settings()
    configure_logging(settings.log_level, settings.log_dir, "pipeline.jsonl")
    clips = list(args.videos)
    if args.glob:
        if not args.camera:
            parser.error("--glob needs --camera")
        clips += sorted((settings.data_root / "raw" / "barbados_traffic" / args.camera).glob(args.glob))
    if not clips:
        parser.error("no input videos")
    geometry = None
    if args.camera and (PROJECT_ROOT / "configs" / "cameras" / f"{args.camera}.yaml").is_file():
        geometry = CameraGeometry.load(PROJECT_ROOT / "configs" / "cameras" / f"{args.camera}.yaml")
    config = load_config(args.config)
    name = args.name or clips[0].stem
    detector = UltralyticsDetector(config.detection, settings.models_root, device=select_device(settings.device))

    def progress(done: int, total: int, message: str) -> None:
        if done % 500 == 0:
            logger.info("%d/%d frames — %s", done, total, message)

    summary = process_clips(
        clips,
        config,
        detector,
        settings.data_root / "runs" / "pipeline" / name,
        geometry=geometry,
        camera_id=args.camera or "uploaded",
        stride=args.stride,
        max_frames=args.max_frames,
        write_video=not args.no_video,
        progress=progress,
        run_name=name,
    )
    logger.info(
        "done: %d frames, %d tracks, %d count events in %.0fs -> %s",
        summary.frames,
        summary.tracks,
        summary.events,
        summary.wall_s,
        summary.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
