"""Process one or more time-contiguous clips from one camera into counts, tracks and an annotated video.

Clips are treated as one continuous stream (the tracker keeps state across clip boundaries), which matches the
Barbados layout of consecutive one-minute files.

Outputs in `out_dir`:
  run.json          provenance (config + hash, input checksums, model, env) and summary
  frames.csv        per processed frame: condition statistics, reliability, counts, stage latencies
  detections.csv    every detection (frame, class, confidence, box)
  tracks.csv        one row per track (class, first/last frame, length, displacement, heading)
  track_points.csv  trajectory points (track, frame, x, y) for visualisation
  events.csv        one row per line crossing (track, line, direction, class, time, reliability)
  intervals.csv     per-interval counts per line/direction/class (CountSource = estimated)
  annotated.mp4     H.264 video with boxes, IDs, trails, lines and running counts
  snapshots/*.jpg   a few annotated frames (local use only; not for publication — see RISK R-20)
"""

from __future__ import annotations

import csv
import logging
import math
import shutil
import subprocess
import time
from collections import Counter, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.common.config import PipelineConfig
from src.common.provenance import sha256_file
from src.common.types import ReliabilityFlag
from src.counting.line_counter import CameraGeometry, LineCounter, aggregate_intervals
from src.detection.detector import UltralyticsDetector
from src.detection.visualize import CLASS_COLOURS
from src.evaluation.latency import latency_stats
from src.evaluation.runs import write_run_record
from src.ingestion.video import VideoFileSource, validate_video
from src.monitoring.envelope import EnvelopeMonitor
from src.tracking.tracker import VehicleTracker

logger = logging.getLogger(__name__)
ProgressFn = Callable[[int, int, str], None]  # (frames_done, frames_total, message)


@dataclass
class RunSummary:
    out_dir: Path
    frames: int
    duration_s: float
    events: int
    tracks: int
    wall_s: float


def _draw_overlay(
    image: np.ndarray,
    tracked,  # noqa: ANN001
    histories,  # noqa: ANN001
    geometry: CameraGeometry | None,
    totals: Counter,
    condition_text: str,
    degraded: bool,
) -> np.ndarray:
    """Draw lines, boxes, IDs, trails and a counts panel, scaled to the frame width (960 px = scale 1)."""
    canvas = image.copy()
    k = max(image.shape[1] / 960.0, 0.75)
    thick = max(int(round(2 * k)), 1)
    thin = max(int(round(k)), 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if geometry is not None:
        for line in geometry.lines:
            p1 = tuple(int(v) for v in line.p1)
            p2 = tuple(int(v) for v in line.p2)
            cv2.line(canvas, p1, p2, (0, 215, 255), thick)
            cv2.putText(
                canvas, line.name, (p1[0], max(p1[1] - int(6 * k), int(12 * k))), font, 0.45 * k, (0, 215, 255), thin
            )
    for obj in tracked:
        hist = histories[obj.track_id]
        colour = CLASS_COLOURS.get(hist.vehicle_class, (255, 255, 255))
        pts = np.array([(int(x), int(y)) for _, x, y in hist.points[-30:]], dtype=np.int32)
        if len(pts) > 1:
            cv2.polylines(canvas, [pts], False, colour, thick)
        b = obj.bbox
        cv2.rectangle(canvas, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), colour, thick)
        cv2.putText(
            canvas,
            f"#{obj.track_id} {hist.vehicle_class.value} {obj.confidence:.2f}",
            (int(b.x1), max(int(b.y1) - int(4 * k), int(10 * k))),
            font,
            0.4 * k,
            colour,
            thin,
        )
    panel = [f"{line}:{direction} {n}" for (line, direction), n in sorted(totals.items())] + [condition_text]
    row = int(18 * k)
    width = int(max((cv2.getTextSize(t, font, 0.45 * k, thin)[0][0] for t in panel), default=100) + 12 * k)
    cv2.rectangle(
        canvas, (int(6 * k), int(30 * k)), (int(6 * k) + width, int(36 * k) + row * len(panel)), (0, 0, 0), -1
    )
    y = int(46 * k)
    for text in panel:
        colour = (80, 80, 255) if (text == condition_text and degraded) else (255, 255, 255)
        cv2.putText(canvas, text, (int(12 * k), y), font, 0.45 * k, colour, thin)
        y += row
    return canvas


class H264Writer:
    """Stream BGR frames into FFmpeg's libx264 encoder (no large intermediate file).

    Falls back to OpenCV's mp4v writer when FFmpeg is unavailable (that file will not play in most browsers).
    """

    def __init__(self, path: Path, fps: float, size: tuple[int, int]) -> None:
        self.path = path
        self.h264 = False
        exe = shutil.which("ffmpeg")
        if exe is not None:
            cmd = [
                exe,
                "-v",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{size[0]}x{size[1]}",
                "-r",
                f"{fps:.3f}",
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "28",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ]
            try:
                self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                self.h264 = True
                return
            except OSError as exc:
                logger.warning("could not start ffmpeg (%s); falling back to mp4v", exc)
        self._cv = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def write(self, frame: np.ndarray) -> None:
        if self.h264:
            self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            self._cv.write(frame)

    def release(self) -> bool:
        """Close the stream; returns True when an H.264 file was produced successfully."""
        if not self.h264:
            self._cv.release()
            return False
        self._proc.stdin.close()
        _, err = self._proc.communicate(timeout=600)
        if self._proc.returncode != 0:
            logger.warning("ffmpeg exited with %s: %s", self._proc.returncode, err.decode(errors="replace")[-500:])
            return False
        return True


def process_clips(
    clips: Sequence[Path],
    config: PipelineConfig,
    detector: UltralyticsDetector,
    out_dir: Path,
    geometry: CameraGeometry | None = None,
    camera_id: str = "camera",
    stride: int = 2,
    max_frames: int | None = None,
    write_video: bool = True,
    snapshot_every_s: float = 60.0,
    progress: ProgressFn | None = None,
    run_name: str = "run",
    extra_inputs: list[dict] | None = None,
    max_width: int | None = None,
) -> RunSummary:
    """Process clips; `max_width` downscales frames (and the counting lines) once on entry, e.g. 4K -> 1280 px."""
    if not clips:
        raise ValueError("no clips given")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "snapshots").mkdir(exist_ok=True)
    reports = []
    for clip in clips:
        report = validate_video(clip)
        if not report.ok:
            raise ValueError(f"{clip.name} rejected: {'; '.join(report.errors)}")
        reports.append(report.meta)
    fps = reports[0].fps
    if any(abs(m.fps - fps) > 0.01 or (m.width, m.height) != (reports[0].width, reports[0].height) for m in reports):
        raise ValueError("clips differ in fps or resolution; process them separately")
    width, height = reports[0].width, reports[0].height
    scale = min(1.0, max_width / width) if max_width else 1.0
    if scale < 1.0:
        width, height = int(width * scale) // 2 * 2, int(height * scale) // 2 * 2
        if geometry is not None:
            geometry = geometry.scaled(scale)
    total_frames = sum(m.frame_count for m in reports) // stride
    if max_frames:
        total_frames = min(total_frames, max_frames)

    tracker = VehicleTracker(config.tracking, frame_rate=fps / stride)
    counter = LineCounter(geometry, config.counting.min_track_length) if geometry else None
    monitor = EnvelopeMonitor()
    writer = H264Writer(out_dir / "annotated.mp4", fps / stride, (width, height)) if write_video else None
    video_ok = False

    stage_ms: dict[str, list[float]] = {"monitor": [], "detect": [], "track": [], "count": [], "total": []}
    totals: Counter = Counter()
    recent_labels: deque[str] = deque(maxlen=int(fps / stride * 2))
    processed = 0
    offset_frames = 0
    next_snapshot = 0.0
    started = time.perf_counter()
    f_frames = (out_dir / "frames.csv").open("w", newline="", encoding="utf-8")
    f_dets = (out_dir / "detections.csv").open("w", newline="", encoding="utf-8")
    f_events = (out_dir / "events.csv").open("w", newline="", encoding="utf-8")
    w_frames = csv.writer(f_frames)
    w_dets = csv.writer(f_dets)
    w_events = csv.writer(f_events)
    w_frames.writerow(
        [
            "frame",
            "t_s",
            "clip",
            "detections",
            "tracks",
            "luma",
            "contrast",
            "sharpness",
            "occluded_fraction",
            "condition",
            "reliability",
            "monitor_ms",
            "detect_ms",
            "track_ms",
            "count_ms",
            "total_ms",
        ]
    )
    w_dets.writerow(["frame", "t_s", "class", "confidence", "x1", "y1", "x2", "y2"])
    w_events.writerow(["track_id", "line", "direction", "class", "frame", "t_s", "reliability"])
    try:
        for clip, meta in zip(clips, reports, strict=True):
            for frame in VideoFileSource(clip, stride=stride):
                if max_frames and processed >= max_frames:
                    break
                global_index = offset_frames + frame.index
                t_s = global_index / fps
                t0 = time.perf_counter_ns()
                image = (
                    cv2.resize(frame.image, (width, height), interpolation=cv2.INTER_AREA)
                    if scale < 1.0
                    else frame.image
                )
                dets = detector.detect([image], [global_index])[0]
                t1 = time.perf_counter_ns()
                condition = monitor.assess(image, [d.bbox for d in dets])
                t2 = time.perf_counter_ns()
                tracked = tracker.update(global_index, dets)
                t3 = time.perf_counter_ns()
                events = []
                if counter is not None:
                    events = counter.update(
                        tracker.histories, {o.track_id for o in tracked}, fps, condition.reliability
                    )
                    for e in events:
                        totals[(e.line, e.direction)] += 1
                        w_events.writerow(
                            [
                                e.track_id,
                                e.line,
                                e.direction,
                                e.vehicle_class.value,
                                e.frame_index,
                                round(e.timestamp_s, 3),
                                e.reliability.value,
                            ]
                        )
                t4 = time.perf_counter_ns()
                recent_labels.append("|".join(condition.labels) or "nominal")
                cond_text = f"condition: {Counter(recent_labels).most_common(1)[0][0]}"
                if writer is not None or t_s >= next_snapshot:
                    canvas = _draw_overlay(
                        image,
                        tracked,
                        tracker.histories,
                        geometry,
                        totals,
                        cond_text,
                        condition.reliability is not ReliabilityFlag.OK,
                    )
                    if writer is not None:
                        writer.write(canvas)
                    if t_s >= next_snapshot:
                        cv2.imwrite(
                            str(out_dir / "snapshots" / f"t{int(t_s):05d}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 85]
                        )
                        next_snapshot += snapshot_every_s
                t5 = time.perf_counter_ns()
                for name, (a, b) in {
                    "detect": (t0, t1),
                    "monitor": (t1, t2),
                    "track": (t2, t3),
                    "count": (t3, t4),
                    "total": (t0, t5),
                }.items():
                    stage_ms[name].append((b - a) / 1e6)
                for d in dets:
                    w_dets.writerow(
                        [
                            global_index,
                            round(t_s, 3),
                            d.vehicle_class.value,
                            round(d.confidence, 4),
                            round(d.bbox.x1, 1),
                            round(d.bbox.y1, 1),
                            round(d.bbox.x2, 1),
                            round(d.bbox.y2, 1),
                        ]
                    )
                row = condition.as_row()
                w_frames.writerow(
                    [
                        global_index,
                        round(t_s, 3),
                        clip.stem,
                        len(dets),
                        len(tracked),
                        round(row["luma"], 1),
                        round(row["contrast"], 1),
                        round(row["sharpness"], 1),
                        round(row["occluded_fraction"], 3),
                        row["labels"],
                        row["reliability"],
                        *(round(stage_ms[k][-1], 2) for k in ("monitor", "detect", "track", "count", "total")),
                    ]
                )
                processed += 1
                if progress and processed % 25 == 0:
                    progress(processed, total_frames, f"{clip.name}: {sum(totals.values())} vehicles counted")
            offset_frames += meta.frame_count
            if max_frames and processed >= max_frames:
                break
    finally:
        f_frames.close()
        f_dets.close()
        f_events.close()
        if writer is not None:
            video_ok = writer.release()
    wall = time.perf_counter() - started
    duration_s = offset_frames / fps if not max_frames else processed * stride / fps

    # tracks
    with (
        (out_dir / "tracks.csv").open("w", newline="", encoding="utf-8") as fh,
        (out_dir / "track_points.csv").open("w", newline="", encoding="utf-8") as fp,
    ):
        wt, wp = csv.writer(fh), csv.writer(fp)
        wt.writerow(
            [
                "track_id",
                "class",
                "first_frame",
                "last_frame",
                "first_t_s",
                "last_t_s",
                "points",
                "dx",
                "dy",
                "distance_px",
                "heading_deg",
            ]
        )
        wp.writerow(["track_id", "frame", "x", "y"])
        for h in tracker.histories.values():
            (_, x0, y0), (_, x1, y1) = h.points[0], h.points[-1]
            dx, dy = x1 - x0, y1 - y0
            wt.writerow(
                [
                    h.track_id,
                    h.vehicle_class.value,
                    h.first_frame,
                    h.last_frame,
                    round(h.first_frame / fps, 2),
                    round(h.last_frame / fps, 2),
                    len(h.points),
                    round(dx, 1),
                    round(dy, 1),
                    round(math.hypot(dx, dy), 1),
                    round(math.degrees(math.atan2(-dy, dx)) % 360, 1),
                ]
            )
            for f, x, y in h.points[:: max(1, len(h.points) // 40)]:
                wp.writerow([h.track_id, f, round(x, 1), round(y, 1)])

    # intervals
    if counter is not None:
        intervals = aggregate_intervals(counter.events, camera_id, config.counting.interval_seconds, duration_s)
        with (out_dir / "intervals.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(
                [
                    "camera_id",
                    "interval_start_s",
                    "interval_end_s",
                    "line",
                    "direction",
                    "class",
                    "count",
                    "source",
                    "reliability",
                ]
            )
            for r in intervals:
                line, direction = (r.direction or ":").split(":", 1)
                w.writerow(
                    [
                        r.camera_id,
                        r.interval_start_s,
                        r.interval_end_s,
                        line,
                        direction,
                        r.vehicle_class.value,
                        r.count,
                        r.source.value,
                        r.reliability.value,
                    ]
                )

    warm = {k: v[1:] if len(v) > 1 else v for k, v in stage_ms.items()}
    results = {
        "camera_id": camera_id,
        "clips": len(clips),
        "frames_processed": processed,
        "stride": stride,
        "source_fps": fps,
        "duration_s": round(duration_s, 1),
        "wall_time_s": round(wall, 1),
        "throughput_fps": round(processed / wall, 2) if wall else None,
        "tracks": len(tracker.histories),
        "count_events": len(counter.events) if counter else 0,
        "counts_by_line_direction": {f"{k[0]}:{k[1]}": v for k, v in sorted(totals.items())},
        "counts_by_class": dict(Counter(e.vehicle_class.value for e in (counter.events if counter else []))),
        "degraded_frame_share": None,
        "latency_ms": {k: latency_stats(v).as_dict() for k, v in warm.items() if v},
        "annotated_video_h264": video_ok,
        "envelope_thresholds": monitor.t.__dict__,
        "geometry": None
        if geometry is None
        else {
            "camera_id": geometry.camera_id,
            "lines": [
                {"name": ln.name, "p1": ln.p1, "p2": ln.p2, "positive": ln.positive, "negative": ln.negative}
                for ln in geometry.lines
            ],
        },
        "processing_scale": round(scale, 4),
        "processed_size": [width, height],
        "validation_status": "NOT VALIDATED: no reference counts exist for these clips yet.",
    }
    with (out_dir / "frames.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if rows:
        results["degraded_frame_share"] = round(sum(r["reliability"] != "ok" for r in rows) / len(rows), 4)
        results["condition_share"] = {
            k: round(v / len(rows), 4) for k, v in Counter(r["condition"] for r in rows).items()
        }
    inputs = [
        {"clip": c.name, "sha256": sha256_file(c), "fps": m.fps, "frames": m.frame_count}
        for c, m in zip(clips, reports, strict=True)
    ] + (extra_inputs or [])
    write_run_record(
        out_dir,
        "run",
        config,
        inputs=inputs,
        model=detector.info.__dict__,
        results=results,
        notes=[
            f"run name: {run_name}",
            "Counts are pipeline ESTIMATES. Condition thresholds are provisional (src/monitoring/envelope.py).",
        ],
    )
    return RunSummary(
        out_dir, processed, duration_s, len(counter.events) if counter else 0, len(tracker.histories), wall
    )
