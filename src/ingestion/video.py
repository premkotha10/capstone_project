"""Video probing, validation and frame sources (FR-1, NFR-3).

`probe_video` extracts metadata with OpenCV and cross-checks it with `ffprobe` when available.
`validate_video` applies machine-checkable quality gates and returns a structured report rather than raising for
soft issues, so the caller can decide whether to reject, flag or proceed.
`VideoFileSource` / `ImageSequenceSource` yield `Frame`s with a common interface (UA-DETRAC is a JPEG sequence,
Barbados is MP4).
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.common.types import VideoMeta

logger = logging.getLogger(__name__)

SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".avi")
SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


class IngestionError(Exception):
    """Base class for ingestion failures."""


class UnsupportedVideoError(IngestionError):
    pass


class CorruptVideoError(IngestionError):
    pass


@dataclass(frozen=True)
class Frame:
    index: int  # 0-based index in the source
    timestamp_s: float
    image: np.ndarray = field(repr=False)  # BGR uint8, HxWx3


@dataclass(frozen=True)
class ValidationRules:
    min_width: int = 320
    min_height: int = 240
    min_fps: float = 1.0
    max_fps: float = 240.0
    min_duration_s: float = 0.5
    duration_tolerance_s: float = 1.0  # OpenCV vs ffprobe duration disagreement allowed
    sample_frames: int = 10  # frames decoded across the file for the corruption scan


@dataclass
class ValidationReport:
    meta: VideoMeta | None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _ffprobe(path: Path) -> dict | None:
    exe = shutil.which("ffprobe")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [
                exe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,r_frame_rate,nb_frames:format=duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        ).stdout
        return json.loads(out)
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError) as exc:
        logger.warning("ffprobe failed for %s: %s", path, exc)
        return None


def probe_video(path: str | Path, video_id: str | None = None, camera_id: str | None = None) -> VideoMeta:
    """Read container metadata. Raises UnsupportedVideoError / CorruptVideoError on hard failures."""
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        raise UnsupportedVideoError(f"{path.name}: extension {path.suffix!r} not in {SUPPORTED_VIDEO_EXTENSIONS}")
    if not path.is_file():
        raise IngestionError(f"{path}: file not found")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise CorruptVideoError(f"{path.name}: OpenCV cannot open the file")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    finally:
        capture.release()
    if width <= 0 or height <= 0 or fps <= 0:
        raise CorruptVideoError(f"{path.name}: invalid stream properties ({width}x{height} @ {fps} fps)")
    codec = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip("\x00 ") or None
    probe = _ffprobe(path)
    if probe and probe.get("streams"):
        codec = probe["streams"][0].get("codec_name", codec)
    return VideoMeta(
        video_id=video_id or path.stem,
        path=str(path),
        width=width,
        height=height,
        fps=fps,
        frame_count=max(frame_count, 0),
        duration_s=frame_count / fps if frame_count > 0 else 0.0,
        codec=codec,
        camera_id=camera_id,
    )


def validate_video(path: str | Path, rules: ValidationRules | None = None, **probe_kwargs: str) -> ValidationReport:
    """Probe and apply quality gates; decode sampled frames to detect corruption."""
    rules = rules or ValidationRules()
    try:
        meta = probe_video(path, **probe_kwargs)
    except IngestionError as exc:
        return ValidationReport(meta=None, errors=[str(exc)])
    report = ValidationReport(meta=meta)
    if meta.width < rules.min_width or meta.height < rules.min_height:
        report.errors.append(f"resolution {meta.width}x{meta.height} below {rules.min_width}x{rules.min_height}")
    if not rules.min_fps <= meta.fps <= rules.max_fps:
        report.errors.append(f"fps {meta.fps} outside [{rules.min_fps}, {rules.max_fps}]")
    if meta.frame_count <= 0:
        report.errors.append("container reports no frames")
    elif meta.duration_s < rules.min_duration_s:
        report.errors.append(f"duration {meta.duration_s:.2f}s below {rules.min_duration_s}s")

    probe = _ffprobe(Path(path))
    if probe is None:
        report.warnings.append("ffprobe unavailable: duration cross-check skipped")
    else:
        try:
            ff_duration = float(probe["format"]["duration"])
            if abs(ff_duration - meta.duration_s) > rules.duration_tolerance_s:
                report.warnings.append(
                    f"duration mismatch: OpenCV {meta.duration_s:.2f}s vs ffprobe {ff_duration:.2f}s"
                )
        except (KeyError, ValueError):
            report.warnings.append("ffprobe returned no duration")

    if meta.frame_count > 0:
        bad = _scan_frames(Path(path), meta.frame_count, rules.sample_frames)
        if bad:
            report.errors.append(f"{len(bad)} of {rules.sample_frames} sampled frames failed to decode: {bad[:5]}")
    return report


def _scan_frames(path: Path, frame_count: int, samples: int) -> list[int]:
    indices = sorted({int(i) for i in np.linspace(0, max(frame_count - 2, 0), num=max(samples, 1))})
    capture = cv2.VideoCapture(str(path))
    failed: list[int] = []
    try:
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = capture.read()
            if not ok or image is None or image.size == 0:
                failed.append(index)
    finally:
        capture.release()
    return failed


class VideoFileSource:
    """Iterate frames of a video file with an optional stride."""

    def __init__(self, path: str | Path, stride: int = 1, max_frames: int | None = None) -> None:
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.meta = probe_video(path)
        self.stride = stride
        self.max_frames = max_frames

    def __iter__(self) -> Iterator[Frame]:
        capture = cv2.VideoCapture(self.meta.path)
        if not capture.isOpened():
            raise CorruptVideoError(f"{self.meta.path}: cannot open")
        index = 0
        yielded = 0
        try:
            while True:
                if index % self.stride == 0:
                    ok, image = capture.read()
                    if not ok:
                        break
                    yield Frame(index=index, timestamp_s=index / self.meta.fps, image=image)
                    yielded += 1
                    if self.max_frames is not None and yielded >= self.max_frames:
                        break
                elif not capture.grab():  # skip decoding work for strided-out frames
                    break
                index += 1
        finally:
            capture.release()


class ImageSequenceSource:
    """Iterate a directory of numbered images (e.g. UA-DETRAC img00001.jpg) as frames at a nominal fps."""

    def __init__(self, directory: str | Path, fps: float, stride: int = 1, max_frames: int | None = None) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.directory = Path(directory)
        self.files = sorted(p for p in self.directory.iterdir() if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS)
        if not self.files:
            raise IngestionError(f"{self.directory}: no images found")
        first = cv2.imread(str(self.files[0]))
        if first is None:
            raise CorruptVideoError(f"{self.files[0]}: unreadable image")
        self.fps = fps
        self.stride = stride
        self.max_frames = max_frames
        self.meta = VideoMeta(
            video_id=self.directory.name,
            path=str(self.directory),
            width=first.shape[1],
            height=first.shape[0],
            fps=fps,
            frame_count=len(self.files),
            duration_s=len(self.files) / fps,
            codec="image-sequence",
        )

    def __iter__(self) -> Iterator[Frame]:
        yielded = 0
        for index in range(0, len(self.files), self.stride):
            image = cv2.imread(str(self.files[index]))
            if image is None:
                raise CorruptVideoError(f"{self.files[index]}: unreadable image")
            yield Frame(index=index, timestamp_s=index / self.fps, image=image)
            yielded += 1
            if self.max_frames is not None and yielded >= self.max_frames:
                break
