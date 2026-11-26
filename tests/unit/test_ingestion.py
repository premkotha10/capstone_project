from pathlib import Path

import cv2
import numpy as np
import pytest

from src.ingestion.video import (
    ImageSequenceSource,
    IngestionError,
    UnsupportedVideoError,
    ValidationRules,
    VideoFileSource,
    probe_video,
    validate_video,
)


def _write_video(path: Path, frames: int = 30, fps: float = 10.0, size: tuple[int, int] = (640, 360)) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert writer.isOpened()
    for i in range(frames):
        image = np.full((size[1], size[0], 3), i * 7 % 255, dtype=np.uint8)
        cv2.putText(image, str(i), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        writer.write(image)
    writer.release()
    return path


@pytest.fixture()
def video(tmp_path: Path) -> Path:
    return _write_video(tmp_path / "clip.mp4")


def test_probe_reads_metadata(video: Path) -> None:
    meta = probe_video(video, camera_id="cam1")
    assert (meta.width, meta.height) == (640, 360)
    assert meta.fps == pytest.approx(10.0)
    assert meta.frame_count == 30
    assert meta.duration_s == pytest.approx(3.0)
    assert meta.camera_id == "cam1"


def test_valid_video_passes(video: Path) -> None:
    report = validate_video(video)
    assert report.ok, report.errors


def test_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "clip.mkv"
    path.write_bytes(b"x")
    with pytest.raises(UnsupportedVideoError):
        probe_video(path)
    assert not validate_video(path).ok


def test_random_bytes_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "garbage.mp4"
    path.write_bytes(np.random.default_rng(0).integers(0, 255, 50_000, dtype=np.uint8).tobytes())
    report = validate_video(path)
    assert not report.ok


def test_truncated_video_is_rejected(tmp_path: Path) -> None:
    good = _write_video(tmp_path / "full.mp4", frames=60)
    truncated = tmp_path / "truncated.mp4"
    truncated.write_bytes(good.read_bytes()[: good.stat().st_size // 3])
    assert not validate_video(truncated).ok


def test_low_resolution_rejected(tmp_path: Path) -> None:
    path = _write_video(tmp_path / "small.mp4", size=(160, 120))
    report = validate_video(path, ValidationRules(min_width=320, min_height=240))
    assert any("resolution" in e for e in report.errors)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(IngestionError):
        probe_video(tmp_path / "nope.mp4")


def test_video_source_stride_and_timestamps(video: Path) -> None:
    frames = list(VideoFileSource(video, stride=5))
    assert [f.index for f in frames] == [0, 5, 10, 15, 20, 25]
    assert frames[1].timestamp_s == pytest.approx(0.5)
    assert frames[0].image.shape == (360, 640, 3)


def test_video_source_max_frames(video: Path) -> None:
    assert len(list(VideoFileSource(video, max_frames=4))) == 4


def test_image_sequence_source(tmp_path: Path) -> None:
    for i in range(1, 8):
        cv2.imwrite(str(tmp_path / f"img{i:05d}.jpg"), np.full((540, 960, 3), i, dtype=np.uint8))
    source = ImageSequenceSource(tmp_path, fps=25.0, stride=3)
    frames = list(source)
    assert [f.index for f in frames] == [0, 3, 6]
    assert source.meta.frame_count == 7
    assert frames[2].timestamp_s == pytest.approx(6 / 25.0)


def test_image_sequence_empty_dir(tmp_path: Path) -> None:
    with pytest.raises(IngestionError):
        ImageSequenceSource(tmp_path, fps=25.0)
