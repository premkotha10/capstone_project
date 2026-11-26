import pytest
from pydantic import ValidationError

from src.common.types import (
    BoundingBox,
    CountSource,
    IntervalCount,
    ReliabilityFlag,
    Track,
    VehicleClass,
)


def test_bbox_geometry() -> None:
    box = BoundingBox(x1=10, y1=20, x2=30, y2=60)
    assert (box.width, box.height, box.area) == (20, 40, 800)
    assert box.bottom_center == (20.0, 60)


@pytest.mark.parametrize("coords", [(10, 10, 10, 20), (10, 10, 5, 20), (0, 30, 10, 20)])
def test_degenerate_bbox_rejected(coords: tuple[float, float, float, float]) -> None:
    x1, y1, x2, y2 = coords
    with pytest.raises(ValidationError):
        BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _count(**overrides: object) -> IntervalCount:
    fields: dict[str, object] = dict(
        camera_id="cam1",
        interval_start_s=0,
        interval_end_s=60,
        vehicle_class=VehicleClass.CAR,
        count=5,
        source=CountSource.ESTIMATED,
    )
    fields.update(overrides)
    return IntervalCount(**fields)  # type: ignore[arg-type]


def test_count_source_is_required() -> None:
    with pytest.raises(ValidationError):
        IntervalCount(camera_id="c", interval_start_s=0, interval_end_s=1, vehicle_class="car", count=1)  # type: ignore[call-arg]


def test_demo_and_abstained_counts_are_not_evidence() -> None:
    assert _count().is_valid_evidence
    assert not _count(source=CountSource.DEMO).is_valid_evidence
    assert not _count(reliability=ReliabilityFlag.ABSTAIN).is_valid_evidence
    assert _count(reliability=ReliabilityFlag.DEGRADED).is_valid_evidence  # usable but must be labelled


def test_invalid_interval_and_negative_count_rejected() -> None:
    with pytest.raises(ValidationError):
        _count(interval_end_s=0)
    with pytest.raises(ValidationError):
        _count(count=-1)


def test_track_requires_aligned_increasing_frames() -> None:
    box = BoundingBox(x1=0, y1=0, x2=1, y2=1)
    Track(track_id=1, vehicle_class=VehicleClass.BUS, frame_indices=(1, 2), boxes=(box, box))
    with pytest.raises(ValidationError):
        Track(track_id=1, vehicle_class=VehicleClass.BUS, frame_indices=(1,), boxes=(box, box))
    with pytest.raises(ValidationError):
        Track(track_id=1, vehicle_class=VehicleClass.BUS, frame_indices=(2, 1), boxes=(box, box))


def test_types_are_immutable() -> None:
    c = _count()
    with pytest.raises(ValidationError):
        c.count = 99  # type: ignore[misc]
