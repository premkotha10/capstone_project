from pathlib import Path

import numpy as np
import pytest

from src.common.config import TrackingConfig
from src.common.types import BoundingBox, Detection, ReliabilityFlag, VehicleClass
from src.counting.line_counter import CameraGeometry, CountingLine, LineCounter, aggregate_intervals
from src.forecasting.baselines import linear_trend, moving_average, persistence, rolling_origin_backtest
from src.monitoring.envelope import EnvelopeMonitor, occluded_fraction
from src.tracking.tracker import TrackHistory, VehicleTracker

LINE = CountingLine("approach", (0.0, 100.0), (200.0, 100.0), positive="exiting", negative="entering")


def _geometry() -> CameraGeometry:
    return CameraGeometry("cam", (200, 200), (LINE,))


def _hist(tid: int, ys: list[float], cls: VehicleClass = VehicleClass.CAR) -> TrackHistory:
    h = TrackHistory(tid, 0, len(ys) - 1, points=[(i, 50.0, y) for i, y in enumerate(ys)])
    h.class_votes[cls] = 1.0
    return h


def _run_counter(histories: list[TrackHistory]) -> LineCounter:
    counter = LineCounter(_geometry(), min_track_length=2)
    longest = max(len(h.points) for h in histories)
    for step in range(2, longest + 1):
        view = {
            h.track_id: TrackHistory(h.track_id, 0, step - 1, points=h.points[:step], class_votes=h.class_votes)
            for h in histories
            if len(h.points) >= step
        }
        counter.update(view, set(view), fps=10.0)
    return counter


def test_crossing_down_is_entering_and_up_is_exiting() -> None:
    counter = _run_counter([_hist(1, [80, 95, 105, 120]), _hist(2, [130, 110, 90], VehicleClass.BUS)])
    by_track = {e.track_id: e for e in counter.events}
    assert by_track[1].direction == "entering"
    assert by_track[2].direction == "exiting" and by_track[2].vehicle_class is VehicleClass.BUS


def test_jitter_on_the_line_counts_once() -> None:
    counter = _run_counter([_hist(1, [95, 105, 95, 105, 95, 105])])
    assert len(counter.events) == 1


def test_track_not_reaching_line_is_not_counted() -> None:
    assert _run_counter([_hist(1, [10, 30, 50, 70, 90])]).events == []


def test_crossing_outside_segment_is_not_counted() -> None:
    h = TrackHistory(1, 0, 2, points=[(0, 300.0, 90.0), (1, 300.0, 110.0)])  # x=300 is beyond the line's end
    h.class_votes[VehicleClass.CAR] = 1
    counter = LineCounter(CameraGeometry("cam", (400, 200), (LINE,)), min_track_length=2)
    counter.update({1: h}, {1}, fps=10)
    assert counter.events == []


def test_geometry_validation(tmp_path: Path) -> None:
    bad = tmp_path / "cam.yaml"
    bad.write_text(
        "camera_id: c\nframe_size: [100, 100]\nlines:\n  - {name: a, p1: [0, 0], p2: [500, 0], "
        "positive: p, negative: n}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        CameraGeometry.load(bad)


def test_camera1_config_loads() -> None:
    geometry = CameraGeometry.load(Path(__file__).resolve().parents[2] / "configs" / "cameras" / "normanniles1.yaml")
    assert {line.name for line in geometry.lines} == {"approach", "circulating_west"}


def test_aggregate_intervals_includes_zeros_and_degraded_flag() -> None:
    counter = _run_counter([_hist(1, [80, 95, 105])])
    events = counter.events + [
        type(counter.events[0])(9, "approach", "entering", VehicleClass.CAR, 900, 90.0, ReliabilityFlag.DEGRADED)
    ]
    rows = aggregate_intervals(events, "cam", 60.0, 180.0)
    counts = [r.count for r in rows]
    assert counts == [1, 1, 0]
    assert [r.reliability for r in rows] == [ReliabilityFlag.OK, ReliabilityFlag.DEGRADED, ReliabilityFlag.OK]
    assert all(r.source.value == "estimated" for r in rows)


def test_tracker_keeps_identity_for_moving_box() -> None:
    tracker = VehicleTracker(TrackingConfig(), frame_rate=10)
    ids = set()
    for f in range(12):
        box = BoundingBox(x1=100 + 5 * f, y1=100, x2=160 + 5 * f, y2=140)
        tracked = tracker.update(
            f, [Detection(frame_index=f, bbox=box, confidence=0.9, vehicle_class=VehicleClass.CAR, source_class="car")]
        )
        ids.update(t.track_id for t in tracked)
    assert len(ids) == 1
    hist = next(iter(tracker.histories.values()))
    assert hist.vehicle_class is VehicleClass.CAR and hist.length >= 10


def test_tracker_handles_empty_frames() -> None:
    tracker = VehicleTracker(TrackingConfig(), frame_rate=10)
    assert tracker.update(0, []) == []


def test_envelope_flags_dark_frame_and_dense_overlap() -> None:
    monitor = EnvelopeMonitor()
    dark = np.full((540, 960, 3), 20, dtype=np.uint8)
    assert "night" in monitor.assess(dark, []).labels
    assert monitor.assess(dark, []).reliability is ReliabilityFlag.DEGRADED
    rng = np.random.default_rng(0)
    bright = rng.integers(60, 255, (540, 960, 3), dtype=np.uint8)
    stacked = [BoundingBox(x1=10 + i, y1=10, x2=60 + i, y2=50) for i in range(10)]
    cond = monitor.assess(bright, stacked)
    assert "dense_occlusion" in cond.labels and cond.occluded_fraction == 1.0
    assert occluded_fraction([BoundingBox(x1=0, y1=0, x2=10, y2=10)], 0.3) == 0.0


def test_forecast_baselines() -> None:
    hist = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert persistence(hist, 1) == 5.0
    assert moving_average(3)(hist, 1) == 4.0
    assert linear_trend(5)(hist, 2) == pytest.approx(7.0)
    results = rolling_origin_backtest([1, 2, 3, 4, 5, 6, 7, 8], horizons=(1,), min_history=3)
    by_model = {r.model: r for r in results}
    assert by_model["linear_trend_5"].mae == pytest.approx(0.0, abs=1e-9)
    assert by_model["persistence"].mae == pytest.approx(1.0)
