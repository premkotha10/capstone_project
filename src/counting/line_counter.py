"""Virtual-line counting with direction and duplicate prevention (Module 5, O2-A image-space variant).

A track is counted on a line when the segment between two consecutive bottom-centre positions intersects the
line segment. Each (track, line) pair is counted at most once, so jitter around the line or a track lingering on it
cannot double-count. The class reported for a count is the track's majority class at the moment of crossing.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.common.types import CountSource, IntervalCount, ReliabilityFlag, VehicleClass
from src.tracking.tracker import TrackHistory


@dataclass(frozen=True)
class CountingLine:
    name: str
    p1: tuple[float, float]
    p2: tuple[float, float]
    positive: str
    negative: str

    def side(self, x: float, y: float) -> float:
        """Signed area: >0 on one side, <0 on the other, 0 on the line."""
        (x1, y1), (x2, y2) = self.p1, self.p2
        return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


@dataclass(frozen=True)
class CameraGeometry:
    camera_id: str
    frame_size: tuple[int, int]
    lines: tuple[CountingLine, ...]

    def scaled(self, factor: float) -> CameraGeometry:
        """The same lines in a frame resized by `factor` (used when frames are downscaled for speed)."""
        w, h = self.frame_size
        return CameraGeometry(
            self.camera_id,
            (int(w * factor), int(h * factor)),
            tuple(
                CountingLine(
                    ln.name,
                    (ln.p1[0] * factor, ln.p1[1] * factor),
                    (ln.p2[0] * factor, ln.p2[1] * factor),
                    ln.positive,
                    ln.negative,
                )
                for ln in self.lines
            ),
        )

    @classmethod
    def custom_line(
        cls, frame_size: tuple[int, int], orientation: str, position: float, start: float = 0.02, end: float = 0.98
    ) -> CameraGeometry:
        """One line across the frame, placed by fractions (for uploaded videos without a camera config).

        orientation 'horizontal': a line at height `position` (0 = top); directions 'up' / 'down'.
        orientation 'vertical': a line at width `position` (0 = left); directions 'left' / 'right'.
        """
        if not 0.0 < position < 1.0 or not 0.0 <= start < end <= 1.0:
            raise ValueError("position must be in (0, 1) and 0 <= start < end <= 1")
        w, h = frame_size
        if orientation == "horizontal":
            line = CountingLine("line", (start * w, position * h), (end * w, position * h), "up", "down")
        elif orientation == "vertical":
            line = CountingLine("line", (position * w, start * h), (position * w, end * h), "right", "left")
        else:
            raise ValueError("orientation must be 'horizontal' or 'vertical'")
        return cls("custom", (w, h), (line,))

    @classmethod
    def load(cls, path: str | Path) -> CameraGeometry:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        width, height = data["frame_size"]
        lines = []
        for spec in data["lines"]:
            line = CountingLine(spec["name"], tuple(spec["p1"]), tuple(spec["p2"]), spec["positive"], spec["negative"])
            for x, y in (line.p1, line.p2):
                if not (0 <= x <= width and 0 <= y <= height):
                    raise ValueError(f"{path}: line {line.name} point ({x},{y}) outside {width}x{height}")
            if line.p1 == line.p2:
                raise ValueError(f"{path}: line {line.name} has zero length")
            lines.append(line)
        if len({ln.name for ln in lines}) != len(lines):
            raise ValueError(f"{path}: duplicate line names")
        return cls(data["camera_id"], (width, height), tuple(lines))


def _segments_intersect(a: tuple[float, float], b: tuple[float, float], line: CountingLine) -> bool:
    def orient(p, q, r) -> float:  # noqa: ANN001
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    c, d = line.p1, line.p2
    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    return (o1 * o2 < 0) and (o3 * o4 < 0)


@dataclass(frozen=True)
class CountEvent:
    track_id: int
    line: str
    direction: str
    vehicle_class: VehicleClass
    frame_index: int
    timestamp_s: float
    reliability: ReliabilityFlag


class LineCounter:
    def __init__(self, geometry: CameraGeometry, min_track_length: int = 3) -> None:
        self.geometry = geometry
        self.min_track_length = min_track_length
        self._counted: set[tuple[int, str]] = set()
        self.events: list[CountEvent] = []

    def update(
        self,
        histories: dict[int, TrackHistory],
        active_ids: set[int],
        fps: float,
        reliability: ReliabilityFlag = ReliabilityFlag.OK,
    ) -> list[CountEvent]:
        new: list[CountEvent] = []
        for tid in active_ids:
            hist = histories[tid]
            if hist.length < max(self.min_track_length, 2):
                continue
            (_, xa, ya), (frame, xb, yb) = hist.points[-2], hist.points[-1]
            for line in self.geometry.lines:
                key = (tid, line.name)
                if key in self._counted or not _segments_intersect((xa, ya), (xb, yb), line):
                    continue
                direction = line.positive if line.side(xb, yb) < line.side(xa, ya) else line.negative
                self._counted.add(key)
                event = CountEvent(tid, line.name, direction, hist.vehicle_class, frame, frame / fps, reliability)
                new.append(event)
        self.events.extend(new)
        return new

    def totals(self) -> Counter[tuple[str, str, VehicleClass]]:
        return Counter((e.line, e.direction, e.vehicle_class) for e in self.events)


def aggregate_intervals(
    events: list[CountEvent], camera_id: str, interval_s: float, duration_s: float, offset_s: float = 0.0
) -> list[IntervalCount]:
    """Per-interval counts for every (line, direction, class) seen, including zero intervals.

    An interval is DEGRADED if any event in it was produced under a degraded reliability flag.
    """
    keys = sorted({(e.line, e.direction, e.vehicle_class) for e in events}, key=str)
    n = max(int(-(-duration_s // interval_s)), 1)
    counts = Counter((int(e.timestamp_s // interval_s), e.line, e.direction, e.vehicle_class) for e in events)
    degraded = {int(e.timestamp_s // interval_s) for e in events if e.reliability is not ReliabilityFlag.OK}
    rows: list[IntervalCount] = []
    for i in range(n):
        for line, direction, cls in keys:
            rows.append(
                IntervalCount(
                    camera_id=camera_id,
                    interval_start_s=offset_s + i * interval_s,
                    interval_end_s=offset_s + min((i + 1) * interval_s, max(duration_s, (i + 1) * interval_s)),
                    vehicle_class=cls,
                    direction=f"{line}:{direction}",
                    count=counts.get((i, line, direction, cls), 0),
                    source=CountSource.ESTIMATED,
                    reliability=ReliabilityFlag.DEGRADED if i in degraded else ReliabilityFlag.OK,
                )
            )
    return rows
