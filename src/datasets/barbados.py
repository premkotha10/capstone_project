"""Zindi Barbados Traffic Analysis Challenge — clip naming, Drive listing and subset selection.

Facts verified 2026-11-23 (DATASET_RESEARCH §2.1):
* 4 fixed cameras (normanniles1..4) at the Norman Niles roundabout, ~1-minute MP4 clips.
* Probed clip: H.264, 960x540, 20 fps, ~59.9 s, ~4.4 MB.
* File-name times span 06:00-17:59, 2025-10-20 .. 2025-10-26. The burned-in camera overlay is 4 h behind the file
  name (checked 06:00:45 -> 02:00:54 and 13:13:45 -> 09:13:44), consistent with file names in UTC and Barbados
  local time AST = UTC-4. Local coverage is therefore 02:00-13:59: night (~02:00-05:50) plus morning and midday.
* Licence: Zindi states CC BY-SA 4.0, but frames carry "(c) IWCP Inc. Re-Use/Re-Streaming Forbidden".
  Treat as local analysis only; never redistribute video (RISK R-20).
* Congestion labels are in Train.csv (requires Zindi sign-in).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.datasets.remote import DriveEntry

DATASET_NAME = "barbados_traffic"
SOURCE_URL = "https://zindi.africa/competitions/barbados-traffic-analysis-challenge"
LICENCE = "CC-BY-SA-4.0"
CAMERAS = ("normanniles1", "normanniles2", "normanniles3", "normanniles4")
DRIVE_FOLDERS = {
    "normanniles1": "1kqn5DNcUGeIOr1Qm_58IQIAv5wCgmZUY",
    "normanniles2": "1Z9U8_cchaZb06r_WiUPVqs0f8G4tK333",
    "normanniles3": "16tUFln0scBAsaWpNDe8o56Bh_8w56I53",
    "normanniles4": "1uQ0baBBIdmSozvc5XrzsIiNbDOSYSW0W",
    "added": "1ltQPsFoqKQTj69GvOdJz0ZzDZg7Zvdme",
}

_CLIP_RE = re.compile(r"^(?P<camera>normanniles[1-4])_(?P<ts>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})\.mp4$")


@dataclass(frozen=True)
class ClipRef:
    camera_id: str
    start_time: datetime  # time encoded in the file name (apparently UTC; local = start_time - 4 h)
    entry: DriveEntry

    @property
    def clip_id(self) -> str:
        return self.entry.name.removesuffix(".mp4")


def parse_clip_name(name: str) -> tuple[str, datetime]:
    match = _CLIP_RE.match(name)
    if match is None:
        raise ValueError(f"not a Barbados clip name: {name!r}")
    return match.group("camera"), datetime.strptime(match.group("ts"), "%Y-%m-%d-%H-%M-%S")


def to_clip_refs(entries: list[DriveEntry]) -> list[ClipRef]:
    """Convert Drive entries to clip references, skipping non-clip files and de-duplicating by name."""
    refs: dict[str, ClipRef] = {}
    for entry in entries:
        try:
            camera, start = parse_clip_name(entry.name)
        except ValueError:
            continue
        refs.setdefault(entry.name, ClipRef(camera, start, entry))
    return sorted(refs.values(), key=lambda r: (r.camera_id, r.start_time))


@dataclass(frozen=True)
class Window:
    """A contiguous time window to download for every selected camera."""

    name: str
    start: datetime
    minutes: int
    condition_hint: str  # e.g. "dawn", "midday", "dusk" — a hint from clock time, NOT a verified label

    @property
    def end(self) -> datetime:
        return self.start + timedelta(minutes=self.minutes)


def select_clips(refs: list[ClipRef], windows: list[Window], cameras: tuple[str, ...]) -> list[tuple[ClipRef, Window]]:
    """Return clips whose start time lies in [window.start, window.end) for the selected cameras."""
    selected: list[tuple[ClipRef, Window]] = []
    for window in windows:
        for ref in refs:
            if ref.camera_id in cameras and window.start <= ref.start_time < window.end:
                selected.append((ref, window))
    return selected
