"""UA-DETRAC annotation parser and class mapping.

Facts (DATASET_RESEARCH §2.3, XML inspected 2026-11-23):
* 100 sequences (60 train / 40 test), 960x540 JPEG frames at 25 fps, fixed elevated roadside cameras.
* Per-sequence attributes: `sence_weather` (sic) in {sunny, cloudy, night, rainy}, `camera_state` {stable, unstable}.
* Per-frame `density`; per-target box, track id, vehicle_type {car, bus, van, others}, truncation ratio, speed,
  orientation, and optional occlusion overlap regions. `ignored_region` boxes mark areas that are not annotated.
* Licence of the original release is unverified; treated as academic/non-commercial use with citation.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from src.common.types import BoundingBox, VehicleClass

DATASET_NAME = "ua_detrac"
SOURCE_URL = "https://sites.google.com/view/daweidu/projects/ua-detrac"
CITATION = (
    "L. Wen et al., UA-DETRAC: A new benchmark and protocol for multi-object detection and tracking, "
    "Computer Vision and Image Understanding, 2020."
)
IMAGES_ARCHIVE_DRIVE_ID = "1fQC6CEWOeL9pnJ4ZdD7U2gromcnWDRgm"  # DETRAC-Images.zip (~9.85 GB)
FPS = 25.0
FRAME_SIZE = (960, 540)

# UA-DETRAC classes -> project taxonomy. "others" covers trucks, tankers etc. (paper) -> mapped to TRUCK.
CLASS_MAP: dict[str, VehicleClass] = {
    "car": VehicleClass.CAR,
    "bus": VehicleClass.BUS,
    "van": VehicleClass.VAN,
    "others": VehicleClass.TRUCK,
}


@dataclass(frozen=True)
class GTObject:
    track_id: int
    bbox: BoundingBox
    vehicle_class: VehicleClass
    source_class: str
    truncation_ratio: float
    occluded: bool  # at least one occlusion overlap region recorded
    occlusion_area: float  # total overlapped area in px^2 (sum of region_overlap boxes)


@dataclass(frozen=True)
class GTFrame:
    frame_number: int  # 1-based as in UA-DETRAC (img00001.jpg)
    density: int
    objects: tuple[GTObject, ...]


@dataclass
class SequenceAnnotation:
    name: str
    weather: str
    camera_state: str
    ignored_regions: list[BoundingBox] = field(default_factory=list)
    frames: list[GTFrame] = field(default_factory=list)

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def track_ids(self) -> set[int]:
        return {obj.track_id for frame in self.frames for obj in frame.objects}

    @property
    def mean_density(self) -> float:
        return sum(f.density for f in self.frames) / len(self.frames) if self.frames else 0.0

    @property
    def occluded_fraction(self) -> float:
        total = sum(len(f.objects) for f in self.frames)
        occluded = sum(o.occluded for f in self.frames for o in f.objects)
        return occluded / total if total else 0.0


def _box(element: ET.Element) -> BoundingBox:
    left, top = float(element.attrib["left"]), float(element.attrib["top"])
    width, height = float(element.attrib["width"]), float(element.attrib["height"])
    return BoundingBox(x1=left, y1=top, x2=left + width, y2=top + height)


def _iter_objects(frame: ET.Element) -> Iterator[GTObject]:
    target_list = frame.find("target_list")
    if target_list is None:
        return
    for target in target_list.findall("target"):
        box_el = target.find("box")
        attr = target.find("attribute")
        if box_el is None or attr is None:
            raise ValueError(f"target {target.attrib.get('id')} lacks box or attribute")
        source_class = attr.attrib["vehicle_type"]
        if source_class not in CLASS_MAP:
            raise ValueError(f"unknown UA-DETRAC vehicle_type {source_class!r}")
        overlaps = target.findall("occlusion/region_overlap")
        area = sum(float(o.attrib["width"]) * float(o.attrib["height"]) for o in overlaps)
        yield GTObject(
            track_id=int(target.attrib["id"]),
            bbox=_box(box_el),
            vehicle_class=CLASS_MAP[source_class],
            source_class=source_class,
            truncation_ratio=float(attr.attrib.get("truncation_ratio", 0.0)),
            occluded=bool(overlaps),
            occlusion_area=area,
        )


def parse_sequence_xml(source: str | Path | bytes) -> SequenceAnnotation:
    """Parse one UA-DETRAC sequence XML (path or raw bytes)."""
    root = ET.fromstring(source) if isinstance(source, bytes) else ET.parse(source).getroot()
    if root.tag != "sequence":
        raise ValueError(f"expected <sequence> root, got <{root.tag}>")
    attrs = root.find("sequence_attribute")
    if attrs is None:
        raise ValueError("missing <sequence_attribute>")
    sequence = SequenceAnnotation(
        name=root.attrib["name"],
        weather=attrs.attrib["sence_weather"],
        camera_state=attrs.attrib.get("camera_state", "unknown"),
        ignored_regions=[_box(b) for b in root.findall("ignored_region/box")],
    )
    for frame in root.findall("frame"):
        sequence.frames.append(
            GTFrame(
                frame_number=int(frame.attrib["num"]),
                density=int(frame.attrib.get("density", 0)),
                objects=tuple(_iter_objects(frame)),
            )
        )
    return sequence


def image_member_name(sequence: str, frame_number: int) -> str:
    """Member path inside DETRAC-Images.zip (layout verified from the archive central directory, 2026-11-23)."""
    return f"DETRAC-Images/{sequence}/img{frame_number:05d}.jpg"


def site_proxy(sequence: str) -> str:
    """Group key used to keep sequences from the same scene in the same partition.

    UA-DETRAC does not publish a sequence->location table in the files we have; consecutive IDs such as
    MVI_40851..MVI_40855 are recordings from one camera position. We therefore group by the first four digits
    (e.g. 'MVI_4085'). This is a documented proxy, not a verified site label.
    """
    if not sequence.startswith("MVI_") or len(sequence) < 8:
        raise ValueError(f"unexpected UA-DETRAC sequence name {sequence!r}")
    return sequence[:8]
