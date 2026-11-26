import io
import zipfile
from datetime import date, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.common.types import VehicleClass
from src.datasets import barbados, ua_detrac
from src.datasets.manifest import DatasetManifest, ManifestItem, sha256_directory, verify_items
from src.datasets.remote import DriveEntry, RangeFile, parse_drive_folder_html

# ---------- remote ----------


def test_range_file_lets_zipfile_read_selected_members() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for i in range(50):
            zf.writestr(f"DETRAC-Images/MVI_{i:05d}/img00001.jpg", bytes([i]) * 5000)
    blob = buffer.getvalue()
    calls: list[tuple[int, int]] = []

    def fetch(start: int, end: int) -> bytes:
        calls.append((start, end))
        return blob[start : end + 1]

    remote = RangeFile(fetch, len(blob), block_size=4096)
    with zipfile.ZipFile(remote) as zf:
        assert zf.read("DETRAC-Images/MVI_00007/img00001.jpg") == bytes([7]) * 5000
    assert remote.bytes_fetched < len(blob)  # only the needed parts were transferred
    assert calls


def test_range_file_seek_semantics() -> None:
    data = bytes(range(256))
    rf = RangeFile(lambda s, e: data[s : e + 1], len(data), block_size=16)
    rf.seek(-6, io.SEEK_END)
    assert rf.read() == data[-6:]
    rf.seek(10)
    assert rf.read(3) == data[10:13]
    assert rf.read(0) == b""
    with pytest.raises(ValueError):
        rf.seek(-1)


def test_parse_drive_folder_html() -> None:
    page = (
        '<div id="entry-AbC_1-x"><div class="flip-entry-title">normanniles1_2025-10-20-06-00-45.mp4</div></div>'
        '<div id="entry-ZZ9"><div class="flip-entry-title">notes &amp; readme.txt</div></div>'
    )
    entries = parse_drive_folder_html(page)
    assert entries == [
        DriveEntry("AbC_1-x", "normanniles1_2025-10-20-06-00-45.mp4"),
        DriveEntry("ZZ9", "notes & readme.txt"),
    ]
    assert "id=AbC_1-x" in entries[0].download_url


# ---------- barbados ----------


def test_parse_clip_name() -> None:
    camera, start = barbados.parse_clip_name("normanniles3_2025-10-21-17-37-45.mp4")
    assert camera == "normanniles3"
    assert start == datetime(2025, 10, 21, 17, 37, 45)
    with pytest.raises(ValueError):
        barbados.parse_clip_name("normanniles5_2025-10-21-17-37-45.mp4")


def test_select_clips_window_and_dedup() -> None:
    entries = [DriveEntry(f"id{m}", f"normanniles1_2025-10-21-06-{m:02d}-45.mp4") for m in range(0, 30)] + [
        DriveEntry("dup", "normanniles1_2025-10-21-06-05-45.mp4"),
        DriveEntry("x", "readme.txt"),
    ]
    refs = barbados.to_clip_refs(entries)
    assert len(refs) == 30
    window = barbados.Window("w", datetime(2025, 10, 21, 6, 0), 23, "dawn")
    selected = barbados.select_clips(refs, [window], ("normanniles1",))
    assert len(selected) == 23
    assert barbados.select_clips(refs, [window], ("normanniles2",)) == []


# ---------- ua-detrac ----------

XML = b"""<?xml version="1.0" encoding="utf-8"?>
<sequence name="MVI_40742">
  <sequence_attribute camera_state="stable" sence_weather="night"/>
  <ignored_region><box left="0" top="0" width="50" height="40"/></ignored_region>
  <frame density="2" num="1"><target_list>
    <target id="1"><box left="100" top="100" width="60" height="40"/>
      <attribute orientation="1" speed="2" trajectory_length="5" truncation_ratio="0.1" vehicle_type="car"/></target>
    <target id="2"><box left="300" top="120" width="120" height="80"/>
      <attribute orientation="1" speed="2" trajectory_length="5" truncation_ratio="0" vehicle_type="others"/>
      <occlusion><region_overlap left="300" top="120" width="10" height="5" occlusion_id="1" occlusion_status="1"/></occlusion>
    </target>
  </target_list></frame>
  <frame density="1" num="2"><target_list>
    <target id="1"><box left="104" top="100" width="60" height="40"/>
      <attribute orientation="1" speed="2" trajectory_length="6" truncation_ratio="0" vehicle_type="bus"/></target>
  </target_list></frame>
</sequence>"""


def test_parse_sequence_xml() -> None:
    seq = ua_detrac.parse_sequence_xml(XML)
    assert (seq.name, seq.weather, seq.camera_state) == ("MVI_40742", "night", "stable")
    assert seq.num_frames == 2
    assert seq.track_ids == {1, 2}
    assert seq.ignored_regions[0].x2 == 50
    obj = seq.frames[0].objects[1]
    assert obj.vehicle_class is VehicleClass.TRUCK and obj.source_class == "others"
    assert obj.occluded and obj.occlusion_area == 50
    assert seq.occluded_fraction == pytest.approx(1 / 3)
    assert seq.mean_density == 1.5


def test_unknown_vehicle_type_rejected() -> None:
    with pytest.raises(ValueError):
        ua_detrac.parse_sequence_xml(XML.replace(b'vehicle_type="bus"', b'vehicle_type="tram"'))


def test_member_name_and_site_proxy() -> None:
    assert ua_detrac.image_member_name("MVI_40742", 7) == "DETRAC-Images/MVI_40742/img00007.jpg"
    assert ua_detrac.site_proxy("MVI_40742") == "MVI_4074"


# ---------- manifest ----------


def _manifest(items: list[ManifestItem]) -> DatasetManifest:
    return DatasetManifest(
        dataset="t",
        version="1",
        source_url="https://example.org",
        access_date=date(2026, 11, 23),
        licence="CC-BY-4.0",
        licence_verified=True,
        custodian="me",
        items=items,
    )


def test_manifest_roundtrip_and_verify(tmp_path: Path) -> None:
    (tmp_path / "seq").mkdir()
    (tmp_path / "seq" / "a.jpg").write_bytes(b"abc")
    (tmp_path / "f.bin").write_bytes(b"hello")
    digest, size = sha256_directory(tmp_path / "seq")
    from src.common.provenance import sha256_file

    m = _manifest(
        [
            ManifestItem(id="seq", file="seq", sha256=digest, size_bytes=size),
            ManifestItem(id="f", file="f.bin", sha256=sha256_file(tmp_path / "f.bin"), size_bytes=5),
        ]
    )
    m.save(tmp_path / "m.yaml")
    loaded = DatasetManifest.load(tmp_path / "m.yaml")
    assert loaded == m
    assert verify_items(loaded, tmp_path, recompute_hash=True) == []
    (tmp_path / "f.bin").write_bytes(b"HELLO")
    assert verify_items(loaded, tmp_path, recompute_hash=True) == ["f: sha256 mismatch"]
    with pytest.raises(KeyError):
        loaded.item("missing")


def test_manifest_rejects_duplicates_and_bad_hash() -> None:
    item = ManifestItem(id="a", file="a", sha256="0" * 64, size_bytes=1)
    with pytest.raises(ValidationError):
        _manifest([item, item])
    with pytest.raises(ValidationError):
        ManifestItem(id="a", file="a", sha256="xyz", size_bytes=1)
