"""Download a configured subset of Barbados clips from the public Google Drive folders and register them.

Usage:  python scripts/download_barbados_subset.py configs/data/barbados_subset_v1.yaml [--dry-run]

Clips land in DATA_ROOT/raw/barbados_traffic/<camera>/<clip>.mp4 and are recorded in
data/manifests/barbados_traffic.yaml with SHA-256, size, Drive file ID and window/condition hint.
Already-registered files with a matching size are skipped, so the script is resumable.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import PROJECT_ROOT, Settings  # noqa: E402
from src.common.logging_setup import configure_logging  # noqa: E402
from src.datasets import barbados  # noqa: E402
from src.datasets.manifest import DatasetManifest, ManifestItem  # noqa: E402
from src.datasets.remote import download_file, list_drive_folder  # noqa: E402

logger = logging.getLogger("download_barbados")
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / f"{barbados.DATASET_NAME}.yaml"


NOTES = [
    "File-name times 06:00-17:59 (2025-10-20..26) appear to be UTC: the burned-in overlay (local AST = UTC-4) "
    "reads 4 h earlier, i.e. 02:00-13:59 local, which includes night.",
    "Video frames carry the burned-in notice '(c) IWCP Inc. Re-Use/Re-Streaming Forbidden', which conflicts with "
    "the CC BY-SA 4.0 statement on Zindi: analyse locally, never redistribute or re-stream the video.",
    "No vehicle-count, box, track, weather or light annotations are provided by the source.",
    "Congestion labels (Train.csv) require Zindi sign-in and acceptance of competition terms.",
]


def load_manifest() -> DatasetManifest:
    if MANIFEST_PATH.exists():
        manifest = DatasetManifest.load(MANIFEST_PATH)
        manifest.notes = list(NOTES)
        manifest.licence_verified = False  # conflicting statements (Zindi CC BY-SA vs. burned-in IWCP notice)
        return manifest
    return DatasetManifest(
        dataset=barbados.DATASET_NAME,
        version="zindi-2026 re-encoded (Google Drive folders)",
        source_url=barbados.SOURCE_URL,
        access_date=date.today(),
        licence=barbados.LICENCE + " (stated by Zindi; video overlay says re-use forbidden — see notes)",
        licence_verified=False,
        custodian="Zindi / competition host; video overlay names IWCP Inc.",
        citation="Zindi, Barbados Traffic Analysis Challenge, 2025-2026.",
        notes=list(NOTES),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("subset_config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    settings = Settings()
    configure_logging(settings.log_level, settings.log_dir, "download.jsonl")
    spec = yaml.safe_load(args.subset_config.read_text(encoding="utf-8"))
    cameras = tuple(spec["cameras"])
    windows = [
        barbados.Window(w["name"], datetime.fromisoformat(w["start"]), int(w["minutes"]), w["condition_hint"])
        for w in spec["windows"]
    ]

    refs: list[barbados.ClipRef] = []
    for folder in (*cameras, "added"):
        entries = list_drive_folder(barbados.DRIVE_FOLDERS[folder])
        logger.info("listed %d entries in Drive folder %s", len(entries), folder)
        refs.extend(barbados.to_clip_refs(entries))
    refs = barbados.to_clip_refs([r.entry for r in refs])  # de-duplicate across main and 'added' folders

    selected = barbados.select_clips(refs, windows, cameras)
    expected = len(windows) * len(cameras)
    logger.info("selected %d clips across %d window-camera pairs", len(selected), expected)
    for window in windows:
        for camera in cameras:
            n = sum(1 for r, w in selected if w is window and r.camera_id == camera)
            if n < window.minutes:
                logger.warning("window %s camera %s: %d/%d minutes available", window.name, camera, n, window.minutes)
    if args.dry_run:
        return 0

    manifest = load_manifest()
    raw_root = settings.data_root / "raw" / barbados.DATASET_NAME
    offset = timedelta(hours=float(spec.get("local_offset_hours", 0)))
    selected_ids = {ref.clip_id for ref, _ in selected}
    stale = [item for item in manifest.items if item.id not in selected_ids]
    for item in stale:  # clips from a superseded window definition: remove file and manifest entry
        (settings.data_root / item.file).unlink(missing_ok=True)
        manifest.items.remove(item)
        logger.info("removed %s (no longer in subset %s)", item.id, spec["version"])
    for index, (ref, window) in enumerate(selected, start=1):
        relative = f"raw/{barbados.DATASET_NAME}/{ref.camera_id}/{ref.entry.name}"
        destination = settings.data_root / relative
        sha = None
        try:
            existing = manifest.item(ref.clip_id)
            if destination.is_file() and destination.stat().st_size == existing.size_bytes:
                sha = existing.sha256  # already downloaded: refresh attributes only
        except KeyError:
            pass
        if sha is None:
            sha = download_file(ref.entry.download_url, destination)
        manifest.upsert(
            ManifestItem(
                id=ref.clip_id,
                file=relative,
                sha256=sha,
                size_bytes=destination.stat().st_size,
                source_ref=f"gdrive:{ref.entry.file_id}",
                attributes={
                    "camera_id": ref.camera_id,
                    "start_time_filename": ref.start_time.isoformat(),
                    "start_time_local": (ref.start_time + offset).isoformat(),
                    "local_offset_hours": offset.total_seconds() / 3600,
                    "window": window.name,
                    "condition_hint": window.condition_hint,
                    "subset_version": spec["version"],
                },
                used_by=["O2", "O4", "KPI-2"],
            )
        )
        if index % 20 == 0:
            manifest.save(MANIFEST_PATH)
            logger.info("downloaded %d/%d", index, len(selected))
    manifest.save(MANIFEST_PATH)
    total = sum(item.size_bytes for item in manifest.items) / 1024**3
    logger.info("manifest now holds %d items (%.2f GB) under %s", len(manifest.items), total, raw_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
