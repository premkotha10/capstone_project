"""Download a configured subset of UA-DETRAC sequences and register them in the evidence manifest.

Usage:  python scripts/download_ua_detrac_subset.py configs/data/ua_detrac_subset_v1.yaml

* Annotation zips (train/test XML, ~25 MB) are downloaded whole.
* Images are read *selectively* from the remote ~9.85 GB DETRAC-Images.zip using HTTP range requests, so only the
  chosen sequences are transferred.
* Frames land in DATA_ROOT/raw/ua_detrac/images/<sequence>/imgNNNNN.jpg; XML in raw/ua_detrac/annotations/.
* One manifest item per sequence (directory digest) plus one per annotation zip.
"""

from __future__ import annotations

import argparse
import logging
import sys
import zipfile
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import PROJECT_ROOT, Settings  # noqa: E402
from src.common.logging_setup import configure_logging  # noqa: E402
from src.datasets import ua_detrac  # noqa: E402
from src.datasets.manifest import DatasetManifest, ManifestItem, sha256_directory  # noqa: E402
from src.datasets.remote import DRIVE_DOWNLOAD_URL, RangeFile, download_file, http_range_fetcher  # noqa: E402

logger = logging.getLogger("download_ua_detrac")
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / f"{ua_detrac.DATASET_NAME}.yaml"
ANNOTATION_ZIPS = {
    "DETRAC-Train-Annotations-XML.zip": "1f22g3puG164gemY_njcPG8beuCcuMjqb",
    "DETRAC-Test-Annotations-XML.zip": "1zc8KZWUa3Cw0DbcFqTCMBmUZzbfuhGKr",
}


def load_manifest() -> DatasetManifest:
    if MANIFEST_PATH.exists():
        return DatasetManifest.load(MANIFEST_PATH)
    return DatasetManifest(
        dataset=ua_detrac.DATASET_NAME,
        version="UA-DETRAC (author Google Drive release: DETRAC-Images.zip + XML annotations)",
        source_url=ua_detrac.SOURCE_URL,
        access_date=date.today(),
        licence="unverified — treated as academic, non-commercial use with citation",
        licence_verified=False,
        custodian="UA-DETRAC authors (University at Albany / Tianjin University)",
        citation=ua_detrac.CITATION,
        notes=[
            "Original site detrac-db.rit.albany.edu is offline; files from the author page's Google Drive links.",
            "Only selected sequences were extracted from DETRAC-Images.zip via HTTP range requests.",
            "Site grouping uses a documented proxy (first four digits of the sequence ID).",
        ],
    )


def sequence_index(annotation_dir: Path) -> dict[str, tuple[str, ua_detrac.SequenceAnnotation]]:
    index: dict[str, tuple[str, ua_detrac.SequenceAnnotation]] = {}
    for zip_name, split in (("DETRAC-Train-Annotations-XML.zip", "train"), ("DETRAC-Test-Annotations-XML.zip", "test")):
        with zipfile.ZipFile(annotation_dir / zip_name) as zf:
            for member in zf.namelist():
                if member.endswith(".xml"):
                    seq = ua_detrac.parse_sequence_xml(zf.read(member))
                    index[seq.name] = (split, seq)
    return index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("subset_config", type=Path)
    args = parser.parse_args()

    settings = Settings()
    configure_logging(settings.log_level, settings.log_dir, "download.jsonl")
    spec = yaml.safe_load(args.subset_config.read_text(encoding="utf-8"))
    wanted = [(cond, name) for cond, names in spec["sequences"].items() for name in names]

    manifest = load_manifest()
    base = settings.data_root / "raw" / ua_detrac.DATASET_NAME
    annotation_dir = base / "annotations"
    for zip_name, file_id in ANNOTATION_ZIPS.items():
        relative = f"raw/{ua_detrac.DATASET_NAME}/annotations/{zip_name}"
        path = settings.data_root / relative
        if not path.exists():
            sha = download_file(DRIVE_DOWNLOAD_URL.format(file_id=file_id), path)
        else:
            from src.common.provenance import sha256_file

            sha = sha256_file(path)
        manifest.upsert(
            ManifestItem(
                id=zip_name.removesuffix(".zip"),
                file=relative,
                sha256=sha,
                size_bytes=path.stat().st_size,
                source_ref=f"gdrive:{file_id}",
                attributes={"kind": "annotations"},
                used_by=["O2", "O3", "KPI-1", "KPI-4", "KPI-5"],
            )
        )

    index = sequence_index(annotation_dir)
    fetch, size = http_range_fetcher(DRIVE_DOWNLOAD_URL.format(file_id=ua_detrac.IMAGES_ARCHIVE_DRIVE_ID))
    remote = RangeFile(fetch, size)
    with zipfile.ZipFile(remote) as archive:
        members = {info.filename: info for info in archive.infolist()}
        for condition, name in wanted:
            split, seq = index[name]
            if seq.weather != condition:
                raise ValueError(f"{name}: config says {condition}, XML says {seq.weather}")
            out_dir = base / "images" / name
            frame_numbers = [f.frame_number for f in seq.frames]
            prefix = f"DETRAC-Images/{name}/"
            jpgs = sorted(m for m in members if m.startswith(prefix) and m.endswith(".jpg"))
            done = out_dir.is_dir() and len(list(out_dir.glob("*.jpg"))) == len(jpgs)
            if not done:
                out_dir.mkdir(parents=True, exist_ok=True)
                for member in jpgs:
                    target = out_dir / Path(member).name
                    if target.exists() and target.stat().st_size == members[member].file_size:
                        continue
                    target.write_bytes(archive.read(member))
                logger.info(
                    "%s: extracted %d frames (%.0f MB fetched so far)", name, len(jpgs), remote.bytes_fetched / 1e6
                )
            digest, total = sha256_directory(out_dir)
            manifest.upsert(
                ManifestItem(
                    id=name,
                    file=f"raw/{ua_detrac.DATASET_NAME}/images/{name}",
                    sha256=digest,
                    size_bytes=total,
                    source_ref=f"gdrive:{ua_detrac.IMAGES_ARCHIVE_DRIVE_ID}#{prefix}",
                    attributes={
                        "split_original": split,
                        "weather": seq.weather,
                        "camera_state": seq.camera_state,
                        "site_proxy": ua_detrac.site_proxy(name),
                        "frames_annotated": len(frame_numbers),
                        "frames_images": len(jpgs),
                        "tracks": len(seq.track_ids),
                        "mean_density": round(seq.mean_density, 2),
                        "occluded_fraction": round(seq.occluded_fraction, 3),
                        "subset_version": spec["version"],
                    },
                    used_by=["O2", "O3", "KPI-1", "KPI-4", "KPI-5", "NT-1", "NT-2", "NT-4"],
                )
            )
            manifest.save(MANIFEST_PATH)
    total_gb = sum(i.size_bytes for i in manifest.items) / 1024**3
    logger.info(
        "done: %d manifest items, %.2f GB on disk, %.2f GB transferred",
        len(manifest.items),
        total_gb,
        remote.bytes_fetched / 1024**3,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
