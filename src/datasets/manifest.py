"""Evidence manifest (PDF p.12 "Provenance and independent partition").

Each dataset has `data/manifests/<dataset>.yaml` recording source URL, access date, licence, custodian, item IDs,
checksums, exclusions, transformations, partition assignment and the objectives/KPIs using each item.
Loaders refuse files that are not registered (FR-1).
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.common.provenance import sha256_file


class Partition(StrEnum):
    UNASSIGNED = "unassigned"
    DEV_TRAIN = "dev_train"
    DEV_VAL = "dev_val"
    ACCEPTANCE = "acceptance_ac3"  # reserved independent evidence; never used for fitting or tuning


class ManifestItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    file: str  # path relative to DATA_ROOT, forward slashes
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    source_ref: str | None = None  # e.g. Drive file ID or archive member path
    partition: Partition = Partition.UNASSIGNED
    attributes: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    used_by: list[str] = Field(default_factory=list)


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str
    version: str
    source_url: str
    access_date: date
    licence: str
    licence_verified: bool
    custodian: str
    citation: str | None = None
    notes: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)
    transformations: list[str] = Field(default_factory=list)
    items: list[ManifestItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_ids(self) -> DatasetManifest:
        ids = [item.id for item in self.items]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(f"duplicate manifest item ids: {sorted(duplicates)[:5]}")
        return self

    def item(self, item_id: str) -> ManifestItem:
        for item in self.items:
            if item.id == item_id:
                return item
        raise KeyError(f"{item_id!r} is not registered in the {self.dataset} manifest")

    def upsert(self, new: ManifestItem) -> None:
        self.items = [item for item in self.items if item.id != new.id] + [new]
        self.items.sort(key=lambda item: item.id)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(mode="json")
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> DatasetManifest:
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def sha256_directory(directory: Path) -> tuple[str, int]:
    """Deterministic digest of a directory of files: SHA-256 over sorted 'relative_path:file_sha256' lines.

    Used for image-sequence items (one manifest entry per sequence instead of per frame). Returns (digest, bytes).
    """
    import hashlib

    lines: list[str] = []
    total = 0
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        lines.append(f"{path.relative_to(directory).as_posix()}:{sha256_file(path)}")
        total += path.stat().st_size
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest(), total


def _size_on_disk(path: Path) -> int:
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return path.stat().st_size


class UnregisteredEvidenceError(KeyError):
    """Raised when a file is used that is not recorded in an evidence manifest."""


def verify_items(manifest: DatasetManifest, data_root: Path, recompute_hash: bool = False) -> list[str]:
    """Return a list of problems (missing files, size or checksum mismatch). Empty list == consistent."""
    problems: list[str] = []
    for item in manifest.items:
        path = data_root / item.file
        if not path.exists():
            problems.append(f"{item.id}: missing {path}")
            continue
        size = _size_on_disk(path)
        if size != item.size_bytes:
            problems.append(f"{item.id}: size {size} != {item.size_bytes}")
        elif recompute_hash:
            digest = sha256_directory(path)[0] if path.is_dir() else sha256_file(path)
            if digest != item.sha256:
                problems.append(f"{item.id}: sha256 mismatch")
    return problems
