"""Environment settings and validated, hashable pipeline configuration (NFR-5).

* `Settings` — machine-specific values from environment variables / `.env` (paths, device, log level).
* `PipelineConfig` — experiment configuration loaded from YAML. Unknown keys are rejected so that a typo
  cannot silently change an experiment. `config_hash()` identifies the exact configuration of every run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Machine-level settings. Large artefacts default to a folder outside OneDrive (ADR-7)."""

    model_config = SettingsConfigDict(env_prefix="DSCI41_", env_file=".env", extra="ignore")

    data_root: Path = Field(default=Path.home() / "dsci41" / "data")
    models_root: Path = Field(default=Path.home() / "dsci41" / "models")
    reports_root: Path = Field(default=PROJECT_ROOT / "reports")
    log_dir: Path = Field(default=Path.home() / "dsci41" / "logs")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    device: Literal["auto", "cpu", "cuda", "xpu", "openvino"] = "auto"
    hosted: bool = False  # public demo mode (e.g. Hugging Face Space): uploads only, size limits
    hosted_max_upload_mb: int = 200
    hosted_max_seconds: float = 60.0
    hosted_max_width: int = 1280


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DetectionConfig(_Strict):
    model: str = "yolo11n.pt"
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    image_size: int = Field(default=640, ge=32, multiple_of=32)
    frame_stride: int = Field(default=1, ge=1)


class TrackingConfig(_Strict):
    algorithm: Literal["bytetrack"] = "bytetrack"
    track_activation_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    lost_track_buffer: int = Field(default=30, ge=1)
    minimum_matching_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    minimum_consecutive_frames: int = Field(default=1, ge=1)


class CountingConfig(_Strict):
    mode: Literal["image_line", "ground_plane"] = "image_line"  # O2-A vs O2-B
    interval_seconds: float = Field(default=60.0, gt=0)
    min_track_length: int = Field(default=3, ge=1)


class RobustnessConfig(_Strict):
    """O3 contribution switches. All disabled == O2 reference behaviour."""

    enabled: bool = False
    condition_gate: bool = False
    low_light_enhancement: bool = False
    rain_suppression: bool = False
    occlusion_aware_tracking: bool = False


class RuntimeConfig(_Strict):
    seed: int = 42
    num_threads: int = Field(default=0, ge=0)  # 0 = library default


class PipelineConfig(_Strict):
    name: str
    description: str = ""
    detection: DetectionConfig = DetectionConfig()
    tracking: TrackingConfig = TrackingConfig()
    counting: CountingConfig = CountingConfig()
    robustness: RobustnessConfig = RobustnessConfig()
    runtime: RuntimeConfig = RuntimeConfig()

    def config_hash(self) -> str:
        """Stable SHA-256 of the canonical configuration (first 16 hex chars)."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into a copy of `base`."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")
    return data


def load_config(path: str | Path) -> PipelineConfig:
    """Load a pipeline config. A file may declare `extends: <relative path>` to inherit another config."""
    path = Path(path).resolve()
    data = _read_yaml(path)
    parent = data.pop("extends", None)
    if parent is not None:
        parent_path = (path.parent / parent).resolve()
        if parent_path == path:
            raise ValueError(f"{path}: config cannot extend itself")
        base = load_config(parent_path).model_dump(mode="json")
        data = deep_merge(base, data)
    return PipelineConfig.model_validate(data)
