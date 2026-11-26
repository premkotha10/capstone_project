import hashlib
import json
import logging
from pathlib import Path

import pytest

from src.common.hardware import HardwareReport, select_device
from src.common.logging_setup import configure_logging
from src.common.provenance import environment_snapshot, sha256_file


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    payload = b"dsci41" * 100_000
    path = tmp_path / "blob.bin"
    path.write_bytes(payload)
    assert sha256_file(path, chunk_size=4096) == hashlib.sha256(payload).hexdigest()


def test_environment_snapshot_has_required_fields() -> None:
    snap = environment_snapshot()
    for key in ("timestamp_utc", "python", "platform", "git_commit", "packages"):
        assert key in snap
    assert snap["python"].startswith("3.11")


def _report(**overrides: object) -> HardwareReport:
    fields: dict[str, object] = dict(cpu_physical_cores=4, cpu_logical_cores=8, ram_total_gb=8.0, cuda_available=False)
    fields.update(overrides)
    return HardwareReport(**fields)  # type: ignore[arg-type]


def test_select_device_cpu_fallback() -> None:
    assert select_device("auto", _report()) == "cpu"
    assert select_device("cuda", _report()) == "cpu"
    assert select_device("cpu", _report(cuda_available=True)) == "cpu"


def test_select_device_prefers_accelerators() -> None:
    assert select_device("auto", _report(cuda_available=True)) == "cuda"
    assert select_device("auto", _report(xpu_available=True)) == "xpu"
    assert select_device("openvino", _report(openvino_devices=["CPU", "GPU"])) == "openvino"


def test_select_device_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        select_device("tpu", _report())


def test_json_log_file_contains_extra_fields(tmp_path: Path) -> None:
    configure_logging("INFO", log_dir=tmp_path, filename="t.jsonl")
    logging.getLogger("dsci41.test").info("processed", extra={"video_id": "v1", "fps": 12.5})
    for handler in logging.getLogger().handlers:
        handler.flush()
    record = json.loads((tmp_path / "t.jsonl").read_text(encoding="utf-8").strip().splitlines()[-1])
    assert record["msg"] == "processed"
    assert record["video_id"] == "v1"
    assert record["fps"] == 12.5
    configure_logging("WARNING")  # detach file handler so tmp_path can be removed on Windows
