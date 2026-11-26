import json
from pathlib import Path

import pytest

from src.common.config import PipelineConfig
from src.evaluation.latency import latency_stats
from src.evaluation.runs import write_run_record


def test_latency_percentiles() -> None:
    stats = latency_stats([float(i) for i in range(1, 101)], failures=2)
    assert stats.trials == 100 and stats.failures == 2
    assert stats.p50_ms == pytest.approx(50.5)
    assert stats.p95_ms == pytest.approx(95.05)
    assert stats.p99_ms == pytest.approx(99.01)
    assert stats.max_ms == 100.0


def test_latency_requires_samples() -> None:
    with pytest.raises(ValueError):
        latency_stats([])


def test_run_record_contains_provenance(tmp_path: Path) -> None:
    cfg = PipelineConfig(name="t")
    path = write_run_record(
        tmp_path, "r1", cfg, inputs=[{"id": "x", "sha256": "0" * 64}], model={"sha256": "m"}, results={"ap50": 0.5}
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["config_hash"] == cfg.config_hash()
    assert record["inputs"][0]["id"] == "x"
    assert "git_commit" in record["environment"] and "packages" in record["environment"]
    assert record["results"] == {"ap50": 0.5}
