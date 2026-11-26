from pathlib import Path

import pytest
from pydantic import ValidationError

from src.common.config import PROJECT_ROOT, PipelineConfig, deep_merge, load_config

CONFIGS = PROJECT_ROOT / "configs"


def test_base_config_loads() -> None:
    cfg = load_config(CONFIGS / "base.yaml")
    assert cfg.name == "base"
    assert cfg.robustness.enabled is False


@pytest.mark.parametrize("name, mode", [("o2_a.yaml", "image_line"), ("o2_b.yaml", "ground_plane")])
def test_o2_candidates_inherit_base(name: str, mode: str) -> None:
    base = load_config(CONFIGS / "base.yaml")
    cfg = load_config(CONFIGS / name)
    assert cfg.counting.mode == mode
    assert cfg.detection == base.detection  # same detector => fair O2-A vs O2-B comparison
    assert cfg.tracking == base.tracking


def test_unknown_key_rejected(tmp_path: Path) -> None:
    path = tmp_path / "typo.yaml"
    path.write_text("name: t\ndetection:\n  confidence_treshold: 0.5\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(path)


def test_out_of_range_threshold_rejected() -> None:
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate({"name": "t", "detection": {"confidence_threshold": 1.5}})


def test_self_extension_rejected(tmp_path: Path) -> None:
    path = tmp_path / "loop.yaml"
    path.write_text("extends: loop.yaml\nname: loop\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_config_hash_is_stable_and_sensitive() -> None:
    a1 = PipelineConfig(name="x")
    a2 = PipelineConfig(name="x")
    b = PipelineConfig.model_validate({"name": "x", "detection": {"confidence_threshold": 0.3}})
    assert a1.config_hash() == a2.config_hash()
    assert a1.config_hash() != b.config_hash()
    assert len(a1.config_hash()) == 16


def test_o2_variants_have_distinct_hashes() -> None:
    assert load_config(CONFIGS / "o2_a.yaml").config_hash() != load_config(CONFIGS / "o2_b.yaml").config_hash()


def test_deep_merge_does_not_mutate_inputs() -> None:
    base = {"a": {"b": 1, "c": 2}}
    merged = deep_merge(base, {"a": {"b": 9}})
    assert merged == {"a": {"b": 9, "c": 2}}
    assert base == {"a": {"b": 1, "c": 2}}
