import importlib.util
import sys
from pathlib import Path

import pytest

from src.counting.line_counter import CameraGeometry

ROOT = Path(__file__).resolve().parents[2]


def test_custom_horizontal_line() -> None:
    g = CameraGeometry.custom_line((1000, 500), "horizontal", 0.6)
    line = g.lines[0]
    assert line.p1 == (20.0, 300.0) and line.p2 == (980.0, 300.0)
    assert (line.positive, line.negative) == ("up", "down")
    # a point moving down the image ends on the 'negative' (down) side
    assert line.side(500, 350) > line.side(500, 250)


def test_custom_vertical_line_and_validation() -> None:
    line = CameraGeometry.custom_line((1000, 500), "vertical", 0.25).lines[0]
    assert line.p1[0] == line.p2[0] == 250.0
    with pytest.raises(ValueError):
        CameraGeometry.custom_line((1000, 500), "diagonal", 0.5)
    with pytest.raises(ValueError):
        CameraGeometry.custom_line((1000, 500), "horizontal", 1.0)


def test_scaled_geometry() -> None:
    g = CameraGeometry.custom_line((2000, 1000), "horizontal", 0.5).scaled(0.5)
    assert g.frame_size == (1000, 500)
    assert g.lines[0].p1 == (20.0, 250.0)


def _load_builder():  # noqa: ANN202
    spec = importlib.util.spec_from_file_location("build_hf_space", ROOT / "scripts" / "build_hf_space.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_hf_space"] = module
    spec.loader.exec_module(module)
    return module


def test_space_bundle_contains_app_and_no_data(tmp_path: Path) -> None:
    builder = _load_builder()
    files = {p.relative_to(tmp_path / "b").as_posix() for p in builder.build(tmp_path / "b")}
    for required in (
        "Dockerfile",
        "README.md",
        "pyproject.toml",
        "uv.lock",
        "LICENSE",
        "dashboard/app.py",
        "src/pipeline/runner.py",
        "configs/o2_a.yaml",
    ):
        assert required in files
    assert not any(f.startswith(("data/", "tests/", "docs/", "reports/", "site/")) for f in files)
    readme = (tmp_path / "b" / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("---\n") and "sdk: docker" in readme and "app_port: 7860" in readme


def test_all_app_sources_compile() -> None:
    """Streamlit page renders can hide a SyntaxError; compile every source file explicitly."""
    import py_compile

    for path in [*(ROOT / "dashboard").rglob("*.py"), *(ROOT / "src").rglob("*.py"), *(ROOT / "scripts").glob("*.py")]:
        py_compile.compile(str(path), doraise=True)
