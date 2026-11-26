from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.evaluation.reference_counts import (
    CountCell,
    PlanClip,
    ReferenceRecord,
    ReferenceStoreError,
    consolidate,
    icc_2_1,
    load_records,
    pipeline_clip_counts,
    save_record,
    score,
)

T0 = datetime(2026, 11, 23, 10, 0)
SHA = "a" * 64


def _rec(annotator: str, role: str, counts: dict, clip: str = "c1", **kw) -> ReferenceRecord:
    cells = [CountCell(line=ln, direction=d, vehicle_class=c, count=n) for (ln, d, c), n in counts.items()]
    return ReferenceRecord(
        clip_id=clip,
        clip_sha256=SHA,
        geometry_sha256=SHA,
        annotator=annotator,
        role=role,
        started_at=T0,
        finished_at=T0 + timedelta(minutes=5),
        cells=cells,
        **kw,
    )


def test_record_validation() -> None:
    with pytest.raises(ValidationError):
        _rec("Full Name With Spaces", "primary", {})
    with pytest.raises(ValidationError):
        _rec("ab", "adjudicator", {})  # adjudication must say what it resolves
    with pytest.raises(ValidationError):
        CountCell(line="a", direction="in", vehicle_class="tram", count=1)
    with pytest.raises(ValidationError):
        CountCell(line="a", direction="in", vehicle_class="car", count=-1)


def test_save_checks_checksums(tmp_path: Path) -> None:
    geometry = tmp_path / "cam.yaml"
    geometry.write_text("x", encoding="utf-8")
    from src.common.provenance import sha256_file

    rec = _rec("ab", "primary", {("l", "in", "car"): 3})
    rec.geometry_sha256 = sha256_file(geometry)

    class Clip(PlanClip):
        @property
        def geometry_path(self) -> Path:  # type: ignore[override]
            return geometry

    clip = Clip("c1", tmp_path / "c1.mp4", "cam", "run", "night", SHA)
    path = save_record(rec, clip, root=tmp_path / "refs")
    assert path.is_file() and load_records(tmp_path / "refs")[0].total == 3
    with pytest.raises(ReferenceStoreError):
        save_record(rec, clip, root=tmp_path / "refs")  # no silent overwrite
    bad = rec.model_copy(update={"clip_sha256": "b" * 64})
    with pytest.raises(ReferenceStoreError):
        save_record(bad, clip, root=tmp_path / "refs", overwrite=True)


def test_consolidate_agreement_and_disagreement() -> None:
    p = _rec("ab", "primary", {("l", "in", "car"): 5, ("l", "out", "car"): 2})
    s = _rec("cd", "secondary", {("l", "in", "car"): 5, ("l", "out", "car"): 3, ("l", "in", "bus"): 1})
    table, agreement = consolidate([p, s])
    t = table.set_index(["direction", "vehicle_class"])
    assert t.loc[("in", "car"), "status"] == "agreed" and t.loc[("in", "car"), "reference"] == 5
    assert t.loc[("out", "car"), "status"] == "disagree" and np.isnan(t.loc[("out", "car"), "reference"])
    assert t.loc[("in", "bus"), "primary"] == 0  # primary saved the clip, so an absent cell means zero
    assert agreement.cells == 3 and agreement.exact_agreement == pytest.approx(1 / 3)
    assert agreement.within_one == 1.0


def test_adjudication_resolves_disagreement() -> None:
    p = _rec("ab", "primary", {("l", "out", "car"): 2})
    s = _rec("cd", "secondary", {("l", "out", "car"): 3})
    a = _rec("ef", "adjudicator", {("l", "out", "car"): 3}, adjudicates=["ab", "cd"])
    table, _ = consolidate([p, s, a])
    assert table.iloc[0]["status"] == "adjudicated" and table.iloc[0]["reference"] == 3


def test_single_count_is_not_a_reference() -> None:
    table, agreement = consolidate([_rec("ab", "primary", {("l", "in", "car"): 4})])
    assert table.iloc[0]["status"] == "single" and np.isnan(table.iloc[0]["reference"])
    assert agreement is None


def test_icc_perfect_and_noisy() -> None:
    assert icc_2_1(np.array([[1, 1], [5, 5], [9, 9]], dtype=float)) == pytest.approx(1.0)
    assert icc_2_1(np.array([[1, 2], [5, 4], [9, 9]], dtype=float)) < 1.0


def test_score_wape_and_ci() -> None:
    table = pd.DataFrame({"clip_id": ["a", "a", "b"], "reference": [10, 0, 10], "estimate": [8, 1, 10]})
    s = score(table, n_boot=200)
    assert s.wape == pytest.approx(3 / 20) and s.accuracy == pytest.approx(0.85)
    assert s.accuracy_ci95[0] <= s.accuracy <= s.accuracy_ci95[1]
    assert score(table.assign(reference=np.nan)) is None


def test_pipeline_clip_counts_attributes_by_frame(tmp_path: Path) -> None:
    pd.DataFrame({"frame": [0, 2, 1200, 1202], "clip": ["c1", "c1", "c2", "c2"]}).to_csv(
        tmp_path / "frames.csv", index=False
    )
    pd.DataFrame(
        {
            "track_id": [1, 2, 3],
            "line": ["l"] * 3,
            "direction": ["in", "in", "out"],
            "class": ["car", "car", "bus"],
            "frame": [2, 1200, 1202],
            "t_s": [0.1, 60, 60.1],
            "reliability": ["ok"] * 3,
        }
    ).to_csv(tmp_path / "events.csv", index=False)
    c2 = pipeline_clip_counts(tmp_path, "c2").set_index(["direction", "vehicle_class"])["estimate"]
    assert c2.to_dict() == {("in", "car"): 1, ("out", "bus"): 1}
