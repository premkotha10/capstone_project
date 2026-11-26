"""Integration: reference-count comparison against a real pipeline run.

Uses SYNTHETIC test records written to a temporary folder (never the real reference store). Their values are set from
the pipeline's own per-clip counts plus a known offset, so the expected accuracy is known exactly.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.common.provenance import sha256_file
from src.evaluation.reference_counts import (
    PLAN_PATH,
    CountCell,
    ReferenceRecord,
    compare_with_pipeline,
    consolidate,
    load_plan,
    load_records,
    pipeline_clip_counts,
    save_record,
    score,
)
from src.services.results import RunStore

pytestmark = [pytest.mark.slow, pytest.mark.needs_data]


def _usable_clip():  # noqa: ANN202
    if not PLAN_PATH.is_file():
        return None
    for clip in load_plan():
        if (
            clip.path.is_file()
            and (RunStore().root / clip.run / "events.csv").is_file()
            and clip.run.endswith("morning")
        ):
            return clip
    return None


CLIP = _usable_clip()


@pytest.mark.skipif(CLIP is None, reason="needs the counting plan, a morning clip and its pipeline run")
def test_comparison_scores_known_offset(tmp_path) -> None:  # noqa: ANN001
    est = pipeline_clip_counts(RunStore().root / CLIP.run, CLIP.path.stem)
    assert not est.empty, "the planned morning clip should contain pipeline crossings"
    # synthetic reference: pipeline estimate + 1 on the first cell, identical elsewhere
    cells = [
        CountCell(
            line=r.line,
            direction=r.direction,
            vehicle_class=r.vehicle_class,
            count=int(r.estimate) + (1 if i == 0 else 0),
        )
        for i, r in enumerate(est.itertuples())
    ]
    common = dict(
        clip_id=CLIP.clip_id,
        clip_sha256=sha256_file(CLIP.path),
        geometry_sha256=sha256_file(CLIP.geometry_path),
        started_at=datetime(2026, 1, 1),
        finished_at=datetime(2026, 1, 1) + timedelta(minutes=3),
        cells=cells,
        notes="synthetic test record",
    )
    root = tmp_path / "refs"
    save_record(ReferenceRecord(annotator="test-a", role="primary", **common), CLIP, root=root)
    save_record(ReferenceRecord(annotator="test-b", role="secondary", **common), CLIP, root=root)

    table, agreement = consolidate(load_records(root))
    assert agreement.exact_agreement == 1.0
    comparison = compare_with_pipeline(table, [CLIP], RunStore().root)
    summary = score(comparison, n_boot=100)
    reference_total = int(est["estimate"].sum()) + 1
    assert summary.reference_total == reference_total
    assert summary.estimate_total == reference_total - 1
    assert summary.wape == pytest.approx(1 / reference_total)
