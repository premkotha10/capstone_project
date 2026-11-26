"""Blind reference-counting tool (docs/COUNTING_PROTOCOL.md).

Shows the raw clip with the counting lines only. It never shows detections, pipeline counts or another annotator's
numbers, so the counts stay independent of the system they are used to evaluate.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from dashboard.common import pill, setup, source_note
from src.common.config import Settings
from src.common.provenance import sha256_file
from src.counting.line_counter import CameraGeometry
from src.evaluation.reference_counts import (
    CLASSES,
    PLAN_PATH,
    PROTOCOL_VERSION,
    CountCell,
    ReferenceRecord,
    ReferenceStoreError,
    consolidate,
    load_plan,
    load_records,
    record_path,
    save_record,
)
from src.pipeline.runner import H264Writer

setup("Reference Counting", "Count vehicles by hand so the system can be scored. Blind to model output.")
settings = Settings()
CACHE = settings.data_root / "annotation_cache"
MAX_PREVIEW_WIDTH = 1080

if settings.hosted:
    st.info("Reference counting runs in the local project, where the evidence clips are stored.")
    st.stop()
if not PLAN_PATH.is_file():
    st.error("No counting plan found. Run `python scripts/make_counting_plan.py` first.")
    st.stop()
plan = load_plan()
records = load_records()


def scale_for(width: int) -> float:
    return min(1.0, MAX_PREVIEW_WIDTH / width)


def draw_lines(image: np.ndarray, geometry: CameraGeometry, k: float) -> np.ndarray:
    out = image.copy()
    t = max(int(2 * out.shape[1] / 960), 2)
    for line in geometry.lines:
        p1 = tuple(int(v * k) for v in line.p1)
        p2 = tuple(int(v * k) for v in line.p2)
        cv2.line(out, p1, p2, (0, 215, 255), t)
        cv2.putText(
            out,
            line.name,
            (p1[0] + 4, max(p1[1] - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5 * t / 2,
            (0, 215, 255),
            max(t // 2, 1),
        )
    return out


@st.cache_data(show_spinner="Preparing the clip with counting lines…", max_entries=8)
def preview(clip_path: str, camera_id: str, geometry_sha: str) -> str:
    """Render the clip with only the counting lines drawn (cached on disk)."""
    target = CACHE / camera_id / f"{geometry_sha[:12]}_{sha256_file(clip_path)[:12]}.mp4"
    if target.is_file():
        return str(target)
    geometry = CameraGeometry.load(settings_path(camera_id))
    cap = cv2.VideoCapture(clip_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    k = scale_for(w)
    size = (int(w * k) // 2 * 2, int(h * k) // 2 * 2)
    target.parent.mkdir(parents=True, exist_ok=True)
    writer = H264Writer(target, fps, size)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if k != 1.0:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        writer.write(draw_lines(frame, geometry, k))
    cap.release()
    writer.release()
    return str(target)


def settings_path(camera_id: str):  # noqa: ANN201
    from src.common.config import PROJECT_ROOT

    return PROJECT_ROOT / "configs" / "cameras" / f"{camera_id}.yaml"


def frame_at(clip_path: str, index: int, camera_id: str) -> np.ndarray | None:
    cap = cv2.VideoCapture(clip_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    k = scale_for(frame.shape[1])
    if k != 1.0:
        frame = cv2.resize(frame, (int(frame.shape[1] * k), int(frame.shape[0] * k)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(draw_lines(frame, CameraGeometry.load(settings_path(camera_id)), k), cv2.COLOR_BGR2RGB)


# ---------------- who is counting ----------------
c1, c2 = st.columns([1, 1])
annotator = c1.text_input(
    "Your annotator ID",
    key="annotator",
    max_chars=32,
    help="Initials or roll number only (letters, digits, - and _). No full names.",
)
role = c2.selectbox(
    "Your role for this clip",
    ["primary", "secondary", "adjudicator"],
    help="Two people count every clip independently (primary and secondary). "
    "An adjudicator recounts only cells where they disagree.",
)
valid_id = bool(annotator) and annotator.replace("-", "").replace("_", "").isalnum()
if annotator and not valid_id:
    st.error("Use letters, digits, '-' or '_' only.")

# ---------------- progress (no values shown) ----------------
done_by_clip: dict[str, Counter] = {}
for r in records:
    done_by_clip.setdefault(r.clip_id, Counter())[r.role] += 1
table, _ = consolidate(records)
progress_rows = []
for clip in plan:
    roles = done_by_clip.get(clip.clip_id, Counter())
    mine = valid_id and record_path(clip.clip_id, annotator).is_file()
    disagree = (
        int(((table["clip_id"] == clip.clip_id) & (table["status"] == "disagree")).sum()) if not table.empty else 0
    )
    progress_rows.append(
        {
            "clip": clip.clip_id,
            "condition": clip.condition_hint,
            "primary": "✓" if roles["primary"] else "",
            "secondary": "✓" if roles["secondary"] else "",
            "cells to adjudicate": disagree or "",
            "you": "saved" if mine else "",
        }
    )
progress = pd.DataFrame(progress_rows)
k1, k2, k3 = st.columns(3)
k1.metric("Clips in plan", len(plan))
k2.metric("Counted twice", int(((progress["primary"] == "✓") & (progress["secondary"] == "✓")).sum()))
k3.metric(
    "Cells awaiting adjudication", int(pd.to_numeric(progress["cells to adjudicate"], errors="coerce").fillna(0).sum())
)
with st.expander("Progress by clip (no counts are shown here)", expanded=not records):
    st.dataframe(progress, hide_index=True, width="stretch")

st.divider()
clip = st.selectbox("Clip to count", plan, format_func=lambda c: f"{c.clip_id} · {c.condition_hint}")
if not clip.path.is_file():
    st.error(f"Clip file not found: {clip.path}")
    st.stop()
geometry = CameraGeometry.load(clip.geometry_path)
geometry_sha = sha256_file(clip.geometry_path)

left, right = st.columns([1.35, 1])
with left:
    st.video(preview(str(clip.path), clip.camera_id, geometry_sha))
    source_note("raw clip with counting lines only · use the player's speed menu to slow it down")
    with st.expander("Step frame by frame (for dense scenes)"):
        cap = cv2.VideoCapture(str(clip.path))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        cap.release()
        idx = st.slider("Frame", 0, max(n_frames - 1, 0), 0, key=f"frame_{clip.clip_id}")
        st.caption(f"t = {idx / fps:.2f} s of {n_frames / fps:.1f} s")
        image = frame_at(str(clip.path), idx, clip.camera_id)
        if image is not None:
            st.image(image, width="stretch")

with right:
    st.subheader("Your counts")
    st.caption("Count each vehicle once per line, when its wheels' road contact crosses the line (protocol §3).")
    if role == "adjudicator" and not table.empty:
        dis = table[(table["clip_id"] == clip.clip_id) & (table["status"] == "disagree")]
        if dis.empty:
            st.info("No disagreeing cells on this clip.")
        else:
            st.markdown(
                "Recount only these cells: "
                + ", ".join(f"`{r.line} · {r.direction} · {r.vehicle_class}`" for r in dis.itertuples())
            )
    start_key = f"start_{clip.clip_id}"
    st.session_state.setdefault(start_key, datetime.now())
    cells: list[CountCell] = []
    for line in geometry.lines:
        for direction in (line.negative, line.positive):
            st.markdown(f"**{line.name}** · {direction.replace('_', ' ')}")
            cols = st.columns(3)
            for i, cls in enumerate(CLASSES):
                n = cols[i % 3].number_input(
                    cls, min_value=0, max_value=500, step=1, key=f"n_{clip.clip_id}_{line.name}_{direction}_{cls}"
                )
                if n:
                    cells.append(CountCell(line=line.name, direction=direction, vehicle_class=cls, count=int(n)))
    uncertain = st.checkbox("I am uncertain about some of these counts", key=f"unc_{clip.clip_id}")
    notes = st.text_area("Notes (what was uncertain, occlusions, unusual vehicles)", key=f"notes_{clip.clip_id}")
    resolves = []
    if role == "adjudicator":
        others = sorted({r.annotator for r in records if r.clip_id == clip.clip_id and r.role != "adjudicator"})
        resolves = st.multiselect("Resolves the counts of", others, default=others)
    replace = st.checkbox("Replace my saved count for this clip", key=f"rep_{clip.clip_id}")
    st.markdown(
        pill(f"protocol v{PROTOCOL_VERSION}", "info")
        + " "
        + pill(f"{sum(c.count for c in cells)} vehicles entered", "ok" if cells else "warn"),
        unsafe_allow_html=True,
    )
    if st.button("Save count", type="primary", disabled=not valid_id):
        try:
            record = ReferenceRecord(
                clip_id=clip.clip_id,
                clip_sha256=sha256_file(clip.path),
                geometry_sha256=geometry_sha,
                annotator=annotator,
                role=role,
                started_at=st.session_state[start_key],
                finished_at=datetime.now(),
                cells=cells,
                uncertain=uncertain,
                notes=notes,
                adjudicates=resolves,
            )
            path = save_record(record, clip, overwrite=replace)
        except (ReferenceStoreError, ValueError) as exc:
            st.error(str(exc))
        else:
            st.success(f"Saved {record.total} vehicles for {clip.clip_id} as {annotator} ({role}). File: {path.name}")
            st.caption("A clip with zero vehicles is a valid count: saving records it.")
