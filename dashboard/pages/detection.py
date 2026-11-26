from __future__ import annotations

import plotly.express as px
import streamlit as st

from dashboard.common import CLASS_COLOURS, estimate_banner, kpi, select_run, setup, source_note

setup("Vehicle Detection", "Per-frame YOLO11n detections: classes, confidence and density over time.")
run = select_run()
if run is None:
    st.stop()
estimate_banner()
det = run.detections
if det.empty:
    st.info("This run has no detections.")
    st.stop()

threshold = st.sidebar.slider(
    "Confidence threshold (view)",
    0.25,
    0.95,
    0.25,
    0.05,
    help="Filters what this page shows. The pipeline ran at 0.25.",
)
view = det[det["confidence"] >= threshold]
frames = run.frames

c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi("Detections", f"{len(view):,}", f"at confidence ≥ {threshold:.2f}")
with c2:
    kpi("Per frame (mean)", f"{len(view) / max(len(frames), 1):.1f}", f"{len(frames):,} frames")
with c3:
    kpi("Median confidence", f"{view['confidence'].median():.2f}" if len(view) else "—")
with c4:
    small = ((view["x2"] - view["x1"]) * (view["y2"] - view["y1"]) < 32 * 32).mean() if len(view) else 0
    kpi("Small boxes", f"{small:.0%}", "area under 32×32 px, reported separately")

left, right = st.columns([1.3, 1])
with left:
    st.subheader("Annotated frame")
    snaps = run.snapshots
    if snaps:
        pick = st.select_slider(
            "Time in window", snaps, format_func=lambda p: f"{int(p.stem[1:]) // 60:02d}:{int(p.stem[1:]) % 60:02d}"
        )
        st.image(str(pick), width="stretch")
        source_note("snapshots/ · one annotated frame per minute (local only)")
with right:
    st.subheader("Confidence by class")
    fig = px.histogram(
        view,
        x="confidence",
        color="class",
        nbins=30,
        barmode="overlay",
        opacity=0.7,
        height=330,
        color_discrete_map=CLASS_COLOURS,
        labels={"confidence": "Detector confidence"},
    )
    st.plotly_chart(fig, width="stretch")
    by_class = view["class"].value_counts().rename_axis("class").reset_index(name="boxes")
    st.dataframe(by_class, hide_index=True, width="stretch")
    st.caption("COCO classes only: vans appear as car or truck; the model cannot claim other vehicle types.")

st.subheader("Detections per frame over time")
per = view.groupby("t_s").size().rename("detections").reset_index()
per["minute"] = per["t_s"] / 60
fig = px.line(per, x="minute", y="detections", height=260, labels={"minute": "Minute", "detections": "Boxes"})
fig.update_traces(line=dict(width=1, color="#0F6E8C"))
st.plotly_chart(fig, width="stretch")
source_note("detections.csv · boxes per processed frame (not vehicles: one vehicle appears in many frames)")
