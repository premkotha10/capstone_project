from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from dashboard.common import CLASS_COLOURS, estimate_banner, kpi, select_run, setup, source_note

setup("Vehicle Tracking", "ByteTrack identities, trajectories and movement directions.")
run = select_run()
if run is None:
    st.stop()
estimate_banner()
tracks = run.tracks
if tracks.empty:
    st.info("No tracks in this run.")
    st.stop()

min_points = st.sidebar.slider("Minimum track length (points)", 2, 60, 10)
moving = tracks[(tracks["points"] >= min_points) & (tracks["distance_px"] > 40)]
static = tracks[(tracks["points"] >= min_points) & (tracks["distance_px"] <= 40)]

c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi("Track IDs", f"{len(tracks):,}", "all identities created")
with c2:
    kpi("Moving tracks", f"{len(moving):,}", f"≥ {min_points} points, moved > 40 px")
with c3:
    kpi("Stationary tracks", f"{len(static):,}", "parked vehicles or static false positives")
with c4:
    short = (tracks["points"] < 5).mean()
    kpi("Fragments", f"{short:.0%}", "tracks under 5 points (ID-switch indicator)")

left, right = st.columns([1.5, 1])
with left:
    st.subheader("Trajectories")
    pts = run.track_points
    sample_ids = moving.sort_values("points", ascending=False)["track_id"].head(150)
    pts = pts[pts["track_id"].isin(sample_ids)].merge(tracks[["track_id", "class"]], on="track_id")
    fig = go.Figure()
    snaps = run.snapshots
    if snaps:
        img = Image.open(snaps[len(snaps) // 2])
        fig.add_layout_image(
            source=img,
            x=0,
            y=0,
            sizex=img.width,
            sizey=img.height,
            xref="x",
            yref="y",
            sizing="stretch",
            layer="below",
            opacity=0.55,
        )
        fig.update_xaxes(range=[0, img.width], showgrid=False, visible=False)
        fig.update_yaxes(range=[img.height, 0], showgrid=False, visible=False, scaleanchor="x")
    for (tid, cls), g in pts.groupby(["track_id", "class"]):
        fig.add_scatter(
            x=g["x"],
            y=g["y"],
            mode="lines",
            line=dict(width=1.6, color=CLASS_COLOURS.get(cls, "#888")),
            name=cls,
            legendgroup=cls,
            showlegend=False,
            hovertemplate=f"track {tid} · {cls}<extra></extra>",
        )
    fig.update_layout(height=470, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")
    source_note(f"track_points.csv · {len(sample_ids)} longest moving tracks over a mid-window frame")
with right:
    st.subheader("Direction of movement")
    moving = moving.assign(sector=(moving["heading_deg"] // 30 * 30).astype(int))
    rose = moving.groupby(["sector", "class"]).size().rename("tracks").reset_index()
    fig = px.bar_polar(rose, r="tracks", theta="sector", color="class", color_discrete_map=CLASS_COLOURS, height=330)
    fig.update_polars(angularaxis=dict(direction="counterclockwise", rotation=0))
    st.plotly_chart(fig, width="stretch")
    st.caption("Heading from first to last point: 0° = right, 90° = up the image.")
    st.subheader("Track duration")
    dur = moving.assign(seconds=moving["last_t_s"] - moving["first_t_s"])
    fig = px.histogram(dur, x="seconds", nbins=30, height=200, labels={"seconds": "Seconds tracked"})
    fig.update_traces(marker_color="#0F6E8C")
    st.plotly_chart(fig, width="stretch")

st.subheader("Track table")
st.dataframe(moving.sort_values("points", ascending=False), hide_index=True, width="stretch", height=280)
st.caption("Tracking accuracy (IDF1, MOTA, ID switches) needs ground-truth tracks; it will be measured on UA-DETRAC.")
