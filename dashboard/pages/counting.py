from __future__ import annotations

import plotly.express as px
import streamlit as st

from dashboard.common import CLASS_COLOURS, estimate_banner, kpi, movement_label, select_run, setup, source_note

setup("Traffic Counting", "Line crossings by movement, direction and vehicle class, aggregated per minute.")
run = select_run()
if run is None:
    st.stop()
estimate_banner()
ev, iv = run.events, run.intervals
if ev.empty:
    st.info("This run has no counting lines or no crossings. Process it with a camera configuration to count.")
    st.stop()
ev["movement"] = ev["line"] + ":" + ev["direction"]

movements = sorted(ev["movement"].unique())
cols = st.columns(len(movements) + 1)
with cols[0]:
    kpi("Total counted", f"{len(ev):,}", f"{run.results.get('duration_s', 0) / 60:.0f} min window")
for col, m in zip(cols[1:], movements, strict=True):
    with col:
        n = int((ev["movement"] == m).sum())
        kpi(movement_label(m), f"{n:,}", f"{n / max(run.results.get('duration_s', 1) / 3600, 1e-9):,.0f} veh/h")

left, right = st.columns([1.6, 1])
with left:
    st.subheader("Counts per minute by class")
    pick = st.selectbox(
        "Movement",
        ["All movements", *movements],
        format_func=lambda m: m if m == "All movements" else movement_label(m),
    )
    data = iv if pick == "All movements" else iv[iv["movement"] == pick]
    per = data.groupby(["minute", "class"], as_index=False)["count"].sum()
    fig = px.bar(
        per,
        x="minute",
        y="count",
        color="class",
        color_discrete_map=CLASS_COLOURS,
        height=340,
        labels={"minute": "Minute", "count": "Vehicles"},
    )
    for m in data[data["reliability"] != "ok"]["minute"].unique():
        fig.add_vrect(x0=m - 0.5, x1=m + 0.5, fillcolor="#E0A100", opacity=0.12, line_width=0)
    st.plotly_chart(fig, width="stretch")
    source_note("intervals.csv · CountSource = estimated · shaded = minute contains degraded-condition counts")
with right:
    st.subheader("By class and movement")
    table = ev.pivot_table(index="class", columns="movement", values="track_id", aggfunc="count", fill_value=0)
    table.columns = [movement_label(c) for c in table.columns]
    table["Total"] = table.sum(axis=1)
    st.dataframe(table.sort_values("Total", ascending=False), width="stretch")
    rare = table[table["Total"] < 0.05 * table["Total"].sum()]
    if not rare.empty:
        st.caption(f"Rare classes ({', '.join(rare.index)}) are listed separately so the aggregate cannot hide them.")
    st.subheader("Reliability of counts")
    rel = ev["reliability"].value_counts()
    st.dataframe(rel.rename_axis("flag").reset_index(name="crossings"), hide_index=True, width="stretch")

st.subheader("Crossing events")
st.dataframe(
    ev.assign(minute=(ev["t_s"] // 60).astype(int)).drop(columns=["movement"]),
    hide_index=True,
    width="stretch",
    height=260,
)
st.caption(
    "Each track is counted at most once per line. Known risk: an ID switch inside a queue at the line can "
    "double-count. Accuracy will be measured against hand counts before any figure is reported as validated."
)
