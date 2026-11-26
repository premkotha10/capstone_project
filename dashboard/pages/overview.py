from __future__ import annotations

from collections import Counter

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.common import (
    CLASS_COLOURS,
    estimate_banner,
    kpi,
    movement_label,
    pill,
    select_run,
    setup,
    source_note,
)
from src.common.config import Settings
from src.forecasting.baselines import DEFAULT_MODELS, forecast_ahead, rolling_origin_backtest

setup("Traffic Overview", "Roadside video turned into vehicle counts, conditions and a short-term volume outlook.")
run = select_run()
if run is None:
    st.stop()
estimate_banner()
r = run.results
counts = r.get("counts_by_line_direction", {})
total = sum(counts.values())
minutes = max(r.get("duration_s", 0) / 60, 1e-9)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    kpi("Vehicles counted", f"{total:,}", f"over {minutes:.0f} min of video")
with c2:
    kpi("Flow rate", f"{total / minutes * 60:,.0f}", "vehicles per hour, all movements")
with c3:
    kpi("Entering roundabout", f"{counts.get('approach:entering', 0):,}", "approach line, downward")
with c4:
    kpi("Leaving roundabout", f"{counts.get('approach:exiting', 0):,}", "approach line, upward")
with c5:
    kpi("Tracks", f"{r.get('tracks', 0):,}", f"{r.get('frames_processed', 0):,} frames processed")

st.write("")
left, mid, right = st.columns([1.1, 1.6, 1.1])

with left:
    st.subheader("Vehicle mix")
    by_class = r.get("counts_by_class", {})
    if by_class:
        fig = go.Figure(
            go.Pie(
                labels=list(by_class),
                values=list(by_class.values()),
                hole=0.62,
                sort=False,
                marker=dict(colors=[CLASS_COLOURS.get(c, "#8A8F98") for c in by_class]),
                textinfo="label+percent",
            )
        )
        fig.update_layout(
            showlegend=False,
            height=290,
            annotations=[dict(text=f"<b>{total:,}</b><br>counted", showarrow=False, font_size=15)],
        )
        st.plotly_chart(fig, width="stretch")
    source_note("events.csv · class = confidence-weighted majority over each track")

with mid:
    st.subheader("Traffic volume per minute")
    iv = run.intervals
    if not iv.empty:
        per = iv.groupby(["minute", "movement"], as_index=False)["count"].sum()
        per["movement"] = per["movement"].map(movement_label)
        fig = px.bar(
            per,
            x="minute",
            y="count",
            color="movement",
            height=290,
            labels={"minute": "Minute of window", "count": "Vehicles", "movement": ""},
        )
        degraded = iv[iv["reliability"] != "ok"]["minute"].unique()
        for m in degraded:
            fig.add_vrect(x0=m - 0.5, x1=m + 0.5, fillcolor="#E0A100", opacity=0.12, line_width=0)
        st.plotly_chart(fig, width="stretch")
        source_note("intervals.csv · shaded minutes contain counts made under a degraded condition flag")

with right:
    st.subheader("Measured condition")
    fr = run.frames
    if not fr.empty:
        last = fr[fr["t_s"] >= fr["t_s"].max() - 60]
        dominant = Counter(last["condition"]).most_common(1)[0][0]
        degraded_share = (last["reliability"] != "ok").mean()
        kind = "ok" if dominant == "nominal" else "warn"
        st.markdown(pill(dominant.replace("|", " + ").replace("_", " "), kind), unsafe_allow_html=True)
        st.metric("Frames flagged in last minute", f"{degraded_share:.0%}")
        st.metric("Mean brightness (0–255)", f"{last['luma'].mean():.0f}")
        st.caption("Night, low contrast, blur or dense occlusion mark results as degraded (provisional thresholds).")

st.divider()
f1, f2 = st.columns([1.6, 1.4])
with f1:
    st.subheader("Next 5 minutes: volume outlook")
    movement = "approach:entering" if "approach:entering" in counts else (next(iter(counts), None))
    series = run.volume_series(movement)
    if movement and len(series) >= 6:
        y = series["count"].tolist()
        bt = rolling_origin_backtest(y, horizons=(1,))
        best = min(bt, key=lambda b: b.mae)
        preds = forecast_ahead(y, DEFAULT_MODELS[best.model], range(1, 6))
        fig = go.Figure()
        fig.add_scatter(
            x=series["minute"], y=y, mode="lines+markers", name="Estimated (observed video)", line=dict(color="#0F6E8C")
        )
        fut = [series["minute"].max() + h for h in range(1, 6)]
        fig.add_scatter(
            x=[series["minute"].max(), *fut],
            y=[y[-1], *preds],
            mode="lines+markers",
            name=f"Predicted ({best.model})",
            line=dict(color="#D55E00", dash="dash"),
        )
        fig.update_layout(height=280, xaxis_title="Minute", yaxis_title="Vehicles / min")
        st.plotly_chart(fig, width="stretch")
        source_note(
            f"{movement_label(movement)} · best baseline by 1-minute backtest MAE ({best.mae:.2f} veh/min, "
            f"n={best.n}). Illustrative: {len(y)} points only."
        )
    else:
        st.info("Not enough minutes in this run for a forecast.")
with f2:
    st.subheader("Congestion level")
    st.markdown(pill("Not available yet", "warn"), unsafe_allow_html=True)
    st.write(
        "Congestion classes (free flowing → heavy delay) come from the Barbados challenge labels in "
        "`Train.csv`, which need a Zindi sign-in. Until then the dashboard shows volume forecasts only and "
        "invents no congestion labels."
    )
    st.subheader("System status")
    s = Settings()
    checks = {
        "Detector weights": (s.models_root / "detection" / "yolo11n.pt").is_file(),
        "Data folder": s.data_root.is_dir(),
        "Annotated video": run.video is not None,
        "Reference counts": False,
    }
    st.markdown(" ".join(pill(k, "ok" if v else "bad") for k, v in checks.items()), unsafe_allow_html=True)
    st.caption("Reference counts are still to be hand-annotated, so accuracy cannot be reported yet.")
