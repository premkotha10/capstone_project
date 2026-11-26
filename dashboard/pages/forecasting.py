from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.common import estimate_banner, kpi, movement_label, pill, select_run, setup, source_note
from src.forecasting.baselines import DEFAULT_MODELS, forecast_ahead, rolling_origin_backtest

setup(
    "Congestion Forecasting",
    "Short-term outlook from estimated volumes. Observed, estimated and predicted are kept apart.",
)
run = select_run()
if run is None:
    st.stop()
estimate_banner()
iv = run.intervals
if iv.empty:
    st.info("This run has no interval counts.")
    st.stop()

c1, c2 = st.columns([1, 1])
movement = c1.selectbox("Movement", sorted(iv["movement"].unique()), format_func=movement_label)
horizon = c2.slider("Forecast horizon (minutes)", 1, 10, 5)
series = run.volume_series(movement)
y = series["count"].tolist()

if len(y) < 6:
    st.warning("At least 6 minutes are needed for a forecast.")
    st.stop()

bt = pd.DataFrame([b.__dict__ for b in rolling_origin_backtest(y, horizons=sorted({1, 2, 3, min(5, horizon)}))])
best_name = bt[bt["horizon"] == 1].sort_values("mae").iloc[0]["model"]
preds = forecast_ahead(y, DEFAULT_MODELS[best_name], range(1, horizon + 1))

k1, k2, k3 = st.columns(3)
with k1:
    kpi("Last minute (estimated)", f"{y[-1]:,.0f}", "vehicles")
with k2:
    kpi(f"Next {horizon} min (predicted)", f"{sum(preds):,.0f}", f"{best_name}")
with k3:
    best_mae = bt[(bt["model"] == best_name) & (bt["horizon"] == 1)]["mae"].iloc[0]
    kpi("Backtest error", f"{best_mae:.2f}", "MAE, vehicles/min, 1-min horizon")

fig = go.Figure()
fig.add_scatter(x=series["minute"], y=y, mode="lines+markers", name="Estimated from video", line=dict(color="#0F6E8C"))
deg = series[series["degraded"]]
if not deg.empty:
    fig.add_scatter(
        x=deg["minute"],
        y=deg["count"],
        mode="markers",
        name="Degraded-condition minute",
        marker=dict(color="#E0A100", size=11, symbol="diamond-open", line=dict(width=2)),
    )
last = series["minute"].max()
fig.add_scatter(
    x=[last, *[last + h for h in range(1, horizon + 1)]],
    y=[y[-1], *preds],
    mode="lines+markers",
    name=f"Predicted ({best_name})",
    line=dict(color="#D55E00", dash="dash"),
)
fig.add_vrect(x0=last + 0.5, x1=last + horizon + 0.5, fillcolor="#D55E00", opacity=0.06, line_width=0)
fig.update_layout(height=360, xaxis_title="Minute", yaxis_title="Vehicles per minute")
st.plotly_chart(fig, width="stretch")
source_note(f"intervals.csv → {movement_label(movement)}; models in src/forecasting/baselines.py")

left, right = st.columns([1.2, 1])
with left:
    st.subheader("Baseline comparison (rolling-origin backtest)")
    fig = px.line(
        bt,
        x="horizon",
        y="mae",
        color="model",
        markers=True,
        height=300,
        labels={"horizon": "Horizon (minutes)", "mae": "MAE (vehicles/min)"},
    )
    st.plotly_chart(fig, width="stretch")
    st.dataframe(bt.round(3), hide_index=True, width="stretch")
with right:
    st.subheader("Congestion classification")
    st.markdown(pill("Waiting for labels", "warn"), unsafe_allow_html=True)
    st.write(
        "The target is the Barbados challenge's 4-class congestion rating, predicted 5 minutes ahead after a "
        "15-minute input and 2-minute delay. Those labels are in `Train.csv` (Zindi sign-in needed). "
        "No congestion level is shown until real labels are loaded."
    )
    st.subheader("Why only baselines")
    st.write(
        f"This window has {len(y)} one-minute points. Random Forest and XGBoost need far more history; they "
        "are added in Phase 10 and must beat these baselines to be kept."
    )
    st.caption("Forecast confidence intervals are shown only once their coverage has been checked.")
