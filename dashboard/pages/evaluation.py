from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.common import kpi, pill, setup, source_note
from src.common.config import Settings
from src.evaluation.reference_counts import (
    PLAN_PATH,
    compare_with_pipeline,
    consolidate,
    load_plan,
    load_records,
    score,
)
from src.services.results import RunStore

setup("Model Evaluation", "What has been measured, under which conditions, and what is still open.")
settings = Settings()

ref_records = load_records()
ref_table, agreement = consolidate(ref_records)
plan = load_plan() if PLAN_PATH.is_file() else []
comparison = compare_with_pipeline(ref_table, plan, RunStore().root) if not ref_table.empty else pd.DataFrame()
overall = score(comparison) if not comparison.empty else None
if overall is not None:
    counting_status = ("Development result", "info", f"{overall.clips} clip(s) with final hand counts.")
elif ref_records:
    counting_status = ("Counting in progress", "warn", "Hand counts exist but no clip has two agreeing counts yet.")
else:
    counting_status = (
        "Not measured",
        "bad",
        "No hand counts yet." if settings.hosted else "No hand counts yet: open Reference Counting.",
    )

st.subheader("Acceptance status")
rows = [
    ("O2 vs O3 comparison", "Not started", "warn", "O3 is built after O2 is frozen (Phase 9)."),
    ("Detection (P, R, mAP50, mAP50-95)", "Development run", "info", "UA-DETRAC subset, pretrained model, by weather."),
    ("Tracking (IDF1, MOTA, ID switches)", "Not measured", "warn", "Needs UA-DETRAC track evaluation (Phase 6)."),
    ("Counting (MAE, RMSE, accuracy)", *counting_status),
    (
        "Forecasting (MAE, RMSE by horizon)",
        "Baselines only",
        "info",
        "Volume backtests per run; congestion labels pending.",
    ),
    ("KPI-1 … KPI-6", "Not measured", "warn", "All targets are relative to the frozen O2 reference."),
]
for name, status, kind, note in rows:
    c1, c2, c3 = st.columns([1.3, 0.7, 2])
    c1.write(f"**{name}**")
    c2.markdown(pill(status, kind), unsafe_allow_html=True)
    c3.caption(note)

st.divider()
st.subheader("Counting accuracy against hand counts")
if not ref_records:
    st.info(
        "Counting accuracy is measured in the project against human hand counts (two independent counters per "
        "clip, docs/COUNTING_PROTOCOL.md). No hand counts have been entered yet."
        if settings.hosted
        else f"No hand counts yet. The plan has {len(plan)} clips; each needs two independent counts in "
        "**Reference Counting** (protocol: docs/COUNTING_PROTOCOL.md)."
    )
else:
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        kpi("Clips counted twice", f"{agreement.clips_double_counted if agreement else 0}", f"of {len(plan)} planned")
    with a2:
        kpi(
            "Annotator agreement",
            f"{agreement.exact_agreement:.0%}" if agreement else "—",
            "cells with identical counts",
        )
    with a3:
        kpi("Within ±1", f"{agreement.within_one:.0%}" if agreement else "—", "cells")
    with a4:
        kpi(
            "ICC(2,1)",
            f"{agreement.icc:.2f}" if agreement and agreement.icc == agreement.icc else "—",
            "absolute agreement",
        )
    status_counts = ref_table["status"].value_counts().to_dict()
    st.caption(
        "Cell status: "
        + ", ".join(f"{k} {v}" for k, v in status_counts.items())
        + ". Only agreed or adjudicated cells are used for accuracy."
    )
    if overall is None:
        st.warning("No cell has a final reference yet, so accuracy cannot be computed.")
    else:
        b1, b2, b3, b4 = st.columns(4)
        lo, hi = overall.accuracy_ci95
        with b1:
            kpi("Counting accuracy", f"{overall.accuracy:.1%}", f"95% CI {lo:.1%} – {hi:.1%} (clip bootstrap)")
        with b2:
            kpi("WAPE", f"{overall.wape:.1%}", "Σ|estimate − reference| / Σ reference")
        with b3:
            kpi("MAE per cell", f"{overall.mae:.2f}", f"RMSE {overall.rmse:.2f}")
        with b4:
            kpi("Vehicles", f"{overall.estimate_total} / {overall.reference_total}", "estimated / reference")
        st.caption("Accuracy = 1 − WAPE is the proposed KPI-1 definition (REQUIREMENTS §4), not yet frozen.")
        breakdown = []
        for (dim, key), g in [(("condition", c), g) for c, g in comparison.groupby("condition")] + [
            (("class", c), g) for c, g in comparison.groupby("vehicle_class")
        ]:
            s = score(g, n_boot=500)
            if s is not None:
                breakdown.append(
                    {
                        "by": dim,
                        "group": key,
                        "clips": s.clips,
                        "reference": s.reference_total,
                        "estimate": s.estimate_total,
                        "accuracy": round(s.accuracy, 3),
                        "ci95 low": round(s.accuracy_ci95[0], 3),
                        "ci95 high": round(s.accuracy_ci95[1], 3),
                        "MAE": round(s.mae, 2),
                    }
                )
        if breakdown:
            st.dataframe(pd.DataFrame(breakdown), hide_index=True, width="stretch")
            st.caption(
                "Condition-wise and class-wise results are listed separately so an aggregate cannot hide a "
                "failing condition or a rare class (NFR-4, NT-4)."
            )
        with st.expander("Cell-level comparison"):
            st.dataframe(comparison, hide_index=True, width="stretch")
    source_note("data/annotations/reference_counts/ (human) vs pipeline events.csv attributed to clips")

st.divider()
st.subheader("Detection on UA-DETRAC, by condition")
reports = sorted((settings.reports_root / "phase5").glob("detection_ua_detrac_*.json"))
if not reports:
    st.info(
        "The UA-DETRAC detection evaluation has not been run yet."
        + ("" if settings.hosted else " Run `python scripts/eval_detection_ua_detrac.py`.")
    )
else:
    rec = json.loads(reports[-1].read_text(encoding="utf-8"))
    by = pd.DataFrame(rec["results"]["by_condition"]).T.reset_index(names="condition")
    cols = ["condition", "images", "gt_objects", "precision", "recall", "ap50", "ap50_95", "small_recall"]
    st.dataframe(by[cols].round(3), hide_index=True, width="stretch")
    long = by.melt(id_vars="condition", value_vars=["ap50", "ap50_95", "recall"], var_name="metric")
    fig = px.bar(long, x="condition", y="value", color="metric", barmode="group", height=320, range_y=[0, 1])
    st.plotly_chart(fig, width="stretch")
    source_note(f"{reports[-1].name} · development data, not acceptance evidence")

st.divider()
st.subheader("Condition-wise comparison of processed runs")
st.caption(
    "Observational only: these show how the pipeline behaves at night versus by day. They are not accuracy "
    "figures because there are no reference counts yet."
)
runs = [r for r in RunStore().list() if r.name != "smoke"]
if runs:
    table = []
    for r in runs:
        res = r.results
        cond = res.get("condition_share", {})
        table.append(
            {
                "run": r.name,
                "minutes": round(res.get("duration_s", 0) / 60, 1),
                "vehicles counted": res.get("count_events"),
                "tracks": res.get("tracks"),
                "degraded frames": res.get("degraded_frame_share"),
                "night share": round(sum(v for k, v in cond.items() if "night" in k), 3),
                "dense-occlusion share": round(sum(v for k, v in cond.items() if "dense" in k), 3),
                "median confidence": round(r.detections["confidence"].median(), 3) if not r.detections.empty else None,
                "p95 latency ms": round(res["latency_ms"]["total"]["p95_ms"], 1),
            }
        )
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
