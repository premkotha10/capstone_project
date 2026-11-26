"""DSCI-41 Traffic Intelligence dashboard.

Run:  streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

import src  # noqa: E402,F401  (sets YOLO_AUTOINSTALL=false before any Ultralytics import)
from src.common.config import Settings  # noqa: E402

st.set_page_config(page_title="DSCI-41 Traffic Intelligence", page_icon="🚦", layout="wide")

PAGES = Path(__file__).parent / "pages"
navigation = st.navigation(
    {
        "Operations": [
            st.Page(PAGES / "overview.py", title="Overview", icon=":material/dashboard:", default=True),
            st.Page(PAGES / "video_analytics.py", title="Video Analytics", icon=":material/videocam:"),
        ],
        "Analysis": [
            st.Page(PAGES / "detection.py", title="Vehicle Detection", icon=":material/center_focus_strong:"),
            st.Page(PAGES / "tracking.py", title="Vehicle Tracking", icon=":material/route:"),
            st.Page(PAGES / "counting.py", title="Traffic Counting", icon=":material/bar_chart:"),
            st.Page(PAGES / "forecasting.py", title="Congestion Forecasting", icon=":material/trending_up:"),
        ],
        "Assurance": [
            st.Page(PAGES / "evaluation.py", title="Model Evaluation", icon=":material/fact_check:"),
            *(
                []
                if Settings().hosted
                else [st.Page(PAGES / "reference_counting.py", title="Reference Counting", icon=":material/edit_note:")]
            ),
            st.Page(PAGES / "monitoring.py", title="System Monitoring", icon=":material/monitor_heart:"),
        ],
    }
)
st.sidebar.caption("DSCI-41 · KL University capstone")
navigation.run()
