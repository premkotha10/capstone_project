"""Shared dashboard helpers: styling, run selection, provenance labels, chart theme."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import plotly.graph_objects as go  # noqa: E402
import plotly.io as pio  # noqa: E402
import streamlit as st  # noqa: E402

from src.common.config import Settings  # noqa: E402
from src.services.results import PipelineRun, RunStore  # noqa: E402

CLASS_COLOURS = {
    "car": "#0072B2",
    "truck": "#CC79A7",
    "bus": "#009E73",
    "motorcycle": "#E69F00",
    "bicycle": "#56B4E9",
    "van": "#D55E00",
    "other": "#8A8F98",
}
MOVEMENT_LABELS = {
    "approach:entering": "Approach · entering roundabout",
    "approach:exiting": "Approach · leaving roundabout",
    "circulating_west:eastbound": "Ring road · eastbound",
    "circulating_west:westbound": "Ring road · westbound",
}

pio.templates["dsci41"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Inter, Segoe UI, sans-serif", size=13, color="#14212B"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#DDE3E8", zerolinecolor="#C9D1D8"),
        yaxis=dict(gridcolor="#DDE3E8", zerolinecolor="#C9D1D8"),
        margin=dict(l=10, r=10, t=40, b=10),
        colorway=list(CLASS_COLOURS.values()),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
)
pio.templates.default = "plotly_white+dsci41"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
h1, h2, h3 { font-family: 'Space Grotesk', 'Inter', sans-serif !important; letter-spacing: -0.01em; }
.block-container { padding-top: 1.6rem; max-width: 1400px; }
.kpi { background: #FFFFFF; border: 1px solid #DDE3E8; border-radius: 10px; padding: 14px 16px; height: 100%; }
.kpi .label { color: #5B6B78; font-size: 0.78rem; text-transform: uppercase; letter-spacing: .05em; }
.kpi .value { font-family: 'Space Grotesk', sans-serif; font-size: 1.9rem; font-weight: 600; line-height: 1.2;
              font-variant-numeric: tabular-nums; color: #14212B; }
.kpi .sub { color: #5B6B78; font-size: 0.8rem; }
.pill { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.78rem; font-weight: 500; }
.pill-ok { background: #E3F4EA; color: #17663A; }
.pill-warn { background: #FFF1D6; color: #8A5300; }
.pill-bad { background: #FBE3E3; color: #9B1C1C; }
.pill-info { background: #E2EFF5; color: #0F4E66; }
.banner { border-left: 4px solid #E0A100; background: #FFF8E6; padding: 8px 14px; border-radius: 6px;
          color: #5C4300; font-size: 0.86rem; margin-bottom: 0.6rem; }
.src { color: #5B6B78; font-size: 0.75rem; }
</style>
"""


def setup(title: str, subtitle: str | None = None) -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.title(title)
    if subtitle:
        st.caption(subtitle)


@st.cache_data(ttl=30, show_spinner=False)
def _run_names() -> list[str]:
    return [r.name for r in RunStore().list()]


@st.cache_resource(show_spinner=False)
def _load(name: str, mtime: float) -> PipelineRun:  # mtime busts the cache when a run is rewritten
    return RunStore().get(name)


def select_run(key: str = "run") -> PipelineRun | None:
    names = [n for n in _run_names() if n != "smoke"] or _run_names()
    if not names:
        hosted = Settings().hosted
        st.info(
            "No processed runs yet. Open **Video Analytics** to "
            + ("upload a video." if hosted else "process a sample clip or upload a video.")
        )
        return None
    busiest = max(names, key=lambda n: RunStore().get(n).results.get("count_events") or 0)
    default = st.session_state.get("selected_run", busiest)
    name = st.sidebar.selectbox(
        "Processed run", names, index=names.index(default) if default in names else 0, key=f"{key}_select"
    )
    st.session_state["selected_run"] = name
    path = RunStore().root / name / "run.json"
    run = _load(name, path.stat().st_mtime)
    r = run.results
    st.sidebar.markdown(
        f"<div class='src'>Camera <b>{r.get('camera_id')}</b> · {r.get('clips')} clip(s) · "
        f"{r.get('duration_s', 0) / 60:.1f} min · stride {r.get('stride')}<br>"
        f"Config hash <code>{run.record.get('config_hash')}</code></div>",
        unsafe_allow_html=True,
    )
    return run


def estimate_banner() -> None:
    st.markdown(
        "<div class='banner'><b>Estimated values.</b> Counts come from the O2-A reference pipeline "
        "(YOLO11n + ByteTrack "
        "+ image-space lines) and have <b>not yet been validated against reference counts</b>. Condition flags use "
        "provisional thresholds.</div>",
        unsafe_allow_html=True,
    )


def kpi(label: str, value: str, sub: str = "") -> None:
    st.markdown(
        f"<div class='kpi'><div class='label'>{label}</div><div class='value'>{value}</div>"
        f"<div class='sub'>{sub}</div></div>",
        unsafe_allow_html=True,
    )


def pill(text: str, kind: str = "info") -> str:
    return f"<span class='pill pill-{kind}'>{text}</span>"


def source_note(text: str) -> None:
    st.markdown(f"<div class='src'>Source: {text}</div>", unsafe_allow_html=True)


def movement_label(movement: str) -> str:
    if movement in MOVEMENT_LABELS:
        return MOVEMENT_LABELS[movement]
    line, _, direction = movement.partition(":")
    return f"{line.replace('_', ' ').capitalize()} · {direction.replace('_', ' ')}".strip(" ·")
