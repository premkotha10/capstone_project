from __future__ import annotations

import json
from collections import deque

import pandas as pd
import plotly.express as px
import psutil
import streamlit as st

from dashboard.common import kpi, select_run, setup, source_note
from src.common.config import Settings
from src.common.hardware import detect_hardware

setup("System Monitoring", "Throughput, latency per stage, machine load, errors and model provenance.")
settings = Settings()
run = select_run()

hw = detect_hardware()
c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi("CPU load", f"{psutil.cpu_percent(interval=0.3):.0f}%", f"{hw.cpu_logical_cores} logical cores")
with c2:
    mem = psutil.virtual_memory()
    kpi("Memory", f"{mem.percent:.0f}%", f"{mem.used / 1024**3:.1f} of {hw.ram_total_gb} GB")
with c3:
    kpi(
        "Accelerator",
        "CPU" if not (hw.cuda_available or hw.xpu_available) else "GPU",
        "no CUDA device; OpenVINO not installed" if not hw.cuda_available else ", ".join(hw.cuda_devices),
    )
with c4:
    disk = psutil.disk_usage(str(settings.data_root.anchor or "C:\\"))
    kpi("Disk free", f"{disk.free / 1024**3:.1f} GB", "data drive")

if run is not None:
    r = run.results
    st.subheader("Pipeline performance for this run")
    k1, k2, k3 = st.columns(3)
    with k1:
        kpi("Throughput", f"{r.get('throughput_fps', 0):.1f} fps", f"processed frames, stride {r.get('stride')}")
    with k2:
        kpi(
            "Real-time factor",
            f"{(r.get('throughput_fps', 0) * r.get('stride', 1)) / r.get('source_fps', 1):.2f}×",
            "≥ 1 means faster than the camera",
        )
    with k3:
        kpi("End-to-end p95", f"{r['latency_ms']['total']['p95_ms']:.0f} ms", "per processed frame")
    lat = pd.DataFrame(r["latency_ms"]).T.reset_index(names="stage")
    left, right = st.columns([1, 1.3])
    with left:
        st.dataframe(
            lat[["stage", "trials", "p50_ms", "p95_ms", "p99_ms", "max_ms"]].round(2), hide_index=True, width="stretch"
        )
        st.caption("Not a KPI-6 measurement: other jobs shared the CPU. KPI-6 uses a frozen, idle machine.")
    with right:
        fr = run.frames
        fr = fr.assign(minute=fr["t_s"] / 60)
        fig = px.line(
            fr.iloc[:: max(len(fr) // 1500, 1)],
            x="minute",
            y=["detect_ms", "total_ms"],
            height=260,
            labels={"value": "ms", "minute": "Minute", "variable": ""},
        )
        st.plotly_chart(fig, width="stretch")
    source_note("run.json latency_ms and frames.csv · time.perf_counter_ns")

    st.subheader("Model and configuration")
    m = run.record["model"]
    env = run.record["environment"]
    st.dataframe(
        pd.DataFrame(
            {
                "item": [
                    "Detector",
                    "Weights SHA-256",
                    "Device",
                    "Input size",
                    "Config hash",
                    "Code commit",
                    "Python",
                    "PyTorch",
                    "Ultralytics",
                ],
                "value": [
                    str(v)
                    for v in (
                        m["name"],
                        m["sha256"],
                        m["device"],
                        m["image_size"],
                        run.record["config_hash"],
                        env.get("git_commit"),
                        env.get("python"),
                        env["packages"].get("torch"),
                        env["packages"].get("ultralytics"),
                    )
                ],
            }
        ),
        hide_index=True,
        width="stretch",
    )

st.subheader("Recent warnings and errors")
rows = []
for log in sorted(settings.log_dir.glob("*.jsonl")):
    with log.open(encoding="utf-8", errors="replace") as fh:
        for line in deque(fh, maxlen=400):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("level") in {"WARNING", "ERROR", "CRITICAL"}:
                rows.append(
                    {
                        "time": rec.get("ts"),
                        "level": rec.get("level"),
                        "logger": rec.get("logger"),
                        "message": rec.get("msg"),
                        "file": log.name,
                    }
                )
if rows:
    st.dataframe(pd.DataFrame(rows).sort_values("time", ascending=False).head(50), hide_index=True, width="stretch")
else:
    st.success("No warnings or errors in the logs.")
source_note(f"{settings.log_dir}")
