from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import streamlit as st

from dashboard.common import select_run, setup, source_note
from src.common.config import PROJECT_ROOT, Settings, load_config
from src.common.hardware import select_device
from src.counting.line_counter import CameraGeometry
from src.ingestion.video import IngestionError, probe_video
from src.pipeline.runner import process_clips

settings = Settings()
HOSTED = settings.hosted
setup(
    "Video Analytics",
    "Upload a short traffic video, place a counting line, and process it."
    if HOSTED
    else "Process a sample window or your own video, then watch and download the results.",
)
CAMERAS = sorted(p.stem for p in (PROJECT_ROOT / "configs" / "cameras").glob("*.yaml"))
CUSTOM = "custom line (place it below)"
NONE = "none (detect + track only)"


@st.cache_resource(show_spinner="Loading detector…")
def detector(model: str):  # noqa: ANN201
    from src.detection.detector import UltralyticsDetector

    cfg = load_config(PROJECT_ROOT / "configs" / "o2_a.yaml")
    return UltralyticsDetector(cfg.detection, settings.models_root, device=select_device(settings.device))


def first_frame(path: Path):  # noqa: ANN201
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


tab_run, tab_view = st.tabs(["Process video", "View results"])

with tab_run:
    if HOSTED:
        st.info(
            f"Public demo on a shared CPU. Videos up to {settings.hosted_max_upload_mb} MB; only the first "
            f"{settings.hosted_max_seconds:.0f} s are processed, downscaled to {settings.hosted_max_width} px wide. "
            "Uploads are processed on this server and deleted when it restarts. Only upload footage you are allowed "
            "to share, and avoid identifiable people."
        )
        source = "Upload a video"
    else:
        source = st.radio("Input", ["Sample clips (Barbados, local copy)", "Upload a video"], horizontal=True)
    clips: list[Path] = []
    camera_id = "uploaded"
    if source.startswith("Sample"):
        raw = settings.data_root / "raw" / "barbados_traffic"
        cams = sorted(p.name for p in raw.iterdir() if p.is_dir()) if raw.is_dir() else []
        if not cams:
            st.warning("No sample clips found under the data folder.")
        else:
            c1, c2, c3 = st.columns(3)
            camera_id = c1.selectbox("Camera", cams)
            files = sorted((raw / camera_id).glob("*.mp4"))

            def local(p: Path) -> str:
                ts = datetime.strptime(
                    re.search(r"_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})", p.name).group(1), "%Y-%m-%d-%H-%M-%S"
                )
                return (ts - timedelta(hours=4)).strftime("%d %b %H:%M")

            start = c2.selectbox("Start (local time)", files, format_func=local)
            n = c3.number_input("Minutes", 1, 23, 2)
            i = files.index(start)
            clips = files[i : i + int(n)]
    else:
        up = st.file_uploader("MP4 or AVI", type=["mp4", "avi"])
        if up is not None:
            if HOSTED and up.size > settings.hosted_max_upload_mb * 1024 * 1024:
                st.error(
                    f"This file is {up.size / 1024**2:.0f} MB; the limit here is {settings.hosted_max_upload_mb} MB."
                )
            else:
                key = f"upload_{up.file_id}"
                if key not in st.session_state:  # save once per upload, under a server-chosen name
                    suffix = Path(up.name).suffix.lower()
                    dest = settings.data_root / "uploads" / f"{uuid.uuid4().hex[:12]}{suffix}"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(up.getbuffer())
                    st.session_state[key] = str(dest)
                clips = [Path(st.session_state[key])]

    geometry_options = [CUSTOM, NONE, *([] if HOSTED else CAMERAS)]
    default = geometry_options.index(camera_id) if camera_id in geometry_options else 0
    c1, c2, c3 = st.columns(3)
    geometry_choice = c1.selectbox("Counting lines", geometry_options, index=default)
    stride = c2.select_slider("Frame stride", [1, 2, 3, 4], value=2, help="Process every n-th frame")
    name = c3.text_input("Run name", value=f"{camera_id}_{datetime.now():%H%M%S}", max_chars=60)
    name = re.sub(r"[^A-Za-z0-9_-]", "_", name) or "run"

    geometry = None
    meta = None
    if clips:
        try:
            meta = probe_video(clips[0])
        except IngestionError as exc:
            st.error(f"This video cannot be read: {exc}")
            clips = []
    if clips and meta is not None and geometry_choice == CUSTOM:
        g1, g2 = st.columns([1, 2])
        orientation = g1.radio(
            "Line direction",
            ["horizontal", "vertical"],
            horizontal=True,
            help="Horizontal counts traffic moving up/down the image; vertical counts left/right.",
        )
        position = (
            g2.slider(
                "Line position (% of frame)",
                5,
                95,
                65,
                help="From the top for a horizontal line, from the left for a vertical one. "
                "Place it where vehicles are fully visible and separated.",
            )
            / 100
        )
        geometry = CameraGeometry.custom_line((meta.width, meta.height), orientation, position)
        frame = first_frame(clips[0])
        if frame is not None:
            line = geometry.lines[0]
            preview = frame.copy()
            t = max(int(preview.shape[1] / 320), 2)
            cv2.line(preview, tuple(int(v) for v in line.p1), tuple(int(v) for v in line.p2), (0, 215, 255), t)
            k = min(1.0, 640 / preview.shape[1])
            preview = cv2.resize(preview, (int(preview.shape[1] * k), int(preview.shape[0] * k)))
            st.image(
                cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                caption="First frame with your counting line",
                width=min(640, preview.shape[1]),
            )
    elif geometry_choice not in (CUSTOM, NONE):
        geometry = CameraGeometry.load(PROJECT_ROOT / "configs" / "cameras" / f"{geometry_choice}.yaml")

    max_frames = None
    if HOSTED and meta is not None:
        max_frames = max(int(settings.hosted_max_seconds * meta.fps / stride), 1)
    st.caption(
        "Processing runs inside this page: stay on it until it finishes. Switching pages or changing a "
        "setting stops the run." + ("" if HOSTED else " For long windows use scripts/process_video.py.")
    )
    if st.button("Process", type="primary", disabled=not clips):
        bar = st.progress(0.0, text="Starting…")

        def progress(done: int, total: int, message: str) -> None:
            bar.progress(min(done / max(total, 1), 1.0), text=f"{done:,}/{total:,} frames · {message}")

        try:
            summary = process_clips(
                clips,
                load_config(PROJECT_ROOT / "configs" / "o2_a.yaml"),
                detector("yolo11n"),
                settings.data_root / "runs" / "pipeline" / name,
                geometry=geometry,
                camera_id=geometry.camera_id if geometry else camera_id,
                stride=stride,
                max_frames=max_frames,
                progress=progress,
                run_name=name,
                max_width=settings.hosted_max_width if HOSTED else None,
            )
        except ValueError as exc:
            bar.empty()
            st.error(f"Could not process the video: {exc}")
        else:
            bar.progress(1.0, text="Done")
            st.success(
                f"{summary.frames:,} frames, {summary.tracks:,} tracks, {summary.events:,} vehicles counted "
                f"in {summary.wall_s:.0f} s. Open **View results**, or pick **{name}** in the sidebar on any page."
            )
            st.session_state["selected_run"] = name
            st.cache_data.clear()

with tab_view:
    run = select_run("va")
    if run is not None:
        if run.video:
            st.video(str(run.video))
            source_note("annotated.mp4 · boxes, track IDs, trails, counting lines and running totals")
        else:
            st.info("This run has no annotated video.")
        st.subheader("Downloads")
        cols = st.columns(5)
        for col, fname in zip(
            cols, ["intervals.csv", "events.csv", "tracks.csv", "detections.csv", "run.json"], strict=True
        ):
            p = run.path / fname
            if p.is_file():
                col.download_button(fname, p.read_bytes(), file_name=f"{run.name}_{fname}", width="stretch")
        if not HOSTED:
            st.caption(
                "Barbados video carries an IWCP re-use notice: keep annotated videos local, do not publish them."
            )
