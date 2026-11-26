"""Export the project's current status as small JSON files for the status dashboard.

Usage:  python scripts/export_status_data.py [--junit PATH] [--detection-run PATH]

Everything is read from project artefacts, never typed in by hand:
  phases.json          DEVELOPMENT_ROADMAP.md phase table
  ua_detrac.json       UA-DETRAC subset config + parsed XML statistics + whether frames are on disk (manifest)
  barbados.json        Barbados subset windows x cameras: clips selected in the manifest
  detection_run.json   one detection run record (per-class boxes, latency percentiles)
  tests.json           pytest JUnit XML summary per suite
Output: reports/status/*.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import PROJECT_ROOT, Settings  # noqa: E402
from src.datasets import ua_detrac  # noqa: E402
from src.datasets.manifest import DatasetManifest  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "status"
STATUS_WORDS = {"✅": "done", "🟡": "in progress", "⬜": "not started", "⏸": "awaiting approval"}


def phases() -> list[dict]:
    rows = []
    for line in (PROJECT_ROOT / "DEVELOPMENT_ROADMAP.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|[^|]*\|[^|]*\|\s*([^|]+?)\s*\|$", line)
        if not m:
            continue
        status_cell = m.group(3)
        status = next((word for icon, word in STATUS_WORDS.items() if status_cell.startswith(icon)), "unknown")
        if "⏸" in status_cell and status_cell.startswith("🟡"):
            status = "drafted, awaiting review"
        gate = {1: "O1", 2: "O1", 3: "O1", 4: "O1", 5: "O2", 6: "O2", 7: "O2", 8: "O2", 9: "O3"}.get(int(m.group(1)))
        gate = gate or ("O4" if int(m.group(1)) <= 13 else "O5")
        rows.append({"phase": int(m.group(1)), "title": m.group(2), "objective": gate, "status": status})
    return rows


def ua_detrac_rows(settings: Settings) -> list[dict]:
    spec = yaml.safe_load((PROJECT_ROOT / "configs" / "data" / "ua_detrac_subset_v1.yaml").read_text(encoding="utf-8"))
    manifest = DatasetManifest.load(PROJECT_ROOT / "data" / "manifests" / "ua_detrac.yaml")
    on_disk = {i.id for i in manifest.items if i.attributes.get("weather")}
    stats: dict[str, tuple[str, ua_detrac.SequenceAnnotation]] = {}
    ann = settings.data_root / "raw" / "ua_detrac" / "annotations"
    for zip_name, split in (("DETRAC-Train-Annotations-XML.zip", "train"), ("DETRAC-Test-Annotations-XML.zip", "test")):
        with zipfile.ZipFile(ann / zip_name) as zf:
            for member in zf.namelist():
                if member.endswith(".xml"):
                    seq = ua_detrac.parse_sequence_xml(zf.read(member))
                    stats[seq.name] = (split, seq)
    rows = []
    for _condition, names in spec["sequences"].items():
        for name in names:
            split, seq = stats[name]
            rows.append(
                {
                    "sequence": name,
                    "weather": seq.weather,
                    "original_split": split,
                    "frames": seq.num_frames,
                    "vehicles": len(seq.track_ids),
                    "mean_density": round(seq.mean_density, 1),
                    "occluded_pct": round(100 * seq.occluded_fraction, 1),
                    "downloaded": "yes" if name in on_disk else "pending",
                }
            )
    return rows


def barbados_rows() -> list[dict]:
    spec = yaml.safe_load((PROJECT_ROOT / "configs" / "data" / "barbados_subset_v1.yaml").read_text(encoding="utf-8"))
    manifest = DatasetManifest.load(PROJECT_ROOT / "data" / "manifests" / "barbados_traffic.yaml")
    counts = Counter((i.attributes.get("window"), i.attributes.get("camera_id")) for i in manifest.items)
    offset = timedelta(hours=spec["local_offset_hours"])
    rows = []
    for w in spec["windows"]:
        start_local = datetime.fromisoformat(w["start"]) + offset
        for cam in spec["cameras"]:
            rows.append(
                {
                    "window": w["name"],
                    "local_start": start_local.strftime("%d %b %H:%M"),
                    "condition": w["condition_hint"],
                    "camera": cam.replace("normanniles", "Norman Niles #"),
                    "clips_planned": w["minutes"],
                    "clips_downloaded": counts.get((w["name"], cam), 0),
                }
            )
    return rows


def detection_rows(run_path: Path) -> dict:
    run = json.loads(run_path.read_text(encoding="utf-8"))
    res = run["results"]
    lat = res["detector_latency_ms"]
    clip = Path(run["inputs"][0]["video"]).stem
    camera, ts = clip.split("_", 1)
    local = datetime.strptime(ts, "%Y-%m-%d-%H-%M-%S") - timedelta(hours=4)
    return {
        "rows": [
            {"measure": "boxes_" + cls, "label": cls, "value": n}
            for cls, n in sorted(res["detections_per_class"].items())
        ]
        + [
            {"measure": "latency_p50", "label": "p50", "value": round(lat["p50_ms"], 1)},
            {"measure": "latency_p95", "label": "p95", "value": round(lat["p95_ms"], 1)},
            {"measure": "latency_p99", "label": "p99", "value": round(lat["p99_ms"], 1)},
            {"measure": "frames", "label": "frames processed", "value": res["frames_processed"]},
            {"measure": "throughput_fps", "label": "throughput fps", "value": res["throughput_fps"]},
        ],
        "meta": [
            {
                "label": "Clip",
                "value": f"{camera.replace('normanniles', 'Norman Niles #')}, {local:%d %b %Y %H:%M} local",
            },
            {"label": "Model", "value": f"{run['model']['name']} (sha256 {run['model']['sha256'][:12]}…)"},
            {"label": "Run finished (UTC)", "value": run["environment"]["timestamp_utc"]},
            {"label": "Config hash", "value": run["config_hash"]},
            {"label": "Frame stride", "value": str(res["stride"])},
            {"label": "Caveat", "value": "Measured while two data downloads were using the CPU; not a KPI-6 value."},
        ],
    }


def test_rows(junit: Path) -> list[dict]:
    root = ET.parse(junit).getroot()
    suites = Counter()
    failed = Counter()
    skipped = Counter()
    for case in root.iter("testcase"):
        suite = case.attrib.get("classname", "").split(".")[1] if "." in case.attrib.get("classname", "") else "other"
        suites[suite] += 1
        if case.find("failure") is not None or case.find("error") is not None:
            failed[suite] += 1
        if case.find("skipped") is not None:
            skipped[suite] += 1
    return [
        {"suite": s, "tests": n, "passed": n - failed[s] - skipped[s], "failed": failed[s], "skipped": skipped[s]}
        for s, n in sorted(suites.items())
    ]


def main() -> int:
    settings = Settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--junit", type=Path, default=settings.log_dir / "pytest_junit.xml")
    parser.add_argument(
        "--detection-run",
        type=Path,
        default=settings.data_root / "runs/detection/normanniles1_2025-10-21-13-13-45/run.json",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    outputs = {
        "phases.json": phases(),
        "ua_detrac.json": ua_detrac_rows(settings),
        "barbados.json": barbados_rows(),
        "detection_run.json": detection_rows(args.detection_run),
        "tests.json": test_rows(args.junit),
    }
    for name, data in outputs.items():
        (OUT / name).write_text(json.dumps(data, indent=1), encoding="utf-8")
        print(name, len(data) if isinstance(data, list) else len(data["rows"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
