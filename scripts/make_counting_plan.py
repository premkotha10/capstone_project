"""Generate the reference-counting sampling plan (docs/COUNTING_PROTOCOL.md §7). Deterministic for a given seed.

Usage:  python scripts/make_counting_plan.py [--per-window 8] [--seed 41] [--out PATH]

Refuses to overwrite an existing plan, because counts may already refer to it.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import PROJECT_ROOT, Settings  # noqa: E402
from src.common.provenance import sha256_file  # noqa: E402
from src.datasets.manifest import DatasetManifest  # noqa: E402

WINDOWS = {  # window name in the Barbados manifest -> pipeline run that processed it
    "d21_local0200": "cam1_2025-10-21_0200_night",
    "d21_local0900": "cam1_2025-10-21_0900_morning",
}
CAMERA = "normanniles1"
USER_SAMPLE = {
    "clip_id": "user_sample_1",
    "path": "tests/Video/sample 1.mp4",
    "camera_id": "sample1",
    "run": "sample1_expressway",
    "condition_hint": "dense_urban_dusk",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-window", type=int, default=8)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data/annotations/reference_counts/plan_v1.csv")
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"{args.out} exists; plans are immutable once counting may have started. Use a new version.")

    settings = Settings()
    manifest = DatasetManifest.load(PROJECT_ROOT / "data" / "manifests" / "barbados_traffic.yaml")
    rng = random.Random(args.seed)
    rows = []
    for window, run in WINDOWS.items():
        items = sorted(
            (
                i
                for i in manifest.items
                if i.attributes.get("window") == window and i.attributes.get("camera_id") == CAMERA
            ),
            key=lambda i: i.id,
        )
        if len(items) < args.per_window:
            parser.error(f"window {window} has only {len(items)} clips")
        # stratified: split the window into equal blocks and draw one clip per block
        blocks = [
            items[round(k * len(items) / args.per_window) : round((k + 1) * len(items) / args.per_window)]
            for k in range(args.per_window)
        ]
        for block in blocks:
            item = rng.choice(block)
            rows.append(
                {
                    "clip_id": item.id,
                    "path": f"DATA_ROOT/{item.file}",
                    "camera_id": CAMERA,
                    "run": run,
                    "condition_hint": item.attributes.get("condition_hint"),
                    "sha256": item.sha256,
                    "start_time_local": item.attributes.get("start_time_local"),
                }
            )
    sample = PROJECT_ROOT / USER_SAMPLE["path"]
    if sample.is_file():
        rows.append({**USER_SAMPLE, "sha256": sha256_file(sample), "start_time_local": ""})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["clip_id", "path", "camera_id", "run", "condition_hint", "sha256", "start_time_local"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} clips to {args.out} (seed {args.seed}); data root {settings.data_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
