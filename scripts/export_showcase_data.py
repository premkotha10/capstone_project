"""Export aggregate results for the static showcase site (site/data/showcase.json).

Usage:  python scripts/export_showcase_data.py

Only aggregates are exported: counts, rates, distributions, latencies, statuses. No video frames, images, number
plates or per-vehicle crops leave the machine (Barbados licence notice, RISK R-20; personal data in user video).
Every figure is read from a pipeline run record, its CSV outputs, the roadmap, or the reference-count store.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import PROJECT_ROOT  # noqa: E402
from src.evaluation.reference_counts import consolidate, load_plan, load_records  # noqa: E402
from src.forecasting.baselines import DEFAULT_MODELS, forecast_ahead, rolling_origin_backtest  # noqa: E402
from src.services.results import RunStore  # noqa: E402

RUNS = {
    "cam1_2025-10-21_0900_morning": {"label": "Barbados camera 1 · 09:00 morning peak", "kind": "barbados"},
    "cam1_2025-10-21_0200_night": {"label": "Barbados camera 1 · 02:00 night", "kind": "barbados"},
    "sample1_expressway": {"label": "Sample video · congested expressway at dusk", "kind": "sample"},
}
OUT = PROJECT_ROOT / "site" / "data" / "showcase.json"


def _round(x: float, n: int = 3) -> float | None:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def run_summary(name: str, meta: dict) -> dict:
    run = RunStore().get(name)
    r = run.results
    tracks = run.tracks
    moving = tracks[(tracks["points"] >= 10) & (tracks["distance_px"] > 40)]
    stationary = tracks[(tracks["points"] >= 10) & (tracks["distance_px"] <= 40)]
    det = run.detections
    bins = np.linspace(0.25, 1.0, 16)
    conf_hist = (
        {cls: np.histogram(g["confidence"], bins=bins)[0].tolist() for cls, g in det.groupby("class")}
        if not det.empty
        else {}
    )
    iv = run.intervals
    per_minute = []
    if not iv.empty:
        grp = iv.groupby(["minute", "movement"])["count"].sum().unstack(fill_value=0)
        degraded = iv.groupby("minute")["reliability"].apply(lambda s: bool((s != "ok").any()))
        for minute, row in grp.iterrows():
            per_minute.append(
                {
                    "minute": int(minute),
                    "degraded": bool(degraded.get(minute, False)),
                    **{k: int(v) for k, v in row.items()},
                }
            )
    events = run.events
    by_line_class = {}
    if not events.empty:
        ct = pd.crosstab(events["line"] + ":" + events["direction"], events["class"])
        by_line_class = {mv: {c: int(n) for c, n in row.items() if n} for mv, row in ct.iterrows()}
    forecast = None
    movements = sorted(iv["movement"].unique()) if not iv.empty else []
    target = "approach:entering" if "approach:entering" in movements else None
    if target:
        y = run.volume_series(target)["count"].tolist()
        if len(y) >= 8:
            bt = rolling_origin_backtest(y, horizons=(1, 2, 3, 5))
            best = min((b for b in bt if b.horizon == 1), key=lambda b: b.mae)
            forecast = {
                "movement": target,
                "observed": y,
                "best_model": best.model,
                "predicted_next_5": [_round(v, 2) for v in forecast_ahead(y, DEFAULT_MODELS[best.model], range(1, 6))],
                "backtest": [
                    {"model": b.model, "horizon": b.horizon, "n": b.n, "mae": _round(b.mae), "rmse": _round(b.rmse)}
                    for b in bt
                ],
            }
    lat = {k: {m: _round(v[m], 1) for m in ("p50_ms", "p95_ms", "p99_ms")} for k, v in r["latency_ms"].items()}
    return {
        "id": name,
        **meta,
        "minutes": _round(r["duration_s"] / 60, 1),
        "frames": r["frames_processed"],
        "stride": r["stride"],
        "source_fps": _round(r["source_fps"], 2),
        "throughput_fps": r["throughput_fps"],
        "realtime_factor": _round(r["throughput_fps"] * r["stride"] / r["source_fps"], 2),
        "vehicles_counted": r["count_events"],
        "flow_per_hour": round(r["count_events"] / r["duration_s"] * 3600) if r["duration_s"] else None,
        "counts_by_movement": r["counts_by_line_direction"],
        "counts_by_class": r["counts_by_class"],
        "counts_by_movement_and_class": by_line_class,
        "tracks": r["tracks"],
        "moving_tracks": len(moving),
        "stationary_tracks": len(stationary),
        "fragment_share": _round((tracks["points"] < 5).mean()),
        "detections": len(det),
        "median_confidence": _round(det["confidence"].median()) if not det.empty else None,
        "confidence_bins": [_round(b, 2) for b in bins],
        "confidence_hist": conf_hist,
        "condition_share": r.get("condition_share", {}),
        "degraded_frame_share": r.get("degraded_frame_share"),
        "per_minute": per_minute,
        "forecast": forecast,
        "latency_ms": lat,
        "model": run.record["model"]["name"],
        "model_sha256": run.record["model"]["sha256"][:12],
        "config_hash": run.record["config_hash"],
    }


def phases() -> list[dict]:
    path = PROJECT_ROOT / "reports" / "status" / "phases.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []


def main() -> int:
    runs = [run_summary(n, m) for n, m in RUNS.items() if (RunStore().root / n / "run.json").is_file()]
    records = load_records()
    table, agreement = consolidate(records)
    payload = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "repository": "https://github.com/Pavansailahar/Traffic-Volume-Estimation-and-Short-Term-Congestion-Forecasting-from-Roadside-Video",
        "validation": {
            "plan_clips": len(load_plan()),
            "records": len(records),
            "clips_double_counted": agreement.clips_double_counted if agreement else 0,
            "statement": "Counts are pipeline estimates. They have not yet been validated against human reference "
            "counts; the hand-counting protocol and tool are in place.",
        },
        "phases": phases(),
        "runs": runs,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB, {len(runs)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
