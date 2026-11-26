# EXPERIMENT_LOG — DSCI-41

Every entry links to a machine-written run record (config + hash, input manifest IDs + SHA-256, model SHA-256, git
commit, environment). Numbers in this log are copied from run records, never typed by hand from memory.
"Development" entries use data that is not yet partitioned and **are not acceptance (AC-3) evidence**.

| ID | Date | Phase | Purpose | Data | Run record | Status |
|----|------|-------|---------|------|------------|--------|
| E-000 | 2026-11-23 | 4 | Environment feasibility probe (torch CPU) | none | console only | done |
| E-001 | 2026-11-23 | 5 | Pretrained YOLO11n detection baseline on UA-DETRAC subset, by weather | UA-DETRAC subset v1 | `reports/phase5/detection_ua_detrac_pretrained-*.json` | pending data |
| E-002 | 2026-11-23 | 5 | First end-to-end O2-A vertical slice on Barbados camera 1 (morning peak vs night) | Barbados subset v1, normanniles1 | `DATA_ROOT/runs/pipeline/cam1_2025-10-21_{0900_morning,0200_night}/run.json` | done (unvalidated) |
| E-003 | 2026-11-23 | 5 | User sample video: dense expressway, line-placement sensitivity | `tests/Video/sample 1.mp4` (user-supplied, local only) | `DATA_ROOT/runs/pipeline/sample1_expressway/run.json` | done (unvalidated) |

---

## E-000 — Environment feasibility probe
- MobileNetV3-Small, 640×640 input, torch 2.15.0+cpu, 16 threads: ≈ 38 ms/iteration (10 iterations, after 1 warm-up).
- Purpose: confirm CPU inference is viable. Not a project KPI.

## E-001 — Pretrained detector baseline on UA-DETRAC (development)
- **Question:** how well does an off-the-shelf COCO detector find vehicles in fixed roadside views, and how does
  that change with weather/lighting? This sets the starting point for O2. It does not set the O2 reference values,
  which are measured on frozen partitions in Phase 8.
- **Protocol:** `scripts/eval_detection_ua_detrac.py`. Class-agnostic; AP at confidence 0.001; P/R at 0.25; IoU 0.5
  for P/R; detections ≥ 50 % inside ignored regions dropped; every 5th annotated frame; small objects < 32² px.
- **Results:** _to be filled from the run record._

## E-002 — End-to-end O2-A vertical slice on Barbados camera 1 (development, unvalidated)
- **Pipeline:** `configs/o2_a.yaml` (YOLO11n, conf 0.25, 640 px) → ByteTrack (Ultralytics) → image-space lines in
  `configs/cameras/normanniles1.yaml` → 60 s intervals. Stride 2 (10 processed fps of 20). CPU only, while a data
  download was running.
- **Inputs:** 23 consecutive one-minute clips per window; SHA-256 per clip in each run record.

| Window (local) | Frames | Count events | Approach in / out | Ring road east / west | Tracks | Frames flagged | Throughput |
|---|---|---|---|---|---|---|---|
| 21 Oct 09:00–09:23 | 13,775 | 1,659 | 364 / 464 | 831 / 0 | 2,715 | 14.4 % dense occlusion | 19.4 fps (1.9× real time) |
| 21 Oct 02:00–02:23 | 13,780 | 62 | 3 / 23 | 36 / 0 | 176 | 100 % night | 17.7 fps |

- **Sanity checks passed:** zero westbound crossings on the ring road (one-way circulation); static objects
  (container, billboard) form stationary tracks that never cross a line.
- **Not established:** accuracy. No reference counts exist yet. Morning in/out asymmetry (364 vs 464) and the low
  night counts may be real or may be misses/ID switches in the queue at the approach line. The planned hand counts
  will decide.
- **Volume forecast baselines (morning, approach entering, 1-min horizon):** best = 5-minute moving average,
  MAE 3.73 vehicles/min over 18 origins. Illustrative only (23 points).

## E-003 — User sample: congested expressway, counting-line sensitivity (development, unvalidated)
- **Input:** `tests/Video/sample 1.mp4`, supplied by the user: portrait 2160×3840, ~19.4 fps, 20.7 s, 401 frames,
  overpass view of a congested urban expressway at dusk. Licence/provenance unknown → local only, not committed.
- **Pipeline:** same O2-A configuration as E-002, every frame (stride 1), geometry `configs/cameras/sample1.yaml`.
- **Result:** 116 tracks; 15.5 % of frames flagged dense occlusion; 4.5 fps at 4K on CPU.

| Line | car | bus | truck | motorcycle | total |
|---|---|---|---|---|---|
| `carriageway` (y = 2590) | 11 | 2 | 2 | **0** | 15 |
| `carriageway_low` (y = 3150, added after inspection) | 9 | 0 | 0 | **5** | 14 |

- **Finding (NT-1 / NT-4 evidence):** motorcycles filtering between queued cars are occluded until they are already
  below the upper line, so their tracks start after it and they are never counted there. Totals agree (15 vs 14)
  while the class mix differs completely: an aggregate count hides a total miss of one class. Many queued vehicles in
  the dense middle section get no detection at all. Both are direct targets for O3 (occlusion-aware detection and
  tracking) and for the NT-1/NT-4 fixtures.
- **Caution:** the lower line was added after seeing this clip, so it is a sensitivity analysis, not a tuned result.
