# DATASET_RESEARCH — DSCI-41

Research date / access date: **2026-11-23**. No dataset files were downloaded. Facts come from official pages, papers, or
listed mirrors visited on that date. Items marked **unverified** must be checked after the first download.
Status: **v0.1 — awaiting dataset-selection approval.**

---

## 1. Annotation-type matrix (the deciding table)

| Dataset | View | Video? | Detection boxes | Track IDs | **Vehicle count GT** | **Congestion labels** | Weather / light labels | Calibration | Long continuous time series |
|---------|------|--------|-----------------|-----------|----------------------|-----------------------|------------------------|-------------|----------------------------|
| Zindi Barbados Traffic Analysis | Fixed roadside, 4 cameras at one roundabout | Yes (~1-min segments, timestamped) | No | No | **No** | **Yes** — 4-class enter/exit rating per segment | No labels, but **night (02:00–05:50 local) is present** — see §2.1 update | No | **Yes** |
| AI City 2020 Track 1 | Fixed, 20 cameras | Yes | No | No | Yes (per movement, per class) | No | No (dawn/rain/snow present, unlabelled) | No | No (31 clips, ~9 h) |
| AI City 2021 Track 1 (accessible successor) | Same description as 2020 | Yes | No | No | Yes; **public GT completeness unverified** | No | No | No | No |
| UA-DETRAC | Fixed elevated roadside, 24 sites | Frame sequences, 25 fps | **Yes** (~1.21 M) | **Yes** | Derivable from track IDs (not native) | No | **Yes** (per-sequence: cloudy, night, sunny, rainy) + occlusion/truncation attrs | No | No (short sequences) |
| BDD100K | **Dashcam** | Yes (40 s) | Yes (1 keyframe/video; MOT subset at 5 fps) | MOT subset | No | No | Yes (weather, time of day) | No | No |
| ACDC | **Dashcam**, stills | No | Segmentation (detection listed) | No | No | No | Yes (fog/night/rain/snow) | No | No |
| BrnoCompSpeed | Fixed roadside/overpass, 6 sites × 3 views | Yes (~1 h each) | No | Per-vehicle passages | **Yes** (passage GT) | No | No (likely daytime only, unverified) | **Yes** | Partly (18 × 1 h) |
| CityFlowV2 | Fixed, 46 cameras, 16 sites | Yes (~3.6 h) | Yes, **only multi-camera vehicles** | Yes (880 IDs) | No (annotation not exhaustive) | No | No | Yes | No |

**Conclusion:** no dataset alone satisfies the PDF's primary evidence, *"Roadside traffic video with reference vehicle
counts across weather and lighting"*. A role-based combination is required, plus a small amount of documented
human counting.

## 2. Per-dataset findings

### 2.1 Zindi — Barbados Traffic Analysis Challenge (project reference source)
- **Source:** https://zindi.africa/competitions/barbados-traffic-analysis-challenge (now redirects to zindi.world; `/data` and `/rules` pages visited).
- **Status:** ran 07 Nov 2025 – 25 Jan 2026; shows "Completed".
- **Task:** forecast roundabout congestion 5 minutes ahead. 15 min input video → 2 min embargo → 5 min prediction window. Targets: `congestion_enter_rating`, `congestion_exit_rating` (4 classes: free flowing, light delay, moderate delay, heavy delay). Metric: 70 % macro-F1 + 30 % accuracy. Reports say exit ratings were later dropped from scoring; **unverified**.
- **Data:** 4 fixed cameras (Norman Niles #1–#4), each viewing one approach of the Norman Niles roundabout. Synchronised ~1-min segments with timestamped filenames (e.g. `normanniles1_2025-10-20-12-11-45.mp4`, per participant post). `Train.csv` 4.9 MB; ~16 k training rows (top team's README), ~63 % free flowing → **class imbalance**.
- **Not provided:** vehicle counts, boxes, tracks, weather/light labels, calibration. Resolution, fps, date range and night/rain coverage **unverified**. A top team used a brightness feature for day/night and rain, which suggests the conditions vary.
- **Size:** originals ≈ 620 GB in total; re-encoded ≈ 126 GB; compressed ≈ 140 GB (Kaggle mirror sizes). Hosted on Google Drive and GCS (`brb-traffic`, `brb-traffic-full`); public listing of the GCS bucket was reported by a participant (**untested**).
- **Licence:** **CC BY-SA 4.0** (data page and rules). Kaggle mirrors relabel it CC BY / CC0; ignore that and follow CC BY-SA 4.0. The competition rule *"Back-propagation is not allowed in training or inference"* applied to competition entries, not to this project; we will note it anyway.
- **Role for us:** primary operational domain and **real congestion labels for O4 forecasting**. Its count reference has to be created by us (see §4).

> **Update after first download (2026-11-23, Phase 5) — supersedes earlier statements:**
> - **Clips:** 4,633 per camera folder plus 184 in `added`. They are H.264, 960×540, 20 fps, about 59.9 s and about 4.4 MB each.
> - **Time base:** the times in the file names run 06:00–17:59 from 20 to 26 Oct 2025. The camera's burned-in clock overlay reads **4 hours earlier**: 06:00:45 shows as 02:00:54, and 13:13:45 shows as 09:13:44. That fits file names in UTC and local Barbados time AST = UTC−4.
> - **Night coverage:** local coverage is therefore **02:00–13:59**, so the dataset **does contain night footage** (02:00 to about 05:50, lit by street lamps) as well as morning-peak and midday traffic. My earlier "no night" reading was wrong.
> - **Licence conflict:** every frame carries the burned-in notice **"(c) IWCP Inc. Re-Use/Re-Streaming Forbidden"**, which conflicts with the CC BY-SA 4.0 statement on Zindi. Until this is clarified, we only analyse the video locally. We never redistribute it, publish clips, or put frames in the repository. Any frames shown in reports or the demo need permission first (RISK R-20).
> - **Official mirror cost:** the official GCS bucket `brb-traffic` is *requester-pays*, so downloading from it bills a Google Cloud project; we do not use it. `brb-traffic-full` returns "bucket does not exist". The public Google Drive folders work and are the source used.
> - **Labels need sign-in:** `Train.csv` and the other CSVs return HTTP 401 without a Zindi sign-in and acceptance of the competition terms.
> - **Static false positives:** a roadside billboard is detected as "truck" (confidence 0.26) on camera #2. Stationary false positives need a region of interest and track-motion filtering before counting.

### 2.2 AI City Challenge 2020 Track 1
- **Source:** https://www.aicitychallenge.org/2020-data-and-evaluation/
- 31 clips, ~9 h, 20 fixed cameras; ≥ 960p, mostly 10 fps; classes **car, truck** only; per-camera ROI + movements of interest; manual count GT; dawn, rain and snow present; metric nwRMSE + efficiency; 5.43 GB.
- **Availability:** the 2020 release is **no longer accessible** (official access page). The **2021 Track 1** counting set is listed, password-free via Google Drive, with an identical description. It is probably the same data, but that is not stated.
- **Licence:** research only, no commercial use (request form). Whether the public release has full count GT, or only Dataset A's demo subset, is **unverified**.
- **Role:** independent count oracle across many cameras (AC-1/AC-3, KPI-3), if the GT check passes.

### 2.3 UA-DETRAC
- **Sources:** arXiv:1511.04136; author page https://sites.google.com/view/daweidu/projects/ua-detrac (Google Drive links). The original detrac-db.rit.albany.edu is offline.
- 100 sequences (60 train / 40 test), ~140 k frames, 960×540 at 25 fps, 24 locations, fixed elevated camera. Classes: car, bus, van, others. Boxes, track IDs, occlusion/truncation, and per-sequence **cloudy / night / sunny / rainy** labels.
- **Size:** ~9.9 GB (full images). **Licence:** original **unverified**; treat as academic, non-commercial, cite Wen et al., CVIU 2020. Mirrors (Zenodo 14030107, Kaggle, Hugging Face) carry uploader-chosen licences with no authority behind them; prefer the author's links.
- **Role:** detector/tracker evaluation and fine-tuning, and the **main adverse-condition evidence** (night/rain/occlusion labelled). Counts are derived from GT track IDs crossing a virtual line. These are derived from human annotation, so they can serve as an oracle, but the derivation must be versioned and reported as "derived reference counts".

### 2.4 BDD100K
- arXiv:1805.04687; https://github.com/bdd100k/bdd100k. Official download hosts were unreachable on 2026-11-23 (DNS/TLS failures).
- Dashcam, 100 k × 40 s videos; weather and time-of-day tags; detection subset ~7.1 GB. Data licence not found.
- **Role:** optional auxiliary training data for night/rain only. Wrong viewpoint for counting.

### 2.5 ACDC
- arXiv:2104.13395; https://acdc.vision.ee.ethz.ch. 8,012 dashcam stills (fog/night/rain/snow, paired with normal-condition images); segmentation annotations. Registration required; non-commercial research licence.
- **Role:** optional robustness sanity checks only. Not recommended for the core.

### 2.6 BrnoCompSpeed
- https://github.com/JakubSochor/BrnoCompSpeed; arXiv:1702.06441. 18 × ~1 h full-HD videos, 6 sites × 3 views, 20,865 vehicles with passage and speed GT, camera calibration. ~200 GB; access by e-mail request. Licence not found.
- **Role:** optional. It is the best evidence for perspective/calibration (O2-B) and long count series, but it is too large for the current disk and likely daytime only.

### 2.7 CityFlowV2
- aicitychallenge.org (2022 Track 1 = CityFlowV2); arXiv:1903.09254. 46 cameras, 16 sites, ~3.6 h, calibration provided; boxes cover only vehicles seen in ≥ 2 cameras. Clicking the download accepts a research-only licence agreement. 2020 version was 15.7 GB.
- **Role:** optional cross-camera tracking. **Not valid as count GT** because annotation is not exhaustive.

## 3. Recommended combination

| Role in project | Dataset | Objective / KPI served |
|-----------------|---------|------------------------|
| Operational domain + congestion forecasting labels | **Barbados** (subset) | O4, KPI-2, demo |
| Counting reference on Barbados | **Barbados subset hand-counted by two annotators** (stratified by camera, hour, brightness/rain) | O2, O5, KPI-1, KPI-4, KPI-5 on the target domain |
| Detection/tracking + labelled night/rain/occlusion | **UA-DETRAC** | O2, O3, KPI-1, KPI-4, KPI-5, NT-1, NT-2, NT-4 |
| Independent multi-camera count oracle | **AI City 2021 Track 1** (if public GT is sufficient) | AC-3, KPI-3, NT-3 |
| Optional calibration study | BrnoCompSpeed (one site, if obtainable) | O2-B, NT-3 |
| Not recommended for core | BDD100K, ACDC, CityFlowV2 | — |

Barbados plus hand-counting is required because the PDF frames the Zindi challenge as the reference source. UA-DETRAC is
the only accessible dataset with *labelled* night and rain on fixed roadside cameras.

## 4. Gaps and how they are closed honestly

| Gap | Closure | Integrity control |
|-----|---------|-------------------|
| Barbados has no count GT | Two annotators count a stratified sample of 1-min segments per class and direction, using a counting tool built in Phase 5 | Inter-annotator agreement reported; disagreements adjudicated and logged; counts never produced by our model |
| Barbados has no weather/light labels | Label segments night/day/rain by inspection plus timestamps; record the labeller | Labels made blind to model output; stored in manifest |
| UA-DETRAC has no native counts | Derive line-crossing counts from GT tracks with a versioned script | Script unit-tested on synthetic tracks; reported as "derived reference" |
| Barbados resolution/fps/night/rain coverage unknown | **Resolved for resolution/fps/night** (960×540, 20 fps, night 02:00–05:50 local). Rain still to be checked by inspection | Recorded in manifest |
| AIC21 count-GT completeness unknown | Inspect after download (5.4 GB) | Dataset dropped from AC-3 role if GT is demo-only |
| Disk: 11.6 GB free | First subset ≤ 3 GB; data root outside OneDrive; external drive recommended | Download plan approved before each pull |

## 5. Proposed first download (needs approval)

| Item | Approx. size | Purpose |
|------|--------------|---------|
| Barbados `Train.csv`, `TestInputSegments.csv` | ~5 MB | Label structure, time coverage |
| Barbados: e.g. 1–2 hours of re-encoded/compressed segments from 2 cameras, spread across day and night timestamps | target ≤ 1.5 GB (exact per-segment size measured on first file) | Pipeline vertical slice, condition inspection |
| UA-DETRAC: a subset of ~10 sequences (balanced sunny/cloudy/night/rainy) + annotations XML | target ≤ 1.5 GB (depends on how the Drive archives are split; **unverified**) | O2/O3 counting under labelled conditions |

AI City 2021 Track 1 (~5.4 GB) is deferred until disk space is resolved.

## 6. Provenance manifest template

Each dataset gets `data/manifests/<dataset>.yaml`:

```yaml
dataset: barbados_traffic
version: "zindi-2026-01 (re-encoded set)"
source_url: https://zindi.africa/competitions/barbados-traffic-analysis-challenge
access_date: 2026-11-23
licence: CC-BY-SA-4.0
custodian: Zindi / competition host
items:
  - id: normanniles1_2025-10-20-12-11-45
    file: raw/barbados/normanniles1/normanniles1_2025-10-20-12-11-45.mp4
    sha256: <filled on download>
    partition: <dev_train|dev_val|acceptance_ac3>
    conditions: {light: <labelled later>, weather: <labelled later>}
    used_by: [O2, O4, KPI-1, KPI-2]
exclusions: []
transformations: []
```
