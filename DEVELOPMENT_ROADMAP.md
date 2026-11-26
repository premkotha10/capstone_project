# DEVELOPMENT_ROADMAP — DSCI-41

Status: **v0.1 — 2026-11-23.** ✅ done · 🟡 in progress · ⬜ not started · ⏸ awaiting approval

Mapping to PDF gates: Phases 1–4 → **CP1 Gate 1 (O1)**; Phases 5–8 → **CP1 Gate 2 (O2)**; Phase 9 → **CP2 Gate 3
(O3)**; Phases 10–13 → **CP2 Gate 4 (O4)**; Phases 14–15 → **CP2 Final Gate (O5)**.

| Phase | Title | Key outputs | Exit criterion | Status |
|-------|-------|-------------|----------------|--------|
| 1 | PDF analysis & requirements extraction | PROJECT_ANALYSIS.md, REQUIREMENTS.md | Requirements reviewed by team/guide | 🟡 drafted, ⏸ review |
| 2 | Dataset research & selection | DATASET_RESEARCH.md, access requests, subset plan | Datasets chosen; licences recorded; access confirmed | 🟡 drafted, ⏸ approval |
| 3 | Architecture & technical design | SYSTEM_ARCHITECTURE.md, ADRs, RISK_REGISTER.md, **frozen KPI definitions + rubrics** | Rubrics tagged in git **before any O2 measurement** | 🟡 drafted, ⏸ review |
| 4 | Environment & project structure | Own git repo, signed Python 3.11.9 venv, `pyproject.toml` + `uv.lock` + `requirements.txt`, skeleton packages, `src/common` (types, config, provenance, hardware, logging), configs, `check_env.py`, `setup_env.ps1`, `.env.example`, Dockerfile, CLAUDE.md | `pytest` green; `check_env.py` PASS; setup script re-runs cleanly | ✅ 2026-11-23 (Docker image build not yet run) |
| 5 | Video ingestion & detection | `src/ingestion`, `src/detection`, manifest builder, first data subset + converters | Detection P/R/mAP on dev-val subset; ingestion unit tests | 🟡 in progress (data downloading; ingestion + detection built) |
| 6 | Classification & tracking | Class map, ByteTrack integration, trajectories | IDF1/MOTA/IDSW on sequences with track GT | 🟡 built (vertical slice); metrics pending |
| 7 | Traffic volume estimation | Line/zone counting, direction, per-class, interval aggregation | Counting MAE/RMSE/WAPE vs reference counts; synthetic oracle fixture passes | 🟡 built (vertical slice); counting protocol, plan (17 clips) and blind counting tool ready; hand counts pending |
| 8 | Perspective correction | Per-camera calibration files, homography, O2-A vs O2-B comparison, **O2 frozen** | O2 KPI-1…KPI-6 baselines recorded under AC-1/AC-4; **AC-3 partition locked** | ⬜ |
| 9 | Adverse-condition improvement (O3) | `src/enhancement`, tracker options, fine-tune, ablation matrix | O3 vs O2 with CIs; positive or negative conclusion documented | ⬜ |
| 10 | Congestion forecasting | Features, baselines, RF/XGBoost, horizon-wise eval | Forecast contracts + backtest on dev data | ⬜ |
| 11 | FastAPI backend | Upload/jobs/results/forecast/metrics/health endpoints, OpenAPI, API_DOCUMENTATION.md | Integration tests green | ⬜ |
| 12 | Streamlit dashboard | 8 pages per master instruction, data-source labelling | Every widget backed by API; demo data watermarked | 🟡 8 pages built on run files; API not yet in between |
| 13 | System integration | Docker Compose, degraded mode, recovery, telemetry | End-to-end system test; recovery demonstrated | ⬜ |
| 14 | Testing, acceptance & performance | AC-1…AC-4, NT-1…NT-5, KPI-1…KPI-6 harness; rater scoring | Acceptance report; every target passes or explicit hold/revise | ⬜ |
| 15 | Documentation & demo | INSTALLATION_GUIDE, USER_GUIDE, FINAL_REPORT, paper/poster drafts, demo script | Clean-environment rebuild reproduces reported numbers | ⬜ |

## Immediate next steps

1. ✅ Dedicated git repository; planning documents committed.
2. ✅ Phase 4 environment setup.
3. Phase 5: download the approved first subset (Barbados CSVs + ~1.5 GB segments, ~1.5 GB UA-DETRAC subset), build
   manifests, implement ingestion + detection.
4. Team + guide (parallel, blocking O2 measurement): confirm stakeholder (OI-1), roles (OI-3), raters (OI-5), freeze
   KPI-1/4/5 operational definitions and KPI-2/3 rubrics.

## Dependencies that need explicit approval

| Item | Why approval | Est. size/cost |
|------|-------------|----------------|
| Dataset downloads beyond a ~2–3 GB first subset | Disk (11.6 GB free) | see DATASET_RESEARCH.md |
| Any remote GPU (Colab/Kaggle/cloud) | Possible cost; data leaves machine | — |
| Initialising a new git repository here | Changes the repo layout (outer Desktop repo exists) | — |
