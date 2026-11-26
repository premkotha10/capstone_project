# DSCI-41 — Traffic Volume Estimation and Short-Term Congestion Forecasting from Roadside Video

KL University two-semester engineering capstone. Specification (single source of truth):
[docs/UseCase.pdf](docs/UseCase.pdf).

> **Status (2026-11-23): Phase 5 of 15.** A working O2-A vertical slice (ingestion → YOLO11n detection →
> ByteTrack → line counting → volume baselines → Streamlit dashboard) runs on real Barbados video. Counts are
> **estimates not yet validated against reference counts**; no KPI has been measured. See
> [DEVELOPMENT_ROADMAP.md](DEVELOPMENT_ROADMAP.md).

## What this project builds

Roadside video → detection → classification → tracking → (perspective-aware) counting → interval traffic volumes
→ short-term congestion forecast → operator dashboard. Every result goes through an acceptance and negative-test
harness that checks it against reference counts across occlusion, rain and night conditions.

| Objective | Meaning |
|-----------|---------|
| O2 | Reference: mixed vehicle types and oblique perspective |
| O3 | Contribution: sustain accuracy under occlusion, rain and low light (separately switchable) |
| O4 | Integrated system: forecast near-term congestion from estimated volumes |
| O5 | Independent acceptance across conditions (AC-1…AC-4, NT-1…NT-5, KPI-1…KPI-6) |

## Showcase site

`site/` is a static results page (no build step) that shows aggregate figures from the pipeline runs. It contains no
video frames. Refresh its data with `python scripts/export_showcase_data.py`, then commit. To host it on Vercel,
import this repository at vercel.com/new and set **Root Directory** to `site` (framework preset: Other).

## Live demo (Hugging Face Space)

The full dashboard, with upload → detect → track → count, can run on a free Hugging Face Space (Docker, CPU).
`deploy/hf_space/` holds the Space card and Dockerfile; `scripts/build_hf_space.py` assembles a small bundle (app
code, configs and lock file only); `.github/workflows/sync-hf-space.yml` pushes it to the Space on every push to
`main`.

One-time setup:
1. On huggingface.co, create a **Space** (SDK: Docker, template: Blank, hardware: CPU basic, free).
2. Create an access token with **Write** permission (Settings → Access Tokens).
3. In this GitHub repository: Settings → Secrets and variables → Actions:
   add the secret `HF_TOKEN` (the token) and the variable `HF_SPACE` (`<hf-username>/<space-name>`).
4. Actions tab → **Sync Hugging Face Space** → Run workflow. The Space then builds (about 10 minutes the first
   time) and serves the dashboard in hosted mode: uploads only, first 60 s of each video, 1280 px.

## Planning documents

[PROJECT_ANALYSIS](PROJECT_ANALYSIS.md) · [REQUIREMENTS](REQUIREMENTS.md) · [DATASET_RESEARCH](DATASET_RESEARCH.md) ·
[SYSTEM_ARCHITECTURE](SYSTEM_ARCHITECTURE.md) · [DEVELOPMENT_ROADMAP](DEVELOPMENT_ROADMAP.md) ·
[RISK_REGISTER](RISK_REGISTER.md)

## Setup (Windows)

Prerequisites: Windows 10/11, the Python Install Manager (`py`), [uv](https://docs.astral.sh/uv/), FFmpeg on PATH,
Git.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
```

This script:
1. installs the signed python.org Python 3.11 if it is missing,
2. creates a virtual environment at `%USERPROFILE%\dsci41\venv`, outside OneDrive,
3. installs the exact locked dependencies from `uv.lock` (CPU PyTorch),
4. runs `scripts/check_env.py`.

The script uses signed python.org Python because Windows Smart App Control blocks uv's unsigned interpreter and
launcher. If you have no Smart App Control restriction, `uv sync` alone also works.

Run the tests:

```powershell
& "$HOME\dsci41\venv\Scripts\python.exe" -m pytest -q
```

Environment report:

```powershell
& "$HOME\dsci41\venv\Scripts\python.exe" scripts\check_env.py
```

Using pip instead of uv: `requirements.txt` is exported from `uv.lock`. The torch wheels come from the CPU index:
`pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu`.

Process a video end to end (detect → track → count), e.g. a 23-minute Barbados window from camera 1:

```powershell
& "$HOME\dsci41env\Scripts\python.exe" scripts\process_video.py --glob "normanniles1_2025-10-21-13-*.mp4" --camera normanniles1 --name cam1_morning
```

Open the operator dashboard (http://localhost:8501):

```powershell
& "$HOME\dsci41env\Scripts\python.exe" -m streamlit run dashboardpp.py
```

Hand-count reference (two people per clip, blind to model output): open the dashboard's **Reference Counting**
page, enter your annotator ID and role, and count the clips in `data/annotations/reference_counts/plan_v1.csv`
following [docs/COUNTING_PROTOCOL.md](docs/COUNTING_PROTOCOL.md). Accuracy appears on **Model Evaluation** once a clip
has two agreeing (or adjudicated) counts. Commit the saved JSON files: they are reference evidence.

Docker (clean-environment test run, CPU):

```powershell
docker compose run --rm tests
```

## Configuration

* Machine settings: environment variables or `.env`. See [.env.example](.env.example).
  Large data, model weights and logs live under `DSCI41_DATA_ROOT`, `DSCI41_MODELS_ROOT` and `DSCI41_LOG_DIR`.
* Experiment settings: [configs/](configs). They are validated YAML, unknown keys are rejected, and each config has a
  stable hash. `o2_a.yaml` and `o2_b.yaml` are the two O2 reference candidates.

## Repository layout

```
src/common/        shared types, config, provenance, hardware, logging     (implemented)
src/ingestion, detection, tracking, counting, monitoring, forecasting (baselines),
    pipeline, services, datasets, evaluation                               (implemented, O2-A slice)
src/perspective, enhancement, classification, api                          (skeleton only)
dashboard/         Streamlit app, 8 pages                                    (reads pipeline runs)
configs/           validated YAML experiment configs
scripts/           setup_env.ps1, check_env.py
tests/             unit / integration / acceptance / negative
data/              manifests and tiny samples only (raw data under DSCI41_DATA_ROOT)
reports/           committed experiment and acceptance reports
```

## Licence

The source code in this repository is licensed under the **GNU Affero General Public License v3.0**
(`AGPL-3.0-only`); see [LICENSE](LICENSE). AGPL-3.0 matches the licence of Ultralytics YOLO, which the detector and
tracker use. If you run a modified version as a network service, you must offer its source code to that service's
users.

The licence covers this project's code and documentation only. It does **not** cover:

- **Datasets**, which are never committed and keep their own terms: Barbados traffic video (Zindi states
  CC BY-SA 4.0, but frames carry an IWCP "Re-Use/Re-Streaming Forbidden" notice: see RISK_REGISTER R-20), UA-DETRAC
  and AI City (research and non-commercial use). See [DATASET_RESEARCH.md](DATASET_RESEARCH.md).
- **Model weights**, which are downloaded from Ultralytics releases at runtime under Ultralytics' licence.
- **`docs/UseCase.pdf`**, the KL University project specification, included for reference.
