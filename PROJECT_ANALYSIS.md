# PROJECT_ANALYSIS — DSCI-41

**Traffic Volume Estimation and Short-Term Congestion Forecasting from Roadside Video**
Source of truth: [docs/UseCase.pdf](docs/UseCase.pdf) (12 pages, generated 23/11/2026 by KL University Capstone Project Portal).
Document status: **DRAFT v0.1 — Phase 1 output, awaiting approval.** Last updated 2026-11-23.

---

## 1. Problem statement (from the PDF, p.2)

Many cities have roadside cameras but no traffic counts. Turning video into reliable vehicle counts requires handling:

- oblique camera angles,
- occlusion in dense traffic,
- mixed vehicle types,
- weather that degrades the image.

Without this, a visual result is unreliable and **can fail silently** under realistic scene and sensor conditions. The
"fail silently" phrase drives a large part of the specification: the system must *know* when it is out of its validated
envelope (NT-1…NT-5, FR-7, NFR-2, NFR-6).

**Final achievement required:** roadside video traffic counting and congestion forecasting validated under occlusion,
rain and night conditions.

## 2. Objective chain

| ID | Semester | Objective (PDF wording) | What it means in engineering terms | Gate |
|----|----------|-------------------------|------------------------------------|------|
| O1 | VII (CP1) | Validate the problem and freeze the engineering contract | Baseline vehicle counter under fixed camera geometry; requirements, ADRs, risk register, source-to-claim map, capture/annotation checks, **frozen KPI rubrics** | CP1 Gate 1 |
| O2 | VII (CP1) | Handle mixed vehicle types and oblique perspective | Reference counting pipeline (detect → track → count with perspective handling). **Compare ≥ 2 feasible reference approaches**, pick one via documented trade-off; measure under AC-1 and AC-4; produce baseline values for KPI-1…KPI-6 | CP1 Gate 2 |
| O3 | VIII (CP2) | Sustain accuracy under occlusion, rain and low light | A separately testable robustness subsystem; compared with O2 on the *same* evidence and resource envelope; ablation/sensitivity; positive **or negative** conclusion is acceptable if defensible | CP2 Gate 3 |
| O4 | VIII (CP2) | Forecast near-term congestion from estimated volumes | Complete operable system: evidence path, O2+O3, operator UI, config, safety controls, telemetry, degraded mode, recovery | CP2 Gate 4 |
| O5 | VIII (CP2) | Validate counting and forecasting against reference counts across conditions | AC-1…AC-4, all 5 negative tests, independent review, validation report | CP2 Final Gate |

Key consequence: **O2 is the yardstick.** Every KPI target is relative to the O2 reference measured on the same evidence,
workload and resource envelope. O2 numbers must therefore be measured honestly and frozen before O3 work begins.

## 3. What the PDF explicitly requires that is easy to miss

1. **Rubrics frozen before O2** (KPI-2, KPI-3): a 0–100 rubric with ≥ 5 observable criteria and named 0/50/80/100
   anchors, scored by **two independent raters**, with inter-rater agreement reported. An AI assistant cannot be one of
   the raters. KPI-2 ("congestion forecast error") is a **rubric score, not MAE/RMSE**; MAE/RMSE will be reported as
   supporting evidence and can be one rubric criterion, but they do not replace the rubric.
2. **AC-3 evidence reserved before O3 tuning**, separated by the independent unit (camera/site/scene), with no
   transformation leaking acceptance information into development (NT-5).
3. **95 % confidence intervals and condition-wise values** for KPI-1, KPI-4, KPI-5; raw confusion/error evidence retained.
4. **Small, rare and safety-critical** objects/scenes reported separately (NFR-4, AC-3, NT-4). Motorcycles, bicycles,
   pedestrians-adjacent vehicles, buses at frame edge, and night/rain scenes are candidates for this slice.
5. **KPI-6 latency protocol:** warm-up, ≥ 30 measured trials per condition, monotonic clock, report p50/p95/p99 and
   failures.
6. **Comparison of ≥ 2 reference approaches in O2** with a documented trade-off.
7. **Negative tests require detection + safe response**, not just a failing test: detect → abstain/degrade → label the
   result → prevent silent use → recover (restore accepted bundle, rerun scene class) → retain evidence and recovery time.
8. **Clean-environment rebuild** of O2, O3, integrated system and acceptance evidence from a pinned toolchain and a
   scripted runbook.
9. **Evidence manifest** fields: source URL, access date, licence/permission, owner/custodian, sample IDs, exclusions,
   transformations, configuration and the objective/KPI using each item.
10. **Explicitly excluded:** notebook-only, dashboard-only or isolated-model deliverables; production certification;
    universal generalisation; unattended high-stakes operation.

## 4. Open items the PDF says must be resolved at approval

| # | Open item | Owner | Why it blocks |
|---|-----------|-------|---------------|
| OI-1 | KL deployment stakeholder and operational decision owner are **not identified** (p.2) | Team + faculty guide | Required before approval |
| OI-2 | No sponsor/funder stated | Faculty guide | Charter completeness |
| OI-3 | Team of 3–4 with named roles (evidence lead, vision/geometry lead, application/deployment lead, critical-case validation lead) | Team | Each member must own reviewable work; individual viva evidence (D7) |
| OI-4 | Dataset access and licence confirmed at approval review (Dependency assumption, p.7) | Team | Unavailable data requires an approved scope revision |
| OI-5 | Two independent raters for KPI-2/KPI-3 rubrics | Team + guide | Rubric KPIs cannot be scored without them |
| OI-6 | Approved non-inferiority margin for KPI-6 guardrail | Team + guide | KPI-6 guardrail references it |
| OI-7 | Definition of "critical scene" and failure threshold for KPI-4 | Team + guide | Must be frozen before O2 |
| OI-8 | Compute facility (PDF expects "GPU and storage"); this laptop has no CUDA GPU and ~11.6 GB free disk | Team + guide | Determines fallback path (subset/tiling/smaller model) |

## 5. Environment inspection (2026-11-23)

| Item | Finding | Implication |
|------|---------|-------------|
| OS | Windows 11 Home 10.0.26300 | Windows-first scripts; Docker for clean-environment rebuild |
| CPU | Intel Core Ultra 9 185H, 16 cores / 22 threads | Viable for small YOLO models on CPU / OpenVINO |
| GPU | Intel Arc integrated ("Intel(R) Graphics", ~2 GB shared reported); **no NVIDIA / CUDA** | No CUDA training; inference via CPU or OpenVINO (iGPU/NPU). Fine-tuning must be small or done on a remote GPU (needs approval) |
| RAM | 15.5 GB | Batch sizes and video buffering must be modest |
| Disk | **11.6 GB free** on C: | Full UA-DETRAC/CityFlow/BDD100K downloads are not feasible locally; subset strategy mandatory |
| Python | System 3.15.0 only; `uv` 0.13.1 can provision 3.11.17 | Phase 4 decision: **signed python.org 3.11.9** venv (uv's launcher was blocked by Smart App Control — ADR-1) |
| PyTorch | torch 2.15.0+cpu / torchvision 0.29.2 import and run on 3.11 (probe: MobileNetV3-S 640² ≈ 38 ms) | CPU path verified |
| OpenCV | opencv-python-headless 5.1.0 + NumPy 2.4.6 verified | OK |
| FFmpeg | 9.0.2 full build on PATH | OK for transcoding/metadata |
| Git | 2.56.0 — but the **repository root is `C:\Users\pavan\OneDrive\Desktop`**, not this folder, and it contains another project's `node_modules` that crash `git status` | Needs its own repo in this folder (see RISK R-15) |
| Docker | 29.9.3 present | Clean-environment build feasible |
| Smart App Control | **Enforcing.** First load of some unsigned native files (`charset_normalizer` `.pyd`, uv-built `_overlapped.pyd`) was blocked, then allowed on retry | Intermittent `ImportError: DLL load failed … Application Control policy`. See RISK R-14 |
| OneDrive | Project lives in a OneDrive-synced folder | Large data/model files must not live here or must be excluded from sync |

## 6. My understanding in one paragraph

We must build a reproducible, *honest* traffic-counting system: a reference (O2) that counts mixed vehicle types from
oblique roadside cameras, a separately switchable robustness contribution (O3) for occlusion/rain/night that is judged
only by measured improvement over O2 on the identical evidence and resource budget, and an integrated application (O4)
that turns estimated counts into short-term congestion forecasts behind an operator dashboard with telemetry, degraded
mode and recovery. Everything is then accepted (O5) on reserved, unseen cameras/conditions with confidence intervals,
condition-wise breakdowns and five negative tests that prove the system refuses to silently report unreliable output.
The largest scientific risk is evidence: we need **reference counts** under **night and rain** on **fixed oblique
cameras**, and few public datasets provide all three. The largest practical risk is compute (no CUDA, 11.6 GB disk).

## 7. Source-to-claim map (initial)

| Claim in the project | Source | Status |
|----------------------|--------|--------|
| Barbados roadside traffic analysis is a published challenge | Zindi competition page (PDF p.1) | To be verified — see DATASET_RESEARCH.md |
| O2/O3/O4/O5, FR, NFR, AC, KPI, NT, D definitions | docs/UseCase.pdf pp.2–12 | Extracted verbatim in REQUIREMENTS.md |
| Any detection/count/forecast performance number | Our own experiments only | None yet — no number may be reported before it is measured |
