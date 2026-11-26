# REQUIREMENTS — DSCI-41

Source: [docs/UseCase.pdf](docs/UseCase.pdf). Requirement text in *italics* is quoted or closely paraphrased from the PDF.
Columns headed "Implementation plan" and "Verification" are **our proposals** and are open to review.
Status: **DRAFT v0.1 — 2026-11-23.** Nothing below is implemented yet.

---

## 1. Functional requirements

| ID | Requirement (PDF) | Implementation plan | Verification |
|----|-------------------|---------------------|--------------|
| FR-1 | *Ingest or acquire "Roadside traffic video with reference vehicle counts"; validate version, provenance, schema or protocol, units and project-specific quality gates before use.* | `src/ingestion/`: video validator (codec, fps, resolution, duration, frame-count consistency, corruption scan); dataset loaders that refuse items with no manifest entry; manifest schema (Pydantic) with SHA-256 per file | Unit tests on valid/corrupt/mislabelled fixtures; manifest validation in CI |
| FR-2 | *Reproduce "Handle mixed vehicle types and oblique perspective" through a one-command, one-protocol or one-test-run reference path with an independent expected-result oracle.* | `scripts/run_o2_reference.py --config configs/o2.yaml`; oracle = human reference counts from the dataset (never produced by our pipeline) plus a small synthetic fixture with analytically known counts | `pytest tests/acceptance/test_o2_reference.py` reproduces frozen O2 numbers within tolerance |
| FR-3 | *Implement "Accuracy under occlusion, rain and low light" as a separately testable subsystem whose configuration and evidence are versioned.* | `src/enhancement/` + O3 tracker/counter options behind a single `robustness:` config block; O2 = same pipeline with block disabled | Ablation matrix runs; unit tests per component; config hash stored with every result |
| FR-4 | *Implement and verify "Forecast near-term congestion from estimated volumes" and its input, output, error, configuration and telemetry contracts.* | `src/forecasting/`: Pydantic I/O contracts; baselines (persistence, seasonal-naive), Random Forest, XGBoost; configurable horizons | Contract tests; rolling-origin backtest on training partition only; acceptance on reserved partition |
| FR-5 | *Implement and verify "Object detection and tracking" and its input, output, error, configuration and telemetry contracts.* | `src/detection/`, `src/tracking/` with typed `Detection`, `Track` dataclasses; ByteTrack | Detection mAP/P/R; tracking IDF1/MOTA/IDSW where track GT exists |
| FR-6 | *Deliver "Traffic Volume Estimation and Short-Term Congestion Forecasting" as the complete demonstrable result; no isolated notebook, model, function or dashboard counts as completion.* | FastAPI backend + Streamlit dashboard + CLI; Docker Compose | End-to-end system test: upload video → counts → forecast → report |
| FR-7 | *Detect the condition described by NT-1 (occlusion in dense traffic), invoke the documented safe or degraded response and retain recovery evidence.* | Runtime "envelope monitor": density/overlap/track-fragmentation indicators; when out of envelope → result labelled `DEGRADED`/`ABSTAIN`, forecast inputs flagged | NT-1 test fixture; recovery evidence written to `reports/negative_tests/` |

## 2. Non-functional requirements

| ID | Requirement (PDF) | How we will satisfy it |
|----|-------------------|------------------------|
| NFR-1 | *Performance: meet every accepted KPI target under AC-1 and AC-4, including tail values and resource cost.* | KPI harness reports targets, tail latency and resource use side by side |
| NFR-2 | *Reliability and recovery: pass every mandatory negative test; a critical failure cannot be hidden by an average.* | NT-1…NT-5 automated; per-condition/per-class reporting is mandatory in every report |
| NFR-3 | *Domain assurance: capture, geometry, annotation and transformation consistency shall be machine-checkable.* | Validators for video metadata, homography reprojection error, annotation schema/box sanity, transformation logs |
| NFR-4 | *Evidence quality: small, rare and safety-critical objects or scenes reported separately from aggregate accuracy.* | Fixed slice definitions (small-object area threshold, rare classes, night/rain/dense) frozen before O2 |
| NFR-5 | *Maintainability: version control, automated tests, documented interfaces, configuration validation and a clean-environment build.* | Git, pytest, Pydantic configs, OpenAPI docs, Dockerfile |
| NFR-6 | *Operability: logs, measurements, health/quality indicators, operator guidance, degraded mode and verified recovery.* | Structured logging, `/health` and `/metrics` endpoints, dashboard System Monitoring page, USER_GUIDE runbook |

## 3. Acceptance conditions

| ID | PDF definition | Planned harness |
|----|----------------|-----------------|
| AC-1 | *Representative operation: validate counting and forecasting against reference counts across conditions.* | `tests/acceptance/test_ac1_representative.py` on held-out representative clips |
| AC-2 | *Boundary and failure operation: exercise occlusion in dense traffic; night and rain degradation; camera-specific calibration.* | Implemented through NT-1, NT-2, NT-3 fixtures plus boundary sweeps (density, illumination, rain intensity, calibration perturbation) |
| AC-3 | *Independent acceptance evidence: unseen scenes, devices, sites or conditions; report small, rare and safety-critical cases separately.* | Partition locked and hashed **before O3 tuning**; separated by camera/site; access logged |
| AC-4 | *Frozen resource envelope: same versions, workload, hardware/facility limits and resource budget for reference and final comparisons.* | `configs/resource_envelope.yaml` (hardware ID, thread count, input resolution, frame stride, package lock hash); harness refuses to compare runs whose envelopes differ |

## 4. KPIs — verbatim contract plus proposed operational definitions

The PDF fixes baseline, direction, unit, target, conditions, method and guardrail. It **does not** fully define the
underlying measure for KPI-1, KPI-4 and KPI-5, and it requires rubrics for KPI-2 and KPI-3. The "Proposed operational
definition" column must be **reviewed and frozen by the team and guide before O2 is measured**; it is not a PDF
requirement.

| ID | Measure | Unit / direction | Target (PDF) | Method (PDF) | Proposed operational definition (to freeze) |
|----|---------|------------------|--------------|--------------|---------------------------------------------|
| KPI-1 | Counting accuracy under adverse conditions | %, higher | ≥ O2 + 3 pp, **or** within 1 pp while improving the primary resource KPI by ≥ 20 % | Leakage-resistant independent partition; condition-wise values, 95 % CI, raw error evidence | Per (clip, interval, class, direction) cell: `acc = 1 − WAPE = 1 − Σ|ŷ−y| / Σy`, computed over adverse-condition clips (night, rain, dense occlusion); CI by clip-level bootstrap |
| KPI-2 | Congestion forecast error | 0–100 rubric, higher | ≥ 80/100 **and** ≥ O2 + 5 | Rubric frozen before O2, ≥ 5 observable criteria, anchors 0/50/80/100, two independent raters, agreement reported, condition-wise | Draft rubric in §5 — MAE/RMSE/horizon skill feed the criteria |
| KPI-3 | Cross-camera generalisation | 0–100 rubric, higher | ≥ 80/100 **and** ≥ O2 + 5 | Same rubric procedure | Draft rubric in §5 |
| KPI-4 | Critical-scene failure rate | %, lower | ≤ 0.80 × O2; no critical condition worse than O2 | Independent partition, condition-wise, 95 % CI | `% of critical scenes (night/rain/dense) whose interval count error > τ`; τ frozen before O2 (proposal: 20 % relative or 3 vehicles absolute, whichever is larger) |
| KPI-5 | Cross-condition performance gap | %, lower | ≤ 0.80 × O2; no critical condition worse than O2 | Independent partition, condition-wise, 95 % CI | `gap = (acc_reference_condition − acc_worst_condition) / acc_reference_condition × 100`, reference condition = clear daytime |
| KPI-6 | Inference latency p95 | ms, lower | ≤ 0.80 × O2 at equal or better functional quality | Warm-up, ≥ 30 measured trials per condition, monotonic clock, p50/p95/p99 + failures | End-to-end per-frame latency (decode → enhance → detect → track → count) with `time.perf_counter_ns` |

KPI-6 guardrail: *the primary correctness, safety or quality metric may not regress by more than the approved
non-inferiority margin* — margin is open item OI-6.
All KPIs guardrail: *no critical subgroup, scene or operating condition may be hidden by an aggregate result.*

### Note on the KPI-1 / KPI-6 tension
O3 adds pre-processing, which usually *increases* latency, while KPI-6 requires p95 ≤ 0.80 × O2. Options to meet both
are (a) condition-gated enhancement (only run on frames the envelope monitor flags), (b) a smaller/faster detector in O3
compensated by robustness gains, (c) OpenVINO/INT8 export applied to O3. Whether the target is met will be reported
honestly; a negative conclusion is permitted by the PDF if defensible.

## 5. Draft rubrics (KPI-2 and KPI-3) — NOT FROZEN

These drafts exist so the team has something concrete to review. Each criterion is scored 0–20; total 0–100.

**KPI-2 Congestion forecast quality rubric (draft)**

| # | Observable criterion | 0 | 10 (≈50 anchor) | 16 (≈80 anchor) | 20 (100 anchor) |
|---|----------------------|---|-----------------|-----------------|-----------------|
| C1 | Volume MAE skill vs. persistence baseline on held-out partition | worse than persistence | equal | ≥ 15 % better | ≥ 30 % better |
| C2 | Congestion-level agreement (macro-F1 vs. reference-derived level) | < 0.3 | 0.5 | 0.7 | ≥ 0.85 |
| C3 | Horizon degradation (error growth from shortest to longest horizon) | > 3× | 2× | 1.5× | ≤ 1.2× |
| C4 | Condition-wise consistency (worst-condition skill vs. aggregate) | worst condition fails C1 | within 50 % | within 25 % | within 10 % |
| C5 | Operator-visible correctness: forecast labelled with inputs' reliability flag, degraded inputs never shown as normal | never | sometimes | always labelled, minor UI gaps | always, verified by NT tests |

**KPI-3 Cross-camera generalisation rubric (draft)** — scored on cameras unseen in development

| # | Criterion | 0 | 50 anchor | 80 anchor | 100 anchor |
|---|-----------|---|-----------|-----------|------------|
| G1 | Count accuracy on unseen cameras relative to seen cameras | < 50 % | 75 % | 90 % | ≥ 97 % |
| G2 | Per-class consistency (rare classes) on unseen cameras | rare classes missed | coarse | most classes within CI | all classes within CI |
| G3 | Calibration effort needed for a new camera | full retrain | per-camera fine-tune | line/ROI + homography only | line/ROI only |
| G4 | Envelope monitor correctly flags unsupported cameras/views | never | half | most | all fixtures |
| G5 | Direction-wise count correctness on unseen cameras | wrong | partially | mostly | fully |

Two independent raters score each; report Cohen's κ / ICC and resolve disagreements by documented discussion.

## 6. Negative tests

All five share the PDF contract: *Trigger → detect → abstain or approved degraded mode → label affected result → prevent
silent use → Recovery: restore accepted preprocessing and model bundle; rerun affected scene class; retain failure example
→ Evidence: fixture ID, precondition, version/config, expected vs observed, logs, recovery time, residual-risk decision.*

| ID | Condition | Planned fixture | Planned detector |
|----|-----------|-----------------|------------------|
| NT-1 | Occlusion in dense traffic | Real dense clips + synthetic occluder overlays at controlled rates | Box-overlap ratio, detections/area, track fragmentation & ID-switch rate above calibrated thresholds |
| NT-2 | Night and rain degradation | Real night/rain clips + controlled synthetic darkening/rain streaks/blur on clear clips (applied only inside the test partition) | Luminance histogram, contrast, blur (Laplacian variance), rain-streak score, detector confidence distribution shift |
| NT-3 | Camera-specific calibration | Clip run with wrong/perturbed homography or counting line from another camera; camera shift/rotation | Homography reprojection error on reference points, background-keypoint drift vs. calibration snapshot, implausible ground-plane speeds |
| NT-4 | Average accuracy hides failure on small, rare or safety-critical targets | Fixture where aggregate count is good but motorcycles/buses are badly miscounted | Per-class and small-object slice checks; report fails if any critical slice breaches its threshold even when aggregate passes |
| NT-5 | Pre-processing/augmentation leaks across evidence partitions | Deliberately fit normalisation/enhancement statistics or forecaster scaler on acceptance data; duplicate frames across splits | Partition hash registry, perceptual-hash duplicate scan across splits, provenance check that every fitted artefact lists only development IDs |

## 7. Deliverables

| ID | Deliverable (PDF) | Where it will live |
|----|-------------------|--------------------|
| D1 | Problem charter, requirements, system boundary, architecture, source-to-claim map | PROJECT_ANALYSIS.md, REQUIREMENTS.md, SYSTEM_ARCHITECTURE.md |
| D2 | Reproducible O2 reference with fixtures/protocol, oracle, tests, baseline results, resource profile | `scripts/run_o2_reference.py`, `tests/acceptance`, `reports/o2/` |
| D3 | Independently testable O3 with rationale, comparative evidence, trade-offs, limitations | `src/enhancement/`, `reports/o3/`, EXPERIMENT_LOG.md |
| D4 | Complete prototype with configuration, interfaces, telemetry, degraded operation, recovery | `src/api/`, `dashboard/`, Docker Compose |
| D5 | Acceptance harness covering AC-1…AC-4, all KPIs, every negative test | `tests/acceptance`, `tests/negative`, `scripts/run_acceptance.py` |
| D6 | Versioned repo, evidence manifest, installation/operating guide, API docs, reproducibility | Git repo, `data/manifests/`, INSTALLATION_GUIDE.md, USER_GUIDE.md, API_DOCUMENTATION.md |
| D7 | Final report, research paper, poster, portfolio case study, demo video, video logs, individual contribution/viva evidence | `reports/final/` — demo video and individual evidence are produced by the team |
