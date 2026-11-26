# CLAUDE.md — working rules for this repository

Project: DSCI-41 capstone. Spec = `docs/UseCase.pdf` (single source of truth). Requirements extracted in
REQUIREMENTS.md; current phase in DEVELOPMENT_ROADMAP.md.

## Environment
- Python 3.11 venv lives at `%USERPROFILE%\dsci41\venv` (outside OneDrive). Run tools with
  `C:/Users/pavan/dsci41/venv/Scripts/python.exe -m pytest -q`.
- Use `uv` with `UV_PROJECT_ENVIRONMENT` and `UV_PYTHON` pointing at that venv. Never let uv create its own venv
  or download its own CPython: Smart App Control blocks unsigned interpreters/launchers (RISK R-14).
- CPU only (no CUDA). Disk is tight (~10–15 GB free): ask before any download larger than ~500 MB.

## Integrity rules (non-negotiable)
- Never fabricate metrics, annotations, reference counts, congestion labels or speeds.
- Every count carries a `CountSource` (reference / estimated / predicted / demo). Demo data is never evidence.
- O2 and O3 run the same pipeline; O3 differs only in the `robustness:` config block. Compare only under the same
  resource envelope (`configs/resource_envelope.yaml`).
- The AC-3 acceptance partition is locked before O3 tuning and must never be used to fit or tune anything.
- KPI-2/KPI-3 are rubric scores by two independent human raters, not MAE/RMSE.
- Do not mark a feature complete until it is tested; report failures with their output.

## Code conventions
- Type hints everywhere; Pydantic for configs and data contracts; `extra="forbid"` on configs.
- No hardcoded paths: use `src.common.config.Settings`.
- Each module is independent and tested under `tests/unit`; integration tests go in `tests/integration`.
- Ask before large dataset downloads, paid services, destructive operations or long GPU/CPU training runs.
