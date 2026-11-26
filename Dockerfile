# Clean-environment build (NFR-5, reproducibility contract). CPU-only.
# Phase 4: builds the locked environment and runs the test suite.
# Phase 11/13 add the API and dashboard entrypoints (see docker-compose.yml).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    DSCI41_DATA_ROOT=/data \
    DSCI41_MODELS_ROOT=/models \
    DSCI41_LOG_DIR=/logs \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project

COPY . .
CMD ["python", "-m", "pytest", "-q"]
