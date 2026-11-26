# Hugging Face Space (Docker SDK) for the DSCI-41 dashboard. Built from the bundle made by scripts/build_hf_space.py.
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv

# Spaces run the container as UID 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.venv/bin:$PATH \
    UV_PROJECT_ENVIRONMENT=/home/user/.venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DSCI41_HOSTED=true \
    DSCI41_DATA_ROOT=/tmp/dsci41/data \
    DSCI41_MODELS_ROOT=/home/user/models \
    DSCI41_LOG_DIR=/tmp/dsci41/logs \
    DSCI41_REPORTS_ROOT=/tmp/dsci41/reports \
    DSCI41_DEVICE=cpu \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    YOLO_AUTOINSTALL=false

WORKDIR /home/user/app
COPY --chown=user pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project
COPY --chown=user . .

# Fetch the detector weights at build time so the first visitor does not wait for a download.
RUN python -c "import src; from src.common.config import Settings; from src.detection.detector import ensure_weights; print(ensure_weights('yolo11n.pt', Settings().models_root))"

EXPOSE 7860
CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port", "7860", "--server.address", "0.0.0.0", \
     "--server.headless", "true", "--server.maxUploadSize", "200", \
     "--server.enableXsrfProtection", "false", "--server.enableCORS", "false", \
     "--browser.gatherUsageStats", "false"]
