"""DSCI-41 traffic intelligence source package."""

import os

# Never let Ultralytics pip-install packages at runtime: every dependency must come from uv.lock (reproducibility).
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
