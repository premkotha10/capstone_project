"""Verify the development environment: Python version, key imports, FFmpeg, hardware and paths.

Usage:  python scripts/check_env.py [--json]

Exit code 0 when every required check passes, 1 otherwise.

Windows Smart App Control can block an unsigned native DLL the first time it loads and allow it on a retry
(RISK_REGISTER R-14), so each import is retried once before it is reported as failed.
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import Settings  # noqa: E402
from src.common.hardware import detect_hardware, select_device  # noqa: E402

REQUIRED_MODULES = (
    "numpy",
    "pandas",
    "cv2",
    "torch",
    "torchvision",
    "ultralytics",
    "lap",
    "sklearn",
    "xgboost",
    "fastapi",
    "pydantic",
    "streamlit",
    "plotly",
    "psutil",
    "yaml",
)
APP_CONTROL_MARKER = "Application Control policy"


def try_import(name: str, retries: int = 1) -> tuple[bool, str]:
    for attempt in range(retries + 1):
        try:
            module = importlib.import_module(name)
            return True, str(getattr(module, "__version__", "ok"))
        except Exception as exc:  # report every import failure, not only ImportError
            message = f"{type(exc).__name__}: {exc}"
            if APP_CONTROL_MARKER in message and attempt < retries:
                sys.modules.pop(name, None)
                time.sleep(1.0)
                continue
            return False, message
    return False, "unreachable"


def ffmpeg_version() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe is None:
        return None
    out = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=30).stdout
    return out.splitlines()[0] if out else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="print a JSON report")
    args = parser.parse_args()

    settings = Settings()
    results: dict[str, object] = {"python": sys.version.split()[0]}
    failures: list[str] = []

    if sys.version_info[:2] != (3, 11):
        failures.append(f"python {results['python']} (expected 3.11)")

    imports: dict[str, str] = {}
    for name in REQUIRED_MODULES:
        ok, info = try_import(name)
        imports[name] = info
        if not ok:
            failures.append(f"import {name}: {info}")
    results["imports"] = imports

    results["ffmpeg"] = ffmpeg_version()
    if results["ffmpeg"] is None:
        failures.append("ffmpeg not found on PATH")

    hw = detect_hardware()
    results["hardware"] = hw.__dict__
    results["selected_device"] = select_device(settings.device, hw)

    paths = {}
    for label in ("data_root", "models_root", "log_dir"):
        path = Path(getattr(settings, label))
        paths[label] = {"path": str(path), "exists": path.exists()}
    results["paths"] = paths
    free_gb = round(shutil.disk_usage(Path.home()).free / 1024**3, 1)
    results["disk_free_gb"] = free_gb
    if free_gb < 5:
        failures.append(f"low disk space: {free_gb} GB free")
    results["failures"] = failures

    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        print(f"Python          {results['python']}")
        for name, info in imports.items():
            print(f"  {name:<14}{info}")
        print(f"FFmpeg          {results['ffmpeg']}")
        print(f"CPU cores       {hw.cpu_physical_cores} physical / {hw.cpu_logical_cores} logical")
        print(f"RAM             {hw.ram_total_gb} GB")
        print(f"CUDA            {hw.cuda_available} {hw.cuda_devices}")
        print(f"XPU             {hw.xpu_available}")
        print(f"OpenVINO        {hw.openvino_devices or 'not installed'}")
        print(f"Selected device {results['selected_device']}")
        for label, info in paths.items():
            print(f"{label:<16}{info['path']}  (exists={info['exists']})")
        print(f"Disk free       {free_gb} GB")
        print("RESULT          " + ("PASS" if not failures else "FAIL:\n  - " + "\n  - ".join(failures)))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
