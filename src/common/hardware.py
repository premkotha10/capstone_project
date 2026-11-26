"""Compute-device discovery and selection with CPU fallback (dev rule 11)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import psutil

logger = logging.getLogger(__name__)

VALID_PREFERENCES = ("auto", "cpu", "cuda", "xpu", "openvino")


@dataclass(frozen=True)
class HardwareReport:
    cpu_physical_cores: int | None
    cpu_logical_cores: int | None
    ram_total_gb: float
    cuda_available: bool
    cuda_devices: list[str] = field(default_factory=list)
    xpu_available: bool = False
    openvino_devices: list[str] = field(default_factory=list)


def _torch_status() -> tuple[bool, list[str], bool]:
    try:
        import torch
    except ImportError:
        return False, [], False
    cuda = bool(torch.cuda.is_available())
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda else []
    xpu = bool(getattr(torch, "xpu", None) is not None and torch.xpu.is_available())
    return cuda, names, xpu


def _openvino_devices() -> list[str]:
    try:
        import openvino as ov  # optional extra
    except ImportError:
        return []
    try:
        return list(ov.Core().available_devices)
    except Exception as exc:  # driver/runtime problems should not crash discovery
        logger.warning("OpenVINO device query failed: %s", exc)
        return []


def detect_hardware() -> HardwareReport:
    cuda, cuda_names, xpu = _torch_status()
    return HardwareReport(
        cpu_physical_cores=psutil.cpu_count(logical=False),
        cpu_logical_cores=psutil.cpu_count(logical=True) or os.cpu_count(),
        ram_total_gb=round(psutil.virtual_memory().total / 1024**3, 1),
        cuda_available=cuda,
        cuda_devices=cuda_names,
        xpu_available=xpu,
        openvino_devices=_openvino_devices(),
    )


def select_device(preference: str = "auto", report: HardwareReport | None = None) -> str:
    """Resolve a device preference to an available backend, falling back to CPU with a warning."""
    if preference not in VALID_PREFERENCES:
        raise ValueError(f"unknown device preference {preference!r}; expected one of {VALID_PREFERENCES}")
    if preference == "cpu":
        return "cpu"
    report = report or detect_hardware()
    available = {
        "cuda": report.cuda_available,
        "xpu": report.xpu_available,
        "openvino": bool(report.openvino_devices),
    }
    if preference == "auto":
        for candidate in ("cuda", "xpu"):
            if available[candidate]:
                return candidate
        return "cpu"
    if available[preference]:
        return preference
    logger.warning("Requested device %r is not available; falling back to CPU", preference)
    return "cpu"
