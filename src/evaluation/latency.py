"""Latency statistics following the KPI-6 method (monotonic clock, p50/p95/p99, failures)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class LatencyStats:
    trials: int
    failures: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def latency_stats(samples_ms: Sequence[float], failures: int = 0) -> LatencyStats:
    if not samples_ms:
        raise ValueError("no latency samples")
    arr = np.asarray(samples_ms, dtype=float)
    p50, p95, p99 = np.percentile(arr, [50, 95, 99])
    return LatencyStats(
        trials=len(arr),
        failures=failures,
        mean_ms=float(arr.mean()),
        p50_ms=float(p50),
        p95_ms=float(p95),
        p99_ms=float(p99),
        max_ms=float(arr.max()),
    )
