"""Short-term volume forecasting baselines with rolling-origin evaluation (FR-4, Module 8 starting point).

Inputs are ESTIMATED interval counts from the counting stage, never reference counts. Outputs are PREDICTED values.
These baselines are the mandatory comparators for any learned model (Random Forest / XGBoost come in Phase 10, once
there is enough history and real congestion labels from the Barbados Train.csv).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

Forecaster = Callable[[np.ndarray, int], float]  # (history, horizon) -> forecast for t+horizon


def persistence(history: np.ndarray, horizon: int) -> float:
    """Last observed value."""
    return float(history[-1])


def moving_average(window: int) -> Forecaster:
    def forecast(history: np.ndarray, horizon: int) -> float:
        return float(history[-window:].mean())

    forecast.__name__ = f"moving_average_{window}"
    return forecast


def linear_trend(window: int) -> Forecaster:
    """Least-squares line through the last `window` points, extrapolated and clipped at zero."""

    def forecast(history: np.ndarray, horizon: int) -> float:
        y = history[-window:]
        if len(y) < 2:
            return float(y[-1])
        x = np.arange(len(y))
        slope, intercept = np.polyfit(x, y, 1)
        return max(float(intercept + slope * (len(y) - 1 + horizon)), 0.0)

    forecast.__name__ = f"linear_trend_{window}"
    return forecast


DEFAULT_MODELS: dict[str, Forecaster] = {
    "persistence": persistence,
    "moving_average_3": moving_average(3),
    "moving_average_5": moving_average(5),
    "linear_trend_5": linear_trend(5),
}


@dataclass(frozen=True)
class BacktestResult:
    model: str
    horizon: int
    n: int
    mae: float
    rmse: float


def rolling_origin_backtest(
    series: Sequence[float],
    models: dict[str, Forecaster] | None = None,
    horizons: Sequence[int] = (1, 2, 3, 5),
    min_history: int = 5,
) -> list[BacktestResult]:
    """For every origin t >= min_history, forecast t+h from data up to t; score against the observed value."""
    y = np.asarray(series, dtype=float)
    models = models or DEFAULT_MODELS
    results: list[BacktestResult] = []
    for name, model in models.items():
        for h in horizons:
            errors = [model(y[: t + 1], h) - y[t + h] for t in range(min_history - 1, len(y) - h)]
            if not errors:
                continue
            e = np.asarray(errors)
            results.append(BacktestResult(name, h, len(e), float(np.abs(e).mean()), float(np.sqrt((e**2).mean()))))
    return results


def forecast_ahead(series: Sequence[float], model: Forecaster, horizons: Sequence[int]) -> list[float]:
    y = np.asarray(series, dtype=float)
    return [model(y, h) for h in horizons]
