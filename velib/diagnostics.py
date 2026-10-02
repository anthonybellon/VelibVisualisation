"""Holdout diagnostics for the debugging dashboard (`index.html`).

Everything here is computed on the holdout window with the model trained
*without* it, so it shows where and when the model is wrong, not how well it
memorised the training data:

- error by station, by local hour and by day of week, for the model and both baselines;
- hourly actual vs predicted series per station over the holdout window;
- permutation feature importance;
- data coverage per station.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from velib.config import RANDOM_STATE
from velib.features import FEATURES, TARGET
from velib.preprocessing import HourlyData

logger = logging.getLogger(__name__)

METHODS = ("model", "seasonal_naive", "weekly_profile")
IMPORTANCE_SAMPLE_ROWS = 20_000
IMPORTANCE_REPEATS = 3


def _clean(values: Any, digits: int = 2) -> list[float | None]:
    """Round to `digits` and turn NaN into None so the result is valid JSON."""
    array = np.asarray(values, dtype=float)
    return [None if np.isnan(v) else round(float(v), digits) for v in array]


def _predictions(model: Any, rows: pd.DataFrame) -> pd.DataFrame:
    capacity = rows["capacity"].to_numpy()
    ratio = np.clip(model.predict(rows[FEATURES]), 0.0, None)
    return pd.DataFrame(
        {
            "time": rows["time"].to_numpy(),
            "stationcode": rows["stationcode"].to_numpy(),
            "hour": rows["hour"].to_numpy(),
            "day_of_week": rows["day_of_week"].to_numpy(),
            "actual": rows["bikes"].to_numpy(),
            "model": ratio * capacity,
            "seasonal_naive": rows["fill_lag_1w"].to_numpy() * capacity,
            "weekly_profile": rows["fill_profile"].to_numpy() * capacity,
        }
    )


def _errors_by(pred: pd.DataFrame, key: str, size: int) -> dict[str, list[float | None]]:
    comparable = pred.dropna(subset=list(METHODS))
    out = {}
    for method in METHODS:
        abs_err = (comparable[method] - comparable["actual"]).abs()
        out[method] = _clean(abs_err.groupby(comparable[key]).mean().reindex(range(size)), 3)
    return out


def _station_table(pred: pd.DataFrame, hourly: HourlyData) -> list[dict[str, Any]]:
    comparable = pred.dropna(subset=list(METHODS))
    abs_err = comparable[list(METHODS)].sub(comparable["actual"], axis=0).abs()
    abs_err["stationcode"] = comparable["stationcode"].to_numpy()
    mae = abs_err.groupby("stationcode").mean()
    rows = comparable.groupby("stationcode").size()
    mean_bikes = comparable.groupby("stationcode")["actual"].mean()

    meta = hourly.stations
    grid_hours = len(hourly.index)
    table = []
    for code in meta.index:
        info = meta.loc[code]
        if pd.isna(info["lat"]) or pd.isna(info["lon"]):
            continue
        entry: dict[str, Any] = {
            "code": str(code),
            "name": str(info["name"]),
            "lat": round(float(info["lat"]), 6),
            "lon": round(float(info["lon"]), 6),
            "capacity": int(info["capacity"]),
            "coverage": round(float(info["observed_hours"]) / grid_hours, 4),
            "test_rows": int(rows.get(code, 0)),
        }
        if code in mae.index:
            scores = {m: round(float(mae.loc[code, m]), 3) for m in METHODS}
            best_baseline = min(scores["seasonal_naive"], scores["weekly_profile"])
            entry["mae"] = scores
            entry["mean_bikes"] = round(float(mean_bikes.loc[code]), 2)
            entry["skill"] = (
                round(1 - scores["model"] / best_baseline, 4) if best_baseline > 0 else None
            )
        else:
            entry["mae"] = None
            entry["skill"] = None
        table.append(entry)
    return table


def _series(pred: pd.DataFrame, index: pd.DatetimeIndex) -> dict[str, dict[str, list]]:
    series: dict[str, dict[str, list]] = {}
    for code, rows in pred.groupby("stationcode", sort=True):
        aligned = rows.set_index("time").reindex(index)
        series[str(code)] = {name: _clean(aligned[name], 1) for name in ("actual", *METHODS)}
    return series


def _importance(model: Any, test_rows: pd.DataFrame) -> list[dict[str, Any]]:
    sample = test_rows.dropna(subset=[TARGET])
    if len(sample) > IMPORTANCE_SAMPLE_ROWS:
        sample = sample.sample(IMPORTANCE_SAMPLE_ROWS, random_state=RANDOM_STATE)
    result = permutation_importance(
        model,
        sample[FEATURES],
        sample[TARGET],
        scoring="neg_mean_absolute_error",
        n_repeats=IMPORTANCE_REPEATS,
        random_state=RANDOM_STATE,
    )
    ranked = sorted(
        zip(FEATURES, result.importances_mean, result.importances_std, strict=True),
        key=lambda item: -item[1],
    )
    # Increase in MAE of the fill ratio, expressed in percentage points of capacity.
    return [
        {"feature": name, "mae_increase_pp": round(100 * mean, 3), "std_pp": round(100 * std, 3)}
        for name, mean, std in ranked
    ]


def build_diagnostics(
    model: Any,
    test_rows: pd.DataFrame,
    hourly: HourlyData,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the diagnostics document from the holdout model and holdout rows."""
    pred = _predictions(model, test_rows)
    start, end = test_rows["time"].min(), test_rows["time"].max()
    index = pd.date_range(start, end, freq="h", tz="UTC")
    logger.info("Computing permutation importance on the holdout")
    return {
        "test_start": start.isoformat(),
        "test_end": end.isoformat(),
        "series_start": start.isoformat(),
        "series_hours": len(index),
        "metrics": metrics,
        "by_hour": _errors_by(pred, "hour", 24),
        "by_day_of_week": _errors_by(pred, "day_of_week", 7),
        "stations": _station_table(pred, hourly),
        "series": _series(pred, index),
        "importance": _importance(model, test_rows),
        "data": {
            "grid_start": hourly.index[0].isoformat(),
            "grid_end": hourly.index[-1].isoformat(),
            "grid_hours": len(hourly.index),
            "stations": int(hourly.bikes.shape[1]),
            "observed_share": round(float(hourly.bikes.notna().to_numpy().mean()), 4),
        },
    }
