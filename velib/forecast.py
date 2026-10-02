"""Forecast the next week and export it (dashboard + external front end).

Forecasting reuses `build_features` unchanged: the hourly grid is extended with
`HORIZON_HOURS` empty future rows, and because every feature only looks at
least `HORIZON_HOURS` back, all future rows get complete features.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from velib.config import (
    EXTRA_CAPACITY_STATIONS,
    HORIZON_HOURS,
    INACTIVE_STATION_DAYS,
    LEGACY_EXPORT_TIMEZONE,
    LOW_CONFIDENCE_HISTORY_HOURS,
    TIMEZONE,
)
from velib.features import FEATURES, build_features
from velib.neighbours import neighbour_adjacency
from velib.preprocessing import HourlyData

logger = logging.getLogger(__name__)

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def forecast(hourly: HourlyData, bundle: dict[str, Any]) -> pd.DataFrame:
    """Predict every active station for the `HORIZON_HOURS` after the last grid hour.

    Stations not seen in the `INACTIVE_STATION_DAYS` before the origin are left
    out: their weekly profile never expires, so they would otherwise keep
    getting plausible-looking forecasts long after being removed.

    Returns one row per station and future hour with `time` (UTC), local
    `day_of_week` / `hour`, `fill_ratio_pred` and `bikes_pred`.
    """
    start = hourly.index[-1] + pd.Timedelta(hours=1)
    extended = hourly.extend(HORIZON_HOURS)
    adjacency = neighbour_adjacency(hourly.stations)
    frame = build_features(extended, adjacency, start=start, require_target=False)

    last_seen = hourly.stations["last_seen"]
    active = last_seen[last_seen >= start - pd.Timedelta(days=INACTIVE_STATION_DAYS)].index
    dropped = len(last_seen) - len(active)
    if dropped:
        logger.info("Skipping %d stations not seen for %d days", dropped, INACTIVE_STATION_DAYS)
    frame = frame[frame["stationcode"].isin(active)].reset_index(drop=True)

    # The fill ratio is converted with the capacity known at the origin (the
    # exports also express percentages of it); a week-old capacity is only the
    # fallback when the latest one is unusable (0, e.g. during maintenance).
    latest = hourly.stations["capacity"].reindex(frame["stationcode"]).to_numpy(dtype=float)
    lagged = frame["capacity"].to_numpy(dtype=float)
    capacity = np.where(latest > 0, latest, np.where(lagged > 0, lagged, 0.0))

    ratio = np.clip(bundle["model"].predict(frame[FEATURES]), 0.0, None)
    result = frame[["time", "stationcode", "day_of_week", "hour"]].copy()
    result["fill_ratio_pred"] = ratio
    result["bikes_pred"] = ratio * capacity
    logger.info(
        "Forecast %d stations from %s to %s",
        result["stationcode"].nunique(),
        result["time"].min(),
        result["time"].max(),
    )
    return result


def _week_grid(station_rows: pd.DataFrame, tz: str = TIMEZONE) -> tuple[np.ndarray, int]:
    """7 x 24 array (day of week x hour, in time zone `tz`) of predicted bikes.

    A daylight-saving change makes one local hour appear twice (averaged) or not
    at all (interpolated from its neighbours within the day). Also returns how
    many cells had no prediction before interpolation.
    """
    local = station_rows["time"].dt.tz_convert(tz)
    grid = (
        station_rows.groupby([local.dt.dayofweek.rename("day"), local.dt.hour.rename("hour")])[
            "bikes_pred"
        ]
        .mean()
        .unstack("hour")
        .reindex(index=range(7), columns=range(24))
    )
    missing = int(grid.isna().to_numpy().sum())
    grid = grid.interpolate(axis=1, limit_direction="both").fillna(0.0)
    return grid.to_numpy(), missing


def _week_matrix(station_rows: pd.DataFrame) -> list[list[float]]:
    grid, _ = _week_grid(station_rows)
    return [[round(float(v), 1) for v in row] for row in grid]


def export_forecast(
    predictions: pd.DataFrame,
    hourly: HourlyData,
    bundle: dict[str, Any],
    synthetic: bool = False,
) -> dict[str, Any]:
    """Build the forecast document read by the debugging dashboard (`index.html`)."""
    start = predictions["time"].min()
    end = predictions["time"].max()
    local_start = start.tz_convert(TIMEZONE)

    # Date of each local weekday's first occurrence inside the forecast window.
    # (Calendar days, not 24 h steps, which go wrong across a DST change.) If the
    # window starts mid-day, that weekday's earlier hours fall a week later;
    # the dashboard labels individual cells from `forecast_start`.
    day_dates: dict[int, str] = {}
    for offset in range(8):
        day = local_start.date() + timedelta(days=offset)
        day_dates.setdefault(day.weekday(), day.isoformat())

    stations = []
    meta = hourly.stations
    for code, rows in predictions.groupby("stationcode", sort=True):
        info = meta.loc[code]
        if pd.isna(info["lat"]) or pd.isna(info["lon"]):
            continue
        stations.append(
            {
                "code": str(code),
                "name": str(info["name"]),
                "lat": round(float(info["lat"]), 6),
                "lon": round(float(info["lon"]), 6),
                "capacity": int(info["capacity"]),
                "extra_capacity": code in EXTRA_CAPACITY_STATIONS,
                "is_renting": bool(info["is_renting"]),
                "low_confidence": int(info["observed_hours"]) < LOW_CONFIDENCE_HISTORY_HOURS,
                "last_seen": info["last_seen"].isoformat(),
                "bikes": _week_matrix(rows),
            }
        )

    metrics = bundle.get("metrics", {})
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "synthetic": synthetic,
        "timezone": TIMEZONE,
        "forecast_start": start.isoformat(),
        "forecast_end": end.isoformat(),
        "day_names": DAY_NAMES,
        "day_dates": [day_dates.get(d) for d in range(7)],
        "model": {
            "trained_until": bundle.get("trained_until"),
            "test_start": metrics.get("test_start"),
            "test_end": metrics.get("test_end"),
            "scores": metrics.get("scores"),
            "skill_vs_best_baseline": metrics.get("skill_vs_best_baseline"),
        },
        "stations": stations,
    }


def export_legacy(
    predictions: pd.DataFrame, hourly: HourlyData, tz: str = LEGACY_EXPORT_TIMEZONE
) -> dict[str, Any]:
    """The compressed format written by the old `prediction_compression.py`.

    External front ends read this file, so its schema is frozen:

    ```
    {"normal_capacity": [station, ...], "extra_capacity": [station, ...]}
    station = {
        "stationcode", "name", "capacity", "is_renting" ("OUI"/"NON"),
        "coordonnees_geo": {"lon", "lat"}, "missing_predictions",
        "predictions": {"0".."6": [24 x int percent of capacity]},   # Monday = "0"
        # day/hour keys are in `tz` (LEGACY_EXPORT_TIMEZONE; v1 used UTC)
        # extra_capacity only: raw predicted bikes before rounding
        "extra_capacity_predictions": {"0".."6": [24 x float]},
    }
    ```

    In `extra_capacity`, stations listed in `EXTRA_CAPACITY_STATIONS` use twice
    their published capacity; all others are identical to `normal_capacity`.
    """
    normal, extra = [], []
    meta = hourly.stations
    for code, rows in predictions.groupby("stationcode", sort=True):
        info = meta.loc[code]
        if pd.isna(info["lat"]) or pd.isna(info["lon"]):
            continue
        grid, missing = _week_grid(rows, tz)
        capacity = int(info["capacity"])
        base = {
            "stationcode": str(code),
            "name": str(info["name"]),
            "is_renting": "OUI" if info["is_renting"] else "NON",
            "coordonnees_geo": {"lon": float(info["lon"]), "lat": float(info["lat"])},
            "missing_predictions": missing,
        }
        rounded = np.rint(grid)
        for target, cap in (
            (normal, capacity),
            (extra, capacity * 2 if code in EXTRA_CAPACITY_STATIONS else capacity),
        ):
            percent = np.rint(rounded / cap * 100) if cap > 0 else np.zeros_like(rounded)
            station = {
                **base,
                "capacity": cap,
                "predictions": {str(d): [int(v) for v in percent[d]] for d in range(7)},
            }
            if target is extra:
                station["extra_capacity_predictions"] = {
                    str(d): [float(v) for v in grid[d]] for d in range(7)
                }
            target.append(station)
    return {"normal_capacity": normal, "extra_capacity": extra}
