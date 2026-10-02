"""Leak-free features for forecasting bike availability up to one week ahead.

The rule
--------
A forecast made at origin `o` for target time `t` (with `0 < t - o <= HORIZON_HOURS`)
may only use observations from times `<= o`. To make one model valid for every
horizon, **every observation-based feature at time `t` uses data from
`t - HORIZON_HOURS` or earlier**. Calendar features and station metadata are
known in advance and are exempt.

That single rule is what makes training and forecasting identical: the forecast
is just `build_features` on the hourly grid extended with empty future rows.
`tests/test_features.py` checks it directly by perturbing future values and
asserting that no feature before `t + HORIZON_HOURS` changes.

The target is the fill ratio `bikes / capacity`, so a single global model can
learn from every station regardless of its size.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from functools import cache

import numpy as np
import pandas as pd
from scipy import sparse

from velib.config import (
    HORIZON_HOURS,
    LAG_WEEKS,
    MIN_PERIODS_DAY,
    MIN_PERIODS_WEEK,
    TIMEZONE,
)
from velib.neighbours import neighbour_adjacency, neighbour_mean
from velib.preprocessing import HourlyData

logger = logging.getLogger(__name__)

CALENDAR_FEATURES = ["hour", "day_of_week", "is_weekend", "is_holiday", "month"]
STATION_FEATURES = ["capacity", "lat", "lon"]
HISTORY_FEATURES = [
    *(f"fill_lag_{w}w" for w in LAG_WEEKS),
    "fill_same_hour_mean",
    "fill_profile",
    "empty_profile",
    "full_profile",
    "profile_weeks",
    "fill_day_mean_lag",
    "fill_week_mean_lag",
]
NEIGHBOUR_FEATURES = [
    "nbr_fill_lag_1w",
    "nbr_fill_profile",
    "nbr_empty_lag_1w",
    "nbr_full_lag_1w",
]
FEATURES = CALENDAR_FEATURES + STATION_FEATURES + HISTORY_FEATURES + NEIGHBOUR_FEATURES
TARGET = "fill_ratio"


# ------------------------------------------------------------------------------
# Calendar
# ------------------------------------------------------------------------------


def _easter(year: int) -> date:
    """Gregorian Easter Sunday (anonymous Gregorian algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    m = (32 + 2 * e + 2 * i - h - k) % 7
    n = (a + 11 * h + 22 * m) // 451
    month, day = divmod(h + m - 7 * n + 114, 31)
    return date(year, month, day + 1)


@cache
def french_holidays(year: int) -> frozenset[date]:
    """French public holidays for a year."""
    easter = _easter(year)
    fixed = [(1, 1), (5, 1), (5, 8), (7, 14), (8, 15), (11, 1), (11, 11), (12, 25)]
    movable = [easter + timedelta(days=n) for n in (1, 39, 50)]  # Easter Mon, Ascension, Whit Mon
    return frozenset([date(year, m, d) for m, d in fixed] + movable)


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar features in local (Paris) time for a UTC index."""
    local = index.tz_convert(TIMEZONE)
    local_dates = local.date
    return pd.DataFrame(
        {
            "hour": local.hour,
            "day_of_week": local.dayofweek,
            "is_weekend": (local.dayofweek >= 5).astype(int),
            "is_holiday": [int(d in french_holidays(d.year)) for d in local_dates],
            "month": local.month,
        },
        index=index,
    )


# ------------------------------------------------------------------------------
# History-based (wide, time x station) features
# ------------------------------------------------------------------------------


def slot_profile(values: pd.DataFrame, horizon: int = HORIZON_HOURS):
    """Expanding mean of past values in the same local weekly slot, and how many it uses.

    A slot is a local (Paris) weekday and hour, so the 08:00 profile stays 08:00
    across daylight-saving changes. Row `t` averages every earlier occurrence of
    its slot that is at least `horizon` hours old (non-missing values only).
    Normally the newest one is exactly a week old. In the week after the clocks
    go forward it would be 167 h old, so it is skipped, and the profile uses
    occurrences from two weeks back and earlier.
    """
    index = values.index
    local = index.tz_convert(TIMEZONE)
    slot = np.asarray(local.dayofweek * 24 + local.hour)

    totals = values.fillna(0.0).groupby(slot).cumsum()
    counts = values.notna().astype(float).groupby(slot).cumsum()

    times = pd.Series(index, index=index)
    previous = times.groupby(slot).shift(1)
    # True where the previous occurrence of this slot is too recent to be used.
    too_recent = ((times - previous) < pd.Timedelta(hours=horizon)).to_numpy()[:, None]

    def before(frame: pd.DataFrame) -> pd.DataFrame:
        one = frame.groupby(slot).shift(1).to_numpy()
        two = frame.groupby(slot).shift(2).to_numpy()
        return pd.DataFrame(
            np.where(too_recent, two, one), index=index, columns=values.columns
        ).fillna(0.0)

    totals, counts = before(totals), before(counts)
    return totals / counts.where(counts > 0), counts


def _mean_ignoring_nan(frames: list[pd.DataFrame]) -> pd.DataFrame:
    totals = sum(f.fillna(0.0) for f in frames)
    counts = sum(f.notna().astype(float) for f in frames)
    return totals / counts.where(counts > 0)


def _wide_history_features(hourly: HourlyData, adjacency: sparse.csr_matrix, horizon: int):
    """Yield (name, time x station frame) for every observation-based feature."""
    bikes = hourly.bikes
    capacity = hourly.capacity
    observed = bikes.notna()
    fill = bikes / capacity.where(capacity > 0)
    empty = (bikes <= 0).astype(float).where(observed)
    full = (bikes >= capacity).astype(float).where(observed)

    lags = []
    for w in LAG_WEEKS:
        lag = fill.shift(w * horizon)
        lags.append(lag)
        yield f"fill_lag_{w}w", lag
    yield "fill_same_hour_mean", _mean_ignoring_nan(lags)
    del lags

    profile, weeks = slot_profile(fill, horizon)
    yield "fill_profile", profile
    yield "profile_weeks", weeks
    yield "empty_profile", slot_profile(empty, horizon)[0]
    yield "full_profile", slot_profile(full, horizon)[0]

    yield "fill_day_mean_lag", fill.rolling(24, min_periods=MIN_PERIODS_DAY).mean().shift(horizon)
    yield (
        "fill_week_mean_lag",
        fill.rolling(horizon, min_periods=MIN_PERIODS_WEEK).mean().shift(horizon),
    )

    nbr_fill = neighbour_mean(fill, adjacency)
    yield "nbr_fill_lag_1w", nbr_fill.shift(horizon)
    yield "nbr_fill_profile", slot_profile(nbr_fill, horizon)[0]
    yield "nbr_empty_lag_1w", neighbour_mean(empty, adjacency).shift(horizon)
    yield "nbr_full_lag_1w", neighbour_mean(full, adjacency).shift(horizon)

    # Capacity is metadata, but it can change, so use the value known at the origin.
    yield "capacity", capacity.shift(horizon)

    yield TARGET, fill
    yield "bikes", bikes


# ------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------


def build_features(
    hourly: HourlyData,
    adjacency: sparse.csr_matrix | None = None,
    start: pd.Timestamp | None = None,
    require_target: bool = True,
    horizon: int = HORIZON_HOURS,
) -> pd.DataFrame:
    """Build the long (one row per station-hour) feature frame.

    Args:
        hourly: Hourly grid. For forecasting, pass a grid extended with empty
            future rows (`HourlyData.extend`).
        adjacency: Station adjacency (defaults to `neighbour_adjacency(hourly.stations)`).
        start: Only emit rows at or after this time.
        require_target: Only emit rows with an observed target and at least one
            week of same-slot history (training/evaluation). Set to False to
            emit every row (forecasting).
        horizon: Minimum age of any observation used by a feature.

    Returns:
        DataFrame with `time`, `stationcode`, every name in `FEATURES`, the
        target `fill_ratio` and the observed `bikes`.
    """
    if adjacency is None:
        adjacency = neighbour_adjacency(hourly.stations)

    index = hourly.index
    stations = hourly.bikes.columns
    n_times, n_stations = len(index), len(stations)

    keep = np.ones((n_times, n_stations), dtype=bool)
    if start is not None:
        keep &= np.asarray(index >= start)[:, None]
    if require_target:
        fill = hourly.bikes / hourly.capacity.where(hourly.capacity > 0)
        _, weeks = slot_profile(fill, horizon)
        keep &= fill.notna().to_numpy() & (weeks.to_numpy() > 0)
    flat_keep = keep.ravel()

    time_idx = np.repeat(np.arange(n_times), n_stations)[flat_keep]
    station_idx = np.tile(np.arange(n_stations), n_times)[flat_keep]

    columns: dict[str, np.ndarray] = {
        "time": index[time_idx],
        "stationcode": stations.to_numpy()[station_idx],
    }

    calendar = calendar_features(index)
    for name in CALENDAR_FEATURES:
        columns[name] = calendar[name].to_numpy()[time_idx]

    meta = hourly.stations.reindex(stations)
    for name in ("lat", "lon"):
        columns[name] = meta[name].to_numpy(dtype="float32")[station_idx]

    for name, wide in _wide_history_features(hourly, adjacency, horizon):
        columns[name] = wide.to_numpy(dtype="float32").ravel()[flat_keep]

    frame = pd.DataFrame(columns)
    logger.info("Built %d feature rows (%d features)", len(frame), len(FEATURES))
    return frame[["time", "stationcode", *FEATURES, TARGET, "bikes"]]
