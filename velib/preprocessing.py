"""Turn raw snapshot records into a regular hourly grid.

Raw snapshots arrive every ~30 minutes, with gaps, duplicates and stations that
stop reporting. Everything downstream assumes one row per UTC hour, so this
module is the only place that deals with that mess:

1. `clean_records`: parse, correct and deduplicate records.
2. `to_hourly`: average each station's observations within each UTC hour and
   lay them out on a continuous hourly index (missing hours stay NaN).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from velib.config import (
    MAX_REPORT_AGE_HOURS,
    MIN_DAILY_STATION_SHARE,
    STATION_COORD_UPDATES,
)

logger = logging.getLogger(__name__)


def _coord(geo: object, key: str) -> float:
    if isinstance(geo, dict):
        value = geo.get(key)
        return float(value) if value is not None else np.nan
    return np.nan


def clean_records(raw: pd.DataFrame) -> pd.DataFrame:
    """Parse and clean raw records.

    - `duedate` is the station's *last report* time, not the fetch time. A
      station that stops reporting keeps returning the same `duedate`, so
      duplicates on (stationcode, duedate) are dropped: repeating a stale value
      would teach the model that a dead station is perfectly stable.
    - Reports more than `MAX_REPORT_AGE_HOURS` older than the fetch that
      returned them are dropped (dead stations), when `fetched_at` is known.
    - Observations from stations that are not installed are dropped; their
      bike counts are meaningless.
    - Any sparse leading period is trimmed (see `trim_sparse_start`).
    """
    df = pd.DataFrame(
        {
            "stationcode": raw["stationcode"],
            "time": pd.to_datetime(raw["duedate"], utc=True, errors="coerce", format="ISO8601"),
            "bikes": pd.to_numeric(raw["numbikesavailable"], errors="coerce"),
            "capacity": pd.to_numeric(raw["capacity"], errors="coerce"),
            "name": raw["name"],
            "is_installed": raw["is_installed"],
            "is_renting": raw["is_renting"],
            "lat": raw["coordonnees_geo"].map(lambda g: _coord(g, "lat")),
            "lon": raw["coordonnees_geo"].map(lambda g: _coord(g, "lon")),
            "fetched_at": pd.to_datetime(
                raw.get("fetched_at"), utc=True, errors="coerce", format="ISO8601"
            ),
        }
    )
    initial = len(df)

    report_age = df["fetched_at"] - df["time"]
    stale = report_age > pd.Timedelta(hours=MAX_REPORT_AGE_HOURS)
    if stale.any():
        logger.info(
            "Dropping %d reports from %d stations that stopped reporting",
            int(stale.sum()),
            df.loc[stale, "stationcode"].nunique(),
        )
    df = df[~stale].drop(columns="fetched_at")

    df = df.dropna(subset=["stationcode", "time", "bikes", "capacity"])
    df["stationcode"] = df["stationcode"].astype(str)
    df = df[df["bikes"] >= 0]
    df = df[df["is_installed"].fillna("OUI") == "OUI"]

    for code, coords in STATION_COORD_UPDATES.items():
        mask = df["stationcode"] == code
        df.loc[mask, "lat"] = coords["lat"]
        df.loc[mask, "lon"] = coords["lon"]

    df = df.sort_values(["stationcode", "time"])
    df = df.drop_duplicates(subset=["stationcode", "time"], keep="last")
    df = trim_sparse_start(df)

    logger.info(
        "Cleaned records: kept %d of %d (%d stations)",
        len(df),
        initial,
        df["stationcode"].nunique(),
    )
    return df.reset_index(drop=True)


def trim_sparse_start(df: pd.DataFrame, min_share: float = MIN_DAILY_STATION_SHARE) -> pd.DataFrame:
    """Drop observations before the first day on which the network is reasonably observed."""
    if df.empty:
        return df
    stations_per_day = df.groupby(df["time"].dt.floor("D"))["stationcode"].nunique()
    dense = stations_per_day[stations_per_day >= min_share * stations_per_day.max()]
    first_day = dense.index.min()
    early = df["time"] < first_day
    if early.any():
        logger.info(
            "Dropping %d observations before %s (too few stations reporting)",
            int(early.sum()),
            first_day.date(),
        )
    return df[~early]


@dataclass
class HourlyData:
    """Station observations on a continuous UTC hourly grid.

    `bikes` and `capacity` share the same index (hourly, UTC) and columns
    (station codes, sorted). `stations` is indexed by station code in the same
    order and holds the latest metadata for each station.
    """

    bikes: pd.DataFrame
    capacity: pd.DataFrame
    stations: pd.DataFrame

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.bikes.index

    @property
    def last_observed(self) -> pd.Timestamp:
        return self.bikes.dropna(how="all").index.max()

    def extend(self, hours: int) -> HourlyData:
        """Append `hours` empty future rows (capacity carries its last known value)."""
        future = pd.date_range(
            self.index[-1] + pd.Timedelta(hours=1), periods=hours, freq="h", tz="UTC"
        )
        index = self.index.append(future).rename(self.index.name)
        return HourlyData(
            bikes=self.bikes.reindex(index),
            capacity=self.capacity.reindex(index).ffill(),
            stations=self.stations,
        )

    def truncate(self, end: pd.Timestamp) -> HourlyData:
        """Keep only rows strictly before `end` (used to simulate a past forecast origin)."""
        keep = self.index < end
        return HourlyData(
            bikes=self.bikes.loc[keep], capacity=self.capacity.loc[keep], stations=self.stations
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(
            {"bikes": self.bikes, "capacity": self.capacity, "stations": self.stations}, path
        )
        logger.info("Saved hourly grid %s to %s", self.bikes.shape, path)

    @classmethod
    def load(cls, path: Path) -> HourlyData:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `python -m velib prepare` first.")
        data = pd.read_pickle(path)  # noqa: S301 (file produced by this pipeline)
        return cls(bikes=data["bikes"], capacity=data["capacity"], stations=data["stations"])


def to_hourly(clean: pd.DataFrame) -> HourlyData:
    """Aggregate clean records onto a continuous UTC hourly grid."""
    if clean.empty:
        raise ValueError("No usable records after cleaning")

    df = clean.assign(hour=clean["time"].dt.floor("h"))
    grouped = df.groupby(["hour", "stationcode"], sort=True)
    bikes = grouped["bikes"].mean().unstack("stationcode")
    capacity = grouped["capacity"].last().unstack("stationcode")

    index = pd.date_range(bikes.index.min(), bikes.index.max(), freq="h", tz="UTC")
    columns = bikes.columns.sort_values()
    bikes = bikes.reindex(index=index, columns=columns).astype("float64")
    # Forward fill only: back-filling would copy future capacity into the past.
    capacity = capacity.reindex(index=index, columns=columns).ffill().astype("float64")
    bikes.index.name = capacity.index.name = "time"

    latest = df.sort_values("time").groupby("stationcode").last()
    stations = pd.DataFrame(
        {
            "name": latest["name"],
            "lat": latest["lat"],
            "lon": latest["lon"],
            "capacity": latest["capacity"],
            "is_renting": latest["is_renting"].fillna("OUI") == "OUI",
            "last_seen": latest["time"],
            "observed_hours": bikes.notna().sum(),
        }
    ).reindex(columns)
    stations.index.name = "stationcode"

    logger.info(
        "Hourly grid: %d hours x %d stations (%s to %s), %.1f%% observed",
        len(index),
        len(columns),
        index[0],
        index[-1],
        100 * bikes.notna().to_numpy().mean(),
    )
    return HourlyData(bikes=bikes, capacity=capacity, stations=stations)
