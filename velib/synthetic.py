"""Synthetic snapshot data with realistic structure, for demos and tests.

Real station locations and capacities come from the committed sample snapshot.
Availability follows commuter patterns (residential stations empty in the
morning, business stations fill up), weekend/holiday leisure patterns, daily
"weather" shocks, neighbourhood-level shocks and per-station noise. The data
also contains the defects the pipeline must handle: missing fetches, stations
that are temporarily not installed, and a station that stops reporting (its
`duedate` freezes, producing duplicates).

None of this says anything about real model accuracy; it only exercises the code.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from velib.config import SAMPLE_SNAPSHOT_PATH, TIMEZONE
from velib.features import french_holidays
from velib.io import append_jsonl

PARIS_CENTRE = (48.8566, 2.3522)


def _stations(n: int, rng: np.random.Generator, snapshot_path: Path | None) -> pd.DataFrame:
    if snapshot_path is not None and Path(snapshot_path).exists():
        with open(snapshot_path, encoding="utf-8") as f:
            records = json.load(f)
        df = pd.DataFrame(
            {
                "stationcode": [str(r["stationcode"]) for r in records],
                "name": [r["name"] for r in records],
                "capacity": [r["capacity"] for r in records],
                "lat": [(r.get("coordonnees_geo") or {}).get("lat") for r in records],
                "lon": [(r.get("coordonnees_geo") or {}).get("lon") for r in records],
            }
        ).dropna()
        df = df[df["capacity"] > 0].drop_duplicates("stationcode")
    else:
        count = max(n, 1)
        df = pd.DataFrame(
            {
                "stationcode": [str(10000 + i) for i in range(count)],
                "name": [f"Synthetic station {i}" for i in range(count)],
                "capacity": rng.integers(15, 60, count),
                "lat": PARIS_CENTRE[0] + rng.normal(0, 0.02, count),
                "lon": PARIS_CENTRE[1] + rng.normal(0, 0.03, count),
            }
        )
    # The stations closest to the centre form a dense network with real neighbours.
    km_lat, km_lon = 111.2, 111.2 * np.cos(np.radians(PARIS_CENTRE[0]))
    df["dist_km"] = np.hypot(
        (df["lat"] - PARIS_CENTRE[0]) * km_lat, (df["lon"] - PARIS_CENTRE[1]) * km_lon
    )
    return df.nsmallest(n, "dist_km").reset_index(drop=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def generate_records(
    n_stations: int = 150,
    days: int = 35,
    start: str = "2024-03-04",
    freq_minutes: int = 30,
    seed: int = 0,
    snapshot_path: Path | None = SAMPLE_SNAPSHOT_PATH,
) -> list[dict[str, Any]]:
    """Generate snapshot records (Open Data schema plus `fetched_at`).

    The default window (4 Mar to 8 Apr 2024) contains a daylight-saving change
    and the Easter Monday holiday.
    """
    rng = np.random.default_rng(seed)
    stations = _stations(n_stations, rng, snapshot_path)
    n = len(stations)
    times = pd.date_range(
        pd.Timestamp(start, tz="UTC"),
        periods=days * 24 * 60 // freq_minutes,
        freq=f"{freq_minutes}min",
    )
    local = times.tz_convert(TIMEZONE)
    hour = (local.hour + local.minute / 60).to_numpy()[:, None]
    day_index = ((times - times[0]) // pd.Timedelta(days=1)).to_numpy()
    leisure_day = np.array([d.weekday() >= 5 or d in french_holidays(d.year) for d in local.date])[
        :, None
    ]

    # Station character: business (1) near the centre, residential (0) further out.
    business = np.clip(1 - stations["dist_km"].to_numpy() / 5 + rng.normal(0, 0.25, n), 0, 1)
    base = rng.uniform(0.35, 0.6, n)

    residential = 0.2 - 0.35 * _sigmoid((hour - 7.5) / 0.7) + 0.35 * _sigmoid((hour - 18.5) / 1.2)
    weekday = (1 - business) * residential - business * residential
    leisure = 0.3 * weekday - 0.15 * np.exp(-((hour - 15) ** 2) / 18)
    # Real stations swing hard between empty and full; amplify the shapes accordingly.
    pattern = 2.0 * np.where(leisure_day, leisure, weekday)

    weather = np.clip(rng.normal(1, 0.25, days + 1), 0.4, 1.4)[day_index][:, None]
    zone = (
        (stations["lat"] * 80).round().astype(int).astype(str)
        + "_"
        + (stations["lon"] * 55).round().astype(int).astype(str)
    )
    zone_ids = pd.factorize(zone)[0]
    zone_shock = rng.normal(0, 0.06, (days + 1, zone_ids.max() + 1))[day_index][:, zone_ids]

    noise = np.zeros((len(times), n))
    innovations = rng.normal(0, 0.03, (len(times), n))
    for t in range(1, len(times)):
        noise[t] = 0.95 * noise[t - 1] + innovations[t]

    ratio = np.clip(base + weather * pattern + zone_shock + noise, 0, 1)
    capacity = stations["capacity"].to_numpy()
    bikes = np.rint(ratio * capacity).astype(int)

    installed = np.ones_like(bikes, dtype=bool)
    for s in rng.choice(n, size=max(1, n // 20), replace=False):
        begin = rng.integers(0, len(times) - 96)
        installed[begin : begin + rng.integers(48, 96), s] = False

    present = rng.random(bikes.shape) > 0.02
    outage = rng.integers(0, len(times) - 12)
    present[outage : outage + 12] = False

    # One station stops reporting halfway: later snapshots repeat its last report.
    stale_station = n - 1 if n > 1 else None
    stale_from = len(times) // 2

    report_lag = rng.uniform(0, 240, bikes.shape)
    codes = stations["stationcode"].to_numpy()
    names = stations["name"].to_numpy()
    lats = stations["lat"].to_numpy()
    lons = stations["lon"].to_numpy()

    records = []
    for t, fetched_at in enumerate(times):
        if not present[t].any():
            continue
        fetched_iso = fetched_at.isoformat(timespec="seconds")
        for s in np.flatnonzero(present[t]):
            src = stale_from if (s == stale_station and t >= stale_from) else t
            due = times[src] - pd.Timedelta(seconds=float(report_lag[src, s]))
            b = int(bikes[src, s])
            cap = int(capacity[s])
            ebike = int(round(b * 0.4))
            records.append(
                {
                    "stationcode": codes[s],
                    "name": names[s],
                    "is_installed": "OUI" if installed[src, s] else "NON",
                    "is_renting": "OUI",
                    "is_returning": "OUI",
                    "capacity": cap,
                    "numbikesavailable": b,
                    "numdocksavailable": max(cap - b, 0),
                    "mechanical": b - ebike,
                    "ebike": ebike,
                    "duedate": due.isoformat(timespec="seconds"),
                    "coordonnees_geo": {"lat": float(lats[s]), "lon": float(lons[s])},
                    "fetched_at": fetched_iso,
                }
            )
    return records


def write_records(records: list[dict[str, Any]], raw_dir: Path) -> list[Path]:
    """Write records to one JSONL file per UTC fetch day (the `velib fetch` layout)."""
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_day[record["fetched_at"][:10]].append(record)
    paths = []
    for day, day_records in sorted(by_day.items()):
        path = Path(raw_dir) / f"{day}.jsonl"
        path.unlink(missing_ok=True)
        append_jsonl(day_records, path)
        paths.append(path)
    return paths
