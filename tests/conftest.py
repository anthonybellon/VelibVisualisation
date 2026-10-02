"""Shared fixtures. Everything is built from synthetic data: no real data is needed."""

from __future__ import annotations

import pandas as pd
import pytest

from velib.io import records_to_frame
from velib.preprocessing import HourlyData, clean_records, to_hourly
from velib.synthetic import generate_records


def make_record(code="1001", duedate="2024-03-04T08:10:00+00:00", bikes=5, capacity=20, **extra):
    record = {
        "stationcode": code,
        "name": f"Station {code}",
        "is_installed": "OUI",
        "is_renting": "OUI",
        "is_returning": "OUI",
        "capacity": capacity,
        "numbikesavailable": bikes,
        "numdocksavailable": None if capacity is None else capacity - bikes,
        "mechanical": bikes,
        "ebike": 0,
        "duedate": duedate,
        "coordonnees_geo": {"lat": 48.86, "lon": 2.35},
    }
    record.update(extra)
    return record


def make_hourly(bikes: pd.DataFrame, capacity: float = 20.0, coords=None) -> HourlyData:
    """Build HourlyData directly from a (UTC hourly index x station) bikes frame."""
    stations = list(bikes.columns)
    coords = coords or {s: (48.86 + 0.002 * i, 2.35 + 0.003 * i) for i, s in enumerate(stations)}
    meta = pd.DataFrame(
        {
            "name": [f"Station {s}" for s in stations],
            "lat": [coords[s][0] for s in stations],
            "lon": [coords[s][1] for s in stations],
            "capacity": capacity,
            "is_renting": True,
            "last_seen": bikes.index[-1],
            "observed_hours": bikes.notna().sum().to_numpy(),
        },
        index=pd.Index(stations, name="stationcode"),
    )
    cap = pd.DataFrame(capacity, index=bikes.index, columns=bikes.columns)
    return HourlyData(bikes=bikes.astype(float), capacity=cap, stations=meta)


@pytest.fixture
def record():
    return make_record


@pytest.fixture(scope="session")
def synthetic_records():
    return generate_records(n_stations=12, days=24, start="2024-03-11", seed=1)


@pytest.fixture(scope="session")
def synthetic_hourly(synthetic_records) -> HourlyData:
    return to_hourly(clean_records(records_to_frame(synthetic_records)))
