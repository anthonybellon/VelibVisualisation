import json

import pandas as pd
import pytest

from velib.config import HORIZON_HOURS, INACTIVE_STATION_DAYS
from velib.forecast import export_forecast, export_legacy, forecast
from velib.model import train
from velib.preprocessing import HourlyData


@pytest.fixture(scope="module")
def bundle(synthetic_hourly):
    return train(synthetic_hourly, test_days=3, with_diagnostics=False)[0]


def active_codes(hourly):
    """Stations seen within INACTIVE_STATION_DAYS of the end of the grid."""
    cutoff = hourly.index[-1] + pd.Timedelta(hours=1) - pd.Timedelta(days=INACTIVE_STATION_DAYS)
    return set(hourly.stations.index[hourly.stations["last_seen"] >= cutoff])


@pytest.fixture(scope="module")
def predictions(synthetic_hourly, bundle):
    return forecast(synthetic_hourly, bundle)


def test_forecast_covers_the_next_week_for_every_station(synthetic_hourly, predictions):
    start = synthetic_hourly.index[-1] + pd.Timedelta(hours=1)
    assert predictions["time"].min() == start
    assert predictions["time"].nunique() == HORIZON_HOURS
    # The synthetic data contains one station that stops reporting halfway.
    assert set(predictions["stationcode"]) == active_codes(synthetic_hourly)
    assert len(active_codes(synthetic_hourly)) == synthetic_hourly.bikes.shape[1] - 1
    assert predictions["bikes_pred"].notna().all()
    assert (predictions["bikes_pred"] >= 0).all()


def test_export_document(synthetic_hourly, predictions, bundle):
    doc = export_forecast(predictions, synthetic_hourly, bundle, synthetic=True)
    json.dumps(doc)  # must be serialisable as-is

    assert doc["synthetic"] is True
    assert doc["model"]["scores"] == bundle["metrics"]["scores"]
    assert len(doc["day_dates"]) == 7 and all(doc["day_dates"])
    station = doc["stations"][0]
    assert len(station["bikes"]) == 7
    assert all(len(day) == 24 for day in station["bikes"])
    assert {s["code"] for s in doc["stations"]} == active_codes(synthetic_hourly)


def test_daylight_saving_week_has_no_holes(synthetic_hourly, bundle):
    # Forecast the week of 31 March 2024, when 02:00 local time does not exist.
    origin = pd.Timestamp("2024-03-25", tz="UTC")
    history = synthetic_hourly.truncate(origin)
    predictions = forecast(history, bundle)
    doc = export_forecast(predictions, history, bundle)
    sunday = doc["day_names"].index("Sunday")
    for station in doc["stations"]:
        assert all(v >= 0 for v in station["bikes"][sunday])
    assert doc["day_dates"][sunday] == "2024-03-31"


def test_legacy_export_keeps_the_frozen_schema(synthetic_hourly, predictions):
    doc = export_legacy(predictions, synthetic_hourly)
    assert set(doc) == {"normal_capacity", "extra_capacity"}
    normal = doc["normal_capacity"][0]
    assert set(normal) == {
        "stationcode",
        "name",
        "capacity",
        "is_renting",
        "coordonnees_geo",
        "missing_predictions",
        "predictions",
    }
    assert normal["is_renting"] in {"OUI", "NON"}
    assert set(normal["coordonnees_geo"]) == {"lon", "lat"}
    assert list(normal["predictions"]) == [str(d) for d in range(7)]
    assert all(
        len(v) == 24 and all(isinstance(x, int) for x in v) for v in normal["predictions"].values()
    )

    extra = doc["extra_capacity"][0]
    assert set(extra) == set(normal) | {"extra_capacity_predictions"}
    assert extra["stationcode"] == normal["stationcode"]


def test_legacy_percentages_and_doubled_capacity(synthetic_hourly, predictions):
    code = predictions["stationcode"].iloc[0]
    rows = predictions[predictions["stationcode"] == code].copy()
    rows["bikes_pred"] = 10.4
    stations = synthetic_hourly.stations.copy()
    stations.loc[code, "capacity"] = 20
    hourly = HourlyData(synthetic_hourly.bikes, synthetic_hourly.capacity, stations)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("velib.forecast.EXTRA_CAPACITY_STATIONS", frozenset({code}))
        doc = export_legacy(rows, hourly)
    normal, extra = doc["normal_capacity"][0], doc["extra_capacity"][0]
    assert normal["capacity"] == 20 and extra["capacity"] == 40
    assert normal["predictions"]["0"][0] == 50  # round(10.4) = 10 bikes of 20
    assert extra["predictions"]["0"][0] == 25  # 10 of 40
    assert extra["extra_capacity_predictions"]["0"][0] == pytest.approx(10.4)


def test_inactive_stations_are_not_forecast(synthetic_hourly, bundle):
    code = synthetic_hourly.bikes.columns[0]
    stations = synthetic_hourly.stations.copy()
    stations.loc[code, "last_seen"] = synthetic_hourly.index[-1] - pd.Timedelta(days=10)
    hourly = HourlyData(synthetic_hourly.bikes, synthetic_hourly.capacity, stations)
    predictions = forecast(hourly, bundle)
    assert set(predictions["stationcode"]) == active_codes(synthetic_hourly) - {code}


def test_bikes_use_the_latest_capacity_and_survive_zero_capacity(synthetic_hourly, bundle):
    code = synthetic_hourly.bikes.columns[0]
    capacity = synthetic_hourly.capacity.copy()
    capacity[code] = 0.0  # under maintenance for the whole history
    stations = synthetic_hourly.stations.copy()
    stations.loc[code, "capacity"] = 30
    hourly = HourlyData(synthetic_hourly.bikes, capacity, stations)
    rows = forecast(hourly, bundle)
    rows = rows[rows["stationcode"] == code]
    assert rows["bikes_pred"].to_numpy() == pytest.approx(30 * rows["fill_ratio_pred"].to_numpy())


def test_day_dates_across_spring_forward(synthetic_hourly, bundle):
    # Window starts Saturday 30 March 23:00 local; Sunday is the next day (DST day).
    history = synthetic_hourly.truncate(pd.Timestamp("2024-03-30 22:00", tz="UTC"))
    doc = export_forecast(forecast(history, bundle), history, bundle)
    assert doc["day_dates"][6] == "2024-03-31"
    assert doc["day_dates"][5] == "2024-03-30"


def test_legacy_export_time_zone_is_configurable(synthetic_hourly, predictions):
    code = predictions["stationcode"].iloc[0]
    rows = predictions[predictions["stationcode"] == code].copy()
    local = rows["time"].dt.tz_convert("Europe/Paris")
    rows["bikes_pred"] = local.dt.hour.astype(float)  # value = local hour
    hourly = synthetic_hourly
    paris = export_legacy(rows, hourly, tz="Europe/Paris")["extra_capacity"][0]
    utc = export_legacy(rows, hourly, tz="UTC")["extra_capacity"][0]
    assert paris["extra_capacity_predictions"]["2"][8] == pytest.approx(8.0)
    # In UTC keys, hour 8 holds local 09:00 or 10:00 depending on the season.
    assert utc["extra_capacity_predictions"]["2"][8] in (pytest.approx(9.0), pytest.approx(10.0))
