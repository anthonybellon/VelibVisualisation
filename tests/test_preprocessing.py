import numpy as np
import pandas as pd

from velib.io import records_to_frame
from velib.preprocessing import HourlyData, clean_records, to_hourly


def _clean(records):
    return clean_records(records_to_frame(records))


def test_stale_reports_are_deduplicated(record):
    # A station that stopped reporting returns the same duedate on every fetch.
    stale = [record("1", duedate="2024-03-04T08:10:00+00:00", bikes=3) for _ in range(5)]
    clean = _clean(stale)
    assert len(clean) == 1


def test_not_installed_and_invalid_rows_are_dropped(record):
    clean = _clean(
        [
            record("1", bikes=4),
            record("2", is_installed="NON"),
            record("3", bikes=-1),
            record("4", duedate="not a date"),
            record("5", capacity=None),
        ]
    )
    assert list(clean["stationcode"]) == ["1"]


def test_mixed_utc_offsets_are_parsed_to_utc(record):
    clean = _clean(
        [
            record("1", duedate="2024-03-04T09:10:00+01:00"),
            record("2", duedate="2024-03-04T08:10:00+00:00"),
        ]
    )
    assert clean["time"].nunique() == 1
    assert str(clean["time"].dt.tz) == "UTC"


def test_known_coordinate_corrections_are_applied(record):
    clean = _clean([record("22504")])
    assert clean.loc[0, ["lat", "lon"]].tolist() == [48.905928, 2.253629]


def test_hourly_grid_averages_within_hour_and_keeps_gaps(record):
    clean = _clean(
        [
            record("1", duedate="2024-03-04T08:05:00+00:00", bikes=4),
            record("1", duedate="2024-03-04T08:35:00+00:00", bikes=6),
            # 09:00 missing
            record("1", duedate="2024-03-04T10:20:00+00:00", bikes=1, capacity=25),
            record("2", duedate="2024-03-04T10:40:00+00:00", bikes=9),
        ]
    )
    hourly = to_hourly(clean)

    assert list(hourly.index.strftime("%H")) == ["08", "09", "10"]
    assert hourly.bikes["1"].tolist()[0] == 5
    assert np.isnan(hourly.bikes.loc["2024-03-04 09:00", "1"])
    # Capacity is carried forward, never backward.
    assert hourly.capacity["1"].tolist() == [20, 20, 25]
    assert hourly.capacity["2"].isna().tolist() == [True, True, False]
    assert hourly.stations.loc["1", "observed_hours"] == 2
    assert hourly.stations.loc["1", "capacity"] == 25


def test_extend_and_truncate_round_trip(synthetic_hourly):
    extended = synthetic_hourly.extend(5)
    assert len(extended.index) == len(synthetic_hourly.index) + 5
    assert extended.bikes.iloc[-5:].isna().all().all()
    assert not extended.capacity.iloc[-5:].isna().all().any()

    back = extended.truncate(synthetic_hourly.index[-1] + pd.Timedelta(hours=1))
    pd.testing.assert_frame_equal(back.bikes, synthetic_hourly.bikes)


def test_save_and_load(tmp_path, synthetic_hourly):
    path = tmp_path / "hourly.pkl"
    synthetic_hourly.save(path)
    loaded = HourlyData.load(path)
    pd.testing.assert_frame_equal(loaded.bikes, synthetic_hourly.bikes)
    pd.testing.assert_frame_equal(loaded.stations, synthetic_hourly.stations)


def test_synthetic_data_contains_the_defects_we_clean(synthetic_records):
    raw = records_to_frame(synthetic_records)
    assert (raw["is_installed"] == "NON").any()
    assert raw.duplicated(subset=["stationcode", "duedate"]).any()
    clean = clean_records(raw)
    assert not clean.duplicated(subset=["stationcode", "time"]).any()


def test_reports_much_older_than_the_fetch_are_dropped(record):
    fetched = "2024-03-04T08:30:00+00:00"
    clean = _clean(
        [
            record("1", duedate="2024-03-04T08:10:00+00:00", fetched_at=fetched),
            record("2", duedate="2018-12-21T14:00:00+00:00", fetched_at=fetched),  # dead station
            record("3", duedate="2018-12-21T14:00:00+00:00"),  # no fetched_at: kept here
        ]
    )
    assert "2" not in set(clean["stationcode"])


def test_sparse_leading_period_does_not_stretch_the_grid(record):
    # One ancient report (no fetched_at to identify it) plus a normal day of data.
    records = [record("old", duedate="2018-12-21T14:00:00+00:00")]
    records += [
        record(str(code), duedate=f"2024-03-04T{hour:02d}:10:00+00:00")
        for code in range(20)
        for hour in range(24)
    ]
    hourly = to_hourly(_clean(records))
    assert len(hourly.index) == 24
    assert "old" not in hourly.bikes.columns
