from datetime import date

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_hourly
from velib.config import HORIZON_HOURS
from velib.features import (
    FEATURES,
    calendar_features,
    french_holidays,
    slot_profile,
)
from velib.features import build_features as _build_features
from velib.neighbours import neighbour_adjacency

H = HORIZON_HOURS
OBSERVATION_FEATURES = [
    f
    for f in FEATURES
    if f not in {"hour", "day_of_week", "is_weekend", "is_holiday", "month", "lat", "lon"}
]


def build_features(hourly, **kwargs):
    return _build_features(hourly, neighbour_adjacency(hourly.stations), **kwargs)


def _feature_frame(hourly, **kwargs):
    frame = build_features(hourly, require_target=False, **kwargs)
    return frame.set_index(["time", "stationcode"]).sort_index()


# ------------------------------------------------------------------------------
# Leakage: the central guarantee of the pipeline
# ------------------------------------------------------------------------------


def test_no_feature_uses_data_less_than_one_horizon_old(synthetic_hourly):
    """Changing observations at and after t0 must not change any feature before t0 + H."""
    t0 = synthetic_hourly.index[len(synthetic_hourly.index) // 2]
    perturbed_bikes = synthetic_hourly.bikes.copy()
    perturbed_bikes.loc[perturbed_bikes.index >= t0] = 0.0
    perturbed_capacity = synthetic_hourly.capacity.copy()
    perturbed_capacity.loc[perturbed_capacity.index >= t0] = 99.0
    perturbed = type(synthetic_hourly)(
        bikes=perturbed_bikes, capacity=perturbed_capacity, stations=synthetic_hourly.stations
    )

    original = _feature_frame(synthetic_hourly)
    changed = _feature_frame(perturbed)
    times = original.index.get_level_values("time")

    safe = times < t0 + pd.Timedelta(hours=H)
    pd.testing.assert_frame_equal(original.loc[safe, FEATURES], changed.loc[safe, FEATURES])

    # Sanity check that the perturbation does reach features one horizon later.
    later = times >= t0 + pd.Timedelta(hours=H)
    assert not original.loc[later, OBSERVATION_FEATURES].equals(
        changed.loc[later, OBSERVATION_FEATURES]
    )


def test_forecast_features_match_features_computed_with_hindsight(synthetic_hourly):
    """Training/forecast parity: features for a future week, computed from history
    only (grid extended with empty rows), equal the features computed later once
    that week has been observed."""
    origin = synthetic_hourly.index[-H]  # last week of data plays the "future"
    history = synthetic_hourly.truncate(origin).extend(H)

    forecast_rows = _feature_frame(history, start=origin)
    hindsight_rows = _feature_frame(synthetic_hourly, start=origin)

    assert len(forecast_rows) == H * synthetic_hourly.bikes.shape[1]
    pd.testing.assert_frame_equal(forecast_rows[FEATURES], hindsight_rows[FEATURES])


# ------------------------------------------------------------------------------
# Individual features
# ------------------------------------------------------------------------------


def _weekly_series(weeks, values_per_week, station="A"):
    index = pd.date_range("2024-01-01", periods=weeks * H, freq="h", tz="UTC")
    data = np.repeat(values_per_week, H).astype(float)
    return make_hourly(pd.DataFrame({station: data}, index=index), capacity=10.0)


def test_slot_profile_is_expanding_mean_of_previous_weeks():
    hourly = _weekly_series(4, [1, 3, 5, 7])
    fill = hourly.bikes / hourly.capacity
    profile, weeks = slot_profile(fill)
    at = [0, H, 2 * H, 3 * H]
    assert np.isnan(profile.iloc[at[0], 0])
    assert profile.iloc[at[1:], 0].tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert weeks.iloc[at, 0].tolist() == [0, 1, 2, 3]


def test_lag_and_profile_values():
    hourly = _weekly_series(5, [1, 2, 3, 4, 5])
    frame = build_features(hourly, require_target=True)
    last = frame[frame["time"] == hourly.index[-1]].iloc[0]
    assert last["fill_lag_1w"] == pytest.approx(0.4)
    assert last["fill_lag_4w"] == pytest.approx(0.1)
    assert last["fill_same_hour_mean"] == pytest.approx(0.25)
    assert last["fill_profile"] == pytest.approx(0.25)
    assert last["fill_week_mean_lag"] == pytest.approx(0.4)
    assert last["fill_ratio"] == pytest.approx(0.5)


def test_training_rows_need_a_target_and_one_week_of_history():
    hourly = _weekly_series(2, [1, 2])
    hourly.bikes.iloc[H + 5] = np.nan
    frame = build_features(hourly, require_target=True)
    assert frame["time"].min() == hourly.index[H]
    assert len(frame) == H - 1


def test_empty_and_full_profiles():
    index = pd.date_range("2024-01-01", periods=3 * H, freq="h", tz="UTC")
    bikes = np.tile([0.0, 10.0, 5.0], H)  # empty, full, half; repeats every 3 hours
    hourly = make_hourly(pd.DataFrame({"A": bikes}, index=index), capacity=10.0)
    frame = build_features(hourly, require_target=True).set_index("time")
    row = frame.loc[index[2 * H]]  # same slot as index 0 (empty) in earlier weeks
    assert (row["empty_profile"], row["full_profile"]) == (1.0, 0.0)


def test_neighbour_features_use_other_stations_only():
    index = pd.date_range("2024-01-01", periods=2 * H, freq="h", tz="UTC")
    bikes = pd.DataFrame({"A": 2.0, "B": 8.0}, index=index)
    hourly = make_hourly(bikes, capacity=10.0)
    frame = build_features(hourly, require_target=True)
    a = frame[frame["stationcode"] == "A"].iloc[0]
    assert a["nbr_fill_lag_1w"] == pytest.approx(0.8)
    assert a["fill_lag_1w"] == pytest.approx(0.2)


# ------------------------------------------------------------------------------
# Calendar
# ------------------------------------------------------------------------------


def test_hour_is_local_time_across_daylight_saving():
    index = pd.DatetimeIndex(["2024-03-29 07:00", "2024-04-02 07:00"], tz="UTC")
    calendar = calendar_features(index)
    # 07:00 UTC is 08:00 in winter (CET) and 09:00 in summer (CEST).
    assert calendar["hour"].tolist() == [8, 9]


def test_day_of_week_follows_local_midnight():
    # 23:00 UTC on Sunday is already Monday in Paris.
    index = pd.DatetimeIndex(["2024-03-10 23:00"], tz="UTC")
    calendar = calendar_features(index)
    assert calendar["day_of_week"].tolist() == [0]
    assert calendar["is_weekend"].tolist() == [0]


def test_french_holidays():
    holidays = french_holidays(2024)
    assert date(2024, 4, 1) in holidays  # Easter Monday
    assert date(2024, 5, 9) in holidays  # Ascension
    assert date(2024, 5, 20) in holidays  # Whit Monday
    assert date(2024, 7, 14) in holidays
    assert date(2024, 4, 2) not in holidays
    assert len(holidays) == 11


def test_slot_profile_follows_local_time_across_daylight_saving():
    # Each value is the local hour, so a correct profile equals the local hour.
    index = pd.date_range("2024-03-04", "2024-04-21 23:00", freq="h", tz="UTC")
    local_hour = index.tz_convert("Europe/Paris").hour.to_numpy().astype(float)
    hourly = make_hourly(pd.DataFrame({"A": local_hour}, index=index), capacity=100.0)
    profile, _ = slot_profile(hourly.bikes / hourly.capacity)

    after = index >= pd.Timestamp("2024-04-01", tz="UTC")  # summer time, 2+ weeks of history
    expected = local_hour[after] / 100
    assert profile.loc[after, "A"].to_numpy() == pytest.approx(expected)


def test_slot_profile_never_uses_data_less_than_a_horizon_old():
    # Spring forward: the same local slot recurs after only 167 UTC hours.
    index = pd.date_range("2024-03-18", "2024-04-14", freq="h", tz="UTC")
    values = pd.DataFrame({"A": np.arange(len(index), dtype=float)}, index=index)
    profile, counts = slot_profile(values)
    for t in range(len(index)):
        if counts.iloc[t, 0] == 0:
            continue
        # The newest value is t - age; with values = row number, the mean of the
        # used rows is at most (t - H) if every used row is >= H old.
        assert profile.iloc[t, 0] <= t - H
