import json

import numpy as np
import pandas as pd
import pytest

from velib.features import FEATURES, build_features
from velib.model import evaluate, fit_model, load_bundle, save_bundle, time_split, train
from velib.neighbours import neighbour_adjacency


@pytest.fixture(scope="module")
def trained(synthetic_hourly):
    return train(synthetic_hourly, test_days=3)


@pytest.fixture(scope="module")
def bundle(trained):
    return trained[0]


@pytest.fixture(scope="module")
def diagnostics(trained):
    return trained[1]


def test_time_split_puts_the_last_days_in_test(synthetic_hourly):
    frame = build_features(synthetic_hourly, neighbour_adjacency(synthetic_hourly.stations))
    train_rows, test_rows = time_split(frame, test_days=3)
    assert train_rows["time"].max() < test_rows["time"].min()
    assert len(train_rows) + len(test_rows) == len(frame)
    test_days = test_rows["time"].dt.floor("D").unique()
    assert len(test_days) == 3
    assert test_days.max() == frame["time"].max().floor("D")


def test_bundle_contents(bundle, synthetic_hourly):
    assert bundle["features"] == FEATURES
    assert bundle["horizon_hours"] == 168
    assert bundle["refit_on_all_data"] is True
    assert pd.Timestamp(bundle["trained_until"]) == synthetic_hourly.index[-1]
    metrics = bundle["metrics"]
    assert set(metrics["scores"]) == {"model", "seasonal_naive", "weekly_profile"}
    assert metrics["rows"] > 0
    assert pd.Timestamp(metrics["test_start"]) > pd.Timestamp(bundle["trained_from"])


def test_model_beats_seasonal_naive_on_synthetic_data(bundle):
    # The synthetic data has a stable weekly pattern plus noise, so one noisy
    # week (seasonal naive) must be clearly worse than the learned model.
    scores = bundle["metrics"]["scores"]
    assert scores["model"]["mae"] < scores["seasonal_naive"]["mae"]


def test_no_refit_keeps_the_holdout_model(synthetic_hourly):
    bundle, diagnostics = train(synthetic_hourly, test_days=3, refit=False, with_diagnostics=False)
    assert diagnostics is None
    assert bundle["refit_on_all_data"] is False
    assert pd.Timestamp(bundle["trained_until"]) < pd.Timestamp(bundle["metrics"]["test_start"])


def test_evaluate_scores_everything_on_the_same_rows():
    class Constant:
        def predict(self, X):
            return np.full(len(X), 0.5)

    test = pd.DataFrame(
        dict.fromkeys(FEATURES, 0.0)
        | {"capacity": [10.0, 10.0, 10.0], "bikes": [5.0, 7.0, 3.0]}
        | {"fill_lag_1w": [0.5, 0.5, np.nan], "fill_profile": [0.6, 0.6, 0.6]}
    )
    metrics = evaluate(Constant(), test)
    assert metrics["rows"] == 2 and metrics["rows_excluded"] == 1
    assert metrics["scores"]["model"]["mae"] == pytest.approx(1.0)
    assert metrics["scores"]["seasonal_naive"]["mae"] == pytest.approx(1.0)
    assert metrics["scores"]["weekly_profile"]["mae"] == pytest.approx(1.0)
    assert metrics["skill_vs_best_baseline"] == pytest.approx(0.0)


def test_too_little_history_is_a_clear_error(synthetic_hourly):
    short = synthetic_hourly.truncate(synthetic_hourly.index[0] + pd.Timedelta(days=9))
    with pytest.raises(ValueError, match="Not enough history"):
        train(short, test_days=3)


def test_bundle_round_trip_and_feature_check(tmp_path, bundle):
    path = tmp_path / "model.joblib"
    save_bundle(bundle, path)
    assert load_bundle(path)["features"] == FEATURES

    stale = dict(bundle, features=FEATURES[:-1])
    save_bundle(stale, path)
    with pytest.raises(ValueError, match="retrain"):
        load_bundle(path)


def test_diagnostics_document(diagnostics, bundle, synthetic_hourly):
    json.dumps(diagnostics)  # NaN must already be None: browsers reject NaN in JSON

    assert diagnostics["metrics"] == bundle["metrics"]
    assert set(diagnostics["by_hour"]) == {"model", "seasonal_naive", "weekly_profile"}
    assert len(diagnostics["by_hour"]["model"]) == 24
    assert len(diagnostics["by_day_of_week"]["model"]) == 7

    stations = {s["code"]: s for s in diagnostics["stations"]}
    assert set(stations) == set(synthetic_hourly.bikes.columns)
    scored = [s for s in stations.values() if s["mae"]]
    assert scored and all(0 <= s["coverage"] <= 1 for s in stations.values())

    hours = diagnostics["series_hours"]
    for code, series in diagnostics["series"].items():
        assert code in stations
        assert set(series) == {"actual", "model", "seasonal_naive", "weekly_profile"}
        assert all(len(values) == hours for values in series.values())

    ranked = [item["feature"] for item in diagnostics["importance"]]
    assert sorted(ranked) == sorted(FEATURES)


def test_diagnostics_station_errors_match_overall_scale(diagnostics):
    # Per-station MAEs average (weighted by rows) to the overall MAE.
    scored = [s for s in diagnostics["stations"] if s["mae"]]
    rows = sum(s["test_rows"] for s in scored)
    weighted = sum(s["mae"]["model"] * s["test_rows"] for s in scored) / rows
    assert weighted == pytest.approx(diagnostics["metrics"]["scores"]["model"]["mae"], rel=1e-2)


def test_features_without_any_data_do_not_break_training(synthetic_hourly):
    # With < 4 weeks of history `fill_lag_4w` is entirely NaN; scikit-learn >= 1.9
    # raises on all-NaN columns, so fit_model must cope (and prediction must too).
    frame = build_features(synthetic_hourly, neighbour_adjacency(synthetic_hourly.stations))
    frame = frame.assign(fill_lag_4w=np.nan, fill_lag_3w=np.nan)
    model = fit_model(frame)
    predictions = model.predict(frame[FEATURES])
    assert np.isfinite(predictions).all()
