"""Training, evaluation and persistence of the global forecasting model.

One HistGradientBoosting model is trained for all stations (instead of one
RandomForest per station). Station identity enters through station-level
features (coordinates, capacity, history profiles), which lets stations with
little data borrow strength from similar ones.

Evaluation is a time-based holdout: the model is trained on everything before
the last `test_days` and scored on those days, next to two baselines it has to
beat to be worth anything:

- `seasonal_naive`: same hour one week earlier.
- `weekly_profile`: the station's historical mean for that weekly slot.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from velib.config import DEFAULT_TEST_DAYS, HORIZON_HOURS, MODEL_PARAMS
from velib.diagnostics import build_diagnostics
from velib.features import FEATURES, TARGET, build_features
from velib.neighbours import neighbour_adjacency
from velib.preprocessing import HourlyData

logger = logging.getLogger(__name__)

BUNDLE_VERSION = 2
BASELINES = {"seasonal_naive": "fill_lag_1w", "weekly_profile": "fill_profile"}


def make_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(**MODEL_PARAMS)


def fit_model(rows: pd.DataFrame) -> HistGradientBoostingRegressor:
    """Fit a fresh model on `rows`.

    A feature with no observed value at all (e.g. `fill_lag_4w` when there is
    less than four weeks of history) is passed as a constant. It carries no
    information either way, so the model never splits on it and predictions
    are unchanged; but scikit-learn >= 1.9 raises on all-NaN columns.
    """
    X = rows[FEATURES]
    empty = [c for c in FEATURES if X[c].isna().all()]
    if empty:
        logger.info("No data yet for features %s; the model ignores them", ", ".join(empty))
        X = X.assign(**dict.fromkeys(empty, 0.0))
    return make_model().fit(X, rows[TARGET])


def time_split(frame: pd.DataFrame, test_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time: the last `test_days` x 24 hours (up to the latest row) form the test set."""
    cutoff = frame["time"].max() - pd.Timedelta(days=test_days) + pd.Timedelta(hours=1)
    train = frame[frame["time"] < cutoff]
    test = frame[frame["time"] >= cutoff]
    return train, test


def predict_bikes(model: Any, frame: pd.DataFrame) -> np.ndarray:
    """Predicted bike counts: predicted fill ratio x capacity known at the origin."""
    ratio = np.clip(model.predict(frame[FEATURES]), 0.0, None)
    return ratio * frame["capacity"].to_numpy()


def _errors(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    diff = predicted - actual
    return {
        "mae": round(float(np.mean(np.abs(diff))), 4),
        "rmse": round(float(np.sqrt(np.mean(diff**2))), 4),
    }


def evaluate(model: Any, test: pd.DataFrame) -> dict[str, Any]:
    """Score the model and the baselines on the same rows, in bikes."""
    rows = test.dropna(subset=["capacity", *BASELINES.values()])
    if rows.empty:
        raise ValueError("No test rows have both baselines available; collect more history.")
    actual = rows["bikes"].to_numpy()
    scores = {"model": _errors(actual, predict_bikes(model, rows))}
    for name, column in BASELINES.items():
        scores[name] = _errors(actual, rows[column].to_numpy() * rows["capacity"].to_numpy())

    best_baseline = min(scores[name]["mae"] for name in BASELINES)
    skill = 1 - scores["model"]["mae"] / best_baseline if best_baseline > 0 else 0.0
    return {
        "rows": len(rows),
        "rows_excluded": len(test) - len(rows),
        "mean_bikes": round(float(actual.mean()), 3),
        "scores": scores,
        # > 0 means the model beats the best baseline (0.1 = 10% lower MAE).
        "skill_vs_best_baseline": round(float(skill), 4),
    }


def train(
    hourly: HourlyData,
    test_days: int = DEFAULT_TEST_DAYS,
    refit: bool = True,
    with_diagnostics: bool = True,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Train, evaluate on a time holdout, optionally refit on all data.

    Returns the model bundle and, if requested, the holdout diagnostics for the
    debugging dashboard (computed with the holdout model, before any refit).
    """
    adjacency = neighbour_adjacency(hourly.stations)
    frame = build_features(hourly, adjacency, require_target=True)
    if frame.empty:
        raise ValueError(
            f"No trainable rows: need more than {HORIZON_HOURS // 24} days of history "
            "for at least one station."
        )

    train_rows, test_rows = time_split(frame, test_days)
    if train_rows.empty or test_rows.empty:
        span = (frame["time"].max() - frame["time"].min()).days + 1
        raise ValueError(
            f"Not enough history for a {test_days}-day holdout: only {span} days of "
            f"trainable rows (each needs {HORIZON_HOURS // 24} days of history first). "
            "Collect more data or lower --test-days."
        )

    logger.info(
        "Training on %d rows (%s to %s), testing on %d rows (%s to %s)",
        len(train_rows),
        train_rows["time"].min(),
        train_rows["time"].max(),
        len(test_rows),
        test_rows["time"].min(),
        test_rows["time"].max(),
    )
    model = fit_model(train_rows)
    metrics = evaluate(model, test_rows)
    metrics["test_start"] = str(test_rows["time"].min())
    metrics["test_end"] = str(test_rows["time"].max())
    _log_metrics(metrics)
    diagnostics = build_diagnostics(model, test_rows, hourly, metrics) if with_diagnostics else None

    if refit:
        logger.info("Refitting on all %d rows", len(frame))
        model = fit_model(frame)

    bundle = {
        "version": BUNDLE_VERSION,
        "model": model,
        "features": list(FEATURES),
        "target": TARGET,
        "horizon_hours": HORIZON_HOURS,
        "params": dict(MODEL_PARAMS),
        "trained_from": str(frame["time"].min()),
        "trained_until": str(frame["time"].max() if refit else train_rows["time"].max()),
        "refit_on_all_data": refit,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": metrics,
    }
    return bundle, diagnostics


def _log_metrics(metrics: dict[str, Any]) -> None:
    for name, score in metrics["scores"].items():
        logger.info("  %-15s MAE %.3f  RMSE %.3f bikes", name, score["mae"], score["rmse"])
    logger.info("  skill vs best baseline: %+.1f%%", 100 * metrics["skill_vs_best_baseline"])
    if metrics["skill_vs_best_baseline"] <= 0:
        logger.warning(
            "The model does not beat the best baseline on the holdout; "
            "treat its forecast with caution (see ARCHITECTURE.md, Evaluation)."
        )


def save_bundle(bundle: dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    logger.info("Saved model bundle to %s", path)


def load_bundle(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m velib train` first.")
    bundle = joblib.load(path)
    if bundle.get("version") != BUNDLE_VERSION or bundle.get("features") != FEATURES:
        raise ValueError(
            f"{path} was trained with a different feature set; retrain with `python -m velib train`."
        )
    return bundle
