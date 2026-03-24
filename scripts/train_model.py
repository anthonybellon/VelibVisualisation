"""Train bike availability prediction models for each station.

Usage:
    python train_model.py [OPTIONS]

Options:
    --data-dir PATH       Directory with cleaned JSON data
    --output-dir PATH     Directory to save models
    --batch-size INT      Number of stations per batch (default: 100)
    --resume              Resume from last completed batch
    --verbose, -v         Enable verbose logging
    --quiet, -q           Suppress info logging
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    HISTORICAL_DATA_DIR,
    MODEL_SAVE_DIR,
    N_ITER_SEARCH,
    N_SPLITS,
    RANDOM_STATE,
    RF_PARAM_DISTRIBUTIONS,
    TEST_SIZE,
)
from feature_engineering import (
    add_capacity_features,
    add_lag_features,
    add_rolling_features,
    add_temporal_features,
    apply_coordinate_updates,
    calculate_nearby_station_status_adjustable,
    compute_nearby_features,
    extract_coordinates,
    handle_large_values,
    preprocess_datetime,
)
from logging_config import get_logger, init_cli_logging

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train bike availability prediction models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=HISTORICAL_DATA_DIR,
        help="Directory with cleaned JSON data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODEL_SAVE_DIR,
        help="Directory to save models",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Number of stations per batch",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last completed batch",
    )
    parser.add_argument(
        "--station-limit",
        type=int,
        default=None,
        help="Limit number of stations to process (for testing)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress info logging",
    )
    return parser.parse_args()


def load_data(data_dir: Path) -> pd.DataFrame:
    """Load and concatenate JSON files from data directory."""
    logger.info(f"Loading data from {data_dir}")

    all_data = []
    json_files = list(data_dir.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")

    for file_path in tqdm(json_files, desc="Loading JSON files"):
        with open(file_path) as f:
            data = json.load(f)
            all_data.extend(data)

    df = pd.DataFrame(all_data)

    if df.empty:
        raise ValueError("Loaded data is empty")

    logger.info(f"Loaded {len(df)} records from {len(json_files)} files")
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess bike data with feature engineering."""
    logger.info("Preprocessing data...")

    # Datetime preprocessing
    df = preprocess_datetime(df)

    # Coordinate extraction and updates
    df = extract_coordinates(df)
    df = apply_coordinate_updates(df)

    # Drop rows with missing coordinates
    initial_rows = len(df)
    df = df.dropna(subset=["lat", "lon"])
    if len(df) < initial_rows:
        logger.info(f"Dropped {initial_rows - len(df)} rows with missing coordinates")

    # Fill NaN values
    df = df.fillna(0)

    # Feature engineering
    df = add_temporal_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_capacity_features(df)

    # Handle large values
    df, problematic_stations = handle_large_values(df)
    if problematic_stations:
        logger.warning(f"Stations with problematic values: {problematic_stations}")

    logger.info("Preprocessing complete")
    return df


# Feature definitions
BASE_FEATURES = [
    "hour",
    "day_of_week",
    "avg_bikes_hour_day",
    "lag_1_hour",
    "lag_1_day",
    "rolling_mean_7_days",
    "rolling_mean_30_days",
    "normalized_bikes_available",
    "normalized_docks_available",
    "usage_ratio",
    "capacity_hour_interaction",
    "capacity_day_interaction",
]
ADDITIONAL_FEATURES = [
    "nearby_stations_closed",
    "nearby_stations_full",
    "nearby_stations_empty",
    "likelihood_fill",
    "likelihood_empty",
]
TARGET = "numbikesavailable"


def get_last_completed_batch(directory: Path) -> int:
    """Determine the last completed batch number."""
    directory = Path(directory)
    model_files = list(directory.glob("model_batch_*.pkl"))
    if not model_files:
        return 0
    batch_numbers = [int(f.stem.split("_")[2]) for f in model_files]
    return max(batch_numbers)


def save_feature_names(output_dir: Path):
    """Save feature names to JSON file."""
    feature_names_path = output_dir / "feature_names.json"
    all_features = BASE_FEATURES + ADDITIONAL_FEATURES
    with open(feature_names_path, "w") as f:
        json.dump(all_features, f)
    logger.info(f"Feature names saved to {feature_names_path}")


def train_station_model(
    station_data: pd.DataFrame,
    features: list,
    station: str,
) -> tuple:
    """Train a model for a single station.

    Note: RandomForest doesn't require feature scaling as it uses
    threshold-based splits that are invariant to monotonic transformations.
    """
    selected_features = features
    station_data = station_data.copy()

    # Check for problematic values
    if np.isinf(station_data[selected_features]).values.any():
        logger.warning(f"Skipping station {station}: infinite values")
        return None, None

    if (np.abs(station_data[selected_features]) > np.finfo(np.float64).max).values.any():
        logger.warning(f"Skipping station {station}: values too large")
        return None, None

    X = station_data[selected_features]
    y = station_data[TARGET]

    if len(X) < 2:
        logger.warning(f"Skipping station {station}: insufficient data")
        return None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    if len(X_train) < 2 or len(X_test) == 0:
        logger.warning(
            f"Skipping station {station}: insufficient data after split (train={len(X_train)}, test={len(X_test)})"
        )
        return None, None

    # Dynamically adjust CV splits (minimum 2 required for KFold)
    n_splits = max(2, min(N_SPLITS, len(X_train)))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    # Hyperparameter tuning
    random_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE),
        param_distributions=RF_PARAM_DISTRIBUTIONS,
        n_iter=N_ITER_SEARCH,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_

    # Evaluate
    cv_score = cross_val_score(
        best_model, X_train, y_train, cv=cv, scoring="neg_mean_squared_error"
    )
    logger.debug(f"Station {station} CV score: {cv_score.mean():.4f}")

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    logger.debug(f"Station {station} MSE: {mse:.4f}")

    return best_model, mse


def train_and_save_batches(
    bike_data: pd.DataFrame,
    stations: list,
    output_dir: Path,
    batch_size: int = BATCH_SIZE,
    start_batch: int = 0,
):
    """Train and save model batches."""
    all_features = BASE_FEATURES + ADDITIONAL_FEATURES
    problematic_stations = []

    for batch_start in range(start_batch * batch_size, len(stations), batch_size):
        batch_num = batch_start // batch_size + 1
        batch_end = min(batch_start + batch_size, len(stations))
        station_batch = stations[batch_start:batch_end]
        batch_models = {}

        logger.info(f"Training batch {batch_num} ({len(station_batch)} stations)")

        for station in tqdm(station_batch, desc=f"Batch {batch_num}", unit="station"):
            nearby_stations, station_data = calculate_nearby_station_status_adjustable(
                bike_data, station
            )
            station_data = station_data.copy()

            # Initialize nearby feature columns
            station_data["nearby_stations_closed"] = 0
            station_data["nearby_stations_full"] = 0
            station_data["nearby_stations_empty"] = 0
            station_data["likelihood_fill"] = 0.0
            station_data["likelihood_empty"] = 0.0

            if nearby_stations is not None and len(nearby_stations) >= 5:
                # Compute per-row nearby features to match inference parity
                for idx, row in station_data.iterrows():
                    date_filtered = nearby_stations[nearby_stations["date"] == row["date"]]
                    features = compute_nearby_features(date_filtered)
                    station_data.loc[idx, "nearby_stations_closed"] = features[
                        "nearby_stations_closed"
                    ]
                    station_data.loc[idx, "nearby_stations_full"] = features["nearby_stations_full"]
                    station_data.loc[idx, "nearby_stations_empty"] = features[
                        "nearby_stations_empty"
                    ]
                    station_data.loc[idx, "likelihood_fill"] = features["likelihood_fill"]
                    station_data.loc[idx, "likelihood_empty"] = features["likelihood_empty"]
            else:
                logger.debug(f"Station {station}: using base features (insufficient nearby)")

            model, mse = train_station_model(station_data, all_features, station)

            if model is not None:
                batch_models[station] = model
            else:
                problematic_stations.append(station)

        # Save batch
        model_path = output_dir / f"model_batch_{batch_num}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(batch_models, f)
        logger.info(f"Saved {len(batch_models)} models to {model_path}")

    if problematic_stations:
        logger.warning(f"Skipped {len(problematic_stations)} problematic stations")

    return problematic_stations


def main():
    """Main entry point."""
    args = parse_args()

    # Initialize logging
    init_cli_logging(verbose=args.verbose, quiet=args.quiet)

    logger.info("Starting model training")

    # Load and preprocess data
    bike_data = load_data(args.data_dir)
    bike_data = preprocess_data(bike_data)

    # Initialize nearby station columns
    bike_data["nearby_stations_closed"] = 0
    bike_data["nearby_stations_full"] = 0
    bike_data["nearby_stations_empty"] = 0
    bike_data["likelihood_fill"] = 0.0
    bike_data["likelihood_empty"] = 0.0

    # Add avg_bikes_hour_day feature
    bike_data["avg_bikes_hour_day"] = bike_data.groupby(["stationcode", "hour", "day_of_week"])[
        "numbikesavailable"
    ].transform("mean")

    # Get stations
    stations = bike_data["stationcode"].unique()
    if args.station_limit:
        stations = stations[: args.station_limit]

    logger.info(f"Found {len(stations)} stations to process")

    # Save feature names
    save_feature_names(args.output_dir)

    # Determine starting batch
    start_batch = get_last_completed_batch(args.output_dir) if args.resume else 0
    if start_batch > 0:
        logger.info(f"Resuming from batch {start_batch + 1}")

    # Train models
    train_and_save_batches(
        bike_data,
        stations,
        args.output_dir,
        batch_size=args.batch_size,
        start_batch=start_batch,
    )

    logger.info("Model training complete")


if __name__ == "__main__":
    main()
