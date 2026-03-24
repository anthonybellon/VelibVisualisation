"""Make predictions and compare with actual values.

Usage:
    python predict_and_compare.py [OPTIONS]

Options:
    --input PATH          Input data file (JSON)
    --models PATH         Combined models file (pickle)
    --output PATH         Output predictions file (JSON)
    --verbose, -v         Enable verbose logging
    --quiet, -q           Suppress info logging
"""

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import (
    COMBINED_MODELS_PATH,
    DATA_DIR,
    PREDICTIONS_INPUT_PATH,
)
from feature_engineering import (
    add_capacity_features,
    add_lag_features,
    add_rolling_features,
    add_temporal_features,
    calculate_nearby_station_status,
    handle_large_values,
    preprocess_datetime,
)
from logging_config import get_logger, init_cli_logging

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Make predictions and compare with actual values",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PREDICTIONS_INPUT_PATH,
        help="Input data file (JSON)",
    )
    parser.add_argument(
        "--models",
        type=Path,
        default=COMBINED_MODELS_PATH,
        help="Combined models file (pickle)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "prediction_results_final.json",
        help="Output predictions file (JSON)",
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


def load_current_data(input_path: Path) -> pd.DataFrame:
    """Load current bike data from JSON file."""
    logger.info(f"Loading data from {input_path}")

    with open(input_path) as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    logger.info(f"Loaded {len(df)} records")
    return df


def load_models(models_path: Path) -> dict:
    """Load combined models."""
    logger.info(f"Loading models from {models_path}")

    with open(models_path, "rb") as f:
        combined_data = pickle.load(f)

    models = combined_data["models"]

    logger.info(f"Loaded {len(models)} models")
    return models


def preprocess_current_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess current bike data for prediction."""
    logger.info("Preprocessing data...")

    # Datetime preprocessing
    df = preprocess_datetime(df)

    # Extract coordinates
    coords = pd.json_normalize(df["coordonnees_geo"])
    df["lat"] = coords["lat"]
    df["lon"] = coords["lon"]

    # Fill NaN values
    df = df.fillna(0)

    # Add temporal features
    df = add_temporal_features(df)

    # Keep unscaled versions for output
    df["hour_unscaled"] = df["date"].dt.hour
    df["day_of_week_unscaled"] = df["date"].dt.dayofweek

    # Add lag and rolling features
    df = add_lag_features(df)
    df = add_rolling_features(df)

    # Add capacity features
    df = add_capacity_features(df)

    # Handle large values
    df, problematic_stations = handle_large_values(df)
    if problematic_stations:
        logger.warning(f"Stations with problematic values: {problematic_stations}")

    logger.info("Preprocessing complete")
    return df


def make_predictions(df: pd.DataFrame, models: dict, show_progress: bool = True) -> pd.DataFrame:
    """Make predictions for all stations.

    Note: RandomForest models don't require scaling - predictions use
    raw feature values directly.
    """
    # Add avg_bikes_hour_day feature
    df["avg_bikes_hour_day"] = df.groupby(["stationcode", "hour", "day_of_week"])[
        "numbikesavailable"
    ].transform("mean")

    stations_to_test = list(models.keys())
    results = []
    missing_stations = []

    logger.info(f"Making predictions for {len(stations_to_test)} stations")

    iterator = tqdm(stations_to_test, desc="Predicting") if show_progress else stations_to_test

    for station in iterator:
        if station not in models:
            missing_stations.append(station)
            continue

        station_data = df[df["stationcode"] == station].copy()
        if station_data.empty:
            logger.debug(f"No data for station {station}")
            continue

        model = models[station]

        # Get features used during training
        trained_features = model.feature_names_in_

        # Ensure all features exist
        for feature in trained_features:
            if feature not in station_data.columns:
                station_data[feature] = 0

        X = station_data[trained_features]
        y_true = station_data["numbikesavailable"]

        # Predict directly (no scaling needed for tree models)
        y_pred = model.predict(X)

        station_data.loc[:, "predicted_bikesavailable"] = y_pred
        station_data.loc[:, "actual_bikesavailable"] = y_true
        results.append(station_data)

    if missing_stations:
        logger.warning(f"Missing models for {len(missing_stations)} stations")

    if not results:
        raise ValueError("No predictions were made")

    results_df = pd.concat(results)
    logger.info(f"Made predictions for {len(results)} stations")

    return results_df


def format_output(df: pd.DataFrame, original_df: pd.DataFrame) -> list:
    """Format prediction results for output."""
    df = df.drop(columns=["date", "lat", "lon"], errors="ignore")

    # Add unscaled values
    df["hour_unscaled"] = original_df["hour_unscaled"]
    df["day_of_week_unscaled"] = original_df["day_of_week_unscaled"]

    # Select output columns
    output_columns = [
        "stationcode",
        "name",
        "is_installed",
        "capacity",
        "numdocksavailable",
        "numbikesavailable",
        "mechanical",
        "ebike",
        "is_renting",
        "is_returning",
        "coordonnees_geo",
        "predicted_bikesavailable",
        "actual_bikesavailable",
        "hour_unscaled",
        "day_of_week_unscaled",
    ]

    # Filter to available columns
    available_columns = [c for c in output_columns if c in df.columns]
    df = df[available_columns]

    return df.to_dict(orient="records")


def main():
    """Main entry point."""
    args = parse_args()

    # Initialize logging
    init_cli_logging(verbose=args.verbose, quiet=args.quiet)

    logger.info("Starting prediction pipeline")

    # Load data
    current_bike_data = load_current_data(args.input)

    # Preprocess
    current_bike_data = preprocess_current_data(current_bike_data)

    # Calculate nearby station status
    logger.info("Calculating nearby station status...")
    current_bike_data = calculate_nearby_station_status(current_bike_data)

    # Load models
    models = load_models(args.models)

    # Make predictions
    results_df = make_predictions(current_bike_data, models)

    # Format output
    results_json = format_output(results_df, current_bike_data)

    # Save results
    with open(args.output, "w") as f:
        json.dump(results_json, f, indent=4)

    logger.info(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
