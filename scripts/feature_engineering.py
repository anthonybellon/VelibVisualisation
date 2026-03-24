"""
Feature engineering module for VelibVisualisation.

This module provides shared feature engineering functions used by both
training and inference pipelines to ensure feature parity.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from tqdm import tqdm

from config import (
    EPSILON,
    LAG_1_DAY,
    LAG_1_HOUR,
    MIN_NEARBY_STATIONS,
    NEARBY_STATION_INITIAL_RADIUS,
    NEARBY_STATION_MAX_RADIUS,
    NEARBY_STATION_RADIUS_INCREMENT,
    NUMERIC_COLUMNS,
    ROLLING_WINDOW_7_DAYS,
    ROLLING_WINDOW_30_DAYS,
    STATION_COORD_UPDATES,
)

logger = logging.getLogger(__name__)


def extract_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract latitude and longitude from coordonnees_geo column.

    Args:
        df: DataFrame with 'coordonnees_geo' column containing dict with lat/lon.

    Returns:
        DataFrame with 'lat' and 'lon' columns added.
    """
    df = df.copy()

    def extract_lat(x):
        if isinstance(x, dict):
            return x.get("lat", np.nan)
        return np.nan

    def extract_lon(x):
        if isinstance(x, dict):
            return x.get("lon", np.nan)
        return np.nan

    df["lat"] = df["coordonnees_geo"].apply(extract_lat)
    df["lon"] = df["coordonnees_geo"].apply(extract_lon)

    logger.info("Extracted latitude and longitude from coordinates")
    return df


def apply_coordinate_updates(df: pd.DataFrame, updates: Optional[dict] = None) -> pd.DataFrame:
    """
    Apply known coordinate corrections for stations with incorrect data.

    Args:
        df: DataFrame with 'stationcode', 'lat', and 'lon' columns.
        updates: Dict mapping station codes to coordinate dicts.
                 Uses STATION_COORD_UPDATES from config if not provided.

    Returns:
        DataFrame with corrected coordinates.
    """
    df = df.copy()
    updates = updates or STATION_COORD_UPDATES

    for station_code, coords in updates.items():
        mask = df["stationcode"] == station_code
        if mask.any():
            df.loc[mask, "lat"] = coords["lat"]
            df.loc[mask, "lon"] = coords["lon"]
            logger.debug(f"Updated coordinates for station {station_code}")

    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add hour and day_of_week features from the date column.

    Args:
        df: DataFrame with 'date' column (datetime).

    Returns:
        DataFrame with 'hour' and 'day_of_week' columns added.
    """
    df = df.copy()
    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek

    logger.info("Added temporal features: hour, day_of_week")
    return df


def add_lag_features(
    df: pd.DataFrame, lag_1_hour: int = LAG_1_HOUR, lag_1_day: int = LAG_1_DAY
) -> pd.DataFrame:
    """
    Add lag features for bike availability.

    Args:
        df: DataFrame sorted by ['stationcode', 'date'].
        lag_1_hour: Number of periods for 1-hour lag (default: 1).
        lag_1_day: Number of periods for 1-day lag (default: 24).

    Returns:
        DataFrame with lag features added.
    """
    df = df.copy()
    df = df.sort_values(by=["stationcode", "date"])

    df["lag_1_hour"] = df.groupby("stationcode")["numbikesavailable"].shift(lag_1_hour)
    df["lag_1_day"] = df.groupby("stationcode")["numbikesavailable"].shift(lag_1_day)

    logger.info(f"Added lag features: lag_1_hour ({lag_1_hour}), lag_1_day ({lag_1_day})")
    return df


def add_rolling_features(
    df: pd.DataFrame,
    window_7_days: int = ROLLING_WINDOW_7_DAYS,
    window_30_days: int = ROLLING_WINDOW_30_DAYS,
) -> pd.DataFrame:
    """
    Add rolling mean features.

    Args:
        df: DataFrame sorted by ['stationcode', 'date'].
        window_7_days: Window size for 7-day rolling mean (default: 168 hours).
        window_30_days: Window size for 30-day rolling mean (default: 720 hours).

    Returns:
        DataFrame with rolling mean features added.
    """
    df = df.copy()

    df["rolling_mean_7_days"] = df.groupby("stationcode")["numbikesavailable"].transform(
        lambda x: x.rolling(window=window_7_days, min_periods=1).mean()
    )
    df["rolling_mean_30_days"] = df.groupby("stationcode")["numbikesavailable"].transform(
        lambda x: x.rolling(window=window_30_days, min_periods=1).mean()
    )

    logger.info("Added rolling mean features: 7-day, 30-day")
    return df


def add_capacity_features(df: pd.DataFrame, epsilon: float = EPSILON) -> pd.DataFrame:
    """
    Add capacity-based features.

    Args:
        df: DataFrame with 'numbikesavailable', 'numdocksavailable',
            'capacity', 'hour', 'day_of_week' columns.
        epsilon: Small value to avoid division by zero.

    Returns:
        DataFrame with capacity-based features added.
    """
    df = df.copy()

    df["normalized_bikes_available"] = df["numbikesavailable"] / (df["capacity"] + epsilon)
    df["normalized_docks_available"] = df["numdocksavailable"] / (df["capacity"] + epsilon)
    df["usage_ratio"] = df["numbikesavailable"] / (df["capacity"] + epsilon)
    df["capacity_hour_interaction"] = df["capacity"] * df["hour"]
    df["capacity_day_interaction"] = df["capacity"] * df["day_of_week"]

    logger.info("Added capacity-based features")
    return df


def handle_large_values(
    df: pd.DataFrame, columns: Optional[list[str]] = None, decimal_places: int = 6
) -> tuple[pd.DataFrame, list[str]]:
    """
    Handle infinite or excessively large values in numeric columns.

    Args:
        df: DataFrame to process.
        columns: List of column names to process. Uses NUMERIC_COLUMNS if not provided.
        decimal_places: Number of decimal places to round to.

    Returns:
        Tuple of (processed DataFrame, list of station codes with problematic values).
    """
    df = df.copy()
    columns = columns or NUMERIC_COLUMNS
    problematic_stations = []

    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column {col} not found in DataFrame, skipping")
            continue

        # Replace infinities with NaN
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        # Round to specified decimal places
        df[col] = df[col].round(decimal_places)

        # Identify rows with NaN after rounding
        problematic = df[col].isna()

        if problematic.any():
            problematic_stations.extend(df.loc[problematic, "stationcode"].unique())
            df[col] = df[col].fillna(0)

    if problematic_stations:
        unique_problematic = list(set(problematic_stations))
        logger.warning(f"Stations with problematic values: {unique_problematic}")

    return df, problematic_stations


def report_data_quality(df: pd.DataFrame) -> dict:
    """
    Report data quality issues including infinite and large values.

    Args:
        df: DataFrame to analyze.

    Returns:
        Dict with data quality information.
    """
    numeric_data = df.select_dtypes(include=[np.number])

    report = {
        "infinite_columns": [],
        "large_value_columns": [],
        "nan_columns": [],
    }

    # Check for infinite values
    infinite_mask = numeric_data.isin([np.inf, -np.inf]).any()
    if infinite_mask.any():
        report["infinite_columns"] = numeric_data.columns[infinite_mask].tolist()
        logger.warning(f"Infinite values found in: {report['infinite_columns']}")

    # Check for large values
    large_mask = (numeric_data.abs() > 1e10).any()
    if large_mask.any():
        report["large_value_columns"] = numeric_data.columns[large_mask].tolist()
        logger.warning(f"Large values found in: {report['large_value_columns']}")

    # Check for NaN values
    nan_mask = numeric_data.isna().any()
    if nan_mask.any():
        report["nan_columns"] = numeric_data.columns[nan_mask].tolist()
        logger.info(f"NaN values found in: {report['nan_columns']}")

    return report


def calculate_nearby_station_status(
    df: pd.DataFrame, radius: int = NEARBY_STATION_INITIAL_RADIUS, show_progress: bool = True
) -> pd.DataFrame:
    """
    Calculate nearby station status metrics.

    Args:
        df: DataFrame with station data.
        radius: Search radius in meters.
        show_progress: Whether to show progress bar.

    Returns:
        DataFrame with nearby station features added.
    """
    df = df.copy()
    stations = df["stationcode"].unique()

    coords = df[["lat", "lon"]].drop_duplicates().values
    kd_tree = KDTree(coords)

    # Initialize columns
    df["nearby_stations_closed"] = 0
    df["nearby_stations_full"] = 0
    df["nearby_stations_empty"] = 0
    df["likelihood_fill"] = 0.0
    df["likelihood_empty"] = 0.0

    iterator = (
        tqdm(stations, desc="Calculating nearby station status") if show_progress else stations
    )

    for station in iterator:
        station_data = df[df["stationcode"] == station]
        station_coords = station_data[["lat", "lon"]].iloc[0].values

        # Convert radius from meters to approximate degrees
        indices = kd_tree.query_ball_point(station_coords, radius / 1000.0 / 111.32)
        nearby_stations = df.iloc[indices]

        for index, row in station_data.iterrows():
            date_filtered = nearby_stations[nearby_stations["date"] == row["date"]]

            nearby_closed = (
                date_filtered["is_installed"].apply(lambda x: 1 if x == "NON" else 0).sum()
            )
            nearby_full = (date_filtered["numbikesavailable"] == date_filtered["capacity"]).sum()
            nearby_empty = (date_filtered["numbikesavailable"] == 0).sum()

            df.loc[index, "nearby_stations_closed"] = nearby_closed
            df.loc[index, "nearby_stations_full"] = nearby_full
            df.loc[index, "nearby_stations_empty"] = nearby_empty

            total_nearby = len(date_filtered)
            if total_nearby > 0:
                likelihood_fill = nearby_full / total_nearby
                likelihood_empty = nearby_empty / total_nearby
            else:
                likelihood_fill = 0.0
                likelihood_empty = 0.0

            df.loc[index, "likelihood_fill"] = likelihood_fill
            df.loc[index, "likelihood_empty"] = likelihood_empty

            # Adjust likelihood based on nearby station statuses
            if nearby_full > total_nearby / 2:
                df.loc[index, "likelihood_fill"] *= 1.5
            if nearby_empty > total_nearby / 2:
                df.loc[index, "likelihood_empty"] *= 1.5

    logger.info("Calculated nearby station status features")
    return df


def calculate_nearby_station_status_adjustable(
    df: pd.DataFrame,
    station: str,
    initial_radius: int = NEARBY_STATION_INITIAL_RADIUS,
    max_radius: int = NEARBY_STATION_MAX_RADIUS,
    increment: int = NEARBY_STATION_RADIUS_INCREMENT,
    min_stations: int = MIN_NEARBY_STATIONS,
) -> tuple[Optional[pd.DataFrame], pd.DataFrame]:
    """
    Calculate nearby station status with adjustable radius.

    Expands the search radius until minimum number of nearby stations is found.

    Args:
        df: DataFrame with station data.
        station: Station code to analyze.
        initial_radius: Starting search radius in meters.
        max_radius: Maximum search radius in meters.
        increment: Radius increment in meters.
        min_stations: Minimum number of nearby stations required.

    Returns:
        Tuple of (nearby stations DataFrame, station data DataFrame).
    """
    station_data = df[df["stationcode"] == station]
    if station_data.empty:
        logger.warning(f"No data found for station {station}")
        return None, station_data

    unique_coords = df[["lat", "lon"]].drop_duplicates().values
    station_codes = df[["stationcode"]].drop_duplicates().values.flatten()
    kd_tree = KDTree(unique_coords)

    station_coords = station_data[["lat", "lon"]].iloc[0].values
    target_station_code = station_data["stationcode"].iloc[0]

    radius = initial_radius
    nearby_stations = pd.DataFrame()

    while radius <= max_radius:
        # Convert radius from meters to approximate degrees
        indices = kd_tree.query_ball_point(station_coords, radius / 1000.0 / 111.32)

        # Filter out invalid indices
        valid_indices = [i for i in indices if i < len(station_codes)]

        if len(valid_indices) != len(indices):
            logger.debug(f"Filtered {len(indices) - len(valid_indices)} out-of-bounds indices")

        # Exclude target station
        nearby_indices = [i for i in valid_indices if station_codes[i] != target_station_code]

        nearby_stations = df.iloc[nearby_indices]
        nearby_stations = nearby_stations[nearby_stations["is_installed"] != "NON"]

        if len(nearby_stations) >= min_stations:
            logger.debug(f"Found {len(nearby_stations)} nearby stations at radius {radius}m")
            return nearby_stations, station_data

        radius += increment

    logger.debug(f"Only found {len(nearby_stations)} nearby stations at max radius")
    return nearby_stations, station_data


def preprocess_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess datetime column with timezone handling.

    Args:
        df: DataFrame with 'duedate' column.

    Returns:
        DataFrame with 'date' column (timezone-aware datetime).
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["duedate"])

    if df["date"].dt.tz is None:
        df["date"] = df["date"].dt.tz_localize("UTC")

    logger.info("Preprocessed datetime column")
    return df


def create_all_features(
    df: pd.DataFrame, include_nearby_status: bool = False, show_progress: bool = True
) -> pd.DataFrame:
    """
    Create all features for the dataset.

    This is the main entry point for feature engineering that ensures
    feature parity between training and inference.

    Args:
        df: Raw DataFrame with bike data.
        include_nearby_status: Whether to calculate nearby station features.
        show_progress: Whether to show progress bars.

    Returns:
        DataFrame with all features added.
    """
    logger.info("Starting feature engineering pipeline")

    # Preprocess datetime
    df = preprocess_datetime(df)

    # Extract and fix coordinates
    if "lat" not in df.columns or "lon" not in df.columns:
        df = extract_coordinates(df)
    df = apply_coordinate_updates(df)

    # Drop rows with missing coordinates
    initial_rows = len(df)
    df = df.dropna(subset=["lat", "lon"])
    if len(df) < initial_rows:
        logger.info(f"Dropped {initial_rows - len(df)} rows with missing coordinates")

    # Fill remaining NaN values
    df = df.fillna(0)

    # Add temporal features
    df = add_temporal_features(df)

    # Add lag features
    df = add_lag_features(df)

    # Add rolling features
    df = add_rolling_features(df)

    # Add capacity features
    df = add_capacity_features(df)

    # Handle large values
    df, problematic_stations = handle_large_values(df)

    # Calculate nearby station status if requested
    if include_nearby_status:
        df = calculate_nearby_station_status(df, show_progress=show_progress)

    # Report data quality
    report_data_quality(df)

    logger.info("Feature engineering complete")
    return df
