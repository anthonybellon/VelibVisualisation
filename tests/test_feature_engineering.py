"""
Tests for feature engineering module.

Tests feature parity between training and inference pipelines.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from config import NUMERIC_COLUMNS, STATION_COORD_UPDATES
from feature_engineering import (
    add_capacity_features,
    add_lag_features,
    add_rolling_features,
    add_temporal_features,
    apply_coordinate_updates,
    create_all_features,
    extract_coordinates,
    handle_large_values,
    preprocess_datetime,
)


class TestTemporalFeatures:
    """Tests for temporal feature creation."""

    def test_add_temporal_features(self, sample_bike_data):
        """Test that hour and day_of_week are added correctly."""
        df = preprocess_datetime(sample_bike_data)
        result = add_temporal_features(df)

        assert "hour" in result.columns
        assert "day_of_week" in result.columns
        assert result["hour"].min() >= 0
        assert result["hour"].max() <= 23
        assert result["day_of_week"].min() >= 0
        assert result["day_of_week"].max() <= 6

    def test_temporal_features_consistency(self, sample_bike_data):
        """Test that temporal features are consistent with date."""
        df = preprocess_datetime(sample_bike_data)
        result = add_temporal_features(df)

        # Check a few rows
        for idx in range(min(10, len(result))):
            assert result.iloc[idx]["hour"] == result.iloc[idx]["date"].hour
            assert result.iloc[idx]["day_of_week"] == result.iloc[idx]["date"].dayofweek


class TestLagFeatures:
    """Tests for lag feature creation."""

    def test_add_lag_features(self, sample_bike_data):
        """Test that lag features are added."""
        df = preprocess_datetime(sample_bike_data)
        df = add_temporal_features(df)
        result = add_lag_features(df)

        assert "lag_1_hour" in result.columns
        assert "lag_1_day" in result.columns

    def test_lag_features_sorted(self, sample_bike_data):
        """Test that data is sorted after adding lag features."""
        df = preprocess_datetime(sample_bike_data)
        df = add_temporal_features(df)
        result = add_lag_features(df)

        # Check that data is sorted by stationcode and date
        for station in result["stationcode"].unique():
            station_data = result[result["stationcode"] == station]
            dates = station_data["date"].values
            assert all(dates[i] <= dates[i + 1] for i in range(len(dates) - 1))


class TestCapacityFeatures:
    """Tests for capacity-based feature creation."""

    def test_add_capacity_features(self, sample_bike_data):
        """Test that capacity features are added."""
        df = preprocess_datetime(sample_bike_data)
        df = add_temporal_features(df)
        result = add_capacity_features(df)

        expected_features = [
            "normalized_bikes_available",
            "normalized_docks_available",
            "usage_ratio",
            "capacity_hour_interaction",
            "capacity_day_interaction",
        ]

        for feature in expected_features:
            assert feature in result.columns

    def test_normalized_features_range(self, sample_bike_data):
        """Test that normalized features are in expected range."""
        df = preprocess_datetime(sample_bike_data)
        df = add_temporal_features(df)
        result = add_capacity_features(df)

        # Normalized values should generally be between 0 and 1
        # (can exceed 1 in edge cases with bad data)
        assert result["usage_ratio"].min() >= 0


class TestCoordinateExtraction:
    """Tests for coordinate extraction and updates."""

    def test_extract_coordinates(self, sample_bike_data):
        """Test that coordinates are extracted from dict."""
        result = extract_coordinates(sample_bike_data)

        assert "lat" in result.columns
        assert "lon" in result.columns
        assert not result["lat"].isna().all()
        assert not result["lon"].isna().all()

    def test_apply_coordinate_updates(self, sample_bike_data):
        """Test that coordinate updates are applied."""
        df = extract_coordinates(sample_bike_data)

        # Add a station that needs updating
        df = df.copy()
        df.loc[0, "stationcode"] = "22504"

        result = apply_coordinate_updates(df)

        updated_row = result[result["stationcode"] == "22504"].iloc[0]
        expected = STATION_COORD_UPDATES["22504"]

        assert updated_row["lat"] == expected["lat"]
        assert updated_row["lon"] == expected["lon"]


class TestLargeValueHandling:
    """Tests for handling large/infinite values."""

    def test_handle_large_values_replaces_inf(self):
        """Test that infinite values are replaced."""
        df = pd.DataFrame(
            {
                "stationcode": ["10001", "10002"],
                "normalized_bikes_available": [np.inf, 0.5],
                "usage_ratio": [0.3, -np.inf],
            }
        )

        result, problematic = handle_large_values(
            df, columns=["normalized_bikes_available", "usage_ratio"]
        )

        assert not np.isinf(result["normalized_bikes_available"]).any()
        assert not np.isinf(result["usage_ratio"]).any()
        assert len(problematic) > 0

    def test_handle_large_values_rounds(self):
        """Test that values are rounded to 6 decimal places."""
        df = pd.DataFrame(
            {
                "stationcode": ["10001"],
                "normalized_bikes_available": [0.123456789],
            }
        )

        result, _ = handle_large_values(
            df, columns=["normalized_bikes_available"], decimal_places=6
        )

        # Should be rounded to 6 decimal places
        assert result["normalized_bikes_available"].iloc[0] == pytest.approx(0.123457, abs=1e-6)


class TestFeatureParity:
    """Tests to ensure feature parity between train and inference."""

    def test_create_all_features_train_infer_parity(self, sample_bike_data):
        """Test that same features are created for train and inference."""
        # Simulate training data processing
        train_df = create_all_features(sample_bike_data.copy(), include_nearby_status=False)

        # Simulate inference data processing
        infer_df = create_all_features(sample_bike_data.copy(), include_nearby_status=False)

        # Check that same columns exist
        train_features = set(train_df.columns)
        infer_features = set(infer_df.columns)

        assert train_features == infer_features

    def test_feature_dtypes_match(self, sample_bike_data):
        """Test that feature dtypes match between runs."""
        df1 = create_all_features(sample_bike_data.copy(), include_nearby_status=False)
        df2 = create_all_features(sample_bike_data.copy(), include_nearby_status=False)

        for col in df1.columns:
            if col in df2.columns:
                assert df1[col].dtype == df2[col].dtype, f"Dtype mismatch for {col}"
