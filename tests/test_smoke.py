"""
End-to-end smoke tests for VelibVisualisation.

These tests verify that the main pipelines can run without errors
on small fixture data.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from feature_engineering import create_all_features


class TestEndToEndSmoke:
    """End-to-end smoke tests."""

    def test_feature_engineering_pipeline(self, sample_bike_data):
        """Test that full feature engineering pipeline runs without errors."""
        result = create_all_features(
            sample_bike_data, include_nearby_status=False, show_progress=False
        )

        assert not result.empty
        assert "hour" in result.columns
        assert "day_of_week" in result.columns
        assert "lag_1_hour" in result.columns
        assert "normalized_bikes_available" in result.columns

    def test_prediction_pipeline_smoke(self, sample_bike_data, sample_model):
        """Test that prediction pipeline runs without errors."""
        model = sample_model

        # Prepare data
        df = create_all_features(sample_bike_data, include_nearby_status=False, show_progress=False)

        # Filter to one station
        df = df[df["stationcode"] == df["stationcode"].iloc[0]].copy()

        # Prepare features
        features = ["hour", "day_of_week", "capacity", "lag_1_hour"]
        for feat in features:
            if feat not in df.columns:
                df[feat] = 0

        X = df[features].fillna(0)

        # Predict directly (no scaling needed for tree models)
        predictions = model.predict(X)

        assert len(predictions) == len(X)
        assert not np.isnan(predictions).any()

    def test_model_training_smoke(self, sample_bike_data):
        """Test that model training runs without errors on tiny data."""
        # Prepare data
        df = create_all_features(sample_bike_data, include_nearby_status=False, show_progress=False)

        # Filter to one station
        station = df["stationcode"].iloc[0]
        station_data = df[df["stationcode"] == station].copy()

        if len(station_data) < 10:
            pytest.skip("Not enough data for training test")

        # Prepare features and target
        features = ["hour", "day_of_week", "capacity"]
        for feat in features:
            if feat not in station_data.columns:
                station_data[feat] = 0

        X = station_data[features].fillna(0)
        y = station_data["numbikesavailable"]

        # Train directly on raw features (no scaling needed)
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)

        # Verify model works
        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_data_loading_smoke(self, sample_json_file):
        """Test that data can be loaded from JSON file."""
        with open(sample_json_file) as f:
            data = json.load(f)

        df = pd.DataFrame(data)

        assert not df.empty
        assert "stationcode" in df.columns
        assert "numbikesavailable" in df.columns

    def test_model_loading_smoke(self, sample_model_file):
        """Test that models can be loaded from pickle file."""
        with open(sample_model_file, "rb") as f:
            data = pickle.load(f)

        assert "models" in data
        assert len(data["models"]) > 0


class TestDataIntegrity:
    """Tests for data integrity checks."""

    def test_no_nan_in_predictions(self, sample_bike_data, sample_model):
        """Test that predictions don't contain NaN values."""
        model = sample_model

        df = create_all_features(sample_bike_data, include_nearby_status=False, show_progress=False)

        features = ["hour", "day_of_week", "capacity", "lag_1_hour"]
        for feat in features:
            if feat not in df.columns:
                df[feat] = 0

        X = df[features].fillna(0)
        predictions = model.predict(X)

        assert not np.isnan(predictions).any()
        assert not np.isinf(predictions).any()

    def test_features_no_unexpected_nan(self, sample_bike_data):
        """Test that feature engineering doesn't introduce unexpected NaN."""
        df = create_all_features(sample_bike_data, include_nearby_status=False, show_progress=False)

        # These columns should not have NaN after processing
        required_complete_cols = ["hour", "day_of_week", "lat", "lon"]

        for col in required_complete_cols:
            if col in df.columns:
                assert not df[col].isna().all(), f"Column {col} is all NaN"

    def test_coordinate_values_reasonable(self, sample_bike_data):
        """Test that coordinates are in reasonable range for Paris."""
        df = create_all_features(sample_bike_data, include_nearby_status=False, show_progress=False)

        # Paris coordinates roughly: lat 48.8-49.0, lon 2.2-2.5
        valid_lat = df[(df["lat"] >= 48.0) & (df["lat"] <= 49.5)]
        valid_lon = df[(df["lon"] >= 1.5) & (df["lon"] <= 3.0)]

        assert len(valid_lat) > 0, "No valid latitude values"
        assert len(valid_lon) > 0, "No valid longitude values"
