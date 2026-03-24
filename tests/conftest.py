"""
Test fixtures for VelibVisualisation tests.
"""

import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor


@pytest.fixture
def sample_bike_data():
    """Create sample bike data for testing."""
    np.random.seed(42)
    n_records = 100

    data = {
        "stationcode": np.random.choice(["10001", "10002", "10003"], n_records),
        "name": ["Station A", "Station B", "Station C"] * (n_records // 3 + 1),
        "is_installed": ["OUI"] * n_records,
        "is_renting": ["OUI"] * n_records,
        "capacity": np.random.randint(20, 50, n_records),
        "numdocksavailable": np.random.randint(0, 30, n_records),
        "numbikesavailable": np.random.randint(0, 30, n_records),
        "mechanical": np.random.randint(0, 20, n_records),
        "ebike": np.random.randint(0, 10, n_records),
        "is_returning": ["OUI"] * n_records,
        "duedate": pd.date_range("2024-01-01", periods=n_records, freq="h").strftime(
            "%Y-%m-%dT%H:%M:%S"
        ),
        "coordonnees_geo": [
            {
                "lat": 48.8566 + np.random.uniform(-0.01, 0.01),
                "lon": 2.3522 + np.random.uniform(-0.01, 0.01),
            }
            for _ in range(n_records)
        ],
        "nom_arrondissement_communes": ["Paris"] * n_records,
    }

    # Trim to exact n_records
    for key in data:
        data[key] = data[key][:n_records]

    return pd.DataFrame(data)


@pytest.fixture
def sample_model():
    """Create a sample model for testing.

    Note: RandomForest doesn't require scaling, so we train on raw features.
    """
    # Create sample training data
    np.random.seed(42)
    n_samples = 100

    X = pd.DataFrame(
        {
            "hour": np.random.randint(0, 24, n_samples),
            "day_of_week": np.random.randint(0, 7, n_samples),
            "capacity": np.random.randint(20, 50, n_samples),
            "lag_1_hour": np.random.randint(0, 30, n_samples),
        }
    )
    y = np.random.randint(0, 30, n_samples)

    # Train model directly on raw features
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)

    return model


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_json_file(temp_dir, sample_bike_data):
    """Create a sample JSON file with bike data."""
    file_path = temp_dir / "sample_data.json"
    data = sample_bike_data.to_dict(orient="records")

    with open(file_path, "w") as f:
        json.dump(data, f)

    return file_path


@pytest.fixture
def sample_model_file(temp_dir, sample_model):
    """Create a sample pickle file with models."""
    model = sample_model

    combined_data = {
        "models": {
            "10001": model,
            "10002": model,
        },
    }

    file_path = temp_dir / "combined_models_and_scalers.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(combined_data, f)

    return file_path
