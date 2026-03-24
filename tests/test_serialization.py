"""
Tests for model serialization contracts.
"""

import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


@pytest.fixture
def temp_pickle_file():
    """Create a temporary pickle file that's cleaned up after the test."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
        temp_path = Path(f.name)
    yield temp_path
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


class TestModelSerialization:
    """Tests for RandomForest model serialization."""

    def test_model_roundtrip(self, sample_model, temp_pickle_file):
        """Test that model can be serialized and deserialized."""
        model = sample_model

        # Serialize
        with open(temp_pickle_file, "wb") as f:
            pickle.dump(model, f)

        # Deserialize
        with open(temp_pickle_file, "rb") as f:
            loaded_model = pickle.load(f)

        # Verify predictions match (use DataFrame with feature names to avoid warnings)
        X_test = pd.DataFrame(
            {
                "hour": np.random.randint(0, 24, 10),
                "day_of_week": np.random.randint(0, 7, 10),
                "capacity": np.random.randint(20, 50, 10),
                "lag_1_hour": np.random.randint(0, 30, 10),
            }
        )
        original_pred = model.predict(X_test)
        loaded_pred = loaded_model.predict(X_test)

        np.testing.assert_array_almost_equal(original_pred, loaded_pred)

    def test_model_preserves_feature_names(self, temp_pickle_file):
        """Test that model preserves feature_names_in_ after serialization."""
        # Create DataFrame with named features to set feature_names_in_
        X = pd.DataFrame(
            {
                "hour": [1, 2, 3],
                "day_of_week": [0, 1, 2],
                "capacity": [30, 40, 50],
                "lag_1_hour": [10, 15, 20],
            }
        )

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, [5, 10, 15])  # Fit with DataFrame to set feature names

        with open(temp_pickle_file, "wb") as f:
            pickle.dump(model, f)

        with open(temp_pickle_file, "rb") as f:
            loaded_model = pickle.load(f)

        assert hasattr(loaded_model, "feature_names_in_")
        np.testing.assert_array_equal(model.feature_names_in_, loaded_model.feature_names_in_)


class TestCombinedModelFormat:
    """Tests for the combined models format."""

    def test_combined_format_structure(self, sample_model_file):
        """Test that combined file has correct structure."""
        with open(sample_model_file, "rb") as f:
            data = pickle.load(f)

        assert "models" in data
        assert isinstance(data["models"], dict)

    def test_model_entry_is_valid(self, sample_model_file):
        """Test that each model entry is a valid RandomForest model."""
        with open(sample_model_file, "rb") as f:
            data = pickle.load(f)

        for station_code, model in data["models"].items():
            assert hasattr(model, "predict"), f"Model for {station_code} missing predict method"
            assert hasattr(
                model, "feature_names_in_"
            ), f"Model for {station_code} missing feature names"
