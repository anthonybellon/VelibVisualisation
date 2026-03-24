"""
Tests for model and scaler serialization contracts.
"""

import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


class TestScalerSerialization:
    """Tests for StandardScaler serialization."""

    def test_scaler_roundtrip(self):
        """Test that scaler can be serialized and deserialized."""
        # Create and fit scaler
        scaler = StandardScaler()
        X = np.random.randn(100, 5)
        scaler.fit(X)

        # Serialize
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            pickle.dump(scaler, f)
            temp_path = f.name

        # Deserialize
        with open(temp_path, "rb") as f:
            loaded_scaler = pickle.load(f)

        # Verify
        X_test = np.random.randn(10, 5)
        original_transform = scaler.transform(X_test)
        loaded_transform = loaded_scaler.transform(X_test)

        np.testing.assert_array_almost_equal(original_transform, loaded_transform)

    def test_scaler_preserves_parameters(self):
        """Test that scaler parameters are preserved after serialization."""
        scaler = StandardScaler()
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler.fit(X)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            pickle.dump(scaler, f)
            temp_path = f.name

        with open(temp_path, "rb") as f:
            loaded_scaler = pickle.load(f)

        np.testing.assert_array_almost_equal(scaler.mean_, loaded_scaler.mean_)
        np.testing.assert_array_almost_equal(scaler.scale_, loaded_scaler.scale_)


class TestModelSerialization:
    """Tests for RandomForest model serialization."""

    def test_model_roundtrip(self, sample_model_and_scaler):
        """Test that model can be serialized and deserialized."""
        model, _ = sample_model_and_scaler

        # Serialize
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            pickle.dump(model, f)
            temp_path = f.name

        # Deserialize
        with open(temp_path, "rb") as f:
            loaded_model = pickle.load(f)

        # Verify predictions match
        X_test = np.random.randn(10, 4)
        original_pred = model.predict(X_test)
        loaded_pred = loaded_model.predict(X_test)

        np.testing.assert_array_almost_equal(original_pred, loaded_pred)

    def test_model_preserves_feature_names(self, sample_model_and_scaler):
        """Test that model preserves feature_names_in_ after serialization."""
        model, scaler = sample_model_and_scaler

        # Create DataFrame with named features to set feature_names_in_
        X = pd.DataFrame(
            {
                "hour": [1, 2, 3],
                "day_of_week": [0, 1, 2],
                "capacity": [30, 40, 50],
                "lag_1_hour": [10, 15, 20],
            }
        )
        scaler.transform(X)  # Verify scaler works

        model2 = RandomForestRegressor(n_estimators=10, random_state=42)
        model2.fit(X, [5, 10, 15])  # Fit with DataFrame to set feature names

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            pickle.dump(model2, f)
            temp_path = f.name

        with open(temp_path, "rb") as f:
            loaded_model = pickle.load(f)

        assert hasattr(loaded_model, "feature_names_in_")
        np.testing.assert_array_equal(model2.feature_names_in_, loaded_model.feature_names_in_)


class TestCombinedModelFormat:
    """Tests for the combined models and scalers format."""

    def test_combined_format_structure(self, sample_model_file):
        """Test that combined file has correct structure."""
        with open(sample_model_file, "rb") as f:
            data = pickle.load(f)

        assert "models" in data
        assert "scalers" in data
        assert isinstance(data["models"], dict)
        assert isinstance(data["scalers"], dict)

    def test_model_entry_structure(self, sample_model_file):
        """Test that each model entry has required fields."""
        with open(sample_model_file, "rb") as f:
            data = pickle.load(f)

        for _station_code, model_data in data["models"].items():
            assert "model" in model_data
            assert "scaler_idx" in model_data
            assert model_data["scaler_idx"] in data["scalers"]

    def test_scaler_model_compatibility(self, sample_model_file):
        """Test that scaler and model are compatible."""
        with open(sample_model_file, "rb") as f:
            data = pickle.load(f)

        for _station_code, model_data in data["models"].items():
            model = model_data["model"]
            scaler = data["scalers"][model_data["scaler_idx"]]

            # Get feature count from model
            if hasattr(model, "n_features_in_"):
                n_features = model.n_features_in_

                # Scaler should have same number of features
                assert scaler.n_features_in_ == n_features
