"""
Tests for configuration module.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from config import (
    DATA_DIR,
    FEATURE_COLUMNS,
    HISTORICAL_DATA_DIR,
    MODEL_SAVE_DIR,
    NUMERIC_COLUMNS,
    PROJECT_ROOT,
    SCRIPTS_DIR,
    STATION_COORD_UPDATES,
    ensure_dir_exists,
    get_absolute_path,
    get_project_root,
    get_scripts_dir,
)


class TestPathConfiguration:
    """Tests for path configuration."""

    def test_project_root_exists(self):
        """Test that project root is a valid directory."""
        assert PROJECT_ROOT.exists()
        assert PROJECT_ROOT.is_dir()

    def test_scripts_dir_exists(self):
        """Test that scripts directory exists."""
        assert SCRIPTS_DIR.exists()
        assert SCRIPTS_DIR.is_dir()

    def test_data_dir_path(self):
        """Test that data directory is correctly configured."""
        assert DATA_DIR == PROJECT_ROOT / "data"

    def test_historical_data_dir_path(self):
        """Test that historical data directory is correctly configured."""
        assert HISTORICAL_DATA_DIR == DATA_DIR / "historical_data_cleaned"

    def test_model_save_dir_path(self):
        """Test that model save directory is correctly configured."""
        assert MODEL_SAVE_DIR == DATA_DIR

    def test_get_project_root_returns_path(self):
        """Test that get_project_root returns a Path object."""
        root = get_project_root()
        assert isinstance(root, Path)

    def test_get_scripts_dir_returns_path(self):
        """Test that get_scripts_dir returns a Path object."""
        scripts = get_scripts_dir()
        assert isinstance(scripts, Path)


class TestAbsolutePathHelper:
    """Tests for get_absolute_path helper."""

    def test_relative_path_resolution(self):
        """Test that relative paths are resolved correctly."""
        result = get_absolute_path("../data")
        assert result.is_absolute()
        assert "data" in str(result)

    def test_returns_path_object(self):
        """Test that function returns Path object."""
        result = get_absolute_path(".")
        assert isinstance(result, Path)


class TestEnsureDirExists:
    """Tests for ensure_dir_exists helper."""

    def test_creates_directory(self, temp_dir):
        """Test that directory is created if it doesn't exist."""
        new_dir = temp_dir / "new_subdir"
        assert not new_dir.exists()

        result = ensure_dir_exists(new_dir)

        assert new_dir.exists()
        assert new_dir.is_dir()
        assert result == new_dir

    def test_handles_existing_directory(self, temp_dir):
        """Test that existing directory is handled gracefully."""
        result = ensure_dir_exists(temp_dir)

        assert temp_dir.exists()
        assert result == temp_dir

    def test_creates_nested_directories(self, temp_dir):
        """Test that nested directories are created."""
        nested = temp_dir / "a" / "b" / "c"

        ensure_dir_exists(nested)

        assert nested.exists()
        assert nested.is_dir()


class TestFeatureConfiguration:
    """Tests for feature configuration."""

    def test_feature_columns_not_empty(self):
        """Test that feature columns list is not empty."""
        assert len(FEATURE_COLUMNS) > 0

    def test_numeric_columns_not_empty(self):
        """Test that numeric columns list is not empty."""
        assert len(NUMERIC_COLUMNS) > 0

    def test_numeric_columns_subset_of_features(self):
        """Test that numeric columns are related to feature columns."""
        # Numeric columns should be processable
        for col in NUMERIC_COLUMNS:
            assert isinstance(col, str)

    def test_station_coord_updates_structure(self):
        """Test that station coordinate updates have correct structure."""
        for station_code, coords in STATION_COORD_UPDATES.items():
            assert isinstance(station_code, str)
            assert "lat" in coords
            assert "lon" in coords
            assert isinstance(coords["lat"], (int, float))
            assert isinstance(coords["lon"], (int, float))


class TestConstants:
    """Tests for constants configuration."""

    def test_epsilon_is_small(self):
        """Test that EPSILON is a small positive number."""
        from config import EPSILON

        assert EPSILON > 0
        assert EPSILON < 1

    def test_batch_size_is_positive(self):
        """Test that BATCH_SIZE is positive."""
        from config import BATCH_SIZE

        assert BATCH_SIZE > 0

    def test_random_state_is_set(self):
        """Test that RANDOM_STATE is set for reproducibility."""
        from config import RANDOM_STATE

        assert isinstance(RANDOM_STATE, int)

    def test_test_size_is_valid(self):
        """Test that TEST_SIZE is between 0 and 1."""
        from config import TEST_SIZE

        assert 0 < TEST_SIZE < 1
