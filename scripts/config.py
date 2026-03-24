"""
Centralized configuration for VelibVisualisation project.
All paths, constants, and model parameters are defined here.
"""

from pathlib import Path

# ==============================================================================
# Path Configuration
# ==============================================================================


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.resolve()


def get_scripts_dir() -> Path:
    """Get the scripts directory."""
    return Path(__file__).parent.resolve()


# Base directories
PROJECT_ROOT = get_project_root()
SCRIPTS_DIR = get_scripts_dir()
DATA_DIR = PROJECT_ROOT / "data"

# Data subdirectories
HISTORICAL_DATA_DIR = DATA_DIR / "historical_data_cleaned"
DAILY_DATA_DIR = DATA_DIR / "do_not_touch" / "daily_velib_data" / "daily"
WEEKLY_DATA_DIR = DATA_DIR / "do_not_touch" / "weekly_velib_data"

# Model files
MODEL_SAVE_DIR = DATA_DIR
COMBINED_MODELS_PATH = DATA_DIR / "combined_models_and_scalers.pkl"
SCALER_PATH = DATA_DIR / "scaler.pkl"

# Data files
PREDICTIONS_INPUT_PATH = DATA_DIR / "2_organized_predictions.json"
PROCESSED_PREDICTIONS_PATH = DATA_DIR / "processed_predictions.json"
REALTIME_DATA_PATH = DATA_DIR / "velib-disponibilite-en-temps-reel.json"
WEEKLY_DATA_PATH = WEEKLY_DATA_DIR / "weekly_velib_data.json"

# External data URL
VELIB_REALTIME_URL = (
    "https://opendata.paris.fr/explore/dataset/"
    "velib-disponibilite-en-temps-reel/download/"
    "?format=json&timezone=Europe/Berlin"
)


# ==============================================================================
# Feature Engineering Constants
# ==============================================================================

# Lag feature windows (in hours)
LAG_1_HOUR = 1
LAG_1_DAY = 24

# Rolling window sizes (in hours)
ROLLING_WINDOW_7_DAYS = 7 * 24  # 168 hours
ROLLING_WINDOW_30_DAYS = 30 * 24  # 720 hours

# Epsilon to avoid division by zero
EPSILON = 1e-5

# Feature columns used for training
FEATURE_COLUMNS = [
    "capacity",
    "hour",
    "day_of_week",
    "lag_1_hour",
    "lag_1_day",
    "rolling_mean_7_days",
    "rolling_mean_30_days",
    "normalized_bikes_available",
    "normalized_docks_available",
    "usage_ratio",
    "capacity_hour_interaction",
    "capacity_day_interaction",
    "nearby_stations_closed",
    "nearby_stations_full",
    "nearby_stations_empty",
    "likelihood_fill",
    "likelihood_empty",
    "lat",
    "lon",
]

# Target column for prediction
TARGET_COLUMN = "numbikesavailable"

# Numeric columns that need special handling for large/infinite values
NUMERIC_COLUMNS = [
    "normalized_bikes_available",
    "normalized_docks_available",
    "usage_ratio",
    "capacity_hour_interaction",
    "capacity_day_interaction",
    "rolling_mean_7_days",
    "rolling_mean_30_days",
    "lag_1_hour",
    "lag_1_day",
]


# ==============================================================================
# Nearby Station Configuration
# ==============================================================================

# Radius parameters for nearby station calculations (in meters)
NEARBY_STATION_INITIAL_RADIUS = 500
NEARBY_STATION_MAX_RADIUS = 2000
NEARBY_STATION_RADIUS_INCREMENT = 500
MIN_NEARBY_STATIONS = 5


# ==============================================================================
# Model Training Configuration
# ==============================================================================

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Cross-validation
N_SPLITS = 5

# RandomizedSearchCV parameters
N_ITER_SEARCH = 50

# RandomForest hyperparameter search space
RF_PARAM_DISTRIBUTIONS = {
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [None, 10, 20, 30, 40],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
}

# Batch processing
BATCH_SIZE = 100


# ==============================================================================
# Station Coordinate Updates
# ==============================================================================

# Known coordinate corrections for stations with incorrect data
STATION_COORD_UPDATES = {
    "22504": {"lon": 2.253629, "lat": 48.905928},
    "25006": {"lon": 2.1961666225454, "lat": 48.862453313908},
    "10001": {"lon": 2.3600032, "lat": 48.8685433},
    "10001_relais": {"lon": 2.3599605, "lat": 48.8687079},
}


# ==============================================================================
# Logging Configuration
# ==============================================================================

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL = "INFO"


# ==============================================================================
# Helper Functions
# ==============================================================================


def ensure_dir_exists(path: Path) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_absolute_path(relative_path: str) -> Path:
    """
    Get absolute path from a path relative to the scripts directory.

    This function maintains backward compatibility with existing code
    that uses relative paths from the scripts directory.
    """
    return (SCRIPTS_DIR / relative_path).resolve()


# ==============================================================================
# Validation
# ==============================================================================


def validate_paths():
    """Validate that critical paths exist or can be created."""
    critical_dirs = [DATA_DIR]

    for dir_path in critical_dirs:
        if not dir_path.exists():
            print(f"Warning: Directory does not exist: {dir_path}")

    return True
