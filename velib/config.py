"""Central configuration: paths, forecasting constants and model parameters.

Everything that a pipeline step needs to agree on with another step lives here,
so that training and forecasting cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_SNAPSHOT_PATH = PROJECT_ROOT / "sample_data" / "velib_snapshot_2024-05-28.json"


@dataclass(frozen=True)
class Paths:
    """All on-disk locations used by the pipeline, derived from one data directory."""

    data_dir: Path

    @property
    def raw_dir(self) -> Path:
        """Snapshots written by `velib fetch` (one JSONL file per UTC day)."""
        return self.data_dir / "raw"

    @property
    def historical_dir(self) -> Path:
        """Pre-existing monthly JSON arrays (same record schema as the API)."""
        return self.data_dir / "historical_data_cleaned"

    @property
    def hourly_path(self) -> Path:
        return self.data_dir / "processed" / "hourly.pkl"

    @property
    def model_path(self) -> Path:
        return self.data_dir / "models" / "model.joblib"

    @property
    def metrics_path(self) -> Path:
        return self.data_dir / "models" / "metrics.json"

    @property
    def diagnostics_path(self) -> Path:
        """Holdout diagnostics read by the debugging dashboard (index.html)."""
        return self.data_dir / "models" / "diagnostics.json"

    @property
    def forecast_path(self) -> Path:
        """Next-week forecast read by the debugging dashboard (index.html)."""
        return self.data_dir / "forecast.json"

    @property
    def legacy_export_path(self) -> Path:
        """Frozen-schema output consumed by the external front end."""
        return self.data_dir / "4_compressed_predictions_final_fix.json"

    def input_dirs(self) -> list[Path]:
        return [self.historical_dir, self.raw_dir]


# ==============================================================================
# Data sources
# ==============================================================================

# Full export of every station in one response: the same data as the "Export"
# page's JSON download link (`/explore/dataset/.../download/?format=json`) that
# v1 used. The paginated `/records` API returns at most 100 stations per call,
# which is why it looked incomplete.
OPENDATA_URL = (
    "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/"
    "velib-disponibilite-en-temps-reel/exports/json"
)
GBFS_STATUS_URL = (
    "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/station_status.json"
)
GBFS_INFO_URL = (
    "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/"
    "station_information.json"
)
HTTP_TIMEOUT_SECONDS = 30
# The export is refreshed every few minutes. Four snapshots per hour give a
# good hourly mean (about 40 MB/day as JSONL; gzip finished days).
DEFAULT_FETCH_INTERVAL_MINUTES = 15

# Fields kept from each raw record. Anything else is dropped at load time.
RECORD_FIELDS = [
    "stationcode",
    "name",
    "is_installed",
    "is_renting",
    "is_returning",
    "capacity",
    "numbikesavailable",
    "numdocksavailable",
    "mechanical",
    "ebike",
    "duedate",
    "coordonnees_geo",
    "fetched_at",  # added by `velib fetch`; absent from older dumps
]

# A report this much older than the fetch that returned it comes from a station
# that has stopped reporting (the live feed still lists stations last seen in 2018).
MAX_REPORT_AGE_HOURS = 24

# The hourly grid starts on the first UTC day on which at least this share of
# the stations (relative to the best-covered day) reported. Without this, a few
# ancient reports in a dump without `fetched_at` stretch the grid over years.
MIN_DAILY_STATION_SHARE = 0.1

# ==============================================================================
# Time handling
# ==============================================================================

# Calendar features (hour, day of week) are computed in local time so that the
# morning peak stays at 08:00 across daylight-saving changes. The hourly grid
# itself is kept in UTC so that "one week ago" is always exactly 168 rows.
TIMEZONE = "Europe/Paris"

# ==============================================================================
# Forecasting problem
# ==============================================================================

# We forecast up to one week ahead. Every feature is built only from values at
# least HORIZON_HOURS old, so a single model is valid for any horizon 1..168h.
HORIZON_HOURS = 168
LAG_WEEKS = (1, 2, 3, 4)

# Neighbourhood used for the neighbour features.
NEIGHBOUR_COUNT = 5
NEIGHBOUR_MAX_RADIUS_M = 2000

# Hourly rows need at least this many observed hours in a rolling window
# before the window mean is trusted.
MIN_PERIODS_DAY = 6
MIN_PERIODS_WEEK = 24

# Stations not seen for this long before the forecast origin are treated as
# removed and left out of both forecast exports.
INACTIVE_STATION_DAYS = 3

# Time zone of the day/hour keys in the front-end export. v1 keyed it on UTC
# hours (so "8" meant 08:00 UTC = 10:00 in Paris in summer). v2 defaults to
# Paris local time; set this to "UTC" to reproduce v1 exactly.
LEGACY_EXPORT_TIMEZONE = TIMEZONE

# Stations with fewer observed hours than this are flagged as low confidence
# in the exported forecast.
LOW_CONFIDENCE_HISTORY_HOURS = 2 * HORIZON_HOURS

# ==============================================================================
# Model
# ==============================================================================

RANDOM_STATE = 42
DEFAULT_TEST_DAYS = 7
MODEL_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 400,
    "max_leaf_nodes": 63,
    "min_samples_leaf": 50,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": RANDOM_STATE,
}

# ==============================================================================
# Station data corrections
# ==============================================================================

# Known coordinate corrections for stations whose published location is wrong.
STATION_COORD_UPDATES = {
    "22504": {"lon": 2.253629, "lat": 48.905928},
    "25006": {"lon": 2.1961666225454, "lat": 48.862453313908},
    "10001": {"lon": 2.3600032, "lat": 48.8685433},
    "10001_relais": {"lon": 2.3599605, "lat": 48.8687079},
}

# Stations whose real capacity is about twice the published one (overflow
# parking). The map can optionally use the doubled capacity for them.
EXTRA_CAPACITY_STATIONS = frozenset(
    {
        "4005",
        "4104",
        "8002",
        "8004",
        "9104",
        "12105",
        "13123",
        "15056",
        "21302",
        "32012",
        "42004",
        "4010",
        "4017",
        "12010",
        "15058",
        "15122",
        "18043",
        "19018",
        "21021",
        "33019",
    }
)

# ==============================================================================
# Logging
# ==============================================================================

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
