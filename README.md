# VELIB Bike Station Prediction

This project predicts bike availability at VELIB stations around Paris using machine learning. It uses historical and current data to make predictions for each hour of the day, Monday through Sunday.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Training Models](#training-models)
  - [Making Predictions](#making-predictions)
  - [Data Pipeline](#data-pipeline)
- [Development](#development)
  - [Running Tests](#running-tests)
  - [Code Quality](#code-quality)
- [CLI Reference](#cli-reference)
- [Contributing](#contributing)
- [License](#license)

## Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd VelibVisualisation

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# OR
.\venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt

# Run tests to verify setup
pytest tests/

# Train models (example)
python scripts/train_model.py --help
```

## Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Setup

1. **Create a virtual environment** (recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # macOS/Linux
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up pre-commit hooks** (for development):

   ```bash
   pre-commit install
   ```

### Verify Installation

```bash
# Check Python version
python --version

# Run tests
pytest tests/ -v

# Check a script
python scripts/train_model.py --help
```

## Project Structure

```
VelibVisualisation/
├── scripts/                    # Main Python scripts
│   ├── config.py              # Centralized configuration
│   ├── feature_engineering.py # Shared feature engineering
│   ├── logging_config.py      # Logging configuration
│   ├── train_model.py         # Model training (CLI)
│   ├── predict_and_compare.py # Predictions (CLI)
│   ├── merge_models.py        # Combine batch models
│   ├── utils.py               # Utility functions
│   └── other_scripts/
│       ├── active/            # Active utility scripts
│       │   ├── fetch_data.py
│       │   └── data_preprocessing.py
│       └── archive/           # Archived/one-time scripts
├── tests/                     # Test suite
│   ├── conftest.py           # Test fixtures
│   ├── test_feature_engineering.py
│   ├── test_serialization.py
│   ├── test_config.py
│   └── test_smoke.py
├── data/                      # Data directory (not in git)
│   ├── historical_data_cleaned/
│   └── combined_models_and_scalers.pkl
├── notebooks/                 # Jupyter notebooks
├── pyproject.toml            # Project configuration
├── .pre-commit-config.yaml   # Pre-commit hooks
└── requirements.txt          # Dependencies
```

## Usage

### Training Models

Train prediction models for each station:

```bash
# Basic training
python scripts/train_model.py

# With options
python scripts/train_model.py \
    --data-dir data/historical_data_cleaned \
    --output-dir data \
    --batch-size 100 \
    --verbose

# Resume from last batch (if interrupted)
python scripts/train_model.py --resume

# Limit stations for testing
python scripts/train_model.py --station-limit 10 -v
```

After training, merge the batch models:

```bash
python scripts/merge_models.py
```

### Making Predictions

Generate predictions using trained models:

```bash
# Basic prediction
python scripts/predict_and_compare.py

# With options
python scripts/predict_and_compare.py \
    --input data/2_organized_predictions.json \
    --models data/combined_models_and_scalers.pkl \
    --output data/prediction_results.json \
    --verbose
```

### Data Pipeline

Full pipeline from data collection to predictions:

```bash
# 1. Fetch latest data
python scripts/other_scripts/active/fetch_data.py

# 2. Preprocess data
python scripts/other_scripts/active/data_preprocessing.py

# 3. Train models
python scripts/train_model.py -v

# 4. Merge models
python scripts/merge_models.py

# 5. Organize prediction data
python scripts/predict_and_compare_organize_bike_data.py

# 6. Generate predictions
python scripts/predict_and_compare.py -v

# 7. Compress output (optional)
python scripts/prediction_compression.py
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_feature_engineering.py

# Run with coverage
pytest tests/ --cov=scripts --cov-report=html
```

### Code Quality

```bash
# Run linter
ruff check scripts/

# Auto-fix issues
ruff check scripts/ --fix

# Format code
ruff format scripts/

# Run all pre-commit hooks
pre-commit run --all-files
```

### Type Checking (Optional)

```bash
# Install mypy
pip install mypy pandas-stubs

# Run type checker
mypy scripts/
```

## CLI Reference

### train_model.py

```
Usage: python scripts/train_model.py [OPTIONS]

Options:
  --data-dir PATH       Directory with cleaned JSON data
  --output-dir PATH     Directory to save models
  --batch-size INT      Number of stations per batch (default: 100)
  --station-limit INT   Limit stations to process (for testing)
  --resume              Resume from last completed batch
  -v, --verbose         Enable verbose logging
  -q, --quiet           Suppress info logging
  --help                Show help message
```

### predict_and_compare.py

```
Usage: python scripts/predict_and_compare.py [OPTIONS]

Options:
  --input PATH          Input data file (JSON)
  --models PATH         Combined models file (pickle)
  --output PATH         Output predictions file (JSON)
  -v, --verbose         Enable verbose logging
  -q, --quiet           Suppress info logging
  --help                Show help message
```

## Configuration

All paths and constants are centralized in `scripts/config.py`:

```python
from config import (
    DATA_DIR,
    HISTORICAL_DATA_DIR,
    COMBINED_MODELS_PATH,
    FEATURE_COLUMNS,
    BATCH_SIZE,
)
```

## File Descriptions

### Core Scripts

| Script                   | Description                           |
| ------------------------ | ------------------------------------- |
| `config.py`              | Centralized paths and constants       |
| `feature_engineering.py` | Shared feature engineering functions  |
| `train_model.py`         | Train station models with CLI         |
| `predict_and_compare.py` | Generate predictions with CLI         |
| `merge_models.py`        | Combine batch models into single file |
| `logging_config.py`      | Structured logging configuration      |

### Utility Scripts

| Script                  | Location                | Description               |
| ----------------------- | ----------------------- | ------------------------- |
| `fetch_data.py`         | `other_scripts/active/` | Fetch VELIB data from API |
| `data_preprocessing.py` | `other_scripts/active/` | Preprocess raw data       |

## Troubleshooting

### Common Issues

**pip command not found:**

```bash
# Use pip3 instead
pip3 install -r requirements.txt

# Or use python -m pip
python -m pip install -r requirements.txt
```

**Module not found errors:**

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

**Tests failing:**

```bash
# Check Python version (need 3.9+)
python --version

# Install test dependencies
pip install pytest pytest-cov
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Install pre-commit hooks (`pre-commit install`)
4. Make your changes
5. Run tests (`pytest tests/`)
6. Run linter (`ruff check scripts/`)
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## License

This project is licensed under the terms in the LICENSE file.
