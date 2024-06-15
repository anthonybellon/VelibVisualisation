# VELIB Bike Station Prediction

This project focuses on predicting bike availability at VELIB stations around Paris. It uses historical and current data to make accurate predictions for each hour of the day from Monday to Sunday.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Fetch Data](#fetch-data)
  - [Train Model](#train-model)
  - [Merge Models](#merge-models)
  - [Predict and Compare](#predict-and-compare)
  - [Prediction Compression](#prediction-compression)
- [File Descriptions](#file-descriptions)
- [Help](#help)
- [Contributing](#contributing)
- [License](#license)

## Installation

Ensure you have Python installed. You can install the necessary packages using:

```bash
pip install -r requirements.txt
```

## Usage

### Fetch Data

To start with, run the `fetch_data.py` script. This script will fetch and organize the data into a proper structure, saving it into a file named `weekly_vidib_data.json`.

```bash
python scripts/fetch_data.py
```

Make sure to then delete any duplicates with the `delete_duplicates.py` script.

```bash
python scripts/other_scripts/delete_duplicates.py
```

### Train Model

Next, run the `train_model.py` script. This script uses a file called `historical_data_cleaned.json` that contains all previously stored data in JSON format. It concatenates this data for training.

The script produces models and scalers in batches of 100 stations. The train model also gives you a `feature_name.json` that you can use to ensure the `predict_and_compare.py` script uses the same features.

```bash
python scripts/train_model.py
```

### Merge Models

Use the `merge_models.py` script to combine the batch models into a single `.pkl` file.

```bash
python scripts/merge_models.py
```

### Predict and Compare

Then, use the `predict_and_compare_organize_bike_data.py` script. It uses a file called `1_use_for_predictions.json` with the most up-to-date data for each hour of the day. The script generates `2_organized_predictions.json`, which contains organized predictions for every hour of the day.

```bash
python scripts/predict_and_compare_organize_bike_data.py
```

Afterwards, use `predict_and_compare.py` to create a file called `3_predictions_results.json` that will add the predicted amount of bikes to each hour of each day of the week.

```bash
python scripts/predict_and_compare.py
```

### Prediction Compression

The file size you get is huge, so to help with that, use `prediction_compression.py`. It takes your `3_predictions_results.json` and compresses it to a smaller file size (just under 2 MB as of June 15th, 2024, version 1.0).

```bash
python scripts/prediction_compression.py
```

And voila! You can apply this to any website, app, or program of your choosing.

## File Descriptions

- **fetch_data.py**: Fetches and organizes VELIB data, saving it to `weekly_vidib_data.json`.
- **delete_duplicates.py**: Deletes duplicate entries from the data.
- **train_model.py**: Trains models using `historical_data_cleaned.json` and outputs models and scalers in batches. It also produces `feature_name.json`.
- **merge_models.py**: Merges batch models into a single `.pkl` file.
- **predict_and_compare_organize_bike_data.py**: Uses up-to-date data to create predictions and saves them in `2_organized_predictions.json`.
- **predict_and_compare.py**: Adds predicted bike availability for each hour of each day of the week, saving results in `3_predictions_results.json`.
- **prediction_compression.py**: Compresses the prediction results to a smaller file size.

## Help

Additional scripts are available in the `other_scripts` directory to assist with various tasks:

- **combine_data.py**: Use this script if you need to concatenate multiple `.json` files.

```bash
python scripts/other_scripts/combine_data.py
```

- **month_splitter_csv.py**: This script is useful if you have old bike data from GitHub and need to make the files more manageable. It converts the data into `.json` files one month at a time.

```bash
python scripts/other_scripts/month_splitter_csv.py
```
