import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm
import json
from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree
from collections import defaultdict

# Function to get the absolute path relative to the script location
def get_absolute_path(relative_path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))

# Load the current data
print("Loading current bike data...")
current_data_path = get_absolute_path('../data/historical_data_cleaned/2021-04.json')
with open(current_data_path, 'r') as f:
    current_data = json.load(f)

# Load the additional weekly data
weekly_data_path = get_absolute_path('../data/historical_data_cleaned/weekly_velib_data.json')
with open(weekly_data_path, 'r') as f:
    weekly_data = json.load(f)

# Combine the datasets
current_data.extend(weekly_data)
current_bike_data = pd.DataFrame(current_data)

# Preprocess current data
print("Preprocessing current bike data...")
current_bike_data['date'] = pd.to_datetime(current_bike_data['duedate'])

# Ensure 'date' column is timezone-aware (localize to UTC if needed)
if current_bike_data['date'].dt.tz is None:
    current_bike_data['date'] = current_bike_data['date'].dt.tz_localize('UTC')

# Extract latitude and longitude
coords = pd.json_normalize(current_bike_data['coordonnees_geo'])
current_bike_data['lat'] = coords['lat']
current_bike_data['lon'] = coords['lon']

# Remove any NaN values
current_bike_data.fillna(0, inplace=True)

# Feature engineering: Add hour and day_of_week
current_bike_data['hour'] = current_bike_data['date'].dt.hour
current_bike_data['day_of_week'] = current_bike_data['date'].dt.dayofweek

# Adding human-readable hour and day_of_week columns
current_bike_data['hour_unscaled'] = current_bike_data['date'].dt.hour
current_bike_data['day_of_week_unscaled'] = current_bike_data['date'].dt.dayofweek

# Adding lag features
print("Adding lag features...")
current_bike_data = current_bike_data.sort_values(by=['stationcode', 'date'])
current_bike_data['lag_1_hour'] = current_bike_data.groupby('stationcode')['numbikesavailable'].shift(1)
current_bike_data['lag_1_day'] = current_bike_data.groupby('stationcode')['numbikesavailable'].shift(24)

# Adding trend features
print("Adding trend features...")
current_bike_data['rolling_mean_7_days'] = current_bike_data.groupby('stationcode')['numbikesavailable'].transform(lambda x: x.rolling(window=7*24, min_periods=1).mean())
current_bike_data['rolling_mean_30_days'] = current_bike_data.groupby('stationcode')['numbikesavailable'].transform(lambda x: x.rolling(window=30*24, min_periods=1).mean())

# Add Capacity-Based Features
# Ensure capacity is not zero to avoid division by zero
current_bike_data['normalized_bikes_available'] = current_bike_data.apply(lambda row: row['numbikesavailable'] / row['capacity'] if row['capacity'] != 0 else 0, axis=1)
current_bike_data['normalized_docks_available'] = current_bike_data.apply(lambda row: row['numdocksavailable'] / row['capacity'] if row['capacity'] != 0 else 0, axis=1)
current_bike_data['usage_ratio'] = current_bike_data.apply(lambda row: row['numbikesavailable'] / (row['capacity'] + 1e-5), axis=1)
current_bike_data['capacity_hour_interaction'] = current_bike_data['capacity'] * current_bike_data['hour']
current_bike_data['capacity_day_interaction'] = current_bike_data['capacity'] * current_bike_data['day_of_week']

# Function to calculate nearby station status with adjustable radius
def calculate_nearby_station_status(data, initial_radius=500, max_radius=2000, increment=500, limit=2):
    data = data.copy()
    stations = data['stationcode'].unique()[:limit]

    coords = data[['lat', 'lon']].drop_duplicates().values
    kd_tree = KDTree(coords)

    data['nearby_stations_closed'] = 0
    data['nearby_stations_full'] = 0
    data['nearby_stations_empty'] = 0
    data['likelihood_fill'] = 0.0
    data['likelihood_empty'] = 0.0

    for station in tqdm(stations, desc="Calculating nearby station status"):
        station_data = data[data['stationcode'] == station]
        if station_data.empty:
            continue
        station_coords = station_data[['lat', 'lon']].iloc[0].values

        radius = initial_radius
        while radius <= max_radius:
            indices = kd_tree.query_ball_point(station_coords, radius / 1000.0)
            nearby_stations = data.iloc[indices]

            # Exclude stations that are open but have no capacity
            nearby_stations = nearby_stations[(nearby_stations['is_renting'] == "OUI") & (nearby_stations['capacity'] > 0)]

            if len(nearby_stations) >= 5:
                break

            radius += increment

        for index, row in station_data.iterrows():
            date_filtered_nearby_stations = nearby_stations[nearby_stations['date'] == row['date']]
            nearby_stations_closed = date_filtered_nearby_stations['is_installed'].apply(lambda x: 1 if x == "NON" else 0).sum()
            nearby_stations_full = (date_filtered_nearby_stations['numbikesavailable'] == 0).sum()
            nearby_stations_empty = (date_filtered_nearby_stations['numdocksavailable'] == 0).sum()

            data.loc[index, 'nearby_stations_closed'] = nearby_stations_closed
            data.loc[index, 'nearby_stations_full'] = nearby_stations_full
            data.loc[index, 'nearby_stations_empty'] = nearby_stations_empty

            total_nearby_stations = len(date_filtered_nearby_stations)
            if total_nearby_stations > 0:
                likelihood_fill = nearby_stations_full / total_nearby_stations
                likelihood_empty = nearby_stations_empty / total_nearby_stations
            else:
                likelihood_fill = 0.0
                likelihood_empty = 0.0

            data.loc[index, 'likelihood_fill'] = likelihood_fill
            data.loc[index, 'likelihood_empty'] = likelihood_empty

            # Adjust likelihood based on nearby station statuses
            if nearby_stations_full > total_nearby_stations / 2:
                data.loc[index, 'likelihood_fill'] *= 1.5  # Increase likelihood of filling up
            if nearby_stations_empty > total_nearby_stations / 2:
                data.loc[index, 'likelihood_empty'] *= 1.5  # Increase likelihood of emptying

    return data

# Apply the function to the current bike data
print("Calculating nearby station status...")
current_bike_data = calculate_nearby_station_status(current_bike_data, limit=2)  # You can adjust the limit as needed

# Load the scaler and models
print("Loading scaler and models...")
scaler_path = get_absolute_path('../data/scaler_final_new.pkl')
model_path = get_absolute_path('../data/model_final_new.pkl')

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

with open(model_path, 'rb') as f:
    models = pickle.load(f)

# Feature engineering: Add 'avg_bikes_hour_day'
current_bike_data['avg_bikes_hour_day'] = current_bike_data.groupby(['stationcode', 'hour', 'day_of_week'])['numbikesavailable'].transform('mean')

# Load feature names
feature_names_file_path = get_absolute_path('../data/feature_names.json')
with open(feature_names_file_path, 'r') as f:
    feature_names = json.load(f)

# Ensure the current data has the same features
current_bike_data[feature_names] = current_bike_data[feature_names].fillna(0)  # Fill any NaN values in features with 0

# Check for infinite or extremely large values and replace them
current_bike_data.replace([np.inf, -np.inf], np.nan, inplace=True)
current_bike_data.fillna(0, inplace=True)

# Normalize features (excluding the unscaled columns)
print("Normalizing features...")
scaled_features = scaler.transform(current_bike_data[feature_names])
current_bike_data[feature_names] = scaled_features

# Make predictions
print("Making predictions and comparing with actual values...")
results = []

for station in tqdm(models.keys(), desc="Predicting for each station"):
    station_data = current_bike_data[current_bike_data['stationcode'] == station].copy()  # Ensure it's a copy
    if station_data.empty:
        continue
    X = station_data[feature_names]
    y_true = station_data['numbikesavailable']
    model = models[station]
    y_pred = model.predict(X)
    station_data.loc[:, 'predicted_bikesavailable'] = y_pred
    station_data.loc[:, 'actual_bikesavailable'] = y_true
    results.append(station_data)

# Concatenate results
results_df = pd.concat(results)

# Drop unnecessary columns to avoid duplication
results_df = results_df.drop(columns=['date', 'lat', 'lon'])

# Add unscaled values to the final JSON output
results_df['hour_unscaled'] = current_bike_data['hour_unscaled']
results_df['day_of_week_unscaled'] = current_bike_data['day_of_week_unscaled']

# Filter to include only necessary columns
results_df = results_df[['stationcode', 'name', 'is_installed', 'capacity', 'numdocksavailable', 'numbikesavailable', 'mechanical', 'ebike', 'is_renting', 'is_returning', 'coordonnees_geo','predicted_bikesavailable', 'actual_bikesavailable', 'hour_unscaled', 'day_of_week_unscaled']]

# Convert to JSON format
results_json = results_df.to_dict(orient='records')

# Save to a JSON file
results_json_path = get_absolute_path('../data/prediction_results_final.json')
with open(results_json_path, 'w') as f:
    json.dump(results_json, f, indent=4)

print(f"Predictions and comparisons saved to {results_json_path}")

# Organize predictions by station, day, and hour
print("Organizing predictions by station, day, and hour...")
organized_predictions = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for station in results_df['stationcode'].unique():
    station_data = results_df[results_df['stationcode'] == station]
    for index, row in station_data.iterrows():
        day = row['day_of_week_unscaled']
        hour = row['hour_unscaled']
        organized_predictions[station][day][hour].append(row['predicted_bikesavailable'])

# Convert to a normal dictionary before saving
organized_predictions = {k: dict(v) for k, v in organized_predictions.items()}
for k, v in organized_predictions.items():
    organized_predictions[k] = {kk: dict(vv) for kk, vv in v.items()}

# Save organized predictions to a JSON file
organized_predictions_path = get_absolute_path('../data/compressed_predictions_final.json')
with open(organized_predictions_path, 'w') as f:
    json.dump(organized_predictions, f, indent=4)

print(f"Organized predictions saved to {organized_predictions_path}")
