import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm
import json
from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree

# Function to get the absolute path relative to the script location
def get_absolute_path(relative_path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))

# Load the current data
print("Loading current bike data...")
current_data_path = get_absolute_path('../data/velib_data.json')
with open(current_data_path, 'r') as f:
    current_data = json.load(f)

# Convert to DataFrame
current_bike_data = pd.DataFrame(current_data)

# Preprocess current data
print("Preprocessing current bike data...")
current_bike_data['date'] = pd.to_datetime(current_bike_data['duedate'])

# Ensure 'date' column is timezone-aware (localize to UTC if needed)
if current_bike_data['date'].dt.tz is None:
    current_bike_data['date'] = current_bike_data['date'].dt.tz_localize('UTC')

# Extract latitude and longitude
latitudes = []
longitudes = []
for coord in tqdm(current_bike_data['coordonnees_geo'], desc="Extracting coordinates"):
    latitudes.append(coord['lat'])
    longitudes.append(coord['lon'])

current_bike_data['lat'] = latitudes
current_bike_data['lon'] = longitudes

# Remove any NaN values
current_bike_data.fillna(0, inplace=True)

# Feature engineering: Add hour and day_of_week
current_bike_data['hour'] = current_bike_data['date'].dt.hour
current_bike_data['day_of_week'] = current_bike_data['date'].dt.dayofweek

# Adding human-readable hour and day_of_week columns
current_bike_data['hour_unscaled'] = current_bike_data['date'].dt.hour
current_bike_data['day_of_week_unscaled'] = current_bike_data['date'].dt.dayofweek

# Function to calculate nearby station status
def calculate_nearby_station_status(data, radius=500, limit=5):  # Limit to the first 5 stations
    data = data.copy()
    stations = data['stationcode'].unique()[:limit]  # Limit to the first 'limit' stations

    coords = data[['lat', 'lon']].drop_duplicates().values
    kd_tree = KDTree(coords)

    data['nearby_stations_closed'] = 0
    data['nearby_stations_full'] = 0
    data['nearby_stations_empty'] = 0
    data['likelihood_fill'] = 0.0
    data['likelihood_empty'] = 0.0

    for station in tqdm(stations, desc="Calculating nearby station status"):
        station_data = data[data['stationcode'] == station]
        station_coords = station_data[['lat', 'lon']].iloc[0].values

        indices = kd_tree.query_ball_point(station_coords, radius / 1000.0)
        nearby_stations = data.iloc[indices]

        for index, row in station_data.iterrows():
            date_filtered_nearby_stations = nearby_stations[nearby_stations['date'] == row['date']]
            nearby_stations_closed = date_filtered_nearby_stations['is_installed'].apply(lambda x: 1 if x == "NON" else 0).sum()
            nearby_stations_full = (date_filtered_nearby_stations['numbikesavailable'] == date_filtered_nearby_stations['capacity']).sum()
            nearby_stations_empty = (date_filtered_nearby_stations['numbikesavailable'] == 0).sum()

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
current_bike_data = calculate_nearby_station_status(current_bike_data, limit=5)

# Load the scaler and models
print("Loading scaler and models...")
scaler_path = get_absolute_path('../data/scaler.pkl')
model_path = get_absolute_path('../data/model_test.pkl')

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

with open(model_path, 'rb') as f:
    models = pickle.load(f)

# Feature engineering: Add 'avg_bikes_hour_day'
current_bike_data['avg_bikes_hour_day'] = current_bike_data.groupby(['stationcode', 'hour', 'day_of_week'])['numbikesavailable'].transform('mean')

# Select features, including the unscaled versions for readability
features = ['hour', 'day_of_week', 'nearby_stations_closed', 'nearby_stations_full', 'nearby_stations_empty', 
            'likelihood_fill', 'likelihood_empty', 'avg_bikes_hour_day']

# Ensure the current data has the same features
current_bike_data[features] = current_bike_data[features].fillna(0)  # Fill any NaN values in features with 0

# Normalize features (excluding the unscaled columns)
print("Normalizing features...")
scaled_features = scaler.transform(current_bike_data[features])
current_bike_data[features] = scaled_features

# Make predictions
print("Making predictions and comparing with actual values...")
results = []

for station in tqdm(models.keys(), desc="Predicting for each station"):
    station_data = current_bike_data[current_bike_data['stationcode'] == station]
    if station_data.empty:
        continue
    X = station_data[features]
    y_true = station_data['numbikesavailable']
    model = models[station]
    y_pred = model.predict(X)
    station_data.loc[:, 'predicted_bikesavailable'] = y_pred
    station_data.loc[:, 'actual_bikesavailable'] = y_true
    results.append(station_data)

# Concatenate results
results_df = pd.concat(results)

# Convert 'date' column to string for JSON serialization
results_df['date'] = results_df['date'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

# Drop unnecessary columns to avoid duplication
results_df = results_df.drop(columns=['date', 'lat', 'lon'])

# Add unscaled values to the final JSON output
results_df['hour_unscaled'] = current_bike_data['hour_unscaled']
results_df['day_of_week_unscaled'] = current_bike_data['day_of_week_unscaled']

# Convert to JSON format
results_json = results_df.to_dict(orient='records')

# Save to a JSON file
results_json_path = get_absolute_path('../data/prediction_results.json')
with open(results_json_path, 'w') as f:
    json.dump(results_json, f, indent=4)

print(f"Predictions and comparisons saved to {results_json_path}")
