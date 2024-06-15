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
current_data_path = get_absolute_path('../data/organized_predictions.json')
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

# Adding new features as in train_model.py
current_bike_data['normalized_bikes_available'] = current_bike_data['numbikesavailable'] / current_bike_data['capacity']
current_bike_data['normalized_docks_available'] = current_bike_data['numdocksavailable'] / current_bike_data['capacity']
current_bike_data['usage_ratio'] = current_bike_data['numbikesavailable'] / (current_bike_data['capacity'] + 1e-5)
current_bike_data['capacity_hour_interaction'] = current_bike_data['capacity'] * current_bike_data['hour']
current_bike_data['capacity_day_interaction'] = current_bike_data['capacity'] * current_bike_data['day_of_week']

# Function to calculate nearby station status with a limit
def calculate_nearby_station_status(data, radius=500):
    data = data.copy()
    stations = data['stationcode'].unique()

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

# Apply the function to the current bike data for specific range
print("Calculating nearby station status...")
current_bike_data = calculate_nearby_station_status(current_bike_data)


combined_file_path = get_absolute_path('../data/combined_models_and_scalers.pkl')

with open(combined_file_path, 'rb') as f:
    combined_data = pickle.load(f)
    combined_models = combined_data['models']
    combined_scalers = combined_data['scalers']

# Verify if scaler for batch 15 is present
if 15 in combined_scalers:
    print("Scaler for batch 15 is present.")
else:
    print("Scaler for batch 15 is missing.")

# Optionally, verify the stations in batch 15
stations_in_batch_15 = [station for station, details in combined_models.items() if details['scaler_idx'] == 15]
print(f"Stations in batch 15: {stations_in_batch_15}")


with open(combined_file_path, 'rb') as f:
    combined_data = pickle.load(f)
    models = combined_data['models']
    scalers = combined_data['scalers']

# Feature engineering: Add 'avg_bikes_hour_day'
current_bike_data['avg_bikes_hour_day'] = current_bike_data.groupby(['stationcode', 'hour', 'day_of_week'])['numbikesavailable'].transform('mean')

# Select features, including the unscaled versions for readability
base_features = ['hour', 'day_of_week', 'avg_bikes_hour_day', 'lag_1_hour', 'lag_1_day', 
                 'rolling_mean_7_days', 'rolling_mean_30_days', 'normalized_bikes_available', 
                 'normalized_docks_available', 'usage_ratio', 'capacity_hour_interaction', 'capacity_day_interaction']
additional_features = ['nearby_stations_closed', 'nearby_stations_full', 'nearby_stations_empty', 
                       'likelihood_fill', 'likelihood_empty']
all_features = base_features + additional_features

# Process a specific range of stations (835th to 844th)
stations_to_test = list(models.keys())

# Make predictions
print("Making predictions and comparing with actual values...")
results = []

# Debugging: Print the station codes in the model and current bike data
print("Station codes in models:", stations_to_test)
print("Station codes in current data:", current_bike_data['stationcode'].unique())

# Check for missing models and scalers
missing_stations = []
for station in stations_to_test:
    if station not in models:
        missing_stations.append(station)
        print(f"Missing model for station {station}. Station data:")
        print(current_bike_data[current_bike_data['stationcode'] == station])
    else:
        scaler_idx = models[station].get('scaler_idx')
        if scaler_idx not in scalers:
            print(f"Missing scaler for station {station} with scaler index {scaler_idx}.")

print(f"Total missing stations: {len(missing_stations)}")

for station in tqdm(stations_to_test, desc="Predicting for each station"):
    if station not in models:
        continue
    
    station_data = current_bike_data[current_bike_data['stationcode'] == station].copy()  # Ensure it's a copy
    if station_data.empty:
        print(f"No data for station {station}")
        continue
    
    # Get the model and scaler for the current station
    model = models[station]['model']
    scaler_idx = models[station]['scaler_idx']
    
    if scaler_idx not in scalers:
        print(f"Missing scaler for station {station} with scaler index {scaler_idx}.")
        continue
    
    scaler = scalers[scaler_idx]

    # Determine which features were used during training for this station
    trained_features = model.feature_names_in_  # Get the feature names used during training
    
    # Ensure that the prediction DataFrame has the same features as the training DataFrame
    for feature in trained_features:
        if feature not in station_data.columns:
            station_data[feature] = 0

    X = station_data[trained_features]
    y_true = station_data['numbikesavailable']

    # Normalize the features
    X_scaled = pd.DataFrame(scaler.transform(X), columns=trained_features)
    
    # Predict
    y_pred = model.predict(X_scaled)
    
    station_data.loc[:, 'predicted_bikesavailable'] = y_pred
    station_data.loc[:, 'actual_bikesavailable'] = y_true
    results.append(station_data)

# Ensure results is not empty before concatenating
if results:
    # Concatenate results
    results_df = pd.concat(results)
else:
    raise ValueError("No predictions were made, please check your data and models.")

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
