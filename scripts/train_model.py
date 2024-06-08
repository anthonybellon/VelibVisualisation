import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm
import json
from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree
from collections import defaultdict

def get_absolute_path(relative_path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))

def calculate_nearby_station_status_adjustable(data, station, initial_radius=500, max_radius=2000, increment=500):
    station_data = data[data['stationcode'] == station]
    if station_data.empty:
        return None, station_data

    unique_coords = data[['lat', 'lon']].drop_duplicates().values
    station_codes = data[['stationcode']].drop_duplicates().values.flatten()
    kd_tree = KDTree(unique_coords)

    station_coords = station_data[['lat', 'lon']].iloc[0].values
    target_station_code = station_data['stationcode'].iloc[0]

    radius = initial_radius
    while (radius <= max_radius):
        indices = kd_tree.query_ball_point(station_coords, radius / 1000.0 / 111.32)
        out_of_bounds_indices = [i for i in indices if i >= len(station_codes)]
        
        if out_of_bounds_indices:
            print(f"Out of bounds indices for station {station}: {out_of_bounds_indices}")
            print(f"Max valid index: {len(station_codes) - 1}")
        
        indices = [i for i in indices if i < len(station_codes)]
        
        print(f"Valid indices for station {station}: {indices}")
        
        nearby_indices = [i for i in indices if station_codes[i] != target_station_code]

        nearby_stations = data.iloc[nearby_indices]
        nearby_stations = nearby_stations[nearby_stations['is_installed'] != 'NON']

        if len(nearby_stations) >= 5:
            return nearby_stations, station_data

        radius += increment

    return nearby_stations, station_data

print("Loading current bike data...")
current_data_path = get_absolute_path('../data/historical_data_cleaned/2021-04.json')
with open(current_data_path, 'r') as f:
    current_data = json.load(f)

current_bike_data = pd.DataFrame(current_data)

print("Preprocessing current bike data...")
current_bike_data['date'] = pd.to_datetime(current_bike_data['duedate'])

if current_bike_data['date'].dt.tz is None:
    current_bike_data['date'] = current_bike_data['date'].dt.tz_localize('UTC')

coords = pd.json_normalize(current_bike_data['coordonnees_geo'])
current_bike_data['lat'] = coords['lat']
current_bike_data['lon'] = coords['lon']

current_bike_data.fillna(0, inplace=True)

current_bike_data['hour'] = current_bike_data['date'].dt.hour
current_bike_data['day_of_week'] = current_bike_data['date'].dt.dayofweek

current_bike_data['hour_unscaled'] = current_bike_data['date'].dt.hour
current_bike_data['day_of_week_unscaled'] = current_bike_data['date'].dt.dayofweek

print("Adding lag features...")
current_bike_data = current_bike_data.sort_values(by=['stationcode', 'date'])
current_bike_data['lag_1_hour'] = current_bike_data.groupby('stationcode')['numbikesavailable'].shift(1)
current_bike_data['lag_1_day'] = current_bike_data.groupby('stationcode')['numbikesavailable'].shift(24)

print("Adding trend features...")
current_bike_data['rolling_mean_7_days'] = current_bike_data.groupby('stationcode')['numbikesavailable'].transform(lambda x: x.rolling(window=7*24, min_periods=1).mean())
current_bike_data['rolling_mean_30_days'] = current_bike_data.groupby('stationcode')['numbikesavailable'].transform(lambda x: x.rolling(window=30*24, min_periods=1).mean())

current_bike_data['normalized_bikes_available'] = current_bike_data.apply(lambda row: row['numbikesavailable'] / row['capacity'] if row['capacity'] != 0 else 0, axis=1)
current_bike_data['normalized_docks_available'] = current_bike_data.apply(lambda row: row['numdocksavailable'] / row['capacity'] if row['capacity'] != 0 else 0, axis=1)
current_bike_data['usage_ratio'] = current_bike_data.apply(lambda row: row['numbikesavailable'] / (row['capacity'] + 1e-5), axis=1)
current_bike_data['capacity_hour_interaction'] = current_bike_data['capacity'] * current_bike_data['hour']
current_bike_data['capacity_day_interaction'] = current_bike_data['capacity'] * current_bike_data['day_of_week']

print("Calculating nearby station status...")
current_bike_data['nearby_stations_closed'] = 0
current_bike_data['nearby_stations_full'] = 0
current_bike_data['nearby_stations_empty'] = 0
current_bike_data['likelihood_fill'] = 0.0
current_bike_data['likelihood_empty'] = 0.0

for station in tqdm(current_bike_data['stationcode'].unique(), desc="Calculating nearby station status"):
    nearby_stations, station_data = calculate_nearby_station_status_adjustable(current_bike_data, station)
    if nearby_stations is None:
        continue
    for index, row in station_data.iterrows():
        date_filtered_nearby_stations = nearby_stations[nearby_stations['date'] == row['date']]
        nearby_stations_closed = date_filtered_nearby_stations['is_installed'].apply(lambda x: 1 if x == "NON" else 0).sum()
        nearby_stations_full = (date_filtered_nearby_stations['numbikesavailable'] == 0).sum()
        nearby_stations_empty = (date_filtered_nearby_stations['numdocksavailable'] == 0).sum()

        current_bike_data.loc[index, 'nearby_stations_closed'] = nearby_stations_closed
        current_bike_data.loc[index, 'nearby_stations_full'] = nearby_stations_full
        current_bike_data.loc[index, 'nearby_stations_empty'] = nearby_stations_empty

        total_nearby_stations = len(date_filtered_nearby_stations)
        if total_nearby_stations > 0:
            likelihood_fill = nearby_stations_full / total_nearby_stations
            likelihood_empty = nearby_stations_empty / total_nearby_stations
        else:
            likelihood_fill = 0.0
            likelihood_empty = 0.0

        current_bike_data.loc[index, 'likelihood_fill'] = likelihood_fill
        current_bike_data.loc[index, 'likelihood_empty'] = likelihood_empty

        if nearby_stations_full > total_nearby_stations / 2:
            current_bike_data.loc[index, 'likelihood_fill'] *= 1.5
        if nearby_stations_empty > total_nearby_stations / 2:
            current_bike_data.loc[index, 'likelihood_empty'] *= 1.5

print("Loading scaler and models...")
scaler_path = get_absolute_path('../data/scaler_final_new.pkl')
model_path = get_absolute_path('../data/model_final_new.pkl')

with open(scaler_path, 'rb') as f:
    scalers = pickle.load(f)

with open(model_path, 'rb') as f:
    models = pickle.load(f)

current_bike_data['avg_bikes_hour_day'] = current_bike_data.groupby(['stationcode', 'hour', 'day_of_week'])['numbikesavailable'].transform('mean')

feature_names_file_path = get_absolute_path('../data/feature_names.json')
with open(feature_names_file_path, 'r') as f:
    feature_names = json.load(f)

current_bike_data[feature_names] = current_bike_data[feature_names].fillna(0)

current_bike_data.replace([np.inf, -np.inf], np.nan, inplace=True)
current_bike_data.fillna(0, inplace=True)

print("Making predictions and comparing with actual values...")
results = []

for station in tqdm(models.keys(), desc="Predicting for each station"):
    station_data = current_bike_data[current_bike_data['stationcode'] == station].copy()
    if station_data.empty:
        continue

    scaler = scalers.get(station)
    if not scaler:
        print(f"No scaler found for station {station}")
        continue

    station_data_features = station_data[feature_names]

    scaled_features = scaler.transform(station_data_features)
    station_data[feature_names] = scaled_features

    X = station_data[feature_names]
    y_true = station_data['numbikesavailable']
    model = models[station]
    y_pred = model.predict(X)
    station_data['predicted_bikesavailable'] = y_pred
    station_data['actual_bikesavailable'] = y_true
    results.append(station_data)

results_df = pd.concat(results)

results_df = results_df.drop(columns=['date', 'lat', 'lon'])

results_df['hour_unscaled'] = current_bike_data['hour_unscaled']
results_df['day_of_week_unscaled'] = current_bike_data['day_of_week_unscaled']

results_df = results_df[['stationcode', 'name', 'is_installed', 'capacity', 'numdocksavailable', 'numbikesavailable', 'mechanical', 'ebike', 'is_renting', 'is_returning', 'coordonnees_geo','predicted_bikesavailable', 'actual_bikesavailable', 'hour_unscaled', 'day_of_week_unscaled']]

results_file_path = get_absolute_path('../data/predictions_vs_actuals_final.pkl')
with open(results_file_path, 'wb') as f:
    pickle.dump(results_df, f)

print("Predictions and actual values saved to", results_file_path)
