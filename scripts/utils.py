import os
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm
import json

def get_absolute_path(relative_path):
    """Get the absolute path from a relative path."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))

def calculate_nearby_station_status_adjustable(data, station, initial_radius=500, max_radius=2000, increment=500):
    """Calculate the status of nearby stations within a certain radius."""
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

def report_infinite_and_large_values(data):
    """Report infinite and large values in the dataframe."""
    numeric_data = data.select_dtypes(include=[np.number])
    infinite_values = numeric_data.isin([np.inf, -np.inf]).any()
    if infinite_values.any():
        print("Infinite values found in columns:", numeric_data.columns[infinite_values].tolist())

    large_values = (numeric_data.abs() > 1e10).any()
    if large_values.any():
        print("Large values found in columns:", numeric_data.columns[large_values].tolist())

    nan_values = numeric_data.isna().any()
    if nan_values.any():
        print("NaN values found in columns:", numeric_data.columns[nan_values].tolist())

def create_features(data):
    """Create and add new features to the dataset."""
    print("Creating features...")
    data = data.copy()
    data['hour'] = data['date'].dt.hour
    data['day_of_week'] = data['date'].dt.dayofweek

    data = data.sort_values(by=['stationcode', 'date'])
    data['lag_1_hour'] = data.groupby('stationcode')['numbikesavailable'].shift(1)
    data['lag_1_day'] = data.groupby('stationcode')['numbikesavailable'].shift(24)

    data['rolling_mean_7_days'] = data.groupby('stationcode')['numbikesavailable'].transform(lambda x: x.rolling(window=7*24, min_periods=1).mean())
    data['rolling_mean_30_days'] = data.groupby('stationcode')['numbikesavailable'].transform(lambda x: x.rolling(window=30*24, min_periods=1).mean())

    data['normalized_bikes_available'] = data['numbikesavailable'] / (data['capacity'] + 1e-5)
    data['normalized_docks_available'] = data['numdocksavailable'] / (data['capacity'] + 1e-5)
    data['usage_ratio'] = data['numbikesavailable'] / (data['capacity'] + 1e-5)
    data['capacity_hour_interaction'] = data['capacity'] * data['hour']
    data['capacity_day_interaction'] = data['capacity'] * data['day_of_week']

    data['avg_bikes_hour_day'] = data.groupby(['stationcode', 'hour', 'day_of_week'])['numbikesavailable'].transform('mean')

    # Create indicator variables for missing values
    data['lag_1_hour_missing'] = data['lag_1_hour'].isna().astype(int)
    data['lag_1_day_missing'] = data['lag_1_day'].isna().astype(int)

    # Fill NaN values with the mean of the column
    data['lag_1_hour'] = data['lag_1_hour'].fillna(data['lag_1_hour'].mean())
    data['lag_1_day'] = data['lag_1_day'].fillna(data['lag_1_day'].mean())
    
    report_infinite_and_large_values(data)
    return data

def save_json(data, file_path):
    """Save data to a JSON file."""
    data.to_json(file_path, orient='records', lines=True)
    print(f"Data saved to {file_path}")

def load_json(file_path):
    """Load data from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_feature_names(feature_names):
    """Save feature names to a JSON file."""
    feature_name_path = get_absolute_path('../data/feature_names.json')
    with open(feature_name_path, 'w') as file:
        json.dump(feature_names, file)
    print(f"Feature names saved to {feature_name_path}")

def load_feature_names():
    """Load feature names from a JSON file."""
    feature_name_path = get_absolute_path('../data/feature_names.json')
    with open(feature_name_path, 'r') as file:
        feature_names = json.load(file)
    return feature_names
