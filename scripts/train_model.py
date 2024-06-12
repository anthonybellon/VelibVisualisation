import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle
import os
from tqdm import tqdm
from geopy.distance import geodesic
import numpy as np
from scipy.spatial import KDTree
import json
from scipy.stats import randint

# Function to get the absolute path relative to the script location
def get_absolute_path(relative_path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))

# Directory containing the cleaned JSON files
clean_data_directory = get_absolute_path('../data/historical_data_cleaned')
model_save_directory = get_absolute_path('../data')

# Load and concatenate JSON files
print("Loading cleaned bike data...")
all_data = []
json_files = [f for f in os.listdir(clean_data_directory) if f.endswith('.json')]
for filename in tqdm(json_files, desc="Loading JSON files"):
    file_path = os.path.join(clean_data_directory, filename)
    with open(file_path, 'r') as f:
        data = json.load(f)
        all_data.extend(data)

# Convert to DataFrame
bike_data = pd.DataFrame(all_data)

# Check if the dataframe is empty
if bike_data.empty:
    raise ValueError("The concatenated bike data is empty. Please check the JSON files and their loading process.")

# Preprocess bike data
print("Preprocessing bike data...")

# Convert date chunks
bike_data['date'] = pd.to_datetime(bike_data['duedate'])
if bike_data['date'].dt.tz is None:
    bike_data['date'] = bike_data['date'].dt.tz_localize('UTC')

# Update station coordinates
station_coord_updates = {
    "22504": {"lon": 2.253629, "lat": 48.905928},
    "25006": {"lon": 2.1961666225454, "lat": 48.862453313908},
    "10001": {"lon": 2.3600032, "lat": 48.8685433},
    "10001_relais": {"lon": 2.3599605, "lat": 48.8687079}
}

# Apply coordinate updates
for station, coords in station_coord_updates.items():
    bike_data.loc[bike_data['stationcode'] == station, 'coordonnees_geo'] = bike_data.loc[bike_data['stationcode'] == station, 'coordonnees_geo'].apply(lambda x: coords if isinstance(x, dict) else x)

# Extract latitude and longitude
print("Extracting latitude and longitude...")
bike_data['lat'] = bike_data['coordonnees_geo'].apply(lambda x: x.get('lat', np.nan) if isinstance(x, dict) else np.nan)
bike_data['lon'] = bike_data['coordonnees_geo'].apply(lambda x: x.get('lon', np.nan) if isinstance(x, dict) else np.nan)

# Handle missing values by filling or dropping
bike_data.dropna(subset=['lat', 'lon'], inplace=True)

# Remove any NaN values
print("Removing NaN values...")
bike_data.fillna(0, inplace=True)

# Feature engineering: Add hour and day_of_week
print("Adding hour and day of week...")
bike_data['hour'] = bike_data['date'].dt.hour
bike_data['day_of_week'] = bike_data['date'].dt.dayofweek

# Adding lag features
print("Adding lag features...")
bike_data = bike_data.sort_values(by=['stationcode', 'date'])
bike_data['lag_1_hour'] = bike_data.groupby('stationcode')['numbikesavailable'].shift(1)
bike_data['lag_1_day'] = bike_data.groupby('stationcode')['numbikesavailable'].shift(24)

# Adding trend features
print("Adding trend features...")
bike_data['rolling_mean_7_days'] = bike_data.groupby('stationcode')['numbikesavailable'].transform(lambda x: x.rolling(window=7*24, min_periods=1).mean())
bike_data['rolling_mean_30_days'] = bike_data.groupby('stationcode')['numbikesavailable'].transform(lambda x: x.rolling(window=30*24, min_periods=1).mean())

# Add Capacity-Based Features
bike_data['normalized_bikes_available'] = bike_data['numbikesavailable'] / bike_data['capacity']
bike_data['normalized_docks_available'] = bike_data['numdocksavailable'] / bike_data['capacity']
bike_data['usage_ratio'] = bike_data['numbikesavailable'] / (bike_data['capacity'] + 1e-5)
bike_data['capacity_hour_interaction'] = bike_data['capacity'] * bike_data['hour']
bike_data['capacity_day_interaction'] = bike_data['capacity'] * bike_data['day_of_week']

# Function to handle infinite or excessively large values
def handle_large_values(df, columns):
    problematic_stations = []
    for col in columns:
        # Replace infinities with NaN
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        # Round to 6 decimal places
        df[col] = df[col].round(6)
        
        # Identify rows that still contain NaN after rounding
        problematic = df[col].isna()
        
        if problematic.any():
            problematic_stations.extend(df.loc[problematic, 'stationcode'].unique())
            
            # Fill remaining NaN with 0
            df[col] = df[col].fillna(0)
    
    return df, problematic_stations

# Handle and round large values
numeric_columns = ['normalized_bikes_available', 'normalized_docks_available', 'usage_ratio',
                   'capacity_hour_interaction', 'capacity_day_interaction',
                   'rolling_mean_7_days', 'rolling_mean_30_days', 'lag_1_hour', 'lag_1_day']

bike_data, problematic_stations = handle_large_values(bike_data, numeric_columns)
if problematic_stations:
    print(f"Stations with problematic values after handling: {problematic_stations}")

print("Preprocessing complete.")

# Function to convert degrees to radians for KDTree
def deg_to_rad(coords):
    return np.radians(coords)

# Function to calculate nearby station status with adjustable radius
def calculate_nearby_station_status_adjustable(data, station, initial_radius=500, max_radius=2000, increment=500):
    station_data = data[data['stationcode'] == station]
    if station_data.empty:
        return None, station_data

    # Extract unique coordinates and station codes
    unique_coords = data[['lat', 'lon']].drop_duplicates().values
    station_codes = data[['stationcode']].drop_duplicates().values.flatten()
    kd_tree = KDTree(unique_coords)
    
    # Get the coordinates of the target station
    station_coords = station_data[['lat', 'lon']].iloc[0].values
    target_station_code = station_data['stationcode'].iloc[0]

    radius = initial_radius
    while (radius <= max_radius):
        # Query the KDTree with radius in degrees converted to approximate meters
        indices = kd_tree.query_ball_point(station_coords, radius / 1000.0 / 111.32)  # Approx conversion from meters to degrees
        
        # Identify and print out-of-bounds indices
        out_of_bounds_indices = [i for i in indices if i >= len(station_codes)]
        if out_of_bounds_indices:
            print(f"Out of bounds indices for station {station}: {out_of_bounds_indices}")
            print(f"Max valid index: {len(station_codes) - 1}")
        
        # Filter out-of-bounds indices
        indices = [i for i in indices if i < len(station_codes)]
        
        # Filter out the target station itself
        nearby_indices = [i for i in indices if station_codes[i] != target_station_code]
        
        nearby_stations = data.iloc[nearby_indices]
        
        # Filter out stations that are not installed
        nearby_stations = nearby_stations[nearby_stations['is_installed'] != 'NON']
        
        if len(nearby_stations) >= 5:
            return nearby_stations, station_data
        
        radius += increment
    
    return nearby_stations, station_data

# Apply the function to the bike data
print("Calculating nearby station status...")

# Optional limit for the number of stations to process
station_limit = None

# Filter stations if a limit is specified
stations = bike_data['stationcode'].unique()[:station_limit] if station_limit is not None else bike_data['stationcode'].unique()

# Initialize the columns for nearby station statuses
bike_data['nearby_stations_closed'] = 0
bike_data['nearby_stations_full'] = 0
bike_data['nearby_stations_empty'] = 0
bike_data['likelihood_fill'] = 0.0
bike_data['likelihood_empty'] = 0.0

# Feature engineering: Add 'avg_bikes_hour_day'
bike_data['avg_bikes_hour_day'] = bike_data.groupby(['stationcode', 'hour', 'day_of_week'])['numbikesavailable'].transform('mean')

# Select features and target
base_features = ['hour', 'day_of_week', 'avg_bikes_hour_day', 'lag_1_hour', 'lag_1_day', 
                 'rolling_mean_7_days', 'rolling_mean_30_days', 'normalized_bikes_available', 
                 'normalized_docks_available', 'usage_ratio', 'capacity_hour_interaction', 'capacity_day_interaction']
additional_features = ['nearby_stations_closed', 'nearby_stations_full', 'nearby_stations_empty', 
                       'likelihood_fill', 'likelihood_empty']
target = 'numbikesavailable'

# Save feature names
feature_names_file_path = get_absolute_path('../data/feature_names.json')
with open(feature_names_file_path, 'w') as f:
    json.dump(base_features + additional_features, f)
print(f"Feature names saved to {feature_names_file_path}")

# Determine the last completed batch
def get_last_completed_batch(directory):
    model_files = [f for f in os.listdir(directory) if f.startswith('model_batch_') and f.endswith('.pkl')]
    if not model_files:
        return 0
    batch_numbers = [int(f.split('_')[2].split('.')[0]) for f in model_files]
    return max(batch_numbers)

last_completed_batch = get_last_completed_batch(model_save_directory)

# Normalize features
print("Normalizing features...")
scaler = StandardScaler()

# Train a model for each station
models = {}

print(f"Found {len(stations)} unique stations. Training models from batch {last_completed_batch + 1}...")

# Hyperparameter space for RandomizedSearchCV
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': [10, 20, None],
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 4)
}

# Function to train and save model batches
def train_and_save_batches(stations, batch_size=100, start_batch=0):
    problematic_stations = []
    
    for batch_start in range(start_batch * batch_size, len(stations), batch_size):
        batch_end = min(batch_start + batch_size, len(stations))
        station_batch = stations[batch_start:batch_end]
        batch_models = {}
        batch_scaler = StandardScaler()

        for station in tqdm(station_batch, desc=f"Training batch {batch_start // batch_size + 1}", unit="station"):
            nearby_stations, station_data = calculate_nearby_station_status_adjustable(bike_data, station)

            if nearby_stations is None or len(nearby_stations) < 5:
                print(f"Using base features for station {station} due to insufficient nearby stations. Prediction may be less than optimal.")
                station_data = station_data.copy()
                station_data['nearby_stations_closed'] = 0
                station_data['nearby_stations_full'] = 0
                station_data['nearby_stations_empty'] = 0
                station_data['likelihood_fill'] = 0.0
                station_data['likelihood_empty'] = 0.0
            else:
                # Update station data with nearby station status
                station_data = station_data.copy()
                station_data['nearby_stations_closed'] = nearby_stations['is_renting'].apply(lambda x: 1 if x == 'NON' else 0).sum()
                station_data['nearby_stations_full'] = (nearby_stations['numbikesavailable'] == 0).sum()
                station_data['nearby_stations_empty'] = (nearby_stations['numdocksavailable'] == 0).sum()

                # Calculate likelihoods
                station_data['likelihood_fill'] = station_data['nearby_stations_full'] / (len(nearby_stations) + 1e-5)
                station_data['likelihood_empty'] = station_data['nearby_stations_empty'] / (len(nearby_stations) + 1e-5)

                selected_features = base_features + additional_features

            # Normalize the selected features
            station_data[selected_features] = batch_scaler.fit_transform(station_data[selected_features])

            # Check for infinite or excessively large values
            if np.isinf(station_data[selected_features]).values.any() or (np.abs(station_data[selected_features]) > np.finfo(np.float64).max).values.any():
                print(f"Skipping station {station} due to infinite or excessively large values after rounding.")
                problematic_stations.append(station)
                continue

            X = station_data[selected_features]
            y = station_data[target]

            # Check the size of the data for the current station
            if len(X) < 2:
                print(f"Skipping station {station} due to insufficient data (less than 2 samples).")
                continue

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Check if we have enough data after splitting
            if len(X_train) == 0 or len(X_test) == 0:
                print(f"Skipping station {station} due to insufficient data after splitting.")
                continue

            # Dynamically adjust the number of splits
            n_splits = min(5, len(X_train))
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

            # Hyperparameter tuning using Randomized Search
            random_search = RandomizedSearchCV(RandomForestRegressor(random_state=42), param_distributions=param_dist, n_iter=20, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
            random_search.fit(X_train, y_train)

            best_model = random_search.best_estimator_

            # Cross-Validation score
            cv_score = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
            print(f"Cross-Validation Score for station {station}: {cv_score.mean()}")

            # Fit and evaluate the model
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            print(f"Mean Squared Error for station {station}: {mse}")

            batch_models[station] = best_model

        # Save scaler and models for this batch
        scaler_file_path = get_absolute_path(f'../data/scaler_batch_{batch_start // batch_size + 1}.pkl')
        with open(scaler_file_path, 'wb') as f:
            pickle.dump(batch_scaler, f)
            print(f"Scaler for batch {batch_start // batch_size + 1} saved to {scaler_file_path}")

        model_file_path = get_absolute_path(f'../data/model_batch_{batch_start // batch_size + 1}.pkl')
        with open(model_file_path, 'wb') as f:
            pickle.dump(batch_models, f)
            print(f"Models for batch {batch_start // batch_size + 1} saved to {model_file_path}")
    
    if problematic_stations:
        print(f"Problematic stations skipped due to large/infinite values: {problematic_stations}")

# Call the function to train and save models in batches, starting from the last completed batch
train_and_save_batches(stations, batch_size=100, start_batch=last_completed_batch)

print("Model training and saving complete.")
