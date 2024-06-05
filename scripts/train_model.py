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

# Extract latitude and longitude
print("Extracting latitude and longitude...")
bike_data['lat'] = bike_data['coordonnees_geo'].apply(lambda x: x.get('lat', np.nan) if x else np.nan)
bike_data['lon'] = bike_data['coordonnees_geo'].apply(lambda x: x.get('lon', np.nan) if x else np.nan)

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
station_limit = None  # Set to 1 for processing only the first station

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

# Normalize features
print("Normalizing features...")
scaler = StandardScaler()

# Train a model for each station
models = {}

print(f"Found {len(stations)} unique stations. Training models...")

# Hyperparameter space for RandomizedSearchCV
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': [10, 20, None],
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 4)
}

# Skip the first 99 stations and start from the 100th station
for i, station in enumerate(tqdm(stations[100:], desc="Training models", unit="station")):
    actual_station_index = i + 100  # To display the correct station index
    nearby_stations, station_data = calculate_nearby_station_status_adjustable(bike_data, station)

    # If we have fewer than 5 nearby stations, we omit the additional features
    if nearby_stations is None or len(nearby_stations) < 5:
        print(f"Using base features for station {station} due to insufficient nearby stations. Prediction may be less than optimal.")
        selected_features = base_features
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
    station_data[selected_features] = scaler.fit_transform(station_data[selected_features])

    X = station_data[selected_features]
    y = station_data[target]

    # Check the size of the data for the current station
    print(f"Station {station} (Index {actual_station_index}) has {len(X)} samples.")

    # Check if we have enough data to create a train/test split
    if len(X) < 2:
        print(f"Skipping station {station} (Index {actual_station_index}) due to insufficient data (less than 2 samples).")
        continue

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Check if we have enough data after splitting
    if len(X_train) == 0 or len(X_test) == 0:
        print(f"Skipping station {station} (Index {actual_station_index}) due to insufficient data after splitting.")
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
    print(f"Cross-Validation Score for station {station} (Index {actual_station_index}): {cv_score.mean()}")

    # Fit and evaluate the model
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error for station {station} (Index {actual_station_index}): {mse}")

    models[station] = best_model

    # Print progress message for each station
    print(f"Trained model for station {actual_station_index+1}/{len(stations)}: {station}")
    print(f"Model score: {best_model.score(X_test, y_test)}")

# Save models
print("Saving scaler and models...")
scaler_file_path = get_absolute_path('../data/scaler_final.pkl')
with open(scaler_file_path, 'wb') as f:
    pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_file_path}")

model_file_path = get_absolute_path('../data/model_final.pkl')
with open(model_file_path, 'wb') as f:
    pickle.dump(models, f)
    print(f"Models saved to {model_file_path}")

print("Model training complete.")
