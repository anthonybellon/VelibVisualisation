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

# Function to convert date chunks
def convert_date_chunk(chunk):
    return pd.to_datetime(chunk)

# Preprocess bike data
print("Preprocessing bike data...")

# Chunk size for date conversion
chunk_size = 1000  # Adjust the chunk size based on your data and performance needs

# Split the 'duedate' column into chunks
num_chunks = (len(bike_data) // chunk_size) + 1
duedate_chunks = np.array_split(bike_data['duedate'], num_chunks)

# Convert each chunk and show progress
converted_dates = []
for chunk in tqdm(duedate_chunks, desc="Converting dates"):
    converted_dates.append(convert_date_chunk(chunk))

# Concatenate the converted dates back into a single series
bike_data['date'] = pd.concat(converted_dates)

# Ensure 'date' column is timezone-aware (localize to UTC if needed)
if bike_data['date'].dt.tz is None:
    bike_data['date'] = bike_data['date'].dt.tz_localize('UTC')

# Extract latitude and longitude
print("Extracting latitude and longitude...")
latitudes = []
longitudes = []
for coord in tqdm(bike_data['coordonnees_geo'], desc="Extracting coordinates"):
    latitudes.append(coord['lat'])
    longitudes.append(coord['lon'])

bike_data['lat'] = latitudes
bike_data['lon'] = longitudes

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

# Lag features for the number of bikes available
bike_data['lag_1_hour'] = bike_data.groupby('stationcode')['numbikesavailable'].shift(1)
bike_data['lag_1_day'] = bike_data.groupby('stationcode')['numbikesavailable'].shift(24)

# Adding trend features
print("Adding trend features...")
bike_data['rolling_mean_7_days'] = bike_data.groupby('stationcode')['numbikesavailable'].transform(lambda x: x.rolling(window=7*24, min_periods=1).mean())
bike_data['rolling_mean_30_days'] = bike_data.groupby('stationcode')['numbikesavailable'].transform(lambda x: x.rolling(window=30*24, min_periods=1).mean())

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
    while radius <= max_radius:
        # Query the KDTree with radius in degrees converted to approximate meters
        indices = kd_tree.query_ball_point(station_coords, radius / 1000.0 / 111.32)  # Approx conversion from meters to degrees
        
        # Filter out the target station itself
        nearby_indices = [i for i in indices if station_codes[i] != target_station_code]
        
        # If the radius is greater than the initial radius and there are more than 5 stations, limit to 5
        if radius > initial_radius and len(nearby_indices) > 5:
            nearby_indices = nearby_indices[:5]
        
        nearby_stations = data.iloc[nearby_indices]
        
        # Check the number of nearby stations found

        # Manually check distances for a few points to verify
        for idx in nearby_indices[:5]:  # Check first 5 points for verification
            point_coords = unique_coords[idx]  # Use original coordinates in degrees
            distance = geodesic(station_coords, point_coords).meters

        if len(nearby_stations) >= 5:
            return nearby_stations, station_data
        
        radius += increment
    
    return None, station_data

# Apply the function to the bike data
print("Calculating nearby station status...")

# Optional limit for the number of stations to process
station_limit = 500  # Set to 1 for processing only the first station

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
features = ['hour', 'day_of_week', 'nearby_stations_closed', 'nearby_stations_full', 'nearby_stations_empty', 
            'likelihood_fill', 'likelihood_empty', 'avg_bikes_hour_day', 'lag_1_hour', 'lag_1_day', 
            'rolling_mean_7_days', 'rolling_mean_30_days']
target = 'numbikesavailable'

# Normalize features
print("Normalizing features...")
scaler = StandardScaler()
bike_data[features] = scaler.fit_transform(bike_data[features])

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

for i, station in enumerate(tqdm(stations, desc="Training models", unit="station")):
    nearby_stations, station_data = calculate_nearby_station_status_adjustable(bike_data, station)
    
    if nearby_stations is None or len(station_data) < 5:
        print(f"Skipping station {station} due to insufficient nearby stations or data.")
        continue

    X = station_data[features]
    y = station_data[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if len(X_train) < 5:
        print(f"Skipping station {station} due to insufficient data.")
        continue
    
    # Dynamically adjust the number of splits
    n_splits = min(5, len(X_train))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Hyperparameter tuning using Randomized Search
    random_search = RandomizedSearchCV(RandomForestRegressor(random_state=42), param_distributions=param_dist, n_iter=20, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
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
    
    models[station] = best_model

    # Print progress message for each station
    print(f"Trained model for station {i+1}/{len(stations)}: {station}")
    print(f"Model score: {best_model.score(X_test, y_test)}")

# Save models
print("Saving scaler and models...")
scaler_file_path = get_absolute_path('../data/scaler.pkl')
with open(scaler_file_path, 'wb') as f:
    pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_file_path}")

model_file_path = get_absolute_path('../data/model_randomizedcv.pkl')
with open(model_file_path, 'wb') as f:
    pickle.dump(models, f)
    print(f"Models saved to {model_file_path}")

print("Model training complete.")
