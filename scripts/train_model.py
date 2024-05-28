import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
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

# Function to get the absolute path relative to the script location
def get_absolute_path(relative_path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))

# Directory containing the cleaned JSON files
clean_data_directory = get_absolute_path('../monthly_data_clean')

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

print("Preprocessing complete.")

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

            data.at[index, 'nearby_stations_closed'] = nearby_stations_closed
            data.at[index, 'nearby_stations_full'] = nearby_stations_full
            data.at[index, 'nearby_stations_empty'] = nearby_stations_empty

            total_nearby_stations = len(date_filtered_nearby_stations)
            if total_nearby_stations > 0:
                likelihood_fill = nearby_stations_full / total_nearby_stations
                likelihood_empty = nearby_stations_empty / total_nearby_stations
            else:
                likelihood_fill = 0.0
                likelihood_empty = 0.0

            data.at[index, 'likelihood_fill'] = likelihood_fill
            data.at[index, 'likelihood_empty'] = likelihood_empty

            # Adjust likelihood based on nearby station statuses
            if nearby_stations_full > total_nearby_stations / 2:
                data.at[index, 'likelihood_fill'] *= 1.5  # Increase likelihood of filling up
            if nearby_stations_empty > total_nearby_stations / 2:
                data.at[index, 'likelihood_empty'] *= 1.5  # Increase likelihood of emptying

    return data

# Apply the function to the bike data
print("Calculating nearby station status...")
bike_data = calculate_nearby_station_status(bike_data, limit=5)

# Feature engineering: Add 'avg_bikes_hour_day'
bike_data['avg_bikes_hour_day'] = bike_data.groupby(['stationcode', 'hour', 'day_of_week'])['numbikesavailable'].transform('mean')

# Select features and target
features = ['hour', 'day_of_week', 'nearby_stations_closed', 'nearby_stations_full', 'nearby_stations_empty', 
            'likelihood_fill', 'likelihood_empty', 'avg_bikes_hour_day']
target = 'numbikesavailable'

# Normalize features
print("Normalizing features...")
scaler = StandardScaler()
bike_data[features] = scaler.fit_transform(bike_data[features])

# Train a model for each station
models = {}
stations = bike_data['stationcode'].unique()[:5]  # Limit to the first 5 stations

print(f"Found {len(stations)} unique stations. Training models...")

for i, station in enumerate(tqdm(stations, desc="Training models", unit="station")):
    station_data = bike_data[bike_data['stationcode'] == station]
    X = station_data[features]
    y = station_data[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning using Grid Search
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Cross-Validation score
    cv_score = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
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

model_file_path = get_absolute_path('../data/model_test.pkl')
with open(model_file_path, 'wb') as f:
    pickle.dump(models, f)
    print(f"Models saved to {model_file_path}")

print("Model training complete.")
