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
# Mapping historical station names to live data names if necessary
station_name_mapping = {
    'Benjamin Godard - Victor Hugo': 'Benjamin Godard - Victor Hugo',
    # Add other mappings if necessary
}

# Load historical bike data
print("Loading historical bike data...")
bike_data = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/historique_stations.csv'), header=None)
bike_data.columns = ["date", "capacity", "available_mechanical", "available_electrical", "station_name", "station_geo", "operative"]

# Normalize station names
print("Normalizing station names...")
bike_data['station_name'] = bike_data['station_name'].replace(station_name_mapping)

# Preprocess bike data
print("Preprocessing bike data...")
bike_data['date'] = pd.to_datetime(bike_data['date'])

# Ensure 'date' column is timezone-aware (localize to UTC if needed)
if bike_data['date'].dt.tz is None:
    bike_data['date'] = bike_data['date'].dt.tz_localize('UTC')

bike_data[['lat', 'lon']] = bike_data['station_geo'].str.split(',', expand=True).astype(float)
bike_data['numbikesavailable'] = bike_data['available_mechanical'] + bike_data['available_electrical']

# Load weather data
print("Loading weather data...")
weather_data = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/weather.csv'))

# Manually set column names
weather_data.columns = [
    "name", "datetime", "tempmax", "tempmin", "temp", "feelslikemax", "feelslikemin", 
    "feelslike", "dew", "humidity", "precip", "precipprob", "precipcover", "preciptype", 
    "snow", "snowdepth", "windgust", "windspeed", "winddir", "sealevelpressure", 
    "cloudcover", "visibility", "solarradiation", "solarenergy", "uvindex", "severerisk", 
    "sunrise", "sunset", "moonphase", "conditions", "description", "icon", "stations"
]

# Debug: print the column names to verify
print("Column names after setting manually:", weather_data.columns.tolist())

# Rename 'datetime' to 'date'
weather_data.rename(columns={'datetime': 'date'}, inplace=True)

# Ensure 'date' column is in datetime format and timezone-aware
weather_data['date'] = pd.to_datetime(weather_data['date'], errors='coerce').dt.tz_localize('UTC')

# Debug: print data types to verify 'date' conversion
print("Data types after converting 'date' to datetime:", weather_data.dtypes)

# Select necessary columns
weather_data = weather_data[['date', 'tempmax', 'tempmin', 'temp', 'humidity', 'precip', 'windspeed', 'conditions']]
print("Weather data preview after selecting necessary columns:")
print(weather_data.head())

# Ensure there are no NaN values in weather_data by filling forward and backward
weather_data.fillna(method='ffill', inplace=True)
weather_data.fillna(method='bfill', inplace=True)

# Ensure both 'date' columns have the same timezone information
bike_data['date'] = bike_data['date'].dt.tz_convert('UTC')
weather_data['date'] = weather_data['date'].dt.tz_convert('UTC')

# Merge weather data with bike data
print("Merging weather data with bike data...")
merged_data = pd.merge(bike_data, weather_data, on='date', how='left')

# Forward-fill and backward-fill the weather data to fill NaNs for each 15-minute interval within the same day
merged_data.fillna(method='ffill', inplace=True)
merged_data.fillna(method='bfill', inplace=True)

# Ensure no NaNs in the important columns
important_columns = ['tempmax', 'tempmin', 'temp', 'humidity', 'precip', 'windspeed', 'conditions']
merged_data[important_columns] = merged_data[important_columns].fillna(0)

# Print out the first few rows of the merged data to verify
print("First few rows of the merged data:")
print(merged_data.head(10))

# Add hour and day_of_week from the bike data
merged_data['hour'] = merged_data['date'].dt.hour
merged_data['day_of_week'] = merged_data['date'].dt.dayofweek

def calculate_nearby_station_status(data, radius=500, limit=10):
    data = data.copy()
    stations = data['station_name'].unique()[:limit]  # Limit to the first 'limit' stations

    coords = data[['lat', 'lon']].drop_duplicates().values
    kd_tree = KDTree(coords)

    data['nearby_stations_closed'] = 0
    data['nearby_stations_full'] = 0
    data['nearby_stations_empty'] = 0
    data['likelihood_fill'] = 0.0
    data['likelihood_empty'] = 0.0

    for station in tqdm(stations, desc="Calculating nearby station status"):
        station_data = data[data['station_name'] == station]
        station_coords = station_data[['lat', 'lon']].iloc[0].values

        indices = kd_tree.query_ball_point(station_coords, radius / 1000.0)
        nearby_stations = data.iloc[indices]

        for index, row in station_data.iterrows():
            date_filtered_nearby_stations = nearby_stations[nearby_stations['date'] == row['date']]
            nearby_stations_closed = date_filtered_nearby_stations['operative'].apply(lambda x: 1 if not x else 0).sum()
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

# Apply the function to the merged data
merged_data = calculate_nearby_station_status(merged_data, limit=5)

# Select features and target
features = ['hour', 'day_of_week', 'tempmax', 'tempmin', 'temp', 'humidity', 'precip', 'windspeed', 
            'nearby_stations_closed', 'nearby_stations_full', 'nearby_stations_empty', 
            'likelihood_fill', 'likelihood_empty']
target = 'numbikesavailable'

# Normalize features
scaler = StandardScaler()
merged_data[features] = scaler.fit_transform(merged_data[features])

# Train a model for each station
models = {}
stations = merged_data['station_name'].unique()[:5]  # Limit to the first 5 stations

print(f"Found {len(stations)} unique stations. Training models...")

for i, station in enumerate(tqdm(stations, desc="Training models", unit="station")):
    station_data = merged_data[merged_data['station_name'] == station]
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
model_file_path = os.path.join(os.path.dirname(__file__), '../data/model_test.pkl')
with open(model_file_path, 'wb') as f:
    pickle.dump(models, f)
    print(f"Models saved to {model_file_path}")