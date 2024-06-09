import os
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import randint
from tqdm import tqdm
from data_preprocessing import get_absolute_path
from utils import calculate_nearby_station_status_adjustable

def load_feature_names():
    feature_name_path = get_absolute_path('../data/feature_names.json')
    with open(feature_name_path, 'r') as file:
        feature_names = json.load(file)
    return feature_names

def check_for_large_values(data):
    numeric_data = data.select_dtypes(include=[np.number])
    large_values = (numeric_data.abs() > 1e10).any()
    if large_values.any():
        print("Large values found in columns during feature creation:", numeric_data.columns[large_values].tolist())
    return large_values.any()

def create_features(data):
    print ("Creating features...")
    data = data.copy()
    data['hour'] = data['date'].dt.hour
    data['day_of_week'] = data['date'].dt.dayofweek

    data = data.sort_values(by=['stationcode', 'date'])
    data['lag_1_hour'] = data.groupby('stationcode')['numbikesavailable'].shift(1)
    data['lag_1_day'] = data.groupby('stationcode')['numbikesavailable'].shift(24)

    data['rolling_mean_7_days'] = data.groupby('stationcode')['numbikesavailable'].transform(lambda x: x.rolling(window=7*24, min_periods=1).mean())
    data['rolling_mean_30_days'] = data.groupby('stationcode')['numbikesavailable'].transform(lambda x: x.rolling(window=30*24, min_periods=1).mean())

    # Add small constants to avoid division by zero
    data['normalized_bikes_available'] = data['numbikesavailable'] / (data['capacity'] + 1e-5)
    data['normalized_docks_available'] = data['numdocksavailable'] / (data['capacity'] + 1e-5)
    data['usage_ratio'] = data['numbikesavailable'] / (data['capacity'] + 1e-5)
    data['capacity_hour_interaction'] = data['capacity'] * data['hour']
    data['capacity_day_interaction'] = data['capacity'] * data['day_of_week']

    data['avg_bikes_hour_day'] = data.groupby(['stationcode', 'hour', 'day_of_week'])['numbikesavailable'].transform('mean')

    # Check for large values after feature creation
    if check_for_large_values(data):
        raise ValueError("Large values found in feature creation")

    return data

def train_model(data, stations, target='numbikesavailable'):
    print("Training models...")

    # Create necessary features
    data = create_features(data)
    
    # Debug: Print columns to ensure features are created
    print("Columns after feature creation:", data.columns)

    # Normalize features
    scaler = StandardScaler()
    
    # Load feature names from JSON
    feature_names = load_feature_names()
    base_features = feature_names[:12]
    additional_features = feature_names[12:]

    models = {}
    param_dist = {
        'n_estimators': randint(50, 200),
        'max_depth': [10, 20, None],
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 4)
    }

    for station in tqdm(stations, desc="Training models", unit="station"):
        nearby_stations, station_data = calculate_nearby_station_status_adjustable(data, station)

        if nearby_stations is None or len(nearby_stations) < 5:
            selected_features = base_features
        else:
            station_data = station_data.copy()
            station_data['nearby_stations_closed'] = nearby_stations['is_renting'].apply(lambda x: 1 if x == 'NON' else 0).sum()
            station_data['nearby_stations_full'] = (nearby_stations['numbikesavailable'] == 0).sum()
            station_data['nearby_stations_empty'] = (nearby_stations['numdocksavailable'] == 0).sum()
            station_data['likelihood_fill'] = station_data['nearby_stations_full'] / (len(nearby_stations) + 1e-5)
            station_data['likelihood_empty'] = station_data['nearby_stations_empty'] / (len(nearby_stations) + 1e-5)
            selected_features = base_features + additional_features

        # Debug: Check if selected features exist in the data
        missing_features = [feat for feat in selected_features if feat not in station_data.columns]
        if missing_features:
            print(f"Missing features for station {station}: {missing_features}")
            continue

        # Handle infinite and large values
        # station_data = handle_infinite_and_large_values(station_data)

        station_data[selected_features] = scaler.fit_transform(station_data[selected_features])

        X = station_data[selected_features]
        y = station_data[target]

        if len(X) < 2:
            continue

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        n_splits = min(5, len(X_train))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        random_search = RandomizedSearchCV(RandomForestRegressor(random_state=42), param_distributions=param_dist, n_iter=20, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_
        cv_score = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
        print(f"Cross-Validation Score for station {station}: {cv_score.mean()}")
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error for station {station}: {mse}")

        models[station] = best_model

    return models, scaler

def save_intermediate_results(models, scalers, batch_number):
    model_file_path = get_absolute_path(f'../data/model_batch_{batch_number}.pkl')
    scaler_file_path = get_absolute_path(f'../data/scaler_batch_{batch_number}.pkl')

    with open(model_file_path, 'wb') as f:
        pickle.dump(models, f)
    with open(scaler_file_path, 'wb') as f:
        pickle.dump(scalers, f)
    print(f"Intermediate results saved for batch {batch_number}")

def load_intermediate_results(batch_number):
    model_file_path = get_absolute_path(f'../data/model_batch_{batch_number}.pkl')
    scaler_file_path = get_absolute_path(f'../data/scaler_batch_{batch_number}.pkl')

    if os.path.exists(model_file_path) and os.path.exists(scaler_file_path):
        with open(model_file_path, 'rb') as f:
            models = pickle.load(f)
        with open(scaler_file_path, 'rb') as f:
            scalers = pickle.load(f)
        return models, scalers
    return {}, {}

def main():
    processed_data_file_path = get_absolute_path('../data/processed_bike_data.parquet')
    bike_data = pd.read_parquet(processed_data_file_path)
    
    if bike_data.empty:
        raise ValueError("The processed bike data is empty. Please check the Parquet file and its loading process.")
    
    # Batch process stations in chunks of 100
    stations = bike_data['stationcode'].unique()
    batch_size = 100

    all_models = {}
    all_scalers = {}

    # Load intermediate results if they exist
    current_batch = 0
    while os.path.exists(get_absolute_path(f'../data/model_batch_{current_batch}.pkl')):
        models, scalers = load_intermediate_results(current_batch)
        all_models.update(models)
        all_scalers.update(scalers)
        current_batch += 1

    for i in range(current_batch * batch_size, len(stations), batch_size):
        batch_stations = stations[i:i + batch_size]
        models, scaler = train_model(bike_data, batch_stations)
        all_models.update(models)
        all_scalers.update({station: scaler for station in batch_stations})
        
        # Save intermediate results
        save_intermediate_results(models, {station: scaler for station in batch_stations}, current_batch)
        current_batch += 1

    # Save the final combined results
    scaler_file_path = get_absolute_path('../data/scaler_final.pkl')
    model_file_path = get_absolute_path('../data/model_final.pkl')

    with open(scaler_file_path, 'wb') as f:
        pickle.dump(all_scalers, f)
    print(f"Scaler saved to {scaler_file_path}")

    with open(model_file_path, 'wb') as f:
        pickle.dump(all_models, f)
    print(f"Models saved to {model_file_path}")

    print("Model training complete.")

if __name__ == "__main__":
    main()
