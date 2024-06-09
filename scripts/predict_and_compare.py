import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime, timedelta

def load_feature_names():
    feature_name_path = '../data/feature_names.json'
    with open(feature_name_path, 'r') as file:
        feature_names = json.load(file)
    return feature_names

def create_features(data):
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

    return data

def load_model_and_scaler(station):
    model_file_path = f'path/to/model_{station}.pkl'
    scaler_file_path = f'path/to/scaler_{station}.pkl'
    
    with open(model_file_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_file_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler

def predict_bike_availability(data, station, hours=24):
    model, scaler = load_model_and_scaler(station)
    feature_names = load_feature_names()

    # Create necessary features
    data = create_features(data)

    # Ensure all required features are present
    for feature in feature_names:
        if feature not in data.columns:
            data[feature] = 0  # Or some appropriate default value
    
    X = data[feature_names]
    X = scaler.transform(X)
    
    predictions = model.predict(X)
    return predictions[:hours]

def generate_prediction_json(data, station_details, hours=24):
    predictions_by_station = {}

    for station in station_details:
        station_code = station['stationcode']
        station_name = station['name']
        capacity = station['capacity']
        is_renting = station['is_renting']
        coordonnees_geo = station['coordonnees_geo']

        station_data = data[data['stationcode'] == station_code]
        
        if station_data.empty:
            predictions = {str(i+1): [None] * hours for i in range(hours)}
        else:
            predictions = {}
            for i in range(1, 7):
                future_data = station_data.copy()
                future_data['date'] = future_data['date'] + timedelta(days=i)
                predictions[str(i)] = predict_bike_availability(future_data, station_code, hours).tolist()

        predictions_by_station[station_code] = {
            "name": station_name,
            "capacity": capacity,
            "is_renting": is_renting,
            "coordonnees_geo": coordonnees_geo,
            "predictions": predictions
        }

    return predictions_by_station

# Example usage
data = pd.read_parquet('path/to/your/prediction/data.parquet')
station_details = [
    {
        "stationcode": "12109",
        "name": "Mairie du 12ème",
        "capacity": 30,
        "is_renting": "OUI",
        "coordonnees_geo": {"lon": 2.3875549435616, "lat": 48.840855311763}
    },
    # Add more stations as needed
]

predictions_json = generate_prediction_json(data, station_details, hours=24)

# Save the predictions to a JSON file
with open('path/to/predictions.json', 'w') as json_file:
    json.dump(predictions_json, json_file, indent=2)

print(f"Predictions saved to 'path/to/predictions.json'")
