import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os
from tqdm import tqdm
from scipy.spatial import KDTree
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json

# Load trained models
model_file_path = os.path.join(os.path.dirname(__file__), '../data/model_test.pkl')
with open(model_file_path, 'rb') as f:
    models = pickle.load(f)

# Load up-to-date bike data
print("Loading up-to-date bike data...")
velib_data = pd.read_json(os.path.join(os.path.dirname(__file__), '../data/velib_data.json'))

# Preprocess up-to-date bike data
print("Preprocessing up-to-date bike data...")
velib_data['date'] = pd.to_datetime(velib_data['duedate'])
if velib_data['date'].dt.tz is None:
    velib_data['date'] = velib_data['date'].dt.tz_localize('UTC')
else:
    velib_data['date'] = velib_data['date'].dt.tz_convert('UTC')

velib_data['lat'] = velib_data['coordonnees_geo'].apply(lambda x: x['lat'])
velib_data['lon'] = velib_data['coordonnees_geo'].apply(lambda x: x['lon'])
velib_data['operative'] = velib_data['is_installed'] == "OUI"
velib_data['hour'] = velib_data['date'].dt.hour
velib_data['day_of_week'] = velib_data['date'].dt.dayofweek

# Select features
original_features = ['hour', 'day_of_week', 'tempmax', 'tempmin', 'temp', 'humidity', 'precip', 'windspeed', 
                     'nearby_stations_closed', 'nearby_stations_full', 'nearby_stations_empty', 
                     'likelihood_fill', 'likelihood_empty']

# Create missing features with zero values
for feature in original_features:
    if feature not in velib_data.columns:
        velib_data[feature] = 0

# Select features for normalization
features = ['hour', 'day_of_week', 
            'nearby_stations_closed', 'nearby_stations_full', 'nearby_stations_empty', 
            'likelihood_fill', 'likelihood_empty']

# Normalize features
scaler = StandardScaler()

def calculate_nearby_station_status(data, radius=500):
    data = data.copy()
    coords = data[['lat', 'lon']].values
    kd_tree = KDTree(coords)

    data['nearby_stations_closed'] = 0
    data['nearby_stations_full'] = 0
    data['nearby_stations_empty'] = 0
    data['likelihood_fill'] = 0.0
    data['likelihood_empty'] = 0.0
    
    for i, row in tqdm(data.iterrows(), total=len(data), desc="Calculating nearby station status"):
        station_coords = (row['lat'], row['lon'])
        indices = kd_tree.query_ball_point(station_coords, radius / 1000.0)
        nearby_stations = data.iloc[indices]
        
        nearby_stations_closed = nearby_stations['operative'].apply(lambda x: 1 if not x else 0).sum()
        nearby_stations_full = (nearby_stations['numbikesavailable'] == nearby_stations['capacity']).sum()
        nearby_stations_empty = (nearby_stations['numbikesavailable'] == 0).sum()

        data.at[i, 'nearby_stations_closed'] = nearby_stations_closed
        data.at[i, 'nearby_stations_full'] = nearby_stations_full
        data.at[i, 'nearby_stations_empty'] = nearby_stations_empty

        total_nearby_stations = len(nearby_stations)
        if total_nearby_stations > 0:
            likelihood_fill = nearby_stations_full / total_nearby_stations
            likelihood_empty = nearby_stations_empty / total_nearby_stations
        else:
            likelihood_fill = 0.0
            likelihood_empty = 0.0

        data.at[i, 'likelihood_fill'] = likelihood_fill
        data.at[i, 'likelihood_empty'] = likelihood_empty

    return data

# Apply the function to the up-to-date data
velib_data = calculate_nearby_station_status(velib_data)

# Normalize the merged data
velib_data[original_features] = scaler.fit_transform(velib_data[original_features])

# Prediction and comparison
print("Predicting and comparing bike availability...")
results = []

for station in tqdm(models.keys(), desc="Predicting for each station"):
    station_data = velib_data[velib_data['name'] == station]
    if station_data.empty:
        continue

    X = station_data[original_features]
    actuals = station_data['numbikesavailable'].values
    predictions = models[station].predict(X)

    comparison = pd.DataFrame({
        'station_name': station,
        'date': station_data['date'],
        'lat': station_data['lat'],
        'lon': station_data['lon'],
        'actual': actuals,
        'predicted': predictions
    })

    results.append(comparison)

comparison_results = pd.concat(results, ignore_index=True)

# Save the results as a JSON file
output_file_path = os.path.join(os.path.dirname(__file__), '../data/predictions_comparison.json')
comparison_results.to_json(output_file_path, orient='records', date_format='iso')

print("Prediction and comparison completed. Results saved to predictions_comparison.json")

# Calculate metrics
mae = mean_absolute_error(comparison_results['actual'], comparison_results['predicted'])
rmse = mean_squared_error(comparison_results['actual'], comparison_results['predicted'], squared=False)

metrics = {
    'MAE': mae,
    'RMSE': rmse
}

# Save metrics to a JSON file
metrics_file_path = os.path.join(os.path.dirname(__file__), '../data/prediction_metrics.json')
with open(metrics_file_path, 'w') as f:
    json.dump(metrics, f)

print("Metrics calculated and saved to prediction_metrics.json")
