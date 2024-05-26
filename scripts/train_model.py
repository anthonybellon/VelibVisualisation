import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import os
from tqdm import tqdm  # For progress bar

# Mapping historical station names to live data names if necessary
station_name_mapping = {
    # 'Historical Name': 'Live Data Name',
    'Benjamin Godard - Victor Hugo': 'Benjamin Godard - Victor Hugo',
    # Add other mappings if necessary
}

print("Loading historical data...")
# Load historical data
data = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/historique_stations.csv'), header=None)
data.columns = ["date", "capacity", "available_mechanical", "available_electrical", "station_name", "station_geo", "operative"]

# Normalize station names
print("Normalizing station names...")
data['station_name'] = data['station_name'].replace(station_name_mapping)

# Preprocess data
print("Preprocessing data...")
data['date'] = pd.to_datetime(data['date'])
data['hour'] = data['date'].dt.hour
data[['lat', 'lon']] = data['station_geo'].str.split(',', expand=True).astype(float)
data['numbikesavailable'] = data['available_mechanical'] + data['available_electrical']

# Train a model for each station
models = {}
stations = data['station_name'].unique()

print(f"Found {len(stations)} unique stations. Training models...")

for i, station in enumerate(tqdm(stations), start=1):  # Using tqdm for progress bar
    station_data = data[data['station_name'] == station]
    X = station_data[['hour']]
    y = station_data['numbikesavailable']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    models[station] = model

    # Print progress message for each station
    print(f"Trained model for station {i}/{len(stations)}: {station}")

# Save models
model_file_path = os.path.join(os.path.dirname(__file__), '../data/model.pkl')
with open(model_file_path, 'wb') as f:
    pickle.dump(models, f)
    print(f"Models saved to {model_file_path}")
