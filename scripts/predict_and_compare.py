import pickle
import json
import pandas as pd
import os

# Load models
with open(os.path.join(os.path.dirname(__file__), '../data/model.pkl'), 'rb') as f:
    models = pickle.load(f)

# Load real-time data
with open(os.path.join(os.path.dirname(__file__), '../data/velib_data.json')) as f:
    real_time_data = json.load(f)

# Prepare predictions
predictions = []

for station in real_time_data:
    station_name = station['name']
    hour = pd.to_datetime(station['duedate']).hour
    model = models.get(station_name)
    if model:
        # Create a DataFrame with the correct feature name
        input_data = pd.DataFrame({'hour': [hour]})
        predicted_bikes = model.predict(input_data)[0]
        station['predicted_numbikesavailable'] = predicted_bikes
        predictions.append(station)

# Save predictions
predictions_file_path = os.path.join(os.path.dirname(__file__), '../data/predictions.json')
with open(predictions_file_path, 'w') as f:
    json.dump(predictions, f)
    print(f"Predictions saved to {predictions_file_path}")
