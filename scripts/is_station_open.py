import json
import os

def get_absolute_path(relative_path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))

# Load real-time availability data
realtime_data_path = get_absolute_path('../data/velib-disponibilite-en-temps-reel.json')
with open(realtime_data_path, 'r') as file:
    realtime_data = json.load(file)

# Load processed predictions data
processed_data_path = get_absolute_path('../data/processed_predictions.json')
with open(processed_data_path, 'r') as file:
    processed_data = json.load(file)

# Create a mapping of station codes to their is_renting status from real-time data
realtime_status = {station['stationcode']: station['is_renting'] for station in realtime_data}

# Update the is_renting status in the processed predictions data
for stationcode in processed_data:
    if stationcode in realtime_status:
        processed_data[stationcode]['is_renting'] = realtime_status[stationcode]

# Save the updated processed predictions data
with open(processed_data_path, 'w') as file:
    json.dump(processed_data, file, indent=4)

print("Station status updated in processed_predictions.json.")
