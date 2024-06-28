import json
import os
from datetime import datetime

def get_absolute_path(relative_path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))

# Load the JSON files
with open(get_absolute_path('../data/2_organized_predictions.json'), 'r') as f:
    organized_predictions = json.load(f)

with open(get_absolute_path('../data/3_predictions_results.json'), 'r') as f:
    predictions_results = json.load(f)

# Create a dictionary to store the latest capacity for each station
latest_capacity = {}

# Iterate over the organized predictions to find the latest capacity for each station
for entry in organized_predictions:
    station_id = (entry['stationcode'], entry['name'])
    duedate = datetime.strptime(entry['duedate'], "%Y-%m-%dT%H:%M:%S%z")

    if station_id not in latest_capacity or duedate > latest_capacity[station_id]['duedate']:
        latest_capacity[station_id] = {
            'capacity': entry['capacity'],
            'duedate': duedate
        }

# Update the capacities in the predictions results
for entry in predictions_results:
    station_id = (entry['stationcode'], entry['name'])

    if station_id in latest_capacity:
        entry['capacity'] = latest_capacity[station_id]['capacity']

# Save the updated predictions results to a new JSON file
with open(get_absolute_path('../data/3_predictions_results_updated.json'), 'w') as f:
    json.dump(predictions_results, f, indent=4)

print("Updated capacities in 3_predictions_results_updated.json")
