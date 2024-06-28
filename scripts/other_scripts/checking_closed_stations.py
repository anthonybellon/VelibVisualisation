import json
from datetime import datetime
import os

def get_absolute_path(relative_path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))

# Get the absolute path of the JSON file
json_file_path = get_absolute_path('../../data/velib-disponibilite-en-temps-reel-5.json')

# Load the JSON data
with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

closed_stations = []
last_update_date = None

for record in data:
    try:
        fields = record.get('fields', {})
        if fields.get('is_renting') != "OUI" or fields.get('is_installed') != "OUI" or fields.get('is_returning') != "OUI":
            closed_stations.append(record)
        record_timestamp = record.get('record_timestamp')
        record_date = datetime.fromisoformat(record_timestamp)
        if last_update_date is None or record_date > last_update_date:
            last_update_date = record_date
    except Exception as e:
        # Skip records that are not properly structured
        continue

# Save the closed stations to a new JSON file in the same directory as the input file
output_file_path = get_absolute_path('../../data/closed_stations.json')
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    json.dump(closed_stations, output_file, ensure_ascii=False, indent=4)

# Display the results
print(f"Number of closed stations: {len(closed_stations)}")
print(f"Last update date: {last_update_date.strftime('%Y-%m-%d %H:%M:%S') if last_update_date else 'No valid date found'}")
print(f"Closed stations data saved to: {output_file_path}")
