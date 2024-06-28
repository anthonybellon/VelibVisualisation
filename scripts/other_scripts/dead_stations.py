import json
import os
from tqdm import tqdm

def get_absolute_path(relative_path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))

# Paths to the data files
current_data_path = get_absolute_path('../../data/velib-disponibilite-en-temps-reel.json')
historical_data_folder = get_absolute_path('../../data/historical_data_cleaned/')
closed_stations_output_path = get_absolute_path('../data/closed_stations.json')

# Ensure the output directory exists
output_directory = os.path.dirname(closed_stations_output_path)
os.makedirs(output_directory, exist_ok=True)

# Load the current station data
with open(current_data_path, 'r', encoding='utf-8') as file:
    current_data = json.load(file)

# Extract currently open station codes
if isinstance(current_data, dict) and 'records' in current_data:
    current_station_codes = {station['fields']['stationcode'] for station in current_data['records']}
elif isinstance(current_data, list):
    current_station_codes = {station['fields']['stationcode'] for station in current_data}
else:
    raise ValueError("Unexpected data structure for current_data")

# Initialize a list to hold the closed stations data
closed_stations = []
unique_historical_station_codes = set()

# Iterate through the historical data files with a progress bar
historical_files = [file for file in os.listdir(historical_data_folder) if file.endswith('.json')]
for historical_file in tqdm(historical_files, desc="Processing historical data files"):
    historical_file_path = os.path.join(historical_data_folder, historical_file)
    
    with open(historical_file_path, 'r', encoding='utf-8') as file:
        historical_data = json.load(file)
        
        for station in historical_data:
            unique_historical_station_codes.add(station['stationcode'])
            if station['stationcode'] not in current_station_codes:
                closed_stations.append({
                    'stationcode': station['stationcode'],
                    'name': station['name']
                })

# Write the closed stations data to a JSON file
with open(closed_stations_output_path, 'w', encoding='utf-8') as file:
    json.dump(closed_stations, file, ensure_ascii=False, indent=4)

# Print the number of unique station codes in historical data
print(f"Closed stations data has been written to {closed_stations_output_path}")
print(f"Number of unique station codes in historical data: {len(unique_historical_station_codes)}")
