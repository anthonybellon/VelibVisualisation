import json
import os

def get_absolute_path(relative_path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))

# Load JSON data from files
with open(get_absolute_path('../../data/do_not_touch/weekly_velib_data/weekly_velib_data.json'), 'r') as file:
    velib_data_single_time = json.load(file)

with open(get_absolute_path('../../data/velib-disponibilite-en-temps-reel.json'), 'r') as file:
    velib_disponibilite_en_temps_reel = json.load(file)

# Extract station codes from both datasets
station_codes_single_time = {station['stationcode'] for station in velib_data_single_time}
station_codes_real_time = {station['stationcode'] for station in velib_disponibilite_en_temps_reel}

# Print the number of stations in each dataset
print(f'Number of stations in cleaned_weekly_data.json: {len(station_codes_single_time)}')
print(f'Number of stations in velib-disponibilite-en-temps-reel.json: {len(station_codes_real_time)}')

# Find stations that are in real-time data but not in single time data
missing_stations = [station for station in velib_disponibilite_en_temps_reel if station['stationcode'] not in station_codes_single_time]

# Save the missing stations to a new JSON file
with open('missing_stations-test.json', 'w') as file:
    json.dump(missing_stations, file, indent=4)

print(f'Found {len(missing_stations)} stations missing from velib_data_single_time.json')
