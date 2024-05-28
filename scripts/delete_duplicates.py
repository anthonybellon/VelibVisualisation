import os
import json
from tqdm import tqdm
from collections import defaultdict

# Function to get the absolute path relative to the script location
def get_absolute_path(relative_path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))

# Define the directories and paths
input_directory = get_absolute_path('../monthly_data')
output_clean_directory = get_absolute_path('../monthly_data_clean')
output_duplicates_directory = get_absolute_path('../monthly_data_duplicates')

# Ensure the output directories exist
os.makedirs(output_clean_directory, exist_ok=True)
os.makedirs(output_duplicates_directory, exist_ok=True)

# Process each JSON file in the input directory
json_files = [f for f in os.listdir(input_directory) if f.endswith('.json')]

for filename in tqdm(json_files, desc="Processing JSON files"):
    file_path = os.path.join(input_directory, filename)
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Dictionary to track entries by station code and date
    entry_tracker = defaultdict(dict)
    duplicates = []

    # Check for duplicates
    for entry in tqdm(data, desc=f"Checking duplicates in {filename}", leave=False):
        station_code = entry['stationcode']
        duedate = entry['duedate']
        
        if station_code in entry_tracker and duedate in entry_tracker[station_code]:
            duplicates.append(entry)
        else:
            entry_tracker[station_code][duedate] = entry

    # Collect the clean data
    clean_data = [entry for station_entries in entry_tracker.values() for entry in station_entries.values()]

    # Save the clean data
    clean_file_path = os.path.join(output_clean_directory, filename)
    with open(clean_file_path, 'w') as f:
        json.dump(clean_data, f, indent=4)
    
    # Save the duplicates
    if duplicates:
        duplicates_file_path = os.path.join(output_duplicates_directory, filename)
        with open(duplicates_file_path, 'w') as f:
            json.dump(duplicates, f, indent=4)

print("Duplicate check and clean data processing complete.")
