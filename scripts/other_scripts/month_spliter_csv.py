import pandas as pd
import json
from datetime import datetime
import os
from tqdm import tqdm

print("Current Working Directory:", os.getcwd())


# File paths
historical_csv_path = '../data/historique_stations.csv'
current_json_path = '../data/velib_data.json'
output_directory = '../monthly_data'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Read the CSV data
data = pd.read_csv(historical_csv_path, header=None, names=[
    "date", "capacity", "available_mechanical", "available_electrical", 
    "station_name", "station_geo", "operative"
])

# Read the current JSON data
with open(current_json_path, 'r') as f:
    current_data = json.load(f)

# Convert current JSON data to a DataFrame
current_df = pd.DataFrame(current_data)

# Convert date column to datetime
data['date'] = pd.to_datetime(data['date'])

# Extract year and month for grouping
data['year_month'] = data['date'].dt.to_period('M')

# Merge the historical data with the current data on station_name
merged_data = pd.merge(data, current_df, left_on='station_name', right_on='name', how='left', suffixes=('', '_current'))

# Function to convert historical data row to current JSON format
def convert_row_to_json(row):
    if pd.isna(row['stationcode']):
        return None  # Skip if stationcode is not found
    
    # Extract latitude and longitude from station_geo
    lat, lon = map(float, row['station_geo'].strip('"').split(','))
    
    # Create the JSON structure
    json_data = {
        "stationcode": row['stationcode'],
        "name": row['station_name'],
        "is_installed": "OUI" if row['operative'] else "NON",
        "capacity": row['capacity'],
        "numdocksavailable": row['capacity'] - row['available_mechanical'] - row['available_electrical'],
        "numbikesavailable": row['available_mechanical'] + row['available_electrical'],
        "mechanical": row['available_mechanical'],
        "ebike": row['available_electrical'],
        "is_renting": "OUI" if row['operative'] else "NON",
        "is_returning": "OUI" if row['operative'] else "NON",
        "duedate": row['date'].isoformat(),
        "coordonnees_geo": { "lon": lon, "lat": lat },
        "nom_arrondissement_communes": row['nom_arrondissement_communes'],
        "code_insee_commune": row['code_insee_commune']
    }
    return json_data

# Split data into monthly groups
monthly_groups = merged_data.groupby('year_month')

# Process each monthly group and save to JSON files
for period, group in tqdm(monthly_groups, desc="Processing months"):
    month_str = period.strftime('%Y-%m')
    json_list = group.apply(convert_row_to_json, axis=1).dropna().tolist()  # Drop None values
    
    # Write to JSON file
    output_path = os.path.join(output_directory, f'{month_str}.json')
    with open(output_path, 'w') as f:
        json.dump(json_list, f, indent=4)

    # Also save the data in CSV format
    csv_output_path = os.path.join(output_directory, f'{month_str}.csv')
    group.to_csv(csv_output_path, index=False)

print("Data processing complete.")
