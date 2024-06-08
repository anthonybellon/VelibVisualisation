import pandas as pd
import os
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Function to get the absolute path relative to the script location
def get_absolute_path(relative_path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))

# Directory containing the cleaned JSON files
clean_data_directory = get_absolute_path('../data/historical_data_cleaned')

# Load and concatenate JSON files
print("Loading cleaned bike data...")
all_data = []
json_files = [f for f in os.listdir(clean_data_directory) if f.endswith('.json')]
for filename in tqdm(json_files, desc="Loading JSON files"):
    file_path = os.path.join(clean_data_directory, filename)
    with open(file_path, 'r') as f:
        data = json.load(f)
        all_data.extend(data)

# Convert to DataFrame
bike_data = pd.DataFrame(all_data)

# Check if the dataframe is empty
if bike_data.empty:
    raise ValueError("The concatenated bike data is empty. Please check the JSON files and their loading process.")

# Remove station codes with only one data point
station_counts = bike_data['stationcode'].value_counts()
stations_to_keep = station_counts[station_counts > 1].index
bike_data = bike_data[bike_data['stationcode'].isin(stations_to_keep)]

# Preprocess bike data
print("Preprocessing bike data...")

# Convert date chunks
bike_data['date'] = pd.to_datetime(bike_data['duedate'], errors='coerce')
if bike_data['date'].dt.tz is None:
    bike_data['date'] = bike_data['date'].dt.tz_localize('UTC')

# Extract latitude and longitude
print("Extracting latitude and longitude...")
bike_data['lat'] = bike_data['coordonnees_geo'].apply(lambda x: x.get('lat', np.nan) if x else np.nan)
bike_data['lon'] = bike_data['coordonnees_geo'].apply(lambda x: x.get('lon', np.nan) if x else np.nan)

# Update specific coordinates
station_coord_updates = {
    "22504": {"lon": 2.253629, "lat": 48.905928},
    "25006": {"lon": 2.1961666225454, "lat": 48.862453313908},
    "10001": {"lon": 2.3600032, "lat": 48.8685433},
    "10001_relais": {"lon": 2.3599605, "lat": 48.8687079}
}

# Update coordinates for the specified stations
for station_code, new_coords in station_coord_updates.items():
    bike_data.loc[bike_data['stationcode'] == station_code, ['lat', 'lon']] = new_coords['lat'], new_coords['lon']

# Remove 'code_insee_commune' field
bike_data = bike_data.drop(columns=['code_insee_commune'], errors='ignore')

# Identify unique station code duplicates based on lat and lon
bike_data_unique = bike_data.drop_duplicates(subset=['stationcode', 'lat', 'lon'])

# Find station codes with the same coordinates
duplicate_coords = bike_data_unique[bike_data_unique.duplicated(['lat', 'lon'], keep=False)]
print(f"Stations with duplicate coordinates found:\n{duplicate_coords}")

# Group by coordinates to see how many unique stations share the same coordinates
duplicate_groups = duplicate_coords.groupby(['lat', 'lon']).size().reset_index(name='count')
print(f"Duplicate coordinate groups:\n{duplicate_groups}")

# Convert duplicates to a dictionary for JSON serialization
duplicates_dict = duplicate_coords.to_dict(orient='records')

# Handle Timestamp serialization
def convert_timestamps(item):
    for key, value in item.items():
        if isinstance(value, pd.Timestamp):
            item[key] = value.isoformat()
    return item

duplicates_dict = [convert_timestamps(item) for item in duplicates_dict]

# Save duplicates to JSON file
duplicates_file_path = get_absolute_path('../data/duplicate_coordinates.json')
with open(duplicates_file_path, 'w') as f:
    json.dump(duplicates_dict, f, indent=4)
    print(f"Duplicate coordinates data saved to {duplicates_file_path}")

# Handle missing values by filling or dropping
bike_data.dropna(subset=['lat', 'lon'], inplace=True)

# Remove any NaN values
print("Removing NaN values...")
bike_data.fillna(0, inplace=True)

# Extract 'hour' and 'day_of_week' from 'date' column
print("Extracting 'hour' and 'day_of_week' from 'date' column...")
bike_data['hour'] = bike_data['date'].dt.hour
bike_data['day_of_week'] = bike_data['date'].dt.dayofweek

# Optimize data types
print("Optimizing data types...")
bike_data['lat'] = bike_data['lat'].astype('float32')
bike_data['lon'] = bike_data['lon'].astype('float32')
bike_data['numbikesavailable'] = bike_data['numbikesavailable'].astype('int16')
bike_data['numdocksavailable'] = bike_data['numdocksavailable'].astype('int16')
bike_data['capacity'] = bike_data['capacity'].astype('int16')
bike_data['hour'] = bike_data['hour'].astype('int8')
bike_data['day_of_week'] = bike_data['day_of_week'].astype('int8')

# Convert date column to datetime
print("Converting date column to datetime...")
bike_data['date'] = pd.to_datetime(bike_data['date'])

# Select only numeric columns for correlation matrix
numeric_cols = bike_data.select_dtypes(include=[np.number])

# Check for potential data leakage by inspecting feature correlations
plt.figure(figsize=(12, 8))
correlation_matrix = numeric_cols.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Feature Correlation Matrix')
plt.show()

# Save the preprocessed data as a Parquet file
processed_data_file_path = get_absolute_path('../data/processed_bike_data.parquet')

# Validate DataFrame before saving
print("Validating DataFrame before saving to Parquet...")
print(bike_data.info())
print(bike_data.head())

# Ensure pyarrow is used as the engine
try:
    bike_data.to_parquet(processed_data_file_path, engine='pyarrow', index=False)
    print(f"Data saved to {processed_data_file_path}")
except Exception as e:
    print(f"Failed to save Parquet file: {e}")
