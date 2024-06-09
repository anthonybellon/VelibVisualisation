import pandas as pd
import os
import json
from tqdm import tqdm
import numpy as np
from utils import get_absolute_path, create_features, report_infinite_and_large_values, save_feature_names

def preprocess_data():
    clean_data_directory = get_absolute_path('../data/historical_data_cleaned')

    print("Loading cleaned bike data...")
    all_data = []
    json_files = [f for f in os.listdir(clean_data_directory) if f.endswith('.json')]
    for filename in tqdm(json_files, desc="Loading JSON files"):
        file_path = os.path.join(clean_data_directory, filename)
        with open(file_path, 'r') as f:
            data = json.load(f)
            all_data.extend(data)

    bike_data = pd.DataFrame(all_data)

    if bike_data.empty:
        raise ValueError("The concatenated bike data is empty. Please check the JSON files and their loading process.")

    station_counts = bike_data['stationcode'].value_counts()
    stations_to_keep = station_counts[station_counts > 1].index
    bike_data = bike_data[bike_data['stationcode'].isin(stations_to_keep)]

    print("Preprocessing bike data...")

    bike_data['date'] = pd.to_datetime(bike_data['duedate'], errors='coerce')
    if bike_data['date'].dt.tz is None:
        bike_data['date'] = bike_data['date'].dt.tz_localize('UTC')

    print("Extracting latitude and longitude...")
    bike_data['lat'] = bike_data['coordonnees_geo'].apply(lambda x: x.get('lat', np.nan) if x else np.nan).astype('float32')
    bike_data['lon'] = bike_data['coordonnees_geo'].apply(lambda x: x.get('lon', np.nan) if x else np.nan).astype('float32')

    station_coord_updates = {
        "22504": {"lon": 2.253629, "lat": 48.905928},
        "25006": {"lon": 2.1961666225454, "lat": 48.862453313908},
        "10001": {"lon": 2.3600032, "lat": 48.8685433},
        "10001_relais": {"lon": 2.3599605, "lat": 48.8687079}
    }

    for station_code, new_coords in station_coord_updates.items():
        bike_data.loc[bike_data['stationcode'] == station_code, ['lat', 'lon']] = (
            np.float32(new_coords['lat']),
            np.float32(new_coords['lon'])
        )

    bike_data = bike_data.drop(columns=['code_insee_commune'], errors='ignore')

    bike_data_unique = bike_data.drop_duplicates(subset=['stationcode', 'lat', 'lon'])
    duplicate_coords = bike_data_unique[bike_data_unique.duplicated(['lat', 'lon'], keep=False)]
    print(f"Stations with duplicate coordinates found:\n{duplicate_coords}")

    duplicate_groups = duplicate_coords.groupby(['lat', 'lon']).size().reset_index(name='count')
    print(f"Duplicate coordinate groups:\n{duplicate_groups}")

    duplicates_dict = duplicate_coords.to_dict(orient='records')

    def convert_timestamps(item):
        for key, value in item.items():
            if isinstance(value, pd.Timestamp):
                item[key] = value.isoformat()
        return item

    duplicates_dict = [convert_timestamps(item) for item in duplicates_dict]

    duplicates_file_path = get_absolute_path('../data/duplicate_coordinates.json')
    with open(duplicates_file_path, 'w') as f:
        json.dump(duplicates_dict, f, indent=4)
        print(f"Duplicate coordinates data saved to {duplicates_file_path}")

    bike_data.dropna(subset=['lat', 'lon'], inplace=True)
    print("Removing NaN values...")
    bike_data.fillna(0, inplace=True)

    report_infinite_and_large_values(bike_data)

    # Create features before saving
    bike_data = create_features(bike_data)

    # Save feature names
    feature_names = bike_data.columns.tolist()
    save_feature_names(feature_names)

    processed_data_file_path = get_absolute_path('../data/processed_bike_data.parquet')
    bike_data.to_parquet(processed_data_file_path)
    print(f"Data saved to {processed_data_file_path}")

if __name__ == "__main__":
    preprocess_data()
