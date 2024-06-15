import requests
import json
import os
import time
from datetime import datetime

JSON_URL = "https://opendata.paris.fr/explore/dataset/velib-disponibilite-en-temps-reel/download/?format=json&timezone=Europe/Berlin"
OUTPUT_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/do_not_touch/weekly_velib_data/weekly_velib_data.json'))
DAILY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/do_not_touch/daily_velib_data/daily'))

def get_absolute_path(relative_path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))

def transform_record(record):
    if 'fields' in record:
        fields = record['fields']
        transformed = {
            "stationcode": fields.get("stationcode"),
            "name": fields.get("name"),
            "is_installed": fields.get("is_installed"),
            "capacity": fields.get("capacity"),
            "numdocksavailable": fields.get("numdocksavailable"),
            "numbikesavailable": fields.get("numbikesavailable"),
            "mechanical": fields.get("mechanical"),
            "ebike": fields.get("ebike"),
            "is_renting": fields.get("is_renting"),
            "is_returning": fields.get("is_returning"),
            "duedate": fields.get("duedate"),
            "coordonnees_geo": {
                "lon": fields["coordonnees_geo"][1],
                "lat": fields["coordonnees_geo"][0]
            } if fields.get("coordonnees_geo") else None,
            "nom_arrondissement_communes": fields.get("nom_arrondissement_communes"),
            "code_insee_commune": None
        }
        return transformed
    else:
        return record

def fetch_data():
    response = requests.get(JSON_URL)
    
    if response.status_code != 200:
        print(f"Failed to fetch data: {response.status_code}")
        print(response.text)
        return
    
    try:
        data = response.json()
    except json.JSONDecodeError:
        print("Failed to decode JSON response")
        print(response.text)
        return

    transformed_records = [transform_record(record) for record in data]

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    os.makedirs(DAILY_DIR, exist_ok=True)

    # Read existing data
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    # Append new data to existing data
    combined_data = existing_data + transformed_records

    # Debugging statement to show data before saving
    print(f"Total records after combining: {len(combined_data)}")

    # Write combined data to file
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=4)
        print(f"Data saved to {OUTPUT_FILE}")

    # Split data by day of the week and save to separate files
    daily_data = {i: [] for i in range(7)}
    for record in transformed_records:
        duedate = record.get('duedate')
        if duedate:
            day_of_week = datetime.strptime(duedate, '%Y-%m-%dT%H:%M:%S%z').weekday()
            daily_data[day_of_week].append(record)

    for day in range(7):
        daily_file = os.path.join(DAILY_DIR, f'day_{day}.json')
        if os.path.exists(daily_file):
            with open(daily_file, 'r') as f:
                try:
                    existing_daily_data = json.load(f)
                except json.JSONDecodeError:
                    existing_daily_data = []
        else:
            existing_daily_data = []

        combined_daily_data = existing_daily_data + daily_data[day]

        with open(daily_file, 'w') as f:
            json.dump(combined_daily_data, f, ensure_ascii=False, indent=4)
            print(f"Data for day {day} saved to {daily_file}")

if __name__ == "__main__":
    while True:
        fetch_data()
        time.sleep(1800)  # Wait for 30 minutes before fetching data again
