import requests
import json
import time
import os
from datetime import datetime

API_URL = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/velib-disponibilite-en-temps-reel/records"
OUTPUT_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data_to_clean/velib_data.json'))
DAILY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/daily'))

def fetch_data():
    all_stations_data = []
    start = 0
    limit = 100
    total_stations = 1471 

    while start < total_stations:
        response = requests.get(API_URL, params={"start": start, "limit": limit})
        
        if response.status_code != 200:
            print(f"Failed to fetch data: {response.status_code}")
            print(response.text)
            break
        
        try:
            data = response.json()
            records = data.get('results', [])
            if not records:
                break
            all_stations_data.extend(records)
            start += limit
            print(f"Fetched {len(records)} records. Total so far: {len(all_stations_data)}")
            time.sleep(1)  # Respect API rate limit of 1 call per second
        except json.JSONDecodeError:
            print("Failed to decode JSON response")
            print(response.text)
            break
        except KeyError:
            print("Unexpected JSON structure")
            print(data)
            break

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
    combined_data = existing_data + all_stations_data

    # Debugging statement to show data before saving
    print(f"Total records after combining: {len(combined_data)}")

    # Write combined data to file
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(combined_data, f)
        print(f"Data saved to {OUTPUT_FILE}")

    # Split data by day of the week and save to separate files
    daily_data = {i: [] for i in range(7)}
    for record in all_stations_data:
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
            json.dump(combined_daily_data, f)
            print(f"Data for day {day} saved to {daily_file}")

if __name__ == "__main__":
    while True:
        fetch_data()
        time.sleep(1800)  # Wait for 30 minutes before fetching data again
