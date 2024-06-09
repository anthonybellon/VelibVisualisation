import requests
import json
import time
import os

API_URL = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/velib-disponibilite-en-temps-reel/records"
OUTPUT_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/velib_data_single_time.json'))

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
            records = data.get('records', [])
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

    # Debugging statement to show data before saving
    print(f"Total records fetched: {len(all_stations_data)}")

    # Write data to file
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_stations_data, f, indent=4)
        print(f"Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    fetch_data()
