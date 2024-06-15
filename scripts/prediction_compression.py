import json
import os

def calculate_trend(previous, next, position):
    return previous + (next - previous) * position

def get_absolute_path(relative_path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))

def process_data(data):
    processed_data = {}
    total_missing_predictions = 0
    for day in range(7):  # There are 7 days in a week
        hour_values = [None] * 24  # Initialize all hours with None

        for hour in range(24):
            hour_values[hour] = data.get(str(day), {}).get(str(hour), None)
            # If there are multiple values, take the average
            if hour_values[hour] is not None and isinstance(hour_values[hour], list):
                hour_values[hour] = sum(hour_values[hour]) / len(hour_values[hour])
        
        # Count missing values before filling them
        total_missing_predictions += hour_values.count(None)
        
        # Fill in missing values based on the trends of previous and next hours
        for i in range(24):
            if hour_values[i] is None:
                prev_hour = next((j for j in range(i-1, -1, -1) if hour_values[j] is not None), None)
                next_hour = next((j for j in range(i+1, 24) if hour_values[j] is not None), None)
                if prev_hour is not None and next_hour is not None:
                    position = (i - prev_hour) / (next_hour - prev_hour)
                    hour_values[i] = calculate_trend(hour_values[prev_hour], hour_values[next_hour], position)
                elif prev_hour is not None:
                    hour_values[i] = hour_values[prev_hour]
                elif next_hour is not None:
                    hour_values[i] = hour_values[next_hour]
        
        # If all hours for the day are missing, set default value
        if all(hour is None for hour in hour_values):
            hour_values = [0] * 24

        # Store the processed values back into the dictionary
        processed_data[day] = hour_values
    
    return processed_data, total_missing_predictions

def round_predictions(data):
    for station in data:
        for day, hours in station["predictions"].items():
            station["predictions"][day] = [round(hour) if hour is not None else 0 for hour in hours]
    return data

def compute_percentage(data):
    for station in data:
        capacity = station["capacity"]
        for day, hours in station["predictions"].items():
            station["predictions"][day] = [(round((hour / capacity) * 100) if capacity > 0 else 0) for hour in hours]
    return data

def validate_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
            print(f"File '{file_path}' read successfully. Number of stations processed: {len(data)}")
    else:
        print(f"File '{file_path}' does not exist.")

# Read data from compressed_predictions_final.json
data_path = get_absolute_path('../data/3_predictions_results.json')

with open(data_path, 'r') as file:
    data = json.load(file)

# Transform the data into the required format
processed_data = []
for item in data:
    stationcode = item['stationcode']
    station = next((s for s in processed_data if s["stationcode"] == stationcode), None)
    if not station:
        station = {
            "stationcode": stationcode,
            "name": item["name"],
            "capacity": item["capacity"],
            "is_renting": item["is_renting"],
            "coordonnees_geo": item["coordonnees_geo"],
            "missing_predictions": 0,
            "predictions": {},
        }
        processed_data.append(station)
    
    day = item['day_of_week_unscaled']
    hour = item['hour_unscaled']
    if str(day) not in station["predictions"]:
        station["predictions"][str(day)] = {}
    if str(hour) not in station["predictions"][str(day)]:
        station["predictions"][str(day)][str(hour)] = []

    station["predictions"][str(day)][str(hour)].append(item["predicted_bikesavailable"])

# Process the data for each station
for station in processed_data:
    station['predictions'], missing_count = process_data(station['predictions'])
    station['missing_predictions'] = missing_count

# Round the predictions to the nearest integer
rounded_data = round_predictions(processed_data)

# Compute the percentage of fullness
percentage_data = compute_percentage(rounded_data)

# Define the output path
output_path = get_absolute_path('../data/4_compressed_predictions_final.json')

# Save the processed and rounded data
with open(output_path, 'w') as file:
    json.dump(percentage_data, file, indent=4)

print(f"Data processing complete. Processed and rounded data saved to {output_path}.")

# Validate the saved file
validate_file(output_path)
