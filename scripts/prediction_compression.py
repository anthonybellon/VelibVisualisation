import json
import os

extra_capacity_stations = ["4005", "4104", "8002", "8004", "9104", "12105", "13123", "15056", "21302", "32012", "42004", "4010", "4017", "12010", "15058", "15122", "18043", "19018", "21021", "33019"]

def calculate_trend(previous, next, position):
    return previous + (next - previous) * position

def get_absolute_path(relative_path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))

def process_data(data):
    processed_data = {}
    total_missing_predictions = 0
    for day in range(7):
        hour_values = [None] * 24

        for hour in range(24):
            hour_values[hour] = data.get(str(day), {}).get(str(hour), None)
            if hour_values[hour] is not None and isinstance(hour_values[hour], list):
                hour_values[hour] = sum(hour_values[hour]) / len(hour_values[hour])
        
        total_missing_predictions += hour_values.count(None)
        
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
        
        if all(hour is None for hour in hour_values):
            hour_values = [0] * 24

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

data_path = get_absolute_path('../data/3_predictions_results_updated.json')

with open(data_path, 'r') as file:
    data = json.load(file)

processed_data_normal = []
processed_data_extra = []

for item in data:
    stationcode = item['stationcode']
    normal_capacity = item['capacity']
    extra_capacity = item['capacity'] * 2 if stationcode in extra_capacity_stations else item['capacity']
    
    station_normal = next((s for s in processed_data_normal if s["stationcode"] == stationcode), None)
    if not station_normal:
        station_normal = {
            "stationcode": stationcode,
            "name": item["name"],
            "capacity": normal_capacity,
            "is_renting": item["is_renting"],
            "coordonnees_geo": item["coordonnees_geo"],
            "missing_predictions": 0,
            "predictions": {},
        }
        processed_data_normal.append(station_normal)
    
    station_extra = next((s for s in processed_data_extra if s["stationcode"] == stationcode), None)
    if not station_extra:
        station_extra = {
            "stationcode": stationcode,
            "name": item["name"],
            "capacity": extra_capacity,
            "is_renting": item["is_renting"],
            "coordonnees_geo": item["coordonnees_geo"],
            "missing_predictions": 0,
            "predictions": {},
            "extra_capacity_predictions": {}
        }
        processed_data_extra.append(station_extra)
    
    day = item['day_of_week_unscaled']
    hour = item['hour_unscaled']
    if str(day) not in station_normal["predictions"]:
        station_normal["predictions"][str(day)] = {}
    if str(day) not in station_extra["predictions"]:
        station_extra["predictions"][str(day)] = {}
    if str(hour) not in station_normal["predictions"][str(day)]:
        station_normal["predictions"][str(day)][str(hour)] = []
    if str(hour) not in station_extra["predictions"][str(day)]:
        station_extra["predictions"][str(day)][str(hour)] = []

    station_normal["predictions"][str(day)][str(hour)].append(item["predicted_bikesavailable"])
    station_extra["predictions"][str(day)][str(hour)].append(item["predicted_bikesavailable"])

for station in processed_data_normal:
    station['predictions'], missing_count = process_data(station['predictions'])
    station['missing_predictions'] = missing_count

for station in processed_data_extra:
    station['predictions'], missing_count = process_data(station['predictions'])
    station['missing_predictions'] = missing_count
    station['extra_capacity_predictions'] = {k: v[:] for k, v in station['predictions'].items()}  # Copy predictions for extra capacity processing

rounded_data_normal = round_predictions(processed_data_normal)
rounded_data_extra = round_predictions(processed_data_extra)

percentage_data_normal = compute_percentage(rounded_data_normal)
percentage_data_extra = compute_percentage(rounded_data_extra)

output_data = {
    "normal_capacity": percentage_data_normal,
    "extra_capacity": percentage_data_extra
}

output_path = get_absolute_path('../data/4_compressed_predictions_final_fix.json')

with open(output_path, 'w') as file:
    json.dump(output_data, file, indent=4)

print(f"Data processing complete. Processed and rounded data saved to {output_path}.")

validate_file(output_path)
