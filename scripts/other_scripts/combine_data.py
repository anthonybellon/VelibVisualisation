import os
import json

def get_absolute_path(relative_path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))

def remove_keys(data, keys_to_remove):
    if isinstance(data, dict):
        return {key: remove_keys(value, keys_to_remove) for key, value in data.items() if key not in keys_to_remove}
    elif isinstance(data, list):
        return [remove_keys(item, keys_to_remove) for item in data]
    else:
        return data

def simplify_predictions(predictions_data):
    simplified_predictions = {}
    for station_code, days_data in predictions_data.items():
        simplified_predictions[station_code] = {}
        for day, hours_data in days_data.items():
            daily_predictions = [None] * 24  # Initialize with None for 24 hours
            for hour, predictions in hours_data.items():
                hour_index = int(hour)
                daily_predictions[hour_index] = predictions[0] if predictions else None
            simplified_predictions[station_code][day] = daily_predictions
    return simplified_predictions

def convert_general_data_to_dict(general_data):
    general_dict = {}
    for station in general_data:
        station_code = station["stationcode"]
        general_dict[station_code] = station
    return general_dict

def combine_data(general_data, predictions_data):
    combined_data = {}
    for station_code, station_info in general_data.items():
        combined_data[station_code] = {
            "name": station_info["name"],
            "capacity": station_info["capacity"],
            "is_renting": station_info.get("is_renting"),      # Use .get() to handle missing keys
            "coordonnees_geo": station_info["coordonnees_geo"],
            "predictions": predictions_data.get(station_code, {})
        }
    return combined_data

def main():
    general_file_path = get_absolute_path('../data/velib_data_single_time.json')  # Change this to your actual general data file path
    predictions_file_path = get_absolute_path('../data/compressed_predictions.json')  # Change this to your actual predictions data file path
    output_file_path = get_absolute_path('../data/combined_data.json')  # Change this to your desired output file path


    keys_to_remove = [
        "is_installed",  
        "numdocksavailable", 
        "numbikesavailable", 
        "mechanical", 
        "ebike",
        "is_returning",
        "duedate",
        "nom_arrondissement_communes",
        "code_insee_commune"
    ]

    with open(general_file_path, 'r', encoding='utf-8') as f:
        general_data = json.load(f)

    with open(predictions_file_path, 'r', encoding='utf-8') as f:
        predictions_data = json.load(f)

    general_dict = convert_general_data_to_dict(general_data)
    cleaned_general_data = remove_keys(general_dict, keys_to_remove)
    simplified_predictions = simplify_predictions(predictions_data)
    combined_data = combine_data(cleaned_general_data, simplified_predictions)

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
