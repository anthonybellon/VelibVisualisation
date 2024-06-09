import json
import numpy as np

def calculate_trend(previous, next, position):
    return previous + (next - previous) * position

def process_data(data):
    processed_data = {}
    for day, hours in data.items():
        processed_data[day] = {}
        hour_values = []
        
        for hour in range(24):
            hour_str = str(hour)
            if hour_str in hours:
                # Average the values if there are multiple entries for an hour
                values = hours[hour_str]
                avg_value = np.mean(values)
            else:
                # If the hour is missing, we will fill it later based on trends
                avg_value = None
            hour_values.append(avg_value)

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

        # Store the processed values back into the dictionary
        for hour in range(24):
            processed_data[day][str(hour)] = [hour_values[hour]]
    
    return processed_data

# Read data from compressed_predictions.json
with open('compressed_predictions.json', 'r') as file:
    data = json.load(file)

# Process the data
processed_data = {}
for relais, days in data.items():
    processed_data[relais] = process_data(days)

# Save the processed data to processed_predictions.json
with open('processed_predictions.json', 'w') as file:
    json.dump(processed_data, file, indent=4)

print("Data processing complete. Processed data saved to processed_predictions.json.")
