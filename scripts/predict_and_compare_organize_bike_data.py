import pandas as pd
import os
import json

# Function to get the absolute path relative to the script location
def get_absolute_path(relative_path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))

# Directory containing the original JSON file
data_directory = get_absolute_path('../data')
input_file_path = os.path.join(data_directory, '1_use_for_predictions.json')
output_file_path = os.path.join(data_directory, '2_organized_predictions.json')

# Load the data
print("Loading bike data...")
with open(input_file_path, 'r') as f:
    data = json.load(f)

# Convert to DataFrame
bike_data = pd.DataFrame(data)

# Convert 'duedate' to datetime
bike_data['duedate'] = pd.to_datetime(bike_data['duedate'])

# Extract additional time-related features
bike_data['hour'] = bike_data['duedate'].dt.hour
bike_data['day_of_week'] = bike_data['duedate'].dt.dayofweek

# Function to keep the most recent record for each station, day of the week, and hour
def get_most_recent(df):
    return df.loc[df['duedate'].idxmax()]

# Apply the function
print("Organizing data to keep the most recent record for each station, day of the week, and hour...")
grouped = bike_data.groupby(['stationcode', 'day_of_week', 'hour'], group_keys=False).apply(get_most_recent).reset_index(drop=True)

# Convert 'duedate' back to string format for JSON serialization
grouped['duedate'] = grouped['duedate'].dt.strftime('%Y-%m-%dT%H:%M:%S%z')

# Convert the organized data back to JSON format
organized_data_dict = grouped.to_dict(orient='records')

# Save the organized data to a new JSON file
with open(output_file_path, 'w') as f:
    json.dump(organized_data_dict, f, indent=4)

print(f"Organized data saved to {output_file_path}")
