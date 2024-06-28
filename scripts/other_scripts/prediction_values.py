import json
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# Load data from JSON file

def get_absolute_path(relative_path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))

# Get the absolute path of the JSON file
json_file_path = get_absolute_path('../../data/prediction_results_final.json')

with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Extract predicted and actual values
predicted = []
actual = []

for entry in data:
    predicted.append(entry['predicted_bikesavailable'])
    actual.append(entry['actual_bikesavailable'])

# Convert lists to numpy arrays for better performance
predicted = np.array(predicted)
actual = np.array(actual)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(actual, predicted))

# Calculate MAE
mae = mean_absolute_error(actual, predicted)

# Calculate R-squared
r2 = r2_score(actual, predicted)

# Print results
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R²: {r2}")

# Additional analysis if needed
# For example, residuals analysis
residuals = actual - predicted

print(f"Residuals: {residuals}")
print(f"Mean of residuals: {np.mean(residuals)}")
print(f"Standard deviation of residuals: {np.std(residuals)}")
