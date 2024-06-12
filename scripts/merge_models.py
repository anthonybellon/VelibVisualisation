import os
import pickle

# Function to get the absolute path relative to the script location
def get_absolute_path(relative_path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))

# Directory containing the saved model and scaler batch files
save_directory = get_absolute_path('../data')

# Load all scaler and model batch files
scaler_files = [f for f in os.listdir(save_directory) if f.startswith('scaler_batch_') and f.endswith('.pkl')]
model_files = [f for f in os.listdir(save_directory) if f.startswith('model_batch_') and f.endswith('.pkl')]

# Sort files by batch number
scaler_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
model_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))

# Dictionary to hold all models and scalers
combined_models = {}
combined_scalers = {}

# Load and combine scalers
for scaler_file in scaler_files:
    batch_index = int(scaler_file.split('_')[2].split('.')[0])
    scaler_path = os.path.join(save_directory, scaler_file)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        combined_scalers[batch_index] = scaler  # Use batch_index instead of idx

# Load and combine models
for model_file in model_files:
    batch_index = int(model_file.split('_')[2].split('.')[0])
    model_path = os.path.join(save_directory, model_file)
    with open(model_path, 'rb') as f:
        models = pickle.load(f)
        for station, model in models.items():
            combined_models[station] = {'model': model, 'scaler_idx': batch_index}  # Use batch_index

# Save the combined models and scalers to a single file
combined_file_path = get_absolute_path('../data/combined_models_and_scalers.pkl')
with open(combined_file_path, 'wb') as f:
    pickle.dump({'models': combined_models, 'scalers': combined_scalers}, f)
    print(f"Combined models and scalers saved to {combined_file_path}")
