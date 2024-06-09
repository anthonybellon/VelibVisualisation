import os
import pickle

def get_absolute_path(relative_path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))

# Function to load pickle file with error handling
def load_pickle(file_path):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"Error loading pickle file {file_path}: {e}")
            return None
    else:
        print(f"File {file_path} does not exist or is empty.")
        return None

# Function to save pickle file
def save_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {file_path}")

# Load all batch models and scalers
intermediate_models = {}
intermediate_scalers = {}
batch_number = 0

while True:
    model_file_path = get_absolute_path(f'../data/model_batch_{batch_number}.pkl')
    scaler_file_path = get_absolute_path(f'../data/scaler_batch_{batch_number}.pkl')
    
    if not os.path.exists(model_file_path) or not os.path.exists(scaler_file_path):
        break

    models = load_pickle(model_file_path)
    scalers = load_pickle(scaler_file_path)
    
    if models is not None:
        intermediate_models.update(models)
    if scalers is not None:
        intermediate_scalers.update(scalers)
    
    batch_number += 1

# Save the combined models and scalers to new final files
final_model_path = get_absolute_path('../data/model_final_new.pkl')
final_scaler_path = get_absolute_path('../data/scaler_final_new.pkl')

save_pickle(intermediate_models, final_model_path)
save_pickle(intermediate_scalers, final_scaler_path)

print(f"Number of combined models: {len(intermediate_models)}")
print(f"Number of combined scalers: {len(intermediate_scalers)}")
