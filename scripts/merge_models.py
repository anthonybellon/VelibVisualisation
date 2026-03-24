import os
import pickle


# Function to get the absolute path relative to the script location
def get_absolute_path(relative_path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))


# Directory containing the saved model batch files
save_directory = get_absolute_path("../data")

# Load all model batch files
model_files = [
    f for f in os.listdir(save_directory) if f.startswith("model_batch_") and f.endswith(".pkl")
]

# Sort files by batch number
model_files.sort(key=lambda x: int(x.split("_")[2].split(".")[0]))

# Dictionary to hold all models
combined_models = {}

# Load and combine models
for model_file in model_files:
    model_path = os.path.join(save_directory, model_file)
    with open(model_path, "rb") as f:
        models = pickle.load(f)
        for station, model in models.items():
            combined_models[station] = model

# Save the combined models to a single file
combined_file_path = get_absolute_path("../data/combined_models_and_scalers.pkl")
with open(combined_file_path, "wb") as f:
    pickle.dump({"models": combined_models}, f)
    print(f"Combined models saved to {combined_file_path}")
