import os
import json

def get_absolute_path(relative_path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def save_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Data saved to {file_path}")

def concat_json_files(file1, file2, output_file):
    data1 = load_json(get_absolute_path(file1))
    data2 = load_json(get_absolute_path(file2))
    
    # Assuming data1 and data2 are lists of dictionaries
    concatenated_data = data1 + data2
    
    save_json(concatenated_data, get_absolute_path(output_file))

if __name__ == "__main__":
    file1 = '../data/2024-05-to-concat.json'
    file2 = '../data/2021-04-to-concat.json'
    output_file = '../data/use_for_predictions.json'
    
    concat_json_files(file1, file2, output_file)
