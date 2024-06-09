import os
import pickle
from utils import get_absolute_path

def merge_models_and_scalers():
    all_models = {}
    all_scalers = {}
    all_nearby_station_results = {}

    current_batch = 0
    while os.path.exists(get_absolute_path(f'../data/model_batch_{current_batch}.pkl')):
        model_file_path = get_absolute_path(f'../data/model_batch_{current_batch}.pkl')
        scaler_file_path = get_absolute_path(f'../data/scaler_batch_{current_batch}.pkl')
        nearby_stations_file_path = get_absolute_path(f'../data/nearby_stations_batch_{current_batch}.pkl')

        with open(model_file_path, 'rb') as f:
            models = pickle.load(f)
        with open(scaler_file_path, 'rb') as f:
            scalers = pickle.load(f)
        with open(nearby_stations_file_path, 'rb') as f:
            nearby_station_results = pickle.load(f)

        all_models.update(models)
        all_scalers.update(scalers)
        all_nearby_station_results.update(nearby_station_results)

        current_batch += 1

    model_file_path = get_absolute_path('../data/model_final.pkl')
    scaler_file_path = get_absolute_path('../data/scaler_final.pkl')
    nearby_stations_file_path = get_absolute_path('../data/nearby_stations_final.pkl')

    with open(model_file_path, 'wb') as f:
        pickle.dump(all_models, f)
    print(f"Final models saved to {model_file_path}")

    with open(scaler_file_path, 'wb') as f:
        pickle.dump(all_scalers, f)
    print(f"Final scalers saved to {scaler_file_path}")

    with open(nearby_stations_file_path, 'wb') as f:
        pickle.dump(all_nearby_station_results, f)
    print(f"Final nearby station results saved to {nearby_stations_file_path}")

if __name__ == "__main__":
    merge_models_and_scalers()
