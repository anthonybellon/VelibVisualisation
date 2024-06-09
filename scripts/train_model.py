import os
import pandas as pd
import numpy as np
import pickle
import json
import logging
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import randint
from tqdm import tqdm
from utils import get_absolute_path, calculate_nearby_station_status_adjustable, create_features, load_feature_names

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(data, stations, target='numbikesavailable'):
    logger.info("Training models...")

    # Create necessary features
    data = create_features(data)
    
    # Normalize features
    scaler = StandardScaler()
    
    # Load feature names from JSON
    feature_names = load_feature_names()
    base_features = feature_names[:12]
    additional_features = feature_names[12:]

    models = {}
    nearby_station_results = {}
    param_dist = {
        'n_estimators': randint(50, 200),
        'max_depth': [10, 20, None],
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 4)
    }

    for station in tqdm(stations, desc="Training models", unit="station"):
        nearby_stations, station_data = calculate_nearby_station_status_adjustable(data, station)
        nearby_station_results[station] = nearby_stations

        if nearby_stations is None or len(nearby_stations) < 5:
            selected_features = base_features
        else:
            station_data = station_data.copy()
            station_data['nearby_stations_closed'] = nearby_stations['is_renting'].apply(lambda x: 1 if x == 'NON' else 0).sum()
            station_data['nearby_stations_full'] = (nearby_stations['numbikesavailable'] == 0).sum()
            station_data['nearby_stations_empty'] = (nearby_stations['numdocksavailable'] == 0).sum()
            station_data['likelihood_fill'] = station_data['nearby_stations_full'] / (len(nearby_stations) + 1e-5)
            station_data['likelihood_empty'] = station_data['nearby_stations_empty'] / (len(nearby_stations) + 1e-5)
            selected_features = base_features + additional_features

        # Ensure only numeric features are selected
        numeric_features = station_data[selected_features].select_dtypes(include=[np.number]).columns.tolist()
        missing_features = [feat for feat in numeric_features if feat not in station_data.columns]
        if missing_features:
            logger.warning(f"Missing features for station {station}: {missing_features}")
            continue

        station_data[numeric_features] = scaler.fit_transform(station_data[numeric_features])

        X = station_data[numeric_features]
        y = station_data[target]

        if len(X) < 2:
            continue

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        n_splits = min(5, len(X_train))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        random_search = RandomizedSearchCV(RandomForestRegressor(random_state=42), param_distributions=param_dist, n_iter=20, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_
        cv_score = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
        logger.info(f"Cross-Validation Score for station {station}: {cv_score.mean()}")
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        logger.info(f"Mean Squared Error for station {station}: {mse}")

        models[station] = best_model

    return models, scaler, nearby_station_results

def save_intermediate_results(models, scalers, nearby_station_results, batch_number):
    model_file_path = get_absolute_path(f'../data/model_batch_{batch_number}.pkl')
    scaler_file_path = get_absolute_path(f'../data/scaler_batch_{batch_number}.pkl')
    nearby_stations_file_path = get_absolute_path(f'../data/nearby_stations_batch_{batch_number}.pkl')

    with open(model_file_path, 'wb') as f:
        pickle.dump(models, f)
    with open(scaler_file_path, 'wb') as f:
        pickle.dump(scalers, f)
    with open(nearby_stations_file_path, 'wb') as f:
        pickle.dump(nearby_station_results, f)
    logger.info(f"Intermediate results saved for batch {batch_number}")

def load_intermediate_results(batch_number):
    model_file_path = get_absolute_path(f'../data/model_batch_{batch_number}.pkl')
    scaler_file_path = get_absolute_path(f'../data/scaler_batch_{batch_number}.pkl')
    nearby_stations_file_path = get_absolute_path(f'../data/nearby_stations_batch_{batch_number}.pkl')

    if os.path.exists(model_file_path) and os.path.exists(scaler_file_path) and os.path.exists(nearby_stations_file_path):
        with open(model_file_path, 'rb') as f:
            models = pickle.load(f)
        with open(scaler_file_path, 'rb') as f:
            scalers = pickle.load(f)
        with open(nearby_stations_file_path, 'rb') as f:
            nearby_station_results = pickle.load(f)
        return models, scalers, nearby_station_results
    return {}, {}, {}

def main():
    processed_data_file_path = get_absolute_path('../data/processed_bike_data.parquet')
    bike_data = pd.read_parquet(processed_data_file_path)
    
    if bike_data.empty:
        raise ValueError("The processed bike data is empty. Please check the Parquet file and its loading process.")
    
    stations = bike_data['stationcode'].unique()[:150]  # Process only the first 150 stations for testing
    batch_size = 100

    all_models = {}
    all_scalers = {}
    all_nearby_station_results = {}

    current_batch = 0
    while os.path.exists(get_absolute_path(f'../data/model_batch_{current_batch}.pkl')):
        models, scalers, nearby_station_results = load_intermediate_results(current_batch)
        all_models.update(models)
        all_scalers.update(scalers)
        all_nearby_station_results.update(nearby_station_results)
        current_batch += 1

    for i in range(current_batch * batch_size, len(stations), batch_size):
        batch_stations = stations[i:i + batch_size]
        models, scaler, nearby_station_results = train_model(bike_data, batch_stations)
        all_models.update(models)
        all_scalers.update({station: scaler for station in batch_stations})
        all_nearby_station_results.update(nearby_station_results)
        
        save_intermediate_results(models, {station: scaler for station in batch_stations}, nearby_station_results, current_batch)
        current_batch += 1

    scaler_file_path = get_absolute_path('../data/scaler_final.pkl')
    model_file_path = get_absolute_path('../data/model_final.pkl')
    nearby_stations_file_path = get_absolute_path('../data/nearby_stations_final.pkl')

    with open(scaler_file_path, 'wb') as f:
        pickle.dump(all_scalers, f)
    logger.info(f"Scaler saved to {scaler_file_path}")

    with open(model_file_path, 'wb') as f:
        pickle.dump(all_models, f)
    logger.info(f"Models saved to {model_file_path}")

    with open(nearby_stations_file_path, 'wb') as f:
        pickle.dump(all_nearby_station_results, f)
    logger.info(f"Nearby station results saved to {nearby_stations_file_path}")

    logger.info("Model training complete.")

if __name__ == "__main__":
    main()
