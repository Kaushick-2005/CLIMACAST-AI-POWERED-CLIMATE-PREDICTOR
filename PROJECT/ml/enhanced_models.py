import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib
import json
import os

class ClimatePredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.model_types = {}
        
    def prepare_data_for_prophet(self, df, variable):
        """Prepare data for Prophet model"""
        df_prophet = df.reset_index()[['date', variable]].rename(columns={'date': 'ds', variable: 'y'})
        df_prophet = df_prophet.dropna()
        return df_prophet
    
    def prepare_data_for_lstm(self, df, variable, lookback=60):
        """Prepare data for LSTM model"""
        data = df[variable].dropna().values.reshape(-1, 1)
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y, scaler
    
    def train_prophet_model(self, df, variable, save_path=None):
        """Train Prophet model for a specific variable"""
        print(f"Training Prophet model for {variable}...")
        
        df_prophet = self.prepare_data_for_prophet(df, variable)
        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )
        model.fit(df_prophet)
        
        self.models[f"{variable}_prophet"] = model
        self.model_types[f"{variable}_prophet"] = "prophet"
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(model_to_json(model))
        
        return model
    
    def train_lstm_model(self, df, variable, lookback=60, epochs=50, batch_size=32, save_path=None):
        """Train LSTM model for a specific variable"""
        print(f"Training LSTM model for {variable}...")
        
        X, y, scaler = self.prepare_data_for_lstm(df, variable, lookback)
        
        # Split data
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Build LSTM model
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(50, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(25),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train model
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)
        
        self.models[f"{variable}_lstm"] = model
        self.scalers[f"{variable}_lstm"] = scaler
        self.model_types[f"{variable}_lstm"] = "lstm"
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            model.save(save_path)
            joblib.dump(scaler, save_path.replace('.h5', '_scaler.pkl'))
        
        return model, scaler
    
    def predict_prophet(self, variable, periods, freq='D'):
        """Make predictions using Prophet model"""
        model_key = f"{variable}_prophet"
        if model_key not in self.models:
            raise ValueError(f"Prophet model for {variable} not found")
        
        model = self.models[model_key]
        future = model.make_future_dataframe(periods=periods, freq=freq)
        forecast = model.predict(future)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    def predict_lstm(self, variable, periods, last_data=None):
        """Make predictions using LSTM model"""
        model_key = f"{variable}_lstm"
        if model_key not in self.models:
            raise ValueError(f"LSTM model for {variable} not found")
        
        model = self.models[model_key]
        scaler = self.scalers[model_key]
        
        if last_data is None:
            raise ValueError("Need recent data for LSTM prediction")
        
        # Use last 60 data points for prediction
        last_60_days = last_data[-60:].values.reshape(-1, 1)
        last_60_days_scaled = scaler.transform(last_60_days)
        
        predictions = []
        current_batch = last_60_days_scaled.reshape(1, 60, 1)
        
        for _ in range(periods):
            current_pred = model.predict(current_batch, verbose=0)[0]
            predictions.append(current_pred[0])
            current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)
        
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        return predictions.flatten()
    
    def load_prophet_model(self, variable, model_path):
        """Load saved Prophet model"""
        with open(model_path, 'r') as f:
            model = model_from_json(f.read())
        
        self.models[f"{variable}_prophet"] = model
        self.model_types[f"{variable}_prophet"] = "prophet"
        
        return model
    
    def load_lstm_model(self, variable, model_path, scaler_path):
        """Load saved LSTM model"""
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
        
        self.models[f"{variable}_lstm"] = model
        self.scalers[f"{variable}_lstm"] = scaler
        self.model_types[f"{variable}_lstm"] = "lstm"
        
        return model, scaler

def train_region_models(region_slug: str, data_path: str, models_root: str, quick: bool = False):
    """Train models for a specific region and save under models_root/region_slug"""
    df = pd.read_csv(data_path, index_col='date', parse_dates=True)
    predictor = ClimatePredictor()
    variables = ['T2M', 'PRECTOTCORR', 'RH2M']
    region_dir = os.path.join(models_root, region_slug)
    os.makedirs(region_dir, exist_ok=True)

    for variable in variables:
        if variable in df.columns:
            print(f"\nTraining models for region={region_slug}, variable={variable}")
            prophet_path = os.path.join(region_dir, f'{variable}_prophet.json')
            predictor.train_prophet_model(df, variable, prophet_path)
            lstm_path = os.path.join(region_dir, f'{variable}_lstm.h5')
            epochs = 5 if quick else 20
            try:
                predictor.train_lstm_model(df, variable, epochs=epochs, save_path=lstm_path)
            except Exception as e:
                print(f"⚠️  LSTM training failed for {region_slug}:{variable} - {e}")

    print(f"\nAll models trained successfully for region {region_slug}!")
    return predictor

if __name__ == '__main__':
    import argparse
    import sys
    # CLI to train models per region
    parser = argparse.ArgumentParser(description='Train models for regions')
    parser.add_argument('--region', type=str, default='nyc', help='Region slug (e.g., nyc, la, ldn, mum, syd)')
    parser.add_argument('--data-path', type=str, default='C:\\HACKATHON\\PROJECT\\data\\preprocessed_nyc_weather_data.csv', help='Path to preprocessed data CSV for the region')
    parser.add_argument('--models-root', type=str, default='C:\\HACKATHON\\PROJECT\\src\\ml\\models', help='Root directory to save models')
    parser.add_argument('--quick', action='store_true', help='Use quick training (fewer epochs)')
    args = parser.parse_args()

    os.makedirs(args.models_root, exist_ok=True)
    train_region_models(args.region, args.data_path, args.models_root, quick=args.quick)
