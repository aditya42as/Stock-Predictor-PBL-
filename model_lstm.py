#LSTM Module
import os
import joblib
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense

MODELS_DIR = "data/models"
os.makedirs(MODELS_DIR, exist_ok=True)

def build_lstm_model(n_weeks):
    model = Sequential()
    model.add(LSTM(50, input_shape=(n_weeks,1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model

def load_trained(n_weeks=8):
    model_path = os.path.join(MODELS_DIR, f"lstm_{n_weeks}.h5")
    scaler_path = os.path.join(MODELS_DIR, f"scaler_{n_weeks}.joblib")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    model = load_model(model_path, compile=False)  
    scaler = joblib.load(scaler_path)
    return model, scaler
