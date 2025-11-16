# lstm module

import os
import json
import joblib
import numpy as np
from datetime import datetime
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler

MODELS_DIR = "data/models"
os.makedirs(MODELS_DIR, exist_ok=True)


def _meta_path(n_weeks):
    return os.path.join(MODELS_DIR, f"lstm_{n_weeks}_meta.json")


def build_lstm_model(n_weeks, n_features=1, units=50, dropout=0.0):
    model = Sequential()
    model.add(LSTM(units, input_shape=(n_weeks, n_features)))
    if dropout and dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model


def load_trained(n_weeks=8):
    """Return (model, scaler, meta) or (None, None, None) if missing."""
    model_path = os.path.join(MODELS_DIR, f"lstm_{n_weeks}.h5")
    scaler_path = os.path.join(MODELS_DIR, f"scaler_{n_weeks}.joblib")
    meta_path = _meta_path(n_weeks)

    if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(meta_path):
        return None, None, None

    model = load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    return model, scaler, meta


def _ensure_numpy_1d(arr):
    if isinstance(arr, (list, tuple)):
        arr = np.array(arr)
    if hasattr(arr, "values"):
        arr = arr.values
    arr = np.asarray(arr, dtype=float)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return arr


def preprocess_data(data, n_weeks):
    """Prepare X, y and scaler from 1D numeric input."""
    data = _ensure_numpy_1d(data)

    if len(data) <= n_weeks:
        raise ValueError(f"Not enough data points: need > {n_weeks}, got {len(data)}")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))

    X, y = [], []
    for i in range(n_weeks, len(scaled_data)):
        X.append(scaled_data[i - n_weeks:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    return X, y, scaler


def train_and_save_lstm(
    data,
    n_weeks=8,
    epochs=50,
    batch_size=16,
    validation_split=0.1,
    units=50,
    dropout=0.0,
    patience=6,
    verbose=0
):
    """Train LSTM and save model, scaler, metadata."""
    X, y, scaler = preprocess_data(data, n_weeks)

    batch_size = min(batch_size, max(1, len(X)))

    model = build_lstm_model(n_weeks, n_features=1, units=units, dropout=dropout)

    tmp_path = os.path.join(MODELS_DIR, f"lstm_{n_weeks}_best_tmp.h5")
    final_path = os.path.join(MODELS_DIR, f"lstm_{n_weeks}.h5")
    scaler_path = os.path.join(MODELS_DIR, f"scaler_{n_weeks}.joblib")
    meta_path = _meta_path(n_weeks)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        ModelCheckpoint(tmp_path, monitor="val_loss", save_best_only=True)
    ]

    history = model.fit(
        X,
        y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        shuffle=False,
        verbose=verbose
    )

    
    if os.path.exists(tmp_path):
        os.replace(tmp_path, final_path)
        model = load_model(final_path, compile=False)
    else:
        model.save(final_path)

    joblib.dump(scaler, scaler_path)

    meta = {
        "n_weeks": n_weeks,
        "date_trained": datetime.utcnow().isoformat() + "Z",
        "epochs_ran": len(history.history["loss"]),
        "train_loss_last": float(history.history["loss"][-1]),
        "val_loss_last": float(history.history["val_loss"][-1]) if "val_loss" in history.history else None,
        "units": units,
        "dropout": dropout
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return model, scaler, meta


def predict_next(model, scaler, data, n_weeks):
    """Predict next single-step weekly value."""
    data = _ensure_numpy_1d(data)

    if len(data) < n_weeks:
        raise ValueError("Not enough data for prediction.")

    seq = data[-n_weeks:].reshape(-1, 1)
    seq_scaled = scaler.transform(seq).reshape(1, n_weeks, 1)

    pred_scaled = model.predict(seq_scaled, verbose=0)
    pred = scaler.inverse_transform(pred_scaled)[0][0]

    return float(pred)


def predict_multi_step(model, scaler, data, n_weeks, steps=4):
    """Iterative multi-step forecasting."""
    buffer = list(_ensure_numpy_1d(data)[-n_weeks:])
    preds = []

    for _ in range(steps):
        val = predict_next(model, scaler, np.array(buffer), n_weeks)
        preds.append(val)
        buffer.append(val)
        buffer.pop(0)

    return preds
