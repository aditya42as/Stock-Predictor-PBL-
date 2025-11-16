import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from model_prophet import prophet_weekly_forecast
from model_lstm import load_trained, train_and_save_lstm, predict_next
import warnings

warnings.filterwarnings("ignore")

def compare_next_step(weekly_df: pd.DataFrame,
                      n_weeks: int = 8,
                      lstm_model: Optional[Any] = None,
                      lstm_scaler: Optional[Any] = None,
                      retrain_lstm: bool = False,
                      lstm_train_params: Optional[Dict] = None) -> Dict:
    if lstm_train_params is None:
        lstm_train_params = {"epochs": 30, "batch_size": 16, "validation_split": 0.12, "units": 50, "dropout": 0.0}

    df = weekly_df.sort_values("Date").reset_index(drop=True)
    last_actual = float(df["Close"].iloc[-1])

    try:
        prophet_forecast = prophet_weekly_forecast(df, periods=1)
        prophet_pred = float(prophet_forecast["yhat"].iloc[-1])
    except:
        prophet_pred = None

    lstm_pred = None
    try:
        if lstm_model is None or lstm_scaler is None:
            model, scaler, meta = load_trained(n_weeks)
            if model is None or retrain_lstm:
                arr = df["Close"].dropna().values.astype(float)
                if len(arr) > n_weeks:
                    model, scaler, meta = train_and_save_lstm(arr, n_weeks=n_weeks, **lstm_train_params)
                else:
                    model = scaler = None
            lstm_model, lstm_scaler = model, scaler
        if lstm_model and lstm_scaler:
            arr = df["Close"].dropna().values.astype(float)
            if len(arr) >= n_weeks:
                lstm_pred = float(predict_next(lstm_model, lstm_scaler, arr, n_weeks))
    except:
        lstm_pred = None

    return {
        "prophet_pred": prophet_pred,
        "lstm_pred": lstm_pred,
        "last_actual": last_actual
    }


def small_backtest(weekly_df: pd.DataFrame,
                   n_weeks: int = 8,
                   backtest_weeks: int = 6,
                   quick_train_epochs: int = 20,
                   quick_batch: int = 8) -> pd.DataFrame:
    df = weekly_df.sort_values("Date").reset_index(drop=True)
    results = []
    T = len(df)
    actual_backtest_weeks = min(backtest_weeks, T - (n_weeks + 1))
    start_idx = T - actual_backtest_weeks

    for t in range(start_idx, T):
        train_df = df.iloc[:t]
        true_val = float(df["Close"].iloc[t])
        try:
            pf = prophet_weekly_forecast(train_df, periods=1)
            prop_pred = float(pf["yhat"].iloc[-1])
        except:
            prop_pred = None
        lstm_pred = None
        try:
            arr = train_df["Close"].dropna().values.astype(float)
            if len(arr) > n_weeks:
                model, scaler, meta = train_and_save_lstm(
                    arr,
                    n_weeks=n_weeks,
                    epochs=quick_train_epochs,
                    batch_size=quick_batch,
                    validation_split=0.12,
                    units=30,
                    dropout=0.0,
                    patience=4,
                    verbose=0
                )
                lstm_pred = float(predict_next(model, scaler, arr, n_weeks))
        except:
            lstm_pred = None

        results.append({
            "index": t,
            "date": df["Date"].iloc[t],
            "actual": true_val,
            "prophet_pred": prop_pred,
            "lstm_pred": lstm_pred,
            "prophet_error_abs": abs(prop_pred - true_val) if prop_pred is not None else None,
            "lstm_error_abs": abs(lstm_pred - true_val) if lstm_pred is not None else None
        })

    return pd.DataFrame(results).set_index("index")
