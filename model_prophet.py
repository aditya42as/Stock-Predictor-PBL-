#Prophet Module
from prophet import Prophet
import pandas as pd

def prophet_weekly_forecast(df, periods=1):
    df = df[["Date","Close"]].rename(columns={"Date":"ds","Close":"y"})
    model = Prophet(weekly_seasonality=True, daily_seasonality=False)
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq="W-FRI")
    forecast = model.predict(future)
    return forecast[["ds","yhat","yhat_lower","yhat_upper"]].tail(periods)
