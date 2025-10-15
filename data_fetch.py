#Data Fetching Module
import yfinance as yf
import pandas as pd
import os
import datetime as dt

def fetch_daily_data(ticker, start="2018-01-01", end=None):
    os.makedirs("data", exist_ok=True)
    if end is None:
        end = dt.date.today()
    file = f"data/{ticker}_{start}_{end}.csv"
    if os.path.exists(file):
        df = pd.read_csv(file, parse_dates=["Date"], low_memory=False)
        if not df.empty:
            return df
    data = yf.download(ticker, start=start, end=end, progress=False)
    if data.empty:
        return pd.DataFrame()
    data.reset_index(inplace=True)
    data.to_csv(file, index=False)
    return data
