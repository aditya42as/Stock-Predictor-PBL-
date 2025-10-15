#Data Preprocessing Module
import pandas as pd

def resample_weekly_close(df_daily):
    import pandas as pd
    df = df_daily.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(map(str, c)).lower().strip() for c in df.columns]
    else:
        df.columns = [str(c).lower().replace(".", "_").strip() for c in df.columns]

    if "date" not in df.columns:
        if "datetime" in df.columns:
            df.rename(columns={"datetime": "date"}, inplace=True)
        elif "index" in df.columns:
            df.rename(columns={"index": "date"}, inplace=True)
        elif df.index.name:
            df.reset_index(inplace=True)
            df.rename(columns={df.columns[0]: "date"}, inplace=True)
        else:
            df.reset_index(inplace=True)
            df.rename(columns={df.columns[0]: "date"}, inplace=True)

    rename_map = {}
    for col in df.columns:
        name = col.replace(" ", "_")
        if "close" in name:
            rename_map[col] = "close"
        elif "open" in name:
            rename_map[col] = "open"
        elif "high" in name:
            rename_map[col] = "high"
        elif "low" in name:
            rename_map[col] = "low"
        elif "vol" in name:
            rename_map[col] = "volume"
    df.rename(columns=rename_map, inplace=True)

    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = pd.Series(dtype=float)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date"]).set_index("date")
    if "close" not in df.columns or df["close"].dropna().empty:
        for c in df.columns:
            if "close" in c:
                df["close"] = pd.to_numeric(df[c], errors="coerce")
                break
    if "close" not in df.columns or df["close"].dropna().empty:
        raise ValueError("No valid closing price data found for this ticker.")

    weekly = df.resample("W-FRI").last().dropna(subset=["close"]).reset_index()
    weekly = weekly[["date", "open", "high", "low", "close", "volume"]]
    return weekly


def add_weekly_features(weekly_df: pd.DataFrame) -> pd.DataFrame:
    df = weekly_df.copy().sort_values("Date").reset_index(drop=True)

    # Ensure numeric
    for col in ["Open","High","Low","Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["SMA_3"] = df["Close"].rolling(window=3).mean()
    df["SMA_6"] = df["Close"].rolling(window=6).mean()
    df["Weekly_Return"] = df["Close"].pct_change()

    df = df.dropna().reset_index(drop=True)
    return df

def create_lstm_sequences(series, n_weeks=8):
    X, y = [], []
    for i in range(len(series) - n_weeks):
        X.append(series[i:i + n_weeks])
        y.append(series[i + n_weeks])
    return X, y
