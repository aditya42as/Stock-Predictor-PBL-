#APP(web UI)
import streamlit as st
import pandas as pd
from auth import signup, login
from data_fetch import fetch_daily_data
from data_preprocess import resample_weekly_close, add_weekly_features
from model_prophet import prophet_weekly_forecast
import numpy as np
import warnings

warnings.filterwarnings("ignore")

st.title("📊 Weekly Stock Predictor")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.header("Login / Signup")
    option = st.radio("Choose:", ["Login", "Sign Up"])
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Continue"):
        msg = login(u, p) if option == "Login" else signup(u, p)
        st.info(msg)
        if msg == "Login successful.":
            st.session_state.logged_in = True
            st.session_state.username = u
            st.rerun()
    st.stop()

st.success(f"Welcome, {st.session_state.username}!")

ticker = st.text_input("Enter stock ticker (e.g. TSLA)", "TSLA")

if st.button("Fetch Weekly Data"):
    st.markdown("### Fetching Data...")
    try:
        import datetime as dt
        today = dt.date.today()
        df_daily = fetch_daily_data(ticker, "2018-01-01", today)
        if df_daily is None or df_daily.empty:
            st.error("⚠️ No valid data returned for this ticker. Try another one or check your internet connection.")
            st.stop()
        if df_daily.empty:
            st.error("No data found. Please check the ticker symbol or your internet connection.")
            st.stop()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

    if "Date" in df_daily.columns:
        df_daily["Date"] = pd.to_datetime(df_daily["Date"], errors="coerce")

    st.markdown("### Daily Data Preview")
    st.dataframe(df_daily.tail())

    weekly = resample_weekly_close(df_daily)
    weekly.columns = [c.capitalize() for c in weekly.columns]


    if "Date" in weekly.columns:
        weekly["Date"] = pd.to_datetime(weekly["Date"], errors="coerce")

    st.markdown("### Weekly (Friday Close) Data")
    st.dataframe(weekly.tail())

    weekly["Close"] = pd.to_numeric(weekly["Close"], errors="coerce")
    weekly = weekly.dropna(subset=["Date", "Close"]).sort_values("Date")
    weekly = weekly.loc[~weekly["Date"].duplicated(keep="last")]
    weekly_plot = weekly.set_index("Date")[["Close"]].copy()

    weekly_plot = weekly_plot[np.isfinite(weekly_plot["Close"])]
    weekly_plot = weekly_plot[
        (weekly_plot["Close"] > 0)
        & (weekly_plot["Close"] < weekly_plot["Close"].quantile(0.99))
    ]

    st.markdown("### Weekly Close Chart")
    st.line_chart(weekly_plot.tail(150))

    with st.expander("Debug Info"):
        st.write(weekly.dtypes)
        st.write("Close min/max:", weekly["Close"].min(), weekly["Close"].max())
        st.write(weekly.head())

    st.session_state.weekly = weekly

if "weekly" in st.session_state:
    weekly = st.session_state.weekly
    weekly_feat = add_weekly_features(weekly)

    if "Date" in weekly_feat.columns:
        weekly_feat["Date"] = pd.to_datetime(weekly_feat["Date"], errors="coerce")

    st.markdown("### Weekly Features Preview")
    st.dataframe(weekly_feat.tail())

    if st.button("Predict Next Week (Prophet)"):
        st.markdown("### 🔮 Prophet Forecast")
        forecast = prophet_weekly_forecast(weekly, periods=1)

        if "ds" in forecast.columns:
            forecast["ds"] = pd.to_datetime(forecast["ds"], errors="coerce")

        st.dataframe(forecast)
        st.success("Prophet forecast generated successfully!")
