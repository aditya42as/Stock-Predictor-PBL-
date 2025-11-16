# App
import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt

from auth import signup, login
from data_fetch import fetch_daily_data
from data_preprocess import resample_weekly_close, add_weekly_features
from model_prophet import prophet_weekly_forecast
from model_lstm import (
    train_and_save_lstm,
    load_trained,
    predict_next,
    predict_multi_step
)
from sentiment import score_texts, aggregate_scores
from news_fetcher import fetch_headlines
from model_compare import compare_next_step, small_backtest

st.set_page_config(page_title="Weekly Stock Predictor", layout="wide")
st.title("📈 Weekly Stock Predictor")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.header("Login / Signup")
    mode = st.radio("Choose:", ["Login", "Sign Up"])
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Continue"):
        msg = login(u, p) if mode == "Login" else signup(u, p)
        st.info(msg)
        if msg == "Login successful.":
            st.session_state.logged_in = True
            st.session_state.username = u
            st.rerun()
    st.stop()

st.success(f"Welcome, {st.session_state.username}!")

with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Stock Ticker", "TSLA")
    n_weeks = st.number_input("LSTM sequence length (weeks)", 2, 52, 8)
    epochs = st.number_input("Epochs", 1, 500, 50)
    batch_size = st.number_input("Batch size", 1, 256, 16)
    steps = st.number_input("Forecast steps (LSTM)", 1, 12, 1)
    show_debug = st.checkbox("Show Debug Info", value=False)
    st.markdown("---")
    st.write("News fetch")
    newsapi_key = st.text_input("NewsAPI key (optional)", type="password")
    prefer_api = st.checkbox("Prefer NewsAPI (else scrape Google News)", value=True)
    st.markdown("---")
    st.write("Model comparison / backtest")
    backtest_weeks = st.number_input("Backtest weeks", 1, 12, 4)
    quick_epochs = st.number_input("Quick train epochs (backtest)", 1, 200, 20)
    quick_batch = st.number_input("Quick train batch (backtest)", 1, 128, 8)

if st.button("Fetch Weekly Data"):
    try:
        today = dt.date.today()
        df_daily = fetch_daily_data(ticker, "2018-01-01", today)
        if df_daily is None or df_daily.empty:
            st.error("Invalid ticker or no data found.")
        else:
            df_daily["Date"] = pd.to_datetime(df_daily["Date"])
            weekly = resample_weekly_close(df_daily)
            weekly.columns = [c.capitalize() for c in weekly.columns]
            st.session_state.weekly = weekly
            st.success("Weekly data loaded.")
    except Exception as e:
        st.error(f"Error fetching data: {e}")

if "weekly" in st.session_state:
    weekly = st.session_state.weekly
    st.subheader("Weekly Close Data")
    st.dataframe(weekly.tail(30))
    st.line_chart(weekly.set_index("Date")["Close"].tail(150))

st.markdown("---")
st.header("LSTM Model")

if "weekly" not in st.session_state:
    st.info("Fetch data first.")
else:
    weekly = st.session_state.weekly
    close_arr = weekly.sort_values("Date")["Close"].dropna().values.astype(float)
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Train LSTM"):
            try:
                with st.spinner("Training LSTM..."):
                    model, scaler, meta = train_and_save_lstm(
                        close_arr,
                        n_weeks=n_weeks,
                        epochs=int(epochs),
                        batch_size=int(batch_size),
                        validation_split=0.1,
                        units=50,
                        dropout=0.0,
                        patience=6,
                        verbose=0
                    )
                    st.success("Model trained!")
                    st.json(meta)
            except Exception as e:
                st.error(f"Training error: {e}")

    with col2:
        if st.button("Predict (LSTM)"):
            try:
                model, scaler, meta = load_trained(n_weeks)
                if model is None:
                    st.warning("No model found. Train first.")
                else:
                    if steps == 1:
                        pred = predict_next(model, scaler, close_arr, n_weeks)
                        st.metric("Next Week Prediction (LSTM)", f"{pred:.2f}")
                    else:
                        preds = predict_multi_step(model, scaler, close_arr, n_weeks, steps=int(steps))
                        st.dataframe(pd.DataFrame({"Predicted": preds}))
            except Exception as e:
                st.error(f"Prediction error: {e}")

    if show_debug:
        st.write({"len_data": len(close_arr), "n_weeks": n_weeks})

st.markdown("---")
st.header("🔮 Prophet Forecast")

if "weekly" in st.session_state:
    try:
        forecast = prophet_weekly_forecast(weekly, periods=1)
        st.dataframe(forecast)
    except Exception as e:
        st.error(f"Prophet error: {e}")

st.markdown("---")
st.header("📰 Market Sentiment & News")

col_a, col_b = st.columns([2, 1])

with col_a:
    txt = st.text_area("Paste headlines (one per line):")
    if st.button("Analyze Sentiment"):
        lines = [x.strip() for x in txt.split("\n") if x.strip()]
        if not lines:
            st.warning("Enter at least one line.")
        else:
            scores = score_texts(lines)
            agg = aggregate_scores(scores, weight_recency=True)
            st.metric("Sentiment Score (Compound)", f"{agg['compound_mean']:.3f}")
            st.write(agg)
            st.subheader("Detailed Scores")
            st.dataframe(pd.DataFrame(scores))

with col_b:
    if st.button("Fetch Headlines for Ticker"):
        try:
            query = ticker + " stock"
            items = fetch_headlines(query, api_key=newsapi_key or None, prefer_api=prefer_api, max_items=25)
            if not items:
                st.info("No headlines found.")
            else:
                titles = [it.get("title") or it.get("snippet") or "" for it in items]
                st.session_state.fetched_headlines = titles
                st.write(pd.DataFrame(items))
        except Exception as e:
            st.error(f"Headline fetch error: {e}")

    if "fetched_headlines" in st.session_state:
        if st.button("Analyze Fetched Headlines"):
            try:
                texts = st.session_state.fetched_headlines
                scores = score_texts(texts)
                agg = aggregate_scores(scores, weight_recency=True)
                st.metric("Headline Sentiment (compound)", f"{agg['compound_mean']:.3f}")
                st.dataframe(pd.DataFrame(scores))
            except Exception as e:
                st.error(f"Analysis error: {e}")

st.markdown("---")
st.header("⚖️ Model Comparison & Backtest")

colc, cold = st.columns(2)

with colc:
    if st.button("Compare Next-Step (Prophet vs LSTM)"):
        try:
            if "weekly" not in st.session_state:
                st.warning("Fetch weekly data first.")
            else:
                result = compare_next_step(weekly, n_weeks=n_weeks, retrain_lstm=False,
                                           lstm_train_params={"epochs": int(quick_epochs), "batch_size": int(quick_batch), "validation_split": 0.12, "units": 50, "dropout": 0.0})
                st.write(result)
                if result.get("prophet_pred") is not None:
                    st.metric("Prophet next-step", f"{result['prophet_pred']:.2f}")
                if result.get("lstm_pred") is not None:
                    st.metric("LSTM next-step", f"{result['lstm_pred']:.2f}")
        except Exception as e:
            st.error(f"Compare error: {e}")

with cold:
    if st.button("Run Quick Backtest"):
        try:
            if "weekly" not in st.session_state:
                st.warning("Fetch weekly data first.")
            else:
                df_bt = small_backtest(weekly, n_weeks=n_weeks, backtest_weeks=int(backtest_weeks),
                                       quick_train_epochs=int(quick_epochs), quick_batch=int(quick_batch))
                st.dataframe(df_bt)
                prop_mean_err = df_bt["prophet_error_abs"].dropna().mean() if "prophet_error_abs" in df_bt.columns else None
                lstm_mean_err = df_bt["lstm_error_abs"].dropna().mean() if "lstm_error_abs" in df_bt.columns else None
                st.write({"prophet_mean_abs_error": prop_mean_err, "lstm_mean_abs_error": lstm_mean_err})
        except Exception as e:
            st.error(f"Backtest error: {e}")

st.markdown("---")


