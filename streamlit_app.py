"""Local Streamlit dashboard for the trained AAPL example."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from stock_lstm import FEATURES, add_indicators, fetch_history, forecast_latest, load_saved_model


HERE = Path(__file__).resolve().parent
TICKER = "AAPL"

st.set_page_config(page_title="Stock Trend Prediction | LSTM", page_icon="📈", layout="wide")
st.title("Stock Price Trend Prediction with LSTM")
st.caption("One trading session ahead · AAPL · Yahoo Finance adjusted daily prices")

metrics_path = HERE / "metrics.json"
forecast_path = HERE / "forecast.json"
history_path = HERE / f"{TICKER}_history.csv"
weights_path = HERE / f"{TICKER}_lstm.weights.h5"
scaler_path = HERE / f"{TICKER}_scalers.joblib"
needed = [metrics_path, forecast_path, history_path, weights_path, scaler_path]
if not all(path.exists() for path in needed):
    st.error("Model artifacts are missing. Run `python train.py` from this folder first.")
    st.stop()

metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
saved_forecast = json.loads(forecast_path.read_text(encoding="utf-8"))
frame = pd.read_csv(history_path, parse_dates=["Date"], index_col="Date")

try:
    model, scaler_bundle = load_saved_model(HERE, TICKER)
except Exception as exc:
    st.error(f"Could not load saved model artifacts: {exc}")
    st.stop()

forecast_value = saved_forecast["next_session_adjusted_close_estimate"]
forecast_as_of = saved_forecast["as_of_date"]
left, center, right = st.columns(3)
left.metric("Last adjusted close", f"${saved_forecast['last_adjusted_close']:,.2f}", forecast_as_of)
center.metric("Next-session estimate", f"${forecast_value:,.2f}",
              f"{saved_forecast['change_percent_vs_last_close']:+.2f}% vs last close")
right.metric("Test MAE", f"${metrics['mae']:,.2f}", f"Naive baseline ${metrics['naive_last_close_mae']:,.2f}")

st.info("This is a one-step educational forecast, not a long-term price target. Test predictions use actual prior-session data at every step; the next-session estimate is not a promise of future performance.")

st.subheader("Historical price, moving averages, and RSI")
fig, (ax, rsi_ax) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
ax.plot(frame.index, frame["Close"], label="Adjusted close", color="#1f4e79")
ax.plot(frame.index, frame["SMA_20"], label="20-day SMA", color="#f28e2b")
ax.plot(frame.index, frame["SMA_50"], label="50-day SMA", color="#59a14f")
ax.set_ylabel("USD")
ax.legend(ncol=3)
ax.grid(alpha=0.2)
rsi_ax.plot(frame.index, frame["RSI_14"], color="#9467bd")
rsi_ax.axhline(70, color="#e15759", linestyle="--", linewidth=0.8)
rsi_ax.axhline(30, color="#59a14f", linestyle="--", linewidth=0.8)
rsi_ax.set_ylabel("RSI")
rsi_ax.set_ylim(0, 100)
rsi_ax.grid(alpha=0.2)
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.subheader("Chronological holdout: actual vs one-step prediction")
test_plot = HERE / "test_predictions.png"
st.image(str(test_plot), use_container_width=True)

with st.expander("Model evaluation details"):
    st.write(f"Test window: {metrics['test_start']} to {metrics['test_end']} ({metrics['test_rows']} sessions)")
    st.write(f"Features: {', '.join(FEATURES)}; input lookback: {metrics['lookback_sessions']} sessions")
    st.write("MAE and RMSE are in USD. MAPE is percent error. Directional accuracy compares the predicted and actual direction relative to the previous close.")
    st.json({key: metrics[key] for key in ["mae", "rmse", "mape_percent", "directional_accuracy_percent", "naive_last_close_mae"]})

with st.expander("Refresh latest Yahoo Finance data"):
    st.write("Download recent prices and apply the saved AAPL model. The trained weights remain fixed.")
    if st.button("Fetch latest AAPL history and forecast"):
        with st.spinner("Fetching Yahoo Finance history..."):
            try:
                fresh = fetch_history(TICKER, "5y")
                live_data = add_indicators(fresh)
                live_estimate = forecast_latest(fresh, model, scaler_bundle)
                st.success(f"Latest available close ({live_data.index[-1].date()}): ${live_data['Close'].iloc[-1]:,.2f} · next-session estimate: ${live_estimate:,.2f}")
            except Exception as exc:
                st.error(f"Refresh failed: {exc}")

st.caption("Data source: Yahoo Finance via yfinance. This dashboard runs locally at http://localhost:8501 when launched with Streamlit.")
