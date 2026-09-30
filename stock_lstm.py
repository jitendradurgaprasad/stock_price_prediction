"""End-to-end daily stock close forecasting with Keras 3 and Yahoo Finance."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("KERAS_BACKEND", "torch")

import torch

# Small recurrent models often run more efficiently with one CPU worker on
# desktop machines; users can override this with LSTM_TORCH_THREADS.
torch.set_num_threads(max(1, int(os.environ.get("LSTM_TORCH_THREADS", "1"))))

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


FEATURES = ["Daily_Return", "SMA_20_Gap", "SMA_50_Gap", "RSI_14", "Log_Volume"]
TARGET = "Daily_Return"
LOOKBACK = 60
SEED = 42


@dataclass
class PreparedData:
    frame: pd.DataFrame
    x_scaler: MinMaxScaler
    y_scaler: MinMaxScaler
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    test_dates: pd.DatetimeIndex
    train_end: int
    val_end: int


def fetch_history(ticker: str = "AAPL", period: str = "5y") -> pd.DataFrame:
    """Download split/dividend adjusted daily history through yfinance."""
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Enter a ticker symbol, such as AAPL.")
    frame = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    if frame is None or frame.empty:
        raise RuntimeError(f"Yahoo Finance returned no daily history for {ticker}.")
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    required = {"Close", "Volume"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"Yahoo Finance data is missing required columns: {required - set(frame.columns)}")
    return frame[[c for c in ["Open", "High", "Low", "Close", "Volume"] if c in frame]].dropna()


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Add backward-looking simple moving averages and Wilder-style RSI."""
    data = frame.copy().sort_index()
    data["SMA_20"] = data["Close"].rolling(window=20, min_periods=20).mean()
    data["SMA_50"] = data["Close"].rolling(window=50, min_periods=50).mean()
    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    data["RSI_14"] = 100 - (100 / (1 + rs))
    data["RSI_14"] = data["RSI_14"].where(avg_loss.ne(0), 100)
    data["Daily_Return"] = data["Close"].pct_change()
    data["SMA_20_Gap"] = data["SMA_20"] / data["Close"] - 1
    data["SMA_50_Gap"] = data["SMA_50"] / data["Close"] - 1
    data["Log_Volume"] = np.log1p(data["Volume"].clip(lower=0))
    return data.dropna(subset=FEATURES).copy()


def prepare_data(frame: pd.DataFrame, lookback: int = LOOKBACK) -> PreparedData:
    """Chronologically split rows 70/15/15 and fit both scalers on train rows only."""
    data = add_indicators(frame) if not set(FEATURES).issubset(frame.columns) else frame.dropna(subset=FEATURES).copy()
    n_rows = len(data)
    train_end = int(n_rows * 0.70)
    val_end = int(n_rows * 0.85)
    if train_end <= lookback or val_end <= train_end or n_rows <= val_end:
        raise ValueError("Not enough usable rows. Download at least 2 years of daily history.")

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    x_scaled = np.empty((n_rows, len(FEATURES)), dtype=np.float32)
    x_scaler.fit(data.iloc[:train_end][FEATURES])
    x_scaled[:] = x_scaler.transform(data[FEATURES]).astype(np.float32)
    y_values = data[[TARGET]].to_numpy(dtype=np.float32)
    y_scaler.fit(y_values[:train_end])
    y_scaled = y_scaler.transform(y_values).astype(np.float32)

    def make_windows(start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
        targets = range(max(start, lookback), stop)
        xs = np.stack([x_scaled[i - lookback : i] for i in targets])
        ys = np.stack([y_scaled[i] for i in targets])
        return xs, ys

    train_x, train_y = make_windows(lookback, train_end)
    val_x, val_y = make_windows(train_end, val_end)
    test_x, test_y = make_windows(val_end, n_rows)
    test_dates = pd.DatetimeIndex(data.index[val_end:])
    return PreparedData(data, x_scaler, y_scaler, train_x, train_y, val_x, val_y, test_x, test_y,
                       test_dates, train_end, val_end)


def build_model(lookback: int = LOOKBACK, n_features: int = len(FEATURES)) -> keras.Model:
    """Build a compact stacked LSTM for next-session adjusted close."""
    keras.utils.set_random_seed(SEED)
    model = keras.Sequential([
        keras.layers.Input(shape=(lookback, n_features)),
        keras.layers.LSTM(32, return_sequences=True),
        keras.layers.Dropout(0.15),
        keras.layers.LSTM(16),
        keras.layers.Dropout(0.15),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),
    ], name="daily_stock_close_lstm")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model


def _inverse(scaler: MinMaxScaler, values: np.ndarray) -> np.ndarray:
    return scaler.inverse_transform(np.asarray(values).reshape(-1, 1)).reshape(-1)


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray, previous: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    previous = np.asarray(previous, dtype=float).reshape(-1)
    error = predicted - actual
    baseline_error = previous - actual
    denominator = np.maximum(np.abs(actual), 1e-8)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape_percent": float(np.mean(np.abs(error) / denominator) * 100),
        "directional_accuracy_percent": float(np.mean(np.sign(predicted - previous) == np.sign(actual - previous)) * 100),
        "naive_last_close_mae": float(np.mean(np.abs(baseline_error))),
    }


def _save_plots(data: PreparedData, predictions: np.ndarray, history: keras.callbacks.History,
                output_dir: Path, ticker: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = data.frame

    fig, (ax_price, ax_rsi) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                          gridspec_kw={"height_ratios": [3, 1]})
    ax_price.plot(frame.index, frame["Close"], label="Adjusted close", linewidth=1.2, color="#1f4e79")
    ax_price.plot(frame.index, frame["SMA_20"], label="20-day SMA", linewidth=1, color="#f28e2b")
    ax_price.plot(frame.index, frame["SMA_50"], label="50-day SMA", linewidth=1, color="#59a14f")
    ax_price.set_title(f"{ticker}: adjusted close and moving averages")
    ax_price.set_ylabel("Price (USD)")
    ax_price.legend(loc="upper left", ncol=3)
    ax_price.grid(alpha=0.2)
    ax_rsi.plot(frame.index, frame["RSI_14"], color="#9467bd", linewidth=0.9)
    ax_rsi.axhline(70, color="#e15759", linestyle="--", linewidth=0.8)
    ax_rsi.axhline(30, color="#59a14f", linestyle="--", linewidth=0.8)
    ax_rsi.set_ylabel("RSI (14)")
    ax_rsi.set_ylim(0, 100)
    ax_rsi.grid(alpha=0.2)
    ax_rsi.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    fig.savefig(output_dir / "price_indicators.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    test_actual = data.frame["Close"].iloc[data.val_end:].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(data.test_dates, test_actual, label="Actual adjusted close", color="#1f4e79", linewidth=1.5)
    ax.plot(data.test_dates, predictions, label="LSTM one-step prediction", color="#e15759", linewidth=1.2)
    ax.set_title(f"{ticker}: one-session-ahead predictions on chronological test set")
    ax.set_ylabel("Price (USD)")
    ax.set_xlabel("Test period")
    ax.legend()
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    fig.savefig(output_dir / "test_predictions.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(history.history.get("loss", []), label="Training loss")
    ax.plot(history.history.get("val_loss", []), label="Validation loss")
    ax.set_title("Training and validation mean squared error")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Scaled MSE")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "training_history.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def train_pipeline(ticker: str = "AAPL", period: str = "5y", output_dir: str | Path = ".",
                   epochs: int = 20, batch_size: int = 64, verbose: int = 1,
                   downloaded_frame: pd.DataFrame | None = None) -> dict[str, Any]:
    """Fetch, prepare, train, evaluate, and save all requested model artifacts."""
    ticker = ticker.strip().upper()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = downloaded_frame if downloaded_frame is not None else fetch_history(ticker, period)
    raw = raw.copy()
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw.to_csv(out / f"{ticker}_raw_history.csv", index_label="Date")
    frame = add_indicators(raw)
    prepared = prepare_data(frame)
    model = build_model()
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
    ]
    fit_history = model.fit(prepared.train_x, prepared.train_y,
                            validation_data=(prepared.val_x, prepared.val_y),
                            epochs=epochs, batch_size=batch_size, shuffle=False,
                            callbacks=callbacks, verbose=verbose)
    scaled_predictions = model.predict(prepared.test_x, verbose=0)
    predictions = _inverse(prepared.y_scaler, scaled_predictions)
    predicted_returns = predictions
    test_start_position = prepared.val_end
    previous_close = prepared.frame["Close"].iloc[test_start_position - 1 : -1].to_numpy(dtype=float)
    actual = prepared.frame["Close"].iloc[test_start_position:].to_numpy(dtype=float)
    predicted_prices = previous_close * (1 + predicted_returns)
    predictions = predicted_prices
    metrics = calculate_metrics(actual, predicted_prices, previous_close)
    metrics.update({
        "ticker": ticker,
        "target": "next trading session adjusted close (USD)",
        "lookback_sessions": LOOKBACK,
        "feature_columns": FEATURES,
        "rows_after_indicator_warmup": int(len(prepared.frame)),
        "train_rows": int(prepared.train_end),
        "validation_rows": int(prepared.val_end - prepared.train_end),
        "test_rows": int(len(prepared.frame) - prepared.val_end),
        "test_start": str(prepared.test_dates.min().date()),
        "test_end": str(prepared.test_dates.max().date()),
        "epochs_completed": int(len(fit_history.history.get("loss", []))),
    })
    last_window = prepared.x_scaler.transform(prepared.frame[FEATURES].iloc[-LOOKBACK:]).astype(np.float32)
    next_scaled = model.predict(last_window[np.newaxis, ...], verbose=0)
    next_return = float(_inverse(prepared.y_scaler, next_scaled)[0])
    last_close = float(prepared.frame["Close"].iloc[-1])
    next_close = last_close * (1 + next_return)
    forecast = {
        "ticker": ticker,
        "as_of_date": str(prepared.frame.index[-1].date()),
        "last_adjusted_close": last_close,
        "next_session_adjusted_close_estimate": next_close,
        "predicted_change_percent": next_return * 100,
        "change_percent_vs_last_close": next_return * 100,
        "horizon": "one trading session; one-step forecast using the latest 60 observed sessions",
    }

    weights_path = out / f"{ticker}_lstm.weights.h5"
    model.save_weights(weights_path)
    import joblib

    joblib.dump({"x_scaler": prepared.x_scaler, "y_scaler": prepared.y_scaler,
                 "features": FEATURES, "lookback": LOOKBACK, "ticker": ticker},
                out / f"{ticker}_scalers.joblib")
    frame.to_csv(out / f"{ticker}_history.csv", index_label="Date")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "forecast.json").write_text(json.dumps(forecast, indent=2), encoding="utf-8")
    _save_plots(prepared, predictions, fit_history, out, ticker)

    return {"model": model, "prepared": prepared, "history": fit_history,
            "predictions": predictions, "actual": actual, "metrics": metrics,
            "forecast": forecast, "weights_path": weights_path}


def load_saved_model(output_dir: str | Path = ".", ticker: str = "AAPL") -> tuple[keras.Model, dict[str, Any]]:
    """Rebuild the network and load its portable Keras weights and scalers."""
    import joblib

    out = Path(output_dir)
    ticker = ticker.strip().upper()
    bundle = joblib.load(out / f"{ticker}_scalers.joblib")
    model = build_model(bundle["lookback"], len(bundle["features"]))
    model.load_weights(out / f"{ticker}_lstm.weights.h5")
    return model, bundle


def forecast_latest(history: pd.DataFrame, model: keras.Model, bundle: dict[str, Any]) -> float:
    """Return the next-session close estimate from a fresh historical frame."""
    data = add_indicators(history)
    features = bundle["features"]
    lookback = bundle["lookback"]
    if len(data) < lookback:
        raise ValueError(f"Need at least {lookback + 50} raw observations to warm up indicators and form a sequence.")
    sequence = bundle["x_scaler"].transform(data[features].iloc[-lookback:]).astype(np.float32)
    estimate_scaled = model.predict(sequence[np.newaxis, ...], verbose=0)
    predicted_return = float(_inverse(bundle["y_scaler"], estimate_scaled)[0])
    return float(data["Close"].iloc[-1] * (1 + predicted_return))
