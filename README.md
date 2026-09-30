# Stock Price Trend Prediction with LSTM

This project uses Yahoo Finance daily history, technical indicators, and a Keras LSTM to estimate the **next trading session's adjusted closing price**. The default example is Apple (`AAPL`) using five years of history.

## Deliverables

| File | Purpose |
| --- | --- |
| `Stock_Price_Trend_Prediction_LSTM.ipynb` | Executed, walkthrough notebook: data, features, chronological split, LSTM, evaluation, and plots |
| `AAPL_lstm.weights.h5` | Trained Keras model weights |
| `AAPL_scalers.joblib` | Train-fitted feature and target scalers plus model metadata; needed with the weights |
| `AAPL_history.csv` | Downloaded daily adjusted OHLCV data with SMA and RSI features |
| `AAPL_raw_history.csv` | Raw adjusted OHLCV download used for reproducible notebook retraining |
| `test_predictions.png` | Actual vs. one-session-ahead test-set estimates |
| `price_indicators.png` | Adjusted close, 20/50-day SMA, and RSI(14) visualization |
| `training_history.png` | Training and validation loss by epoch |
| `metrics.json`, `forecast.json` | Machine-readable holdout metrics and most-recent next-session estimate |
| `streamlit_app.py` | Local interactive dashboard source |
| `stock_lstm.py`, `train.py` | Reusable implementation and command-line training entry point |
| `requirements.txt` | Python package requirements |
| `launch_app.bat` | Windows one-click launcher; creates `.venv`, installs dependencies, then starts the dashboard |
| `Stock_Price_Trend_Prediction_Project_Report.pdf` | Project report covering the methodology, evaluation, plots, limitations, and run instructions |

## Run it

From this folder, install dependencies and train the default model:

```bash
python -m pip install -r requirements.txt
python train.py
```

On Windows, double-click `launch_app.bat` to create the local environment, install the required packages, and start the dashboard. The first run can take a few minutes because it downloads Python packages and the model runtime. Python 3.10 or newer must be installed.

To choose another Yahoo Finance symbol or history period from a terminal:

```bash
python train.py --ticker MSFT --period 5y
```

Then open `Stock_Price_Trend_Prediction_LSTM.ipynb` in Jupyter and run its cells, or launch the dashboard:

```bash
streamlit run streamlit_app.py
```

The local dashboard address is **http://localhost:8501**. It is not a publicly hosted deployment; publish it through a Streamlit hosting service if a public URL is required.

### Vercel-hosted dashboard

The Vercel version is a static dashboard built from the saved metrics and plots. Its project files are in `vercel_site/`; use that folder as the Vercel deployment root. After retraining, run `python build_site.py` from `vercel_site/`, then deploy it with `npx vercel --prod`. Vercel hosts this dashboard; the full Keras inference pipeline and interactive Streamlit refresh stay local.

## Method and evaluation

- Yahoo Finance prices are requested with automatic split/dividend adjustment. SMA(20), SMA(50), RSI(14), and volume features are computed from information available up to each session; price trend features are expressed as returns and gaps from the moving averages.
- The input is the previous 60 sessions. The LSTM predicts the next-session percentage change, which is composed with the latest close to produce a price estimate. Features and target are min-max scaled using training rows only.
- Rows are divided chronologically into 70% train, 15% validation, and 15% test. The test set is never shuffled.
- Test results are rolling one-step forecasts: each prediction uses the preceding 60 observed sessions. This is different from recursively forecasting a long sequence without new observations.
- Metrics include MAE, RMSE, MAPE, directional accuracy, and a naïve last-close MAE baseline.
- No model can reliably infer future market shocks from past prices alone. Results are an educational time-series example, not investment advice or a guarantee.

Yahoo Finance access is provided through the community-maintained `yfinance` package, not an official Yahoo Finance API guarantee. For API details see the [yfinance project documentation](https://github.com/ranaroussi/yfinance). The model uses the [Keras LSTM layer](https://keras.io/api/layers/recurrent_layers/lstm/) with its PyTorch backend.
