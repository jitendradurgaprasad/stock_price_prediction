"""Train the stock forecasting example and write its outputs beside this file."""

import argparse
from pathlib import Path

from stock_lstm import train_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a one-session-ahead stock close LSTM.")
    parser.add_argument("--ticker", default="AAPL", help="Yahoo Finance ticker, e.g. AAPL or MSFT")
    parser.add_argument("--period", default="5y", help="Yahoo Finance history period, e.g. 2y, 5y, max")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    result = train_pipeline(args.ticker, args.period, Path(__file__).resolve().parent,
                            epochs=args.epochs, verbose=1)
    print("\nTest metrics")
    for key, value in result["metrics"].items():
        print(f"{key}: {value}")
    print("\nNext-session estimate")
    print(result["forecast"])


if __name__ == "__main__":
    main()
