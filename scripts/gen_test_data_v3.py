import os

import yfinance as yf


def generate_data():
    # Tech/High Volatility + High Price Tickers: TSLA, NVDA, AMD, BRK-A (Assets), QQQ (Benchmark), BIL (Rf)
    tickers = ["TSLA", "NVDA", "AMD", "BRK-A", "QQQ", "BIL"]
    print(f"Fetching data for {tickers}...")

    # Download 5+ years of daily data
    raw = yf.download(tickers, start="2020-01-01", end="2026-03-01")

    # yfinance multi-index formatting workaround
    prices = raw.xs("Close", level=0, axis=1)

    # Calculate daily returns
    returns = prices.pct_change().dropna()

    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    # Save to CSV
    output_path = "data/test_data_v3.csv"
    returns.to_csv(output_path)
    print(f"Test data saved to {output_path}")


if __name__ == "__main__":
    generate_data()
