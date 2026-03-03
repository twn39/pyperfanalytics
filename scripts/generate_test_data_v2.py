import yfinance as yf
import pandas as pd
import os

def generate_data():
    # New Diversified Tickers: QQQ (Growth), IWM (Small Cap), EEM (Emerging Markets), BIL (Rf)
    tickers = ["QQQ", "IWM", "EEM", "BIL"]
    print(f"Fetching data for {tickers}...")
    
    # Download 5 years of daily data
    data = yf.download(tickers, start="2019-01-01", end="2024-01-01")
    
    # Try to get 'Adj Close', fallback to 'Close'
    if 'Adj Close' in data.columns.levels[0]:
        prices = data['Adj Close']
    else:
        prices = data['Close']
    
    # Calculate daily returns
    returns = prices.pct_change().dropna()
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Save to CSV
    output_path = "data/test_data_v2.csv"
    returns.to_csv(output_path)
    print(f"Test data saved to {output_path}")
    print("\nFirst few rows:")
    print(returns.head())

if __name__ == "__main__":
    generate_data()
