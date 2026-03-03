import yfinance as yf
import pandas as pd
import os

def generate_data():
    # Define tickers: SPY (S&P 500), AGG (Bonds), GLD (Gold), BIL (1-3 Month T-Bill as Rf proxy)
    tickers = ["SPY", "AGG", "GLD", "BIL"]
    print(f"Fetching data for {tickers}...")
    
    # Download 5 years of daily data
    # yf.download returns a MultiIndex (Price, Ticker) by default if multiple tickers
    data = yf.download(tickers, start="2019-01-01", end="2024-01-01")
    
    # Try to get 'Adj Close', fallback to 'Close'
    if 'Adj Close' in data.columns.levels[0]:
        prices = data['Adj Close']
    else:
        prices = data['Close']
    
    # Calculate daily returns
    returns = prices.pct_change().dropna()
    
    # Rename for clarity if needed (ticker symbols are already column names)
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Save to CSV
    output_path = "data/test_data.csv"
    returns.to_csv(output_path)
    print(f"Test data saved to {output_path}")
    print("\nFirst few rows:")
    print(returns.head())
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Save to CSV
    output_path = "data/test_data.csv"
    returns.to_csv(output_path)
    print(f"Test data saved to {output_path}")
    print("\nFirst few rows:")
    print(returns.head())

if __name__ == "__main__":
    generate_data()
