import yfinance as yf
import os

def generate_data():
    # Define tickers: SPY (S&P 500), AGG (Bonds), GLD (Gold), BIL (1-3 Month T-Bill as Rf proxy)
    tickers = ["SPY", "AGG", "GLD", "BIL"]
    print(f"Fetching data for {tickers}...")
    
    # Download 5 years of daily data
    # yf.download returns a MultiIndex (Price, Ticker) by default if multiple tickers
    raw = yf.download(tickers, start="2019-01-01", end="2026-03-01")
    
    # Try to get 'Adj Close', fallback to 'Close'
    prices = raw.xs('Close', level=0, axis=1)
    
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
