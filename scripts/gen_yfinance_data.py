import yfinance as yf

def main():
    tickers = ['SPY', 'GLD', 'TLT']
    print(f"Downloading {tickers} from Yahoo Finance...")
    raw = yf.download(tickers, start="2015-01-01", end="2026-03-01")
    # yfinance returns MultiIndex columns: level 0 is 'Price', level 1 is 'Ticker'
    data = raw.xs('Close', level=0, axis=1)
    
    # Resample to monthly to keep test fast and robust
    monthly = data.resample('ME').last()
    returns = monthly.pct_change().dropna()
    
    # Save returns to CSV
    # Ensure index name is empty to match R expectations easily
    returns.index.name = ""
    returns.to_csv('data/yfinance_etfs.csv')
    print("Saved data/yfinance_etfs.csv with shape:", returns.shape)

if __name__ == "__main__":
    main()
