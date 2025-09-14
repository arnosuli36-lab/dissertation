
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime
from tqdm import tqdm

# Colab file utilities (safe import; only used if available)
try:
    from google.colab import files
    COLAB = True
except Exception:
    COLAB = False

# -------------------------------
# Section 1: Load FTSE ALL SHARE list and try to get VFTSE
# -------------------------------

if COLAB:
    print("Please upload your FTSE ALL SHARE Excel file (e.g., FTSE ALL SHARE.xlsx)")
    uploaded = files.upload()
    ftse_all_share = pd.read_excel(list(uploaded.keys())[0])
else:
    # If not using Colab, set your local path here:
    # ftse_all_share = pd.read_excel("FTSE ALL SHARE.xlsx")
    ftse_all_share = pd.DataFrame()

# Read ticker symbols from the first column if available
if not ftse_all_share.empty:
    stock_symbols = ftse_all_share.iloc[:, 0].tolist()
    print(f"Loaded {len(stock_symbols)} tickers")
else:
    stock_symbols = []
    print("No FTSE ALL SHARE file loaded (not running in Colab or path not set).")

def get_vftse_data():
    """Try several possible Yahoo symbols for the VFTSE; fall back to proxy if unavailable."""
    possible_symbols = ['^VFTSE', 'VFTSE.L', 'VFTSE.IL']
    for symbol in possible_symbols:
        try:
            data = yf.download(symbol, start="2014-01-01", end="2024-12-31", progress=False)
            if not data.empty:
                print(f"Successfully fetched VFTSE data using symbol: {symbol}")
                return data
        except Exception:
            continue

    print("Unable to fetch VFTSE directly from Yahoo Finance; will compute FTSE 100 historical volatility as a proxy.")
    return None

# Get or compute volatility data
vftse_data = get_vftse_data()

if vftse_data is None:
    # If VFTSE is unavailable, compute FTSE 100 historical volatility as a proxy
    ftse100 = yf.download('^FTSE', start="2014-01-01", end="2024-12-31", progress=False)

    # Compute 30-day rolling volatility (annualized, in %)
    ftse100['Returns'] = np.log(ftse100['Close'] / ftse100['Close'].shift(1))
    ftse100['Volatility_30D'] = ftse100['Returns'].rolling(window=30).std() * np.sqrt(252) * 100

    vftse_data = ftse100[['Volatility_30D']].copy()
    vftse_data.columns = ['Close']  # unify name for downstream usage

# Save VFTSE (or proxy) data
vftse_csv = 'VFTSE_2014_2024.csv'
vftse_data.to_csv(vftse_csv)
print(f"VFTSE data saved as {vftse_csv}")
if COLAB:
    files.download(vftse_csv)

# -------------------------------
# Section 2: Alternative FTSE100 volatility export
# -------------------------------

ftse100_alt = yf.download('^FTSE', start="2014-01-01", end="2024-12-31", progress=False)
ftse100_alt['Returns'] = np.log(ftse100_alt['Close'] / ftse100_alt['Close'].shift(1))
ftse100_alt['Volatility_30D'] = ftse100_alt['Returns'].rolling(window=30).std() * np.sqrt(252) * 100

vftse_alternative = ftse100_alt[['Volatility_30D']].copy()
vftse_alternative.columns = ['VFTSE_Alternative']

alt_csv = 'FTSE100_Volatility_2014_2024.csv'
vftse_alternative.to_csv(alt_csv)
print(f"FTSE 100 historical volatility saved as {alt_csv}")
print(vftse_alternative.head())
if COLAB:
    files.download(alt_csv)

# -------------------------------
# Section 3: Bulk download for FTSE ALL SHARE tickers
# -------------------------------

# Append ".L" for LSE listings if we have tickers
if stock_symbols:
    stock_symbols = [s if s.endswith('.L') else s + '.L' for s in stock_symbols]
    print(f"Loaded {len(stock_symbols)} tickers after adding .L where necessary")
    print("First 10 sample tickers:", stock_symbols[:10])

def check_available_symbols(symbols):
    """Return (available, unavailable) based on whether Yahoo returns any data."""
    available = []
    unavailable = []

    print("Checking ticker data availability...")
    for symbol in tqdm(symbols):
        try:
            data = yf.download(symbol, start="2014-01-01", end="2014-01-10", progress=False)  # small sample to test
            if not data.empty:
                available.append(symbol)
            else:
                unavailable.append(symbol)
        except Exception:
            unavailable.append(symbol)

    print(f"\nCheck completed: {len(available)} available, {len(unavailable)} unavailable")
    return available, unavailable

if stock_symbols:
    available_symbols, unavailable_symbols = check_available_symbols(stock_symbols)

    # Save the list of unavailable tickers (if any)
    if unavailable_symbols:
        unavail_csv = 'unavailable_stocks.csv'
        pd.DataFrame({'Unavailable ticker symbols': unavailable_symbols}).to_csv(unavail_csv, index=False)
        if COLAB:
            files.download(unavail_csv)
        print(f"Saved the list of unavailable tickers as {unavail_csv}")

    # Download full data for available tickers
    print(f"Starting full data download for {len(available_symbols)} tickers (2014–2024)...")

    # Initialize data containers
    data_dict = {
        'Open': pd.DataFrame(),
        'Close': pd.DataFrame(),
        'High': pd.DataFrame(),
        'Low': pd.DataFrame(),
        'Volume': pd.DataFrame()
    }

    batch_size = 50
    for i in tqdm(range(0, len(available_symbols), batch_size)):
        batch = available_symbols[i:i + batch_size]
        try:
            data = yf.download(batch, start="2014-01-01", end="2024-12-31", group_by='ticker', progress=False)

            # Process each ticker's data
            for symbol in batch:
                if symbol in data.columns.get_level_values(0):
                    for var in ['Open', 'Close', 'High', 'Low', 'Volume']:
                        if var in data[symbol].columns:
                            if data_dict[var].empty:
                                data_dict[var] = pd.DataFrame(index=data.index)
                            data_dict[var][symbol] = data[symbol][var]
        except Exception as e:
            print(f"Error while downloading batch {i // batch_size + 1}: {str(e)}")
            continue

    print("\nData download complete!")

    # Save data to CSV files
    print("Saving data...")
    if not any(not df.empty for df in data_dict.values()):
        print("Error: No data successfully downloaded")
    else:
        for var, df in data_dict.items():
            if not df.empty:
                # Clean column names (remove .L suffix)
                df.columns = [col.replace('.L', '') for col in df.columns]

                # Save CSV
                filename = f'FTSE_{var}_2014_2024.csv'
                df.to_csv(filename)
                if COLAB:
                    files.download(filename)
                print(f"Saved {filename} (includes {len(df.columns)} tickers)")

print("All done")
