import os
import pickle
import time
import pandas as pd
import yfinance as yf
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

STOCK_LIST_PATH = "Indices/EQUITY_L.csv"
RESULTS_PKL_DIR = "results_pkl"
BATCH_SIZE = 220
MAX_WORKERS = 5
MAX_RETRIES = 1

def read_stock_list(stock_list_path=STOCK_LIST_PATH):
    """
    Read stock tickers from CSV file.
    Assumes the CSV has a column named 'Symbol'.
    """
    try:
        df = pd.read_csv(stock_list_path)
        tickers = df["SYMBOL"].astype(str).tolist()
        return tickers
    except Exception as e:
        print(f"Error reading stock list from {stock_list_path}: {e}")
        return []

def download_single_stock(stock_code, period, interval):
    """
    Download data for a single stock with retries.
    """
    # Append .NS suffix if not present and not an index symbol
    if not stock_code.endswith(".NS") and not stock_code.startswith("^"):
        stock_code = f"{stock_code}.NS"
    attempt = 0
    while attempt <= MAX_RETRIES:
        try:
            ticker = yf.Ticker(stock_code)
            data = ticker.history(
                period=period,
                interval=interval,
                auto_adjust=True,
                rounding=True,
                timeout=10,
            )
            if not data.empty:
                # Round data to 2 decimals as in existing code
                data = data.round(2)
                return stock_code, data
        except Exception as e:
            print(f"Error downloading {stock_code} (attempt {attempt+1}): {e}")
        attempt += 1
        time.sleep(1)
    print(f"Failed to download data for {stock_code} after {MAX_RETRIES+1} attempts.")
    return stock_code, None

def download_batch_stocks(tickers, period="1y", interval="1d"):
    """
    Download stock data in parallel per ticker (like original AssetsManager logic).
    Returns a dict of stock_code -> DataFrame.
    Adds debug output for process and speed.
    """
    import time as time_module
    from concurrent.futures import ThreadPoolExecutor, as_completed
    all_data = {}
    max_workers = 5  # Lowered to reduce rate limit risk
    max_retries = 1
    batch_size = 220
    total = len(tickers)
    print(f"[Batch Download] Starting download for {total} stocks, batch size {batch_size}, max_workers {max_workers}")
    start_time = time_module.time()
    failed = []
    for batch_start in range(0, total, batch_size):
        batch = tickers[batch_start:batch_start+batch_size]
        print(f"[Batch Download] Processing batch {batch_start//batch_size+1}: {len(batch)} stocks")
        batch_success = 0
        batch_failed = 0
        batch_start_time = time_module.time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(download_single_stock, ticker, period, interval): ticker
                for ticker in batch
            }
            for future in as_completed(future_to_ticker):
                stock_code, data = future.result()
                if data is not None:
                    all_data[stock_code] = data
                    batch_success += 1
                else:
                    failed.append(stock_code)
                    batch_failed += 1
        batch_end_time = time_module.time()
        print(f"[Batch Download] Batch finished: Downloaded {batch_success}, Failed {batch_failed} (Time: {batch_end_time - batch_start_time:.2f}s)")
        time_module.sleep(1.5 if len(batch) > 10 else 1.0)
    if failed:
        print(f"[Batch Download] Retrying {len(failed)} failed stocks...")
        for attempt in range(1, max_retries+1):
            if not failed:
                break
            retry_failed = []
            print(f"[Batch Download] Retry attempt {attempt}/{max_retries} for {len(failed)} stocks")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_ticker = {
                    executor.submit(download_single_stock, ticker, period, interval): ticker
                    for ticker in failed
                }
                for future in as_completed(future_to_ticker):
                    stock_code, data = future.result()
                    if data is not None:
                        all_data[stock_code] = data
                    else:
                        retry_failed.append(stock_code)
            failed = retry_failed
            if failed:
                print(f"[Batch Download] Still failed: {len(failed)} stocks. Waiting before next retry...")
                time_module.sleep(1.0)
    elapsed = time_module.time() - start_time
    print(f"[Batch Download] Finished: {len(all_data)} downloaded, {len(failed)} failed. Total time: {elapsed:.2f} seconds")
    return all_data, failed

def save_stock_data(stock_data, save_dir=RESULTS_PKL_DIR):
    """
    Save stock data dict to a pickle file with date and time suffix.
    Converts each DataFrame to dict in 'split' format and normalizes keys by removing '.NS' suffix.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    date_suffix = datetime.now().strftime("%d%m%y_%H%M%S")
    filename = f"stock_data_{date_suffix}.pkl"
    filepath = os.path.join(save_dir, filename)
    try:
        # Convert DataFrames to dicts in 'split' format and normalize keys
        converted_data = {}
        for k, v in stock_data.items():
            new_key = k[:-3] if k.endswith(".NS") else k
            if hasattr(v, "to_dict"):
                # Convert index to ISO8601 strings with timezone info to preserve tz
                df_copy = v.copy()
                # Convert index to pandas Timestamps with Asia/Kolkata timezone to preserve tz info
                df_copy = v.copy()
                if not isinstance(df_copy.index.dtype, pd.DatetimeTZDtype):
                    df_copy.index = pd.to_datetime(df_copy.index).tz_localize('Asia/Kolkata', ambiguous='NaT', nonexistent='shift_forward')
                converted_data[new_key] = df_copy.to_dict("split")
            else:
                converted_data[new_key] = v
        with open(filepath, "wb") as f:
            pickle.dump(converted_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved stock data for {len(converted_data)} tickers to {filepath}")
        return filepath
    except Exception as e:
        print(f"Error saving pickle file: {e}")
        return None

def load_stock_data(pickle_path):
    """
    Load stock data dict from pickle file and convert dicts in 'split' format to DataFrames if needed.
    """
    if not os.path.exists(pickle_path):
        print(f"Pickle file {pickle_path} does not exist.")
        return {}
    try:
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
        # Convert dicts in 'split' format to DataFrames
        for k, v in data.items():
            if isinstance(v, dict) and set(v.keys()) == {"index", "columns", "data"}:
                data[k] = pd.DataFrame(**v)
        print(f"Loaded stock data for {len(data)} tickers from {pickle_path}")
        return data
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return {}

if __name__ == "__main__":
    tickers = read_stock_list()
    if not tickers:
        print("No tickers to download.")
    else:
        # Example: download 1 year daily data
        stock_data, failed = download_batch_stocks(tickers, period="1y", interval="1d")
        save_path = save_stock_data(stock_data)
        loaded_data = load_stock_data(save_path) if save_path else None
