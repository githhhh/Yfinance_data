import os
import pickle
import time
import pandas as pd
import yfinance as yf
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from eps_screener import run_screener as run_eps_screener
from importlib import import_module


RESULTS_PKL_DIR = "results_pkl"
BATCH_SIZE = 100          # smaller batches keep Yahoo responsive
MAX_WORKERS = 8         # more threads = faster, until Yahoo rate-limits
MAX_RETRIES = 1          # retry failed tickers a couple of times

def read_stock_list(stock_list_dir="us", index_tickers=None):
    """Read and merge all CSV data sources under stock_list_dir.
    
    Convention: all CSV files must have a `code` column containing clean ticker symbols.
    """
    if index_tickers is None:
        index_tickers = ["^GSPC", "^IXIC", "^DJI"]
        
    tickers = set()
    merged_sources = []
    
    # Auto-discover and merge ALL CSV data sources under us/
    import glob
    csv_files = sorted(glob.glob(os.path.join(stock_list_dir, "*.csv")))
    
    if not csv_files:
        print(f"[Merge] WARNING: No CSV files found in {stock_list_dir}/")
    
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            if 'code' not in df.columns:
                print(f"[Merge] ⚠️  WARNING: {csv_path} has no 'code' column — skipped (columns found: {list(df.columns)})")
                continue
            codes = df['code'].dropna().astype(str).tolist()
            if not codes:
                print(f"[Merge] {csv_path}: 'code' column is empty — skipped")
                continue
            tickers.update([t.replace(".", "-") for t in codes])
            merged_sources.append(f"{os.path.basename(csv_path)} ({len(codes)} tickers)")
        except Exception as e:
            print(f"[Merge] ⚠️  ERROR reading {csv_path}: {e}")
    
    # Print merge summary
    print(f"\n[Merge] === Data Source Summary ===")
    for src in merged_sources:
        print(f"[Merge]   ✓ {src}")
    
    final_list = list(tickers)
    
    # Insert index tickers at the beginning
    for idx_ticker in reversed(index_tickers):
        if idx_ticker in final_list:
            final_list.remove(idx_ticker)
        final_list.insert(0, idx_ticker)
        
    print(f"[Merge] Total unique tickers after dedup: {len(final_list)} (including {len(index_tickers)} index tickers)")
    return final_list

def download_single_stock(stock_code, period, interval):
    """Download data for a single stock with retries."""
    attempt = 0
    while attempt <= MAX_RETRIES:
        try:
            ticker = yf.Ticker(stock_code)
            data = ticker.history(
                period=period,
                interval=interval,
                auto_adjust=True,
                rounding=True,
                timeout=5,
            )
            if not data.empty:
                return stock_code, data.round(2)
        except Exception as e:
            print(f"Error downloading {stock_code} (attempt {attempt+1}): {e}")
        attempt += 1
        time.sleep(0.5 * attempt)  # exponential backoff
    return stock_code, None

def download_batch_stocks(tickers, period="1y", interval="1d"):
    """Download stock data in parallel batches with retries and timing per batch."""
    all_data = {}
    failed = []
    total = len(tickers)
    print(f"[Batch Download] Starting download for {total} stocks, batch size {BATCH_SIZE}, workers {MAX_WORKERS}")
    overall_start = time.time()

    for batch_start in range(0, total, BATCH_SIZE):
        batch = tickers[batch_start:batch_start+BATCH_SIZE]
        print(f"[Batch Download] Processing batch {batch_start//BATCH_SIZE+1}: {len(batch)} stocks")
        batch_start_time = time.time()
        batch_success = 0
        batch_failed = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
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

        batch_end_time = time.time()
        print(f"[Batch Download] Batch finished: Downloaded {batch_success}, Failed {batch_failed} "
              f"(Time: {batch_end_time - batch_start_time:.2f}s)")

    # Retry failed tickers once more
    if failed:
        print(f"[Batch Download] Retrying {len(failed)} failed stocks...")
        retry_failed = []
        retry_start_time = time.time()
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
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
        retry_end_time = time.time()
        print(f"[Batch Download] Retry finished: "
              f"Recovered {len(failed) - len(retry_failed)}, Still failed {len(retry_failed)} "
              f"(Time: {retry_end_time - retry_start_time:.2f}s)")
        failed = retry_failed

    overall_end = time.time()
    print(f"[Batch Download] Finished: {len(all_data)} downloaded, {len(failed)} failed. "
          f"Total time: {overall_end - overall_start:.2f} seconds")
    return all_data, failed

def save_stock_data(stock_data, save_dir=RESULTS_PKL_DIR, interval="1d"):
    """Save stock data dict to a pickle file."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    filepath = get_stock_pkl_path(interval)
    try:
        converted_data = {}
        for k, v in stock_data.items():
            # 美股不需要去掉.NS后缀，因为我们已经修改了read_stock_list
            new_key = k
            if hasattr(v, "to_dict"):
                df_copy = v.copy()
                if not isinstance(df_copy.index.dtype, pd.DatetimeTZDtype):
                    # 美股使用美国东部时区
                    df_copy.index = pd.to_datetime(df_copy.index).tz_localize(
                        "US/Eastern", ambiguous="NaT", nonexistent="shift_forward"
                    )
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
    """Load stock data dict from pickle file and convert dicts in 'split' format to DataFrames if needed."""
    if not os.path.exists(pickle_path):
        print(f"Pickle file {pickle_path} does not exist.")
        return {}
    try:
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
        for k, v in data.items():
            if isinstance(v, dict) and set(v.keys()) == {"index", "columns", "data"}:
                data[k] = pd.DataFrame(**v)
        print(f"Loaded stock data for {len(data)} tickers from {pickle_path}")
        return data
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return {}

def get_stock_pkl_path(interval="1d"):
    date_suffix = datetime.now().strftime("%d%m%y")
    filename = f"stock_data_{date_suffix}_{interval}.pkl"
    filepath = os.path.join(RESULTS_PKL_DIR, filename)
    return filepath

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download stock data')
    parser.add_argument('--period', default='2y', help='Data period (1y, 2y, etc.)')
    parser.add_argument('--interval', default='1d', help='Data interval (1d, 1wk, etc.)')
    parser.add_argument('--skip-screener', action='store_true', help='Skip the screener step')
    parser.add_argument('--screener-only', action='store_true', help='Only run the screener and merge, do not download')
    parser.add_argument('--min-eps-growth', type=int, default=150, help='Minimum EPS YoY growth for the screener')
    args = parser.parse_args()

    if not args.skip_screener:
        # --- EPS Screener ---
        print(f"\n[DataStore] Running EPS Screener (min_eps_growth>={args.min_eps_growth}%) before download...")
        count, df, new_tickers = run_eps_screener(min_eps_growth=args.min_eps_growth, verbose=False)
        print(f"[DataStore] EPS Screener complete. Found {count} stocks.\n")

        # --- 52-Week New High Screener ---
        try:
            from importlib import import_module
            mod_52wk = import_module("52_wk_new_high_screener")
            print("[DataStore] Running 52-Week New High Screener...")
            count_52, df_52, tickers_52 = mod_52wk.run_screener(verbose=False)
            print(f"[DataStore] 52-Week New High Screener complete. Found {count_52} stocks.\n")
        except Exception as e:
            print(f"[DataStore] 52-Week New High Screener failed: {e}\n")
    else:
        print("\n[DataStore] Skipping Screener phase (--skip-screener specified).\n")

    if args.screener_only:
        print("[DataStore] Exiting early (--screener-only specified).")
        exit(0)

    tickers = read_stock_list()
    if not tickers:
        print("No tickers to download.")
    else:
        stock_data, failed = download_batch_stocks(tickers, period=args.period, interval=args.interval)
        save_path = save_stock_data(stock_data, interval=args.interval)
        loaded_data = load_stock_data(save_path) if save_path else None