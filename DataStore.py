import os
import pickle
import time
import pandas as pd
import yfinance as yf
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from eps_screener import run_screener as run_eps_screener
from stage2_screener import run_screener as run_stage2_screener, load_whitelist, WHITELIST_PATH
from data_providers import DataProviderFactory, YahooDataProvider

FILTER_SNAPSHOT_PATH = os.path.join("us", "stage2", "stage2_screener_filter.csv")
from importlib import import_module


RESULTS_PKL_DIR = "results_pkl"
BATCH_SIZE = 100          # smaller batches keep Yahoo responsive
MAX_WORKERS = 8         # more threads = faster, until Yahoo rate-limits
MAX_RETRIES = 1          # retry failed tickers a couple of times

def read_stock_list(stock_list_dir="us"):
    """Read and merge all CSV data sources under stock_list_dir.
    
    Convention: all CSV files must have a `code` column containing clean ticker symbols.
    """
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
    
    print(f"[Merge] Total unique tickers after dedup: {len(final_list)}")
    return final_list

def download_single_stock(stock_code, period, interval):
    """Download data for a single stock using default Yahoo provider (legacy alias)."""
    provider = YahooDataProvider(batch_size=BATCH_SIZE, max_workers=MAX_WORKERS, max_retries=MAX_RETRIES)
    return provider.download_single_stock(stock_code, period=period, interval=interval)

def download_batch_stocks(tickers, period="1y", interval="1d"):
    """Download stock data in parallel batches using default Yahoo provider (legacy alias)."""
    provider = YahooDataProvider(batch_size=BATCH_SIZE, max_workers=MAX_WORKERS, max_retries=MAX_RETRIES)
    return provider.download_batch_stocks(tickers, period=period, interval=interval)

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
    parser.add_argument('--provider', default='yahoo', choices=['yahoo', 'schwab'], help='Data provider to use (yahoo, schwab)')
    parser.add_argument('--app-key', default=None, help='Schwab App Key (Client ID)')
    parser.add_argument('--app-secret', default=None, help='Schwab App Secret')
    parser.add_argument('--token-path', default=None, help='Path to schwab OAuth token json file')
    parser.add_argument('--callback-url', default=None, help='Schwab App Callback / Redirect URL')
    parser.add_argument('--period', default='2y', help='Data period (1y, 2y, etc.)')
    parser.add_argument('--interval', default='1d', help='Data interval (1d, 1wk, etc.)')
    parser.add_argument('--skip-screener', action='store_true', help='Skip the screener step')
    parser.add_argument('--screener-only', action='store_true', help='Only run the screener and merge, do not download')
    parser.add_argument('--min-eps-growth', type=int, default=150, help='Minimum EPS YoY growth for the screener')
    args = parser.parse_args()

    if not args.skip_screener:
        # --- Stage2 Screener (必须先跑) ---
        print("\n[DataStore] Running Stage2 IBD Screener (must run first)...")
        count_s2, df_s2, tickers_s2 = run_stage2_screener(verbose=False)
        print(f"[DataStore] Stage2 Screener complete. Found {count_s2} stocks.\n")

        # --- EPS Screener (豁免，不做 Stage2 过滤) ---
        print(f"\n[DataStore] Running EPS Screener (min_eps_growth>={args.min_eps_growth}%) before download...")
        count, df, new_tickers = run_eps_screener(min_eps_growth=args.min_eps_growth, verbose=False)
        print(f"[DataStore] EPS Screener complete. Found {count} stocks.\n")

        # --- 52-Week New High Screener ---
        try:
            from importlib import import_module
            mod_52wk = import_module("52_wk_new_high_screener")
            print("[DataStore] Running 52-Week New High Screener...")
            count_52, df_52, tickers_52 = mod_52wk.run_screener(verbose=False)
            print(f"[DataStore] 52-Week New High Screener complete. Found {count_52} stocks (after Stage2 filter).\n")
        except Exception as e:
            print(f"[DataStore] 52-Week New High Screener failed: {e}\n")

        # --- Weekly Volume Breakout Screener ---
        try:
            from importlib import import_module
            mod_weekly_vol = import_module("weekly_vol_screener")
            print("[DataStore] Running Weekly Volume Breakout Screener...")
            count_vol, df_vol, tickers_vol = mod_weekly_vol.run_screener(verbose=False)
            print(f"[DataStore] Weekly Volume Breakout Screener complete. Found {count_vol} stocks (after Stage2 filter).\n")
        except Exception as e:
            print(f"[DataStore] Weekly Volume Breakout Screener failed: {e}\n")
    else:
        print("\n[DataStore] Skipping Screener phase (--skip-screener specified).\n")

    if args.screener_only:
        print("[DataStore] Exiting early (--screener-only specified).")
        exit(0)

    # Fallback 机制: 检查 Stage2 白名单是否存在
    stage2_available = load_whitelist() is not None
    if stage2_available:
        # 正常路径: glob union (screener CSVs 已被 Stage2 过滤)
        tickers = read_stock_list()
        if tickers:
            # 写入 Fallback 快照
            os.makedirs(os.path.dirname(FILTER_SNAPSHOT_PATH), exist_ok=True)
            pd.DataFrame({'code': tickers}).to_csv(FILTER_SNAPSHOT_PATH, index=False)
            print(f"[DataStore] Fallback 快照已更新: {FILTER_SNAPSHOT_PATH} ({len(tickers)} 只)")
    else:
        # Fallback 路径: Stage2 失败，读取上次快照
        print(f"[DataStore] ⚠️  Stage2 白名单不存在，启用 Fallback...")
        if os.path.exists(FILTER_SNAPSHOT_PATH):
            fallback_df = pd.read_csv(FILTER_SNAPSHOT_PATH)
            tickers = fallback_df['code'].dropna().astype(str).tolist()
            if len(tickers) > 1500:
                tickers = tickers[:1500]
            print(f"[DataStore] Fallback 读取 {FILTER_SNAPSHOT_PATH}: {len(tickers)} 只")
        else:
            tickers = read_stock_list()
            print(f"[DataStore] Fallback 快照不存在，使用 glob union: {len(tickers)} 只")

    if not tickers:
        print("No tickers to download.")
    else:
        # 静默加入指数，确保不写入任何 CSV 中
        index_tickers = ["^GSPC", "^IXIC", "^DJI"]
        for idx in reversed(index_tickers):
            if idx in tickers:
                tickers.remove(idx)
            tickers.insert(0, idx)

        provider = DataProviderFactory.get_provider(
            provider_type=args.provider,
            app_key=args.app_key,
            app_secret=args.app_secret,
            token_path=args.token_path,
            callback_url=args.callback_url,
        )
        stock_data, failed = provider.download_batch_stocks(tickers, period=args.period, interval=args.interval)
        save_path = save_stock_data(stock_data, interval=args.interval)
        loaded_data = load_stock_data(save_path) if save_path else None