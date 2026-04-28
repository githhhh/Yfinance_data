"""
Stage 2 IBD 前置过滤器
====================
双 Pass 联合查询（普通股 + ADR），生成 Stage 2 白名单。

条件 (IBD Method B):
  - Close >= SMA40|1W  (Price above 200-day MA)
  - SMA10|1W >= SMA40|1W  (50-day MA above 200-day MA, 金叉)
  - Close >= 15
  - 普通股: type=stock, typespecs=common, !preferred
  - ADR: type=dr (不限 typespecs)

输出: us/stage2/stage2_whitelist.csv
"""

import os
import logging
import pandas as pd
from tradingview_screener import Query, col

WHITELIST_PATH = os.path.join("us", "stage2", "stage2_whitelist.csv")

# Stage 2 均线条件 (IBD Method B)
BASE_FILTERS = [
    col('exchange').isin(['AMEX', 'CBOE', 'NASDAQ', 'NYSE']),
    col('close') >= 15,
    col('active_symbol') == True,
    col('close') >= col('SMA40|1W'),      # Price > 200-day MA
    col('SMA10|1W') >= col('SMA40|1W'),   # 50-day MA > 200-day MA (金叉)
]


def run_screener(output_file=WHITELIST_PATH, verbose=True):
    """
    统一外部调用入口。
    Returns: (total_count, dataframe, tickers_list)
    """
    try:
        count, df, tickers = _query_stage2(output_file, verbose)
        return count, df, tickers
    except Exception as e:
        print(f"[Stage2] Screener 失败: {e}")
        # 失败时主动删除旧白名单，触发下游 Fallback
        _delete_old_whitelist(output_file)
        return 0, pd.DataFrame(), []


def _query_stage2(output_file, verbose):
    """双 Pass 联合查询: 普通股 + ADR"""

    # Pass 1: 普通股（不限 is_primary）
    if verbose:
        print("[Stage2] Pass 1: 普通股查询...")
    count_stock, df_stock = (
        Query()
        .select('name', 'close', 'SMA10|1W', 'SMA40|1W')
        .where(
            *BASE_FILTERS,
            col('typespecs').has('common'),
            col('typespecs').has_none_of('preferred'),
            col('type') == 'stock',
        )
        .limit(2000)
        .set_markets('america')
        .get_scanner_data()
    )
    if verbose:
        print(f"[Stage2] Pass 1 普通股: {len(df_stock)} 只")

    # Pass 2: ADR（不要求 typespecs common）
    if verbose:
        print("[Stage2] Pass 2: ADR 查询...")
    count_dr, df_dr = (
        Query()
        .select('name', 'close', 'SMA10|1W', 'SMA40|1W')
        .where(
            *BASE_FILTERS,
            col('type') == 'dr',
        )
        .limit(500)
        .set_markets('america')
        .get_scanner_data()
    )
    if verbose:
        print(f"[Stage2] Pass 2 ADR: {len(df_dr)} 只")

    # 合并
    df_all = pd.concat([df_stock, df_dr], ignore_index=True)

    if df_all.empty:
        print("[Stage2] WARNING: 查询结果为空")
        _delete_old_whitelist(output_file)
        return 0, df_all, []

    if 'name' in df_all.columns:
        df_all = df_all.rename(columns={'name': 'code'})

    total = len(df_all)
    tickers_list = df_all['code'].tolist()

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 保存白名单
    df_all.to_csv(output_file, index=False)
    if verbose:
        print(f"[Stage2] 白名单已保存: {output_file} ({total} 只 = {len(df_stock)} 普通股 + {len(df_dr)} ADR)")

    return total, df_all, tickers_list


def load_whitelist(whitelist_path=WHITELIST_PATH):
    """加载白名单，返回 code 集合。文件不存在返回 None（表示 Stage2 失败）。"""
    if not os.path.exists(whitelist_path):
        return None
    try:
        df = pd.read_csv(whitelist_path)
        if 'code' not in df.columns:
            print(f"[Stage2] WARNING: {whitelist_path} 无 'code' 列")
            return None
        codes = set(df['code'].dropna().astype(str).str.strip())
        return codes if codes else None
    except Exception as e:
        print(f"[Stage2] WARNING: 读取白名单失败: {e}")
        return None


def _delete_old_whitelist(path=WHITELIST_PATH):
    """删除旧白名单，确保下游触发 Fallback"""
    if os.path.exists(path):
        os.remove(path)
        print(f"[Stage2] 已删除旧白名单: {path}")


if __name__ == "__main__":
    count, df, tickers = run_screener()
    print(f"\n[Stage2] 完成: {count} 只")
