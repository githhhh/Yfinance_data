"""
Stage 2 IBD 前置过滤器
====================
双 Pass 查询，生成 Stage 2 白名单。

Pass 1 - Stage 2 主体 (普通股 + ADR):
  - Close >= SMA40|1W  (Price above 200-day MA)
  - SMA10|1W >= SMA40|1W  (50-day MA above 200-day MA, 金叉)
  - Close >= 15
  - type in [stock, dr], 排除 preferred

Pass 2 - 次新股/IPO 豁免:
  - 上市不足 40 周 (SMA40|1W 为空)
  - Close >= SMA10|1W (中短期趋势向上)
  - 周线相对成交量 >= 1.3 (有资金介入)

输出: us/stage2/stage2_whitelist.csv
"""

import os
import logging
import pandas as pd
from tradingview_screener import Query, col

WHITELIST_PATH = os.path.join("us", "stage2", "stage2_whitelist.csv")

# 公共基础过滤条件
COMMON_FILTERS = [
    col('exchange').isin(['AMEX', 'CBOE', 'NASDAQ', 'NYSE']),
    col('close') >= 15,
    col('active_symbol') == True,
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
    """双 Pass 查询: Stage2 主体 + 次新股豁免"""

    # Pass 1: Stage 2 主体（普通股 + ADR 合并查询）
    # type in [stock, dr] + 排除 preferred，ADR 天然不含 preferred typespecs
    if verbose:
        print("[Stage2] Pass 1: Stage2 主体 (普通股+ADR)...")
    count_main, df_main = (
        Query()
        .select(
            'name', 'close', 'SMA10|1W', 'SMA40|1W', 'sector', 'industry',
            'earnings_per_share_diluted_yoy_growth_fq', 'price_52_week_high',
        )
        .where(
            *COMMON_FILTERS,
            col('type').isin(['stock', 'dr']),
            col('typespecs').has_none_of('preferred'),
            col('close') >= col('SMA40|1W'),      # Price > 200-day MA
            col('SMA10|1W') >= col('SMA40|1W'),   # 50-day MA > 200-day MA (金叉)
        )
        .limit(2500)
        .set_markets('america')
        .get_scanner_data()
    )
    if verbose:
        print(f"[Stage2] Pass 1 Stage2 主体: {len(df_main)} 只")

    # Pass 2: 次新股/动能股豁免 (IPO/Momentum Exemption)
    # 上市不足 40 周，SMA40|1W 为空，与 Pass 1 天然互斥
    if verbose:
        print("[Stage2] Pass 2: 次新股豁免...")
    count_ipo, df_ipo = (
        Query()
        .select(
            'name', 'close', 'SMA10|1W', 'SMA40|1W', 'sector', 'industry',
            'earnings_per_share_diluted_yoy_growth_fq', 'price_52_week_high',
        )
        .where(
            *COMMON_FILTERS,
            col('type').isin(['stock', 'dr']),
            col('typespecs').has_none_of('preferred'),
            col('relative_volume_10d_calc|1W') >= 1.3,
            col('close') >= col('SMA10|1W'),
            col('SMA40|1W').empty(),  # 互斥逻辑：专门抓取没有 40 周均线的次新股
        )
        .limit(200)
        .set_markets('america')
        .get_scanner_data()
    )
    if verbose:
        print(f"[Stage2] Pass 2 次新股: {len(df_ipo)} 只")

    # 合并 (次新股的 SMA40|1W 全为 NaN，统一 dtype 避免 FutureWarning)
    if not df_ipo.empty and 'SMA40|1W' in df_ipo.columns:
        df_ipo['SMA40|1W'] = df_ipo['SMA40|1W'].astype(float)
    df_all = pd.concat([df_main, df_ipo], ignore_index=True)

    if df_all.empty:
        print("[Stage2] WARNING: 查询结果为空")
        _delete_old_whitelist(output_file)
        return 0, df_all, []

    df_all = df_all.rename(
        columns={
            'name': 'code',
            'earnings_per_share_diluted_yoy_growth_fq': 'eps_yoy_growth',
        }
    )

    # 去重 (Pass 1/2 天然互斥，但防御性去重)
    df_all = df_all.drop_duplicates(subset='code', keep='first')

    total = len(df_all)
    tickers_list = df_all['code'].tolist()

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 保存白名单
    df_all.to_csv(output_file, index=False)
    if verbose:
        print(f"[Stage2] 白名单已保存: {output_file} ({total} 只 = {len(df_main)} 主体 + {len(df_ipo)} 次新股)")

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
