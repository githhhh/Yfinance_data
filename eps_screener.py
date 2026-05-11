"""
EPS 季度同比增速筛选器 (周线级别)
使用 TradingView-Screener 筛选:
  - EPS Diluted (Quarterly YoY Growth) >= 150%
  - 价格 >= 15
  - 所有行情数据均为周线级别 (|1W)
"""

from tradingview_screener import Query, col

def run_screener(min_eps_growth=150, min_price=15, limit=200, output_file="us/eps_growth_screener_results.csv", verbose=True):
    """
    统一外部调用入口。
    Returns: (total_count, dataframe, tickers_list)
    """
    count, df = screen_high_eps_growth(min_eps_growth, min_price, limit, output_file, verbose)
    tickers_list = []
    if not df.empty and 'ticker' in df.columns:
        # ticker field is typically 'EXCHANGE:SYMBOL', we extract the 'SYMBOL'
        tickers_list = df['ticker'].apply(lambda x: x.split(':')[-1] if isinstance(x, str) and ':' in x else x).tolist()
    return count, df, tickers_list


def screen_high_eps_growth(min_eps_growth=150, min_price=15, limit=200, output_file="us/eps_growth_screener_results.csv", verbose=True):
    """
    筛选 EPS 季度同比增速 >= min_eps_growth 且价格 >= min_price 的美股
    """
    if verbose:
        print("=" * 60)
        print("筛选条件:")
        print(f"  - EPS Diluted (Quarterly YoY Growth) >= {min_eps_growth}%")
        print(f"  - 价格 >= ${min_price}")
        print("  - 仅美股普通股（排除 OTC / 优先股 / 基金）")
        print("=" * 60)

    count, df = (
        Query()
        .select(
            'name',
            'earnings_per_share_diluted_yoy_growth_fq',
            'relative_volume_10d_calc|1W',
            'close',
            'open|1W',
            'high|1W',
            'low|1W',
            'volume|1W',
            'market_cap_basic',
            'sector',
            'industry',
        )
        .where(
            col('exchange').isin(['AMEX', 'CBOE', 'NASDAQ', 'NYSE']),
            col('is_primary') == True,
            col('typespecs').has('common'),
            col('typespecs').has_none_of('preferred'),
            col('type') == 'stock',
            col('close') >= min_price,
            col('active_symbol') == True,
            col('earnings_per_share_diluted_yoy_growth_fq') >= min_eps_growth,
        )
        .order_by('earnings_per_share_diluted_yoy_growth_fq', ascending=False)
        .limit(limit)
        .set_markets('america')
        .get_scanner_data()
    )

    if verbose:
        print(f"\n总共找到 {count} 只符合条件的股票，返回 {len(df)} 只\n")

    if not df.empty:
        if 'name' in df.columns:
            df = df.rename(columns={
                'name': 'code',
                'earnings_per_share_diluted_yoy_growth_fq': 'eps_growth',
                'relative_volume_10d_calc|1W': 'vol_ratio_1w'
            })
            
        df_display = df.rename(columns={
            'code': '代码',
            'eps_growth': 'EPS季度同比(%)',
            'vol_ratio_1w': '周相对成交量',
            'close': '价格',
            'open|1W': '周开盘',
            'high|1W': '周最高',
            'low|1W': '周最低',
            'volume|1W': '周成交量',
            'market_cap_basic': '市值',
            'sector': '板块',
            'industry': '子行业',
        })
        if verbose:
            print(df_display.to_string(index=False))

    # 保存到 CSV (无结果时也要覆盖旧文件)
    df.to_csv(output_file, index=False)
    if verbose:
        print(f"\n结果已保存到: {output_file}")

    return count, df


if __name__ == "__main__":
    run_screener()
