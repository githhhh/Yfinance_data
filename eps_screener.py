"""
EPS 季度同比增速筛选器
使用 TradingView-Screener 筛选:
  - EPS Diluted (Quarterly YoY Growth) >= 150%
  - 价格 >= 15
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
            'close',
            'volume',
            'market_cap_basic',
            'earnings_per_share_diluted_yoy_growth_fq',
            'relative_volume_10d_calc',
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
            df = df.rename(columns={'name': 'code'})
            
        df_display = df.rename(columns={
            'code': '代码',
            'close': '价格',
            'volume': '成交量',
            'market_cap_basic': '市值',
            'earnings_per_share_diluted_yoy_growth_fq': 'EPS季度同比(%)',
            'relative_volume_10d_calc': '相对10d成交量',
        })
        if verbose:
            print(df_display.to_string(index=False))

        # 保存到 CSV
        df.to_csv(output_file, index=False)
        if verbose:
            print(f"\n结果已保存到: {output_file}")

    return count, df


if __name__ == "__main__":
    run_screener()
