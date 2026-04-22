"""
52周新高筛选器
使用 TradingView-Screener 筛选:
  - 今日最高价 >= 52周最高价 (即创出52周新高)
  - 价格 >= 15
  - 仅美股普通股（排除 OTC / 优先股 / 基金）
"""

from tradingview_screener import Query, col


def run_screener(min_price=15, limit=200, output_file="us/52wk_new_high_results.csv", verbose=True):
    """
    统一外部调用入口。
    Returns: (total_count, dataframe, tickers_list)
    """
    count, df = screen_52wk_new_high(min_price, limit, output_file, verbose)
    tickers_list = []
    if not df.empty and 'ticker' in df.columns:
        # ticker field is typically 'EXCHANGE:SYMBOL', we extract the 'SYMBOL'
        tickers_list = df['ticker'].apply(lambda x: x.split(':')[-1] if isinstance(x, str) and ':' in x else x).tolist()
    return count, df, tickers_list


def screen_52wk_new_high(min_price=15, limit=200, output_file="us/52wk_new_high_results.csv", verbose=True):
    """
    筛选今日创出52周新高且价格 >= min_price 的美股
    """
    if verbose:
        print("=" * 60)
        print("筛选条件:")
        print(f"  - 今日最高价 >= 52周最高价 (52-Week New High)")
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
            'price_52_week_high',
            'High.All',
            'change',
            'relative_volume_10d_calc',
            'earnings_per_share_diluted_yoy_growth_fq',
            'sector',
        )
        .where(
            col('exchange').isin(['AMEX', 'CBOE', 'NASDAQ', 'NYSE']),
            col('is_primary') == True,
            col('typespecs').has('common'),
            col('typespecs').has_none_of('preferred'),
            col('type') == 'stock',
            col('close') >= min_price,
            col('active_symbol') == True,
            col('price_52_week_high') <= 'high',
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
            'price_52_week_high': '52周最高',
            'High.All': '历史最高',
            'change': '涨跌幅(%)',
            'relative_volume_10d_calc': '相对10d成交量',
            'earnings_per_share_diluted_yoy_growth_fq': 'EPS季度同比(%)',
            'sector': '板块',
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
