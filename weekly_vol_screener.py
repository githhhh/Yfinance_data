"""
周线放量突破筛选器
使用 TradingView-Screener 筛选:
  - 周线级别当周成交量 >= 10周均量的 1.3倍 (即 relative_volume_10d_calc|1W >= 1.3)
  - 价格 >= 15
  - 仅美股普通股（排除 OTC / 优先股 / 基金）
"""

from tradingview_screener import Query, col

def run_screener(min_vol_ratio=1.3, min_price=15, limit=200, output_file="us/weekly_vol_screener_results.csv", verbose=True):
    """
    统一外部调用入口。
    Returns: (total_count, dataframe, tickers_list)
    """
    count, df = screen_weekly_vol_breakout(min_vol_ratio, min_price, limit, output_file, verbose)
    tickers_list = []
    if not df.empty and 'ticker' in df.columns:
        tickers_list = df['ticker'].apply(lambda x: x.split(':')[-1] if isinstance(x, str) and ':' in x else x).tolist()
    return count, df, tickers_list


def screen_weekly_vol_breakout(min_vol_ratio=1.3, min_price=15, limit=200, output_file="us/weekly_vol_screener_results.csv", verbose=True):
    """
    筛选 周线相对成交量 >= min_vol_ratio 且价格 >= min_price 的美股
    """
    if verbose:
        print("=" * 60)
        print("筛选条件:")
        print(f"  - 周线成交量/10周均量比值 (Relative Volume 10W) >= {min_vol_ratio}")
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
            'relative_volume_10d_calc|1W',
            'change|1W',
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
            col('relative_volume_10d_calc|1W') >= min_vol_ratio,
        )
        .order_by('relative_volume_10d_calc|1W', ascending=False)
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
            'relative_volume_10d_calc|1W': '周线放量比例',
            'change|1W': '周涨跌幅(%)',
            'sector': '板块',
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
