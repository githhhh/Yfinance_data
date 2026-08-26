import os
from pathlib import Path

import pandas as pd
import pytest

from data_providers.schwab_provider import SchwabCredentials, SchwabDataProvider
from data_providers.yahoo_provider import YahooDataProvider


RUN_FLAG = "RUN_LIVE_PROVIDER_SMOKE"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUANT_TRADE_DIR = PROJECT_ROOT / ".." / "quant_trade"
SPLIT_ADJUSTED_SYMBOLS = ("NVDA", "TSLA")
NON_RECENT_SPLIT_SYMBOLS = ("MSFT", "JPM")
PRICE_COLUMNS = ("Open", "High", "Low", "Close")
SPLIT_WINDOWS = {
    "NVDA": ("2024-05-01", "2024-05-31", 200.0),
    "TSLA": ("2022-08-01", "2022-08-24", 500.0),
}


def _read_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _quant_trade_env() -> dict[str, str]:
    repo_dir = Path(os.environ.get("QUANT_TRADE_REPO", DEFAULT_QUANT_TRADE_DIR))
    return _read_dotenv(repo_dir / ".env")


def _env_value(dotenv: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = os.environ.get(key) or dotenv.get(key)
        if value:
            return value
    return ""


def _schwab_credentials(dotenv: dict[str, str]) -> SchwabCredentials:
    repo_dir = Path(os.environ.get("QUANT_TRADE_REPO", DEFAULT_QUANT_TRADE_DIR))
    default_token = repo_dir / "token.json"
    token_path = _env_value(dotenv, "SCHWAB_TOKEN_PATH")
    if not token_path and default_token.exists():
        token_path = str(default_token)

    return SchwabCredentials(
        app_key=_env_value(dotenv, "SCHWAB_APP_KEY", "SCHWAB_CLIENT_ID"),
        app_secret=_env_value(dotenv, "SCHWAB_APP_SECRET", "SCHWAB_CLIENT_SECRET"),
        callback_url=_env_value(dotenv, "SCHWAB_CALLBACK_URL", "SCHWAB_REDIRECT_URI"),
        token_path=token_path,
    )


def _prepare_live_schwab_credentials() -> SchwabCredentials:
    if os.environ.get(RUN_FLAG) != "1":
        pytest.skip(f"set {RUN_FLAG}=1 to run live Yahoo/Schwab provider smoke test")

    dotenv = _quant_trade_env()
    creds = _schwab_credentials(dotenv)
    if not creds.is_valid() or not Path(creds.token_path).exists():
        pytest.skip("Schwab credentials/token not available from env or quant_trade")
    return creds


def _by_session_date(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.index = pd.to_datetime(normalized.index).tz_localize(None).normalize()
    normalized = normalized[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return normalized[~normalized.index.duplicated(keep="last")].sort_index()


def _comparison_dates(symbol: str, yahoo_by_date: pd.DataFrame, schwab_by_date: pd.DataFrame) -> pd.DatetimeIndex:
    common_dates = yahoo_by_date.index.intersection(schwab_by_date.index)
    assert len(common_dates) >= 20, f"{symbol}: not enough overlapping sessions"

    # Skip the newest shared session to avoid partial/current-day vendor timing differences.
    recent_dates = common_dates[-21:-1]
    assert len(recent_dates) >= 10, f"{symbol}: not enough stable recent sessions to compare"

    if symbol not in SPLIT_WINDOWS:
        return recent_dates

    start, end, _ = SPLIT_WINDOWS[symbol]
    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end)
    split_dates = common_dates[(common_dates >= start_date) & (common_dates <= end_date)]
    assert len(split_dates) >= 5, f"{symbol}: not enough pre-split adjusted sessions to compare"
    return split_dates.union(recent_dates)


def _has_more_than_two_decimal_precision(df: pd.DataFrame) -> bool:
    for col in PRICE_COLUMNS:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if ((values - values.round(2)).abs() > 1e-9).any():
            return True
    return False


def _assert_split_adjusted_scale(symbol: str, yahoo_by_date: pd.DataFrame, schwab_by_date: pd.DataFrame) -> None:
    start, end, max_split_adjusted_close = SPLIT_WINDOWS[symbol]
    for source_name, df in (("Yahoo", yahoo_by_date), ("Schwab", schwab_by_date)):
        window = df.loc[pd.Timestamp(start) : pd.Timestamp(end)]
        assert len(window) >= 5, f"{symbol} {source_name}: missing split-window data"
        assert float(window["Close"].max()) < max_split_adjusted_close, (
            f"{symbol} {source_name}: pre-split Close does not look split-adjusted"
        )


def _assert_source_precision_retained(symbol: str, yahoo_by_date: pd.DataFrame, schwab_by_date: pd.DataFrame) -> None:
    start, end, _ = SPLIT_WINDOWS[symbol]
    for source_name, df in (("Yahoo", yahoo_by_date), ("Schwab", schwab_by_date)):
        window = df.loc[pd.Timestamp(start) : pd.Timestamp(end)]
        assert _has_more_than_two_decimal_precision(window), (
            f"{symbol} {source_name}: split-window OHLC appears truncated to 2 decimals"
        )


def test_comparison_dates_include_split_window_and_recent_sessions():
    index = pd.bdate_range("2022-08-01", "2026-08-20")
    df = pd.DataFrame(
        {
            "Open": 100.001,
            "High": 101.001,
            "Low": 99.001,
            "Close": 100.501,
            "Volume": 1000,
        },
        index=index,
    )

    dates = _comparison_dates("TSLA", df, df)

    assert any(pd.Timestamp("2022-08-01") <= date <= pd.Timestamp("2022-08-24") for date in dates)
    assert dates.max() == pd.Timestamp("2026-08-19")


def test_precision_detector_requires_more_than_two_decimals():
    high_precision = pd.DataFrame({"Open": [10.123], "High": [11.0], "Low": [9.0], "Close": [10.5]})
    rounded = pd.DataFrame({"Open": [10.12], "High": [11.0], "Low": [9.0], "Close": [10.5]})

    assert _has_more_than_two_decimal_precision(high_precision)
    assert not _has_more_than_two_decimal_precision(rounded)


def _assert_price_sources_match(symbol: str, yahoo_df: pd.DataFrame, schwab_df: pd.DataFrame) -> None:
    yahoo_by_date = _by_session_date(yahoo_df)
    schwab_by_date = _by_session_date(schwab_df)
    sample_dates = _comparison_dates(symbol, yahoo_by_date, schwab_by_date)
    if symbol in SPLIT_WINDOWS:
        _assert_split_adjusted_scale(symbol, yahoo_by_date, schwab_by_date)
        _assert_source_precision_retained(symbol, yahoo_by_date, schwab_by_date)

    for date in sample_dates:
        for col in PRICE_COLUMNS:
            yahoo_value = float(yahoo_by_date.loc[date, col])
            schwab_value = float(schwab_by_date.loc[date, col])
            diff = abs(yahoo_value - schwab_value)
            allowed = max(0.10, abs(yahoo_value) * 0.003)
            assert diff <= allowed, (
                f"{symbol} {date.date()} {col}: Yahoo={yahoo_value}, "
                f"Schwab={schwab_value}, allowed_diff={allowed}"
            )

        assert yahoo_by_date.loc[date, "Volume"] > 0
        assert schwab_by_date.loc[date, "Volume"] > 0


def test_live_yahoo_and_schwab_ohlcv_price_contract_smoke():
    creds = _prepare_live_schwab_credentials()
    yahoo = YahooDataProvider(max_retries=0)
    schwab = SchwabDataProvider(creds=creds, max_retries=0, rate_limit_sleep=0)

    for symbol in SPLIT_ADJUSTED_SYMBOLS + NON_RECENT_SPLIT_SYMBOLS:
        yahoo_symbol, yahoo_df = yahoo.download_single_stock(symbol, period="5y", interval="1d")
        schwab_symbol, schwab_df = schwab.download_single_stock(symbol, period="5y", interval="1d")

        assert yahoo_symbol == symbol
        assert schwab_symbol == symbol
        assert yahoo_df is not None and not yahoo_df.empty
        assert schwab_df is not None and not schwab_df.empty
        assert list(yahoo_df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert list(schwab_df.columns) == ["Open", "High", "Low", "Close", "Volume"]

        _assert_price_sources_match(symbol, yahoo_df, schwab_df)
