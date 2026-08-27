"""TradingView/Yahoo/Schwab price-contract audit tests.

This module is deliberately test-only.  It must not be imported by production
screeners or providers.

Contract under audit:
- prices are split-adjusted;
- prices are not dividend-adjusted;
- repository code does not round/truncate provider precision.

The live audit is opt-in because it depends on external vendor state and a
local Schwab OAuth token.  Run with RUN_LIVE_PROVIDER_SMOKE=1.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
import yfinance as yf
from tradingview_screener import Query, col

import weekly_vol_screener
from data_providers.schwab_provider import SchwabCredentials, SchwabDataProvider
from data_providers.yahoo_provider import YahooDataProvider


RUN_FLAG = "RUN_LIVE_PROVIDER_SMOKE"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUANT_TRADE_DIR = PROJECT_ROOT / ".." / "quant_trade"

SPLIT_SYMBOLS = ("NVDA", "TSLA")
DIVIDEND_SYMBOLS = ("KO", "XOM", "VZ", "MO")
AUDIT_SYMBOLS = SPLIT_SYMBOLS + DIVIDEND_SYMBOLS

TV_FIELDS = (
    "name",
    "exchange",
    "close",
    "open|1W",
    "high|1W",
    "low|1W",
    "volume|1W",
    "SMA10|1W",
    "SMA40|1W",
    "price_52_week_high",
    "High.All",
)


class _FakeTradingViewQuery:
    """Minimal fluent Query fake used to prove local code keeps source precision."""

    def __init__(self, frame: pd.DataFrame):
        self._frame = frame

    def select(self, *args, **kwargs):
        return self

    def where(self, *args, **kwargs):
        return self

    def order_by(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def set_markets(self, *args, **kwargs):
        return self

    def get_scanner_data(self):
        return len(self._frame), self._frame.copy()


def test_tradingview_screener_path_does_not_round_source_precision(monkeypatch, tmp_path):
    """Production screener path must persist TradingView floats without local rounding."""

    source = pd.DataFrame(
        [
            {
                "ticker": "NASDAQ:PREC",
                "name": "PREC",
                "relative_volume_10d_calc|1W": 1.543219876,
                "earnings_per_share_diluted_yoy_growth_fq": 12.3456789,
                "close": 123.456789,
                "open|1W": 120.123456,
                "high|1W": 124.987654,
                "low|1W": 119.234567,
                "volume|1W": 1234567,
                "market_cap_basic": 987654321.123456,
                "change|1W": 2.345678,
                "sector": "Technology Services",
                "industry": "Packaged Software",
            }
        ]
    )

    monkeypatch.setattr(
        weekly_vol_screener,
        "Query",
        lambda: _FakeTradingViewQuery(source),
    )
    monkeypatch.setattr(weekly_vol_screener, "load_whitelist", lambda: None)

    output = tmp_path / "weekly_vol.csv"
    _, frame = weekly_vol_screener.screen_weekly_vol_breakout(
        min_vol_ratio=1.3,
        min_price=15,
        limit=10,
        output_file=str(output),
        verbose=False,
    )

    expected = {
        "close": 123.456789,
        "open|1W": 120.123456,
        "high|1W": 124.987654,
        "low|1W": 119.234567,
    }
    for field, value in expected.items():
        assert float(frame.iloc[0][field]) == pytest.approx(value, rel=0, abs=1e-12)

    persisted = pd.read_csv(output)
    for field, value in expected.items():
        assert float(persisted.iloc[0][field]) == pytest.approx(value, rel=0, abs=1e-12)


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


def _prepare_live_schwab_credentials() -> SchwabCredentials:
    if os.environ.get(RUN_FLAG) != "1":
        pytest.skip(f"set {RUN_FLAG}=1 to run live TradingView/Yahoo/Schwab audit")

    dotenv = _quant_trade_env()
    repo_dir = Path(os.environ.get("QUANT_TRADE_REPO", DEFAULT_QUANT_TRADE_DIR))
    default_token = repo_dir / "token.json"
    token_path = _env_value(dotenv, "SCHWAB_TOKEN_PATH")
    if not token_path and default_token.exists():
        token_path = str(default_token)

    creds = SchwabCredentials(
        app_key=_env_value(dotenv, "SCHWAB_APP_KEY", "SCHWAB_CLIENT_ID"),
        app_secret=_env_value(dotenv, "SCHWAB_APP_SECRET", "SCHWAB_CLIENT_SECRET"),
        callback_url=_env_value(dotenv, "SCHWAB_CALLBACK_URL", "SCHWAB_REDIRECT_URI"),
        token_path=token_path,
    )
    if not creds.is_valid() or not Path(creds.token_path).exists():
        pytest.skip("Schwab credentials/token not available from env or quant_trade")
    return creds


def _fetch_tradingview_snapshot(symbols: tuple[str, ...]) -> dict[str, pd.Series]:
    _, frame = (
        Query()
        .select(*TV_FIELDS)
        .where(
            col("exchange").isin(["AMEX", "CBOE", "NASDAQ", "NYSE"]),
            col("active_symbol") == True,
            col("is_primary") == True,
            col("type").isin(["stock", "dr"]),
            col("name").isin(list(symbols)),
        )
        .limit(max(50, len(symbols) * 4))
        .set_markets("america")
        .get_scanner_data()
    )

    assert frame is not None and not frame.empty, "TradingView returned no audit rows"
    missing_columns = set(TV_FIELDS).difference(frame.columns)
    assert not missing_columns, f"TradingView schema missing fields: {sorted(missing_columns)}"

    rows: dict[str, pd.Series] = {}
    for symbol in symbols:
        matched = frame[frame["name"].astype(str).str.upper().eq(symbol)]
        assert len(matched) == 1, (
            f"{symbol}: expected one primary TradingView row, got {len(matched)}"
        )
        rows[symbol] = matched.iloc[0]
    return rows


def _download_provider_frame(provider, symbol: str) -> pd.DataFrame:
    returned_symbol, frame = provider.download_single_stock(
        symbol,
        period="5y",
        interval="1d",
    )
    assert returned_symbol == symbol
    assert frame is not None and not frame.empty, f"{symbol}: provider returned no data"
    assert list(frame.columns) == ["Open", "High", "Low", "Close", "Volume"]
    return _normalize_daily(frame)


def _normalize_daily(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    index = pd.to_datetime(normalized.index)
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    normalized.index = index.normalize()
    normalized = normalized[~normalized.index.duplicated(keep="last")].sort_index()
    return normalized


def _weekly_metrics(frame: pd.DataFrame) -> dict[str, float]:
    daily = _normalize_daily(frame)
    weekly = daily.resample("W-FRI").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
        }
    ).dropna()

    assert len(weekly) >= 52, "not enough weekly history for contract audit"
    return {
        "sma10": float(weekly["Close"].tail(10).mean()),
        "sma40": float(weekly["Close"].tail(40).mean()),
        "high52": float(weekly["High"].tail(52).max()),
    }


def _download_yahoo_raw_and_dividend_adjusted(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = yf.Ticker(symbol).history(
        period="5y",
        interval="1d",
        auto_adjust=False,
        timeout=10,
    )
    assert frame is not None and not frame.empty, f"{symbol}: Yahoo direct history empty"
    required = {"Open", "High", "Low", "Close", "Adj Close"}
    assert required.issubset(frame.columns), f"{symbol}: Yahoo missing {required.difference(frame.columns)}"

    raw = frame[["Open", "High", "Low", "Close"]].copy()
    close = pd.to_numeric(frame["Close"], errors="coerce")
    adj_close = pd.to_numeric(frame["Adj Close"], errors="coerce")
    factor = adj_close.div(close).replace([float("inf"), float("-inf")], pd.NA)

    adjusted = raw.mul(factor, axis=0)
    raw = raw.dropna()
    adjusted = adjusted.dropna()
    return raw, adjusted


def _relative_error(actual: float, expected: float) -> float:
    denominator = max(abs(expected), 1e-12)
    return abs(actual - expected) / denominator


def _assert_close_scale(symbol: str, tv_close: float, provider_close: float, source: str) -> None:
    # Current-day vendor timestamps need breathing room; this is a scale audit,
    # not an execution-price equality test.
    allowed = max(0.50, abs(tv_close) * 0.03)
    assert abs(tv_close - provider_close) <= allowed, (
        f"{symbol} current Close scale mismatch: TradingView={tv_close}, "
        f"{source}={provider_close}, allowed={allowed}"
    )


def test_live_tradingview_yahoo_schwab_price_contract_audit():
    """Opt-in live audit of split basis, dividend basis and three-source scale."""

    creds = _prepare_live_schwab_credentials()
    yahoo = YahooDataProvider(max_retries=0)
    schwab = SchwabDataProvider(creds=creds, max_retries=0, rate_limit_sleep=0)
    tv_rows = _fetch_tradingview_snapshot(AUDIT_SYMBOLS)

    yahoo_frames: dict[str, pd.DataFrame] = {}
    schwab_frames: dict[str, pd.DataFrame] = {}

    for symbol in AUDIT_SYMBOLS:
        yahoo_frames[symbol] = _download_provider_frame(yahoo, symbol)
        schwab_frames[symbol] = _download_provider_frame(schwab, symbol)

        tv_close = float(tv_rows[symbol]["close"])
        _assert_close_scale(
            symbol,
            tv_close,
            float(yahoo_frames[symbol]["Close"].iloc[-1]),
            "Yahoo",
        )
        _assert_close_scale(
            symbol,
            tv_close,
            float(schwab_frames[symbol]["Close"].iloc[-1]),
            "Schwab",
        )

    # Split contract: High.All gives us a historical-scale probe even though
    # tradingview_screener does not expose a historical candle series.
    for symbol in SPLIT_SYMBOLS:
        tv_all_time_high = float(tv_rows[symbol]["High.All"])
        yahoo_five_year_high = float(yahoo_frames[symbol]["High"].max())
        schwab_five_year_high = float(schwab_frames[symbol]["High"].max())

        yahoo_error = _relative_error(tv_all_time_high, yahoo_five_year_high)
        schwab_error = _relative_error(tv_all_time_high, schwab_five_year_high)
        print(
            f"[split] {symbol}: TV High.All={tv_all_time_high:.8f}, "
            f"Yahoo 5Y High={yahoo_five_year_high:.8f} ({yahoo_error:.4%}), "
            f"Schwab 5Y High={schwab_five_year_high:.8f} ({schwab_error:.4%})"
        )
        assert yahoo_error <= 0.03, f"{symbol}: TradingView split scale differs from Yahoo"
        assert schwab_error <= 0.03, f"{symbol}: TradingView split scale differs from Schwab"

    # Dividend contract: first ensure TV historical-derived metrics agree with
    # raw Yahoo/Schwab prices, then use Yahoo Adj Close as a negative control.
    sma40_evidence = 0
    high52_evidence = 0

    for symbol in DIVIDEND_SYMBOLS:
        tv_sma40 = float(tv_rows[symbol]["SMA40|1W"])
        tv_high52 = float(tv_rows[symbol]["price_52_week_high"])

        yahoo_metrics = _weekly_metrics(yahoo_frames[symbol])
        schwab_metrics = _weekly_metrics(schwab_frames[symbol])

        assert _relative_error(tv_sma40, yahoo_metrics["sma40"]) <= 0.02, (
            f"{symbol}: TV SMA40 is not consistent with raw Yahoo weekly closes"
        )
        assert _relative_error(tv_sma40, schwab_metrics["sma40"]) <= 0.02, (
            f"{symbol}: TV SMA40 is not consistent with raw Schwab weekly closes"
        )
        assert _relative_error(tv_high52, yahoo_metrics["high52"]) <= 0.02, (
            f"{symbol}: TV 52W high is not consistent with raw Yahoo highs"
        )
        assert _relative_error(tv_high52, schwab_metrics["high52"]) <= 0.02, (
            f"{symbol}: TV 52W high is not consistent with raw Schwab highs"
        )

        raw_direct, adjusted_direct = _download_yahoo_raw_and_dividend_adjusted(symbol)
        raw_metrics = _weekly_metrics(raw_direct)
        adjusted_metrics = _weekly_metrics(adjusted_direct)

        sma_separation = _relative_error(
            adjusted_metrics["sma40"],
            raw_metrics["sma40"],
        )
        tv_sma_raw_error = _relative_error(tv_sma40, raw_metrics["sma40"])
        tv_sma_adjusted_error = _relative_error(tv_sma40, adjusted_metrics["sma40"])

        if sma_separation >= 0.005:
            sma40_evidence += 1
            assert tv_sma_raw_error < tv_sma_adjusted_error, (
                f"{symbol}: TV SMA40 is closer to dividend-adjusted than raw Yahoo"
            )

        high_separation = _relative_error(
            adjusted_metrics["high52"],
            raw_metrics["high52"],
        )
        tv_high_raw_error = _relative_error(tv_high52, raw_metrics["high52"])
        tv_high_adjusted_error = _relative_error(tv_high52, adjusted_metrics["high52"])

        if high_separation >= 0.005:
            high52_evidence += 1
            assert tv_high_raw_error < tv_high_adjusted_error, (
                f"{symbol}: TV 52W high is closer to dividend-adjusted than raw Yahoo"
            )

        print(
            f"[dividend] {symbol}: "
            f"TV SMA40={tv_sma40:.8f}, raw={raw_metrics['sma40']:.8f}, "
            f"adj={adjusted_metrics['sma40']:.8f}, raw_err={tv_sma_raw_error:.4%}, "
            f"adj_err={tv_sma_adjusted_error:.4%}; "
            f"TV 52W={tv_high52:.8f}, raw={raw_metrics['high52']:.8f}, "
            f"adj={adjusted_metrics['high52']:.8f}, raw_err={tv_high_raw_error:.4%}, "
            f"adj_err={tv_high_adjusted_error:.4%}"
        )

    # The negative control must materially differ on enough high-dividend names
    # for the audit to prove anything about dividend adjustment.
    assert sma40_evidence >= 3, (
        f"insufficient dividend-adjustment separation in SMA40 controls: {sma40_evidence}"
    )
    assert high52_evidence >= 2, (
        f"insufficient dividend-adjustment separation in 52W-high controls: {high52_evidence}"
    )
