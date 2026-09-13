from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

import DataStore
import data_providers.yahoo_provider as yahoo_module
from data_providers.ohlcv_validation import (
    DataIntegrityError,
    expected_latest_us_session,
    validate_download_batch,
    validate_ohlcv_frame,
)
from data_providers.yahoo_provider import YahooDataProvider


def _frame(dates, *, nan_close=False):
    index = pd.to_datetime(dates)
    size = len(index)
    data = pd.DataFrame(
        {
            "Open": [10.0] * size,
            "High": [11.0] * size,
            "Low": [9.0] * size,
            "Close": [10.5] * size,
            "Volume": [1000] * size,
        },
        index=index,
    )
    if nan_close:
        data.iloc[-1, data.columns.get_loc("Close")] = np.nan
    return data


def test_expected_latest_us_session_handles_weekend_and_exchange_holiday():
    eastern = ZoneInfo("America/New_York")

    assert expected_latest_us_session(
        datetime(2026, 9, 12, 12, tzinfo=eastern)
    ).isoformat() == "2026-09-11"

    # Labor Day was Monday 2026-09-07; the latest completed regular
    # session remained Friday 2026-09-04.
    assert expected_latest_us_session(
        datetime(2026, 9, 7, 20, tzinfo=eastern)
    ).isoformat() == "2026-09-04"

    # NYSE does not move a Saturday New Year's Day closure back to Friday.
    assert expected_latest_us_session(
        datetime(2021, 12, 31, 20, tzinfo=eastern)
    ).isoformat() == "2021-12-31"


def test_validate_ohlcv_frame_rejects_any_null_required_value():
    data = _frame(["2026-09-10", "2026-09-11"], nan_close=True)

    with pytest.raises(DataIntegrityError, match="null/non-numeric OHLCV"):
        validate_ohlcv_frame("AAPL", data)


def test_validate_download_batch_rejects_missing_expected_symbol():
    data = {"AAPL": _frame(["2026-09-11"])}

    with pytest.raises(DataIntegrityError, match="missing 1 expected symbols"):
        validate_download_batch(
            data,
            expected_symbols=["AAPL", "MSFT"],
            interval="1d",
        )


def test_validate_download_batch_rejects_missing_latest_session_bar():
    current = _frame(["2026-09-10", "2026-09-11"])
    stale = _frame(["2026-09-10"])
    data = {
        "^GSPC": current.copy(),
        "^IXIC": current.copy(),
        "^DJI": current.copy(),
        "AAPL": stale,
    }

    with pytest.raises(DataIntegrityError, match="do not reach latest market bar"):
        validate_download_batch(
            data,
            expected_symbols=list(data),
            interval="1d",
            now=datetime(
                2026,
                9,
                12,
                12,
                tzinfo=ZoneInfo("America/New_York"),
            ),
        )


def test_validate_download_batch_rejects_globally_stale_consensus():
    stale = _frame(["2026-09-10"])
    data = {
        symbol: stale.copy()
        for symbol in ["^GSPC", "^IXIC", "^DJI", "AAPL"]
    }

    with pytest.raises(
        DataIntegrityError,
        match="latest completed US session 2026-09-11",
    ):
        validate_download_batch(
            data,
            expected_symbols=list(data),
            interval="1d",
            now=datetime(
                2026,
                9,
                12,
                12,
                tzinfo=ZoneInfo("America/New_York"),
            ),
        )


def test_yahoo_single_download_retries_invalid_partial_row(monkeypatch):
    bad = _frame(["2026-09-10", "2026-09-11"], nan_close=True)
    good = _frame(["2026-09-10", "2026-09-11"])

    class FakeTicker:
        calls = 0

        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            FakeTicker.calls += 1
            return bad if FakeTicker.calls == 1 else good

    monkeypatch.setattr(yahoo_module.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(yahoo_module.time, "sleep", lambda _: None)

    provider = YahooDataProvider(max_retries=1)
    symbol, data = provider.download_single_stock(
        "AAPL", period="2y", interval="1d"
    )

    assert symbol == "AAPL"
    assert data is not None
    assert FakeTicker.calls == 2


def test_yahoo_batch_retries_one_stale_latest_bar(monkeypatch):
    current = _frame(["2026-09-10", "2026-09-11"])
    stale = _frame(["2026-09-10"])
    calls = {}
    provider = YahooDataProvider(
        batch_size=4,
        max_workers=1,
        max_retries=0,
    )
    monkeypatch.setattr(
        yahoo_module,
        "expected_latest_us_session",
        lambda: pd.Timestamp("2026-09-11").date(),
    )

    def fake_download(symbol, period="1y", interval="1d"):
        calls[symbol] = calls.get(symbol, 0) + 1
        if symbol == "AAPL" and calls[symbol] == 1:
            return symbol, stale.copy()
        return symbol, current.copy()

    monkeypatch.setattr(provider, "download_single_stock", fake_download)

    data, failed = provider.download_batch_stocks(
        ["^GSPC", "^IXIC", "^DJI", "AAPL"],
        interval="1d",
    )

    assert failed == []
    assert calls["AAPL"] == 2
    assert data["AAPL"].index[-1].date().isoformat() == "2026-09-11"


def test_yahoo_batch_rejects_persistently_stale_latest_bar(monkeypatch):
    current = _frame(["2026-09-10", "2026-09-11"])
    stale = _frame(["2026-09-10"])
    provider = YahooDataProvider(
        batch_size=4,
        max_workers=1,
        max_retries=0,
    )
    monkeypatch.setattr(
        yahoo_module,
        "expected_latest_us_session",
        lambda: pd.Timestamp("2026-09-11").date(),
    )

    def fake_download(symbol, period="1y", interval="1d"):
        return (
            symbol,
            stale.copy() if symbol == "AAPL" else current.copy(),
        )

    monkeypatch.setattr(provider, "download_single_stock", fake_download)

    data, failed = provider.download_batch_stocks(
        ["^GSPC", "^IXIC", "^DJI", "AAPL"],
        interval="1d",
    )

    assert failed == ["AAPL"]
    assert "AAPL" not in data


def test_save_stock_data_rejects_invalid_frame_before_writing(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "invalid.pkl"
    monkeypatch.setattr(
        DataStore,
        "get_stock_pkl_path",
        lambda interval="1d": str(output),
    )

    with pytest.raises(DataIntegrityError, match="invalid OHLCV data"):
        DataStore.save_stock_data(
            {"AAPL": _frame(["2026-09-11"], nan_close=True)},
            save_dir=str(tmp_path),
            interval="1d",
        )

    assert not output.exists()
    assert not (tmp_path / "invalid.pkl.tmp").exists()


def test_save_stock_data_preserves_previous_file_if_temp_round_trip_fails(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "stock_data_existing_1d.pkl"
    previous_bytes = b"previous-good-pkl"
    output.write_bytes(previous_bytes)

    monkeypatch.setattr(
        DataStore,
        "get_stock_pkl_path",
        lambda interval="1d": str(output),
    )
    monkeypatch.setattr(DataStore, "load_stock_data", lambda _: {})

    result = DataStore.save_stock_data(
        {"AAPL": _frame(["2026-09-11"])},
        save_dir=str(tmp_path),
        interval="1d",
    )

    assert result is None
    assert output.read_bytes() == previous_bytes
    assert not (tmp_path / "stock_data_existing_1d.pkl.tmp").exists()
