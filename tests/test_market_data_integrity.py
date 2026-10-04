import numpy as np
import pandas as pd
import pytest

import DataStore
import data_providers.yahoo_provider as yahoo_module
from data_providers.ohlcv_validation import (
    DataIntegrityError,
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
        )


def test_schwab_filters_invalid_ohlc_but_yahoo_validation_remains_strict(capsys):
    invalid = _frame(["2026-09-11"])
    invalid.loc[:, "Close"] = 11.5  # Above High; retain the source value.
    data = {symbol: _frame(["2026-09-11"]) for symbol in ("^GSPC", "^IXIC", "^DJI", "MSFT")}
    data["AAPL"] = invalid

    filtered, excluded = DataStore.filter_schwab_stock_data(
        data, failed=[], expected_symbols=list(data), interval="1wk"
    )
    assert set(filtered) == set(data) - {"AAPL"}
    assert excluded == ["AAPL"]
    output = capsys.readouterr().out
    assert "AAPL: inconsistent OHLC rows" in output
    assert "High=11.0" in output
    assert "Close=11.5" in output
    assert "High<Close" in output

    with pytest.raises(DataIntegrityError, match="inconsistent OHLC rows"):
        validate_download_batch(
            data, expected_symbols=list(data)
        )


def test_schwab_filters_reported_failure_but_keeps_valid_older_history(capsys):
    current = _frame(["2026-09-11"])
    data = {symbol: current.copy() for symbol in ("^GSPC", "^IXIC", "^DJI", "MSFT")}
    data["AAPL"] = _frame(["2026-09-04"])

    filtered, excluded = DataStore.filter_schwab_stock_data(
        data, failed=["IVZ"], expected_symbols=[*data, "IVZ"], interval="1wk"
    )
    assert set(filtered) == set(data)
    assert excluded == ["IVZ"]
    output = capsys.readouterr().out
    assert "IVZ" in output and "filtered" in output
    assert "AAPL: latest bar" not in output


def test_schwab_does_not_hide_unreported_missing_or_missing_market_reference():
    data = {symbol: _frame(["2026-09-11"]) for symbol in ("^GSPC", "^IXIC", "^DJI")}
    with pytest.raises(DataIntegrityError, match="unreported missing"):
        DataStore.filter_schwab_stock_data(
            data, failed=[], expected_symbols=[*data, "IVZ"], interval="1wk"
        )
    with pytest.raises(DataIntegrityError, match="market reference"):
        DataStore.filter_schwab_stock_data(
            {"^GSPC": data["^GSPC"], "^IXIC": data["^IXIC"]},
            failed=["^DJI"], expected_symbols=list(data), interval="1wk"
        )


def test_schwab_keeps_valid_histories_with_different_reference_dates():
    data = {symbol: _frame(["2026-09-11"]) for symbol in ("^GSPC", "^IXIC", "MSFT")}
    data["^DJI"] = _frame(["2026-09-10"])

    filtered, excluded = DataStore.filter_schwab_stock_data(
        data, failed=[], expected_symbols=list(data), interval="1wk"
    )
    assert set(filtered) == set(data)
    assert excluded == []


def test_validate_download_batch_accepts_valid_older_symbol_history():
    current = _frame(["2026-09-10", "2026-09-11"])
    stale = _frame(["2026-09-10"])
    data = {
        "^GSPC": current.copy(),
        "^IXIC": current.copy(),
        "^DJI": current.copy(),
        "AAPL": stale,
    }

    validate_download_batch(data, expected_symbols=list(data))


@pytest.mark.parametrize(
    ("column", "value", "reason"),
    [
        ("Close", np.nan, "null/non-numeric OHLCV"),
        ("Close", np.inf, "non-finite OHLCV"),
        ("Close", 0.0, "non-positive price"),
        ("Volume", -1, "negative volume"),
    ],
)
def test_schwab_still_rejects_other_invalid_rows(column, value, reason):
    invalid = _frame(["2026-09-11"])
    invalid.loc[:, column] = value
    with pytest.raises(DataIntegrityError, match=reason):
        validate_ohlcv_frame(
            "AAPL", invalid,
        )


def test_yahoo_single_download_fails_fast_on_invalid_partial_row(monkeypatch):
    bad = _frame(["2026-09-10", "2026-09-11"], nan_close=True)

    class FakeTicker:
        calls = 0

        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            FakeTicker.calls += 1
            return bad

    monkeypatch.setattr(yahoo_module.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(yahoo_module.time, "sleep", lambda _: None)

    provider = YahooDataProvider(max_retries=1)
    with pytest.raises(DataIntegrityError, match="null/non-numeric OHLCV"):
        provider.download_single_stock("AAPL", period="2y", interval="1d")

    assert FakeTicker.calls == 1


@pytest.mark.parametrize("interval", ["1d", "1wk"])
def test_yahoo_batch_keeps_valid_older_history_without_date_retry(monkeypatch, interval):
    previous = "2026-09-10" if interval == "1d" else "2026-09-04"
    current = _frame([previous, "2026-09-11"])
    stale = _frame([previous])
    calls = {}
    provider = YahooDataProvider(
        batch_size=4,
        max_workers=1,
        max_retries=0,
    )

    def fake_download(symbol, period="1y", interval="1d", *, abort_event=None):
        calls[symbol] = calls.get(symbol, 0) + 1
        if symbol == "AAPL" and calls[symbol] == 1:
            return symbol, stale.copy()
        return symbol, current.copy()

    monkeypatch.setattr(provider, "download_single_stock", fake_download)

    data, failed = provider.download_batch_stocks(
        ["^GSPC", "^IXIC", "^DJI", "AAPL"],
        interval=interval,
    )

    assert failed == []
    assert calls == {symbol: 1 for symbol in data}
    assert data["AAPL"].index[-1].date().isoformat() == previous


@pytest.mark.parametrize("interval", ["1d", "1wk"])
def test_yahoo_batch_keeps_valid_older_history_through_pkl_publication(tmp_path, monkeypatch, interval):
    previous = "2026-09-10" if interval == "1d" else "2026-09-04"
    current = _frame([previous, "2026-09-11"])
    stale = _frame([previous])
    provider = YahooDataProvider(
        batch_size=4,
        max_workers=1,
        max_retries=0,
    )

    def fake_download(symbol, period="1y", interval="1d", *, abort_event=None):
        return (
            symbol,
            stale.copy() if symbol == "AAPL" else current.copy(),
        )

    monkeypatch.setattr(provider, "download_single_stock", fake_download)

    data, failed = provider.download_batch_stocks(
        ["^GSPC", "^IXIC", "^DJI", "AAPL"],
        interval=interval,
    )

    assert failed == []
    assert "AAPL" in data
    output = tmp_path / f"stock_data_120926_{interval}.pkl"
    monkeypatch.setattr(DataStore, "get_stock_pkl_path", lambda interval: str(output))
    saved = DataStore.save_stock_data(
        data, save_dir=str(tmp_path), interval=interval, expected_symbols=list(data)
    )
    assert saved == str(output)
    loaded = DataStore.load_stock_data(saved)
    assert set(loaded) == set(data)
    assert loaded["AAPL"].index[-1].date().isoformat() == previous


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


def test_schwab_filtered_save_round_trips_only_valid_symbols(tmp_path, monkeypatch):
    output = tmp_path / "schwab_1d.pkl"
    monkeypatch.setattr(
        DataStore,
        "get_stock_pkl_path",
        lambda interval="1d": str(output),
    )
    invalid = _frame(["2026-09-11"])
    invalid.loc[:, "Close"] = 11.5

    batch = {symbol: _frame(["2026-09-11"]) for symbol in ("^GSPC", "^IXIC", "^DJI", "MSFT")}
    batch["AAPL"] = invalid
    filtered, excluded = DataStore.filter_schwab_stock_data(
        batch, failed=[], expected_symbols=list(batch), interval="1wk"
    )
    assert excluded == ["AAPL"]
    saved = DataStore.save_stock_data(
        filtered,
        save_dir=str(tmp_path),
        interval="1wk",
        expected_symbols=list(filtered),
    )

    assert saved == str(output)
    loaded = DataStore.load_stock_data(saved)
    assert set(loaded) == set(filtered)
    assert "AAPL" not in loaded


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
