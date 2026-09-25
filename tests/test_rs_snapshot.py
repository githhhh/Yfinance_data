from datetime import date
import pickle

import pandas as pd
import pytest

import rs_snapshot


ANCHORS = [215.26, 123.98, 99.07, 90.41, 79.74, 52.93, 18.46]


def series(values):
    return pd.Series(values, index=pd.bdate_range("2025-01-02", periods=len(values)))


def test_parse_rsrating_csv():
    lines = []
    d = pd.Timestamp("2026-08-20")
    for i, value in enumerate(ANCHORS):
        for j in range(5):
            day = d + pd.Timedelta(days=i * 5 + j)
            lines.append(f"{day.strftime('%Y%m%d')}T,0,1000,0,{value},0")
    anchors, source_date = rs_snapshot.parse_rsrating_csv("\n".join(lines))
    assert anchors == ANCHORS
    assert source_date == date(2026, 9, 23)


def test_raw_rs_matches_pine_formula():
    benchmark = series([100 + i * 0.1 for i in range(300)])
    stock = series([50 + i * 0.2 for i in range(300)])
    score = rs_snapshot.raw_rs(stock, benchmark)
    sp = stock.to_numpy()
    bp = benchmark.to_numpy()
    stock_perf = [sp[-1] / sp[-1 - n] for n in rs_snapshot.LOOKBACKS]
    bench_perf = [bp[-1] / bp[-1 - n] for n in rs_snapshot.LOOKBACKS]
    expected = (
        sum(w * p for w, p in zip(rs_snapshot.WEIGHTS, stock_perf))
        / sum(w * p for w, p in zip(rs_snapshot.WEIGHTS, bench_perf))
        * 100
    )
    assert score == pytest.approx(expected)


def test_raw_rs_uses_short_history_bar_index():
    benchmark = series([100 + i for i in range(300)])
    stock = pd.Series([10, 11, 12, 13, 14], index=benchmark.index[-5:])
    assert rs_snapshot.raw_rs(stock, benchmark) == pytest.approx((14 / 10) / (399 / 395) * 100)


def test_raw_rs_aligns_benchmark_to_stock_bars_with_missing_session():
    benchmark = series([100, 110, 120, 130, 140, 150])
    stock = pd.Series([10, 11, 13, 14, 15], index=benchmark.index[[0, 1, 3, 4, 5]])
    assert rs_snapshot.raw_rs(stock, benchmark) == pytest.approx(100)


def test_anchor_age_counts_us_trading_sessions():
    sessions = [
        date(2026, 9, 18),
        date(2026, 9, 21),
        date(2026, 9, 22),
        date(2026, 9, 23),
        date(2026, 9, 24),
    ]
    assert rs_snapshot.anchor_age(date(2026, 9, 18), sessions) == 4
    assert rs_snapshot.anchor_age(date(2026, 9, 23), sessions) == 1


def test_calendar_anchor_normalizes_to_reference_session():
    sessions = [date(2026, 9, 4), date(2026, 9, 8)]
    assert rs_snapshot.normalize_anchor_date(date(2026, 9, 7), sessions) == date(2026, 9, 4)


def test_compute_rows_rejects_age_five():
    idx = pd.bdate_range("2026-09-14", periods=6)
    data = {
        "^GSPC": pd.DataFrame({"Close": range(100, 106)}, index=idx),
        "FORM": pd.DataFrame({"Close": range(50, 56)}, index=idx),
    }
    with pytest.raises(rs_snapshot.StaleAnchorError):
        rs_snapshot.compute_rows(data, ANCHORS, idx[0].date())


def test_write_snapshot_replaces_old_only_after_new_exists(tmp_path):
    old = tmp_path / "rs_data_230926.csv"
    old.write_text("Ticker,RS,AnchorDate\nOLD,1,2026-09-22\n", encoding="utf-8")
    target = rs_snapshot.write_snapshot(
        [{"Ticker": "FORM", "RS": 88, "AnchorDate": "2026-09-23"}],
        date(2026, 9, 24),
        tmp_path,
    )
    assert target.name == "rs_data_240926.csv"
    assert target.exists()
    assert not old.exists()
    assert target.read_text(encoding="utf-8").splitlines()[0] == "Ticker,RS,AnchorDate"


def test_remote_failure_uses_cache_and_marks_check_day(monkeypatch, tmp_path):
    monkeypatch.setattr(rs_snapshot, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(rs_snapshot, "CACHE_PATH", tmp_path / "cache.json")
    rs_snapshot.write_cache(
        {
            "last_check_date": "2026-09-23",
            "source_date": "2026-09-23",
            "anchors": ANCHORS,
        }
    )

    def fail(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(rs_snapshot.urllib.request, "urlopen", fail)
    anchors, source_date, source = rs_snapshot.get_rsrating_anchors(date(2026, 9, 24))
    assert anchors == ANCHORS
    assert source_date == date(2026, 9, 23)
    assert source == "cache-fallback"
    assert rs_snapshot.read_cache()["last_check_date"] == "2026-09-24"


def test_same_day_cache_skips_remote(monkeypatch, tmp_path):
    monkeypatch.setattr(rs_snapshot, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(rs_snapshot, "CACHE_PATH", tmp_path / "cache.json")
    rs_snapshot.write_cache(
        {
            "last_check_date": "2026-09-24",
            "source_date": "2026-09-23",
            "anchors": ANCHORS,
        }
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("remote should not be called")

    monkeypatch.setattr(rs_snapshot.urllib.request, "urlopen", unexpected)
    _, _, source = rs_snapshot.get_rsrating_anchors(date(2026, 9, 24))
    assert source == "cache"


def test_cleanup_stale_published_snapshots_when_refresh_is_unavailable(tmp_path):
    sessions = [
        date(2026, 9, 18),
        date(2026, 9, 21),
        date(2026, 9, 22),
        date(2026, 9, 23),
        date(2026, 9, 24),
        date(2026, 9, 25),
    ]
    stale = tmp_path / "rs_data_180926.csv"
    fresh = tmp_path / "rs_data_240926.csv"
    stale.write_text(
        "Ticker,RS,AnchorDate\nFORM,88,2026-09-18\n", encoding="utf-8"
    )
    fresh.write_text(
        "Ticker,RS,AnchorDate\nNET,90,2026-09-24\n", encoding="utf-8"
    )

    removed = rs_snapshot.cleanup_stale_published_snapshots(sessions, tmp_path)

    assert removed == [stale]
    assert not stale.exists()
    assert fresh.exists()


def test_rs_rating_matches_pine_band_edges():
    assert rs_snapshot.rs_rating(ANCHORS[0], ANCHORS) == 99
    assert rs_snapshot.rs_rating(ANCHORS[-1], ANCHORS) == 1
    assert 90 <= rs_snapshot.rs_rating(ANCHORS[1], ANCHORS) <= 98


def test_snapshot_output_has_only_required_columns(tmp_path):
    target = rs_snapshot.write_snapshot(
        [
            {"Ticker": "P", "RS": 77, "AnchorDate": "2026-09-23"},
            {"Ticker": "A", "RS": 99, "AnchorDate": "2026-09-23"},
            {"Ticker": "B", "RS": 77, "AnchorDate": "2026-09-23"},
        ],
        date(2026, 9, 24),
        tmp_path,
    )
    frame = pd.read_csv(target)
    assert frame.columns.tolist() == ["Ticker", "RS", "AnchorDate"]
    assert frame["Ticker"].tolist() == ["A", "P", "B"]
    assert b"\r\n" not in target.read_bytes()


def test_manual_pkl_entry_publishes_from_selected_existing_daily_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    results_dir = tmp_path / "results_pkl"
    results_dir.mkdir()
    idx = pd.bdate_range(end="2026-09-23", periods=6)
    data = {
        "^GSPC": pd.DataFrame({"Close": range(100, 106)}, index=idx),
        "FORM": pd.DataFrame({"Close": range(50, 56)}, index=idx),
    }
    pkl_path = results_dir / "stock_data_240926_1d.pkl"
    with pkl_path.open("wb") as fh:
        pickle.dump(data, fh)
    monkeypatch.setattr(
        rs_snapshot,
        "get_rsrating_anchors",
        lambda: (ANCHORS, date(2026, 9, 23), "remote"),
    )

    target = rs_snapshot.run(pkl_path=pkl_path)

    assert target.parent == rs_snapshot.RESULTS_DIR
    frame = pd.read_csv(target)
    assert frame["Ticker"].tolist() == ["FORM"]
    assert frame["AnchorDate"].tolist() == ["2026-09-23"]


def test_parse_rsrating_rejects_incomplete_transport_window():
    lines = []
    d = pd.Timestamp("2026-08-20")
    for i, value in enumerate(ANCHORS):
        for j in range(5):
            day = d + pd.Timedelta(days=i * 5 + j)
            lines.append(f"{day.strftime('%Y%m%d')}T,0,1000,0,{value},0")

    with pytest.raises(rs_snapshot.RSSnapshotError, match="Expected 35 RSRATING rows"):
        rs_snapshot.parse_rsrating_csv("\n".join(lines[:-1]))


def test_diagnostic_stale_anchor_does_not_delete_published_snapshot(monkeypatch, tmp_path):
    idx = pd.bdate_range("2026-09-18", periods=6)
    data = {
        "^GSPC": pd.DataFrame({"Close": range(100, 106)}, index=idx),
        "FORM": pd.DataFrame({"Close": range(50, 56)}, index=idx),
    }
    published = tmp_path / "rs_data_230926.csv"
    published.write_text(
        "Ticker,RS,AnchorDate\nFORM,88,2026-09-18\n", encoding="utf-8"
    )

    monkeypatch.setattr(rs_snapshot, "RESULTS_DIR", tmp_path)
    fake_pkl = tmp_path / "stock_data_240926_1d.pkl"
    fake_pkl.touch()
    monkeypatch.setattr(rs_snapshot, "find_latest_daily_pkl", lambda: fake_pkl)
    monkeypatch.setattr(rs_snapshot, "load_daily_pkl", lambda _: data)
    monkeypatch.setattr(
        rs_snapshot,
        "get_rsrating_anchors",
        lambda: (ANCHORS, date(2026, 9, 18), "cache"),
    )

    with pytest.raises(rs_snapshot.StaleAnchorError):
        rs_snapshot.run(["FORM"])

    assert published.exists()
