from pool_exit_watch_store import active_watch_codes, watch_frame, parse_watch_csv
import pytest

def test_watch_sources_only_active_complete(tmp_path):
    path=tmp_path/"us"/"bf_pool_exit_watch.csv"
    path.parent.mkdir(parents=True)
    rows=[{"code":"A","phase":"OBSERVING"},{"code":"B","phase":"AWAITING_DATA"},{"code":"C","phase":"RETURNED"}]
    watch_frame(rows,snapshot_date="2026-10-30",pool_codes_hash="sha256:x").to_csv(path,index=False)
    assert active_watch_codes(tmp_path)==["A","B"]
    assert len(parse_watch_csv(path))==3
    watch_frame([],snapshot_date="2026-10-30",pool_codes_hash="sha256:x").to_csv(path,index=False)
    assert active_watch_codes(tmp_path)==[]

def test_invalid_watch_fails_closed(tmp_path):
    path=tmp_path/"us"/"bf_pool_exit_watch.csv"
    path.parent.mkdir()
    path.write_text("code,phase\nA,OBSERVING\n")
    with pytest.raises(ValueError):
        active_watch_codes(tmp_path)

def test_midweek_never_extends_universe(tmp_path):
    from market_universe import build_download_universe
    path=tmp_path/"us"/"bf_pool_exit_watch_midweek.csv"
    path.parent.mkdir()
    watch_frame([{"code":"MID","phase":"PROVISIONAL_OUT"}],
                snapshot_date="2026-11-03",pool_codes_hash="x").to_csv(path,index=False)
    assert "MID" not in build_download_universe(data_root=tmp_path)



def test_download_requires_explicit_bootstrap_even_when_no_watch_ever_published(tmp_path):
    from pool_exit_watch_store import active_watch_codes
    with pytest.raises(RuntimeError, match="remotely bootstrapped"):
        active_watch_codes(tmp_path, required=True)
    target = tmp_path / "us" / "bf_pool_exit_watch.csv"
    target.parent.mkdir(parents=True)
    watch_frame([], snapshot_date="2026-10-30", pool_codes_hash="fixture").to_csv(
        target, index=False)
    assert active_watch_codes(tmp_path, required=True) == []



@pytest.mark.parametrize("bad_code", [" A", "A ", " ", "nan", "NaN", None])
def test_watch_csv_rejects_noncanonical_or_missing_codes(tmp_path, bad_code):
    rows = [{"code": bad_code, "phase": "OBSERVING", "events": []}]
    target = tmp_path / "watch.csv"
    watch_frame(rows, snapshot_date="2026-10-30",
                pool_codes_hash="digest").to_csv(target, index=False)
    with pytest.raises(ValueError, match="code"):
        parse_watch_csv(target)


@pytest.mark.parametrize("bad_code", [" ABC", "ABC ", "", "nan", None])
def test_pool_identity_rejects_noncanonical_or_missing_codes(bad_code):
    import pandas as pd
    from pool_exit_watch_store import pool_identity
    with pytest.raises(ValueError, match="code"):
        pool_identity(pd.DataFrame([{
            "code": bad_code, "snapshot_date": "2026-10-30"}]))


@pytest.mark.parametrize(("event_type", "week"), [
    ("POOL_EXIT", "2026-10-27"),  # Tuesday, not canonical Monday
    ("POOL_EXIT", "2026-13-01"),  # impossible calendar day
    ("FOREIGN_EVENT", "2026-10-26"),  # unknown event kind
])
def test_watch_csv_rejects_forged_event_week_or_kind(tmp_path, event_type, week):
    import json
    row = {"code": "ABC", "phase": "OBSERVING",
           "events": [{"id": f"{event_type}:{week}",
                       "type": event_type, "week": week}]}
    target = tmp_path / "watch.csv"
    watch_frame([row], snapshot_date="2026-10-30",
                pool_codes_hash="digest").to_csv(target, index=False)
    with pytest.raises(ValueError, match="event"):
        parse_watch_csv(target)



def test_watch_download_restoration_reuses_market_universe_dot_normalization(tmp_path):
    from market_universe import build_download_universe
    target = tmp_path / "us" / "bf_pool_exit_watch.csv"
    target.parent.mkdir(parents=True)
    watch_frame([{"code": "BRK.B", "phase": "OBSERVING", "events": []},
                 {"code": "BRK-B", "phase": "AWAITING_DATA", "events": []}],
                snapshot_date="2026-10-30",
                pool_codes_hash="digest").to_csv(target, index=False)
    assert active_watch_codes(tmp_path) == ["BRK-B"]
    universe = build_download_universe(data_root=tmp_path)
    assert "BRK-B" in universe and "BRK.B" not in universe
