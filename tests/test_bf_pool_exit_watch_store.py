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
