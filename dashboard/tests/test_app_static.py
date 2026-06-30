from pathlib import Path

from dashboard.app import _csv_cache_fingerprint


def test_table_controls_do_not_use_unsupported_selectbox_horizontal_keyword():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")

    assert "horizontal=True" not in source


def test_csv_cache_fingerprint_changes_when_same_path_is_rewritten(tmp_path):
    csv_path = tmp_path / "breakout_follow_pool.csv"
    csv_path.write_text("code,signal,rank_C_continuous\nAAA,True,1\n", encoding="utf-8")
    first = _csv_cache_fingerprint(str(csv_path))

    csv_path.write_text("code,signal,rank_C_continuous\nBBBB,True,1\n", encoding="utf-8")
    second = _csv_cache_fingerprint(str(csv_path))

    assert second != first
