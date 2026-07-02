from pathlib import Path

from dashboard.app import _csv_cache_fingerprint


def test_table_controls_do_not_use_unsupported_selectbox_horizontal_keyword():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")

    assert "horizontal=True" not in source


def test_custom_filter_ui_uses_trading_decision_funnel_without_presets():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")

    assert "_preset_selector" not in source
    assert "Preset" not in source
    assert '"1 Route"' in source
    assert '"2 Entry Confirmation & Strength"' in source
    assert '"3 Weekly Volume & Price"' in source
    assert '"4 Structure"' in source
    assert '"5 Grouping"' in source
    assert source.index('"1 Route"') < source.index('"2 Entry Confirmation & Strength"')


def test_c_rank_mode_displays_fixed_rules_and_formula_reference():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")

    assert "Fixed Mode Rules" in source
    assert "signal=True" in source
    assert "rank_C_continuous asc" in source
    assert "Custom filters ignored" in source
    assert "2.5 x pct(base_depth_abs)" in source


def test_csv_cache_fingerprint_changes_when_same_path_is_rewritten(tmp_path):
    csv_path = tmp_path / "breakout_follow_pool.csv"
    csv_path.write_text("code,signal,rank_C_continuous\nAAA,True,1\n", encoding="utf-8")
    first = _csv_cache_fingerprint(str(csv_path))

    csv_path.write_text("code,signal,rank_C_continuous\nBBBB,True,1\n", encoding="utf-8")
    second = _csv_cache_fingerprint(str(csv_path))

    assert second != first
