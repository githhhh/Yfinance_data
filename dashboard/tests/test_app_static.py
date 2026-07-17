from pathlib import Path
import ast
from dashboard.app import _csv_cache_fingerprint


def test_table_controls_do_not_use_unsupported_selectbox_horizontal_keyword():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "selectbox":
                for kw in node.keywords:
                    assert kw.arg != "horizontal"


def test_custom_filter_ui_removes_sort_bar_from_custom_mode():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert 'st.subheader("Sort")' not in source
    assert "_sort_specs" not in source
    assert "_render_sort_summary" not in source
    assert "sort_1" not in source
    assert "direction_1" not in source


def test_c_rank_mode_displays_fixed_rules_and_formula_reference():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert "Fixed Mode Rules" in source
    assert "Exclusively evaluates Active Signals (`signal=True`)" in source
    assert "Sorted by `rank_C_continuous` asc" in source
    assert "Top N slice selector only" in source
    assert "2.5 x pct(base_depth_abs)" in source


def test_c_rank_reference_denominator_uses_active_signals_count():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert 'active_signals_count = int((df["signal"] == True).sum()) if "signal" in df.columns else len(df)' in source
    assert 'Showing: {len(ranked)} of {denom} Active Signals · Reference Only' in source


def test_csv_cache_fingerprint_changes_when_same_path_is_rewritten(tmp_path):
    csv_path = tmp_path / "breakout_follow_pool.csv"
    csv_path.write_text("code,signal,rank_C_continuous\nAAA,True,1\n", encoding="utf-8")
    first = _csv_cache_fingerprint(str(csv_path))

    csv_path.write_text("code,signal,rank_C_continuous\nBBBB,True,1\n", encoding="utf-8")
    second = _csv_cache_fingerprint(str(csv_path))

    assert second != first


def test_app_no_obsolete_funnel_or_css_tooltip_code():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert "def _funnel_filters(" not in source
    assert "def _flatten_filters(" not in source
    assert "def _funnel_tab_labels(" not in source
    assert "div[col-id=" not in source
    assert 'div[class*="st-key-btn_status_ACTIONABLE"]:hover::after' not in source
