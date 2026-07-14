from pathlib import Path

from dashboard.app import _csv_cache_fingerprint, _funnel_tab_labels
from dashboard.data_utils import FilterSpec


def test_table_controls_do_not_use_unsupported_selectbox_horizontal_keyword():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    import ast

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "selectbox":
                for kw in node.keywords:
                    assert kw.arg != "horizontal"


def test_custom_filter_ui_uses_trading_decision_funnel_without_presets():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")

    assert "_preset_selector" not in source
    assert "Preset" not in source
    assert 'st.title("Breakout Pool")' not in source
    assert "Breakout Pool Workbench" not in source
    assert '"Route"' in source
    assert '"Entry Status"' in source
    assert '"Optional Quality Filters"' in source
    assert source.index('"Route"') < source.index('"Entry Status"')


def test_custom_filter_ui_removes_sort_bar_from_custom_mode():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")

    assert 'st.subheader("Sort")' not in source
    assert "_sort_specs" not in source
    assert "_render_sort_summary" not in source
    assert "sort_1" not in source
    assert "direction_1" not in source


def test_funnel_tab_labels_return_static_names():
    labels = _funnel_tab_labels(
        {
            "Route": [FilterSpec("ibd_candidate_rule", "equals", "pivot")],
            "Entry Status": [FilterSpec("ibd_entry_status", "equals", "ACTIONABLE")],
            "Optional Quality Filters": [],
        }
    )

    assert labels == [
        "Route",
        "Entry Status",
        "Optional Quality Filters",
    ]


def test_current_filter_summary_is_grouped_by_funnel():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")

    assert "Filtered Rows" in source
    assert "_render_current_filter_summary" in source
    assert "_describe_filter_condition" in source


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


def test_route_funnel_defaults_to_active_signals():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")

    assert '"All Pool Records"' not in source
    assert 'FilterSpec("signal", "is true", label="Signal")' in source



