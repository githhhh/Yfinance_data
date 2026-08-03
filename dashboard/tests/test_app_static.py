from pathlib import Path
import ast
import re
from dashboard.app import _csv_cache_fingerprint


APP_SOURCE = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
STYLE_SOURCE = (Path(__file__).resolve().parents[1] / "review_styles.py").read_text(encoding="utf-8")


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


def test_dashboard_uses_stable_keyed_density_containers():
    for key in [
        "dashboard_shell",
        "dashboard_header",
        "review_queue",
        "status_cards",
        "filters",
        "filter_controls",
        "results_toolbar",
        "selected_row",
        "results_grid",
    ]:
        assert f'key="{key}"' in APP_SOURCE


def test_density_css_is_scoped_and_has_no_visual_compensation_hacks():
    combined_source = APP_SOURCE + STYLE_SOURCE
    assert re.search(
        r'^\s*div\[data-testid="stVerticalBlock"\]\s*>\s*div',
        combined_source,
        flags=re.MULTILINE,
    ) is None
    assert "margin-top:28px" not in combined_source
    assert "margin-bottom:4px" not in combined_source
    assert "margin: -" not in combined_source
    assert re.search(r"(?<!-)\btransform\s*:", combined_source) is None
    assert "min-height: 80px" in STYLE_SOURCE
    assert ".st-key-status_cards" in STYLE_SOURCE
    assert 'div[data-testid="stMainBlockContainer"]:has(.st-key-dashboard_shell)' in STYLE_SOURCE
    assert 'div[data-testid="stAppViewBlockContainer"]:has(.st-key-dashboard_shell)' not in STYLE_SOURCE
    assert 'div[data-testid="stApp"]:has(.st-key-dashboard_shell) [data-testid="stHeader"]' in STYLE_SOURCE
    assert 'div[data-testid="stAppViewContainer"]:has(.st-key-dashboard_shell) [data-testid="stHeader"]' not in STYLE_SOURCE
    assert ':has(.st-key-dashboard_shell)' in STYLE_SOURCE


def test_status_cards_render_exactly_two_text_lines():
    assert 'btn_label = f"{prefix}{display_name} · {count}\\n{meta[\'subtitle\']}"' in APP_SOURCE
    assert 'btn_label = f"{prefix}{display_name}\\n{count}\\n{sub_map[status_name]}"' not in APP_SOURCE
