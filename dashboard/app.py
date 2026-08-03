from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.data_utils import (
    apply_c_rank_mode,
    apply_default_review_order,
    build_chart_data,
    build_entry_status_counts,
    build_kpis,
    build_snapshot_freshness,
    filter_unconfirmed_near_trigger,
    load_pool_csv,
)
from dashboard.field_config import (
    FLOW_CARD_META,
    STATUS_META,
    get_all_table_columns,
    get_column_view_fields,
    get_field_label,
    get_midweek_table_columns,
)
from dashboard.services.bf_midweek_review import (
    ENTRY_VOL_ENABLED_STATUSES,
    PoolAnalysisResult,
    PoolMode,
    analyze_breakout_follow_pool,
    apply_review_filters,
    build_review_filter_counts,
    clear_quick_filters,
    default_review_state,
    materialize_review_view,
    reconcile_review_state,
    reset_to_all_signals,
    sort_review_rows,
    switch_review_mode,
    toggle_status_filter,
)
from dashboard.review_styles import REVIEW_UI_CSS
from dashboard.review_tooltip import FLOW_TOOLTIP_BRIDGE_HTML
from dashboard.table_view import render_table

st.set_page_config(page_title="Breakout Pool Dashboard", layout="wide", initial_sidebar_state="collapsed")


@st.cache_data
def cached_load_pool_csv(path: str, cache_fingerprint: tuple[int, int]) -> pd.DataFrame:
    del cache_fingerprint
    return load_pool_csv(path)


@st.cache_data
def cached_analyze_breakout_follow_pool(
    complete_path: str,
    midweek_path: str,
    window_date_value: str,
    cache_fingerprints: tuple[tuple[int, int], tuple[int, int]],
) -> PoolAnalysisResult:
    del cache_fingerprints
    return analyze_breakout_follow_pool(
        complete_path,
        midweek_path,
        window_date=date.fromisoformat(window_date_value),
    )


def _csv_cache_fingerprint(path: str | Path) -> tuple[int, int]:
    csv_path = Path(path)
    if not csv_path.exists():
        return (0, 0)
    stat = csv_path.stat()
    return (stat.st_mtime_ns, stat.st_size)


def _midweek_has_comparison(analysis: PoolAnalysisResult | None) -> bool:
    return bool(
        analysis is not None
        and getattr(analysis, "midweek_baseline_available", False)
    )


def main() -> None:
    args = _parse_args()

    st.markdown(f"<style>{REVIEW_UI_CSS}</style>", unsafe_allow_html=True)
    st.components.v1.html(FLOW_TOOLTIP_BRIDGE_HTML, height=0, scrolling=False)

    df: pd.DataFrame | None = None
    analysis: PoolAnalysisResult | None = None
    load_err: str | None = None
    try:
        window_date_value = args.window_date or datetime_business_date()
        analysis = cached_analyze_breakout_follow_pool(
            args.csv,
            args.midweek_csv,
            window_date_value,
            (
                _csv_cache_fingerprint(args.csv),
                _csv_cache_fingerprint(args.midweek_csv),
            ),
        )
        if "review_ui_state" not in st.session_state:
            st.session_state["review_ui_state"] = default_review_state(analysis.mode)
        state = dict(st.session_state["review_ui_state"])
        if state.get("mode") == "MIDWEEK" and not analysis.midweek_available:
            state = default_review_state(PoolMode.COMPLETE)
            st.session_state["review_ui_state"] = state
        elif state.get("mode") == "MIDWEEK" and not _midweek_has_comparison(analysis):
            reconciled_state = reconcile_review_state(
                state,
                PoolMode.MIDWEEK_WITHOUT_VALID_BASELINE,
            )
            if reconciled_state != state:
                st.session_state["review_ui_state"] = reconciled_state
            state = reconciled_state
        df = _view_dataframe(analysis, state)
    except Exception as exc:
        load_err = str(exc)

    with st.container(key="dashboard_shell"):
        with st.container(key="dashboard_header"):
            _render_header_bar(df, load_err, analysis)

        if analysis is not None:
            for warning in analysis.warnings:
                st.warning(warning, icon="⚠️")

        if df is None:
            st.error(f"Could not load breakout pool data: {load_err}")
            return

        mode = st.session_state.get("global_mode_selector", "IBD Review")
        if mode == "C Rank Reference":
            reference_df = analysis.complete_pool if analysis is not None else df
            _render_c_rank_reference_view(reference_df)
        else:
            _render_ibd_review_view(df, analysis)


def datetime_business_date() -> str:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    return datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()


def _view_dataframe(analysis: PoolAnalysisResult, state: dict[str, Any]) -> pd.DataFrame:
    if state.get("mode") == "MIDWEEK" and analysis.midweek_available:
        return materialize_review_view(analysis.midweek_review)
    complete = analysis.complete_pool.copy()
    complete["review_watch_active"] = complete["signal"]
    complete["review_effective_entry_status"] = complete["ibd_entry_status"]
    complete["review_priority"] = pd.to_numeric(
        complete.get("rank_C_continuous"), errors="coerce"
    )
    return complete


def _render_header_bar(
    df: pd.DataFrame | None,
    load_err: str | None,
    analysis: PoolAnalysisResult | None = None,
) -> None:
    col_l, col_r = st.columns([3, 1.5])
    with col_l:
        badge_html = (
            '<span class="data-badge data-badge--ready">Data Ready</span>'
            if df is not None
            else '<span class="data-badge data-badge--error">Schema / Data Error</span>'
        )
        state = st.session_state.get("review_ui_state", {})
        is_midweek = state.get("mode") == "MIDWEEK" and analysis is not None
        if is_midweek:
            snapshot_value = analysis.midweek_snapshot_date.isoformat() if analysis.midweek_snapshot_date else "N/A"
            baseline_value = (
                analysis.complete_snapshot_date.isoformat()
                if _midweek_has_comparison(analysis)
                and analysis.complete_snapshot_date
                else "unavailable"
            )
            snapshot_html = (
                f'<span class="snapshot-segment">Snapshot <b>{snapshot_value}</b></span> · '
                f'<span class="snapshot-mode-segment snapshot-mode-segment--midweek">Midweek · baseline {baseline_value}</span>'
            )
        else:
            freshness = build_snapshot_freshness(_get_snapshot_date(df) if df is not None else "N/A")
            age_label = (
                "Unknown"
                if freshness["age_days"] is None
                else f'{freshness["age_days"]}d old · '
                f'<span class="snapshot-freshness snapshot-freshness--{freshness["status"].lower()}">'
                f'{html.escape(freshness["label"])}</span>'
            )
            snapshot_html = (
                '<span class="snapshot-segment">Snapshot '
                f'<b>{html.escape(freshness["snapshot_date_str"])}</b></span> · '
                f'<span class="snapshot-mode-segment">{age_label}</span>'
            )
        total_pool = len(df) if df is not None else 0
        active_signals = int(df["signal"].sum()) if df is not None and "signal" in df.columns else 0
        st.markdown(
            f'<h3 class="dashboard-title">Breakout Pool {badge_html}</h3>'
            f'<div class="dashboard-snapshot">{snapshot_html} · <b>{total_pool}</b> Total Pool · <b>{active_signals}</b> Active Signals</div>',
            unsafe_allow_html=True,
        )

    with col_r:
        c1, c2 = st.columns([0.3, 2.7])
        with c1:
            if st.button("ⓘ", key="btn_info_rules", help="Status → Position → Daily Confirmation → Weekly Volume → C Rank"):
                _render_flow_rules_dialog()
        with c2:
            mode = st.segmented_control(
                "Mode Selector",
                ["IBD Review", "C Rank Reference"],
                default=st.session_state.get("global_mode_selector", "IBD Review") or "IBD Review",
                key="global_mode_selector",
                label_visibility="collapsed",
                help="Switch between IBD Breakout Review and C Rank Reference (evaluates Active Signals only).",
            )
            if mode is None:
                mode = "IBD Review"


@st.dialog("IBD Breakout Review Flow & Rules", width="large")
def _render_flow_rules_dialog() -> None:
    legend_lines = [
        f"- **{meta.get('label', key)}**：{meta.get('tooltip', '')}"
        for key, meta in STATUS_META.items()
    ]
    legend_md = "\n        ".join(legend_lines)
    st.markdown(
        f"""
        ### Review Flow
        `Status → Position → Daily Confirmation → Weekly Volume → C Rank`

        ### Status Legend
        {legend_md}

        ### Volume Definition
        - **Entry / Reason**：日线突破确认和日线量比。
        - **W Vol**：当前周成交量相对 10 周均量。
        - **C Rank**：质量对照，不代替 IBD 入场状态。
        """
    )


def _store_review_state(state: dict[str, Any]) -> None:
    st.session_state["review_ui_state"] = dict(state)


def _selected_button_label(selected: bool, label: str) -> str:
    return f"{'✓' if selected else ' '} {label}"


def _render_ibd_review_view(
    df: pd.DataFrame,
    analysis: PoolAnalysisResult | None = None,
) -> None:
    if "review_ui_state" not in st.session_state:
        st.session_state["review_ui_state"] = default_review_state(
            analysis.mode if analysis is not None else PoolMode.COMPLETE
        )
    state = dict(st.session_state["review_ui_state"])
    counts = build_review_filter_counts(df, state)

    with st.container(key="review_queue"):
        _render_status_queue(df, counts, state, analysis)

    with st.container(key="filters"):
        _render_filter_bar(df, state, counts)

    filtered_df = apply_review_filters(df, state)
    filtered_df = sort_review_rows(filtered_df, state["sort_mode"])
    has_comparison = (
        state["mode"] == "MIDWEEK"
        and _midweek_has_comparison(analysis)
    )

    with st.container(key="results_toolbar"):
        summary_col, actions_col = st.columns([1, 0.34], vertical_alignment="center")
        with summary_col:
            st.markdown(
                f'<div class="results-summary">{len(filtered_df)} results · Sorted by {html.escape(state["sort_mode"])}</div>',
                unsafe_allow_html=True,
            )
        with actions_col:
            with st.container(key="results_actions"):
                copy_col, sort_col = st.columns([1.35, 1], vertical_alignment="center")
                with copy_col:
                    _render_copy_codes_control(
                        filtered_df["code"].tolist(),
                        key_prefix=f'ibd_review_{state["mode"].lower()}',
                    )
                with sort_col:
                    options = ["Review Priority", "C Rank", "Distance"] if has_comparison else ["C Rank", "Distance"]
                    current_sort = state["sort_mode"] if state["sort_mode"] in options else options[0]
                    selected_sort = st.selectbox(
                        "Sort",
                        options,
                        index=options.index(current_sort),
                        key=f'review_sort_{state.get("widget_generation", 0)}',
                        label_visibility="collapsed",
                    )
                    if selected_sort != state["sort_mode"]:
                        state["sort_mode"] = selected_sort
                        _store_review_state(state)
                        st.rerun()

    from dashboard.field_config import get_default_table_columns
    columns = get_midweek_table_columns() if has_comparison else get_default_table_columns()
    grid_key = (
        "review_results_grid_midweek"
        if has_comparison
        else "review_results_grid_weekend"
    )

    with st.container(key="selected_row"):
        detail_container = st.empty()
    with st.container(key="results_grid"):
        selected_code = render_table(
            filtered_df,
            columns,
            grid_key=grid_key,
            show_origin_badge=has_comparison,
            height=480,
        )

    with detail_container.container():
        _render_selected_row_detail(filtered_df, selected_code)

    st.markdown("---")
    _download_current_rows(filtered_df, "ibd_review_filtered.csv")


def _render_mode_scope_controls(
    state: dict[str, Any],
    analysis: PoolAnalysisResult | None,
) -> None:
    midweek_available = analysis.midweek_available if analysis is not None else False
    is_midweek = state["mode"] == "MIDWEEK"
    has_comparison = (
        is_midweek
        and _midweek_has_comparison(analysis)
    )
    change_total = 0
    if analysis is not None:
        change_total = sum(
            analysis.summary.get(value, 0)
            for value in ("BECAME_ACTIONABLE", "LEFT_ACTIONABLE", "OTHER_CHANGES")
        )
    all_total = int(df_active_count_for_state(state, analysis))

    with st.container(key="review_queue_heading"):
        title_col, mode_col, scope_col = st.columns([5.0, 2.1, 2.1], vertical_alignment="center")
        with title_col:
            st.markdown("##### Review Queue")
        with mode_col:
            with st.container(key="review_mode_controls"):
                mode_cols = st.columns(2, gap="small")
                with mode_cols[0]:
                    if st.button(
                        _selected_button_label(state["mode"] == "MIDWEEK", "Midweek Review"),
                        key="btn_mode_midweek",
                        use_container_width=True,
                        disabled=not midweek_available,
                        type="primary" if state["mode"] == "MIDWEEK" else "secondary",
                    ):
                        _store_review_state(
                            switch_review_mode(
                                state,
                                "MIDWEEK",
                                midweek_has_baseline=_midweek_has_comparison(analysis),
                            )
                        )
                        st.rerun()
                with mode_cols[1]:
                    if st.button(
                        _selected_button_label(state["mode"] == "WEEKEND", "Weekend Full Pool"),
                        key="btn_mode_weekend",
                        use_container_width=True,
                        type="primary" if state["mode"] == "WEEKEND" else "secondary",
                    ):
                        _store_review_state(switch_review_mode(state, "WEEKEND"))
                        st.rerun()
        with scope_col:
            with st.container(key="review_scope_controls"):
                scope_cols = st.columns(2, gap="small")
                with scope_cols[0]:
                    if st.button(
                        _selected_button_label(state["scope"] == "CHANGES", f"Changes ({change_total})"),
                        key="btn_scope_changes",
                        use_container_width=True,
                        disabled=not has_comparison,
                        type="primary" if state["scope"] == "CHANGES" else "secondary",
                    ):
                        state["scope"] = "CHANGES"
                        _store_review_state(state)
                        st.rerun()
                with scope_cols[1]:
                    if st.button(
                        _selected_button_label(state["scope"] == "ALL_SIGNALS", f"All Signals ({all_total})"),
                        key="btn_scope_all_signals",
                        use_container_width=True,
                        type="primary" if state["scope"] == "ALL_SIGNALS" else "secondary",
                    ):
                        _store_review_state(reset_to_all_signals(state))
                        st.rerun()


def df_active_count_for_state(
    state: dict[str, Any],
    analysis: PoolAnalysisResult | None,
) -> int:
    if analysis is None:
        return 0
    if state["mode"] == "MIDWEEK":
        return int(analysis.summary.get("ACTIVE_SIGNALS", 0))
    if analysis.complete_pool.empty:
        return 0
    return int(analysis.complete_pool["signal"].map(bool).sum())


def _render_filter_card(
    card_id: str,
    button_label: str,
    metadata: dict[str, str],
    *,
    selected: bool,
    button_key: str,
) -> bool:
    with st.container(key=f"flow_card_{card_id.lower()}"):
        clicked = st.button(
            button_label,
            key=button_key,
            use_container_width=True,
            type="primary" if selected else "secondary",
        )
        tooltip_text = html.escape(metadata["tooltip"], quote=True)
        tooltip_title = html.escape(metadata["tooltip_title"], quote=True)
        tooltip_label = html.escape(f"{metadata['tooltip_title']} info", quote=True)
        st.markdown(
            '<button type="button" class="flow-info-trigger" '
            f'data-flow-tooltip-title="{tooltip_title}" data-flow-tooltip="{tooltip_text}" '
            f'aria-label="{tooltip_label}" aria-expanded="false">i</button>',
            unsafe_allow_html=True,
        )
        return clicked


def _toggle_dimension(state: dict[str, Any], field: str, value: str) -> dict[str, Any]:
    result = dict(state)
    result[field] = "ALL" if result.get(field) == value else value
    return result


def _render_quick_group(
    card_ids: tuple[str, str, str],
    *,
    count_field: str,
    state_field: str,
    counts: dict[str, Any],
    state: dict[str, Any],
) -> None:
    columns = st.columns(3, gap="small")
    for column, card_id in zip(columns, card_ids, strict=True):
        metadata = FLOW_CARD_META[card_id]
        with column:
            if _render_filter_card(
                card_id,
                f"{metadata['label']} · {counts[count_field][card_id]}",
                metadata,
                selected=state[state_field] == card_id,
                button_key=f"btn_{state_field}_{card_id}",
            ):
                _store_review_state(_toggle_dimension(state, state_field, card_id))
                st.rerun()


def _render_review_context(
    df: pd.DataFrame,
    counts: dict[str, Any],
    state: dict[str, Any],
    analysis: PoolAnalysisResult | None,
) -> None:
    del df
    with st.container(key="review_context_slot"):
        if state["mode"] != "MIDWEEK":
            st.markdown(
                '<div class="weekend-context-bar"><strong>Weekend Baseline</strong>'
                '<span>Complete weekly pool</span>'
                '<span>Midweek comparison is not applied in this view.</span></div>',
                unsafe_allow_html=True,
            )
            return
        if not _midweek_has_comparison(analysis):
            st.markdown(
                '<div class="weekend-context-bar"><strong>Midweek Snapshot</strong>'
                '<span>No valid complete-week baseline</span>'
                '<span>Change and Origin comparison is unavailable.</span></div>',
                unsafe_allow_html=True,
            )
            return

        with st.container(key="quick_context_row"):
            change_label, change_group, divider, origin_label, origin_group, clear_col = st.columns(
                [0.48, 3.95, 0.16, 0.48, 3.95, 0.72], gap="small"
            )
            with change_label:
                with st.container(key="quick_label_change"):
                    st.caption("CHANGE")
            with change_group:
                _render_quick_group(
                    ("BECAME_ACTIONABLE", "LEFT_ACTIONABLE", "OTHER_CHANGES"),
                    count_field="change",
                    state_field="change_filter",
                    counts=counts,
                    state=state,
                )
            with divider:
                with st.container(key="quick_divider"):
                    st.markdown('<span aria-hidden="true"></span>', unsafe_allow_html=True)
            with origin_label:
                with st.container(key="quick_label_origin"):
                    st.caption("ORIGIN")
            with origin_group:
                _render_quick_group(
                    ("NEW", "CARRY", "RECONFIRMED"),
                    count_field="origin",
                    state_field="origin_filter",
                    counts=counts,
                    state=state,
                )
            with clear_col:
                if st.button(
                    "Clear",
                    key="btn_clear_quick",
                    use_container_width=True,
                    disabled=(state["change_filter"] == "ALL" and state["origin_filter"] == "ALL"),
                ):
                    _store_review_state(clear_quick_filters(state))
                    st.rerun()


def _render_status_queue(
    df: pd.DataFrame,
    counts: dict[str, Any],
    state: dict[str, Any],
    analysis: PoolAnalysisResult | None,
) -> None:
    _render_mode_scope_controls(state, analysis)
    _render_review_context(df, counts, state, analysis)
    scope_label = "CHANGED SIGNALS" if state["scope"] == "CHANGES" else "ALL SIGNALS"
    st.caption(f"CURRENT ENTRY STATUS · {scope_label}")
    with st.container(key="status_cards"):
        cols = st.columns(4)
        statuses = ["ACTIONABLE", "UNCONFIRMED", "BELOW_TRIGGER", "EXTENDED"]
        for i, status_name in enumerate(statuses):
            with cols[i]:
                count = counts["status"].get(status_name, 0)
                is_active = state["status_filter"] == status_name
                prefix = "✓ " if is_active else "  "
                display_name = status_name.replace("_", " ")
                meta = STATUS_META[status_name]
                btn_label = f"{prefix}{display_name} · {count}\n{meta['subtitle']}"
                if _render_filter_card(
                    status_name,
                    btn_label,
                    meta,
                    selected=is_active,
                    button_key=f"btn_status_{status_name}",
                ):
                    _store_review_state(toggle_status_filter(state, status_name))
                    st.rerun()


def _active_filter_count(state: dict[str, Any]) -> int:
    return sum(
        [
            state.get("route_filter", "All") != "All",
            bool(state.get("distance_min")),
            bool(state.get("distance_max")),
            bool(state.get("entry_volume_min")),
            bool(state.get("weekly_volume_min")),
            bool(state.get("near_trigger_only")),
        ]
    )


def _render_filter_bar(
    df: pd.DataFrame,
    state: dict[str, Any],
    counts: dict[str, Any],
) -> None:
    active_count = _active_filter_count(state)
    summary = "No filters applied" if active_count == 0 else f"{active_count} active"
    with st.container(key="filters_header"):
        if st.button(
            f"Filters · {summary}",
            key="btn_filters_toggle",
            use_container_width=True,
        ):
            state["filters_expanded"] = not state["filters_expanded"]
            _store_review_state(state)
            st.rerun()
    if not state["filters_expanded"]:
        return

    generation = state.get("widget_generation", 0)
    controls = st.container(key="filter_controls")
    cols = controls.columns([1.6, 1.1, 1.1, 1.1, 1.1, 0.8], vertical_alignment="bottom")
    with cols[0]:
        routes = ["All"] + _unique_values(df, "ibd_candidate_rule")
        current_route = state.get("route_filter", "All")
        selected_route = st.selectbox(
            "Route (Rule)",
            routes,
            index=routes.index(current_route) if current_route in routes else 0,
            key=f"review_route_{generation}",
        )
        if selected_route != current_route:
            state["route_filter"] = selected_route
            _store_review_state(state)
            st.rerun()
    with cols[1]:
        val = st.text_input("Distance Min %", value=state.get("distance_min", ""), key=f"review_dist_min_{generation}")
        if val != state.get("distance_min", ""):
            state["distance_min"] = val
            _store_review_state(state)
            st.rerun()
    with cols[2]:
        val = st.text_input("Distance Max %", value=state.get("distance_max", ""), key=f"review_dist_max_{generation}")
        if val != state.get("distance_max", ""):
            state["distance_max"] = val
            _store_review_state(state)
            st.rerun()
    with cols[3]:
        current_status = state.get("status_filter", "ALL")
        if current_status == "UNCONFIRMED":
            val = st.checkbox(
                "Near Trigger ≤ +3%",
                value=state.get("near_trigger_only", False),
                key=f"review_near_{generation}",
            )
            if val != state.get("near_trigger_only", False):
                state["near_trigger_only"] = val
                _store_review_state(state)
                st.rerun()
        else:
            is_disabled = current_status not in ENTRY_VOL_ENABLED_STATUSES
            val = st.text_input(
                "Entry Vol Min (x)",
                value="" if is_disabled else state.get("entry_volume_min", ""),
                placeholder="N/A (Disabled)" if is_disabled else "",
                disabled=is_disabled,
                key=f"review_entry_vol_{generation}",
            )
            if not is_disabled and val != state.get("entry_volume_min", ""):
                state["entry_volume_min"] = val
                _store_review_state(state)
                st.rerun()
    with cols[4]:
        val = st.text_input("Weekly Vol Min (x)", value=state.get("weekly_volume_min", ""), key=f"review_weekly_vol_{generation}")
        if val != state.get("weekly_volume_min", ""):
            state["weekly_volume_min"] = val
            _store_review_state(state)
            st.rerun()
    with cols[5]:
        if st.button("Reset", key="btn_filters_reset", use_container_width=True):
            reset = dict(state)
            reset.update(
                {
                    "route_filter": "All",
                    "status_filter": "ALL",
                    "distance_min": "",
                    "distance_max": "",
                    "entry_volume_min": "",
                    "weekly_volume_min": "",
                    "near_trigger_only": False,
                    "widget_generation": generation + 1,
                }
            )
            _store_review_state(reset)
            st.rerun()


def _render_selected_row_detail(filtered_df: pd.DataFrame, selected_code: str | None) -> None:
    if filtered_df.empty:
        st.markdown(
            '<div class="selected-strip selected-strip--empty" role="status">'
            '<span>No matching records found with current filter criteria.</span></div>',
            unsafe_allow_html=True,
        )
        return

    if selected_code is None or selected_code not in filtered_df["code"].values:
        st.markdown(
            '<div class="selected-strip selected-strip--empty" role="status">'
            '<span>Select a row to inspect review details.</span></div>',
            unsafe_allow_html=True,
        )
        return

    row = filtered_df[filtered_df["code"] == selected_code].iloc[0]

    code = html.escape(str(row.get("code", "N/A")))
    cand_price = html.escape(_format_number(row.get("ibd_candidate_price"), ""))
    cand_rule = html.escape(str(row.get("ibd_candidate_rule", "N/A")))
    dist_pct = html.escape(_format_number(row.get("current_vs_ibd_candidate_pct"), "%"))
    latest_close = html.escape(_format_number(row.get("latest_close"), ""))
    status_name = html.escape(str(row.get("ibd_entry_status", "N/A")))
    vol_or_reject = html.escape(str(row.get("ibd_entry_vol_or_reject", "N/A")))
    rank_c = html.escape(str(row.get("rank_C_continuous", "N/A")))
    c_cont = html.escape(_format_number(row.get("C_continuous"), ""))

    raw_origin = row.get("review_signal_origin")
    origin = (
        str(raw_origin).strip()
        if raw_origin is not None and not pd.isna(raw_origin) and str(raw_origin).strip() not in {"", "NONE"}
        else ""
    )
    raw_change = row.get("review_change_label")
    change_summary = (
        str(raw_change).strip()
        if raw_change is not None and not pd.isna(raw_change) and str(raw_change).strip()
        else ""
    )
    baseline_raw = row.get("review_baseline_entry_status")
    baseline_status = (
        str(baseline_raw).strip().replace("_", " ")
        if baseline_raw is not None and not pd.isna(baseline_raw) and str(baseline_raw).strip()
        else ""
    )
    origin_badge = (
        f'<span class="selected-origin" data-origin="{html.escape(origin)}">{html.escape("RECONF." if origin == "RECONFIRMED" else origin)}</span>'
        if origin
        else ""
    )
    change_markup = (
        f'<div class="selected-change-summary">{html.escape(change_summary)}</div>'
        if change_summary
        else ""
    )
    status_color = STATUS_META.get(str(row.get("ibd_entry_status", "N/A")), {}).get("color", "#4caf50" if status_name == "ACTIONABLE" else "#f2f5f9")
    status_transition = (
        f'{html.escape(baseline_status)} → <span style="color:{status_color};">{status_name.replace("_", " ")}</span>'
        if baseline_status
        else f'<span style="color:{status_color};">{status_name.replace("_", " ")}</span>'
    )

    eps_yoy = html.escape(_format_card_val(row.get("eps_yoy_growth"), "%"))
    dist_52w = html.escape(_format_card_val(row.get("dist_to_52w_high_pct"), "%"))
    p_52w = html.escape(_format_card_val(row.get("price_52_week_high"), ""))
    base_depth = html.escape(_format_card_val(row.get("base_depth_pct"), "%"))
    base_dur = html.escape(_format_card_val(row.get("base_duration_weeks"), "w"))

    pb_depth = html.escape(_format_card_val(row.get("pullback_pct"), "%"))
    pb_off_peak = html.escape(_format_card_val(row.get("pullback_pct_off_peak"), "%"))
    pullback_section = (
        f"""
                            <div class="code-popup-section" data-popup-section="pullback">
                                <div class="code-popup-title">2. Pullback</div>
                                <div class="code-popup-grid-2">
                                    <div><div class="code-popup-item">Pullback Depth</div><div class="code-popup-val">{pb_depth}</div></div>
                                    <div><div class="code-popup-item">Off Pullback Peak</div><div class="code-popup-val">{pb_off_peak}</div></div>
                                </div>
                            </div>
        """
        if pb_depth != "n/a" or pb_off_peak != "n/a"
        else ""
    )

    raw_valid = row.get("ibd_entry_valid")
    is_entry_valid = bool(pd.notna(raw_valid) and (raw_valid is True or str(raw_valid).strip().lower() in ("true", "1")))
    trigger_p = html.escape(_format_card_val(row.get("ibd_trigger_price"), ""))
    raw_reason = row.get("ibd_entry_reject_reason")
    reject_reason_str = html.escape(str(raw_reason).strip() if (raw_reason is not None and not pd.isna(raw_reason) and str(raw_reason).strip() not in ("", "nan", "None", "N/A")) else "n/a")

    if is_entry_valid:
        raw_date = row.get("ibd_entry_date")
        entry_date_str = html.escape(str(raw_date).split("T")[0] if (raw_date is not None and not pd.isna(raw_date) and str(raw_date).strip() not in ("", "nan", "None", "N/A")) else "n/a")
        daily_vol_str = html.escape(_format_card_val(row.get("ibd_entry_volume_ratio"), "x"))
        reject_section = ""
    else:
        entry_date_str = "n/a"
        daily_vol_str = "n/a"
        reject_section = f"""
                                <div class="code-popup-reject" role="alert">
                                    <div class="code-popup-item">Reject Reason</div>
                                    <div class="code-popup-val">{reject_reason_str}</div>
                                </div>
        """

    detail_html = (
        f"""
        <style>
        .st-key-selected_row:has(.code-detail:hover),
        .st-key-selected_row:has(.code-detail[open]),
        .st-key-selected_row:has(.code-hover-trigger:focus-visible) {{
            position: relative;
            z-index: 1000;
        }}
        .st-key-selected_row .code-hover-wrapper {{ position: relative; display: inline-block; }}
        .st-key-selected_row .code-detail {{ display: inline-block; }}
        .st-key-selected_row .code-hover-trigger {{
            anchor-name: --selected-code;
            display: list-item;
            font-size: 18px;
            font-weight: 800;
            color: #1f77b4;
            cursor: pointer;
            border-bottom: 1px dotted #1f77b4;
            list-style: none;
        }}
        .st-key-selected_row .code-hover-trigger::-webkit-details-marker {{ display: none; }}
        .st-key-selected_row .code-hover-trigger::marker {{ content: ""; }}
        .st-key-selected_row .code-hover-popup {{
            display: none;
            position: fixed;
            position-anchor: --selected-code;
            position-area: block-start span-inline-end;
            position-try-fallbacks: flip-block;
            width: min(450px, calc(100vw - 24px));
            max-height: min(360px, calc(50dvh - 12px));
            overflow-y: auto;
            overscroll-behavior: contain;
            box-sizing: border-box;
            padding: 8px 0;
            z-index: 999999;
            text-align: left;
        }}
        .st-key-selected_row .code-hover-surface {{
            background: #1e2631;
            border: 1px solid #4a5a6a;
            border-radius: 8px;
            padding: 12px 14px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.6);
        }}
        .st-key-selected_row .code-detail:hover > .code-hover-popup,
        .st-key-selected_row .code-hover-popup:hover,
        .st-key-selected_row .code-detail:has(> .code-hover-trigger:focus-visible) > .code-hover-popup,
        .st-key-selected_row .code-detail[open] > .code-hover-popup {{
            display: block;
        }}
        .st-key-selected_row .code-popup-section {{ margin-bottom: 10px; border-bottom: 1px solid #303947; padding-bottom: 8px; }}
        .st-key-selected_row .code-popup-section:last-child {{ margin-bottom: 0; border-bottom: none; padding-bottom: 0; }}
        .st-key-selected_row .code-popup-title {{ font-size: 11px; font-weight: 700; color: #8899a6; text-transform: uppercase; margin-bottom: 6px; letter-spacing: 0.5px; }}
        .st-key-selected_row .code-popup-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px 10px; }}
        .st-key-selected_row .code-popup-grid-2 {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 6px 10px; }}
        .st-key-selected_row .code-popup-item {{ font-size: 11px; color: #a0aec0; }}
        .st-key-selected_row .code-popup-val {{ font-size: 13px; font-weight: 700; color: #f2f5f9; }}
        .st-key-selected_row .code-popup-reject {{
            margin-top: 8px;
            padding: 8px 10px;
            border: 1px solid #ffb300;
            border-radius: 6px;
            background: rgba(255, 179, 0, 0.12);
        }}
        .st-key-selected_row .code-popup-reject .code-popup-val {{ color: #ffca54; font-size: 12px; }}
        .st-key-selected_row .selected-summary-cell {{ min-width:0; }}
        .st-key-selected_row .selected-origin {{ display:inline-flex; margin-left:6px; padding:2px 5px; border:1px solid #475569; border-radius:3px; color:#cbd5e1; font-size:9px; line-height:1; vertical-align:middle; }}
        .st-key-selected_row .selected-change-summary {{ margin-top:2px; color:#2dd4bf; font-size:9px; font-weight:700; text-transform:uppercase; }}
        </style>
        <div class="selected-strip">
                <div class="selected-summary-cell selected-code-cell">
                    <div class="code-hover-wrapper">
                        <details class="code-detail" data-selected-code="{code}">
                            <summary class="code-hover-trigger" aria-label="{code} secondary details">{code} ▾ {origin_badge}</summary>
                            <div class="code-hover-popup" role="region" aria-label="{code} secondary details">
                            <div class="code-hover-surface">
                            <div class="code-popup-section" data-popup-section="daily-entry">
                                <div class="code-popup-title">1. Daily Entry</div>
                                <div class="code-popup-grid">
                                    <div><div class="code-popup-item">Trigger</div><div class="code-popup-val">{trigger_p}</div></div>
                                    <div><div class="code-popup-item">Entry Date</div><div class="code-popup-val">{entry_date_str}</div></div>
                                    <div><div class="code-popup-item">Daily Entry Vol</div><div class="code-popup-val">{daily_vol_str}</div></div>
                                </div>
                                {reject_section}
                            </div>
                            {pullback_section}
                            <div class="code-popup-section" data-popup-section="canslim-base">
                                <div class="code-popup-title">3. CANSLIM / Base</div>
                                <div class="code-popup-grid">
                                    <div><div class="code-popup-item">EPS YoY</div><div class="code-popup-val">{eps_yoy}</div></div>
                                    <div><div class="code-popup-item">To 52W High</div><div class="code-popup-val">{dist_52w}</div></div>
                                    <div><div class="code-popup-item">52W High</div><div class="code-popup-val">{p_52w}</div></div>
                                    <div><div class="code-popup-item">Ceiling/Base Depth</div><div class="code-popup-val">{base_depth}</div></div>
                                    <div><div class="code-popup-item">Base Duration</div><div class="code-popup-val">{base_dur}</div></div>
                                </div>
                            </div>
                            </div>
                            </div>
                        </details>
                    </div>
                    {change_markup}
                </div>
                <div class="selected-summary-cell">
                    <div style="font-size:11px; color:#8899a6; text-transform:uppercase;">Candidate Price</div>
                    <div style="font-size:15px; font-weight:700; color:#f2f5f9;">{cand_price} <span style="font-size:11px; font-weight:normal; color:#a0aec0;">({cand_rule})</span></div>
                </div>
                <div class="selected-summary-cell">
                    <div style="font-size:11px; color:#8899a6; text-transform:uppercase;">Current vs Candidate</div>
                    <div style="font-size:15px; font-weight:700; color:#f2f5f9;">{dist_pct} <span style="font-size:11px; font-weight:normal; color:#a0aec0;">(Close: {latest_close})</span></div>
                </div>
                <div class="selected-summary-cell">
                    <div style="font-size:11px; color:#8899a6; text-transform:uppercase;">Entry Status</div>
                    <div style="font-size:15px; font-weight:700; color:#f2f5f9;">{status_transition} <span style="font-size:11px; font-weight:normal; color:#a0aec0;">({vol_or_reject})</span></div>
                </div>
                <div class="selected-summary-cell">
                    <div style="font-size:11px; color:#8899a6; text-transform:uppercase;">C Rank & Continuous</div>
                    <div style="font-size:15px; font-weight:700; color:#f2f5f9;">#{rank_c} <span style="font-size:11px; font-weight:normal; color:#a0aec0;">({c_cont})</span></div>
                </div>
        </div>
        """
    )
    st.markdown(
        "\n".join(line for line in detail_html.splitlines() if line.strip()),
        unsafe_allow_html=True,
    )


def _render_c_rank_reference_view(df: pd.DataFrame) -> None:
    active_signals_count = int((df["signal"] == True).sum()) if "signal" in df.columns else len(df)
    denom = active_signals_count if active_signals_count > 0 else len(df)
    st.markdown("##### C Rank Reference View (`signal=True` · Sorted by `rank_C_continuous` asc)")
    
    with st.expander("ℹ️ C Rank Selection & Reference Rules", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                "\n".join(
                    [
                        "**Fixed Mode Rules**",
                        "- Exclusively evaluates Active Signals (`signal=True`) across the pool.",
                        "- Sorted by `rank_C_continuous` asc to horizontally benchmark quality.",
                        "- Top N slice selector only (custom filters ignored).",
                        "- Auxiliary benchmark; does not replace IBD review status.",
                    ]
                )
            )
        with c2:
            st.markdown(
                "\n".join(
                    [
                        "**Ranking Formula Reference**",
                        "- 2.5 x pct(base_depth_abs)",
                        "- 2.0 x pct(pct_above_ceiling)",
                        "- 0.5 x pct(volume_ratio)",
                        "- 0.5 x fresh_touch / fresh_pullback",
                    ]
                )
            )

    with st.container(key="results_toolbar"):
        col_limit, col_summary, col_copy = st.columns([1.3, 2.3, 2.4], vertical_alignment="bottom")
        with col_limit:
            limit_label = st.selectbox("Top N Slice", ["All rows", "Top 10", "Top 20", "Top 30", "Top 50"], index=0, key="c_rank_top_n_select")
            limit = None if limit_label == "All rows" else int(limit_label.split()[1])

        ranked = apply_c_rank_mode(df, limit=limit)

        with col_summary:
            st.markdown(
                f'<div style="font-size:14px; font-weight:600; color:#c5ceda;">Showing: {len(ranked)} of {denom} Active Signals · Reference Only</div>',
                unsafe_allow_html=True,
            )
        with col_copy:
            _render_copy_codes_control(ranked["code"].tolist(), key_prefix="c_rank_ref")

    from dashboard.field_config import get_column_view_fields
    columns = get_column_view_fields("C Rank Reference")

    with st.container(key="selected_row"):
        detail_container = st.empty()
    with st.container(key="results_grid"):
        selected_code = render_table(
            ranked,
            [column for column in columns if column in ranked.columns],
            grid_key="c_rank_reference_grid",
            show_origin_badge=False,
            height=520,
        )

    with detail_container.container():
        _render_selected_row_detail(ranked, selected_code)

    st.markdown("---")
    _download_current_rows(ranked, "c_rank_reference.csv")


def _render_copy_codes_control(codes: list[str], key_prefix: str = "") -> None:
    valid_codes = [str(code).strip() for code in codes if pd.notna(code) and str(code).strip()]
    codes_str = ", ".join(valid_codes)
    n = len(valid_codes)
    disabled_attr = " disabled" if n == 0 else ""
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        html, body {{
            height: 44px;
            width: 100%;
            overflow: hidden;
            background: transparent;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }}
        .copy-wrapper {{
            display: flex;
            align-items: center;
            justify-content: flex-end;
            height: 44px;
            width: 100%;
        }}
        .copy-btn {{
            background: #2e7d32;
            color: #fff;
            border: none;
            border-radius: 8px;
            padding: 0 14px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            height: 44px;
            width: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
            transition: background 0.2s;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            font-variant-numeric: tabular-nums;
        }}
        .copy-btn:disabled {{
            cursor: not-allowed;
            opacity: 0.58;
            background: #29462f;
        }}
    </style>
    </head>
    <body>
    <div class="copy-wrapper">
        <button id="copyBtn_{key_prefix}" class="copy-btn"{disabled_attr}>
            Copy {n} Codes
        </button>
    </div>
    <script>
        const btn = document.getElementById('copyBtn_{key_prefix}');
        const textToCopy = {json.dumps(codes_str)};
        if (btn) {{
            btn.addEventListener('click', async () => {{
                let success = false;
                if (navigator.clipboard && navigator.clipboard.writeText) {{
                    try {{
                        await navigator.clipboard.writeText(textToCopy);
                        success = true;
                    }} catch (e) {{
                        // fallback to execCommand
                    }}
                }}
                if (!success) {{
                    try {{
                        const ta = document.createElement('textarea');
                        ta.value = textToCopy;
                        ta.style.position = 'fixed';
                        ta.style.left = '-9999px';
                        document.body.appendChild(ta);
                        ta.focus();
                        ta.select();
                        success = document.execCommand('copy');
                        document.body.removeChild(ta);
                    }} catch (e2) {{
                        success = false;
                    }}
                }}
                if (success) {{
                    btn.style.background = '#1b5e20';
                    btn.innerText = '✓ Copied ({n})';
                }} else {{
                    btn.style.background = '#c62828';
                    btn.innerText = 'Copy failed';
                }}
                setTimeout(() => {{
                    btn.style.background = '#2e7d32';
                    btn.innerText = 'Copy {n} Codes';
                }}, 2000);
            }});
        }}
    </script>
    </body>
    </html>
    """
    st.components.v1.html(html_code, height=44, scrolling=False)


def _download_current_rows(df: pd.DataFrame, filename: str) -> None:
    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name=filename,
        mime="text/csv",
    )


def _unique_values(df: pd.DataFrame, field: str) -> list[str]:
    if field not in df.columns:
        return []
    values = df[field].dropna().astype(str).sort_values().unique().tolist()
    return [value for value in values if value]


def _get_snapshot_date(df: pd.DataFrame) -> str:
    if "snapshot_date" in df.columns:
        valid = df["snapshot_date"].dropna()
        if not valid.empty:
            return str(valid.iloc[0]).split(" ")[0].split("T")[0]
    return "N/A"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    default_csv = Path(__file__).resolve().parents[1] / "us" / "breakout_follow_pool.csv"
    parser.add_argument("--csv", default=str(default_csv))
    parser.add_argument(
        "--midweek-csv",
        default=str(default_csv.with_name("breakout_follow_pool_midweek.csv")),
    )
    parser.add_argument("--window-date", default=None)
    args, _ = parser.parse_known_args()
    return args


def _format_card_val(value: object, suffix: str = "") -> str:
    if value is None or pd.isna(value) or str(value).strip() in ("", "nan", "None", "N/A", "n/a"):
        return "n/a"
    try:
        val = float(value)
        if suffix == "%":
            if val > 0:
                return f"+{val:.2f}%"
            elif val < 0:
                return f"{val:.2f}%"
            return "0.00%"
        elif suffix == "x":
            return f"{val:.2f}x"
        elif suffix == "w":
            return f"{int(val)}w" if val.is_integer() else f"{val:.1f}w"
        elif suffix == "":
            return f"{val:.2f}"
        return f"{val:.2f}{suffix}"
    except (ValueError, TypeError):
        return str(value)


def _format_number(value: float | int | None, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if suffix == "%":
        val = float(value)
        if val > 0:
            return f"+{val:.2f}%"
        elif val < 0:
            return f"{val:.2f}%"
        else:
            return "0.00%"
    if suffix == "x":
        return f"{float(value):.2f}x"
    return f"{float(value):.2f}{suffix}"


if __name__ == "__main__":
    main()
