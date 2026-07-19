from __future__ import annotations

import argparse
import html
import json
import sys
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
    STATUS_META,
    get_all_table_columns,
    get_column_view_fields,
    get_field_label,
)
from dashboard.table_view import render_table

st.set_page_config(page_title="Breakout Pool Dashboard", layout="wide", initial_sidebar_state="collapsed")


@st.cache_data
def cached_load_pool_csv(path: str, cache_fingerprint: tuple[int, int]) -> pd.DataFrame:
    del cache_fingerprint
    return load_pool_csv(path)


def _csv_cache_fingerprint(path: str | Path) -> tuple[int, int]:
    stat = Path(path).stat()
    return (stat.st_mtime_ns, stat.st_size)


def main() -> None:
    args = _parse_args()

    st.markdown(
        """
        <style>
        div[data-testid="stApp"]:has(.st-key-dashboard_shell) [data-testid="stHeader"] {
            display: none !important;
        }
        div[data-testid="stApp"]:has(.st-key-dashboard_shell) [data-testid="stSidebar"] {
            display: none !important;
        }
        div[data-testid="stMainBlockContainer"]:has(.st-key-dashboard_shell) {
            padding: 8px 16px 16px !important;
            max-width: 98% !important;
        }
        .st-key-dashboard_shell > div[data-testid="stVerticalBlock"] {
            gap: 6px !important;
        }
        .st-key-dashboard_header > div[data-testid="stVerticalBlock"],
        .st-key-review_queue > div[data-testid="stVerticalBlock"],
        .st-key-filters > div[data-testid="stVerticalBlock"],
        .st-key-results_toolbar > div[data-testid="stVerticalBlock"],
        .st-key-selected_row > div[data-testid="stVerticalBlock"],
        .st-key-results_grid > div[data-testid="stVerticalBlock"] {
            gap: 4px !important;
        }
        /* Target exact status card buttons (4 cards) */
        .st-key-status_cards button {
            height: 78px !important;
            min-height: 78px !important;
            max-height: 78px !important;
            width: 100% !important;
            white-space: pre-line !important;
            padding: 8px 6px !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            line-height: 1.25 !important;
            border-radius: 8px !important;
            font-size: 13px !important;
        }
        .st-key-review_queue div[class*="st-key-btn_status_"] button[kind="secondary"],
        .st-key-review_queue div[class*="st-key-btn_all_signals"] button[kind="secondary"] {
            border: 1px solid #303947 !important;
            background: #141a22 !important;
            color: #cbd5e1 !important;
        }
        .st-key-review_queue div[class*="st-key-btn_status_"] button[kind="secondary"]:hover,
        .st-key-review_queue div[class*="st-key-btn_status_"] button[kind="secondary"]:focus,
        .st-key-review_queue div[class*="st-key-btn_all_signals"] button[kind="secondary"]:hover,
        .st-key-review_queue div[class*="st-key-btn_all_signals"] button[kind="secondary"]:focus {
            border: 1px solid #4a5a6a !important;
            background: #1e2631 !important;
            color: #f8fafc !important;
        }
        .st-key-review_queue div[class*="st-key-btn_status_"] button[kind="primary"],
        .st-key-review_queue div[class*="st-key-btn_all_signals"] button[kind="primary"] {
            border: 1.5px solid #3b82f6 !important;
            background: #1e293b !important;
            color: #ffffff !important;
        }
        .st-key-review_queue div[class*="st-key-btn_status_"] button[kind="primary"]:hover,
        .st-key-review_queue div[class*="st-key-btn_all_signals"] button[kind="primary"]:hover {
            border: 1.5px solid #60a5fa !important;
            background: #334155 !important;
            color: #ffffff !important;
        }
        .st-key-status_cards button * {
            margin: 0 !important;
            padding: 0 !important;
            line-height: 1.25 !important;
        }
        /* All Signals button */
        .st-key-review_queue div[class*="st-key-btn_all_signals"] button {
            height: 44px !important;
            min-height: 44px !important;
            min-width: 190px !important;
            max-width: 210px !important;
            margin-left: auto !important;
            border-radius: 8px !important;
            font-size: 13px !important;
            white-space: nowrap !important;
        }
        /* Ensure horizontal blocks align items vertically centered with exact gap */
        .st-key-dashboard_header div[data-testid="stHorizontalBlock"],
        .st-key-results_toolbar div[data-testid="stHorizontalBlock"]:has(iframe) {
            align-items: center !important;
            gap: 8px !important;
        }

        /* Item 2: Info flow rules button (44x44px, 10px radius) */
        .st-key-dashboard_header div[class*="st-key-btn_info_rules"] {
            display: flex !important;
            align-items: center !important;
            height: 44px !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        .st-key-dashboard_header div[class*="st-key-btn_info_rules"] button {
            width: 44px !important;
            min-width: 44px !important;
            max-width: 44px !important;
            height: 44px !important;
            min-height: 44px !important;
            max-height: 44px !important;
            border-radius: 10px !important;
            padding: 0 !important;
            margin: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            box-sizing: border-box !important;
        }

        /* Item 2: Mode Segmented Control (44px height, 10px radius) */
        .st-key-dashboard_header div[class*="st-key-global_mode_selector"] {
            display: flex !important;
            align-items: center !important;
            height: 44px !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        .st-key-dashboard_header div[class*="st-key-global_mode_selector"] div[role="radiogroup"] {
            height: 44px !important;
            min-height: 44px !important;
            max-height: 44px !important;
            border-radius: 10px !important;
            display: flex !important;
            align-items: center !important;
            box-sizing: border-box !important;
        }
        .st-key-dashboard_header div[class*="st-key-global_mode_selector"] div[role="radiogroup"] button,
        .st-key-dashboard_header div[class*="st-key-global_mode_selector"] div[role="radiogroup"] label {
            height: 44px !important;
            min-height: 44px !important;
            max-height: 44px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            box-sizing: border-box !important;
        }

        /* Precisely scope copy control row inside column copy_c so outer row is untouched and inner buttons align flush with right edge */
        .st-key-results_toolbar div[data-testid="stColumn"] div[data-testid="stHorizontalBlock"]:has(iframe) {
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: nowrap !important;
            align-items: center !important;
            justify-content: flex-end !important;
            gap: 8px !important;
            width: 100% !important;
            min-width: 308px !important;
        }
        .st-key-results_toolbar div[data-testid="stColumn"] div[data-testid="stHorizontalBlock"]:has(iframe) > div[data-testid="stColumn"] {
            display: block !important;
            padding: 0 !important;
            margin: 0 !important;
        }
        .st-key-results_toolbar div[data-testid="stColumn"] div[data-testid="stHorizontalBlock"]:has(iframe) > div[data-testid="stColumn"]:has(iframe) {
            flex: 0 0 180px !important;
            width: 180px !important;
            min-width: 180px !important;
            max-width: 180px !important;
        }
        .st-key-results_toolbar div[data-testid="stColumn"] div[data-testid="stHorizontalBlock"]:has(iframe) > div[data-testid="stColumn"]:has(div[data-testid="stPopover"]) {
            flex: 0 0 120px !important;
            width: 120px !important;
            min-width: 120px !important;
            max-width: 120px !important;
        }
        .st-key-results_toolbar div[data-testid="stPopover"] {
            margin: 0 !important;
            padding: 0 !important;
            height: 44px !important;
            width: 120px !important;
            min-width: 120px !important;
            max-width: 120px !important;
            display: flex !important;
            align-items: center !important;
        }
        .st-key-results_toolbar div[data-testid="stPopover"] > button,
        .st-key-results_toolbar div[data-testid="stPopover"] button {
            height: 44px !important;
            min-height: 44px !important;
            max-height: 44px !important;
            width: 120px !important;
            min-width: 120px !important;
            max-width: 120px !important;
            padding: 0 14px !important;
            font-size: 13px !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            box-sizing: border-box !important;
            white-space: nowrap !important;
            margin: 0 !important;
        }
        .st-key-results_toolbar div[data-testid="stHorizontalBlock"] iframe {
            height: 44px !important;
            min-height: 44px !important;
            max-height: 44px !important;
            width: 180px !important;
            min-width: 180px !important;
            max-width: 180px !important;
            margin: 0 !important;
            padding: 0 !important;
            border: none !important;
            display: block !important;
            border-radius: 8px !important;
            box-sizing: border-box !important;
        }

        .st-key-status_cards button p {
            margin: 0 !important;
            padding: 0 !important;
        }

        .st-key-filter_controls > div[data-testid="stVerticalBlock"] {
            gap: 0 !important;
        }
        .st-key-filter_controls input,
        .st-key-filter_controls button,
        .st-key-filter_controls div[data-baseweb="select"] > div {
            min-height: 44px !important;
            height: 44px !important;
            box-sizing: border-box !important;
        }
        .st-key-filter_controls div[data-testid="stCheckbox"] label {
            min-height: 44px !important;
            display: flex !important;
            align-items: center !important;
            white-space: nowrap !important;
        }
        .st-key-filter_controls label,
        .st-key-filter_controls button,
        .st-key-results_toolbar button {
            white-space: nowrap !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

    df: pd.DataFrame | None = None
    load_err: str | None = None
    try:
        df = cached_load_pool_csv(args.csv, _csv_cache_fingerprint(args.csv))
    except Exception as exc:
        load_err = str(exc)

    with st.container(key="dashboard_shell"):
        with st.container(key="dashboard_header"):
            _render_header_bar(df, load_err)

        if df is None:
            st.error(f"Could not load breakout pool data: {load_err}")
            return

        mode = st.session_state.get("global_mode_selector", "IBD Review")
        if mode == "C Rank Reference":
            _render_c_rank_reference_view(df)
        else:
            _render_ibd_review_view(df)


def _render_header_bar(df: pd.DataFrame | None, load_err: str | None) -> None:
    col_l, col_r = st.columns([3, 1.5])
    with col_l:
        badge_html = (
            '<span style="background-color:#e8f5e9; color:#2e7d32; padding:3px 8px; border-radius:4px; font-size:12px; font-weight:600; margin-left:8px;">Data Ready</span>'
            if df is not None
            else '<span style="background-color:#ffebee; color:#c62828; padding:3px 8px; border-radius:4px; font-size:12px; font-weight:600; margin-left:8px;">Schema / Data Error</span>'
        )
        freshness = build_snapshot_freshness(_get_snapshot_date(df) if df is not None else "N/A")
        snapshot_html = freshness["header_html"]
        total_pool = len(df) if df is not None else 0
        active_signals = int(df["signal"].sum()) if df is not None and "signal" in df.columns else 0
        st.markdown(
            f'<h3 style="margin:0; display:inline-block; font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',Roboto,sans-serif;">Breakout Pool {badge_html}</h3>'
            f'<div style="font-size:13px; color:#8899a6; margin-top:2px;">{snapshot_html} · <b>{total_pool}</b> Total Pool · <b>{active_signals}</b> Active Signals</div>',
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
    st.markdown('<hr style="margin: 4px 0 8px 0; border: none; border-top: 1px solid #e0e0e0;" />', unsafe_allow_html=True)


@st.dialog("IBD Breakout Review Flow & Rules", width="large")
def _render_flow_rules_dialog() -> None:
    legend_lines = [f"- {meta.get('dot', '⚪')} **{meta.get('label', k)}**：{meta.get('tooltip', '')}" for k, meta in STATUS_META.items()]
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


ENTRY_VOL_ENABLED_STATUSES: set[str] = {"ACTIONABLE", "BELOW_TRIGGER", "EXTENDED"}


def _render_ibd_review_view(df: pd.DataFrame) -> None:
    active_df = df[df["signal"] == True].copy() if "signal" in df.columns else df.copy()
    active_signals_count = len(active_df)

    route_val = st.session_state.get("ibd_filter_route", "All")
    status_val = st.session_state.get("ibd_filter_status", "All")
    if status_val != "UNCONFIRMED" and st.session_state.get("ibd_near_trigger_only", False):
        st.session_state["ibd_near_trigger_only"] = False
    dist_min_val = st.session_state.get("ibd_filter_dist_min", "")
    dist_max_val = st.session_state.get("ibd_filter_dist_max", "")
    entry_vol_min_val = st.session_state.get("ibd_filter_entry_vol_min", "")
    weekly_vol_min_val = st.session_state.get("ibd_filter_weekly_vol_min", "")

    if route_val != "All":
        route_df = active_df[active_df["ibd_candidate_rule"] == route_val]
    else:
        route_df = active_df

    status_counts = build_entry_status_counts(route_df)

    with st.container(key="review_queue"):
        _render_status_queue(status_counts, len(route_df), status_val)

    with st.container(key="filters"):
        _render_filter_bar(df, status_val, status_counts)

    filtered_df = route_df.copy()
    if status_val != "All":
        filtered_df = filtered_df[filtered_df["ibd_entry_status"] == status_val]

    if dist_min_val != "":
        try:
            val = float(dist_min_val)
            filtered_df = filtered_df[pd.to_numeric(filtered_df["current_vs_ibd_candidate_pct"], errors="coerce") >= val]
        except ValueError:
            pass

    if dist_max_val != "":
        try:
            val = float(dist_max_val)
            filtered_df = filtered_df[pd.to_numeric(filtered_df["current_vs_ibd_candidate_pct"], errors="coerce") <= val]
        except ValueError:
            pass

    if status_val in ENTRY_VOL_ENABLED_STATUSES and entry_vol_min_val != "":
        try:
            val = float(entry_vol_min_val)
            filtered_df = filtered_df[pd.to_numeric(filtered_df["ibd_entry_volume_ratio"], errors="coerce") >= val]
        except ValueError:
            pass

    if weekly_vol_min_val != "":
        try:
            val = float(weekly_vol_min_val)
            filtered_df = filtered_df[pd.to_numeric(filtered_df["volume_ratio"], errors="coerce") >= val]
        except ValueError:
            pass

    filtered_df = filter_unconfirmed_near_trigger(
        filtered_df, status_val, st.session_state.get("ibd_near_trigger_only", False)
    )

    filtered_df = apply_default_review_order(filtered_df)

    with st.container(key="results_toolbar"):
        sum_c, copy_c = st.columns([2.4, 1.6], vertical_alignment="center")
        with sum_c:
            st.markdown(
                f'<div style="font-size:14px; font-weight:600; color:#c5ceda;">{len(filtered_df)} results · Sorted by Entry Status → C Rank</div>',
                unsafe_allow_html=True,
            )
        with copy_c:
            _render_copy_codes_control(filtered_df["code"].tolist(), key_prefix="ibd_review")

    from dashboard.field_config import get_default_table_columns
    columns = get_default_table_columns()

    with st.container(key="selected_row"):
        detail_container = st.empty()
    with st.container(key="results_grid"):
        selected_code = render_table(filtered_df, columns, height=480)

    with detail_container.container():
        _render_selected_row_detail(filtered_df, selected_code)

    st.markdown("---")
    _download_current_rows(df.loc[filtered_df.index] if not filtered_df.empty else pd.DataFrame(), "ibd_review_filtered.csv")


def _render_status_queue(status_counts: dict[str, int], route_total: int, current_status: str) -> None:
    c_title, c_btn = st.columns([7.5, 1.5], vertical_alignment="center")
    with c_title:
        st.markdown("##### Review Queue")
    with c_btn:
        is_all_active = current_status == "All"
        all_prefix = "✓ " if is_all_active else ""
        if st.button(f"{all_prefix}All Signals ({route_total})", key="btn_all_signals", use_container_width=True, type="primary" if is_all_active else "secondary"):
            st.session_state["ibd_filter_status"] = "All"
            st.session_state["ibd_filter_entry_vol_min"] = ""
            st.session_state["ibd_near_trigger_only"] = False
            st.rerun()

    with st.container(key="status_cards"):
        cols = st.columns(4)
        statuses = ["ACTIONABLE", "UNCONFIRMED", "BELOW_TRIGGER", "EXTENDED"]
        sub_map = {
            "ACTIONABLE": "0%–5% above candidate",
            "UNCONFIRMED": f"{status_counts.get('unconfirmed_within_3pct', 0)} within +3% zone",
            "BELOW_TRIGGER": "＜ 0% below trigger",
            "EXTENDED": "＞ +5% chase limit",
        }
        for i, status_name in enumerate(statuses):
            with cols[i]:
                count = status_counts.get(status_name, 0)
                is_active = current_status == status_name
                prefix = "✓ " if is_active else ""
                display_name = status_name.replace("_", " ")
                meta = STATUS_META.get(status_name, {})
                dot = meta.get("dot", "⚪")
                tooltip = meta.get("tooltip", "")
                btn_label = f"{prefix}{dot} {display_name} · {count}\n{sub_map[status_name]}"
                if st.button(btn_label, key=f"btn_status_{status_name}", use_container_width=True, type="primary" if is_active else "secondary", help=tooltip):
                    if is_active:
                        st.session_state["ibd_filter_status"] = "All"
                        st.session_state["ibd_filter_entry_vol_min"] = ""
                        st.session_state["ibd_near_trigger_only"] = False
                    else:
                        st.session_state["ibd_filter_status"] = status_name
                        if status_name not in ENTRY_VOL_ENABLED_STATUSES:
                            st.session_state["ibd_filter_entry_vol_min"] = ""
                        if status_name != "UNCONFIRMED":
                            st.session_state["ibd_near_trigger_only"] = False
                    st.rerun()


def _render_filter_bar(df: pd.DataFrame, current_status: str, status_counts: dict[str, int]) -> None:
    st.markdown("##### Filters")
    controls = st.container(key="filter_controls")
    cols = controls.columns([1.6, 1.1, 1.1, 1.1, 1.1, 0.8], vertical_alignment="bottom")
    with cols[0]:
        routes = ["All"] + _unique_values(df, "ibd_candidate_rule")
        current_route = st.session_state.get("ibd_filter_route", "All")
        idx = routes.index(current_route) if current_route in routes else 0
        selected_route = st.selectbox("Route (Rule)", routes, index=idx, key="ibd_filter_route_input")
        if selected_route != current_route:
            st.session_state["ibd_filter_route"] = selected_route
            st.rerun()

    with cols[1]:
        val = st.text_input("Distance Min %", value=st.session_state.get("ibd_filter_dist_min", ""), placeholder="", key="ibd_filter_dist_min_input")
        if val != st.session_state.get("ibd_filter_dist_min", ""):
            st.session_state["ibd_filter_dist_min"] = val
            st.rerun()

    with cols[2]:
        val = st.text_input("Distance Max %", value=st.session_state.get("ibd_filter_dist_max", ""), placeholder="", key="ibd_filter_dist_max_input")
        if val != st.session_state.get("ibd_filter_dist_max", ""):
            st.session_state["ibd_filter_dist_max"] = val
            st.rerun()

    with cols[3]:
        if current_status == "UNCONFIRMED":
            near_count = status_counts.get("unconfirmed_within_3pct", 0)
            val = st.checkbox(
                f"Near Trigger ≤ +3% ({near_count})",
                value=st.session_state.get("ibd_near_trigger_only", False),
                key="ibd_near_trigger_only_widget",
            )
            if val != st.session_state.get("ibd_near_trigger_only", False):
                st.session_state["ibd_near_trigger_only"] = val
                st.rerun()
        else:
            is_disabled = current_status not in ENTRY_VOL_ENABLED_STATUSES
            placeholder_text = "N/A (Disabled)" if is_disabled else ""
            val_entry = "" if is_disabled else st.session_state.get("ibd_filter_entry_vol_min", "")
            val = st.text_input("Entry Vol Min (x)", value=val_entry, placeholder=placeholder_text, disabled=is_disabled, key="ibd_filter_entry_vol_min_input")
            if not is_disabled and val != st.session_state.get("ibd_filter_entry_vol_min", ""):
                st.session_state["ibd_filter_entry_vol_min"] = val
                st.rerun()

    with cols[4]:
        val = st.text_input("Weekly Vol Min (x)", value=st.session_state.get("ibd_filter_weekly_vol_min", ""), placeholder="", key="ibd_filter_weekly_vol_min_input")
        if val != st.session_state.get("ibd_filter_weekly_vol_min", ""):
            st.session_state["ibd_filter_weekly_vol_min"] = val
            st.rerun()

    with cols[5]:
        if st.button("Reset", use_container_width=True):
            st.session_state["ibd_filter_route"] = "All"
            st.session_state["ibd_filter_status"] = "All"
            st.session_state["ibd_filter_dist_min"] = ""
            st.session_state["ibd_filter_dist_max"] = ""
            st.session_state["ibd_filter_entry_vol_min"] = ""
            st.session_state["ibd_filter_weekly_vol_min"] = ""
            st.session_state["ibd_near_trigger_only"] = False
            st.rerun()


def _render_selected_row_detail(filtered_df: pd.DataFrame, selected_code: str | None) -> None:
    if filtered_df.empty:
        st.info("No matching records found with current filter criteria.")
        return

    row: pd.Series | None = None
    if selected_code is not None and selected_code in filtered_df["code"].values:
        row = filtered_df[filtered_df["code"] == selected_code].iloc[0]
    else:
        row = filtered_df.iloc[0]

    code = html.escape(str(row.get("code", "N/A")))
    cand_price = html.escape(_format_number(row.get("ibd_candidate_price"), ""))
    cand_rule = html.escape(str(row.get("ibd_candidate_rule", "N/A")))
    dist_pct = html.escape(_format_number(row.get("current_vs_ibd_candidate_pct"), "%"))
    latest_close = html.escape(_format_number(row.get("latest_close"), ""))
    status_name = html.escape(str(row.get("ibd_entry_status", "N/A")))
    vol_or_reject = html.escape(str(row.get("ibd_entry_vol_or_reject", "N/A")))
    rank_c = html.escape(str(row.get("rank_C_continuous", "N/A")))
    c_cont = html.escape(_format_number(row.get("C_continuous"), ""))

    status_color = STATUS_META.get(str(row.get("ibd_entry_status", "N/A")), {}).get("color", "#4caf50" if status_name == "ACTIONABLE" else "#f2f5f9")

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
        </style>
        <div style="background:#141a22; border:1px solid #303947; color:#f2f5f9; border-radius:6px; padding:8px 12px; margin-bottom:8px;">
            <div style="display:flex; justify-content:space-between; align-items:center; text-align:center;">
                <div style="flex:1; border-right:1px solid #303947; text-align:left;">
                    <div class="code-hover-wrapper">
                        <details class="code-detail" data-selected-code="{code}">
                            <summary class="code-hover-trigger"
                                     title="Hover, focus, or click to view details">{code} ▾</summary>
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
                </div>
                <div style="flex:1.5; border-right:1px solid #303947;">
                    <div style="font-size:11px; color:#8899a6; text-transform:uppercase;">Candidate Price</div>
                    <div style="font-size:15px; font-weight:700; color:#f2f5f9;">{cand_price} <span style="font-size:11px; font-weight:normal; color:#a0aec0;">({cand_rule})</span></div>
                </div>
                <div style="flex:1.5; border-right:1px solid #303947;">
                    <div style="font-size:11px; color:#8899a6; text-transform:uppercase;">Current vs Candidate</div>
                    <div style="font-size:15px; font-weight:700; color:#f2f5f9;">{dist_pct} <span style="font-size:11px; font-weight:normal; color:#a0aec0;">(Close: {latest_close})</span></div>
                </div>
                <div style="flex:1.5; border-right:1px solid #303947;">
                    <div style="font-size:11px; color:#8899a6; text-transform:uppercase;">Entry Status</div>
                    <div style="font-size:15px; font-weight:700; color:{status_color};">{status_name.replace('_', ' ')} <span style="font-size:11px; font-weight:normal; color:#a0aec0;">({vol_or_reject})</span></div>
                </div>
                <div style="flex:1.5;">
                    <div style="font-size:11px; color:#8899a6; text-transform:uppercase;">C Rank & Continuous</div>
                    <div style="font-size:15px; font-weight:700; color:#f2f5f9;">#{rank_c} <span style="font-size:11px; font-weight:normal; color:#a0aec0;">({c_cont})</span></div>
                </div>
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
        selected_code = render_table(ranked, [column for column in columns if column in ranked.columns], height=520)

    with detail_container.container():
        _render_selected_row_detail(ranked, selected_code)

    st.markdown("---")
    _download_current_rows(ranked, "c_rank_reference.csv")


def _render_copy_codes_control(codes: list[str], key_prefix: str = "") -> None:
    codes_str = ", ".join([str(c) for c in codes if pd.notna(c) and str(c).strip()])
    n = len(codes)
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
        }}
    </style>
    </head>
    <body>
    <div class="copy-wrapper">
        <button id="copyBtn_{key_prefix}" class="copy-btn">
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
                    btn.innerText = 'Copy failed · use Manual';
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
    col_a, col_b = st.columns([180, 120])
    with col_a:
        st.components.v1.html(html_code, height=44, scrolling=False)
    with col_b:
        with st.popover("Manual", use_container_width=True):
            st.caption("If direct copy is blocked by your browser, copy directly from below:")
            st.code(codes_str, language="text")


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
