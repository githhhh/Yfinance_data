from __future__ import annotations

import argparse
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
    FilterSpec,
    apply_c_rank_mode,
    apply_default_review_order,
    apply_filters,
    build_chart_data,
    build_entry_status_counts,
    build_kpis,
    load_pool_csv,
)
from dashboard.field_config import (
    get_all_table_columns,
    get_column_view_fields,
    get_field_label,
    get_filter_funnel_groups,
)
from dashboard.table_view import render_table

st.set_page_config(page_title="Breakout Pool Dashboard", layout="wide", initial_sidebar_state="collapsed")

FUNNEL_ORDER = [
    "Route",
    "Entry Status",
    "Optional Quality Filters",
]


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
        [data-testid="stHeader"] {
            display: none !important;
        }
        [data-testid="stSidebar"] {
            display: none !important;
        }
        .block-container {
            padding-top: 1.2rem !important;
            padding-bottom: 2rem !important;
            max-width: 98% !important;
        }
        div[data-testid="stVerticalBlock"] > div {
            padding-bottom: 0.25rem !important;
        }
        .status-card {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 12px;
            background-color: #f8f9fa;
            text-align: center;
        }
        .status-card-active {
            border: 2px solid #1f77b4;
            background-color: #e3f2fd;
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
    col_l, col_r = st.columns([3, 2])
    with col_l:
        badge_html = (
            '<span style="background-color:#e8f5e9; color:#2e7d32; padding:3px 8px; border-radius:4px; font-size:12px; font-weight:600; margin-left:8px;">Data Ready</span>'
            if df is not None
            else '<span style="background-color:#ffebee; color:#c62828; padding:3px 8px; border-radius:4px; font-size:12px; font-weight:600; margin-left:8px;">Schema / Data Error</span>'
        )
        snapshot_date = _get_snapshot_date(df) if df is not None else "N/A"
        total_pool = len(df) if df is not None else 0
        active_signals = int(df["signal"].sum()) if df is not None and "signal" in df.columns else 0
        st.markdown(
            f'<h3 style="margin:0; display:inline-block; font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',Roboto,sans-serif;">Breakout Follow Pool {badge_html}</h3>'
            f'<div style="font-size:13px; color:#555; margin-top:4px;">Snapshot <b>{snapshot_date}</b> · <b>{total_pool}</b> Total Pool · <b>{active_signals}</b> Active Signals</div>',
            unsafe_allow_html=True,
        )

    with col_r:
        c1, c2 = st.columns([1, 2.5])
        with c1:
            if st.button("ⓘ Flow & Rules", use_container_width=True):
                _render_flow_rules_dialog()
        with c2:
            mode = st.segmented_control(
                "Mode Selector",
                ["IBD Review", "C Rank Reference"],
                default=st.session_state.get("global_mode_selector", "IBD Review") or "IBD Review",
                key="global_mode_selector",
                label_visibility="collapsed",
            )
            if mode is None:
                mode = "IBD Review"
    st.divider()


@st.dialog("IBD Breakout Review Flow & Rules", width="large")
def _render_flow_rules_dialog() -> None:
    st.markdown(
        r"""
        #### 1. IBD Review Workflow (`signal=True`)
        - **Step 1: Route Filtering (`Route`)**: Select the candidate breakout/pullback rule (e.g., `ceiling_breakout`, `ma10_touch_confirm`, etc.). Status card totals dynamically update to reflect the count of active signals under the selected route.
        - **Step 2: Status Triage (`Status Queue`)**: Click any status card to slice the queue.
          - `ACTIONABLE`: Price $\le +5.0\%$ from trigger and daily volume condition met.
          - `UNCONFIRMED`: Price within breakout zone or pullback zone but volume unconfirmed (subtitle tracks items within $+3.0\%$).
          - `BELOW_TRIGGER`: Price fell below trigger price (< $0.0\%$).
          - `EXTENDED`: Price extended beyond $+5.0\%$ chase limit.
        - **Step 3: Quality Thresholds (`One-line Filter Bar`)**: Apply strict `AND` intersection filtering on Distance Min/Max %, Entry Volume Ratio Min, and Weekly Volume Ratio Min.
          - *Note*: `Entry Vol Min` is automatically disabled and cleared when `UNCONFIRMED` or `All` status is selected (enabled for `ACTIONABLE`, `BELOW_TRIGGER`, and `EXTENDED`).
        - **Step 4: Selected Row Detail**: Click any row in the Decision Table to immediately inspect current price, candidate rule, volume ratio, and `C Rank / C Continuous` scores in the top detail bar.
        - **Step 5: Code Export**: Use the one-click clipboard copy button or popover text to export filtered codes.

        ---
        #### 2. C Rank Reference Mode
        - Isolates and displays all actionable or unconfirmed pool records where `signal=True`, strictly sorted by **`rank_C_continuous` ascending**.
        - **Top N Selector**: Slice the top candidates (`Top 10`, `Top 20`, `Top 30`, `Top 50`, or `All rows`).
        """
    )


ENTRY_VOL_ENABLED_STATUSES: set[str] = {"ACTIONABLE", "BELOW_TRIGGER", "EXTENDED"}


def _render_ibd_review_view(df: pd.DataFrame) -> None:
    active_df = df[df["signal"] == True].copy() if "signal" in df.columns else df.copy()
    active_signals_count = len(active_df)

    route_val = st.session_state.get("ibd_filter_route", "All")
    status_val = st.session_state.get("ibd_filter_status", "All")
    dist_min_val = st.session_state.get("ibd_filter_dist_min", "")
    dist_max_val = st.session_state.get("ibd_filter_dist_max", "")
    entry_vol_min_val = st.session_state.get("ibd_filter_entry_vol_min", "")
    weekly_vol_min_val = st.session_state.get("ibd_filter_weekly_vol_min", "")

    if route_val != "All":
        route_df = active_df[active_df["ibd_candidate_rule"] == route_val]
    else:
        route_df = active_df

    status_counts = build_entry_status_counts(route_df)

    _render_status_queue(status_counts, len(route_df), status_val)

    _render_filter_bar(df, status_val)

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

    filtered_df = apply_default_review_order(filtered_df)

    detail_container = st.empty()

    col_view_c, summary_c, copy_c = st.columns([1.5, 2.5, 2])
    with col_view_c:
        column_view = st.selectbox(
            "Column View",
            ["IBD Decision", "All Fields", "Signal", "IBD Entry", "Volume/Pullback", "Reference"],
            index=0,
            key="ibd_column_view_select",
        )
    with summary_c:
        st.markdown(
            f'<div style="margin-top:28px; font-size:14px;">📊 <b>Filtered Rows: {len(filtered_df)} / {active_signals_count}</b> Active Signals · Sorted by Status → C Rank</div>',
            unsafe_allow_html=True,
        )
    with copy_c:
        st.markdown('<div style="margin-top:20px;"></div>', unsafe_allow_html=True)
        _render_copy_codes_control(filtered_df["code"].tolist(), key_prefix="ibd_review")

    columns = get_column_view_fields(column_view)
    selected_code = render_table(filtered_df, columns, height=620)

    with detail_container.container():
        _render_selected_row_detail(filtered_df, selected_code)

    st.markdown("---")
    _download_current_rows(df.loc[filtered_df.index] if not filtered_df.empty else pd.DataFrame(), "ibd_review_filtered.csv")


def _render_status_queue(status_counts: dict[str, int], route_total: int, current_status: str) -> None:
    c_title, c_btn = st.columns([3, 1])
    with c_title:
        st.markdown("##### 1. Status Queue Triage (`signal=True` + `Route`)")
    with c_btn:
        if st.button(f"⚡ All Signals ({route_total})", use_container_width=True, type="primary" if current_status == "All" else "secondary"):
            st.session_state["ibd_filter_status"] = "All"
            st.session_state["ibd_filter_entry_vol_min"] = ""
            st.rerun()

    cols = st.columns(4)
    statuses = ["ACTIONABLE", "UNCONFIRMED", "BELOW_TRIGGER", "EXTENDED"]
    for i, status_name in enumerate(statuses):
        with cols[i]:
            count = status_counts.get(status_name, 0)
            is_active = current_status == status_name
            bg_color = "#e3f2fd" if is_active else "#f8f9fa"
            border_color = "#1f77b4" if is_active else "#e0e0e0"

            subtitle = "&nbsp;"
            if status_name == "UNCONFIRMED":
                subtitle = f"{status_counts.get('unconfirmed_within_3pct', 0)} within +3%"

            st.markdown(
                f"""
                <div style="border: 2px solid {border_color}; border-radius: 8px; padding: 10px; background-color: {bg_color}; text-align: center; margin-bottom: 8px;">
                    <div style="font-size: 13px; font-weight: 700; color: #555;">{status_name}</div>
                    <div style="font-size: 24px; font-weight: 800; color: #1f77b4; margin: 4px 0;">{count}</div>
                    <div style="font-size: 11px; color: #666;">{subtitle}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            btn_label = f"✓ {status_name} Selected" if is_active else f"Select {status_name}"
            if st.button(btn_label, key=f"btn_status_{status_name}", use_container_width=True):
                if is_active:
                    st.session_state["ibd_filter_status"] = "All"
                    st.session_state["ibd_filter_entry_vol_min"] = ""
                else:
                    st.session_state["ibd_filter_status"] = status_name
                    if status_name not in ENTRY_VOL_ENABLED_STATUSES:
                        st.session_state["ibd_filter_entry_vol_min"] = ""
                st.rerun()
    st.markdown("<div style='margin-bottom:8px;'></div>", unsafe_allow_html=True)


def _render_filter_bar(df: pd.DataFrame, current_status: str) -> None:
    st.markdown("##### 2. One-line Quality Filtering (`AND` Combination)")
    cols = st.columns([1.6, 1.1, 1.1, 1.1, 1.1, 0.8])
    with cols[0]:
        routes = ["All"] + _unique_values(df, "ibd_candidate_rule")
        current_route = st.session_state.get("ibd_filter_route", "All")
        idx = routes.index(current_route) if current_route in routes else 0
        selected_route = st.selectbox("Route (Rule)", routes, index=idx, key="ibd_filter_route_input")
        if selected_route != current_route:
            st.session_state["ibd_filter_route"] = selected_route
            st.rerun()

    with cols[1]:
        val = st.text_input("Distance Min %", value=st.session_state.get("ibd_filter_dist_min", ""), placeholder="-20.0", key="ibd_filter_dist_min_input")
        if val != st.session_state.get("ibd_filter_dist_min", ""):
            st.session_state["ibd_filter_dist_min"] = val
            st.rerun()

    with cols[2]:
        val = st.text_input("Distance Max %", value=st.session_state.get("ibd_filter_dist_max", ""), placeholder="20.0", key="ibd_filter_dist_max_input")
        if val != st.session_state.get("ibd_filter_dist_max", ""):
            st.session_state["ibd_filter_dist_max"] = val
            st.rerun()

    with cols[3]:
        is_disabled = current_status not in ENTRY_VOL_ENABLED_STATUSES
        placeholder_text = "N/A (Disabled)" if is_disabled else "1.5"
        val_entry = "" if is_disabled else st.session_state.get("ibd_filter_entry_vol_min", "")
        val = st.text_input("Entry Vol Min (x)", value=val_entry, placeholder=placeholder_text, disabled=is_disabled, key="ibd_filter_entry_vol_min_input")
        if not is_disabled and val != st.session_state.get("ibd_filter_entry_vol_min", ""):
            st.session_state["ibd_filter_entry_vol_min"] = val
            st.rerun()

    with cols[4]:
        val = st.text_input("Weekly Vol Min (x)", value=st.session_state.get("ibd_filter_weekly_vol_min", ""), placeholder="1.2", key="ibd_filter_weekly_vol_min_input")
        if val != st.session_state.get("ibd_filter_weekly_vol_min", ""):
            st.session_state["ibd_filter_weekly_vol_min"] = val
            st.rerun()

    with cols[5]:
        st.markdown('<div style="margin-top:28px;"></div>', unsafe_allow_html=True)
        if st.button("Reset", use_container_width=True):
            st.session_state["ibd_filter_route"] = "All"
            st.session_state["ibd_filter_status"] = "All"
            st.session_state["ibd_filter_dist_min"] = ""
            st.session_state["ibd_filter_dist_max"] = ""
            st.session_state["ibd_filter_entry_vol_min"] = ""
            st.session_state["ibd_filter_weekly_vol_min"] = ""
            st.rerun()
    st.markdown("<div style='margin-bottom:8px;'></div>", unsafe_allow_html=True)


def _render_selected_row_detail(filtered_df: pd.DataFrame, selected_code: str | None) -> None:
    if filtered_df.empty:
        st.info("No matching records found with current filter criteria.")
        return

    row: pd.Series | None = None
    if selected_code is not None and selected_code in filtered_df["code"].values:
        row = filtered_df[filtered_df["code"] == selected_code].iloc[0]
    else:
        row = filtered_df.iloc[0]

    code = str(row.get("code", "N/A"))
    sector = str(row.get("sector", "N/A"))
    industry = str(row.get("industry", "N/A"))
    signal_src = str(row.get("signal_source", "N/A"))
    cand_price = _format_number(row.get("ibd_candidate_price"), "")
    cand_rule = str(row.get("ibd_candidate_rule", "N/A"))
    dist_pct = _format_number(row.get("current_vs_ibd_candidate_pct"), "%")
    latest_close = _format_number(row.get("latest_close"), "")
    status_name = str(row.get("ibd_entry_status", "N/A"))
    vol_or_reject = str(row.get("ibd_entry_vol_or_reject", "N/A"))
    rank_c = str(row.get("rank_C_continuous", "N/A"))
    c_cont = _format_number(row.get("C_continuous"), "")

    st.markdown(
        f"""
        <div style="background-color:#f8f9fa; border:1px solid #dee2e6; border-radius:8px; padding:12px 16px; margin-bottom:12px;">
            <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #e9ecef; padding-bottom:8px; margin-bottom:10px;">
                <div>
                    <span style="font-size:18px; font-weight:800; color:#1f77b4;">{code}</span>
                    <span style="font-size:14px; color:#495057; margin-left:12px;"><b>Sector/Industry:</b> {sector} / {industry}</span>
                </div>
                <div style="font-size:13px; color:#6c757d;"><b>Signal Source:</b> {signal_src}</div>
            </div>
            <div style="display:flex; justify-content:space-between; text-align:center;">
                <div style="flex:1; border-right:1px solid #e9ecef;">
                    <div style="font-size:11px; color:#6c757d; text-transform:uppercase;">Candidate Price</div>
                    <div style="font-size:16px; font-weight:700; color:#212529;">{cand_price}</div>
                    <div style="font-size:11px; color:#495057;">Route: {cand_rule}</div>
                </div>
                <div style="flex:1; border-right:1px solid #e9ecef;">
                    <div style="font-size:11px; color:#6c757d; text-transform:uppercase;">Current vs Candidate</div>
                    <div style="font-size:16px; font-weight:700; color:#212529;">{dist_pct}</div>
                    <div style="font-size:11px; color:#495057;">Close: {latest_close}</div>
                </div>
                <div style="flex:1; border-right:1px solid #e9ecef;">
                    <div style="font-size:11px; color:#6c757d; text-transform:uppercase;">Entry Status</div>
                    <div style="font-size:16px; font-weight:700; color:#2e7d32;">{status_name}</div>
                    <div style="font-size:11px; color:#495057;">Vol / Reason: {vol_or_reject}</div>
                </div>
                <div style="flex:1;">
                    <div style="font-size:11px; color:#6c757d; text-transform:uppercase;">C Rank & Continuous</div>
                    <div style="font-size:16px; font-weight:700; color:#1f77b4;">#{rank_c}</div>
                    <div style="font-size:11px; color:#495057;">Continuous: {c_cont}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_c_rank_reference_view(df: pd.DataFrame) -> None:
    st.markdown("##### C Rank Reference View (`signal=True` · Sorted by `rank_C_continuous` asc)")
    
    with st.expander("ℹ️ C Rank Selection & Reference Rules", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                "\n".join(
                    [
                        "**Fixed Mode Rules**",
                        "- signal=True",
                        "- rank_C_continuous asc",
                        "- Top N selector only",
                        "- Custom filters ignored",
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

    col_limit, col_summary, col_copy = st.columns([1.5, 2.5, 2])
    with col_limit:
        limit_label = st.selectbox("Top N Slice", ["All rows", "Top 10", "Top 20", "Top 30", "Top 50"], index=0, key="c_rank_top_n_select")
        limit = None if limit_label == "All rows" else int(limit_label.split()[1])
    
    ranked = apply_c_rank_mode(df, limit=limit)
    
    with col_summary:
        st.markdown(
            f'<div style="margin-top:28px; font-size:14px;">📊 <b>Showing: {len(ranked)} / {len(df)}</b> Pool Records · Reference Only</div>',
            unsafe_allow_html=True,
        )
    with col_copy:
        st.markdown('<div style="margin-top:20px;"></div>', unsafe_allow_html=True)
        _render_copy_codes_control(ranked["code"].tolist(), key_prefix="c_rank_ref")

    detail_container = st.empty()

    leading_columns = ["code", "rank_C_continuous", "C_continuous", "ibd_entry_status", "current_vs_ibd_candidate_pct", "ibd_candidate_rule", "volume_ratio", "latest_close"]
    columns = leading_columns + [column for column in get_all_table_columns() if column not in set(leading_columns)]
    selected_code = render_table(ranked, [column for column in columns if column in ranked.columns], height=720)

    with detail_container.container():
        _render_selected_row_detail(ranked, selected_code)

    st.markdown("---")
    _download_current_rows(ranked, "c_rank_reference.csv")


def _render_copy_codes_control(codes: list[str], key_prefix: str = "") -> None:
    codes_str = ", ".join([str(c) for c in codes if pd.notna(c) and str(c).strip()])
    n = len(codes)
    html_code = f"""
    <div style="display:flex; align-items:center; justify-content:flex-end; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        <button id="copyBtn_{key_prefix}" style="background:#1f77b4; color:#fff; border:none; border-radius:4px; padding:6px 14px; font-size:13px; font-weight:600; cursor:pointer; box-shadow:0 1px 3px rgba(0,0,0,0.12); transition: background 0.2s;">
            📋 Copy {n} Codes
        </button>
        <span id="copyMsg_{key_prefix}" style="margin-left:8px; font-size:12px; color:#2e7d32; font-weight:600; display:none;">✓ Copied!</span>
    </div>
    <script>
        const btn = document.getElementById('copyBtn_{key_prefix}');
        const msg = document.getElementById('copyMsg_{key_prefix}');
        const textToCopy = {json.dumps(codes_str)};
        if (btn) {{
            btn.addEventListener('click', () => {{
                const ta = document.createElement('textarea');
                ta.value = textToCopy;
                ta.style.position = 'fixed';
                ta.style.left = '-9999px';
                document.body.appendChild(ta);
                ta.focus();
                ta.select();
                try {{
                    document.execCommand('copy');
                }} catch (e) {{
                    if (navigator.clipboard) {{
                        navigator.clipboard.writeText(textToCopy);
                    }}
                }}
                document.body.removeChild(ta);
                if (msg) msg.style.display = 'inline';
                btn.style.background = '#2e7d32';
                btn.innerText = '✓ Copied {n} Codes';
                setTimeout(() => {{
                    if (msg) msg.style.display = 'none';
                    btn.style.background = '#1f77b4';
                    btn.innerText = '📋 Copy {n} Codes';
                }}, 2000);
            }});
        }}
    </script>
    """
    col_a, col_b = st.columns([1.5, 1])
    with col_a:
        st.components.v1.html(html_code, height=36)
    with col_b:
        with st.popover(f"📋 Manual Copy ({n})"):
            st.caption("If direct copy is blocked by your browser, copy directly from below:")
            st.code(codes_str, language="text")


def _funnel_filters(df: pd.DataFrame) -> tuple[dict[str, list[FilterSpec]], pd.DataFrame]:
    groups = get_filter_funnel_groups()
    filters_by_group: dict[str, list[FilterSpec]] = {group: [] for group in FUNNEL_ORDER}
    route_df = df[df["signal"] == True].copy() if "signal" in df.columns else df.copy()
    filters_by_group["Route"].append(FilterSpec("signal", "is true", label="Signal"))
    return filters_by_group, route_df


def _flatten_filters(filters_by_group: dict[str, list[FilterSpec]]) -> list[FilterSpec]:
    return [spec for specs in filters_by_group.values() for spec in specs]


def _funnel_tab_labels(filters_by_group: dict[str, list[FilterSpec]] | None = None) -> list[str]:
    return [str(group) for group in get_filter_funnel_groups()]


def _render_current_filter_summary(filters_by_group: dict[str, list[FilterSpec]], filtered_count: int, total_count: int) -> None:
    active_groups = {group: filters for group, filters in filters_by_group.items() if filters}
    if not active_groups:
        st.markdown(f"📊 **Filtered Rows: `{filtered_count}/{total_count}`** (All records)")
        return
    summary_parts = []
    for group, filters in active_groups.items():
        short_group = group.split(" & ")[0].split(" Rule")[0].split(" Vol")[0]
        conditions_str = " & ".join(_describe_filter_condition(spec) for spec in filters)
        summary_parts.append(f"**{short_group}**: `{conditions_str}`")
    st.markdown(f"📊 **Filtered Rows: `{filtered_count}/{total_count}`** ｜ " + " → ".join(summary_parts))


def _describe_filter_condition(spec: FilterSpec) -> str:
    def _format_val(v) -> str:
        try:
            val = float(v)
            return f"{val:.2f}"
        except (ValueError, TypeError):
            return str(v)

    label = get_field_label(spec.field)
    operator = spec.operator.lower()
    if operator == "is true":
        return f"{label}: True"
    if operator == "is false":
        return f"{label}: False"
    if operator == "equals":
        return f"{label}: {_format_val(spec.value)}"
    if operator == "in":
        values = ", ".join(_format_val(value) for value in spec.value)
        return f"{label}: {values}"
    if operator == "between":
        if spec.field == "pullback_pct":
            val1 = _format_val(abs(float(spec.value2)))
            val2 = _format_val(abs(float(spec.value)))
            return f"{label} magnitude: {val1} to {val2}"
        return f"{label}: {_format_val(spec.value)} to {_format_val(spec.value2)}"
    return f"{label} {spec.operator} {_format_val(spec.value)}"


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
            return str(valid.iloc[0])
    return "N/A"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    default_csv = Path(__file__).resolve().parents[1] / "us" / "breakout_follow_pool.csv"
    parser.add_argument("--csv", default=str(default_csv))
    args, _ = parser.parse_known_args()
    return args


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
