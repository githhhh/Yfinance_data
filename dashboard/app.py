from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.data_utils import (
    FilterSpec,
    apply_c_rank_mode,
    apply_filters,
    build_chart_data,
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


st.set_page_config(page_title="Breakout Pool Dashboard", layout="wide", initial_sidebar_state="auto")

FUNNEL_ORDER = [
    "Route",
    "Entry Confirmation & Strength",
    "Weekly Volume & Price",
    "Structure",
    "Grouping",
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

    try:
        df = cached_load_pool_csv(args.csv, _csv_cache_fingerprint(args.csv))
    except Exception as exc:
        st.error(f"Could not load CSV: {exc}")
        return

    with st.sidebar:
        st.subheader("🎯 Mode Selector")
        mode = st.radio("Mode", ["Custom Filter", "C Rank Reference"], index=0, key="global_mode_selector")
        st.divider()

    if mode == "C Rank Reference":
        _render_c_rank_mode(df)
    else:
        _render_custom_mode(df)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--csv", default=str(Path(__file__).parent / "data" / "breakout_follow_pool.csv"))
    args, _ = parser.parse_known_args()
    return args


def _render_custom_mode(df: pd.DataFrame) -> None:
    st.markdown(
        """
        <style>
        [data-testid="stHeader"] {
            display: none !important;
        }
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 1rem !important;
        }
        div[data-testid="stVerticalBlock"] > div {
            padding-bottom: 0.3rem !important;
        }
        .streamlit-expanderHeader {
            padding-top: 0.3rem !important;
            padding-bottom: 0.3rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    kpis_charts_container = st.container()
    summary_container = st.container()
    
    filters_by_group = _funnel_filters(df)
    filters = _flatten_filters(filters_by_group)

    filtered = apply_filters(df, filters)

    with kpis_charts_container:
        _render_kpis(filtered)
        _render_charts(filtered)

    with summary_container:
        _render_current_filter_summary(filters_by_group, len(filtered), len(df))

    _render_table_controls(filtered, df)


def _render_c_rank_mode(df: pd.DataFrame) -> None:
    with st.expander("ℹ️ C Rank Selection & Reference Rules", expanded=True):
        _render_c_rank_rules()
    limit_label = st.selectbox("Top N", ["All rows", "Top 10", "Top 20", "Top 30", "Top 50"])
    limit = None if limit_label == "All rows" else int(limit_label.split()[1])
    ranked = apply_c_rank_mode(df, limit=limit)
    st.caption(f"C Rank Reference | Rows: {len(ranked)}/{len(df)} | signal=True | rank_C_continuous asc")
    leading_columns = ["code", "rank_C_continuous", "C_continuous", "is_priority"]
    columns = leading_columns + [column for column in get_all_table_columns() if column not in set(leading_columns)]
    render_table(ranked, [column for column in columns if column in ranked.columns], height=720)
    _download_current_rows(ranked, "c_rank_reference.csv")


def _render_c_rank_rules() -> None:
    left, right = st.columns(2)
    with left:
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
    with right:
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


def _funnel_filters(df: pd.DataFrame) -> dict[str, list[FilterSpec]]:
    groups = get_filter_funnel_groups()
    if list(groups) != FUNNEL_ORDER:
        raise ValueError("Filter funnel configuration is out of sync with the dashboard layout.")
    filters_by_group: dict[str, list[FilterSpec]] = {group: [] for group in FUNNEL_ORDER}

    with st.expander("⏳ Filter Funnel Config Panel", expanded=True):
        cols = st.columns(5)

        with cols[0]:
            st.markdown("##### 1. Route")
            candidate_rules = ["All"] + _unique_values(df, "ibd_candidate_rule")
            candidate_rule = st.selectbox("IBD Candidate Rule", candidate_rules, index=0, key="funnel_route_rule")
            if candidate_rule != "All":
                filters_by_group["Route"].append(FilterSpec("ibd_candidate_rule", "equals", candidate_rule))

        with cols[1]:
            st.markdown("##### 2. Entry & Strength")
            entry_valid = st.radio("IBD Entry Valid", ["All", "Valid only", "Invalid only"], index=0, key="funnel_entry_valid")
            if entry_valid == "Valid only":
                filters_by_group["Entry Confirmation & Strength"].append(FilterSpec("ibd_entry_valid", "is true"))
            elif entry_valid == "Invalid only":
                filters_by_group["Entry Confirmation & Strength"].append(FilterSpec("ibd_entry_valid", "is false"))

            strength_disabled = entry_valid != "Valid only"
            for field in [
                "ibd_entry_volume_ratio",
                "ibd_entry_close_position",
                "ibd_entry_breakout_range_ratio",
            ]:
                spec = _range_filter(df, field, st, key_prefix="entry", disabled=strength_disabled)
                if spec is not None:
                    filters_by_group["Entry Confirmation & Strength"].append(spec)

        with cols[2]:
            st.markdown("##### 3. Weekly Vol & Price")
            spec = _range_filter(df, "volume_ratio", st, key_prefix="weekly")
            if spec is not None:
                filters_by_group["Weekly Volume & Price"].append(spec)
            bullish = st.selectbox("Is Bullish", ["All", "True", "False"], index=0, key="funnel_weekly_is_bullish")
            _append_bool_filter(filters_by_group["Weekly Volume & Price"], "is_bullish", bullish)

        with cols[3]:
            st.markdown("##### 4. Structure")
            for field in ["touched_ema10_count"]:
                spec = _range_filter(df, field, st, key_prefix="structure")
                if spec is not None:
                    filters_by_group["Structure"].append(spec)
            spec = _pullback_magnitude_filter(df)
            if spec is not None:
                filters_by_group["Structure"].append(spec)

        with cols[4]:
            st.markdown("##### 5. Grouping")
            for field in ["sector", "industry"]:
                choices = _unique_values(df, field)
                selected = st.multiselect(get_field_label(field), choices, default=[], key=f"funnel_group_{field}")
                if selected:
                    filters_by_group["Grouping"].append(FilterSpec(field, "in", selected))

    return filters_by_group


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
        
    st.markdown(f"📊 **Filtered Rows: `{filtered_count}/{total_count}`** ｜ " + " ｜ ".join(summary_parts))


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


def _render_kpis(df: pd.DataFrame) -> None:
    kpis = build_kpis(df)
    columns = st.columns(5)
    columns[0].metric("Filtered Rows", kpis["filtered_rows"])
    columns[1].metric("IBD Valid Rate", f"{kpis['ibd_valid_rate_pct']:.2f}%")
    columns[2].metric("Median IBD Volume Ratio", _format_number(kpis["median_ibd_entry_volume_ratio"], "x"))
    columns[3].metric("Median Close Position", _format_number(kpis["median_ibd_entry_close_position"]))
    columns[4].metric("Median Range Ratio [Valid]", _format_number(kpis["median_ibd_entry_breakout_range_ratio_valid"], "x"))



def _render_charts(df: pd.DataFrame) -> None:
    charts = build_chart_data(df)
    col1, col2, col3 = st.columns(3)

    with col1:
        concentration = charts["sector_concentration"]
        if concentration.empty:
            st.info("No rows for Sector Concentration.")
        else:
            top = concentration.head(10)
            fig = px.bar(
                top,
                x="share_pct",
                y="sector",
                orientation="h",
                text="share_pct",
                hover_data=["row_count", "valid_count", "valid_rate_pct", "top_industry"],
                title="Sector Concentration",
                height=240,
            )
            fig.update_layout(margin={"l": 8, "r": 8, "t": 36, "b": 24}, yaxis={"autorange": "reversed"})
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        route_df = charts["route_quality"]
        if route_df.empty:
            st.info("No rows for Route Quality.")
        else:
            fig = px.bar(
                route_df,
                x="ibd_candidate_rule",
                y="total_count",
                color="valid_rate_pct",
                color_continuous_scale="Greens",
                range_color=[0, 100],
                text="total_count",
                hover_data={
                    "ibd_candidate_rule": True,
                    "valid_count": True,
                    "invalid_count": True,
                    "valid_rate_pct": ":.2f",
                    "median_ibd_entry_volume_ratio": ":.2f",
                    "median_ibd_entry_close_position": ":.2f",
                    "median_ibd_entry_breakout_range_ratio": ":.2f",
                },
                title="Route Quality",
                height=240,
            )
            fig.update_layout(
                margin={"l": 8, "r": 8, "t": 36, "b": 16},
                xaxis_title="",
                yaxis_title="",
            )
            st.plotly_chart(fig, use_container_width=True)

    with col3:
        action_map = charts["trend_volume_map"]
        if action_map.empty:
            st.info("No rows for Trend × Volume Map [Valid Only].")
        else:
            fig = px.scatter(
                action_map,
                x="touched_ema10_jittered",
                y="volume_ratio",
                color="ibd_candidate_rule",
                symbol="ibd_candidate_rule",
                hover_data=[
                    "code",
                    "sector",
                    "industry",
                    "signal_source",
                    "ibd_candidate_rule",
                    "dry_status",
                    "touched_ema10_count",
                    "ibd_entry_volume_ratio",
                    "ibd_entry_close_position",
                ],
                title="Trend × Volume Map [Valid Only]",
                height=240,
            )
            fig.add_hline(y=1.3, line_dash="dot", line_color="#546E7A")
            fig.update_layout(
                margin={"l": 8, "r": 8, "t": 36, "b": 24},
                legend={"orientation": "h", "y": -0.2},
                xaxis={"title": "Trend Maturity (10W EMA Touch Count)"},
                yaxis={"title": "Current Volume Ratio"},
            )
            st.plotly_chart(fig, use_container_width=True)


def _render_table_controls(filtered_df: pd.DataFrame, original_df: pd.DataFrame) -> None:
    column_view = st.selectbox(
        "Column View",
        ["All Fields", "Signal", "IBD Entry", "Structure", "Volume/Pullback", "Grouping", "Reference"],
    )
    columns = get_column_view_fields(column_view)
    render_table(filtered_df, columns)
    export_df = original_df.loc[filtered_df.index].copy()
    _download_current_rows(export_df, "breakout_pool_filtered.csv")


def _download_current_rows(df: pd.DataFrame, filename: str) -> None:
    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name=filename,
        mime="text/csv",
    )


def _append_bool_filter(filters: list[FilterSpec], field: str, selected: str) -> None:
    if selected == "True":
        filters.append(FilterSpec(field, "is true"))
    elif selected == "False":
        filters.append(FilterSpec(field, "is false"))


def _range_filter(
    df: pd.DataFrame,
    field: str,
    container: Any,
    default_bounds: tuple[float | None, float | None] | None = None,
    key_prefix: str = "",
    disabled: bool = False,
) -> FilterSpec | None:
    if field not in df.columns:
        return None
    values = pd.to_numeric(df[field], errors="coerce").dropna()
    if values.empty:
        return None
    min_value = float(values.min())
    max_value = float(values.max())
    if min_value == max_value:
        return None
    default_min, default_max = min_value, max_value
    if default_bounds is not None:
        if default_bounds[0] is not None:
            default_min = min(max(float(default_bounds[0]), min_value), max_value)
        if default_bounds[1] is not None:
            default_max = min(max(float(default_bounds[1]), min_value), max_value)
    selected = container.slider(
        get_field_label(field),
        min_value=min_value,
        max_value=max_value,
        value=(default_min, default_max),
        key=f"{key_prefix}_range_{field}" if key_prefix else f"range_{field}",
        disabled=disabled,
        format="%.2f",
    )
    if disabled:
        return None
    if selected == (min_value, max_value):
        return None
    return FilterSpec(field, "between", selected[0], selected[1])


def _pullback_magnitude_filter(df: pd.DataFrame) -> FilterSpec | None:
    field = "pullback_pct"
    if field not in df.columns:
        return None
    values = pd.to_numeric(df[field], errors="coerce").dropna().abs()
    if values.empty:
        return None
    min_value = float(values.min())
    max_value = float(values.max())
    if min_value == max_value:
        return None
    selected = st.slider(
        "Pullback Pct magnitude",
        min_value=min_value,
        max_value=max_value,
        value=(min_value, max_value),
        key="structure_range_pullback_pct_magnitude",
        format="%.2f",
    )
    if selected == (min_value, max_value):
        return None
    return FilterSpec(field, "between", -selected[1], -selected[0])


def _unique_values(df: pd.DataFrame, field: str) -> list[str]:
    if field not in df.columns:
        return []
    values = df[field].dropna().astype(str).sort_values().unique().tolist()
    return [value for value in values if value]


def _format_number(value: float | int | None, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if suffix == "%":
        return f"{value:.2%}"
    return f"{value:.2f}{suffix}"


if __name__ == "__main__":
    main()
