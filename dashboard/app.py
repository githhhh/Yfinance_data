from __future__ import annotations

import argparse
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


st.set_page_config(page_title="Breakout Pool Dashboard", layout="wide", initial_sidebar_state="auto")

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

    try:
        df = cached_load_pool_csv(args.csv, _csv_cache_fingerprint(args.csv))
    except Exception as exc:
        st.error(f"Could not load CSV: {exc}")
        return

    with st.sidebar:
        st.subheader("🎯 Mode Selector")
        mode = st.radio("Mode", ["IBD Review", "C Rank Reference"], index=0, key="global_mode_selector")
        st.divider()

    if mode == "C Rank Reference":
        _render_c_rank_mode(df)
    else:
        _render_custom_mode(df)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    default_csv = Path(__file__).resolve().parents[1] / "us" / "breakout_follow_pool.csv"
    parser.add_argument("--csv", default=str(default_csv))
    args, _ = parser.parse_known_args()
    return args


def _get_snapshot_date(df: pd.DataFrame) -> str:
    if "snapshot_date" in df.columns:
        valid = df["snapshot_date"].dropna()
        if not valid.empty:
            return str(valid.iloc[0])
    return "N/A"


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

    snapshot_date = _get_snapshot_date(df)
    total_pool = len(df)
    active_signals = int(df["signal"].sum()) if "signal" in df.columns else 0
    st.markdown(
        f"#### Breakout Follow Pool ｜ Total Pool: **{total_pool}** ｜ Active Signal: **{active_signals}** ｜ Snapshot Date: **`{snapshot_date}`**"
    )

    filters_by_group, route_df = _funnel_filters(df)
    filters = _flatten_filters(filters_by_group)

    filtered = apply_filters(df, filters)
    filtered = apply_default_review_order(filtered)

    _render_current_filter_summary(filters_by_group, len(filtered), len(df))

    active_count = int(df["signal"].sum()) if "signal" in df.columns else len(df)
    st.markdown(f"**Showing {len(filtered)} of {active_count} Active Signals**")
    _render_kpis(filtered)

    _render_table_controls(filtered, df)

    with st.expander("📈 Route Quality (Optional Chart)", expanded=False):
        _render_charts(route_df)


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


def _funnel_filters(df: pd.DataFrame) -> tuple[dict[str, list[FilterSpec]], pd.DataFrame]:
    groups = get_filter_funnel_groups()
    if list(groups) != FUNNEL_ORDER:
        raise ValueError("Filter funnel configuration is out of sync with the dashboard layout.")
    filters_by_group: dict[str, list[FilterSpec]] = {group: [] for group in FUNNEL_ORDER}

    with st.expander("⏳ Filter Funnel Config Panel", expanded=True):
        cols = st.columns([1, 1, 2])

        with cols[0]:
            st.markdown("##### 1. Route")
            candidate_rules = ["All"] + _unique_values(df, "ibd_candidate_rule")
            candidate_rule = st.selectbox("IBD Candidate Rule", candidate_rules, index=0, key="funnel_route_rule")
            filters_by_group["Route"].append(FilterSpec("signal", "is true", label="Signal"))
            if candidate_rule != "All":
                filters_by_group["Route"].append(FilterSpec("ibd_candidate_rule", "equals", candidate_rule))

        route_df = apply_filters(df, filters_by_group["Route"])
        status_counts = build_entry_status_counts(route_df)

        with cols[1]:
            st.markdown("##### 2. Entry Status")
            status_options = ["All", "ACTIONABLE", "UNCONFIRMED", "BELOW_TRIGGER", "EXTENDED"]
            selected_status = st.radio(
                "IBD Entry Status",
                status_options,
                index=0,
                key="funnel_entry_status",
                format_func=lambda s: f"{s} ({status_counts.get(s, 0)})",
            )
            if selected_status != "All":
                filters_by_group["Entry Status"].append(FilterSpec("ibd_entry_status", "equals", selected_status))

        status_df = apply_filters(route_df, filters_by_group["Entry Status"])

        with cols[2]:
            st.markdown("##### 3. Optional Quality Filters")
            disabled_daily = selected_status == "UNCONFIRMED"

            spec = _range_filter(status_df, "current_vs_ibd_candidate_pct", st, key_prefix="cand_pct", disabled=False)
            if spec is not None:
                filters_by_group["Optional Quality Filters"].append(spec)

            spec = _range_filter(status_df, "ibd_entry_volume_ratio", st, key_prefix="entry", disabled=disabled_daily)
            if spec is not None:
                filters_by_group["Optional Quality Filters"].append(spec)

            spec = _range_filter(status_df, "volume_ratio", st, key_prefix="weekly", disabled=False)
            if spec is not None:
                filters_by_group["Optional Quality Filters"].append(spec)

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


def _render_kpis(df: pd.DataFrame) -> None:
    kpis = build_kpis(df)
    columns = st.columns(4)
    columns[0].metric("Filtered Rows", kpis["filtered_rows"])
    columns[1].metric("Median Candidate Dist", _format_number(kpis["median_current_vs_ibd_candidate_pct"], "%"))
    columns[2].metric("Median IBD Volume Ratio", _format_number(kpis["median_ibd_entry_volume_ratio"], "x"))
    columns[3].metric("Median Volume Ratio", _format_number(kpis["median_volume_ratio"], "x"))


def _render_charts(df: pd.DataFrame) -> None:
    charts = build_chart_data(df)
    route_df = charts["route_quality"]
    if route_df.empty:
        st.info("No rows for Route Quality.")
        return
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


def _render_table_controls(filtered_df: pd.DataFrame, original_df: pd.DataFrame) -> None:
    column_view = st.selectbox(
        "Column View",
        ["IBD Decision", "All Fields", "Signal", "IBD Entry", "Volume/Pullback", "Reference"],
        index=0,
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


def _unique_values(df: pd.DataFrame, field: str) -> list[str]:
    if field not in df.columns:
        return []
    values = df[field].dropna().astype(str).sort_values().unique().tolist()
    return [value for value in values if value]


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
