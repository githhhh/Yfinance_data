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
    SortSpec,
    apply_c_rank_mode,
    apply_filters,
    apply_sort,
    build_active_filter_summary,
    build_chart_data,
    build_kpis,
    load_pool_csv,
)
from dashboard.field_config import (
    get_all_table_columns,
    get_column_view_fields,
    get_field_label,
    get_filter_funnel_groups,
    get_sortable_fields,
)
from dashboard.table_view import render_table


st.set_page_config(page_title="Breakout Pool Dashboard", layout="wide")


@st.cache_data
def cached_load_pool_csv(path: str, cache_fingerprint: tuple[int, int]) -> pd.DataFrame:
    del cache_fingerprint
    return load_pool_csv(path)


def _csv_cache_fingerprint(path: str | Path) -> tuple[int, int]:
    stat = Path(path).stat()
    return (stat.st_mtime_ns, stat.st_size)


def main() -> None:
    args = _parse_args()
    st.title("Breakout Pool")

    try:
        df = cached_load_pool_csv(args.csv, _csv_cache_fingerprint(args.csv))
    except Exception as exc:
        st.error(f"Could not load CSV: {exc}")
        return

    mode = st.radio("Mode", ["Custom Filter", "C Rank Reference"], index=0)
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
    filters = _funnel_filters(df)
    sort_specs = _sort_specs()

    filtered = apply_filters(df, filters)
    sorted_df = apply_sort(filtered, sort_specs)

    chips = [f"Rows: {len(filtered)}/{len(df)}"]
    chips.extend(build_active_filter_summary(filters, sort_specs))
    st.caption(" | ".join(chips) if chips else "No active filters")

    _render_kpis(filtered)
    _render_charts(filtered)
    st.divider()
    _render_sort_summary(sort_specs)
    _render_table_controls(sorted_df, df)


def _render_c_rank_mode(df: pd.DataFrame) -> None:
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
    st.subheader("C Rank Reference Mode")
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


def _funnel_filters(df: pd.DataFrame) -> list[FilterSpec]:
    st.subheader("Filters")
    groups = get_filter_funnel_groups()
    funnel_order = [
        "1 Route",
        "2 Entry Confirmation & Strength",
        "3 Weekly Volume & Price",
        "4 Structure",
        "5 Grouping",
    ]
    if list(groups) != funnel_order:
        raise ValueError("Filter funnel configuration is out of sync with the dashboard layout.")
    tabs = st.tabs(funnel_order)
    filters: list[FilterSpec] = []

    with tabs[0]:
        candidate_rules = ["All"] + _unique_values(df, "ibd_candidate_rule")
        candidate_rule = st.selectbox("IBD Candidate Rule", candidate_rules, index=0)
        if candidate_rule != "All":
            filters.append(FilterSpec("ibd_candidate_rule", "equals", candidate_rule))

    with tabs[1]:
        entry_valid = st.radio("IBD Entry Valid", ["All", "Valid only", "Invalid only"], index=0)
        if entry_valid == "Valid only":
            filters.append(FilterSpec("ibd_entry_valid", "is true"))
        elif entry_valid == "Invalid only":
            filters.append(FilterSpec("ibd_entry_valid", "is false"))

        strength_disabled = entry_valid != "Valid only"
        for field in ["ibd_entry_volume_ratio", "ibd_entry_close_vs_trigger_pct"]:
            spec = _range_filter(df, field, st, key_prefix="entry", disabled=strength_disabled)
            if spec is not None:
                filters.append(spec)

    with tabs[2]:
        spec = _range_filter(df, "volume_ratio", st, key_prefix="weekly")
        if spec is not None:
            filters.append(spec)
        bullish = st.selectbox("Is Bullish", ["All", "True", "False"], index=0)
        _append_bool_filter(filters, "is_bullish", bullish)

    with tabs[3]:
        for field in ["touched_ema10_count"]:
            spec = _range_filter(df, field, st, key_prefix="structure")
            if spec is not None:
                filters.append(spec)
        spec = _pullback_magnitude_filter(df)
        if spec is not None:
            filters.append(spec)

    with tabs[4]:
        for field in ["sector", "industry"]:
            choices = _unique_values(df, field)
            selected = st.multiselect(get_field_label(field), choices, default=[])
            if selected:
                filters.append(FilterSpec(field, "in", selected))

    return filters


def _sort_specs() -> list[SortSpec]:
    st.subheader("Sort")
    sortable = [""] + get_sortable_fields()
    columns = st.columns(3)
    specs: list[SortSpec] = []
    for index in range(3):
        default = SortSpec("", "asc", enabled=False)
        with columns[index]:
            field = st.selectbox(
                f"sort_{index + 1}",
                sortable,
                index=_select_index(sortable, default.field),
                format_func=lambda value: "None" if value == "" else get_field_label(value),
                key=f"sort_field_{index}",
            )
            direction = st.selectbox(
                f"direction_{index + 1}",
                ["asc", "desc"],
                index=0 if default.direction == "asc" else 1,
                key=f"sort_direction_{index}",
            )
        if field:
            specs.append(SortSpec(field, direction))
    return specs


def _render_kpis(df: pd.DataFrame) -> None:
    kpis = build_kpis(df)
    columns = st.columns(4)
    columns[0].metric("Filtered Rows", kpis["filtered_rows"])
    columns[1].metric("IBD Valid Rate", f"{kpis['ibd_valid_rate_pct']:.2f}%")
    columns[2].metric("Median IBD Volume Ratio", _format_number(kpis["median_ibd_entry_volume_ratio"], "x"))
    columns[3].metric("Median Close vs Trigger", _format_number(kpis["median_ibd_entry_close_vs_trigger_pct"], "%"))


def _render_charts(df: pd.DataFrame) -> None:
    charts = build_chart_data(df)
    left, right = st.columns(2)
    with left:
        matrix = charts["signal_quality_matrix"]
        if matrix.empty:
            st.info("No rows for Signal Quality Matrix.")
        else:
            fig = px.scatter(
                matrix,
                x="signal_source",
                y="ibd_candidate_rule",
                size="total_count",
                color="valid_rate_pct",
                color_continuous_scale="Greens",
                range_color=[0, 100],
                text="total_count",
                hover_data={
                    "signal_source": True,
                    "ibd_candidate_rule": True,
                    "valid_count": True,
                    "invalid_count": True,
                    "valid_rate_pct": ":.2f",
                    "median_ibd_entry_volume_ratio": ":.2f",
                    "median_ibd_entry_close_vs_trigger_pct": ":.2%",
                    "median_pct_above_ceiling": ":.1f",
                },
                title="Signal Quality Matrix",
                height=260,
            )
            fig.update_layout(
                margin={"l": 8, "r": 8, "t": 36, "b": 48},
                xaxis_title="Signal Source",
                yaxis_title="Candidate Rule",
            )
            st.plotly_chart(fig, use_container_width=True)

    with right:
        action_map = charts["structure_action_map"]
        if action_map.empty:
            st.info("No rows for Structure Action Map.")
        else:
            fig = px.scatter(
                action_map,
                x="pct_above_ceiling",
                y="volume_ratio",
                color="entry_status",
                symbol="signal_source",
                hover_data=[
                    "code",
                    "sector",
                    "industry",
                    "ibd_candidate_rule",
                    "dry_status",
                    "touched_ema10_count",
                    "ibd_entry_volume_ratio",
                    "ibd_entry_close_vs_trigger_pct",
                ],
                title="Structure Action Map",
                height=260,
            )
            fig.add_vline(x=5, line_dash="dot", line_color="#2E7D32")
            fig.add_vline(x=10, line_dash="dot", line_color="#F9A825")
            fig.add_hline(y=1.3, line_dash="dot", line_color="#546E7A")
            x_upper = min(max(float(action_map["pct_above_ceiling"].quantile(0.95)), 20.0), 120.0)
            fig.update_layout(
                margin={"l": 8, "r": 8, "t": 36, "b": 48},
                legend={"orientation": "h", "y": -0.3},
                xaxis={"range": [0, x_upper], "title": "Pct Above Ceiling"},
                yaxis={"title": "Current Volume Ratio"},
            )
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("Sector concentration", expanded=False):
        concentration = charts["sector_concentration"]
        if concentration.empty:
            st.info("No rows for Sector Concentration.")
        else:
            top = concentration.head(12)
            fig = px.bar(
                top,
                x="share_pct",
                y="sector",
                orientation="h",
                text="share_pct",
                hover_data=["row_count", "valid_count", "valid_rate_pct", "top_industry"],
                title="Sector Concentration",
                height=260,
            )
            fig.update_layout(margin={"l": 8, "r": 8, "t": 36, "b": 24}, yaxis={"autorange": "reversed"})
            st.plotly_chart(fig, use_container_width=True)


def _render_sort_summary(sort_specs: list[SortSpec]) -> None:
    active = [f"{get_field_label(spec.field)} {spec.direction}" for spec in sort_specs if spec.enabled]
    st.caption("Sort Bar: " + (" -> ".join(active) if active else "None"))


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
    )
    if selected == (min_value, max_value):
        return None
    return FilterSpec(field, "between", -selected[1], -selected[0])


def _unique_values(df: pd.DataFrame, field: str) -> list[str]:
    if field not in df.columns:
        return []
    values = df[field].dropna().astype(str).sort_values().unique().tolist()
    return [value for value in values if value]


def _select_index(options: list[Any], value: Any) -> int:
    return options.index(value) if value in options else 0


def _format_number(value: float | int | None, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if suffix == "%":
        return f"{value:.2%}"
    return f"{value:.2f}{suffix}"


if __name__ == "__main__":
    main()
