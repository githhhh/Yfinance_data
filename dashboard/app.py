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
    build_preset_filters,
    build_preset_sort,
    combine_filter_specs,
    load_pool_csv,
)
from dashboard.field_config import (
    FIELD_CONFIG,
    PRESETS,
    get_column_view_fields,
    get_custom_mode_fields,
    get_default_table_columns,
    get_field_label,
    get_field_type,
    get_filterable_fields,
    get_preset_options,
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

    mode = st.sidebar.radio("Mode", ["Custom Filter", "C Rank Reference"], horizontal=False)
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
    preset_key = _preset_selector()
    filters = combine_filter_specs(build_preset_filters(preset_key), _sidebar_filters(df, preset_key))
    sort_specs = _sort_specs(preset_key)

    filtered = apply_filters(df, filters)
    sorted_df = apply_sort(filtered, sort_specs)

    chips = [f"Preset: {PRESETS[preset_key]['label']}", f"Rows: {len(filtered)}/{len(df)}"]
    chips.extend(build_active_filter_summary(filters, sort_specs))
    st.caption(" | ".join(chips))

    _render_kpis(filtered)
    _render_charts(filtered)
    st.divider()
    _render_sort_summary(sort_specs)
    _render_table_controls(sorted_df, df)


def _render_c_rank_mode(df: pd.DataFrame) -> None:
    limit_label = st.sidebar.selectbox("Display range", ["All rows", "Top 10", "Top 20", "Top 30", "Top 50"])
    limit = None if limit_label == "All rows" else int(limit_label.split()[1])
    ranked = apply_c_rank_mode(df, limit=limit)
    st.caption(f"C Rank Reference | Rows: {len(ranked)}/{len(df)} | signal=True | rank_C_continuous asc")
    columns = [
        "code",
        "sector",
        "industry",
        "signal_source",
        "ibd_candidate_rule",
        "C_continuous",
        "rank_C_continuous",
        "is_priority",
        "ibd_entry_valid",
        "ibd_entry_volume_ratio",
        "pct_above_ceiling",
        "touched_ema10_count",
    ]
    render_table(ranked, [column for column in columns if column in ranked.columns], height=720)
    _download_current_rows(ranked, "c_rank_reference.csv")


def _preset_selector() -> str:
    options = get_preset_options()
    labels = [label for _, label in options]
    selected_label = st.sidebar.selectbox("Preset", labels, index=0)
    return dict((label, key) for key, label in options)[selected_label]


def _sidebar_filters(df: pd.DataFrame, preset_key: str) -> list[FilterSpec]:
    st.sidebar.subheader("Core Filters")
    preset_defaults = _preset_default_values(preset_key)
    filters: list[FilterSpec] = []

    signal = st.sidebar.selectbox(
        "signal",
        ["All", "True", "False"],
        index=_select_index(["All", "True", "False"], preset_defaults.get("signal", "All")),
        key=f"{preset_key}_signal",
    )
    _append_bool_filter(filters, "signal", signal)

    signal_sources = ["All"] + _unique_values(df, "signal_source")
    signal_source = st.sidebar.selectbox(
        "signal_source",
        signal_sources,
        index=_select_index(signal_sources, preset_defaults.get("signal_source", "All")),
        key=f"{preset_key}_signal_source",
    )
    if signal_source != "All":
        filters.append(FilterSpec("signal_source", "equals", signal_source))

    candidate_rules = ["All"] + _unique_values(df, "ibd_candidate_rule")
    candidate_rule = st.sidebar.selectbox(
        "ibd_candidate_rule",
        candidate_rules,
        index=_select_index(candidate_rules, preset_defaults.get("ibd_candidate_rule", "All")),
        key=f"{preset_key}_ibd_candidate_rule",
    )
    if candidate_rule != "All":
        filters.append(FilterSpec("ibd_candidate_rule", "equals", candidate_rule))

    ibd_valid = st.sidebar.selectbox(
        "ibd_entry_valid",
        ["All", "True", "False"],
        index=_select_index(["All", "True", "False"], preset_defaults.get("ibd_entry_valid", "All")),
        key=f"{preset_key}_ibd_entry_valid",
    )
    _append_bool_filter(filters, "ibd_entry_valid", ibd_valid)

    for field in ["ibd_entry_volume_ratio", "ibd_entry_close_vs_trigger_pct"]:
        spec = _range_filter(df, field, st.sidebar, preset_defaults.get(field), key_prefix=preset_key)
        if spec is not None:
            filters.append(spec)

    with st.sidebar.expander("Secondary Filters", expanded=False):
        for field in ["pct_above_ceiling", "touched_ema10_count", "volume_ratio"]:
            spec = _range_filter(df, field, st, preset_defaults.get(field), key_prefix=preset_key)
            if spec is not None:
                filters.append(spec)
        for field in ["sector", "industry"]:
            choices = _unique_values(df, field)
            selected = st.multiselect(field, choices, default=[])
            if selected:
                filters.append(FilterSpec(field, "in", selected))

    filters.extend(_advanced_filters(df))
    return filters


def _advanced_filters(df: pd.DataFrame) -> list[FilterSpec]:
    filterable_fields = get_filterable_fields()
    active_specs: list[FilterSpec] = []
    with st.sidebar.expander(f"Advanced filters · {len(active_specs)} active", expanded=False):
        count = st.number_input("+ Add filter", min_value=0, max_value=10, value=0, step=1)
        for index in range(int(count)):
            enabled = st.checkbox(f"Enable {index + 1}", value=True, key=f"advanced_enabled_{index}")
            field = st.selectbox(
                f"Field {index + 1}",
                filterable_fields,
                format_func=get_field_label,
                key=f"advanced_field_{index}",
            )
            spec = _advanced_filter_row(df, field, enabled, index)
            if spec is not None:
                active_specs.append(spec)
    return active_specs


def _advanced_filter_row(df: pd.DataFrame, field: str, enabled: bool, index: int) -> FilterSpec | None:
    field_type = get_field_type(field)
    operators = _operators_for_type(field_type)
    operator = st.selectbox(f"Operator {index + 1}", operators, key=f"advanced_operator_{index}")

    if operator in {"is true", "is false", "is empty", "not empty", "non-empty"}:
        return FilterSpec(field, operator, enabled=enabled)

    if operator in {"in", "not in"}:
        values = _unique_values(df, field)
        selected = st.multiselect(f"Value {index + 1}", values, key=f"advanced_value_{index}")
        return FilterSpec(field, operator, selected, enabled=enabled) if selected else None

    if operator == "between":
        first = st.text_input(f"From {index + 1}", key=f"advanced_value_{index}_from")
        second = st.text_input(f"To {index + 1}", key=f"advanced_value_{index}_to")
        return FilterSpec(field, operator, first, second, enabled=enabled) if first and second else None

    value = st.text_input(f"Value {index + 1}", key=f"advanced_value_{index}")
    return FilterSpec(field, operator, value, enabled=enabled) if value else None


def _sort_specs(preset_key: str) -> list[SortSpec]:
    st.subheader("Sort")
    sortable = [""] + get_sortable_fields()
    defaults = build_preset_sort(preset_key)
    columns = st.columns(3)
    specs: list[SortSpec] = []
    for index in range(3):
        default = defaults[index] if index < len(defaults) else SortSpec("", "asc", enabled=False)
        with columns[index]:
            field = st.selectbox(
                f"sort_{index + 1}",
                sortable,
                index=_select_index(sortable, default.field),
                format_func=lambda value: "None" if value == "" else get_field_label(value),
                key=f"{preset_key}_sort_field_{index}",
            )
            direction = st.selectbox(
                f"direction_{index + 1}",
                ["asc", "desc"],
                index=0 if default.direction == "asc" else 1,
                key=f"{preset_key}_sort_direction_{index}",
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
    column_view = st.selectbox("Column View", ["Core", "IBD", "Risk", "Full Custom"])
    if column_view == "Full Custom":
        all_columns = get_custom_mode_fields()
        columns = st.multiselect("Columns", all_columns, default=get_default_table_columns(), format_func=get_field_label)
    else:
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


def _preset_default_values(preset_key: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for spec in build_preset_filters(preset_key):
        if spec.operator == "is true":
            values[spec.field] = "True"
        elif spec.operator == "is false":
            values[spec.field] = "False"
        elif spec.operator == "equals":
            values[spec.field] = spec.value
        elif spec.operator == ">=":
            values[spec.field] = (spec.value, None)
        elif spec.operator == "<=":
            values[spec.field] = (None, spec.value)
        elif spec.operator == "between":
            values[spec.field] = (spec.value, spec.value2)
    return values


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
    )
    if selected == (min_value, max_value):
        return None
    return FilterSpec(field, "between", selected[0], selected[1])


def _unique_values(df: pd.DataFrame, field: str) -> list[str]:
    if field not in df.columns:
        return []
    values = df[field].dropna().astype(str).sort_values().unique().tolist()
    return [value for value in values if value]


def _operators_for_type(field_type: str) -> list[str]:
    if field_type == "boolean":
        return ["is true", "is false"]
    if field_type == "category":
        return ["in", "not in"]
    if field_type == "number":
        return [">=", "<=", "between", "is empty", "not empty"]
    if field_type == "date":
        return ["after", "before", "between", "is empty", "not empty"]
    return ["contains", "equals", "startswith", "non-empty"]


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
