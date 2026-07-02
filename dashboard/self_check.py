from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.data_utils import (
    FilterSpec,
    SortSpec,
    apply_c_rank_mode,
    apply_filters,
    apply_sort,
    build_chart_data,
    build_preset_filters,
    build_preset_sort,
    load_pool_csv,
)
from dashboard.field_config import (
    EXCLUDED_CUSTOM_FIELDS,
    get_default_table_columns,
    get_filterable_fields,
    get_sortable_fields,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Self-check breakout pool dashboard logic.")
    parser.add_argument("--csv", required=True, help="Path to breakout_follow_pool.csv")
    args = parser.parse_args()

    checks = [
        ("load and normalize", lambda df: _check_load(df)),
        ("preset: Review All Signals", _check_active_signal_quality_preset),
        ("preset: IBD Valid Breakout", _check_ibd_valid_preset),
        ("preset: Action Clean Entry", _check_action_clean_entry_preset),
        ("preset: Ceiling Breakout", _check_ceiling_breakout_preset),
        ("preset: Ceiling Pullback", _check_ceiling_pullback_preset),
        ("preset: Pivot Review", _check_pivot_review_preset),
        ("preset: 10W EMA Touch", _check_ma_touch_preset),
        ("advanced filters AND logic", _check_advanced_filters),
        ("sort specs", _check_sort_specs),
        ("chart: Signal Quality Matrix aggregation", _check_signal_quality_matrix_chart),
        ("chart: Structure Action Map row source", _check_structure_action_map_chart),
        ("chart: Sector Concentration aggregation", _check_sector_concentration_chart),
        ("chart: IBD Valid Rate by Signal Source aggregation", _check_valid_rate_chart),
        ("chart: Volume x Close Strength row source", _check_scatter_chart),
        ("mode isolation", _check_mode_isolation),
    ]

    label = "setup"
    try:
        df = load_pool_csv(args.csv)
        for label, check in checks:
            check(df)
            print(f"[PASS] {label}")
    except Exception as exc:
        print(f"[FAIL] {label}: {exc}", file=sys.stderr)
        return 1
    return 0


def _check_load(df: pd.DataFrame) -> None:
    required = {"code", "signal", "ibd_entry_valid", "signal_source", "rank_C_continuous"}
    missing = required - set(df.columns)
    assert not missing, f"missing required columns: {sorted(missing)}"
    assert len(df) > 0, "CSV has no rows"


def _check_ibd_valid_preset(df: pd.DataFrame) -> None:
    expected = df[_true_mask(df["signal"]) & _true_mask(df["ibd_entry_valid"])]
    actual = apply_filters(df, build_preset_filters("ibd_valid_breakout"))
    assert set(actual["code"]) == set(expected["code"])


def _check_active_signal_quality_preset(df: pd.DataFrame) -> None:
    expected = df[_true_mask(df["signal"])].sort_values(
        by=["ibd_entry_valid", "ibd_entry_volume_ratio", "ibd_entry_close_vs_trigger_pct"],
        ascending=[False, False, False],
        na_position="last",
        kind="mergesort",
    )
    actual = apply_sort(
        apply_filters(df, build_preset_filters("active_signal_quality")),
        build_preset_sort("active_signal_quality"),
    )
    assert actual["code"].tolist() == expected["code"].tolist()


def _check_action_clean_entry_preset(df: pd.DataFrame) -> None:
    expected = df[
        _true_mask(df["signal"])
        & _true_mask(df["ibd_entry_valid"])
        & pd.to_numeric(df["ibd_entry_volume_ratio"], errors="coerce").ge(1.5)
        & pd.to_numeric(df["ibd_entry_close_vs_trigger_pct"], errors="coerce").between(0.0, 0.05, inclusive="both")
        & pd.to_numeric(df["pct_above_ceiling"], errors="coerce").le(10.0)
    ]
    actual = apply_filters(df, build_preset_filters("action_clean_entry"))
    assert set(actual["code"]) == set(expected["code"])


def _check_ceiling_breakout_preset(df: pd.DataFrame) -> None:
    expected = df[
        _true_mask(df["signal"])
        & df["signal_source"].eq("ceiling_breakout").fillna(False)
        & df["ibd_candidate_rule"].eq("ceiling").fillna(False)
    ]
    actual = apply_filters(df, build_preset_filters("ceiling_breakout"))
    assert set(actual["code"]) == set(expected["code"])


def _check_ceiling_pullback_preset(df: pd.DataFrame) -> None:
    expected = df[
        _true_mask(df["signal"])
        & df["signal_source"].eq("ceiling_breakout").fillna(False)
        & df["ibd_candidate_rule"].eq("ceiling_pullback").fillna(False)
    ]
    actual = apply_filters(df, build_preset_filters("ceiling_pullback"))
    assert set(actual["code"]) == set(expected["code"])


def _check_pivot_review_preset(df: pd.DataFrame) -> None:
    expected = df[
        _true_mask(df["signal"])
        & df["signal_source"].eq("pivot").fillna(False)
        & df["ibd_candidate_rule"].eq("pivot").fillna(False)
    ]
    actual = apply_filters(df, build_preset_filters("pivot_quality"))
    assert set(actual["code"]) == set(expected["code"])


def _check_ma_touch_preset(df: pd.DataFrame) -> None:
    expected = df[
        _true_mask(df["signal"])
        & df["signal_source"].eq("10_wk_ema_touch_confirm").fillna(False)
        & df["ibd_candidate_rule"].eq("ma10_touch_confirm").fillna(False)
    ].sort_values(
        by=["ibd_entry_valid", "touched_ema10_count", "volume_ratio"],
        ascending=[False, False, False],
        na_position="last",
        kind="mergesort",
    )
    actual = apply_sort(apply_filters(df, build_preset_filters("ma_touch_count")), build_preset_sort("ma_touch_count"))
    assert actual["code"].tolist() == expected["code"].tolist()


def _check_advanced_filters(df: pd.DataFrame) -> None:
    cases = [
        (
            [
                FilterSpec("volume_ratio", ">=", 1.3),
                FilterSpec("pullback_v_is_dry", "is true"),
            ],
            df[pd.to_numeric(df["volume_ratio"], errors="coerce").ge(1.3) & _true_mask(df["pullback_v_is_dry"])],
        ),
        (
            [
                FilterSpec("signal_source", "in", ["pivot", "ceiling_breakout"]),
                FilterSpec("ibd_entry_volume_ratio", ">=", 1.5),
            ],
            df[
                df["signal_source"].isin(["pivot", "ceiling_breakout"])
                & pd.to_numeric(df["ibd_entry_volume_ratio"], errors="coerce").ge(1.5)
            ],
        ),
        (
            [
                FilterSpec("breakout_date", "after", "2026-05-01"),
                FilterSpec("code", "contains", "A"),
            ],
            df[
                pd.to_datetime(df["breakout_date"], errors="coerce").gt(pd.Timestamp("2026-05-01"))
                & df["code"].astype("string").str.contains("A", case=False, regex=False, na=False)
            ],
        ),
    ]
    for filters, expected in cases:
        actual = apply_filters(df, filters)
        assert set(actual["code"]) == set(expected["code"])


def _check_sort_specs(df: pd.DataFrame) -> None:
    cases = [
        [SortSpec("ibd_entry_volume_ratio", "desc")],
        [SortSpec("ibd_entry_volume_ratio", "desc"), SortSpec("ibd_entry_close_vs_trigger_pct", "desc")],
        [
            SortSpec("ibd_entry_valid", "desc"),
            SortSpec("ibd_entry_volume_ratio", "desc"),
            SortSpec("pct_above_ceiling", "asc"),
        ],
    ]
    for specs in cases:
        expected = df.sort_values(
            by=[spec.field for spec in specs],
            ascending=[spec.direction != "desc" for spec in specs],
            na_position="last",
            kind="mergesort",
        )
        actual = apply_sort(df, specs)
        assert actual["code"].tolist() == expected["code"].tolist()


def _check_valid_rate_chart(df: pd.DataFrame) -> None:
    filtered = apply_filters(df, build_preset_filters("ibd_valid_breakout"))
    chart = build_chart_data(filtered)["ibd_valid_rate_by_signal_source"]
    assert int((chart["valid_count"] + chart["invalid_count"]).sum()) == len(filtered)

    expected = filtered.copy()
    expected["signal_source"] = expected["signal_source"].fillna("(empty)").replace("", "(empty)")
    expected["valid"] = _true_mask(expected["ibd_entry_valid"]).astype(int)
    grouped = expected.groupby("signal_source", dropna=False).agg(total_count=("valid", "size"), valid_count=("valid", "sum"))
    grouped["invalid_count"] = grouped["total_count"] - grouped["valid_count"]
    grouped["valid_rate_pct"] = (grouped["valid_count"] / grouped["total_count"] * 100).round(2)
    expected_chart = grouped.reset_index()[
        ["signal_source", "valid_count", "invalid_count", "total_count", "valid_rate_pct"]
    ].sort_values(["total_count", "signal_source"], ascending=[False, True], kind="mergesort").reset_index(drop=True)

    pd.testing.assert_frame_equal(chart.reset_index(drop=True), expected_chart.reset_index(drop=True), check_dtype=False)


def _check_signal_quality_matrix_chart(df: pd.DataFrame) -> None:
    filtered = apply_filters(df, build_preset_filters("active_signal_quality"))
    chart = build_chart_data(filtered)["signal_quality_matrix"]
    assert int((chart["valid_count"] + chart["invalid_count"]).sum()) == len(filtered)

    expected = filtered.copy()
    expected["signal_source"] = expected["signal_source"].fillna("(empty)").replace("", "(empty)")
    expected["ibd_candidate_rule"] = expected["ibd_candidate_rule"].fillna("(empty)").replace("", "(empty)")
    expected["valid"] = _true_mask(expected["ibd_entry_valid"]).astype(int)
    grouped = expected.groupby(["signal_source", "ibd_candidate_rule"], dropna=False).agg(
        total_count=("code", "size"),
        valid_count=("valid", "sum"),
        median_ibd_entry_volume_ratio=("ibd_entry_volume_ratio", "median"),
        median_ibd_entry_close_vs_trigger_pct=("ibd_entry_close_vs_trigger_pct", "median"),
        median_volume_ratio=("volume_ratio", "median"),
        median_pct_above_ceiling=("pct_above_ceiling", "median"),
    )
    grouped["invalid_count"] = grouped["total_count"] - grouped["valid_count"]
    grouped["valid_rate_pct"] = (grouped["valid_count"] / grouped["total_count"] * 100).round(2)
    expected_chart = grouped.reset_index()[
        [
            "signal_source",
            "ibd_candidate_rule",
            "valid_count",
            "invalid_count",
            "total_count",
            "valid_rate_pct",
            "median_ibd_entry_volume_ratio",
            "median_ibd_entry_close_vs_trigger_pct",
            "median_volume_ratio",
            "median_pct_above_ceiling",
        ]
    ].sort_values(["total_count", "signal_source", "ibd_candidate_rule"], ascending=[False, True, True], kind="mergesort").reset_index(drop=True)
    pd.testing.assert_frame_equal(chart.reset_index(drop=True), expected_chart.reset_index(drop=True), check_dtype=False)


def _check_structure_action_map_chart(df: pd.DataFrame) -> None:
    filtered = apply_filters(df, build_preset_filters("active_signal_quality"))
    action_map = build_chart_data(filtered)["structure_action_map"]
    expected = filtered.dropna(subset=["pct_above_ceiling", "volume_ratio"])
    assert len(action_map) == len(expected)
    assert set(action_map["code"]) == set(expected["code"])
    required = {"entry_status", "dry_status", "sector", "industry", "pct_above_ceiling", "volume_ratio"}
    assert required.issubset(action_map.columns)


def _check_sector_concentration_chart(df: pd.DataFrame) -> None:
    filtered = apply_filters(df, build_preset_filters("active_signal_quality"))
    chart = build_chart_data(filtered)["sector_concentration"]
    assert int(chart["row_count"].sum()) == len(filtered)
    assert abs(float(chart["share_pct"].sum()) - 100.0) <= 0.05
    assert {"sector", "row_count", "share_pct", "valid_count", "valid_rate_pct", "top_industry"}.issubset(chart.columns)


def _check_scatter_chart(df: pd.DataFrame) -> None:
    filtered = apply_filters(df, build_preset_filters("ibd_valid_breakout"))
    scatter = build_chart_data(filtered)["volume_close_strength"]
    expected = filtered.dropna(subset=["ibd_entry_volume_ratio", "ibd_entry_close_vs_trigger_pct"])
    assert len(scatter) == len(expected)
    assert set(scatter["code"]).issubset(set(filtered["code"]))
    required_hover = {"code", "signal_source", "ibd_candidate_rule", "ibd_entry_price", "pct_above_ceiling"}
    assert required_hover.issubset(scatter.columns)


def _check_mode_isolation(df: pd.DataFrame) -> None:
    assert not EXCLUDED_CUSTOM_FIELDS.intersection(get_filterable_fields())
    assert not EXCLUDED_CUSTOM_FIELDS.intersection(get_sortable_fields())
    assert EXCLUDED_CUSTOM_FIELDS.issubset(set(get_default_table_columns()))

    expected = df[_true_mask(df["signal"])].sort_values(
        by=["rank_C_continuous"],
        ascending=[True],
        na_position="last",
        kind="mergesort",
    )
    actual = apply_c_rank_mode(df)
    assert actual["code"].tolist() == expected["code"].tolist()


def _true_mask(series: pd.Series) -> pd.Series:
    return series.map(lambda value: False if pd.isna(value) else bool(value) is True).astype(bool)


if __name__ == "__main__":
    raise SystemExit(main())
