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
        ("advanced filters AND logic", _check_advanced_filters),
        ("sort specs", _check_sort_specs),
        ("chart: Route Quality aggregation", _check_route_quality_chart),
        ("chart: Trend Volume Map row source", _check_trend_volume_map_chart),
        ("chart: Breakout Pattern aggregation", _check_breakout_pattern_chart),
        ("chart: Sector Concentration aggregation", _check_sector_concentration_chart),
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


def _check_advanced_filters(df: pd.DataFrame) -> None:
    cases = [
        (
            [
                FilterSpec("ibd_candidate_rule", "in", ["ceiling_pullback", "pivot"]),
                FilterSpec("ibd_entry_valid", "is true"),
                FilterSpec("ibd_entry_close_position", ">=", 0.5),
            ],
            df[
                df["ibd_candidate_rule"].isin(["ceiling_pullback", "pivot"])
                & _true_mask(df["ibd_entry_valid"])
                & pd.to_numeric(df["ibd_entry_close_position"], errors="coerce").ge(0.5)
            ],
        ),
        (
            [
                FilterSpec("volume_ratio", ">=", 1.2),
                FilterSpec("is_bullish", "is true"),
                FilterSpec("touched_ema10_count", ">=", 2),
            ],
            df[
                pd.to_numeric(df["volume_ratio"], errors="coerce").ge(1.2)
                & _true_mask(df["is_bullish"])
                & pd.to_numeric(df["touched_ema10_count"], errors="coerce").ge(2)
            ],
        ),
        (
            [
                FilterSpec("sector", "in", ["Technology Services", "Finance"]),
                FilterSpec("ibd_entry_breakout_range_ratio", "between", 1.0, 1.5),
            ],
            df[
                df["sector"].isin(["Technology Services", "Finance"])
                & pd.to_numeric(df["ibd_entry_breakout_range_ratio"], errors="coerce").between(1.0, 1.5, inclusive="both")
            ],
        ),
    ]
    for filters, expected in cases:
        actual = apply_filters(df, filters)
        assert set(actual["code"]) == set(expected["code"])


def _check_sort_specs(df: pd.DataFrame) -> None:
    cases = [
        [SortSpec("ibd_entry_volume_ratio", "desc")],
        [
            SortSpec("ibd_entry_volume_ratio", "desc"),
            SortSpec("ibd_entry_close_position", "desc"),
            SortSpec("ibd_entry_breakout_range_ratio", "desc"),
        ],
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


def _check_route_quality_chart(df: pd.DataFrame) -> None:
    chart = build_chart_data(df)["route_quality"]
    valid_df = df[
        df["ibd_candidate_rule"].fillna("").astype("string").str.strip().ne("")
        & df["ibd_candidate_rule"].ne("(empty)")
    ]
    assert int((chart["valid_count"] + chart["invalid_count"]).sum()) == len(valid_df)
    assert "(empty)" not in chart["ibd_candidate_rule"].values
    required = {
        "ibd_candidate_rule",
        "valid_count",
        "invalid_count",
        "total_count",
        "valid_rate_pct",
        "median_ibd_entry_volume_ratio",
        "median_ibd_entry_close_position",
        "median_volume_ratio",
        "median_ibd_entry_breakout_range_ratio",
    }
    assert required.issubset(chart.columns)


def _check_trend_volume_map_chart(df: pd.DataFrame) -> None:
    chart = build_chart_data(df)["trend_volume_map"]
    valid_mask = _true_mask(df.get("ibd_entry_valid", pd.Series(index=df.index, dtype="object")))
    expected = df[valid_mask].dropna(subset=["touched_ema10_count", "volume_ratio"])
    assert len(chart) == len(expected)
    assert set(chart["code"]) == set(expected["code"])
    if not chart.empty:
        assert all(chart["entry_status"] == "IBD valid")
    required = {"entry_status", "dry_status", "sector", "industry", "touched_ema10_count", "touched_ema10_jittered", "volume_ratio"}
    assert required.issubset(chart.columns)


def _check_breakout_pattern_chart(df: pd.DataFrame) -> None:
    chart = build_chart_data(df)["breakout_pattern"]
    expected_patterns = {
        "GAP_UP",
        "SOLID_BREAKOUT",
        "MODERATE_BREAKOUT",
        "MARGINAL_BREAKOUT",
        "BULL_TRAP",
    }
    assert set(chart["pattern"]) == expected_patterns
    assert {"pattern", "count", "share_pct", "tickers", "median_vol_ratio", "median_close_pos"}.issubset(chart.columns)


def _check_sector_concentration_chart(df: pd.DataFrame) -> None:
    chart = build_chart_data(df)["sector_concentration"]
    assert int(chart["row_count"].sum()) == len(df)
    assert abs(float(chart["share_pct"].sum()) - 100.0) <= 0.05
    assert {"sector", "row_count", "share_pct", "valid_count", "valid_rate_pct", "top_industry"}.issubset(chart.columns)


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
