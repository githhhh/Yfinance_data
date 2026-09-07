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
    apply_filters,
    apply_sort,
    build_chart_data,
    load_pool_csv,
)
from dashboard.services.bf_midweek_review import (
    PoolMode,
    build_midweek_review,
    default_review_state,
    default_sort_mode,
    sort_review_rows,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Self-check breakout pool dashboard logic.")
    default_csv = Path(__file__).resolve().parents[1] / "us" / "breakout_follow_pool.csv"
    parser.add_argument(
        "--csv",
        default=str(default_csv),
        help="Path to breakout_follow_pool.csv",
    )
    parser.add_argument(
        "--midweek-csv",
        help="Optional path to breakout_follow_pool_midweek.csv.",
    )
    args = parser.parse_args()

    checks = [
        ("load and normalize", _check_load),
        ("formula recalculations (dist_to_52w & candidate_pct)", _check_formulas),
        ("four-state boundaries check", _check_four_state_boundaries),
        ("status conservation check", _check_status_conservation),
        ("advanced filters AND logic", _check_advanced_filters),
        ("sort specs", _check_sort_specs),
        ("chart: Route Quality aggregation", _check_route_quality_chart),
        ("chart: Trend Volume Map row source", _check_trend_volume_map_chart),
        ("review default ordering", _check_review_default_ordering),
    ]

    label = "setup"
    try:
        df = load_pool_csv(args.csv)
        for label, check in checks:
            check(df)
            print(f"[PASS] {label}")
        if args.midweek_csv:
            label = "midweek projection conservation"
            _check_midweek_projection(df, load_pool_csv(args.midweek_csv))
            print(f"[PASS] {label}")
    except Exception as exc:
        print(f"[FAIL] {label}: {exc}", file=sys.stderr)
        return 1
    return 0


def _check_midweek_projection(complete: pd.DataFrame, midweek: pd.DataFrame) -> None:
    result = build_midweek_review(midweek, complete)
    review = result.current_review
    assert len(review) == len(midweek), "review must preserve the current-pool row count"
    assert review["code"].tolist() == midweek["code"].tolist(), "review must preserve current-pool order"
    assert not review["code"].duplicated().any(), "review codes must remain unique"
    assert set(result.actionable_codes).issubset(set(review["code"])), "actionable codes must come from current pool"
    assert result.summary["CURRENT_POOL"] == len(review)
    assert result.summary["EXITED_POOL"] == len(result.exited_pool)
    assert result.summary["ACTIVE_SIGNALS"] == int(review["review_watch_active"].sum())


def _check_load(df: pd.DataFrame) -> None:
    required = {
        "code",
        "signal",
        "ibd_entry_valid",
        "signal_source",
        "ibd_entry_status",
        "latest_close",
        "current_vs_ibd_candidate_pct",
        "dist_to_52w_high_pct",
    }
    missing = required - set(df.columns)
    assert not missing, f"missing required columns: {sorted(missing)}"
    assert len(df) > 0, "CSV has no rows"


def _check_formulas(df: pd.DataFrame) -> None:
    mask_cand = (
        df["latest_close"].notna()
        & df["ibd_candidate_price"].notna()
        & df["ibd_candidate_price"].ne(0)
        & df["current_vs_ibd_candidate_pct"].notna()
    )
    if mask_cand.any():
        calc_cand = (
            df.loc[mask_cand, "latest_close"]
            / df.loc[mask_cand, "ibd_candidate_price"]
            - 1.0
        ) * 100.0
        diff_cand = (
            calc_cand - df.loc[mask_cand, "current_vs_ibd_candidate_pct"]
        ).abs()
        max_diff_cand = diff_cand.max()
        assert max_diff_cand <= 0.05, (
            "current_vs_ibd_candidate_pct formula discrepancy exceeds tolerance: "
            f"max diff = {max_diff_cand:.4f}"
        )

    mask_52w = (
        df["latest_close"].notna()
        & df["price_52_week_high"].notna()
        & df["price_52_week_high"].ne(0)
        & df["dist_to_52w_high_pct"].notna()
    )
    if mask_52w.any():
        calc_52w = (
            df.loc[mask_52w, "latest_close"]
            / df.loc[mask_52w, "price_52_week_high"]
            - 1.0
        ) * 100.0
        diff_52w = (calc_52w - df.loc[mask_52w, "dist_to_52w_high_pct"]).abs()
        max_diff_52w = diff_52w.max()
        assert max_diff_52w <= 0.05, (
            "dist_to_52w_high_pct formula discrepancy exceeds tolerance: "
            f"max diff = {max_diff_52w:.4f}"
        )


def _check_four_state_boundaries(df: pd.DataFrame) -> None:
    signal_mask = _true_mask(df["signal"])
    active_df = df[signal_mask].copy()
    if active_df.empty:
        return

    test_df = active_df[active_df["current_vs_ibd_candidate_pct"].notna()]
    for _, row in test_df.iterrows():
        pct = float(row["current_vs_ibd_candidate_pct"])
        status = row["ibd_entry_status"]
        valid = bool(row["ibd_entry_valid"]) if not pd.isna(row["ibd_entry_valid"]) else False

        if not valid:
            assert status == "UNCONFIRMED", (
                f"Row {row['code']}: when ibd_entry_valid is not True, "
                f"status must be UNCONFIRMED, got {status}"
            )
        elif pct < 0:
            assert status == "BELOW_TRIGGER", (
                f"Row {row['code']}: pct={pct} < 0 must be BELOW_TRIGGER, got {status}"
            )
        elif pct <= 5.0:
            assert status == "ACTIONABLE", (
                f"Row {row['code']}: pct={pct} in [0, 5.0] must be ACTIONABLE, got {status}"
            )
        else:
            assert status == "EXTENDED", (
                f"Row {row['code']}: pct={pct} > 5.0 must be EXTENDED, got {status}"
            )


def _check_status_conservation(df: pd.DataFrame) -> None:
    non_signal_mask = ~_true_mask(df["signal"])
    if non_signal_mask.any():
        non_signal_statuses = df.loc[non_signal_mask, "ibd_entry_status"].dropna()
        assert non_signal_statuses.empty, (
            "Non-signal rows should not have ibd_entry_status assigned, found "
            f"{len(non_signal_statuses)} rows with status"
        )

    signal_df = df[_true_mask(df["signal"])]
    statuses = ["ACTIONABLE", "UNCONFIRMED", "BELOW_TRIGGER", "EXTENDED"]
    vc = signal_df["ibd_entry_status"].value_counts(dropna=True)
    status_sum = sum(vc.get(status, 0) for status in statuses)
    na_count = signal_df["ibd_entry_status"].isna().sum()
    assert status_sum + na_count == len(signal_df), (
        f"Status conservation check failed: status_sum({status_sum}) + "
        f"na_count({na_count}) != total_active({len(signal_df)})"
    )


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
                & pd.to_numeric(
                    df["ibd_entry_breakout_range_ratio"], errors="coerce"
                ).between(1.0, 1.5, inclusive="both")
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
    valid_mask = _true_mask(
        df.get("ibd_entry_valid", pd.Series(index=df.index, dtype="object"))
    )
    expected = df[valid_mask].dropna(subset=["touched_ema10_count", "volume_ratio"])
    assert len(chart) == len(expected)
    assert set(chart["code"]) == set(expected["code"])
    if not chart.empty:
        assert all(chart["entry_status"] == "IBD valid")
    required = {
        "entry_status",
        "dry_status",
        "sector",
        "industry",
        "touched_ema10_count",
        "touched_ema10_jittered",
        "volume_ratio",
    }
    assert required.issubset(chart.columns)


def _check_review_default_ordering(_: pd.DataFrame) -> None:
    assert default_sort_mode("MIDWEEK", "CHANGES", has_comparison=True) == "Review Priority"
    assert default_sort_mode("MIDWEEK", "ALL_SIGNALS", has_comparison=True) == "Code"
    assert default_sort_mode("MIDWEEK", "ALL_SIGNALS", has_comparison=False) == "Code"
    assert default_sort_mode("WEEKEND", "ALL_SIGNALS", has_comparison=False) == "Code"

    assert default_review_state(PoolMode.MIDWEEK)["sort_mode"] == "Review Priority"
    assert default_review_state(PoolMode.COMPLETE)["sort_mode"] == "Code"
    assert default_review_state(PoolMode.MIDWEEK_WITHOUT_VALID_BASELINE)["sort_mode"] == "Code"

    rows = pd.DataFrame([{"code": "B"}, {"code": "A"}, {"code": "C"}])
    assert sort_review_rows(rows, "Code")["code"].tolist() == ["A", "B", "C"]


def _true_mask(series: pd.Series) -> pd.Series:
    return series.map(
        lambda value: False if pd.isna(value) else bool(value) is True
    ).astype(bool)


if __name__ == "__main__":
    raise SystemExit(main())
