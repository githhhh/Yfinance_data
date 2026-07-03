import warnings

import pandas as pd

from dashboard.data_utils import (
    FilterSpec,
    SortSpec,
    apply_c_rank_mode,
    apply_filters,
    apply_sort,
    normalize_pool_df,
    _false_mask,
    _true_mask,
)
from dashboard.field_config import EXCLUDED_CUSTOM_FIELDS


def sample_pool_df() -> pd.DataFrame:
    return normalize_pool_df(
        pd.DataFrame(
            [
                {
                    "code": "AAA",
                    "signal": "True",
                    "signal_source": "ceiling_breakout",
                    "ibd_candidate_rule": "ceiling_pullback",
                    "ibd_entry_valid": "1",
                    "ibd_entry_volume_ratio": "2.5",
                    "ibd_entry_close_vs_trigger_pct": "0.04",
                    "ibd_entry_close_position": "0.80",
                    "ibd_entry_breakout_range_ratio": "1.20",
                    "volume_ratio": "1.4",
                    "is_bullish": "True",
                    "pullback_pct": "-5.0",
                    "sector": "Technology Services",
                    "industry": "Software - Enterprise",
                    "pullback_v_is_dry": "True",
                    "breakout_date": "2026-05-10",
                    "pct_above_ceiling": "4.0",
                    "touched_ema10_count": "2",
                    "rank_C_continuous": "2",
                },
                {
                    "code": "BBB",
                    "signal": "True",
                    "signal_source": "pivot",
                    "ibd_candidate_rule": "pivot",
                    "ibd_entry_valid": "0",
                    "ibd_entry_volume_ratio": "3.5",
                    "ibd_entry_close_vs_trigger_pct": "0.02",
                    "ibd_entry_close_position": "0.40",
                    "ibd_entry_breakout_range_ratio": "0.60",
                    "volume_ratio": "1.6",
                    "is_bullish": "False",
                    "pullback_pct": "-12.0",
                    "sector": "Health Technology",
                    "industry": "Biotechnology",
                    "pullback_v_is_dry": "False",
                    "breakout_date": "2026-04-15",
                    "pct_above_ceiling": "8.0",
                    "touched_ema10_count": "5",
                    "rank_C_continuous": "1",
                },
                {
                    "code": "DDD",
                    "signal": "True",
                    "signal_source": "10_wk_ema_touch_confirm",
                    "ibd_candidate_rule": "ma10_touch_confirm",
                    "ibd_entry_valid": "1",
                    "ibd_entry_volume_ratio": "1.8",
                    "ibd_entry_close_vs_trigger_pct": "0.01",
                    "ibd_entry_close_position": "0.90",
                    "ibd_entry_breakout_range_ratio": "1.50",
                    "volume_ratio": "1.5",
                    "is_bullish": "True",
                    "pullback_pct": "-8.0",
                    "sector": "Technology Services",
                    "industry": "Software - Infrastructure",
                    "pullback_v_is_dry": "False",
                    "breakout_date": "2026-05-20",
                    "pct_above_ceiling": "12.0",
                    "touched_ema10_count": "3",
                    "rank_C_continuous": "4",
                },
                {
                    "code": "CCC",
                    "signal": "False",
                    "signal_source": "",
                    "ibd_candidate_rule": "",
                    "ibd_entry_valid": "",
                    "ibd_entry_volume_ratio": "",
                    "ibd_entry_close_vs_trigger_pct": "",
                    "ibd_entry_close_position": "",
                    "ibd_entry_breakout_range_ratio": "",
                    "volume_ratio": "0.9",
                    "is_bullish": "",
                    "pullback_pct": "",
                    "sector": "Finance",
                    "industry": "Regional Banks",
                    "pullback_v_is_dry": "",
                    "breakout_date": "2026-06-01",
                    "pct_above_ceiling": "2.0",
                    "touched_ema10_count": "1",
                    "rank_C_continuous": "3",
                },
            ]
        )
    )


def test_normalize_pool_df_converts_core_types():
    df = sample_pool_df()

    assert df.loc[0, "signal"] is True
    assert df.loc[1, "ibd_entry_valid"] is False
    assert pd.isna(df.loc[3, "ibd_entry_valid"])
    assert df.loc[0, "ibd_entry_volume_ratio"] == 2.5
    assert df.loc[0, "breakout_date"] == pd.Timestamp("2026-05-10")


def test_normalize_pool_df_adds_base_duration_weeks_from_ceiling_to_breakout_dates():
    df = normalize_pool_df(
        pd.DataFrame(
            [
                {"code": "AAA", "ceiling_date": "2026-01-01", "breakout_date": "2026-02-12"},
                {"code": "BBB", "ceiling_date": "2026-01-01", "breakout_date": ""},
            ]
        )
    )

    assert df.loc[0, "base_duration_weeks"] == 6
    assert pd.isna(df.loc[1, "base_duration_weeks"])


def test_all_enabled_filters_are_combined_with_and_logic():
    df = sample_pool_df()
    filters = [
        FilterSpec("signal", "is true"),
        FilterSpec("volume_ratio", ">=", 1.3),
        FilterSpec("pullback_v_is_dry", "is true"),
        FilterSpec("code", "contains", "A"),
        FilterSpec("signal_source", "in", ["ceiling_breakout", "pivot"], enabled=False),
    ]

    actual = apply_filters(df, filters)

    assert actual["code"].tolist() == ["AAA"]


def test_date_between_filter_uses_date_semantics():
    df = normalize_pool_df(
        pd.DataFrame(
            [
                {"code": "AAA", "breakout_date": "2026-05-15"},
                {"code": "BBB", "breakout_date": "2026-06-15"},
                {"code": "CCC", "breakout_date": ""},
            ]
        )
    )

    actual = apply_filters(df, [FilterSpec("breakout_date", "between", "2026-05-01", "2026-06-01")])

    assert actual["code"].tolist() == ["AAA"]


def test_invalid_numeric_filter_value_returns_no_rows_instead_of_raising():
    df = sample_pool_df()

    actual = apply_filters(df, [FilterSpec("volume_ratio", ">=", "abc")])

    assert actual.empty


def test_invalid_date_filter_value_returns_no_rows_instead_of_raising():
    df = sample_pool_df()

    actual = apply_filters(df, [FilterSpec("breakout_date", "between", "2026-05-01", "not-a-date")])

    assert actual.empty


def test_apply_sort_supports_three_stable_levels_and_nulls_last():
    df = sample_pool_df()
    sorted_df = apply_sort(
        df,
        [
            SortSpec("ibd_entry_valid", "desc"),
            SortSpec("ibd_entry_volume_ratio", "desc"),
            SortSpec("pct_above_ceiling", "asc"),
        ],
    )

    assert sorted_df["code"].tolist() == ["AAA", "DDD", "BBB", "CCC"]


def test_c_rank_mode_ignores_custom_filters_and_sorts_by_rank():
    df = sample_pool_df()
    custom_filters = [FilterSpec("code", "equals", "AAA")]
    custom_sort = [SortSpec("volume_ratio", "desc")]

    actual = apply_c_rank_mode(df, limit=None)

    assert actual["code"].tolist() == ["BBB", "AAA", "DDD"]


def test_boolean_masks_do_not_emit_pandas_downcasting_warning():
    series = pd.Series([True, False, None], dtype="object")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        true_mask = _true_mask(series)
        false_mask = _false_mask(series)

    assert true_mask.tolist() == [True, False, False]
    assert false_mask.tolist() == [False, True, False]
    assert not any(issubclass(item.category, FutureWarning) for item in caught)


def test_funnel_stage1_route_filtering():
    df = sample_pool_df()

    actual = apply_filters(df, [FilterSpec("ibd_candidate_rule", "in", ["pivot", "ma10_touch_confirm"])])

    assert set(actual["code"]) == {"BBB", "DDD"}


def test_funnel_stage2_entry_confirmation_and_strength_filtering():
    df = sample_pool_df()

    actual = apply_filters(
        df,
        [
            FilterSpec("ibd_entry_valid", "is true"),
            FilterSpec("ibd_entry_volume_ratio", ">=", 2.0),
            FilterSpec("ibd_entry_close_position", ">=", 0.75),
            FilterSpec("ibd_entry_breakout_range_ratio", "between", 1.0, 1.4),
        ],
    )

    assert actual["code"].tolist() == ["AAA"]


def test_funnel_stage3_weekly_volume_and_price_filtering():
    df = sample_pool_df()

    actual = apply_filters(
        df,
        [
            FilterSpec("volume_ratio", ">=", 1.4),
            FilterSpec("is_bullish", "is true"),
        ],
    )

    assert set(actual["code"]) == {"AAA", "DDD"}


def test_funnel_stage4_structure_filtering():
    df = sample_pool_df()

    actual = apply_filters(
        df,
        [
            FilterSpec("touched_ema10_count", ">=", 2),
            FilterSpec("pullback_pct", "between", -10.0, -3.0),
        ],
    )

    assert set(actual["code"]) == {"AAA", "DDD"}


def test_funnel_stage5_grouping_filtering():
    df = sample_pool_df()

    actual = apply_filters(
        df,
        [
            FilterSpec("sector", "in", ["Technology Services"]),
            FilterSpec("industry", "in", ["Software - Enterprise", "Software - Infrastructure"]),
        ],
    )

    assert set(actual["code"]) == {"AAA", "DDD"}


def test_funnel_full_decision_funnel_integration_and_logic():
    df = sample_pool_df()

    actual = apply_filters(
        df,
        [
            FilterSpec("ibd_candidate_rule", "in", ["ceiling_pullback", "ma10_touch_confirm"]),
            FilterSpec("ibd_entry_valid", "is true"),
            FilterSpec("ibd_entry_close_position", ">=", 0.80),
            FilterSpec("volume_ratio", ">=", 1.4),
            FilterSpec("is_bullish", "is true"),
            FilterSpec("touched_ema10_count", ">=", 2),
            FilterSpec("sector", "in", ["Technology Services"]),
        ],
    )

    assert set(actual["code"]) == {"AAA", "DDD"}
    assert all(actual["ibd_entry_valid"] == True)
