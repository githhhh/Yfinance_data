import pandas as pd

from dashboard.data_utils import (
    FilterSpec,
    SortSpec,
    apply_c_rank_mode,
    apply_filters,
    apply_sort,
    build_preset_filters,
    build_preset_sort,
    combine_filter_specs,
    normalize_pool_df,
)
from dashboard.field_config import EXCLUDED_CUSTOM_FIELDS, PRESETS, get_preset_options


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
                    "volume_ratio": "1.4",
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
                    "volume_ratio": "1.6",
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
                    "volume_ratio": "1.5",
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
                    "volume_ratio": "0.9",
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


def test_combine_filter_specs_keeps_preset_filters_as_actual_base_filters():
    preset_filters = [FilterSpec("ibd_entry_volume_ratio", ">=", 99)]
    ui_filters = [FilterSpec("ibd_entry_volume_ratio", "between", 1.5, 10.0), FilterSpec("signal", "is true")]

    combined = combine_filter_specs(preset_filters, ui_filters)

    assert combined == [
        FilterSpec("ibd_entry_volume_ratio", ">=", 99),
        FilterSpec("ibd_entry_volume_ratio", "between", 1.5, 10.0),
        FilterSpec("signal", "is true"),
    ]


def test_default_preset_reviews_all_active_signals_before_action_lists():
    assert get_preset_options()[0] == ("active_signal_quality", "Review: All Signals")

    df = sample_pool_df()
    actual = apply_sort(
        apply_filters(df, build_preset_filters("active_signal_quality")),
        build_preset_sort("active_signal_quality"),
    )

    assert set(actual["code"]) == {"AAA", "BBB", "DDD"}
    assert actual.iloc[0]["ibd_entry_valid"] is True


def test_action_clean_entry_preset_filters_confirmed_non_extended_entries():
    df = sample_pool_df()

    actual = apply_filters(df, build_preset_filters("action_clean_entry"))

    assert actual["code"].tolist() == ["AAA"]


def test_preset_filters_keep_ceiling_pullback_as_candidate_rule():
    df = sample_pool_df()

    actual = apply_filters(df, build_preset_filters("ceiling_pullback"))

    assert actual["code"].tolist() == ["AAA"]


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


def test_preset_sort_for_ibd_valid_breakout_uses_strength_then_confirmation():
    sort_specs = build_preset_sort("ibd_valid_breakout")

    assert sort_specs == [
        SortSpec("ibd_entry_volume_ratio", "desc"),
        SortSpec("ibd_entry_close_vs_trigger_pct", "desc"),
    ]


def test_ma_touch_count_is_a_true_ma10_touch_review_preset():
    df = sample_pool_df()

    actual = apply_filters(df, build_preset_filters("ma_touch_count"))

    assert actual["code"].tolist() == ["DDD"]


def test_all_presets_keep_c_rank_reference_fields_isolated():
    for preset in PRESETS.values():
        fields = {spec["field"] for spec in preset["filters"] + preset["sort"]}
        assert not fields.intersection(EXCLUDED_CUSTOM_FIELDS)


def test_c_rank_mode_ignores_custom_filters_and_sorts_by_rank():
    df = sample_pool_df()
    custom_filters = [FilterSpec("code", "equals", "AAA")]
    custom_sort = [SortSpec("volume_ratio", "desc")]

    actual = apply_c_rank_mode(df, limit=None, filters=custom_filters, sort_specs=custom_sort)

    assert actual["code"].tolist() == ["BBB", "AAA", "DDD"]
