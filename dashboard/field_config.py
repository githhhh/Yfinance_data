from __future__ import annotations

from collections import OrderedDict


EXCLUDED_CUSTOM_FIELDS = {"C_continuous", "rank_C_continuous", "is_priority"}

BOOLEAN_FIELDS = {
    "signal",
    "pullback_v_is_dry",
    "ibd_entry_valid",
    "is_bullish",
    "is_priority",
}

DATE_FIELDS = {
    "snapshot_date",
    "ibd_entry_date",
    "breakout_date",
    "ceiling_date",
}

NUMBER_FIELDS = {
    "ibd_candidate_price",
    "ibd_entry_price",
    "ibd_trigger_price",
    "ibd_entry_volume_ratio",
    "ibd_entry_close_vs_trigger_pct",
    "ibd_entry_close_position",
    "ibd_entry_breakout_range_ratio",
    "volume_ratio",
    "hold_return",
    "pct_above_ceiling",
    "touched_ema10_count",
    "mbox_count",
    "ceiling",
    "base_duration_weeks",
    "base_depth_pct",
    "base_mbox_count",
    "base_depth_abs",
    "C_continuous",
    "rank_C_continuous",
    "pullback_count",
    "pullback_pct",
    "pullback_pct_off_peak",
    "eps_yoy_growth",
    "price_52_week_high",
    "dist_to_52w_high_pct",
    "latest_close",
    "current_vs_ibd_candidate_pct",
}

FILTER_FUNNEL_GROUPS = OrderedDict(
    [
        ("Route", ["ibd_candidate_rule"]),
        ("Entry Status", ["ibd_entry_status"]),
        (
            "Optional Quality Filters",
            ["current_vs_ibd_candidate_pct", "ibd_entry_volume_ratio", "volume_ratio"],
        ),
    ]
)

ALL_TABLE_COLUMNS = [
    "code",
    "snapshot_date",
    "sector",
    "industry",
    "eps_yoy_growth",
    "price_52_week_high",
    "dist_to_52w_high_pct",
    "signal",
    "signal_source",
    "ibd_candidate_rule",
    "ibd_candidate_signal_source",
    "breakout_date",
    "ibd_candidate_price",
    "ibd_trigger_price",
    "ibd_entry_valid",
    "ibd_entry_date",
    "ibd_entry_price",
    "ibd_entry_volume_ratio",
    "ibd_entry_close_vs_trigger_pct",
    "ibd_entry_close_position",
    "ibd_entry_breakout_range_ratio",
    "ibd_entry_rule",
    "ibd_entry_reject_reason",
    "ibd_candidate_extra",
    "ceiling",
    "ceiling_date",
    "base_duration_weeks",
    "pct_above_ceiling",
    "base_depth_abs",
    "base_depth_pct",
    "base_mbox_count",
    "mbox_count",
    "touched_ema10_count",
    "volume_ratio",
    "is_bullish",
    "pullback_count",
    "pullback_pct",
    "pullback_pct_off_peak",
    "pullback_v_is_dry",
    "ibd_entry_status",
    "latest_close",
    "current_vs_ibd_candidate_pct",
    "C_continuous",
    "rank_C_continuous",
    "is_priority",
]

IBD_DECISION_COLUMNS = [
    "code",
    "snapshot_date",
    "ibd_candidate_rule",
    "ibd_entry_status",
    "latest_close",
    "ibd_candidate_price",
    "current_vs_ibd_candidate_pct",
    "ibd_entry_date",
    "ibd_entry_price",
    "ibd_entry_volume_ratio",
    "ibd_entry_reject_reason",
    "volume_ratio",
    "eps_yoy_growth",
    "price_52_week_high",
    "dist_to_52w_high_pct",
    "rank_C_continuous",
]

DEFAULT_TABLE_COLUMNS = IBD_DECISION_COLUMNS

SIGNAL_COLUMNS = [
    "code",
    "snapshot_date",
    "signal",
    "signal_source",
    "ibd_candidate_rule",
    "ibd_candidate_signal_source",
    "breakout_date",
]

IBD_COLUMNS = [
    "code",
    "ibd_candidate_price",
    "ibd_trigger_price",
    "ibd_candidate_extra",
    "ibd_entry_valid",
    "ibd_entry_date",
    "ibd_entry_price",
    "ibd_entry_volume_ratio",
    "ibd_entry_close_vs_trigger_pct",
    "ibd_entry_close_position",
    "ibd_entry_breakout_range_ratio",
    "ibd_entry_rule",
    "ibd_entry_reject_reason",
    "ibd_entry_status",
    "latest_close",
    "current_vs_ibd_candidate_pct",
]

VOLUME_PULLBACK_COLUMNS = [
    "code",
    "volume_ratio",
    "is_bullish",
    "pullback_count",
    "pullback_pct",
    "pullback_pct_off_peak",
    "pullback_v_is_dry",
    "hold_return",
]

REFERENCE_COLUMNS = [
    "code",
    "C_continuous",
    "rank_C_continuous",
    "is_priority",
]

LONG_FIELDS = {"ibd_entry_reject_reason", "ibd_candidate_extra"}


def _field(
    label: str,
    field_type: str,
    group: str,
    *,
    filterable: bool = True,
    sortable: bool = True,
    default_table: bool = False,
    custom_mode: bool = True,
    c_rank_mode: bool = False,
    advanced_filter: bool = True,
    fmt: str | None = None,
    help_text: str = "",
) -> dict[str, object]:
    return {
        "label": label,
        "type": field_type,
        "group": group,
        "filterable": filterable,
        "sortable": sortable,
        "default_table": default_table,
        "custom_mode": custom_mode,
        "c_rank_mode": c_rank_mode,
        "advanced_filter": advanced_filter,
        "format": fmt,
        "help": help_text,
    }


FIELD_CONFIG = OrderedDict(
    [
        ("code", _field("Code", "text", "Identity", sortable=True, default_table=True)),
        ("snapshot_date", _field("Snapshot Date", "date", "Identity")),
        ("signal", _field("Signal", "boolean", "Signal")),
        ("signal_source", _field("Signal Source", "category", "Signal", default_table=True)),
        ("pullback_v_is_dry", _field("Pullback V Is Dry", "boolean", "Risk / Structure", default_table=True)),
        ("ibd_candidate_rule", _field("IBD Candidate Rule", "category", "Candidate", default_table=True)),
        ("ibd_candidate_price", _field("IBD Candidate Price", "number", "Candidate", default_table=True, fmt="0.00")),
        ("ibd_candidate_signal_source", _field("IBD Candidate Signal Source", "category", "Candidate")),
        (
            "ibd_candidate_extra",
            _field(
                "IBD Candidate Extra",
                "text",
                "Candidate",
                filterable=False,
                sortable=False,
                default_table=False,
                advanced_filter=False,
            ),
        ),
        ("ibd_entry_valid", _field("IBD Entry Valid", "boolean", "IBD Entry", default_table=True)),
        ("ibd_entry_date", _field("IBD Entry Date", "date", "IBD Entry", default_table=True)),
        ("ibd_entry_price", _field("IBD Entry Price", "number", "IBD Entry", default_table=True, fmt="0.00")),
        ("ibd_trigger_price", _field("IBD Trigger Price", "number", "IBD Entry", fmt="0.00")),
        (
            "ibd_entry_volume_ratio",
            _field(
                "IBD Entry Volume Ratio",
                "number",
                "IBD Entry",
                default_table=True,
                fmt="0.00x",
                help_text="Breakout-day volume divided by recent average volume.",
            ),
        ),
        (
            "ibd_entry_close_vs_trigger_pct",
            _field(
                "Close vs Trigger",
                "number",
                "IBD Entry",
                default_table=True,
                fmt="0.00%",
                help_text="Close confirmation quality versus trigger price.",
            ),
        ),
        (
            "ibd_entry_close_position",
            _field(
                "Close Position",
                "number",
                "IBD Entry",
                default_table=True,
                fmt="0.00",
                help_text="Relative close position within daily high-low range (0 to 1).",
            ),
        ),
        (
            "ibd_entry_breakout_range_ratio",
            _field(
                "Breakout Range Ratio",
                "number",
                "IBD Entry",
                default_table=True,
                fmt="0.00x",
                help_text="Breakout bar price range relative to typical volatility.",
            ),
        ),
        (
            "ibd_entry_status",
            _field(
                "IBD Entry Status",
                "category",
                "IBD Entry",
                filterable=True,
                default_table=True,
                advanced_filter=True,
                help_text="Lifecycle state of IBD breakout review.",
            ),
        ),
        (
            "latest_close",
            _field(
                "Latest Close",
                "number",
                "IBD Entry",
                filterable=False,
                default_table=True,
                advanced_filter=False,
                fmt="0.00",
                help_text="Latest weekly close price.",
            ),
        ),
        (
            "current_vs_ibd_candidate_pct",
            _field(
                "% vs IBD Candidate",
                "number",
                "IBD Entry",
                filterable=True,
                default_table=True,
                advanced_filter=True,
                fmt="0.00%",
                help_text="Current close relative to IBD candidate price.",
            ),
        ),
        ("ibd_entry_rule", _field("IBD Entry Rule", "category", "IBD Entry")),
        (
            "ibd_entry_reject_reason",
            _field(
                "IBD Entry Reject Reason",
                "text",
                "IBD Entry",
                filterable=False,
                sortable=False,
                default_table=False,
                advanced_filter=False,
            ),
        ),
        (
            "volume_ratio",
            _field(
                "Volume Ratio",
                "number",
                "Volume/Pullback",
                filterable=True,
                default_table=True,
                advanced_filter=True,
                fmt="0.00x",
            ),
        ),
        ("hold_return", _field("Hold Return", "number", "Volume/Pullback", fmt="0.0%")),
        (
            "breakout_date",
            _field("Breakout Date", "date", "Signal", filterable=True, default_table=True, advanced_filter=True),
        ),
        ("pct_above_ceiling", _field("Pct Above Ceiling", "number", "Risk / Structure", default_table=True, fmt="0.0%")),
        ("touched_ema10_count", _field("Touched EMA10 Count", "number", "Risk / Structure", default_table=True)),
        ("mbox_count", _field("M Box Count", "number", "Risk / Structure")),
        ("ceiling", _field("Ceiling", "number", "Risk / Structure", default_table=True, fmt="0.00")),
        ("ceiling_date", _field("Ceiling Date", "date", "Risk / Structure")),
        ("base_duration_weeks", _field("Base Duration Weeks", "number", "Risk / Structure")),
        ("base_depth_pct", _field("Base Depth Pct", "number", "Risk / Structure", fmt="0.0%")),
        ("base_mbox_count", _field("Base M Box Count", "number", "Risk / Structure")),
        ("base_depth_abs", _field("Base Depth Abs", "number", "Risk / Structure")),
        (
            "C_continuous",
            _field(
                "Continuous C",
                "number",
                "C Rank",
                custom_mode=False,
                c_rank_mode=True,
                advanced_filter=False,
            ),
        ),
        (
            "rank_C_continuous",
            _field(
                "C Rank",
                "number",
                "C Rank",
                default_table=True,
                custom_mode=False,
                c_rank_mode=True,
                advanced_filter=False,
            ),
        ),
        ("pullback_count", _field("Pullback Count", "number", "Risk / Structure", default_table=True)),
        ("pullback_pct", _field("Pullback Pct", "number", "Risk / Structure", fmt="0.0%")),
        ("pullback_pct_off_peak", _field("Pullback Pct Off Peak", "number", "Risk / Structure", default_table=True, fmt="0.0%")),
        ("pullback_v_is_dry", _field("Pullback V Is Dry", "boolean", "Risk / Structure", default_table=True)),
        ("is_bullish", _field("Is Bullish", "boolean", "Risk / Structure")),
        (
            "is_priority",
            _field(
                "Is Priority",
                "boolean",
                "C Rank",
                custom_mode=False,
                c_rank_mode=True,
                advanced_filter=False,
            ),
        ),
        ("sector", _field("Sector", "category", "Grouping", default_table=True)),
        ("industry", _field("Industry", "category", "Grouping", default_table=True)),
        ("eps_yoy_growth", _field("EPS YoY Growth", "number", "Grouping", default_table=True, fmt="0.0%")),
        ("price_52_week_high", _field("Price 52 Week High", "number", "Grouping", default_table=True, fmt="0.00")),
        ("dist_to_52w_high_pct", _field("Distance To 52W High", "number", "Grouping", default_table=True, fmt="0.0%")),
    ]
)


def _allowed(field: str) -> bool:
    config = FIELD_CONFIG[field]
    return bool(config.get("custom_mode")) and field not in EXCLUDED_CUSTOM_FIELDS


def get_custom_mode_fields() -> list[str]:
    return [field for field in FIELD_CONFIG if _allowed(field) and field not in LONG_FIELDS]


def get_filterable_fields() -> list[str]:
    fields: list[str] = []
    for group_fields in FILTER_FUNNEL_GROUPS.values():
        fields.extend(
            field
            for field in group_fields
            if field in FIELD_CONFIG and FIELD_CONFIG[field].get("filterable") and FIELD_CONFIG[field].get("advanced_filter")
        )
    return fields


def get_sortable_fields() -> list[str]:
    return [field for field in get_custom_mode_fields() if FIELD_CONFIG[field].get("sortable")]


def get_default_table_columns() -> list[str]:
    return [field for field in IBD_DECISION_COLUMNS if field in FIELD_CONFIG]


def get_column_view_fields(view_name: str) -> list[str]:
    if view_name == "IBD Decision":
        return [field for field in IBD_DECISION_COLUMNS if field in FIELD_CONFIG]
    if view_name == "All Fields":
        return get_all_table_columns()
    if view_name == "Signal":
        return [field for field in SIGNAL_COLUMNS if field in FIELD_CONFIG]
    if view_name == "IBD Entry":
        return [field for field in IBD_COLUMNS if field in FIELD_CONFIG]
    if view_name == "Volume/Pullback":
        return [field for field in VOLUME_PULLBACK_COLUMNS if field in FIELD_CONFIG]
    if view_name == "Reference":
        return [field for field in REFERENCE_COLUMNS if field in FIELD_CONFIG]
    raise ValueError(f"Unknown column view: {view_name}")


def get_filter_funnel_groups() -> OrderedDict[str, list[str]]:
    return OrderedDict(
        (group, [field for field in fields if field in FIELD_CONFIG])
        for group, fields in FILTER_FUNNEL_GROUPS.items()
    )


def get_all_table_columns() -> list[str]:
    return [field for field in ALL_TABLE_COLUMNS if field in FIELD_CONFIG]


def get_field_label(field: str) -> str:
    return str(FIELD_CONFIG.get(field, {}).get("label", field))
