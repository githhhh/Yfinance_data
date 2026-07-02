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
}

FILTER_FUNNEL_GROUPS = OrderedDict(
    [
        ("Route", ["ibd_candidate_rule"]),
        (
            "Entry Confirmation & Strength",
            ["ibd_entry_valid", "ibd_entry_volume_ratio", "ibd_entry_close_vs_trigger_pct"],
        ),
        ("Weekly Volume & Price", ["volume_ratio", "is_bullish"]),
        ("Structure", ["touched_ema10_count", "pullback_pct"]),
        ("Grouping", ["sector", "industry"]),
    ]
)

ALL_TABLE_COLUMNS = [
    "code",
    "snapshot_date",
    "sector",
    "industry",
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
    "hold_return",
    "C_continuous",
    "rank_C_continuous",
    "is_priority",
]

DEFAULT_TABLE_COLUMNS = ALL_TABLE_COLUMNS

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
    "ibd_entry_rule",
    "ibd_entry_reject_reason",
]

STRUCTURE_COLUMNS = [
    "code",
    "ceiling",
    "ceiling_date",
    "base_duration_weeks",
    "pct_above_ceiling",
    "base_depth_abs",
    "base_depth_pct",
    "base_mbox_count",
    "mbox_count",
    "touched_ema10_count",
    "pullback_count",
    "pullback_pct",
    "pullback_pct_off_peak",
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

GROUPING_COLUMNS = [
    "code",
    "sector",
    "industry",
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
        ("volume_ratio", _field("Volume Ratio", "number", "Risk / Structure", default_table=True, fmt="0.00x")),
        ("hold_return", _field("Hold Return", "number", "Result", default_table=True, fmt="0.0%")),
        ("breakout_date", _field("Breakout Date", "date", "Signal", default_table=True)),
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
                "C Continuous",
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
                "Rank C Continuous",
                "number",
                "C Rank",
                custom_mode=False,
                c_rank_mode=True,
                advanced_filter=False,
            ),
        ),
        ("pullback_count", _field("Pullback Count", "number", "Risk / Structure", default_table=True)),
        ("pullback_pct", _field("Pullback Pct", "number", "Risk / Structure", fmt="0.0%")),
        (
            "pullback_pct_off_peak",
            _field("Pullback Pct Off Peak", "number", "Risk / Structure", default_table=True, fmt="0.0%"),
        ),
        ("is_bullish", _field("Is Bullish", "boolean", "Signal")),
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
    ]
)

PRESETS = OrderedDict(
    [
        (
            "active_signal_quality",
            {
                "label": "Review: All Signals",
                "filters": [{"field": "signal", "operator": "is true"}],
                "sort": [
                    {"field": "ibd_entry_valid", "direction": "desc"},
                    {"field": "ibd_entry_volume_ratio", "direction": "desc"},
                    {"field": "ibd_entry_close_vs_trigger_pct", "direction": "desc"},
                ],
            },
        ),
        (
            "ibd_valid_breakout",
            {
                "label": "IBD Valid Breakout",
                "filters": [
                    {"field": "signal", "operator": "is true"},
                    {"field": "ibd_entry_valid", "operator": "is true"},
                ],
                "sort": [
                    {"field": "ibd_entry_volume_ratio", "direction": "desc"},
                    {"field": "ibd_entry_close_vs_trigger_pct", "direction": "desc"},
                ],
            },
        ),
        (
            "action_clean_entry",
            {
                "label": "Action: Clean Entry",
                "filters": [
                    {"field": "signal", "operator": "is true"},
                    {"field": "ibd_entry_valid", "operator": "is true"},
                    {"field": "ibd_entry_volume_ratio", "operator": ">=", "value": 1.5},
                    {"field": "ibd_entry_close_vs_trigger_pct", "operator": "between", "value": 0.0, "value2": 0.05},
                    {"field": "pct_above_ceiling", "operator": "<=", "value": 10.0},
                ],
                "sort": [
                    {"field": "pct_above_ceiling", "direction": "asc"},
                    {"field": "ibd_entry_volume_ratio", "direction": "desc"},
                    {"field": "ibd_entry_close_vs_trigger_pct", "direction": "asc"},
                ],
            },
        ),
        (
            "ceiling_breakout",
            {
                "label": "Review: Ceiling Breakout",
                "filters": [
                    {"field": "signal", "operator": "is true"},
                    {"field": "signal_source", "operator": "equals", "value": "ceiling_breakout"},
                    {"field": "ibd_candidate_rule", "operator": "equals", "value": "ceiling"},
                ],
                "sort": [
                    {"field": "pct_above_ceiling", "direction": "asc"},
                    {"field": "volume_ratio", "direction": "desc"},
                ],
            },
        ),
        (
            "ceiling_pullback",
            {
                "label": "Review: Ceiling Pullback",
                "filters": [
                    {"field": "signal", "operator": "is true"},
                    {"field": "signal_source", "operator": "equals", "value": "ceiling_breakout"},
                    {"field": "ibd_candidate_rule", "operator": "equals", "value": "ceiling_pullback"},
                ],
                "sort": [{"field": "pct_above_ceiling", "direction": "asc"}],
            },
        ),
        (
            "pivot_quality",
            {
                "label": "Review: Pivot",
                "filters": [
                    {"field": "signal", "operator": "is true"},
                    {"field": "signal_source", "operator": "equals", "value": "pivot"},
                    {"field": "ibd_candidate_rule", "operator": "equals", "value": "pivot"},
                ],
                "sort": [
                    {"field": "ibd_entry_valid", "direction": "desc"},
                    {"field": "volume_ratio", "direction": "desc"},
                    {"field": "ibd_entry_close_vs_trigger_pct", "direction": "desc"},
                ],
            },
        ),
        (
            "ma_touch_count",
            {
                "label": "Review: 10W EMA Touch",
                "filters": [
                    {"field": "signal", "operator": "is true"},
                    {"field": "signal_source", "operator": "equals", "value": "10_wk_ema_touch_confirm"},
                    {"field": "ibd_candidate_rule", "operator": "equals", "value": "ma10_touch_confirm"},
                ],
                "sort": [
                    {"field": "ibd_entry_valid", "direction": "desc"},
                    {"field": "touched_ema10_count", "direction": "desc"},
                    {"field": "volume_ratio", "direction": "desc"},
                ],
            },
        ),
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
    return get_all_table_columns()


def get_column_view_fields(view_name: str) -> list[str]:
    if view_name == "All Fields":
        return get_all_table_columns()
    if view_name == "Signal":
        return [field for field in SIGNAL_COLUMNS if field in FIELD_CONFIG]
    if view_name == "IBD":
        return [field for field in IBD_COLUMNS if field in FIELD_CONFIG]
    if view_name == "IBD Entry":
        return [field for field in IBD_COLUMNS if field in FIELD_CONFIG]
    if view_name == "Structure":
        return [field for field in STRUCTURE_COLUMNS if field in FIELD_CONFIG]
    if view_name == "Volume/Pullback":
        return [field for field in VOLUME_PULLBACK_COLUMNS if field in FIELD_CONFIG]
    if view_name == "Grouping":
        return [field for field in GROUPING_COLUMNS if field in FIELD_CONFIG]
    if view_name == "Reference":
        return [field for field in REFERENCE_COLUMNS if field in FIELD_CONFIG]
    if view_name == "Full Custom":
        return get_custom_mode_fields()
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


def get_field_type(field: str) -> str:
    return str(FIELD_CONFIG.get(field, {}).get("type", "text"))


def get_preset_options() -> list[tuple[str, str]]:
    return [(key, str(value["label"])) for key, value in PRESETS.items()]
