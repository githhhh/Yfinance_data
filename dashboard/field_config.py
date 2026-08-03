from __future__ import annotations

from collections import OrderedDict


def _tooltip_meta(definition: str, count_basis: str, click_effect: str) -> dict[str, str]:
    return {
        "definition": definition,
        "count_basis": count_basis,
        "click_effect": click_effect,
        "tooltip": "\n".join([definition, count_basis, click_effect]),
    }


STATUS_META = {
    "ACTIONABLE": {
        "label": "ACTIONABLE",
        "subtitle": "买点上方 0%–5%",
        "tone": "green",
        "color": "#35df65",
        **_tooltip_meta(
            "含义：已完成入场确认，当前价位于买点上方 0%–5%。",
            "数量：当前范围内符合条件的标的数。",
            "点击：只看这类标的，并保留其他已选条件。",
        ),
    },
    "UNCONFIRMED": {
        "label": "UNCONFIRMED",
        "subtitle": "等待日线确认",
        "tone": "yellow",
        "color": "#ffd21f",
        **_tooltip_meta(
            "含义：尚未满足日线入场确认条件。",
            "数量：当前范围内符合条件的标的数。",
            "点击：只看这类标的，并保留其他已选条件。",
        ),
    },
    "BELOW_TRIGGER": {
        "label": "BELOW TRIGGER",
        "subtitle": "低于买点",
        "tone": "red",
        "color": "#f04444",
        **_tooltip_meta(
            "含义：当前价低于有效买点。",
            "数量：当前范围内符合条件的标的数。",
            "点击：只看这类标的，并保留其他已选条件。",
        ),
    },
    "EXTENDED": {
        "label": "EXTENDED",
        "subtitle": "高于买点 5%，不追",
        "tone": "blue",
        "color": "#2791ff",
        **_tooltip_meta(
            "含义：当前价已超过买点 5%，不宜追高。",
            "数量：当前范围内符合条件的标的数。",
            "点击：只看这类标的，并保留其他已选条件。",
        ),
    },
}


FLOW_CARD_META = {
    "BECAME_ACTIONABLE": {
        "label": "进入买区",
        "color": "#22c55e",
        **_tooltip_meta(
            "含义：上周不在买区，本次进入 0%–5% 买区。",
            "数量：当前范围内符合条件的标的数。",
            "点击：只看这类标的，并保留其他已选条件。",
        ),
    },
    "LEFT_ACTIONABLE": {
        "label": "离开买区",
        "color": "#ef5350",
        **_tooltip_meta(
            "含义：上周在买区，本次已跌破、未确认或涨超 5%。",
            "数量：当前范围内符合条件的标的数。",
            "点击：只看这类标的，并保留其他已选条件。",
        ),
    },
    "OTHER_CHANGES": {
        "label": "其他变化",
        "color": "#2dd4bf",
        **_tooltip_meta(
            "含义：状态和上周不同，但不属于进入或离开买区。",
            "数量：当前范围内符合条件的标的数。",
            "点击：只看这类标的，并保留其他已选条件。",
        ),
    },
    "NEW": {
        "label": "新信号",
        "color": "#22d3ee",
        **_tooltip_meta(
            "含义：完整周没有信号，周中首次出现信号。",
            "数量：当前范围内符合条件的标的数。",
            "点击：只看这类标的，并保留其他已选条件。",
        ),
    },
    "CARRY": {
        "label": "延续",
        "color": "#94a3b8",
        **_tooltip_meta(
            "含义：周中没有新信号，但完整周信号仍在观察，状态按当前价格更新。",
            "数量：当前范围内符合条件的标的数。",
            "点击：只看这类标的，并保留其他已选条件。",
        ),
    },
    "RECONFIRMED": {
        "label": "再确认",
        "color": "#93c5fd",
        **_tooltip_meta(
            "含义：完整周和周中都有信号，以周中数据为准。",
            "数量：当前范围内符合条件的标的数。",
            "点击：只看这类标的，并保留其他已选条件。",
        ),
    },
}

QUALITY_META = {
    "Powerful Breakout": {
        "label": "Powerful Breakout",
        "color": "#86efac",
        "borderColor": "#22c55e",
        "borderWidth": "5px",
        "backgroundImage": "linear-gradient(90deg, rgba(34, 197, 94, 0.26), rgba(34, 197, 94, 0.11))",
        "fontWeight": "700",
        "rule": "High close + full trigger clearance",
    },
    "Strong Breakout": {
        "label": "Strong Breakout",
        "color": "#4ade80",
        "borderColor": "rgba(34, 197, 94, 0.78)",
        "borderWidth": "4px",
        "backgroundImage": "linear-gradient(90deg, rgba(34, 197, 94, 0.18), rgba(34, 197, 94, 0.07))",
        "fontWeight": "700",
        "rule": "One dimension strongest, the other solid",
    },
    "Constructive Breakout": {
        "label": "Constructive Breakout",
        "color": "#4ade80",
        "borderColor": "rgba(34, 197, 94, 0.85)",
        "borderWidth": "3px",
        "backgroundImage": "linear-gradient(90deg, rgba(34, 197, 94, 0.10), rgba(34, 197, 94, 0.03))",
        "fontWeight": "600",
        "rule": "Mixed but valid price action",
    },
    "Marginal Breakout": {
        "label": "Marginal Breakout",
        "color": "#86c99d",
        "borderColor": "rgba(134, 239, 172, 0.42)",
        "borderWidth": "2px",
        "backgroundImage": "linear-gradient(90deg, rgba(34, 197, 94, 0.025), rgba(34, 197, 94, 0.005))",
        "fontWeight": "500",
        "rule": "Valid, but close and clearance are both thin",
    },
    "Weak Close": {
        "label": "Weak Close",
        "color": "#9eaaa2",
        "borderColor": "rgba(134, 239, 172, 0.20)",
        "borderWidth": "1px",
        "backgroundImage": "none",
        "fontWeight": "400",
        "rule": "Low close",
    },
}

QUALITY_BY_MATRIX_SCORE = {
    4: "Powerful Breakout",
    3: "Strong Breakout",
    2: "Constructive Breakout",
    1: "Marginal Breakout",
}
QUALITY_ORDER = {quality: rank for rank, quality in enumerate(QUALITY_META)}
QUALITY_ALIASES = {
    "Strong Close": "Strong Breakout",
    "Constructive Close (Tight)": "Constructive Breakout",
    "Constructive Close (High Close / Thin Thrust)": "Constructive Breakout",
    "High Close, Small Breakout": "Constructive Breakout",
    "Constructive Close": "Marginal Breakout",
}
QUALITY_ORDER.update({alias: QUALITY_ORDER[current] for alias, current in QUALITY_ALIASES.items()})

EXCLUDED_CUSTOM_FIELDS = {"C_continuous", "rank_C_continuous", "is_priority"}

BOOLEAN_FIELDS = {
    "signal",
    "pullback_v_is_dry",
    "ibd_entry_valid",
    "is_bullish",
    "is_priority",
    "review_watch_active",
    "review_futu_actionable",
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
    "pullback_duration_weeks",
    "pullback_pct",
    "pullback_pct_off_peak",
    "eps_yoy_growth",
    "price_52_week_high",
    "dist_to_52w_high_pct",
    "latest_close",
    "current_vs_ibd_candidate_pct",
    "review_candidate_price",
    "review_current_vs_candidate_pct",
    "review_priority",
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
    "ibd_entry_vol_or_reject",
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
    "pullback_duration_weeks",
    "pullback_pct",
    "pullback_pct_off_peak",
    "pullback_v_is_dry",
    "ibd_entry_status",
    "latest_close",
    "current_vs_ibd_candidate_pct",
    "ibd_breakout_quality",
    "C_continuous",
    "rank_C_continuous",
    "is_priority",
]

IBD_DECISION_COLUMNS = [
    "code",
    "ibd_entry_status",
    "ibd_candidate_rule",
    "current_vs_ibd_candidate_pct",
    "ibd_breakout_quality",
    "latest_close",
    "ibd_entry_vol_or_reject",
    "volume_ratio",
    "rank_C_continuous",
]

C_RANK_REFERENCE_COLUMNS = [
    "code",
    "rank_C_continuous",
    "C_continuous",
    "ibd_entry_status",
    "current_vs_ibd_candidate_pct",
    "ibd_candidate_rule",
    "volume_ratio",
    "latest_close",
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
    "ibd_entry_vol_or_reject",
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
    "pullback_duration_weeks",
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
        ("code", _field("Code", "text", "Identity", sortable=True, default_table=True, help_text="点击 Code 复制单个代码；点击该行其他位置查看详情。")),
        (
            "review_change_label",
            _field(
                "Change",
                "category",
                "Review",
                filterable=True,
                sortable=True,
                default_table=False,
                help_text="Signal origin and effective entry-status transition for this Midweek Review row.",
            ),
        ),
        ("review_signal_origin", _field("Origin", "category", "Review", default_table=False)),
        ("review_change_group", _field("Change Group", "category", "Review", default_table=False)),
        ("review_priority", _field("Review Priority", "number", "Review", default_table=False)),
        ("snapshot_date", _field("Snapshot Date", "date", "Identity")),
        ("signal", _field("Signal", "boolean", "Signal")),
        ("signal_source", _field("Signal Source", "category", "Signal", default_table=True)),
        ("pullback_v_is_dry", _field("Pullback V Is Dry", "boolean", "Risk / Structure", default_table=True)),
        ("ibd_candidate_rule", _field("Route", "category", "Candidate", default_table=True, help_text="IBD Candidate 触发价的结构来源。")),
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
                help_text="Close-to-trigger distance as a proportion of the day's high-low range.",
            ),
        ),
        (
            "ibd_entry_vol_or_reject",
            _field(
                "Entry / Reason",
                "text",
                "IBD Entry",
                filterable=False,
                sortable=False,
                default_table=True,
                advanced_filter=False,
                help_text="日线突破确认：成功显示日线量比，未确认显示原因。",
            ),
        ),
        (
            "ibd_entry_status",
            _field(
                "Status",
                "category",
                "IBD Entry",
                filterable=True,
                default_table=True,
                advanced_filter=True,
                help_text="当前 IBD Review 状态。",
            ),
        ),
        (
            "ibd_breakout_quality",
            _field(
                "Breakout Price Quality",
                "category",
                "IBD Entry",
                filterable=True,
                default_table=True,
                advanced_filter=True,
                help_text=(
                    "Price-action quality based on Close Position and Trigger Clearance.\n"
                    "Volume confirmation is evaluated separately."
                ),
            ),
        ),
        (
            "latest_close",
            _field(
                "Latest",
                "number",
                "IBD Entry",
                filterable=False,
                default_table=True,
                advanced_filter=False,
                fmt="0.00",
                help_text="当前数据快照的最新收盘价，不是实时价格。",
            ),
        ),
        (
            "current_vs_ibd_candidate_pct",
            _field(
                "Vs Candidate",
                "number",
                "IBD Entry",
                filterable=True,
                default_table=True,
                advanced_filter=True,
                fmt="0.00%",
                help_text="最新收盘价相对 Candidate Price 的距离。",
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
                "W Vol",
                "number",
                "Volume/Pullback",
                filterable=True,
                default_table=True,
                advanced_filter=True,
                fmt="0.00x",
                help_text="当前周成交量相对 10 周均量的倍数。",
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
                help_text="综合质量评分（只对 Active Signals 计算和展示分布）。",
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
                help_text="综合质量对照排名（只对 Active Signals 计算和展示分布），数值越小越靠前。",
            ),
        ),
        ("pullback_count", _field("Pullback Count", "number", "Risk / Structure", default_table=True)),
        (
            "pullback_duration_weeks",
            _field(
                "Pullback Duration Weeks",
                "number",
                "Risk / Structure",
                default_table=True,
                help_text="上游正式产出的回撤/巩固持续时间，用于 Continuation 信号的时长检查。",
            ),
        ),
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


def get_midweek_table_columns() -> list[str]:
    columns = get_default_table_columns()
    return [columns[0], "review_change_label", *columns[1:]]


def get_column_view_fields(view_name: str) -> list[str]:
    if view_name == "IBD Decision":
        return [field for field in IBD_DECISION_COLUMNS if field in FIELD_CONFIG]
    if view_name == "C Rank Reference":
        return [field for field in C_RANK_REFERENCE_COLUMNS if field in FIELD_CONFIG]
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
