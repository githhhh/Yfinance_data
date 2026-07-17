import pytest
import pandas as pd
import numpy as np

from dashboard.field_config import STATUS_META, FIELD_CONFIG
from dashboard.table_view import _column_def
from dashboard.app import _format_card_val


def test_status_meta_consistency():
    expected_statuses = ["ACTIONABLE", "UNCONFIRMED", "BELOW_TRIGGER", "EXTENDED"]
    for s in expected_statuses:
        assert s in STATUS_META
        assert "label" in STATUS_META[s]
        assert "dot" in STATUS_META[s]
        assert "color" in STATUS_META[s]
        assert "tooltip" in STATUS_META[s]
        assert len(STATUS_META[s]["tooltip"]) > 0

    assert STATUS_META["ACTIONABLE"]["dot"] == "🟢"
    assert STATUS_META["UNCONFIRMED"]["dot"] == "🟡"
    assert STATUS_META["BELOW_TRIGGER"]["dot"] == "🔴"
    assert STATUS_META["EXTENDED"]["dot"] == "🔵"


def test_header_tooltip_config():
    expected_tooltips = {
        "code": "点击 Code 复制单个代码；点击该行其他位置查看详情。",
        "ibd_entry_status": "当前 IBD Review 状态。",
        "ibd_candidate_rule": "IBD Candidate 触发价的结构来源。",
        "current_vs_ibd_candidate_pct": "最新收盘价相对 Candidate Price 的距离。",
        "latest_close": "当前数据快照的最新收盘价，不是实时价格。",
        "ibd_entry_vol_or_reject": "日线突破确认：成功显示日线量比，未确认显示原因。",
        "volume_ratio": "当前周成交量相对 10 周均量的倍数。",
        "rank_C_continuous": "综合质量对照排名（只对 Active Signals 计算和展示分布），数值越小越靠前。",
    }
    for col, tip in expected_tooltips.items():
        col_def = _column_def(col)
        assert col_def.get("headerTooltip") == tip, f"Column {col} missing or wrong headerTooltip"
        assert FIELD_CONFIG[col].get("help") == tip


def test_selected_row_card_val_formatting():
    # Test None / NaN / empty
    assert _format_card_val(None, "%") == "n/a"
    assert _format_card_val(np.nan, "x") == "n/a"
    assert _format_card_val("nan", "") == "n/a"
    assert _format_card_val("", "%") == "n/a"

    # Test percentages
    assert _format_card_val(15.234, "%") == "+15.23%"
    assert _format_card_val(-6.812, "%") == "-6.81%"
    assert _format_card_val(0.0, "%") == "0.00%"

    # Test volume ratio
    assert _format_card_val(1.5, "x") == "1.50x"
    assert _format_card_val("2.10", "x") == "2.10x"

    # Test base duration weeks
    assert _format_card_val(8, "w") == "8w"
    assert _format_card_val(8.0, "w") == "8w"
    assert _format_card_val(8.5, "w") == "8.5w"

    # Test raw decimals
    assert _format_card_val(125.4, "") == "125.40"
