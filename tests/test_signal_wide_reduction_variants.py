from types import SimpleNamespace

import pandas as pd

from backtest.ibd_weekly_signal_oracle_eval.evaluate_weekly_signal_oracle import (
    VARIANTS,
    item_allowed,
    item_allowed_for_relaxed_fill,
)


def _item(*, status="UNCONFIRMED", reasons=(), risks=()):
    return SimpleNamespace(
        code="TEST",
        snapshot_date="2026-01-02",
        entry_status=status,
        risk_codes=list(risks),
        reason_codes=list(reasons),
    )


def test_signal_wide_core_variant_allows_non_actionable_without_dry_or_geom_hard_filters():
    cfg = VARIANTS["signal_core_quality_eps_known"]
    row = pd.Series({"eps_yoy_growth": 30.0})
    item = _item(
        reasons=["volume_confirms_breakout", "geometry_caution_not_failure"],
        risks=["pullback_not_dry"],
    )

    assert item_allowed(item, row, True, cfg) is True


def test_signal_wide_core_variant_keeps_freshness_and_breakout_volume_as_hard_core():
    cfg = VARIANTS["signal_core_quality_eps_known"]
    row = pd.Series({"eps_yoy_growth": 30.0})

    assert item_allowed(
        _item(reasons=["volume_confirms_breakout"], risks=["extended_from_buy_point"]),
        row,
        True,
        cfg,
    ) is False
    assert item_allowed(
        _item(reasons=[], risks=["entry_volume_below_standard"]),
        row,
        True,
        cfg,
    ) is False


def test_relaxed_fill_can_fallback_from_eps_pass_to_eps_known_without_relaxing_core_quality():
    cfg = {
        "allow_non_actionable": True,
        "require_core_quality": True,
        "require_eps_pass": True,
        "fill_eps_fallback": "known",
    }
    known_below_pass = pd.Series({"eps_yoy_growth": 12.0})
    missing = pd.Series({"eps_yoy_growth": None})

    assert item_allowed(
        _item(reasons=["volume_confirms_breakout"], risks=[]),
        known_below_pass,
        True,
        cfg,
        relaxed=True,
    ) is False
    assert item_allowed_for_relaxed_fill(
        _item(reasons=["volume_confirms_breakout"], risks=[]),
        known_below_pass,
        True,
        cfg,
    ) is True
    assert item_allowed_for_relaxed_fill(
        _item(reasons=["volume_confirms_breakout"], risks=[]),
        missing,
        True,
        cfg,
    ) is False
    assert item_allowed_for_relaxed_fill(
        _item(reasons=[], risks=[]),
        known_below_pass,
        True,
        cfg,
    ) is False
