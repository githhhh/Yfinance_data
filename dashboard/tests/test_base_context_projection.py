from pathlib import Path

import pandas as pd

from dashboard.build_static import (
    PUBLIC_DASHBOARD_ROW_FIELDS,
    STATIC_ASSETS,
    _add_review_display_context,
)


DASHBOARD = Path(__file__).resolve().parents[1]


def _project(row: dict) -> object:
    frame = _add_review_display_context(pd.DataFrame([row]))
    return frame.iloc[0]["review_buy_point_date"]


def test_buy_point_date_uses_explicit_price_provenance() -> None:
    assert _project(
        {
            "ibd_candidate_rule": "ceiling",
            "ibd_candidate_price": 48.21,
            "ceiling": 48.209999,
            "ceiling_date": pd.Timestamp("2026-06-22"),
        }
    ) == pd.Timestamp("2026-06-22")

    assert _project(
        {
            "ibd_candidate_rule": "pivot",
            "ibd_candidate_price": 93.37,
            "ibd_candidate_extra": (
                '{"pivot_candidates":[{"price":93.37,"resistance_date":"2026-09-07"}],'
                '"selected_pivot":{"price":93.37,"resistance_date":"2026-09-07"}}'
            ),
        }
    ) == "2026-09-07"

    assert _project(
        {
            "ibd_candidate_rule": "ma10_touch_confirm",
            "ibd_candidate_price": 38.9,
            "ibd_candidate_extra": (
                '{"pending_high":38.9,"touch_date":"2026-08-17",'
                '"confirm_date":"2026-09-07"}'
            ),
        }
    ) == "2026-08-17"

    assert _project(
        {
            "ibd_candidate_rule": "ceiling_pullback",
            "ibd_candidate_price": 48.21,
            "ibd_candidate_extra": (
                '{"pending_high":48.21,"touch_date":"2026-08-17",'
                '"confirm_date":"2026-09-07"}'
            ),
        }
    ) == "2026-08-17"


def test_buy_point_date_fails_closed_when_date_is_not_explicit() -> None:
    assert pd.isna(
        _project(
            {
                "ibd_candidate_rule": "three_weeks_tight",
                "ibd_candidate_price": 68.38,
                "ibd_candidate_extra": '{"twk_high":68.38,"tight_weeks":3}',
            }
        )
    )

    assert pd.isna(
        _project(
            {
                "ibd_candidate_rule": "pivot",
                "ibd_candidate_price": 40.0,
                "ibd_candidate_extra": (
                    '{"pivot_candidates":[{"price":41.0,"resistance_date":"2026-09-07"}]}'
                ),
            }
        )
    )


def test_public_payload_exposes_only_compact_base_context() -> None:
    for field in ("review_buy_point_date", "ceiling", "ceiling_date"):
        assert field in PUBLIC_DASHBOARD_ROW_FIELDS
    assert "ibd_candidate_extra" not in PUBLIC_DASHBOARD_ROW_FIELDS


def test_base_context_runtime_is_published_and_keeps_ui_fact_only() -> None:
    assert "base_context_runtime.js" in STATIC_ASSETS

    index = (DASHBOARD / "index.html").read_text(encoding="utf-8")
    runtime = (DASHBOARD / "base_context_runtime.js").read_text(encoding="utf-8")

    assert 'src="./base_context_runtime.js"' in index
    assert "Base Ceiling" in runtime
    assert "Base Ceiling Date" in runtime
    assert "Buy Point Date" in runtime
    assert "review_buy_point_date" in runtime
    assert 'grid-template-columns: repeat(3, minmax(0, 1fr))' in runtime
    assert "ibd_entry_status" not in runtime
    assert "review_priority" not in runtime
