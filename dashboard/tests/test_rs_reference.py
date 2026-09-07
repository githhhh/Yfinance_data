from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from dashboard.rs_reference import (
    RS_COMMIT_API_URL,
    RSReferenceSnapshot,
    attach_rs_reference,
    fetch_latest_rs_reference,
    parse_rs_csv,
    parse_rs_market_date,
)


def test_commit_timestamp_maps_to_us_market_date() -> None:
    market_date, sha = parse_rs_market_date(
        '[{"sha":"abc123","commit":{"committer":{"date":"2026-09-05T01:28:10Z"}}}]'
    )

    assert market_date == date(2026, 9, 4)
    assert sha == "abc123"


def test_holiday_commit_maps_to_previous_actual_market_date() -> None:
    # 2026-07-03 is the observed Independence Day market holiday. rs-log
    # nevertheless has a real artifact commit at 2026-07-04T01:19Z, which is
    # 2026-07-03 evening in New York. It must represent the last completed
    # trading session (2026-07-02), not the holiday artifact date.
    market_date, sha = parse_rs_market_date(
        '[{"sha":"holiday123","commit":{"committer":{"date":"2026-07-04T01:19:42Z"}}}]'
    )

    assert market_date == date(2026, 7, 2)
    assert sha == "holiday123"


def test_preclose_delayed_commit_maps_to_previous_completed_session() -> None:
    market_date, _ = parse_rs_market_date(
        '[{"sha":"delayed","commit":{"committer":{"date":"2026-09-08T17:00:00Z"}}}]'
    )

    # 13:00 New York: today's regular session has not completed yet.
    assert market_date == date(2026, 9, 4)


def test_rs_csv_parses_current_and_prior_percentiles() -> None:
    ratings = parse_rs_csv(
        "Ticker,Percentile,1M_RS_Percentile,3M_RS_Percentile,6M_RS_Percentile\n"
        " crwd ,96,91,84,79\n"
    )

    assert ratings["CRWD"] == {
        "rs_percentile": 96,
        "rs_1m_percentile": 91,
        "rs_3m_percentile": 84,
        "rs_6m_percentile": 79,
    }


def test_rs_csv_rejects_incomplete_schema() -> None:
    with pytest.raises(ValueError, match="schema is incomplete"):
        parse_rs_csv("Ticker,Percentile\nCRWD,96\n")


def test_latest_rs_csv_is_pinned_to_the_same_commit_as_market_date() -> None:
    urls: list[str] = []

    def fetch(url: str) -> str:
        urls.append(url)
        if url == RS_COMMIT_API_URL:
            return '[{"sha":"abc123","commit":{"committer":{"date":"2026-09-05T01:28:10Z"}}}]'
        assert "/abc123/output/rs_stocks.csv" in url
        assert "/main/output/rs_stocks.csv" not in url
        return (
            "Ticker,Percentile,1M_RS_Percentile,3M_RS_Percentile,6M_RS_Percentile\n"
            "CRWD,96,91,84,79\n"
        )

    reference = fetch_latest_rs_reference(fetch_text=fetch)

    assert reference.available is True
    assert reference.commit_sha == "abc123"
    assert reference.market_date == date(2026, 9, 4)
    assert reference.ratings["CRWD"]["rs_percentile"] == 96
    assert len(urls) == 2


def test_rs_is_attached_only_for_exact_market_date_match() -> None:
    reference = RSReferenceSnapshot(
        market_date=date(2026, 9, 4),
        ratings={
            "CRWD": {
                "rs_percentile": 96,
                "rs_1m_percentile": 91,
                "rs_3m_percentile": 84,
                "rs_6m_percentile": 79,
            }
        },
        commit_sha="abc123",
    )
    frame = pd.DataFrame([{"code": " crwd "}, {"code": "MISSING"}])

    matched = attach_rs_reference(
        frame,
        snapshot_date=date(2026, 9, 4),
        reference=reference,
    )
    stale = attach_rs_reference(
        frame,
        snapshot_date=date(2026, 9, 3),
        reference=reference,
    )

    assert matched.loc[0, "rs_percentile"] == 96
    assert pd.isna(matched.loc[1, "rs_percentile"])
    assert stale["rs_percentile"].isna().all()


def test_external_rs_failure_is_fail_soft() -> None:
    def fail(_: str) -> str:
        raise TimeoutError("source unavailable")

    reference = fetch_latest_rs_reference(fetch_text=fail)

    assert reference.available is False
    assert reference.market_date is None
    assert reference.ratings == {}
    assert "TimeoutError" in str(reference.error)
