from __future__ import annotations

import csv
import io
import json
import os
from dataclasses import dataclass
from datetime import date, datetime
from typing import Callable
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd

RS_SOURCE = "Fred6725/rs-log"
RS_CSV_URL = "https://raw.githubusercontent.com/Fred6725/rs-log/main/output/rs_stocks.csv"
RS_COMMIT_API_URL = (
    "https://api.github.com/repos/Fred6725/rs-log/commits"
    "?path=output/rs_stocks.csv&per_page=1"
)
RS_MARKET_TIMEZONE = ZoneInfo("America/New_York")
RS_FIELDS = (
    "rs_percentile",
    "rs_1m_percentile",
    "rs_3m_percentile",
    "rs_6m_percentile",
)


@dataclass(frozen=True)
class RSReferenceSnapshot:
    market_date: date | None
    ratings: dict[str, dict[str, int | None]]
    commit_sha: str | None = None
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.market_date is not None and bool(self.ratings) and self.error is None


def _request_text(url: str, *, timeout: float = 12.0) -> str:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "Yfinance_data-dashboard-rs-reference",
    }
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def _coerce_percentile(value: object) -> int | None:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not 0 <= parsed <= 99:
        return None
    return int(round(parsed))


def parse_rs_market_date(commit_payload: str | list[dict[str, object]]) -> tuple[date, str | None]:
    payload = json.loads(commit_payload) if isinstance(commit_payload, str) else commit_payload
    if not payload:
        raise ValueError("rs-log commit history is empty")
    latest = payload[0]
    commit = latest.get("commit") if isinstance(latest, dict) else None
    if not isinstance(commit, dict):
        raise ValueError("rs-log commit payload is invalid")
    committer = commit.get("committer")
    if not isinstance(committer, dict) or not committer.get("date"):
        raise ValueError("rs-log commit timestamp is missing")
    timestamp = datetime.fromisoformat(str(committer["date"]).replace("Z", "+00:00"))
    market_date = timestamp.astimezone(RS_MARKET_TIMEZONE).date()
    sha = str(latest.get("sha")) if isinstance(latest, dict) and latest.get("sha") else None
    return market_date, sha


def parse_rs_csv(csv_text: str) -> dict[str, dict[str, int | None]]:
    rows: dict[str, dict[str, int | None]] = {}
    reader = csv.DictReader(io.StringIO(csv_text))
    required = {
        "Ticker",
        "Percentile",
        "1M_RS_Percentile",
        "3M_RS_Percentile",
        "6M_RS_Percentile",
    }
    if not required.issubset(reader.fieldnames or []):
        raise ValueError("rs-log CSV schema is incomplete")
    for row in reader:
        ticker = str(row.get("Ticker") or "").strip().upper()
        if not ticker:
            continue
        rows[ticker] = {
            "rs_percentile": _coerce_percentile(row.get("Percentile")),
            "rs_1m_percentile": _coerce_percentile(row.get("1M_RS_Percentile")),
            "rs_3m_percentile": _coerce_percentile(row.get("3M_RS_Percentile")),
            "rs_6m_percentile": _coerce_percentile(row.get("6M_RS_Percentile")),
        }
    if not rows:
        raise ValueError("rs-log CSV contains no ticker rows")
    return rows


def fetch_latest_rs_reference(
    *,
    fetch_text: Callable[[str], str] = _request_text,
) -> RSReferenceSnapshot:
    try:
        market_date, commit_sha = parse_rs_market_date(fetch_text(RS_COMMIT_API_URL))
        ratings = parse_rs_csv(fetch_text(RS_CSV_URL))
        return RSReferenceSnapshot(
            market_date=market_date,
            ratings=ratings,
            commit_sha=commit_sha,
        )
    except Exception as exc:
        return RSReferenceSnapshot(
            market_date=None,
            ratings={},
            error=f"{type(exc).__name__}: {exc}",
        )


def attach_rs_reference(
    frame: pd.DataFrame,
    *,
    snapshot_date: date | None,
    reference: RSReferenceSnapshot | None,
) -> pd.DataFrame:
    result = frame.copy()
    for field in RS_FIELDS:
        result[field] = None
    if (
        result.empty
        or snapshot_date is None
        or reference is None
        or not reference.available
        or reference.market_date != snapshot_date
    ):
        return result

    ratings = reference.ratings
    codes = result.get("code", pd.Series(index=result.index, dtype=object)).astype(str).str.upper()
    for field in RS_FIELDS:
        result[field] = codes.map(lambda code: ratings.get(code, {}).get(field))
    return result


def reference_meta(
    reference: RSReferenceSnapshot | None,
    *,
    complete_snapshot_date: date | None,
    midweek_snapshot_date: date | None,
) -> dict[str, object]:
    market_date = reference.market_date if reference is not None else None
    return {
        "source": RS_SOURCE,
        "market_date": market_date.isoformat() if market_date is not None else None,
        "commit_sha": reference.commit_sha if reference is not None else None,
        "available": bool(reference and reference.available),
        "matches_complete": bool(market_date and complete_snapshot_date and market_date == complete_snapshot_date),
        "matches_midweek": bool(market_date and midweek_snapshot_date and market_date == midweek_snapshot_date),
        "error": reference.error if reference is not None else "RS reference was not fetched",
    }


__all__ = [
    "RS_FIELDS",
    "RS_SOURCE",
    "RSReferenceSnapshot",
    "attach_rs_reference",
    "fetch_latest_rs_reference",
    "parse_rs_csv",
    "parse_rs_market_date",
    "reference_meta",
]
