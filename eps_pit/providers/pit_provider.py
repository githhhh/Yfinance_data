from __future__ import annotations

import datetime as dt
from email.utils import parsedate_to_datetime
import json
import logging
import math
import os
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yfinance as yf

from eps_pit.models import EPSMissingReason


DEFAULT_CACHE_DIR = Path(tempfile.gettempdir()) / "quant_trade_eps_pit_cache"
DEFAULT_CACHE_TTL_SECONDS = 24 * 60 * 60
# Fixed repository-level identity for SEC automated access. This intentionally
# uses GitHub's privacy-preserving noreply address rather than any personal
# email or runtime host configuration.
SEC_USER_AGENT = "Yfinance_data EPS PIT githhhh@users.noreply.github.com"
SEC_REQUESTS_PER_SECOND = 9.0
SEC_DEFAULT_REQUEST_INTERVAL_SECONDS = 1.0 / SEC_REQUESTS_PER_SECOND
SEC_COMPANYFACTS_CACHE_TTL_SECONDS = 30 * 60
SEC_RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
SEC_BULK_COMPANYFACTS_FILENAME = "companyfacts.zip"
SEC_BULK_COMPANYFACTS_URL = (
    "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"
)
SEC_BULK_CACHE_TTL_SECONDS = 24 * 60 * 60

YAHOO_DEFAULT_REQUEST_INTERVAL_SECONDS = 0.35
YAHOO_RATE_LIMIT_BACKOFF_SECONDS = (5.0, 15.0, 30.0)

_SEC_RATE_LOCK = threading.Lock()
_SEC_LAST_REQUEST_AT: float | None = None
_YAHOO_RATE_LOCK = threading.Lock()
_YAHOO_LAST_REQUEST_AT: float | None = None


def build_sec_request_headers() -> dict[str, str]:
    return {
        "User-Agent": SEC_USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
    }


# Compatibility aliases for older imports. Internal code uses EPSMissingReason.
NO_QUARTERLY_EPS = EPSMissingReason.NO_QUARTERLY_EPS.value
NO_PRIOR_YEAR_QUARTER = EPSMissingReason.NO_PRIOR_YEAR_QUARTER.value
PRIOR_YEAR_EPS_ZERO = EPSMissingReason.PRIOR_YEAR_EPS_ZERO.value
NO_VERIFIED_YAHOO_RELEASE_DATE = EPSMissingReason.NO_VERIFIED_YAHOO_RELEASE_DATE.value
PROVIDER_ERROR = EPSMissingReason.PROVIDER_ERROR.value


def normalize_symbol(symbol: object) -> str:
    if symbol is None:
        return ""
    try:
        if pd.isna(symbol):
            return ""
    except Exception:
        pass
    text = str(symbol).strip().upper().replace(".", "-")
    return "" if text in {"", "NAN", "<NA>", "NONE"} else text


def date10(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()[:10]
    if not text:
        return ""
    try:
        dt.date.fromisoformat(text)
    except ValueError:
        return ""
    return text


def safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def availability_date(record: dict[str, Any]) -> str:
    if str(record.get("source") or "").lower().startswith("yahoo"):
        return date10(record.get("earnings_release_at"))
    return date10(record.get("filing_date")) or date10(record.get("accepted_at"))


def duration_days(record: dict[str, Any]) -> int | None:
    start = date10(record.get("start"))
    end = date10(record.get("report_period"))
    if not start or not end:
        return None
    return (dt.date.fromisoformat(end) - dt.date.fromisoformat(start)).days + 1


def is_standalone_quarter(record: dict[str, Any]) -> bool:
    period_type = str(record.get("period_type") or "").strip().lower()
    if period_type == "quarter":
        return True
    duration = duration_days(record)
    if duration is not None:
        return 70 <= duration <= 110
    # Yahoo rows are already quarterly-income-statement observations. SEC rows
    # without duration metadata are never assumed to be standalone quarters.
    return (
        str(record.get("source") or "").lower().startswith("yahoo")
        and bool(date10(record.get("report_period")))
        and bool(date10(record.get("earnings_release_at")))
    )


def _concept_priority(record: dict[str, Any]) -> int:
    concept = str(record.get("concept") or "").lower()
    if "diluted" in concept:
        return 2
    if "basic" in concept:
        return 1
    return 0


def _quarter_label(record: dict[str, Any]) -> str | None:
    raw = str(record.get("fiscal_quarter") or "").strip().upper()
    if raw in {"Q1", "Q2", "Q3", "Q4"}:
        return raw
    if raw == "FY" and is_standalone_quarter(record):
        return "Q4"
    return None


def _latest_visible_fact_versions(
    records: list[dict[str, Any]], snapshot_date: object
) -> list[dict[str, Any]]:
    """Filter by PIT visibility first, then choose the latest visible version.

    Fact identity uses source/concept/start/end. This intentionally keeps a
    standalone quarter and a YTD fact ending on the same date as separate facts.
    """
    snapshot = date10(snapshot_date)
    if not snapshot:
        return []

    visible: list[dict[str, Any]] = []
    for input_order, raw in enumerate(records):
        eps = safe_float(raw.get("eps_diluted"))
        end = date10(raw.get("report_period"))
        available = availability_date(raw)
        if eps is None or not end or not available or available > snapshot:
            continue
        item = dict(raw)
        item["_eps"] = eps
        item["_available_date"] = available
        item["_input_order"] = input_order
        visible.append(item)

    latest: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for item in visible:
        key = (
            str(item.get("source") or ""),
            str(item.get("concept") or "EPS"),
            date10(item.get("start")),
            date10(item.get("report_period")),
        )
        previous = latest.get(key)
        candidate_version = (
            item["_available_date"],
            str(item.get("accepted_at") or item.get("filing_date") or ""),
            int(item.get("_input_order", -1)),
        )
        previous_version = (
            previous["_available_date"],
            str(previous.get("accepted_at") or previous.get("filing_date") or ""),
            int(previous.get("_input_order", -1)),
        ) if previous is not None else None
        if previous is None or candidate_version >= previous_version:
            latest[key] = item
    return list(latest.values())


def _reported_quarter(item: dict[str, Any]) -> dict[str, Any]:
    result = dict(item)
    label = _quarter_label(result)
    if label:
        result["fiscal_quarter"] = label
    result["calculation_method"] = "reported_quarter"
    result["_method_priority"] = 2
    return result


def _compatible_unit(*records: dict[str, Any]) -> bool:
    units = {str(record.get("unit")) for record in records if record.get("unit")}
    return len(units) <= 1


def _derive_record(
    *,
    minuend: dict[str, Any],
    subtrahends: list[dict[str, Any]],
    fiscal_quarter: str,
    method: str,
) -> dict[str, Any] | None:
    records = [minuend, *subtrahends]
    if not _compatible_unit(*records):
        return None
    concept = str(minuend.get("concept") or "EPS")
    if any(str(record.get("concept") or "EPS") != concept for record in subtrahends):
        return None
    values = [safe_float(record.get("_eps", record.get("eps_diluted"))) for record in records]
    if any(value is None for value in values):
        return None
    value = values[0] - sum(values[1:])
    if not math.isfinite(value):
        return None
    available_dates = [
        str(record.get("_available_date") or availability_date(record))
        for record in records
    ]
    if any(not date10(value) for value in available_dates):
        return None
    source_ids = [str(record.get("source_record_id") or "") for record in records]
    effective = max(date10(value) for value in available_dates)
    return {
        "code": minuend.get("code"),
        "fiscal_year": minuend.get("fiscal_year"),
        "fiscal_quarter": fiscal_quarter,
        "report_period": date10(minuend.get("report_period")),
        "eps_diluted": value,
        "_eps": value,
        "filing_date": effective,
        "accepted_at": effective,
        "concept": concept,
        "unit": minuend.get("unit"),
        "period_type": "quarter",
        "source": minuend.get("source") or "SEC",
        "source_record_id": f"{method}({','.join(source_ids)})",
        "_available_date": effective,
        "calculation_method": method,
        "_method_priority": 1,
    }


def _nearest_before(
    records: list[dict[str, Any]],
    end: str,
    *,
    period_type: str | None = None,
    quarter: str | None = None,
) -> dict[str, Any] | None:
    candidates = []
    for record in records:
        report_period = date10(record.get("report_period"))
        if not report_period or report_period >= end:
            continue
        if period_type and str(record.get("period_type") or "") != period_type:
            continue
        if quarter and _quarter_label(record) != quarter:
            continue
        candidates.append(record)
    if not candidates:
        return None
    return max(candidates, key=lambda record: date10(record.get("report_period")))


def _construct_sec_quarters(visible: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sec = [record for record in visible if str(record.get("source") or "").upper() == "SEC"]
    reported = [_reported_quarter(record) for record in sec if is_standalone_quarter(record)]
    derived: list[dict[str, Any]] = []

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in sec:
        start = date10(record.get("start"))
        concept = str(record.get("concept") or "EPS")
        if start:
            groups.setdefault((concept, start), []).append(record)

    # Q2 = H1 YTD - Q1 standalone. Q3 = 9M YTD - H1 YTD.
    for (_, fiscal_start), group in groups.items():
        q1s = [
            record for record in group
            if is_standalone_quarter(record) and _quarter_label(record) == "Q1"
        ]
        ytd6s = [record for record in group if record.get("period_type") == "ytd_6m"]
        ytd9s = [record for record in group if record.get("period_type") == "ytd_9m"]

        for h1 in ytd6s:
            q1 = _nearest_before(q1s, date10(h1.get("report_period")), quarter="Q1")
            if q1 is not None:
                record = _derive_record(
                    minuend=h1,
                    subtrahends=[q1],
                    fiscal_quarter="Q2",
                    method="derived_from_ytd",
                )
                if record is not None:
                    record["fiscal_start"] = fiscal_start
                    derived.append(record)

        for ytd9 in ytd9s:
            h1 = _nearest_before(
                ytd6s,
                date10(ytd9.get("report_period")),
                period_type="ytd_6m",
            )
            if h1 is not None:
                record = _derive_record(
                    minuend=ytd9,
                    subtrahends=[h1],
                    fiscal_quarter="Q3",
                    method="derived_from_ytd",
                )
                if record is not None:
                    record["fiscal_start"] = fiscal_start
                    derived.append(record)

    quarters = [*reported, *derived]

    # Q4 = FY - Q1 - Q2 - Q3. Only derive when all three component quarters
    # are compatible, fall inside the same fiscal-year boundaries, and are
    # already visible as of the snapshot.
    fy_records = [record for record in sec if record.get("period_type") == "fy"]
    for fy in fy_records:
        fiscal_start = date10(fy.get("start"))
        fiscal_end = date10(fy.get("report_period"))
        concept = str(fy.get("concept") or "EPS")
        if not fiscal_start or not fiscal_end:
            continue
        components: list[dict[str, Any]] = []
        for label in ("Q1", "Q2", "Q3"):
            candidates = [
                record for record in quarters
                if str(record.get("concept") or "EPS") == concept
                and _quarter_label(record) == label
                and fiscal_start <= date10(record.get("report_period")) < fiscal_end
            ]
            if not candidates:
                components = []
                break
            components.append(
                max(
                    candidates,
                    key=lambda record: (
                        date10(record.get("report_period")),
                        int(record.get("_method_priority", 0)),
                        str(record.get("_available_date") or ""),
                    ),
                )
            )
        if len(components) == 3:
            record = _derive_record(
                minuend=fy,
                subtrahends=components,
                fiscal_quarter="Q4",
                method="derived_from_fy",
            )
            if record is not None:
                record["fiscal_start"] = fiscal_start
                derived.append(record)

    return [*reported, *derived]


def select_visible_quarters(
    records: list[dict[str, Any]],
    snapshot_date: object,
) -> list[dict[str, Any]]:
    visible = _latest_visible_fact_versions(records, snapshot_date)
    non_sec = [
        _reported_quarter(record)
        for record in visible
        if str(record.get("source") or "").upper() != "SEC" and is_standalone_quarter(record)
    ]
    candidates = [*_construct_sec_quarters(visible), *non_sec]

    # Prefer a directly reported quarter over a derived quarter for the same
    # concept/report period. Within the same method, use the latest visible
    # version. Snapshot filtering above prevents amendment time travel.
    best: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in candidates:
        key = (
            str(item.get("source") or ""),
            str(item.get("concept") or "EPS"),
            date10(item.get("report_period")),
        )
        previous = best.get(key)
        score = (
            int(item.get("_method_priority", 0)),
            str(item.get("_available_date") or availability_date(item)),
            str(item.get("source_record_id") or ""),
        )
        previous_score = (
            int(previous.get("_method_priority", 0)),
            str(previous.get("_available_date") or availability_date(previous)),
            str(previous.get("source_record_id") or ""),
        ) if previous is not None else None
        if previous is None or score >= previous_score:
            best[key] = item

    return sorted(
        best.values(),
        key=lambda record: (
            date10(record.get("report_period")),
            _concept_priority(record),
            str(record.get("_available_date") or availability_date(record)),
        ),
    )


def _find_prior_year_record(
    current: dict[str, Any],
    eligible: list[dict[str, Any]],
) -> dict[str, Any] | None:
    current_period = date10(current.get("report_period"))
    if not current_period:
        return None
    current_date = dt.date.fromisoformat(current_period)
    try:
        target = current_date.replace(year=current_date.year - 1)
    except ValueError:
        target = current_date.replace(year=current_date.year - 1, day=28)

    current_concept = str(current.get("concept") or "EPS")
    current_quarter = _quarter_label(current)
    candidates: list[tuple[int, str, dict[str, Any]]] = []
    for record in eligible:
        if str(record.get("concept") or "EPS") != current_concept:
            continue
        period = date10(record.get("report_period"))
        if not period or period == current_period:
            continue
        candidate_quarter = _quarter_label(record)
        if current_quarter and candidate_quarter != current_quarter:
            continue
        if not current_quarter and candidate_quarter:
            continue
        period_date = dt.date.fromisoformat(period)
        distance = abs((period_date - target).days)
        if distance <= 45:
            candidates.append((distance, str(record.get("_available_date") or ""), record))

    if not candidates:
        return None
    min_distance = min(item[0] for item in candidates)
    nearest = [item for item in candidates if item[0] == min_distance]
    return max(nearest, key=lambda item: item[1])[2]


def calculate_latest_eps_yoy_diagnostic(
    records: list[dict[str, Any]],
    snapshot_date: object,
) -> tuple[dict[str, Any] | None, EPSMissingReason | None]:
    eligible = select_visible_quarters(records, snapshot_date)
    if not eligible:
        return None, EPSMissingReason.NO_QUARTERLY_EPS

    latest_period = max(date10(record.get("report_period")) for record in eligible)
    current_candidates = [
        record for record in eligible if date10(record.get("report_period")) == latest_period
    ]
    current_candidates.sort(key=_concept_priority, reverse=True)

    saw_prior = False
    saw_zero = False
    for current in current_candidates:
        prior = _find_prior_year_record(current, eligible)
        if prior is None:
            continue
        saw_prior = True
        current_eps = safe_float(current.get("_eps", current.get("eps_diluted")))
        prior_eps = safe_float(prior.get("_eps", prior.get("eps_diluted")))
        if current_eps is None or prior_eps is None:
            continue
        if prior_eps == 0:
            saw_zero = True
            continue

        growth = (current_eps - prior_eps) / abs(prior_eps) * 100.0
        return {
            "eps_yoy_growth": round(growth, 10),
            "source": current.get("source") or "SEC/Yahoo",
            "effective_date": str(current.get("_available_date") or availability_date(current)),
            "current_eps": current_eps,
            "prior_year_eps": prior_eps,
            "current_period": latest_period,
            "prior_year_period": date10(prior.get("report_period")),
            "calculation_method": current.get("calculation_method") or "reported_quarter",
            "source_record_id": current.get("source_record_id"),
        }, None

    if saw_prior and saw_zero:
        return None, EPSMissingReason.PRIOR_YEAR_EPS_ZERO
    return None, EPSMissingReason.NO_PRIOR_YEAR_QUARTER


def calculate_latest_eps_yoy(
    records: list[dict[str, Any]],
    snapshot_date: object,
) -> dict[str, Any] | None:
    result, _ = calculate_latest_eps_yoy_diagnostic(records, snapshot_date)
    return result


class TTLJSONCache:
    def __init__(self, ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS):
        self.ttl_seconds = ttl_seconds

    def load(self, path: Path) -> Any | None:
        if not path.exists():
            return None
        try:
            if time.time() - path.stat().st_mtime > self.ttl_seconds:
                return None
            return json.loads(path.read_text())
        except Exception:
            return None

    @staticmethod
    def load_any(path: Path) -> Any | None:
        """Read a cache entry regardless of TTL for safe stale fallbacks."""
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception:
            return None

    @staticmethod
    def write(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(data, handle)
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)


class SECProvider:
    TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

    def __init__(
        self,
        cache_dir: Path | None = None,
        rate_limit_sleep: float = SEC_DEFAULT_REQUEST_INTERVAL_SECONDS,
        cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
        companyfacts_cache_ttl_seconds: int = SEC_COMPANYFACTS_CACHE_TTL_SECONDS,
        bulk_cache_ttl_seconds: int = SEC_BULK_CACHE_TTL_SECONDS,
        max_retries: int = 2,
        bulk_companyfacts_zip: Path | None = None,
    ):
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR) / "sec"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_sleep = max(float(rate_limit_sleep), 0.0)
        self.ticker_cache = TTLJSONCache(cache_ttl_seconds)
        self.companyfacts_cache = TTLJSONCache(companyfacts_cache_ttl_seconds)
        self.bulk_cache_ttl_seconds = max(int(bulk_cache_ttl_seconds), 0)
        self.max_retries = max(int(max_retries), 0)
        self._blocked_error: RuntimeError | None = None
        self._cik_map: dict[str, str] | None = None
        self._cik_map_error: Exception | None = None
        self._bulk_members: dict[str, str] | None = None
        self._bulk_prepare_attempted = False
        self._bulk_prepare_error: Exception | None = None
        self.bulk_companyfacts_zip = Path(
            bulk_companyfacts_zip
            or (self.cache_dir / SEC_BULK_COMPANYFACTS_FILENAME)
        )
        self.session = requests.Session()
        self.session.headers.update(build_sec_request_headers())

    @property
    def headers(self) -> dict[str, str]:
        return dict(self.session.headers)

    def _throttle(self) -> None:
        global _SEC_LAST_REQUEST_AT
        if self.rate_limit_sleep <= 0:
            return
        with _SEC_RATE_LOCK:
            now = time.monotonic()
            if _SEC_LAST_REQUEST_AT is not None:
                remaining = self.rate_limit_sleep - (now - _SEC_LAST_REQUEST_AT)
                if remaining > 0:
                    time.sleep(remaining)
            _SEC_LAST_REQUEST_AT = time.monotonic()

    @staticmethod
    def _retry_delay(response: requests.Response, attempt: int) -> float:
        retry_after = str(response.headers.get("Retry-After") or "").strip()
        if retry_after:
            try:
                return max(float(retry_after), 0.0)
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(retry_after)
                    if retry_at.tzinfo is None:
                        retry_at = retry_at.replace(tzinfo=dt.timezone.utc)
                    return max(
                        (retry_at - dt.datetime.now(dt.timezone.utc)).total_seconds(),
                        0.0,
                    )
                except (TypeError, ValueError, OverflowError):
                    pass
        return 0.5 * (2 ** attempt)

    def _get_json(self, url: str, *, label: str) -> Any:
        if self._blocked_error is not None:
            raise self._blocked_error

        attempts = self.max_retries + 1
        for attempt in range(attempts):
            self._throttle()
            response = self.session.get(url, timeout=15)

            if response.status_code == 200:
                try:
                    return response.json()
                except Exception as exc:
                    raise RuntimeError(f"{label} returned invalid JSON") from exc

            if (
                response.status_code in SEC_RETRYABLE_STATUS_CODES
                and attempt + 1 < attempts
            ):
                time.sleep(self._retry_delay(response, attempt))
                continue

            error = RuntimeError(f"{label} HTTP {response.status_code}")
            if response.status_code == 403:
                # Hosted/cloud egress can be blocked independently of request
                # identity. Stop the entire SEC client for this run after the
                # first access-denied response instead of hammering every CIK.
                self._blocked_error = error
            raise error

        raise RuntimeError(f"{label} request failed")

    @staticmethod
    def _partial_download_path(destination: Path) -> Path:
        return destination.with_name(f".{destination.name}.part")

    @staticmethod
    def _validate_downloaded_zip(path: Path, *, label: str) -> None:
        if not zipfile.is_zipfile(path):
            raise RuntimeError(f"{label} downloaded file is not a valid ZIP")

    def _download_file(self, url: str, destination: Path, *, label: str) -> None:
        """Download a large SEC archive with resumable Range requests.

        A stable .part file is preserved across retries and process restarts.
        HTTP 206 appends to the partial file. If the server ignores Range and
        returns HTTP 200, the partial file is safely restarted from byte zero.
        """
        if self._blocked_error is not None:
            raise self._blocked_error

        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = self._partial_download_path(destination)
        attempts = self.max_retries + 1

        for attempt in range(attempts):
            offset = partial.stat().st_size if partial.exists() else 0
            request_headers = {"Range": f"bytes={offset}-"} if offset > 0 else None

            self._throttle()
            response = self.session.get(
                url,
                timeout=60,
                stream=True,
                headers=request_headers,
            )

            try:
                if response.status_code == 416 and offset > 0:
                    content_range = str(response.headers.get("Content-Range") or "")
                    total_text = content_range.rsplit("/", 1)[-1] if "/" in content_range else ""
                    try:
                        total_size = int(total_text)
                    except ValueError:
                        total_size = -1
                    if total_size == offset:
                        self._validate_downloaded_zip(partial, label=label)
                        os.replace(partial, destination)
                        return
                    partial.unlink(missing_ok=True)
                    if attempt + 1 < attempts:
                        continue

                if response.status_code in {200, 206}:
                    append = response.status_code == 206 and offset > 0
                    if append:
                        content_range = str(response.headers.get("Content-Range") or "")
                        expected_prefix = f"bytes {offset}-"
                        if not content_range.startswith(expected_prefix):
                            partial.unlink(missing_ok=True)
                            raise RuntimeError(
                                f"{label} invalid Content-Range for resume: {content_range!r}"
                            )

                    mode = "ab" if append else "wb"
                    try:
                        with partial.open(mode) as handle:
                            for chunk in response.iter_content(chunk_size=1024 * 1024):
                                if chunk:
                                    handle.write(chunk)
                    except requests.RequestException:
                        if attempt + 1 < attempts:
                            time.sleep(0.5 * (2 ** attempt))
                            continue
                        raise

                    self._validate_downloaded_zip(partial, label=label)
                    os.replace(partial, destination)
                    return

                if (
                    response.status_code in SEC_RETRYABLE_STATUS_CODES
                    and attempt + 1 < attempts
                ):
                    time.sleep(self._retry_delay(response, attempt))
                    continue

                error = RuntimeError(f"{label} HTTP {response.status_code}")
                if response.status_code == 403:
                    self._blocked_error = error
                raise error
            finally:
                response.close()

        raise RuntimeError(f"{label} download failed")

    def ensure_bulk_companyfacts(self) -> Path:
        """Refresh the nightly SEC bulk archive at most once per provider run."""
        path = self.bulk_companyfacts_zip
        if path.exists():
            try:
                age = time.time() - path.stat().st_mtime
                if age <= self.bulk_cache_ttl_seconds:
                    return path
            except OSError:
                pass

        if self._bulk_prepare_attempted:
            if path.exists():
                return path
            if self._bulk_prepare_error is not None:
                raise self._bulk_prepare_error

        self._bulk_prepare_attempted = True
        try:
            self._download_file(
                SEC_BULK_COMPANYFACTS_URL,
                path,
                label="SEC companyfacts bulk",
            )
            self._bulk_members = None
            return path
        except Exception as exc:
            self._bulk_prepare_error = exc
            if path.exists():
                logging.warning(
                    "SEC bulk refresh failed; using stale local companyfacts.zip: %s",
                    exc,
                )
                return path
            raise

    def _load_bulk_companyfacts(self, cik: str) -> dict[str, Any] | None:
        path = self.bulk_companyfacts_zip
        if not path.exists():
            return None
        try:
            with zipfile.ZipFile(path) as archive:
                if self._bulk_members is None:
                    self._bulk_members = {
                        Path(name).name: name
                        for name in archive.namelist()
                        if name.lower().endswith(".json")
                    }
                member = self._bulk_members.get(f"CIK{cik}.json")
                if not member:
                    return None
                with archive.open(member) as handle:
                    return json.load(handle)
        except Exception as exc:
            logging.warning("SEC bulk companyfacts cache unreadable: %s", exc)
            return None

    def fetch_quarterly_history(
        self,
        symbol: str,
        *,
        prefer_bulk: bool = False,
    ) -> list[dict[str, Any]]:
        cik = self.get_cik(symbol)
        if not cik:
            return []
        cache_file = self.cache_dir / f"{normalize_symbol(symbol)}.json"
        facts = self.companyfacts_cache.load(cache_file)

        if facts is None and prefer_bulk:
            try:
                self.ensure_bulk_companyfacts()
            except Exception:
                # A direct companyfacts request may still work after a transient
                # bulk-download failure. A 403 circuit breaker will prevent it
                # automatically when the egress itself is blocked.
                pass

        if facts is None:
            facts = self._load_bulk_companyfacts(cik)
        if facts is None:
            facts = self._get_json(
                self.FACTS_URL.format(cik=cik),
                label=f"SEC companyfacts for {normalize_symbol(symbol)}",
            )
        self.companyfacts_cache.write(cache_file, facts)
        return self._parse_company_facts(normalize_symbol(symbol), facts)

    def get_cik(self, symbol: str) -> str | None:
        sym = normalize_symbol(symbol)
        if not sym:
            return None
        mapping = self._get_cik_map()
        cik = mapping.get(sym) or mapping.get(sym.replace("-", "."))
        if cik is None and self._cik_map_error is not None:
            raise RuntimeError(
                f"SEC ticker map unavailable and stale cache has no mapping for {sym}"
            ) from self._cik_map_error
        return cik

    def _get_cik_map(self) -> dict[str, str]:
        if self._cik_map is not None:
            return self._cik_map
        cache_file = self.cache_dir / "company_tickers.json"
        data = self.ticker_cache.load(cache_file)
        if data is None:
            stale = TTLJSONCache.load_any(cache_file)
            try:
                data = self._get_json(self.TICKERS_URL, label="SEC ticker map")
                self.ticker_cache.write(cache_file, data)
            except Exception as exc:
                if stale is None:
                    raise
                self._cik_map_error = exc
                data = stale
                logging.warning(
                    "SEC ticker map network refresh failed; using stale local mapping: %s",
                    exc,
                )

        mapping: dict[str, str] = {}
        if isinstance(data, dict):
            for value in data.values():
                if not isinstance(value, dict):
                    continue
                ticker = normalize_symbol(value.get("ticker"))
                cik_raw = value.get("cik_str")
                if cik_raw is None:
                    continue
                cik = str(cik_raw).zfill(10)
                if ticker and cik:
                    mapping[ticker] = cik
                    mapping[ticker.replace("-", ".")] = cik
        self._cik_map = mapping
        return mapping

    def _parse_company_facts(self, symbol: str, facts: dict[str, Any]) -> list[dict[str, Any]]:
        gaap = facts.get("facts", {}).get("us-gaap", {})
        records: list[dict[str, Any]] = []
        for concept in ("EarningsPerShareDiluted", "EarningsPerShareBasic"):
            units = gaap.get(concept, {}).get("units", {})
            for entry in units.get("USD/shares", []):
                value = safe_float(entry.get("val"))
                filed = date10(entry.get("filed"))
                start = date10(entry.get("start"))
                end = date10(entry.get("end"))
                form = str(entry.get("form") or "")
                if value is None or not filed or not start or not end:
                    continue
                if form not in {"10-Q", "10-Q/A", "10-K", "10-K/A"}:
                    continue
                duration = (dt.date.fromisoformat(end) - dt.date.fromisoformat(start)).days + 1
                if 70 <= duration <= 110:
                    period_type = "quarter"
                elif 150 <= duration <= 210:
                    period_type = "ytd_6m"
                elif 240 <= duration <= 300:
                    period_type = "ytd_9m"
                elif 330 <= duration <= 380:
                    period_type = "fy"
                else:
                    period_type = "unknown"
                fiscal_year = entry.get("fy")
                try:
                    fiscal_year = int(fiscal_year) if fiscal_year is not None else None
                except (TypeError, ValueError):
                    fiscal_year = None
                records.append(
                    {
                        "code": symbol,
                        "fiscal_year": fiscal_year,
                        "fiscal_quarter": str(entry.get("fp") or ""),
                        "report_period": end,
                        "start": start,
                        "eps_diluted": value,
                        "filing_date": filed,
                        # Preserve accepted timestamp for deterministic same-day ordering;
                        # PIT visibility still uses the conservative filed calendar date.
                        "accepted_at": str(entry.get("accepted") or filed),
                        "frame": entry.get("frame"),
                        "form": form,
                        "concept": concept,
                        "unit": "USD/shares",
                        "period_type": period_type,
                        "source": "SEC",
                        "source_record_id": (
                            f"{entry.get('accn')}_{concept}_{start}_{end}_{entry.get('fp')}"
                        ),
                    }
                )
        return sorted(
            records,
            key=lambda record: (
                date10(record.get("report_period")),
                date10(record.get("filing_date")),
                str(record.get("accepted_at") or ""),
                str(record.get("source_record_id") or ""),
            ),
        )


class YahooFundamentalsProvider:
    def __init__(
        self,
        cache_dir: Path | None = None,
        cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
        rate_limit_sleep: float = YAHOO_DEFAULT_REQUEST_INTERVAL_SECONDS,
        max_rate_limit_retries: int = len(YAHOO_RATE_LIMIT_BACKOFF_SECONDS),
    ):
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR) / "yahoo"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = TTLJSONCache(cache_ttl_seconds)
        self.rate_limit_sleep = max(float(rate_limit_sleep), 0.0)
        self.max_rate_limit_retries = max(int(max_rate_limit_retries), 0)
        self.missing_release_periods: list[str] = []

    def _throttle(self) -> None:
        global _YAHOO_LAST_REQUEST_AT
        if self.rate_limit_sleep <= 0:
            return
        with _YAHOO_RATE_LOCK:
            now = time.monotonic()
            if _YAHOO_LAST_REQUEST_AT is not None:
                remaining = self.rate_limit_sleep - (now - _YAHOO_LAST_REQUEST_AT)
                if remaining > 0:
                    time.sleep(remaining)
            _YAHOO_LAST_REQUEST_AT = time.monotonic()

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return (
            exc.__class__.__name__ == "YFRateLimitError"
            or "too many requests" in text
            or "rate limit" in text
            or "http 429" in text
        )

    def _fetch_payload(self, symbol: str, *, include_events: bool) -> dict[str, Any]:
        attempts = self.max_rate_limit_retries + 1
        for attempt in range(attempts):
            self._throttle()
            ticker = yf.Ticker(symbol)
            try:
                events = self._fetch_earnings_dates(ticker) if include_events else []
                income = self._fetch_income_stmt_eps(ticker)
                return {"events": events, "income": income}
            except Exception as exc:
                if not self._is_rate_limit_error(exc) or attempt + 1 >= attempts:
                    raise
                delay_index = min(attempt, len(YAHOO_RATE_LIMIT_BACKOFF_SECONDS) - 1)
                time.sleep(YAHOO_RATE_LIMIT_BACKOFF_SECONDS[delay_index])

        raise RuntimeError(f"Yahoo fundamentals request failed for {symbol}")

    @property
    def last_missing_release_date_count(self) -> int:
        return len(self.missing_release_periods)

    @last_missing_release_date_count.setter
    def last_missing_release_date_count(self, value: int) -> None:
        # Compatibility with the previous mutable counter. Setting zero resets
        # diagnostics; non-zero synthetic values are not meaningful.
        if value == 0:
            self.missing_release_periods = []

    def fetch_quarterly_history(
        self,
        symbol: str,
        *,
        require_release_date: bool = True,
        observed_on: object | None = None,
        refresh: bool = False,
    ) -> list[dict[str, Any]]:
        """Return Yahoo quarterly EPS facts under explicit availability semantics.

        Historical/replay reconstruction requires a matched Yahoo earnings
        release timestamp. A true current LIVE observation may instead use the
        fact that the income statement is observable now; in that case every
        returned row is effective no earlier than observed_on.
        """
        sym = normalize_symbol(symbol)
        if not sym:
            return []

        observation = date10(observed_on) if observed_on is not None else ""
        if not require_release_date and not observation:
            raise ValueError("Yahoo live observation requires observed_on")

        cache_file = self.cache_dir / f"{sym}.json"
        if require_release_date:
            data = None if refresh else self.cache.load(cache_file)
            if data is None:
                data = self._fetch_payload(sym, include_events=True)
                self.cache.write(cache_file, data)
        else:
            # Current LIVE observation needs only what Yahoo exposes now.
            # Do not make it depend on the often-incomplete earnings calendar,
            # and do not overwrite the historical reconstruction cache.
            data = self._fetch_payload(sym, include_events=False)

        events = sorted(
            data.get("events", []),
            key=lambda event: date10(event.get("earnings_release_at")),
        )
        records: list[dict[str, Any]] = []
        self.missing_release_periods = []
        for item in data.get("income", []):
            eps = safe_float(item.get("eps_diluted"))
            period_end = date10(item.get("period_end"))
            if eps is None or not period_end:
                continue

            if require_release_date:
                release_at = self._match_release_date(period_end, events)
                if not release_at:
                    self.missing_release_periods.append(period_end)
                    continue
                source = "Yahoo"
                source_record_id = f"yahoo_income_{period_end}"
            else:
                # This does not claim to know the historical release timestamp.
                # It records only that the current Yahoo statement was observed
                # by this run on the stated date.
                release_at = observation
                source = "YahooLiveObserved"
                source_record_id = f"yahoo_live_{observation}_{period_end}"

            records.append(
                {
                    "code": sym,
                    "report_period": period_end,
                    "eps_diluted": eps,
                    "earnings_release_at": release_at,
                    "period_type": "quarter",
                    "source": source,
                    "concept": "DilutedEPS",
                    "unit": "USD/shares",
                    "source_record_id": source_record_id,
                }
            )

        # Historical mode keeps only release-dated rows. LIVE observation mode
        # keeps current Yahoo rows but never backdates them before observed_on.
        return sorted(records, key=lambda record: record["report_period"])

    @staticmethod
    def _fetch_earnings_dates(ticker: yf.Ticker) -> list[dict[str, Any]]:
        events = ticker.get_earnings_dates(limit=32)
        if events is None or events.empty:
            return []
        result = []
        for index, row in events.iterrows():
            eps = safe_float(row.get("Reported EPS"))
            if eps is None:
                continue
            result.append(
                {
                    "earnings_release_at": pd.to_datetime(index).isoformat(),
                    "eps_diluted": eps,
                }
            )
        return result

    @staticmethod
    def _fetch_income_stmt_eps(ticker: yf.Ticker) -> list[dict[str, Any]]:
        income = ticker.quarterly_income_stmt
        if income is None or income.empty:
            return []
        eps_row = None
        for label in ("Diluted EPS", "Basic EPS"):
            if label in income.index:
                eps_row = income.loc[label]
                break
        if eps_row is None:
            return []
        result = []
        for period_end, value in eps_row.items():
            eps = safe_float(value)
            if eps is None:
                continue
            result.append(
                {
                    "period_end": pd.to_datetime(period_end).strftime("%Y-%m-%d"),
                    "eps_diluted": eps,
                }
            )
        return result

    @staticmethod
    def _match_release_date(period_end: str, events: list[dict[str, Any]]) -> str | None:
        period = date10(period_end)
        if not period:
            return None
        period_date = dt.date.fromisoformat(period)
        candidates: list[tuple[int, str]] = []
        for event in events:
            release = date10(event.get("earnings_release_at"))
            if not release:
                continue
            release_date = dt.date.fromisoformat(release)
            delta = (release_date - period_date).days
            if 0 <= delta <= 75:
                candidates.append((delta, str(event.get("earnings_release_at") or release)))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    @staticmethod
    def _income_statement_is_stale(
        records: list[dict[str, Any]], events: list[dict[str, Any]]
    ) -> bool:
        """Compatibility helper; no longer used to discard historical rows."""
        if not records or not events:
            return False
        record_dates = [date10(record.get("earnings_release_at")) for record in records]
        event_dates = [date10(event.get("earnings_release_at")) for event in events]
        record_dates = [value for value in record_dates if value]
        event_dates = [value for value in event_dates if value]
        if not record_dates or not event_dates:
            return False
        return (
            dt.date.fromisoformat(max(event_dates)) - dt.date.fromisoformat(max(record_dates))
        ).days > 45
