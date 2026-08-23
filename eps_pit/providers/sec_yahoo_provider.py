from __future__ import annotations

import datetime as dt
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yfinance as yf


DEFAULT_CACHE_DIR = Path(tempfile.gettempdir()) / "quant_trade_eps_pit_cache"


def _normalize_symbol(symbol: object) -> str:
    return str(symbol or "").strip().upper().replace(".", "-")


def _date10(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()[:10]


def _effective_date(record: dict[str, Any]) -> str:
    return (
        _date10(record.get("earnings_release_at"))
        or _date10(record.get("filing_date"))
        or _date10(record.get("accepted_at"))
        or _date10(record.get("report_period"))
    )


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def calculate_latest_eps_yoy(
    records: list[dict[str, Any]],
    snapshot_date: object,
) -> dict[str, Any] | None:
    snapshot = _date10(snapshot_date)
    if not snapshot:
        return None

    eligible: list[dict[str, Any]] = []
    for record in records:
        eps = _safe_float(record.get("eps_diluted"))
        effective = _effective_date(record)
        if eps is None or not effective or effective > snapshot:
            continue
        item = dict(record)
        item["_eps"] = eps
        item["_effective_date"] = effective
        eligible.append(item)

    eligible.sort(key=lambda r: (r.get("_effective_date") or "", r.get("report_period") or ""))
    if len(eligible) < 5:
        return None

    current = eligible[-1]
    prior = _find_prior_year_record(current, eligible)
    if prior is None:
        return None

    current_eps = current["_eps"]
    prior_eps = prior["_eps"]
    if prior_eps == 0:
        return None

    growth = (current_eps - prior_eps) / abs(prior_eps) * 100.0
    return {
        "eps_yoy_growth": round(growth, 10),
        "source": current.get("source") or "SEC/Yahoo",
        "effective_date": current["_effective_date"],
        "current_eps": current_eps,
        "prior_year_eps": prior_eps,
        "current_period": current.get("report_period"),
        "prior_year_period": prior.get("report_period"),
    }


def _find_prior_year_record(
    current: dict[str, Any],
    eligible: list[dict[str, Any]],
) -> dict[str, Any] | None:
    by_report_period = _find_prior_by_report_period(current, eligible)
    if by_report_period is not None:
        return by_report_period

    return None


def _find_prior_by_report_period(
    current: dict[str, Any],
    eligible: list[dict[str, Any]],
) -> dict[str, Any] | None:
    current_period = _date10(current.get("report_period"))
    if not current_period:
        return None
    try:
        curr_date = dt.date.fromisoformat(current_period)
        target = curr_date.replace(year=curr_date.year - 1)
    except ValueError:
        return None

    candidates: list[tuple[int, dict[str, Any]]] = []
    for record in eligible:
        if record is current:
            continue
        period = _date10(record.get("report_period"))
        if not period:
            continue
        try:
            period_date = dt.date.fromisoformat(period)
        except ValueError:
            continue
        distance = abs((period_date - target).days)
        if distance <= 45:
            candidates.append((distance, record))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1].get("report_period") or ""))
    return candidates[0][1]


class SECProvider:
    SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    SEC_FACTS_URL_TMPL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    USER_AGENT = "QuantTradeEPSPIT/1.0 contact@example.com"

    def __init__(self, cache_dir: Path | None = None, rate_limit_sleep: float = 0.1):
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR) / "sec"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_sleep = rate_limit_sleep
        self.headers = {"User-Agent": self.USER_AGENT}
        self._cik_map: dict[str, str] | None = None

    def fetch_quarterly_history(self, symbol: str) -> list[dict[str, Any]]:
        cik = self.get_cik(symbol)
        if not cik:
            return []

        facts = self._load_json(self.cache_dir / f"{_normalize_symbol(symbol)}.json")
        if facts is None:
            try:
                time.sleep(self.rate_limit_sleep)
                response = requests.get(
                    self.SEC_FACTS_URL_TMPL.format(cik=cik),
                    headers=self.headers,
                    timeout=15,
                )
                if response.status_code != 200:
                    return []
                facts = response.json()
                self._write_json(self.cache_dir / f"{_normalize_symbol(symbol)}.json", facts)
            except Exception:
                return []

        return self._parse_company_facts(_normalize_symbol(symbol), facts)

    def get_cik(self, symbol: str) -> str | None:
        cik_map = self._get_cik_map()
        sym = _normalize_symbol(symbol)
        return cik_map.get(sym) or cik_map.get(sym.replace("-", "."))

    def _get_cik_map(self) -> dict[str, str]:
        if self._cik_map is not None:
            return self._cik_map

        cache_file = self.cache_dir / "company_tickers.json"
        data = self._load_json(cache_file)
        if data is None:
            try:
                response = requests.get(self.SEC_TICKERS_URL, headers=self.headers, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    self._write_json(cache_file, data)
            except Exception:
                data = None

        cik_map: dict[str, str] = {}
        if isinstance(data, dict):
            for value in data.values():
                ticker = _normalize_symbol(value.get("ticker"))
                cik = str(value.get("cik_str", "")).zfill(10)
                if ticker and cik:
                    cik_map[ticker] = cik
                    cik_map[ticker.replace("-", ".")] = cik
        self._cik_map = cik_map
        return cik_map

    def _parse_company_facts(self, symbol: str, facts: dict[str, Any]) -> list[dict[str, Any]]:
        gaap = facts.get("facts", {}).get("us-gaap", {})
        eps_units = None
        for field in ("EarningsPerShareDiluted", "EarningsPerShareBasic"):
            units = gaap.get(field, {}).get("units", {})
            if "USD/shares" in units:
                eps_units = units["USD/shares"]
                break
        if not eps_units:
            return []

        records: list[dict[str, Any]] = []
        for entry in eps_units:
            value = _safe_float(entry.get("val"))
            filed = _date10(entry.get("filed"))
            end = _date10(entry.get("end"))
            fiscal_quarter = entry.get("fp")
            form = entry.get("form")
            fiscal_year = entry.get("fy")
            if value is None or not filed or not end or fiscal_quarter not in {"Q1", "Q2", "Q3", "Q4"}:
                continue
            if form not in {"10-Q", "10-Q/A", "10-K", "10-K/A"}:
                continue
            records.append(
                {
                    "code": symbol,
                    "fiscal_year": int(fiscal_year) if fiscal_year is not None else None,
                    "fiscal_quarter": fiscal_quarter,
                    "report_period": end,
                    "eps_diluted": value,
                    "filing_date": filed,
                    "accepted_at": filed,
                    "source": "SEC",
                    "source_record_id": f"{entry.get('accn')}_{end}_{fiscal_quarter}",
                }
            )

        dedup: dict[tuple[str, str], dict[str, Any]] = {}
        for record in records:
            key = (record["report_period"], record["fiscal_quarter"])
            prev = dedup.get(key)
            if prev is None or str(record.get("filing_date")) >= str(prev.get("filing_date")):
                dedup[key] = record
        return sorted(dedup.values(), key=lambda r: (r["report_period"], r["filing_date"]))

    @staticmethod
    def _load_json(path: Path) -> Any | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception:
            return None

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))


class YahooFundamentalsProvider:
    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR) / "yahoo"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_quarterly_history(self, symbol: str) -> list[dict[str, Any]]:
        sym = _normalize_symbol(symbol)
        cache_file = self.cache_dir / f"{sym}.json"
        data = SECProvider._load_json(cache_file)
        if data is None:
            try:
                ticker = yf.Ticker(sym)
                events = self._fetch_earnings_dates(ticker)
                income = self._fetch_income_stmt_eps(ticker)
                data = {"events": events, "income": income}
                SECProvider._write_json(cache_file, data)
            except Exception:
                return []

        records = []
        events = sorted(data.get("events", []), key=lambda e: _date10(e.get("earnings_release_at")))
        for item in data.get("income", []):
            eps = _safe_float(item.get("eps_diluted"))
            period_end = _date10(item.get("period_end"))
            if eps is None or not period_end:
                continue
            release_at = self._match_release_date(period_end, events)
            records.append(
                {
                    "code": sym,
                    "report_period": period_end,
                    "eps_diluted": eps,
                    "filing_date": release_at or period_end,
                    "earnings_release_at": release_at,
                    "source": "Yahoo",
                    "source_record_id": f"yahoo_income_{period_end}",
                }
            )

        if self._income_statement_is_stale(records, events):
            return []
        return sorted(records, key=lambda r: r["report_period"])

    @staticmethod
    def _fetch_earnings_dates(ticker: yf.Ticker) -> list[dict[str, Any]]:
        result = []
        events = ticker.get_earnings_dates(limit=32)
        if events is None or events.empty:
            return result
        for index, row in events.iterrows():
            eps = _safe_float(row.get("Reported EPS"))
            if eps is None:
                continue
            timestamp = pd.to_datetime(index)
            result.append(
                {
                    "earnings_release_at": timestamp.isoformat(),
                    "eps_diluted": eps,
                }
            )
        return result

    @staticmethod
    def _fetch_income_stmt_eps(ticker: yf.Ticker) -> list[dict[str, Any]]:
        result = []
        income = ticker.quarterly_income_stmt
        if income is None or income.empty:
            return result
        eps_row = None
        for label in ("Diluted EPS", "Basic EPS"):
            if label in income.index:
                eps_row = income.loc[label]
                break
        if eps_row is None:
            return result
        for period_end, value in eps_row.items():
            eps = _safe_float(value)
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
        try:
            period_date = dt.date.fromisoformat(period_end)
        except ValueError:
            return None
        for event in events:
            release = _date10(event.get("earnings_release_at"))
            if not release:
                continue
            try:
                release_date = dt.date.fromisoformat(release)
            except ValueError:
                continue
            if 0 <= (release_date - period_date).days <= 75:
                return event.get("earnings_release_at") or release
        return None

    @staticmethod
    def _income_statement_is_stale(
        records: list[dict[str, Any]],
        events: list[dict[str, Any]],
    ) -> bool:
        if not records or not events:
            return False
        record_dates = [_date10(record.get("earnings_release_at") or record.get("report_period")) for record in records]
        event_dates = [_date10(event.get("earnings_release_at")) for event in events]
        record_dates = [value for value in record_dates if value]
        event_dates = [value for value in event_dates if value]
        if not record_dates or not event_dates:
            return False
        try:
            latest_record = dt.date.fromisoformat(max(record_dates))
            latest_event = dt.date.fromisoformat(max(event_dates))
        except ValueError:
            return False
        return (latest_event - latest_record).days > 45


class SECYahooEPSProvider:
    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
        self.sec = SECProvider(self.cache_dir)
        self.yahoo = YahooFundamentalsProvider(self.cache_dir)

    def fetch_eps_yoy(self, symbol: str, snapshot_date: object) -> dict[str, Any] | None:
        sec_result = calculate_latest_eps_yoy(self.sec.fetch_quarterly_history(symbol), snapshot_date)
        if sec_result is not None:
            return sec_result
        return calculate_latest_eps_yoy(self.yahoo.fetch_quarterly_history(symbol), snapshot_date)
