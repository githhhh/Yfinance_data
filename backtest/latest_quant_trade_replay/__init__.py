"""Replay historical complete-week pools with the latest quant_trade logic."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
import os
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SnapshotWeek:
    snapshot_date: str
    expected_last_trading_day: str


@dataclass
class ClipResult:
    data: dict[str, pd.DataFrame]
    max_date_before_clip: str | None
    max_date_after_clip: str | None
    has_future_data_before_clip: bool
    replay_used_clipped_data: bool


@dataclass
class SchemaAudit:
    schema_validation_status: str
    missing_critical_fields: list[str] = field(default_factory=list)
    missing_repairable_fields: list[str] = field(default_factory=list)
    missing_optional_fields: list[str] = field(default_factory=list)
    repaired_fields: list[str] = field(default_factory=list)
    repair_sources: dict[str, str] = field(default_factory=dict)
    unrepaired_fields: list[str] = field(default_factory=list)
    schema_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HistoricalPklCandidate:
    commit: str
    commit_date: str
    daily_path: str
    weekly_path: str
    daily_max_date: str | None
    weekly_max_date: str | None


CRITICAL_FIELDS = [
    "code",
    "snapshot_date",
    "signal",
    "signal_source",
    "latest_close",
    "volume_ratio",
    "ceiling",
]
REPAIRABLE_FIELDS = ["industry", "sector"]
OPTIONAL_FIELDS = [
    "eps_yoy_growth",
    "price_52_week_high",
    "dist_to_52w_high_pct",
    "ibd_entry_status",
    "ibd_candidate_rule",
    "pullback_count",
    "pullback_pct",
    "pullback_pct_off_peak",
]
EXPECTED_POOL_FIELDS = [
    "code",
    "snapshot_date",
    "signal",
    "signal_source",
    "pullback_v_is_dry",
    "ibd_candidate_rule",
    "ibd_candidate_price",
    "ibd_candidate_signal_source",
    "ibd_candidate_extra",
    "ibd_entry_valid",
    "ibd_entry_date",
    "ibd_entry_price",
    "ibd_trigger_price",
    "ibd_entry_volume_ratio",
    "ibd_entry_close_vs_trigger_pct",
    "ibd_entry_close_position",
    "ibd_entry_breakout_range_ratio",
    "ibd_entry_rule",
    "ibd_entry_reject_reason",
    "latest_close",
    "current_vs_ibd_candidate_pct",
    "ibd_entry_status",
    "volume_ratio",
    "hold_return",
    "breakout_date",
    "pct_above_ceiling",
    "touched_ema10_count",
    "mbox_count",
    "ceiling",
    "ceiling_date",
    "base_depth_pct",
    "base_mbox_count",
    "base_duration_weeks",
    "base_depth_abs",
    "C_continuous",
    "rank_C_continuous",
    "pullback_count",
    "pullback_duration_weeks",
    "pullback_pct",
    "pullback_pct_off_peak",
    "is_bullish",
    "is_priority",
    "eps_yoy_growth",
    "price_52_week_high",
    "dist_to_52w_high_pct",
    "sector",
    "industry",
]
IBD_RESOLVER_FIELDS = [
    "ibd_entry_valid",
    "ibd_entry_date",
    "ibd_entry_price",
    "ibd_trigger_price",
    "ibd_entry_volume_ratio",
    "ibd_entry_close_position",
    "ibd_entry_breakout_range_ratio",
    "ibd_entry_reject_reason",
]
VALID_IBD_ENTRY_FIELDS = [
    "ibd_entry_date",
    "ibd_entry_price",
    "ibd_trigger_price",
    "ibd_entry_volume_ratio",
    "ibd_entry_close_position",
    "ibd_entry_breakout_range_ratio",
]
CORE_NON_EMPTY_FIELDS = [
    "code",
    "snapshot_date",
    "signal",
    "latest_close",
    "volume_ratio",
    "hold_return",
    "breakout_date",
    "pct_above_ceiling",
    "touched_ema10_count",
    "mbox_count",
    "ceiling",
    "ceiling_date",
    "base_depth_pct",
    "base_mbox_count",
    "base_duration_weeks",
    "price_52_week_high",
    "dist_to_52w_high_pct",
    "pullback_count",
    "is_bullish",
    "is_priority",
]
SIGNAL_REQUIRED_FIELDS = [
    "signal_source",
    "ibd_candidate_rule",
    "ibd_candidate_price",
    "ibd_candidate_signal_source",
    "ibd_entry_valid",
    "current_vs_ibd_candidate_pct",
    "ibd_entry_status",
    "base_depth_abs",
    "C_continuous",
    "rank_C_continuous",
]
VALID_IBD_ENTRY_EXTRA_FIELDS = [
    "ibd_entry_close_vs_trigger_pct",
    "ibd_entry_rule",
]
OPTIONAL_GAP_FIELDS = [
    "pullback_v_is_dry",
    "ibd_candidate_extra",
    "pullback_duration_weeks",
    "pullback_pct",
    "pullback_pct_off_peak",
]

REPLAY_STRATEGY_ENV = {
    "STRATEGY_TYPE": "1",
    "STRATEGY_RECENT_N": "3",
    "BOX_PERIOD_L": "10",
    "BOX_PERIOD_M": "5",
    "BOX_PERIOD_S": "2",
    "LAST_PERIOD_L_BREAKOUT_RESISTANCE_COUNT": "1",
    "LAST_PERIOD_M_BREAKOUT_RESISTANCE_COUNT": "2",
    "TARGET_GROUP_NAME_DAILY": "",
    "TARGET_GROUP_NAME_WEEKLY": "",
    "TARGET_GROUP_NAME_MONTHLY": "",
    "TARGET_GROUP_NAME_VOL": "",
    "TARGET_GROUP_NAME_EPS": "",
    "TARGET_GROUP_NAME_TREND": "",
    "TARGET_GROUP_NAME_IBD_W": "",
}


NYSE_HOLIDAYS_2026 = {
    "2026-01-01",
    "2026-01-19",
    "2026-02-16",
    "2026-04-03",
    "2026-05-25",
    "2026-06-19",
    "2026-07-03",
}


def _to_date(value: str | date | datetime | pd.Timestamp) -> date:
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def _normalize_dates(index: Any) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(index, errors="coerce"))
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    return idx.normalize()


def _trading_days(start_date: str | date, end_date_exclusive: str | date) -> list[date]:
    start = _to_date(start_date)
    end = _to_date(end_date_exclusive)
    out: list[date] = []
    cur = start
    while cur < end:
        if cur.weekday() < 5 and cur.isoformat() not in NYSE_HOLIDAYS_2026:
            out.append(cur)
        cur += timedelta(days=1)
    return out


def enumerate_complete_snapshot_weeks(
    *,
    start_date: str,
    exclude_week_ending: str,
) -> list[SnapshotWeek]:
    excluded = _to_date(exclude_week_ending)
    excluded_week_start = excluded - timedelta(days=excluded.weekday())
    days = _trading_days(start_date, excluded_week_start)
    by_week: dict[date, date] = {}
    for day in days:
        week_start = day - timedelta(days=day.weekday())
        by_week[week_start] = max(by_week.get(week_start, day), day)
    return [
        SnapshotWeek(snapshot_date=last.isoformat(), expected_last_trading_day=last.isoformat())
        for _, last in sorted(by_week.items())
    ]


def select_historical_pkl_pair(
    *,
    snapshot_date: str,
    expected_last_trading_day: str,
    candidates: list[HistoricalPklCandidate],
) -> HistoricalPklCandidate | None:
    expected = pd.Timestamp(expected_last_trading_day).normalize()
    expected_week_label = expected - pd.Timedelta(days=expected.weekday())
    valid = []
    for candidate in candidates:
        if candidate.daily_max_date != expected.strftime("%Y-%m-%d"):
            continue
        if candidate.weekly_max_date is None:
            continue
        weekly_max = pd.Timestamp(candidate.weekly_max_date).normalize()
        if weekly_max < expected_week_label or weekly_max > expected:
            continue
        commit_date = pd.Timestamp(candidate.commit_date).tz_localize(None)
        if commit_date.normalize() < pd.Timestamp(snapshot_date).normalize():
            continue
        valid.append(candidate)
    valid.sort(key=lambda item: (pd.Timestamp(item.commit_date).tz_localize(None), item.commit, item.daily_path, item.weekly_path))
    return valid[0] if valid else None


def _read_env_file(env_path: str | Path | None) -> dict[str, str]:
    if env_path is None:
        return {}
    path = Path(env_path)
    if not path.exists():
        return {}
    try:
        from dotenv import dotenv_values

        return {k: str(v) for k, v in dotenv_values(path).items() if v is not None}
    except Exception:
        values: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            values[key.strip()] = value.strip().strip("\"'")
        return values


def apply_replay_strategy_env(env_path: str | Path | None = None) -> dict[str, str]:
    env_values = _read_env_file(env_path)
    applied = dict(REPLAY_STRATEGY_ENV)
    for key in REPLAY_STRATEGY_ENV:
        if key in env_values:
            applied[key] = env_values[key]
    os.environ.update(applied)
    return applied


def max_price_date(data: dict[str, pd.DataFrame]) -> str | None:
    max_date: pd.Timestamp | None = None
    for df in data.values():
        if df is None or df.empty:
            continue
        idx = _normalize_dates(df.index)
        idx = idx.dropna()
        if idx.empty:
            continue
        cur = pd.Timestamp(idx.max()).normalize()
        if max_date is None or cur > max_date:
            max_date = cur
    return None if max_date is None else max_date.strftime("%Y-%m-%d")


def clip_price_data_asof(
    data: dict[str, pd.DataFrame],
    expected_last_trading_day: str,
) -> ClipResult:
    cutoff = pd.Timestamp(expected_last_trading_day).normalize()
    before = max_price_date(data)
    clipped: dict[str, pd.DataFrame] = {}
    used_clip = False
    for code, df in data.items():
        if df is None or df.empty:
            clipped[code] = df
            continue
        idx = _normalize_dates(df.index)
        mask = idx <= cutoff
        if not bool(mask.all()):
            used_clip = True
        clipped_df = df.loc[mask].copy()
        clipped[code] = clipped_df
    after = max_price_date(clipped)
    has_future = before is not None and pd.Timestamp(before) > cutoff
    return ClipResult(
        data=clipped,
        max_date_before_clip=before,
        max_date_after_clip=after,
        has_future_data_before_clip=has_future,
        replay_used_clipped_data=used_clip,
    )


def enrich_pool_with_asof_52w_high(
    pool: pd.DataFrame,
    daily_data: dict[str, pd.DataFrame],
    expected_last_trading_day: str,
) -> pd.DataFrame:
    result = pool.copy()
    if "code" not in result.columns:
        result["price_52_week_high"] = pd.NA
        result["dist_to_52w_high_pct"] = pd.NA
        return result

    cutoff = pd.Timestamp(expected_last_trading_day).normalize()
    start = cutoff - pd.Timedelta(days=364)
    high_by_code: dict[str, float] = {}

    for code, df in daily_data.items():
        if df is None or df.empty or "High" not in df.columns:
            continue
        idx = _normalize_dates(df.index)
        high = pd.to_numeric(df["High"], errors="coerce")
        mask = (idx >= start) & (idx <= cutoff) & high.notna()
        if not bool(mask.any()):
            continue
        max_high = high.loc[mask].max()
        if pd.notna(max_high) and float(max_high) > 0:
            high_by_code[str(code)] = float(max_high)

    result["price_52_week_high"] = result["code"].astype(str).map(high_by_code)
    latest_close = pd.to_numeric(
        result.get("latest_close", pd.Series(pd.NA, index=result.index)),
        errors="coerce",
    )
    high_52w = pd.to_numeric(result["price_52_week_high"], errors="coerce")
    result["dist_to_52w_high_pct"] = ((latest_close / high_52w - 1.0) * 100.0).where(high_52w > 0, pd.NA)
    return result


def clear_snapshot_contaminated_eps(pool: pd.DataFrame) -> pd.DataFrame:
    """Clear live/current EPS facts before historical REPLAY re-resolution."""
    result = pool.copy()
    for column in (
        "eps_yoy_growth",
        "eps_yoy_growth_source",
        "eps_yoy_growth_status",
        "eps_yoy_growth_missing_reason",
        "eps_growth_type",
    ):
        if column in result.columns or column == "eps_yoy_growth":
            result[column] = pd.NA
    return result


def normalize_empty_pool_schema(pool: pd.DataFrame) -> pd.DataFrame:
    if not pool.empty:
        return pool
    return pd.DataFrame(columns=EXPECTED_POOL_FIELDS)


class ReplayPoolSink:
    """Pool sink that writes only to the replay directory and disables publish paths."""

    def __init__(self, output_path: str | Path):
        self._path = Path(output_path)

    @property
    def name(self) -> str:
        return "replay"

    @property
    def path(self) -> str:
        return str(self._path)

    def save_snapshot(self, pool: pd.DataFrame) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        pool.to_csv(self._path, index=False, encoding="utf-8-sig")

    def ensure_current_snapshot(self) -> pd.DataFrame:
        return pd.read_csv(self._path, dtype={"code": str}, encoding="utf-8-sig")

    def publish(self) -> None:
        raise RuntimeError("publish is disabled for replay")

    def commit(self) -> None:
        raise RuntimeError("commit is disabled for replay")

    def validate_for_publish(self) -> None:
        raise RuntimeError("validate_for_publish is disabled for replay")

    def load_actionable_codes(self) -> list[str]:
        raise RuntimeError("load_actionable_codes is disabled for replay")


def _truthy_series(series: pd.Series) -> pd.Series:
    return series.map(lambda value: str(value).strip().lower() in {"true", "1", "1.0"})


def _has_value_series(series: pd.Series) -> pd.Series:
    return series.map(lambda value: not (pd.isna(value) or str(value).strip() == ""))


def _signal_candidate_mask(pool: pd.DataFrame) -> pd.Series:
    if "signal" not in pool.columns or "ibd_candidate_rule" not in pool.columns:
        return pd.Series(False, index=pool.index)
    return _truthy_series(pool["signal"]) & _has_value_series(pool["ibd_candidate_rule"])


def _empty_mask(pool: pd.DataFrame, col: str) -> pd.Series:
    if col not in pool.columns:
        return pd.Series(True, index=pool.index)
    series = pool[col]
    return series.isna() | series.astype(str).str.strip().eq("")


def _add_count(target: dict[str, int], key: str, count: int) -> None:
    if count:
        target[key] = int(count)


def audit_pool_null_semantics(
    pool: pd.DataFrame,
    *,
    expected_fields: list[str] | None = None,
) -> dict[str, Any]:
    """Classify empty pool values by whether they violate replay semantics."""

    missing_fields = [field for field in expected_fields or [] if field not in pool.columns]
    abnormal: dict[str, int] = {}
    normal: dict[str, int] = {}
    repairable_fallbacks: dict[str, int] = {}
    optional_gaps: dict[str, int] = {}

    signal_mask = _truthy_series(pool["signal"]) if "signal" in pool.columns else pd.Series(False, index=pool.index)
    valid_entry_mask = (
        signal_mask & _truthy_series(pool["ibd_entry_valid"])
        if "ibd_entry_valid" in pool.columns
        else pd.Series(False, index=pool.index)
    )
    invalid_entry_mask = signal_mask & ~valid_entry_mask

    for field in CORE_NON_EMPTY_FIELDS:
        if field in pool.columns:
            _add_count(abnormal, field, int(_empty_mask(pool, field).sum()))

    if "signal_source" in pool.columns:
        empty = _empty_mask(pool, "signal_source")
        _add_count(abnormal, "signal_source_signal", int((signal_mask & empty).sum()))
        _add_count(normal, "signal_source_non_signal", int((~signal_mask & empty).sum()))

    for field in SIGNAL_REQUIRED_FIELDS:
        if field in pool.columns:
            empty = _empty_mask(pool, field)
            _add_count(abnormal, f"{field}_signal", int((signal_mask & empty).sum()))
            _add_count(normal, f"{field}_non_signal", int((~signal_mask & empty).sum()))

    for field in VALID_IBD_ENTRY_FIELDS + VALID_IBD_ENTRY_EXTRA_FIELDS:
        if field in pool.columns:
            empty = _empty_mask(pool, field)
            _add_count(abnormal, f"{field}_valid_entry", int((valid_entry_mask & empty).sum()))
            _add_count(normal, f"{field}_invalid_or_non_signal", int((~valid_entry_mask & empty).sum()))

    if "ibd_entry_reject_reason" in pool.columns:
        empty = _empty_mask(pool, "ibd_entry_reject_reason")
        _add_count(abnormal, "ibd_entry_reject_reason_invalid_entry", int((invalid_entry_mask & empty).sum()))
        _add_count(normal, "ibd_entry_reject_reason_valid_or_non_signal", int((~invalid_entry_mask & empty).sum()))

    if "eps_yoy_growth" in pool.columns:
        empty = _empty_mask(pool, "eps_yoy_growth")
        _add_count(abnormal, "eps_yoy_growth_signal", int((signal_mask & empty).sum()))
        _add_count(normal, "eps_yoy_growth_non_signal", int((~signal_mask & empty).sum()))

    for field in REPAIRABLE_FIELDS:
        if field in pool.columns:
            empty = _empty_mask(pool, field)
            unknown = pool[field].astype(str).str.strip().eq("Unknown")
            _add_count(abnormal, field, int(empty.sum()))
            _add_count(repairable_fallbacks, field, int(unknown.sum()))

    for field in OPTIONAL_GAP_FIELDS:
        if field in pool.columns:
            _add_count(optional_gaps, field, int(_empty_mask(pool, field).sum()))

    status = "failed" if missing_fields or abnormal else "passed"
    return {
        "status": status,
        "row_count": int(len(pool)),
        "column_count": int(len(pool.columns)),
        "missing_fields": missing_fields,
        "abnormal_empty_fields": abnormal,
        "normal_empty_fields": normal,
        "repairable_fallback_fields": repairable_fallbacks,
        "optional_gap_fields": optional_gaps,
        "signal_rows": int(signal_mask.sum()),
        "valid_ibd_entry_rows": int(valid_entry_mask.sum()),
        "invalid_ibd_entry_rows": int(invalid_entry_mask.sum()),
    }


def audit_pool_schema(pool: pd.DataFrame) -> SchemaAudit:
    missing_critical = []
    for col in CRITICAL_FIELDS:
        if col not in pool.columns:
            missing_critical.append(col)
            continue
        if col == "signal_source":
            signal_mask = _truthy_series(pool["signal"]) if "signal" in pool.columns else pd.Series(True, index=pool.index)
            if pool.loc[signal_mask, col].isna().any():
                missing_critical.append(col)
            continue
        if pool[col].isna().any():
            missing_critical.append(col)
    signal_candidate_mask = _signal_candidate_mask(pool)
    if signal_candidate_mask.any():
        for col in IBD_RESOLVER_FIELDS:
            if col not in pool.columns:
                missing_critical.append(col)
                continue
        if "ibd_entry_valid" in pool.columns and pool.loc[signal_candidate_mask, "ibd_entry_valid"].isna().any():
            missing_critical.append("ibd_entry_valid")
        if "ibd_entry_valid" in pool.columns:
            valid_mask = signal_candidate_mask & _truthy_series(pool["ibd_entry_valid"])
            for col in VALID_IBD_ENTRY_FIELDS:
                if col not in pool.columns:
                    continue
                if pool.loc[valid_mask, col].isna().any():
                    missing_critical.append(col)
            invalid_mask = signal_candidate_mask & ~_truthy_series(pool["ibd_entry_valid"])
            if "ibd_entry_reject_reason" in pool.columns and (
                pool.loc[invalid_mask, "ibd_entry_reject_reason"].isna()
                | (pool.loc[invalid_mask, "ibd_entry_reject_reason"].astype(str).str.strip() == "")
            ).any():
                missing_critical.append("ibd_entry_reject_reason")
    missing_critical = sorted(set(missing_critical))
    missing_repairable = [col for col in REPAIRABLE_FIELDS if col not in pool.columns]
    missing_optional = [col for col in OPTIONAL_FIELDS if col not in pool.columns]
    repaired_fields: list[str] = []
    repair_sources: dict[str, str] = {}

    for col in missing_repairable:
        repaired_fields.append(col)
        repair_sources[col] = "default_unknown_repair_for_research_schema"

    status = "passed"
    unrepaired = []
    if missing_critical:
        status = "failed_critical_schema"
        unrepaired = list(missing_critical)
    elif missing_repairable or missing_optional:
        status = "passed_with_repairs_or_optional_gaps"

    notes = []
    if missing_repairable:
        notes.append("Repairable display fields may be filled with Unknown without blocking replay.")
    if missing_optional:
        notes.append("Optional fields affect explanation quality but do not block replay.")

    return SchemaAudit(
        schema_validation_status=status,
        missing_critical_fields=missing_critical,
        missing_repairable_fields=missing_repairable,
        missing_optional_fields=missing_optional,
        repaired_fields=repaired_fields,
        repair_sources=repair_sources,
        unrepaired_fields=unrepaired,
        schema_notes=notes,
    )


def repair_research_fields(pool: pd.DataFrame) -> pd.DataFrame:
    result = pool.copy()
    for col in REPAIRABLE_FIELDS:
        if col not in result.columns:
            result[col] = "Unknown"
        else:
            result[col] = result[col].fillna("Unknown")
    return result
