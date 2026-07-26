from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo
import zlib

import pandas as pd

import numpy as np

from dashboard.field_config import (
    BOOLEAN_FIELDS,
    DATE_FIELDS,
    FIELD_CONFIG,
    NUMBER_FIELDS,
    QUALITY_ALIASES,
    QUALITY_ORDER,
    get_field_label,
)


@dataclass(frozen=True)
class FilterSpec:
    field: str
    operator: str
    value: Any = None
    value2: Any = None
    enabled: bool = True
    label: str | None = None


@dataclass(frozen=True)
class SortSpec:
    field: str
    direction: str = "asc"
    enabled: bool = True


REQUIRED_CORE_FIELDS = {
    "code",
    "signal",
    "latest_close",
    "ibd_candidate_price",
    "ibd_entry_valid",
    "ibd_entry_status",
    "current_vs_ibd_candidate_pct",
    "ibd_candidate_rule",
    "ibd_entry_volume_ratio",
    "ibd_entry_reject_reason",
    "volume_ratio",
    "rank_C_continuous",
    "C_continuous",
}


def validate_pool_schema(df: pd.DataFrame) -> None:
    missing = REQUIRED_CORE_FIELDS - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Schema / Data Error: missing required IBD Review columns: {missing_list}")


def validate_pool_semantics(df: pd.DataFrame) -> None:
    if df.empty or len(df) == 0:
        raise ValueError("Schema / Data Error: dataset cannot be empty")

    if "code" not in df.columns:
        raise ValueError("Schema / Data Error: missing required IBD Review columns: code")

    for val in df["code"]:
        if pd.isna(val) or str(val).strip() == "" or str(val).strip().lower() == "nan":
            raise ValueError("Schema / Data Error: code cannot be empty")

    if df["code"].nunique() != len(df):
        raise ValueError("Schema / Data Error: code cannot be duplicate")

    for val in df["signal"]:
        bool_val = _to_bool_or_na(val)
        if bool_val is not True and bool_val is not False:
            raise ValueError("Schema / Data Error: signal must be convertible to valid boolean")

    active_mask = df["signal"].map(_to_bool_or_na) == True
    active_df = df[active_mask]

    for col in ["latest_close", "ibd_candidate_price", "current_vs_ibd_candidate_pct", "rank_C_continuous", "C_continuous", "volume_ratio"]:
        nums = pd.to_numeric(active_df[col], errors="coerce")
        if nums.isna().any() or not np.isfinite(nums).all():
            raise ValueError(f"Schema / Data Error: {col} must be valid finite numerical values for active signals")

    if (pd.to_numeric(active_df["ibd_candidate_price"], errors="coerce") <= 0).any():
        raise ValueError("Schema / Data Error: ibd_candidate_price must be > 0 for active signals")

    for val in active_df["ibd_candidate_rule"]:
        if pd.isna(val) or str(val).strip() == "" or str(val).strip().lower() == "nan":
            raise ValueError("Schema / Data Error: ibd_candidate_rule cannot be empty for active signals")

    valid_statuses = {"UNCONFIRMED", "BELOW_TRIGGER", "ACTIONABLE", "EXTENDED"}
    if not active_df["ibd_entry_status"].isin(valid_statuses).all():
        raise ValueError("Schema / Data Error: ibd_entry_status must belong to the four states for active signals")

    status_sum = active_df["ibd_entry_status"].isin(valid_statuses).sum()
    if status_sum != len(active_df):
        raise ValueError("Schema / Data Error: sum of four states must strictly equal Active Signals")

    for _, row in active_df.iterrows():
        valid = _to_bool_or_na(row.get("ibd_entry_valid"))
        pct = float(row["current_vs_ibd_candidate_pct"])
        status = row["ibd_entry_status"]
        if valid is not True:
            expected = "UNCONFIRMED"
        elif pct < 0:
            expected = "BELOW_TRIGGER"
        elif pct <= 5.0:
            expected = "ACTIONABLE"
        else:
            expected = "EXTENDED"
        if status != expected:
            raise ValueError(
                f"Schema / Data Error: status formula mismatch for {row.get('code')}: got {status}, expected {expected}"
            )

    non_active_df = df[~active_mask]
    if not non_active_df.empty and "ibd_entry_status" in non_active_df.columns:
        for val in non_active_df["ibd_entry_status"]:
            if pd.notna(val) and str(val).strip() != "" and str(val).strip().lower() not in ("nan", "none"):
                raise ValueError("Schema / Data Error: ibd_entry_status must be empty for non-signal rows")


def load_pool_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    raw_df = pd.read_csv(csv_path, encoding="utf-8-sig")
    raw_df.columns = [str(column).lstrip("\ufeff") for column in raw_df.columns]
    validate_pool_schema(raw_df)
    normalized = normalize_pool_df(raw_df)
    validate_pool_semantics(normalized)
    return normalized


def normalize_pool_df(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.columns = [str(column).lstrip("\ufeff") for column in result.columns]

    for column in BOOLEAN_FIELDS.intersection(result.columns):
        result[column] = result[column].map(_to_bool_or_na).astype("object")

    for column in NUMBER_FIELDS.intersection(result.columns):
        result[column] = pd.to_numeric(result[column], errors="coerce")

    for column in DATE_FIELDS.intersection(result.columns):
        result[column] = pd.to_datetime(result[column], errors="coerce")

    if "base_duration_weeks" not in result.columns and {"breakout_date", "ceiling_date"}.issubset(result.columns):
        duration_days = (result["breakout_date"] - result["ceiling_date"]).dt.days
        result["base_duration_weeks"] = (duration_days / 7).round()

    if {"latest_close", "price_52_week_high"}.issubset(result.columns):
        latest = pd.to_numeric(result["latest_close"], errors="coerce")
        high_52w = pd.to_numeric(result["price_52_week_high"], errors="coerce")
        dist = (latest / high_52w - 1.0) * 100.0
        result["dist_to_52w_high_pct"] = dist.where(latest.notna() & high_52w.gt(0), pd.NA)
    elif "dist_to_52w_high_pct" not in result.columns:
        result["dist_to_52w_high_pct"] = pd.Series(pd.NA, index=result.index)

    for column in result.columns:
        if column not in BOOLEAN_FIELDS and column not in NUMBER_FIELDS and column not in DATE_FIELDS:
            result[column] = result[column].replace("", pd.NA)

    if "ibd_entry_status" not in result.columns:
        result["ibd_entry_status"] = result.apply(_compute_ibd_entry_status, axis=1)

    if "ibd_entry_vol_or_reject" not in result.columns:
        def _vol_or_reject(row: pd.Series) -> str | Any:
            valid = _to_bool_or_na(row.get("ibd_entry_valid"))
            if valid is not True:
                reason = str(row.get("ibd_entry_reject_reason", "")).strip()
                if not reason or reason.lower() == "nan":
                    return "Volume not confirmed"
                return reason
            vol = pd.to_numeric(row.get("ibd_entry_volume_ratio"), errors="coerce")
            if pd.isna(vol):
                return "n/a"
            return f"{vol:.2f}x"
        result["ibd_entry_vol_or_reject"] = result.apply(_vol_or_reject, axis=1)
    if "ibd_breakout_quality" not in result.columns:
        result["ibd_breakout_quality"] = result.apply(_compute_breakout_quality, axis=1)
    else:
        result["ibd_breakout_quality"] = result["ibd_breakout_quality"].replace(QUALITY_ALIASES)

    return result


def _compute_breakout_quality(row: pd.Series) -> str | Any:
    close_vs_trigger = pd.to_numeric(row.get("ibd_entry_close_vs_trigger_pct"), errors="coerce")
    pos = pd.to_numeric(row.get("ibd_entry_close_position"), errors="coerce")
    rr = pd.to_numeric(row.get("ibd_entry_breakout_range_ratio"), errors="coerce")
    if pd.isna(pos) or pd.isna(rr):
        return pd.NA
    if not pd.isna(close_vs_trigger) and close_vs_trigger <= 0:
        return pd.NA
    if rr <= 0:
        return pd.NA
    if rr > 1.0:
        return "Powerful Breakout"
    if pos >= 0.80 and rr >= 0.50:
        return "Strong Close"
    if pos >= 0.80 and rr < 0.50:
        return "Constructive Close (Tight)"
    if 0.65 <= pos < 0.80:
        return "Constructive Close"
    if pos < 0.65:
        return "Weak Close"
    return pd.NA


def apply_filters(df: pd.DataFrame, filters: list[FilterSpec]) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    mask = pd.Series(True, index=df.index)
    for spec in filters:
        if not spec.enabled:
            continue
        _ensure_column(df, spec.field)
        mask &= _filter_mask(df[spec.field], spec)

    return df.loc[mask].copy()


def apply_sort(df: pd.DataFrame, sort_specs: list[SortSpec]) -> pd.DataFrame:
    enabled_specs = [spec for spec in sort_specs if spec.enabled and spec.field]
    if not enabled_specs:
        return df.copy()

    work = df.copy()
    temp_cols = []
    by_cols = []
    ascending_list = []

    for i, spec in enumerate(enabled_specs):
        _ensure_column(work, spec.field)
        if spec.field == "ibd_breakout_quality":
            temp_col = f"_quality_rank_{i}"
            work[temp_col] = work["ibd_breakout_quality"].map(QUALITY_ORDER)
            temp_cols.append(temp_col)
            by_cols.append(temp_col)
        else:
            by_cols.append(spec.field)
        ascending_list.append(spec.direction.lower() != "desc")

    sorted_df = work.sort_values(
        by=by_cols,
        ascending=ascending_list,
        na_position="last",
        kind="mergesort",
    ).copy()

    return sorted_df.drop(columns=temp_cols, errors="ignore")


STATUS_REVIEW_ORDER = {
    "ACTIONABLE": 0,
    "UNCONFIRMED": 1,
    "BELOW_TRIGGER": 2,
    "EXTENDED": 3,
    "Pending": 4,
}


def apply_default_review_order(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    work = df.copy()
    status_col = (
        work["ibd_entry_status"].fillna("Pending")
        if "ibd_entry_status" in work.columns
        else pd.Series("Pending", index=work.index)
    )
    work["_status_rank"] = status_col.map(lambda s: STATUS_REVIEW_ORDER.get(str(s), 99))
    sort_cols = ["_status_rank"]
    ascending = [True]
    if "ibd_breakout_quality" in work.columns:
        work["_quality_rank"] = work["ibd_breakout_quality"].map(QUALITY_ORDER)
        sort_cols.append("_quality_rank")
        ascending.append(True)
    if "code" in work.columns:
        sort_cols.append("code")
        ascending.append(True)
    sorted_df = work.sort_values(by=sort_cols, ascending=ascending, na_position="last", kind="mergesort")
    return sorted_df.drop(columns=["_status_rank", "_quality_rank"], errors="ignore")


def apply_c_rank_mode(df: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
    _ensure_column(df, "signal")
    _ensure_column(df, "rank_C_continuous")

    ranked = apply_filters(df, [FilterSpec("signal", "is true")])
    ranked = apply_sort(ranked, [SortSpec("rank_C_continuous", "asc")])
    if limit is not None:
        ranked = ranked.head(limit)
    return ranked.copy()



def build_kpis(df: pd.DataFrame) -> dict[str, float | int | None]:
    row_count = len(df)
    return {
        "filtered_rows": row_count,
        "median_current_vs_ibd_candidate_pct": _median_or_none(df, "current_vs_ibd_candidate_pct"),
        "median_ibd_entry_volume_ratio": _median_or_none(df, "ibd_entry_volume_ratio"),
        "median_volume_ratio": _median_or_none(df, "volume_ratio"),
    }


def build_chart_data(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "route_quality": _build_route_quality_data(df),
        "trend_volume_map": _build_trend_volume_map_data(df),
        "volume_close_matrix": _build_volume_close_matrix_data(df),
    }


def _ensure_column(df: pd.DataFrame, field: str) -> None:
    if field not in df.columns:
        raise KeyError(f"Missing field: {field}")


def _to_bool_or_na(value: Any) -> bool | pd._libs.missing.NAType:
    if pd.isna(value):
        return pd.NA
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1"}:
        return True
    if text in {"false", "f", "no", "n", "0"}:
        return False
    return pd.NA


def _filter_mask(series: pd.Series, spec: FilterSpec) -> pd.Series:
    operator = spec.operator.lower().strip()

    if operator in {"is true", "true"}:
        return _true_mask(series)
    if operator in {"is false", "false"}:
        return _false_mask(series)
    if operator in {"equals", "=", "=="}:
        return series.eq(spec.value).fillna(False)
    if operator == "in":
        return series.isin(_as_list(spec.value)).fillna(False)
    if operator == "not in":
        return (~series.isin(_as_list(spec.value))).fillna(False)
    if operator in {">", "gt"}:
        threshold = _coerce_float(spec.value)
        if threshold is None:
            return _false_mask_like(series)
        return pd.to_numeric(series, errors="coerce").gt(threshold).fillna(False)
    if operator in {"<", "lt"}:
        threshold = _coerce_float(spec.value)
        if threshold is None:
            return _false_mask_like(series)
        return pd.to_numeric(series, errors="coerce").lt(threshold).fillna(False)
    if operator == ">=":
        threshold = _coerce_float(spec.value)
        if threshold is None:
            return _false_mask_like(series)
        return pd.to_numeric(series, errors="coerce").ge(threshold).fillna(False)
    if operator == "<=":
        threshold = _coerce_float(spec.value)
        if threshold is None:
            return _false_mask_like(series)
        return pd.to_numeric(series, errors="coerce").le(threshold).fillna(False)
    if operator == "between":
        if spec.field in DATE_FIELDS or pd.api.types.is_datetime64_any_dtype(series):
            start = _coerce_timestamp(spec.value)
            end = _coerce_timestamp(spec.value2)
            if start is None or end is None:
                return _false_mask_like(series)
            dates = pd.to_datetime(series, errors="coerce")
            return dates.between(start, end, inclusive="both").fillna(False)
        start = _coerce_float(spec.value)
        end = _coerce_float(spec.value2)
        if start is None or end is None:
            return _false_mask_like(series)
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.between(start, end, inclusive="both").fillna(False)
    if operator == "after":
        threshold = _coerce_timestamp(spec.value)
        if threshold is None:
            return _false_mask_like(series)
        return pd.to_datetime(series, errors="coerce").gt(threshold).fillna(False)
    if operator == "before":
        threshold = _coerce_timestamp(spec.value)
        if threshold is None:
            return _false_mask_like(series)
        return pd.to_datetime(series, errors="coerce").lt(threshold).fillna(False)
    if operator == "contains":
        return series.astype("string").str.contains(str(spec.value), case=False, regex=False, na=False)
    if operator == "startswith":
        return series.astype("string").str.lower().str.startswith(str(spec.value).lower(), na=False)
    if operator in {"is empty", "empty"}:
        return _empty_mask(series)
    if operator in {"not empty", "non-empty"}:
        return ~_empty_mask(series)

    raise ValueError(f"Unsupported operator: {spec.operator}")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _coerce_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(result) else result


def _coerce_timestamp(value: Any) -> pd.Timestamp | None:
    result = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(result) else result


def _false_mask_like(series: pd.Series) -> pd.Series:
    return pd.Series(False, index=series.index)


def _empty_mask(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype("string").str.strip().eq("")


def _true_mask(series: pd.Series) -> pd.Series:
    return pd.Series(series.to_numpy(dtype=bool, na_value=False), index=series.index)


def _false_mask(series: pd.Series) -> pd.Series:
    return ~pd.Series(series.to_numpy(dtype=bool, na_value=True), index=series.index)


def _median_or_none(df: pd.DataFrame, field: str) -> float | None:
    if field not in df.columns:
        return None
    value = pd.to_numeric(df[field], errors="coerce").median()
    return None if pd.isna(value) else float(value)


def _build_route_quality_data(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "ibd_candidate_rule",
        "valid_count",
        "invalid_count",
        "total_count",
        "valid_rate_pct",
        "median_ibd_entry_volume_ratio",
        "median_ibd_entry_close_position",
        "median_volume_ratio",
        "median_ibd_entry_breakout_range_ratio",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    needed = [
        "ibd_candidate_rule",
        "ibd_entry_valid",
        "ibd_entry_volume_ratio",
        "ibd_entry_close_position",
        "volume_ratio",
        "ibd_entry_breakout_range_ratio",
    ]
    available = [c for c in needed if c in df.columns]
    working = df[available].copy()
    for column in [
        "ibd_entry_volume_ratio",
        "ibd_entry_close_position",
        "volume_ratio",
        "ibd_entry_breakout_range_ratio",
    ]:
        if column not in working.columns:
            working[column] = pd.NA
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working["ibd_candidate_rule"] = _label_series(working, "ibd_candidate_rule")
    working = working[working["ibd_candidate_rule"] != "(empty)"].copy()
    if working.empty:
        return pd.DataFrame(columns=columns)
    working["valid"] = _true_mask(working.get("ibd_entry_valid", pd.Series(index=working.index, dtype="object"))).astype(int)

    grouped = working.groupby(["ibd_candidate_rule"], dropna=False).agg(
        total_count=("valid", "size"),
        valid_count=("valid", "sum"),
        median_ibd_entry_volume_ratio=("ibd_entry_volume_ratio", "median"),
        median_ibd_entry_close_position=("ibd_entry_close_position", "median"),
        median_volume_ratio=("volume_ratio", "median"),
        median_ibd_entry_breakout_range_ratio=("ibd_entry_breakout_range_ratio", "median"),
    )
    grouped["invalid_count"] = grouped["total_count"] - grouped["valid_count"]
    grouped["valid_rate_pct"] = (grouped["valid_count"] / grouped["total_count"] * 100).round(2)
    return (
        grouped.reset_index()[columns]
        .sort_values(["total_count", "ibd_candidate_rule"], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )


def _build_trend_volume_map_data(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "code",
        "sector",
        "industry",
        "signal_source",
        "ibd_candidate_rule",
        "ibd_entry_valid",
        "entry_status",
        "pullback_v_is_dry",
        "dry_status",
        "touched_ema10_count",
        "touched_ema10_jittered",
        "volume_ratio",
        "ibd_entry_volume_ratio",
        "ibd_entry_close_position",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    available = [c for c in columns if c != "touched_ema10_jittered" and c in df.columns]
    working = df[available].copy()
    for column in [c for c in columns if c != "touched_ema10_jittered"]:
        if column not in working.columns:
            working[column] = pd.NA
    working["touched_ema10_count"] = pd.to_numeric(working["touched_ema10_count"], errors="coerce")
    working["volume_ratio"] = pd.to_numeric(working["volume_ratio"], errors="coerce")
    valid_mask = _true_mask(working.get("ibd_entry_valid", pd.Series(index=working.index, dtype="object")))
    working = working[valid_mask].dropna(subset=["touched_ema10_count", "volume_ratio"]).copy()
    if working.empty:
        working["touched_ema10_jittered"] = pd.Series(dtype="float64")
        return working[columns].copy()

    working["entry_status"] = working["ibd_entry_valid"].map(_entry_status)
    working["dry_status"] = working["pullback_v_is_dry"].map(_dry_status)
    jitter = working["code"].apply(
        lambda x: (zlib.crc32(str(x).encode("utf-8")) % 301) / 300.0 * 0.3 - 0.15 if pd.notna(x) else 0.0
    )
    working["touched_ema10_jittered"] = working["touched_ema10_count"] + jitter
    return working[columns].copy()


def _build_volume_close_matrix_data(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "code",
        "sector",
        "industry",
        "signal_source",
        "ibd_candidate_rule",
        "ibd_entry_valid",
        "entry_status",
        "volume_ratio",
        "ibd_entry_volume_ratio",
        "ibd_entry_close_position",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    available = [c for c in columns if c in df.columns]
    working = df[available].copy()
    for column in columns:
        if column not in working.columns:
            working[column] = pd.NA

    working["volume_ratio"] = pd.to_numeric(working["volume_ratio"], errors="coerce")
    working["ibd_entry_close_position"] = pd.to_numeric(working["ibd_entry_close_position"], errors="coerce")
    working = working.dropna(subset=["volume_ratio", "ibd_entry_close_position"]).copy()
    if working.empty:
        return working[columns].copy()

    working["entry_status"] = working["ibd_entry_valid"].map(_entry_status)
    return working[columns].copy()





def _label_series(df: pd.DataFrame, field: str) -> pd.Series:
    if field not in df.columns:
        return pd.Series("(empty)", index=df.index, dtype="object")
    return df[field].fillna("(empty)").replace("", "(empty)")


def _entry_status(value: Any) -> str:
    if pd.isna(value):
        return "Pending"
    return "IBD valid" if bool(value) is True else "IBD invalid"


def _dry_status(value: Any) -> str:
    if pd.isna(value):
        return "n/a"
    return "Dry pullback" if bool(value) is True else "Not dry"



def _compute_ibd_entry_status(row: pd.Series) -> str | Any:
    signal = _to_bool_or_na(row.get("signal"))
    if signal is not True:
        return pd.NA
    pct = pd.to_numeric(row.get("current_vs_ibd_candidate_pct"), errors="coerce")
    if pd.isna(pct) or not np.isfinite(pct):
        return pd.NA
    valid = _to_bool_or_na(row.get("ibd_entry_valid"))
    if valid is not True:
        return "UNCONFIRMED"
    if pct < 0:
        return "BELOW_TRIGGER"
    if pct <= 5.0:
        return "ACTIONABLE"
    return "EXTENDED"


def build_entry_status_counts(signal_df: pd.DataFrame) -> dict[str, int]:
    if "signal" in signal_df.columns:
        signal_df = signal_df[signal_df["signal"] == True]
    statuses = ["ACTIONABLE", "UNCONFIRMED", "BELOW_TRIGGER", "EXTENDED"]
    counts = {status: 0 for status in statuses}
    unconfirmed_within_3pct = 0
    if not signal_df.empty and "ibd_entry_status" in signal_df.columns:
        vc = signal_df["ibd_entry_status"].value_counts(dropna=True)
        for status in statuses:
            counts[status] = int(vc.get(status, 0))
        if "current_vs_ibd_candidate_pct" in signal_df.columns:
            unconf_mask = (signal_df["ibd_entry_status"] == "UNCONFIRMED") & pd.to_numeric(
                signal_df["current_vs_ibd_candidate_pct"], errors="coerce"
            ).between(0.0, 3.0, inclusive="both")
            unconfirmed_within_3pct = int(unconf_mask.sum())
    total = len(signal_df)
    return {"All": total, "ALL": total, "unconfirmed_within_3pct": unconfirmed_within_3pct, **counts}


def build_snapshot_freshness(
    snapshot_date: Any,
    today: date | datetime | str | None = None,
) -> dict[str, Any]:
    if today is None:
        today_date = datetime.now(ZoneInfo("America/New_York")).date()
    elif isinstance(today, str):
        try:
            today_date = datetime.strptime(today.strip().split(" ")[0].split("T")[0], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            today_date = datetime.now(ZoneInfo("America/New_York")).date()
    elif isinstance(today, datetime):
        today_date = today.date()
    elif isinstance(today, date):
        today_date = today
    else:
        today_date = datetime.now(ZoneInfo("America/New_York")).date()

    if snapshot_date is None or pd.isna(snapshot_date) or str(snapshot_date).strip() in ("", "N/A", "Unknown", "nan", "None"):
        return {
            "status": "UNKNOWN",
            "label": "Unknown",
            "age_days": None,
            "snapshot_date_str": "N/A",
            "header_html": "Snapshot Unknown",
        }

    try:
        s_str = str(snapshot_date).strip().split(" ")[0].split("T")[0]
        s_date = datetime.strptime(s_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return {
            "status": "UNKNOWN",
            "label": "Unknown",
            "age_days": None,
            "snapshot_date_str": "N/A",
            "header_html": "Snapshot Unknown",
        }

    age_days = max((today_date - s_date).days, 0)
    if age_days <= 3:
        status = "FRESH"
        label = "Fresh"
        color = "#2e7d32"
    elif age_days <= 6:
        status = "AGING"
        label = "Aging"
        color = "#f57c00"
    else:
        status = "STALE"
        label = "Stale"
        color = "#c62828"

    return {
        "status": status,
        "label": label,
        "age_days": age_days,
        "snapshot_date_str": s_str,
        "header_html": f'Snapshot <b>{s_str}</b> · {age_days}d old · <span style="color:{color}; font-weight:600;">{label}</span>',
    }


def filter_unconfirmed_near_trigger(
    df: pd.DataFrame,
    selected_status: str,
    near_trigger_only: bool,
) -> pd.DataFrame:
    if selected_status != "UNCONFIRMED" or not near_trigger_only or df.empty:
        return df
    if "ibd_entry_status" not in df.columns or "current_vs_ibd_candidate_pct" not in df.columns:
        return df
    distance = pd.to_numeric(
        df["current_vs_ibd_candidate_pct"],
        errors="coerce",
    )
    mask = (
        df["ibd_entry_status"].eq("UNCONFIRMED")
        & distance.notna()
        & distance.between(0.0, 3.0, inclusive="both")
    )
    return df[mask]
