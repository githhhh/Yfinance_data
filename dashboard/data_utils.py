from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from dashboard.field_config import (
    BOOLEAN_FIELDS,
    DATE_FIELDS,
    FIELD_CONFIG,
    NUMBER_FIELDS,
    PRESETS,
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


def load_pool_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return normalize_pool_df(pd.read_csv(csv_path, encoding="utf-8-sig"))


def normalize_pool_df(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.columns = [str(column).lstrip("\ufeff") for column in result.columns]

    for column in BOOLEAN_FIELDS.intersection(result.columns):
        result[column] = result[column].map(_to_bool_or_na).astype("object")

    for column in NUMBER_FIELDS.intersection(result.columns):
        result[column] = pd.to_numeric(result[column], errors="coerce")

    for column in DATE_FIELDS.intersection(result.columns):
        result[column] = pd.to_datetime(result[column], errors="coerce")

    for column in result.columns:
        if column not in BOOLEAN_FIELDS and column not in NUMBER_FIELDS and column not in DATE_FIELDS:
            result[column] = result[column].replace("", pd.NA)

    return result





def combine_filter_specs(preset_filters: list[FilterSpec], ui_filters: list[FilterSpec]) -> list[FilterSpec]:
    combined: list[FilterSpec] = []
    seen: set[tuple[str, str, str, str, bool]] = set()
    for spec in preset_filters + ui_filters:
        key = (spec.field, spec.operator, repr(spec.value), repr(spec.value2), spec.enabled)
        if key in seen:
            continue
        combined.append(spec)
        seen.add(key)
    return combined


def build_preset_filters(preset_key: str) -> list[FilterSpec]:
    preset = _get_preset(preset_key)
    return [
        FilterSpec(
            field=spec["field"],
            operator=spec["operator"],
            value=spec.get("value"),
            value2=spec.get("value2"),
        )
        for spec in preset["filters"]
    ]


def build_preset_sort(preset_key: str) -> list[SortSpec]:
    preset = _get_preset(preset_key)
    return [SortSpec(field=spec["field"], direction=spec["direction"]) for spec in preset["sort"]]


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

    for spec in enabled_specs:
        _ensure_column(df, spec.field)

    return df.sort_values(
        by=[spec.field for spec in enabled_specs],
        ascending=[spec.direction.lower() != "desc" for spec in enabled_specs],
        na_position="last",
        kind="mergesort",
    ).copy()


def apply_c_rank_mode(df: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
    _ensure_column(df, "signal")
    _ensure_column(df, "rank_C_continuous")

    ranked = apply_filters(df, [FilterSpec("signal", "is true")])
    ranked = apply_sort(ranked, [SortSpec("rank_C_continuous", "asc")])
    if limit is not None:
        ranked = ranked.head(limit)
    return ranked.copy()


def build_active_filter_summary(filters: list[FilterSpec], sort_specs: list[SortSpec]) -> list[str]:
    chips: list[str] = []
    for spec in filters:
        if not spec.enabled:
            continue
        label = spec.label or get_field_label(spec.field)
        chips.append(_describe_filter(label, spec))
    active_sorts = [spec for spec in sort_specs if spec.enabled and spec.field]
    if active_sorts:
        chips.append(
            "Sort: "
            + " -> ".join(f"{get_field_label(spec.field)} {spec.direction.lower()}" for spec in active_sorts)
        )
    return chips


def build_kpis(df: pd.DataFrame) -> dict[str, float | int | None]:
    row_count = len(df)
    valid_count = _true_mask(df.get("ibd_entry_valid", pd.Series(dtype="object"))).sum()
    return {
        "filtered_rows": row_count,
        "ibd_valid_rate_pct": round((valid_count / row_count) * 100, 2) if row_count else 0.0,
        "median_ibd_entry_volume_ratio": _median_or_none(df, "ibd_entry_volume_ratio"),
        "median_ibd_entry_close_vs_trigger_pct": _median_or_none(df, "ibd_entry_close_vs_trigger_pct"),
    }


def build_chart_data(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "signal_quality_matrix": _build_signal_quality_matrix_data(df),
        "structure_action_map": _build_structure_action_map_data(df),
        "sector_concentration": _build_sector_concentration_data(df),
        "ibd_valid_rate_by_signal_source": _build_valid_rate_data(df),
        "volume_close_strength": _build_scatter_data(df),
    }


def _get_preset(preset_key: str) -> dict[str, Any]:
    if preset_key not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_key}")
    return PRESETS[preset_key]


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
    return series.fillna(False).astype(bool)


def _false_mask(series: pd.Series) -> pd.Series:
    return ~series.fillna(True).astype(bool)


def _median_or_none(df: pd.DataFrame, field: str) -> float | None:
    if field not in df.columns:
        return None
    value = pd.to_numeric(df[field], errors="coerce").median()
    return None if pd.isna(value) else float(value)


def _build_valid_rate_data(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["signal_source", "valid_count", "invalid_count", "total_count", "valid_rate_pct"]
    if df.empty:
        return pd.DataFrame(columns=columns)

    signal_source = df.get("signal_source", pd.Series(index=df.index, dtype="object")).fillna("(empty)")
    signal_source = signal_source.replace("", "(empty)")
    valid = _true_mask(df.get("ibd_entry_valid", pd.Series(index=df.index, dtype="object"))).astype(int)

    working = pd.DataFrame({"signal_source": signal_source, "valid": valid}, index=df.index)
    grouped = working.groupby("signal_source", dropna=False).agg(total_count=("valid", "size"), valid_count=("valid", "sum"))
    grouped["invalid_count"] = grouped["total_count"] - grouped["valid_count"]
    grouped["valid_rate_pct"] = (grouped["valid_count"] / grouped["total_count"] * 100).round(2)
    result = grouped.reset_index()[columns]
    return result.sort_values(["total_count", "signal_source"], ascending=[False, True], kind="mergesort").reset_index(drop=True)


def _build_signal_quality_matrix_data(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "signal_source",
        "ibd_candidate_rule",
        "valid_count",
        "invalid_count",
        "total_count",
        "valid_rate_pct",
        "median_ibd_entry_volume_ratio",
        "median_ibd_entry_close_vs_trigger_pct",
        "median_volume_ratio",
        "median_pct_above_ceiling",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    needed = [
        "signal_source",
        "ibd_candidate_rule",
        "ibd_entry_valid",
        "ibd_entry_volume_ratio",
        "ibd_entry_close_vs_trigger_pct",
        "volume_ratio",
        "pct_above_ceiling",
    ]
    available = [c for c in needed if c in df.columns]
    working = df[available].copy()
    for column in [
        "ibd_entry_volume_ratio",
        "ibd_entry_close_vs_trigger_pct",
        "volume_ratio",
        "pct_above_ceiling",
    ]:
        if column not in working.columns:
            working[column] = pd.NA
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working["signal_source"] = _label_series(working, "signal_source")
    working["ibd_candidate_rule"] = _label_series(working, "ibd_candidate_rule")
    working["valid"] = _true_mask(working.get("ibd_entry_valid", pd.Series(index=working.index, dtype="object"))).astype(int)

    grouped = working.groupby(["signal_source", "ibd_candidate_rule"], dropna=False).agg(
        total_count=("valid", "size"),
        valid_count=("valid", "sum"),
        median_ibd_entry_volume_ratio=("ibd_entry_volume_ratio", "median"),
        median_ibd_entry_close_vs_trigger_pct=("ibd_entry_close_vs_trigger_pct", "median"),
        median_volume_ratio=("volume_ratio", "median"),
        median_pct_above_ceiling=("pct_above_ceiling", "median"),
    )
    grouped["invalid_count"] = grouped["total_count"] - grouped["valid_count"]
    grouped["valid_rate_pct"] = (grouped["valid_count"] / grouped["total_count"] * 100).round(2)
    return (
        grouped.reset_index()[columns]
        .sort_values(["total_count", "signal_source", "ibd_candidate_rule"], ascending=[False, True, True], kind="mergesort")
        .reset_index(drop=True)
    )


def _build_structure_action_map_data(df: pd.DataFrame) -> pd.DataFrame:
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
        "pct_above_ceiling",
        "pullback_pct_off_peak",
        "touched_ema10_count",
        "volume_ratio",
        "ibd_entry_volume_ratio",
        "ibd_entry_close_vs_trigger_pct",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    available = [c for c in columns if c in df.columns]
    working = df[available].copy()
    for column in columns:
        if column not in working.columns:
            working[column] = pd.NA
    working["entry_status"] = working["ibd_entry_valid"].map(_entry_status)
    working["dry_status"] = working["pullback_v_is_dry"].map(_dry_status)
    return working.dropna(subset=["pct_above_ceiling", "volume_ratio"])[columns].copy()


def _build_sector_concentration_data(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["sector", "row_count", "share_pct", "valid_count", "valid_rate_pct", "top_industry"]
    if df.empty:
        return pd.DataFrame(columns=columns)

    needed = ["sector", "industry", "code", "ibd_entry_valid"]
    available = [c for c in needed if c in df.columns]
    working = df[available].copy()
    working["sector"] = _label_series(working, "sector")
    working["industry"] = _label_series(working, "industry")
    working["valid"] = _true_mask(working.get("ibd_entry_valid", pd.Series(index=working.index, dtype="object"))).astype(int)

    grouped = working.groupby("sector", dropna=False).agg(row_count=("code", "size"), valid_count=("valid", "sum"))
    grouped["share_pct"] = (grouped["row_count"] / len(working) * 100).round(2)
    grouped["valid_rate_pct"] = (grouped["valid_count"] / grouped["row_count"] * 100).round(2)
    top_industry = (
        working.groupby(["sector", "industry"], dropna=False)
        .size()
        .rename("industry_count")
        .reset_index()
        .sort_values(["sector", "industry_count", "industry"], ascending=[True, False, True], kind="mergesort")
        .drop_duplicates("sector")
        .set_index("sector")["industry"]
    )
    result = grouped.reset_index()
    result["top_industry"] = result["sector"].map(top_industry)
    return result[columns].sort_values(["row_count", "sector"], ascending=[False, True], kind="mergesort").reset_index(drop=True)


def _build_scatter_data(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "code",
        "signal_source",
        "ibd_candidate_rule",
        "ibd_entry_price",
        "pct_above_ceiling",
        "ibd_entry_volume_ratio",
        "ibd_entry_close_vs_trigger_pct",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    available = [c for c in columns if c in df.columns]
    working = df[available].copy()
    for column in columns:
        if column not in working.columns:
            working[column] = pd.NA
    return working.dropna(subset=["ibd_entry_volume_ratio", "ibd_entry_close_vs_trigger_pct"])[columns].copy()


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


def _describe_filter(label: str, spec: FilterSpec) -> str:
    operator = spec.operator.lower()
    if operator in {"is true", "is false", "not empty", "non-empty", "is empty"}:
        return f"{label} {spec.operator}"
    if operator == "between":
        return f"{label} between {spec.value} and {spec.value2}"
    return f"{label} {spec.operator} {spec.value}"
