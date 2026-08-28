from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import pandas as pd

from eps_pit.models import (
    EPS_RESOLVER_VERSION,
    EPSGrowthType,
    EPSMissingReason,
    EPSResult,
    EPSStatus,
)
from eps_pit.providers.pit_provider import date10, normalize_symbol, safe_float


PIT_COLUMNS = [
    "snapshot_date",
    "code",
    "eps_yoy_growth",
    "source",
    "effective_date",
    "current_eps",
    "prior_year_eps",
    "current_period",
    "prior_year_period",
    "calculation_method",
    "growth_type",
    "status",
    "missing_reason",
    "sec_cik",
    "sec_current_eps",
    "sec_prior_year_eps",
    "sec_current_period",
    "sec_prior_year_period",
    "sec_effective_date",
    "sec_source_record_id",
    "yahoo_current_eps",
    "yahoo_prior_year_eps",
    "yahoo_current_period",
    "yahoo_prior_year_period",
    "yahoo_effective_date",
    "yahoo_source_record_id",
    "source_record_id",
    "resolver_version",
    "retrieved_at",
]


class EPSPITStoreError(RuntimeError):
    pass


class EPSPITStore:
    def __init__(self, csv_path: str = "us/signal_eps_pit.csv"):
        self.csv_path = csv_path

    @staticmethod
    def _norm_code(code: object) -> str:
        return normalize_symbol(code)

    @staticmethod
    def _norm_date(value: object) -> str:
        return date10(value)

    @staticmethod
    def _norm_cik(value: object) -> str | None:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        text = str(value).strip()
        if text.endswith(".0") and text[:-2].isdigit():
            text = text[:-2]
        return text.zfill(10) if text.isdigit() else None

    @staticmethod
    def _norm_text(value: object) -> str:
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except Exception:
            pass
        return str(value).strip()

    def _read(self) -> pd.DataFrame:
        path = Path(self.csv_path)
        if not path.exists():
            return pd.DataFrame(columns=PIT_COLUMNS)
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            raise EPSPITStoreError(f"Cannot read EPS PIT store: {path}") from exc
        missing = {"snapshot_date", "code"}.difference(df.columns)
        if missing:
            raise EPSPITStoreError(
                f"EPS PIT store missing required columns: {sorted(missing)}"
            )
        return df

    @staticmethod
    def _validate_unique(df: pd.DataFrame) -> None:
        if df.empty:
            return
        keys = pd.DataFrame(
            {
                "snapshot_date": df["snapshot_date"].map(date10),
                "code": df["code"].map(normalize_symbol),
            }
        )
        valid = keys["snapshot_date"].ne("") & keys["code"].ne("")
        if keys.loc[valid].duplicated(["snapshot_date", "code"]).any():
            raise EPSPITStoreError("EPS PIT store contains duplicate snapshot/code keys")

    def get(self, snapshot_date: object, code: object) -> EPSResult | None:
        snap = self._norm_date(snapshot_date)
        sym = self._norm_code(code)
        if not snap or not sym:
            return None
        df = self._read()
        self._validate_unique(df)
        if df.empty:
            return None

        normalized_snap = df["snapshot_date"].map(date10)
        normalized_code = df["code"].map(normalize_symbol)
        rows = df.loc[normalized_snap.eq(snap) & normalized_code.eq(sym)]
        if rows.empty:
            return None
        row = rows.iloc[0]
        resolver_version = self._norm_text(row.get("resolver_version"))
        if resolver_version != EPS_RESOLVER_VERSION:
            # Resolver policy changes can alter source ordering, historical
            # visibility, or calculation semantics. Never let a legacy cached
            # observation silently bypass the active policy.
            return None
        eps = safe_float(row.get("eps_yoy_growth"))
        status_raw = self._norm_text(row.get("status")).lower()
        if status_raw not in {
            EPSStatus.RESOLVED.value,
            EPSStatus.EXPECTED_UNAVAILABLE.value,
        }:
            return None
        if status_raw == EPSStatus.RESOLVED.value and eps is None:
            raise EPSPITStoreError(
                f"EPS PIT resolved record has no numeric value: {sym} {snap}"
            )
        if status_raw == EPSStatus.EXPECTED_UNAVAILABLE.value and eps is not None:
            raise EPSPITStoreError(
                f"EPS PIT expected_unavailable record unexpectedly has value: {sym} {snap}"
            )

        def f(name: str) -> float | None:
            return safe_float(row.get(name))

        def s(name: str) -> str | None:
            value = row.get(name)
            try:
                if pd.isna(value):
                    return None
            except Exception:
                pass
            text = str(value).strip()
            return text or None

        effective = s("effective_date")
        if effective:
            normalized_effective = date10(effective)
            if not normalized_effective:
                raise EPSPITStoreError(
                    f"EPS PIT record has invalid effective_date: {sym} {snap}"
                )
            if normalized_effective > snap:
                raise EPSPITStoreError(
                    f"EPS PIT record leaks future data: {sym} {snap} -> {normalized_effective}"
                )
            effective = normalized_effective

        growth_type_raw = s("growth_type")
        try:
            growth_type = EPSGrowthType(growth_type_raw) if growth_type_raw else None
        except ValueError:
            growth_type = None
        missing_reason_raw = s("missing_reason")
        try:
            missing_reason = (
                EPSMissingReason(missing_reason_raw)
                if missing_reason_raw
                else None
            )
        except ValueError:
            missing_reason = None

        return EPSResult(
            code=sym,
            snapshot_date=snap,
            status=EPSStatus(status_raw),
            eps_yoy_growth=eps,
            source=s("source"),
            effective_date=effective,
            current_eps=f("current_eps"),
            prior_year_eps=f("prior_year_eps"),
            current_period=date10(s("current_period")) or None,
            prior_year_period=date10(s("prior_year_period")) or None,
            calculation_method=s("calculation_method"),
            growth_type=growth_type,
            missing_reason=missing_reason,
            sec_cik=self._norm_cik(row.get("sec_cik")),
            sec_current_eps=f("sec_current_eps"),
            sec_prior_year_eps=f("sec_prior_year_eps"),
            sec_current_period=date10(s("sec_current_period")) or None,
            sec_prior_year_period=date10(s("sec_prior_year_period")) or None,
            sec_effective_date=date10(s("sec_effective_date")) or None,
            sec_source_record_id=s("sec_source_record_id"),
            yahoo_current_eps=f("yahoo_current_eps"),
            yahoo_prior_year_eps=f("yahoo_prior_year_eps"),
            yahoo_current_period=date10(s("yahoo_current_period")) or None,
            yahoo_prior_year_period=date10(s("yahoo_prior_year_period")) or None,
            yahoo_effective_date=date10(s("yahoo_effective_date")) or None,
            yahoo_source_record_id=s("yahoo_source_record_id"),
            source_record_id=s("source_record_id"),
            resolver_version=resolver_version,
        )

    def get_sec_cik(self, code: object) -> str | None:
        """Return a stable SEC CIK previously bound under the active policy."""
        sym = self._norm_code(code)
        if not sym:
            return None
        df = self._read()
        if df.empty or "sec_cik" not in df.columns:
            return None
        self._validate_unique(df)
        codes = df["code"].map(normalize_symbol)
        versions = (
            df["resolver_version"].astype(str)
            if "resolver_version" in df.columns
            else pd.Series("", index=df.index)
        )
        rows = df.loc[codes.eq(sym) & versions.eq(EPS_RESOLVER_VERSION)]
        if rows.empty:
            return None
        values = {
            cik
            for value in rows["sec_cik"]
            if (cik := self._norm_cik(value)) is not None
        }
        if len(values) > 1:
            raise EPSPITStoreError(
                f"EPS PIT store has conflicting SEC CIK bindings for {sym}: {sorted(values)}"
            )
        return next(iter(values), None)

    def upsert(self, result: EPSResult) -> None:
        has_evidence = any(
            value is not None
            for value in (
                result.current_eps,
                result.prior_year_eps,
                result.sec_current_eps,
                result.sec_prior_year_eps,
                result.yahoo_current_eps,
                result.yahoo_prior_year_eps,
            )
        )
        persistable = result.is_resolved or (
            result.status is EPSStatus.EXPECTED_UNAVAILABLE and has_evidence
        )
        if not persistable:
            return
        snap = self._norm_date(result.snapshot_date)
        sym = self._norm_code(result.code)
        eps = safe_float(result.eps_yoy_growth)
        if not snap or not sym:
            raise EPSPITStoreError("EPS PIT result has invalid key")
        if result.is_resolved and (eps is None or not math.isfinite(eps)):
            raise EPSPITStoreError("Resolved EPS result has invalid value")
        if result.status is EPSStatus.EXPECTED_UNAVAILABLE and eps is not None:
            raise EPSPITStoreError(
                "Expected-unavailable EPS result cannot have numeric growth"
            )
        effective = self._norm_date(result.effective_date) if result.effective_date else None
        if result.effective_date and not effective:
            raise EPSPITStoreError("Resolved EPS result has invalid effective_date")
        if effective and effective > snap:
            raise EPSPITStoreError("Resolved EPS result effective_date exceeds snapshot_date")

        path = Path(self.csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self._read()
        self._validate_unique(df)
        for col in PIT_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
        if not df.empty:
            normalized_snap = df["snapshot_date"].map(date10)
            normalized_code = df["code"].map(normalize_symbol)
            df = df.loc[~(normalized_snap.eq(snap) & normalized_code.eq(sym))].copy()

        record = result.to_record()
        record["snapshot_date"] = snap
        record["code"] = sym
        record["eps_yoy_growth"] = eps
        record["effective_date"] = effective
        record["resolver_version"] = EPS_RESOLVER_VERSION
        record["retrieved_at"] = pd.Timestamp.utcnow().isoformat()
        rows = df[PIT_COLUMNS].to_dict("records")
        rows.append({column: record.get(column) for column in PIT_COLUMNS})
        df = pd.DataFrame.from_records(rows, columns=PIT_COLUMNS)
        df = df.sort_values(["snapshot_date", "code"], kind="stable")

        fd, temp_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        os.close(fd)
        try:
            df.to_csv(temp_name, index=False)
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
