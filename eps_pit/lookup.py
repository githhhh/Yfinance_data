from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from eps_pit.models import EPSMissingReason, EPSResolveMode, EPSResult, EPSStatus
from eps_pit.providers.pit_provider import date10, normalize_symbol, safe_float
from eps_pit.providers.sec_yahoo_provider import SECYahooEPSProvider
from eps_pit.providers.tradingview_provider import TradingViewEPSProvider
from eps_pit.store import EPSPITStore


_UNSET = object()
_CURRENT_EPS_TIMEZONE = ZoneInfo("America/New_York")


def current_eps_observation_date() -> str:
    """Calendar date on which current-state providers are being observed."""
    return dt.datetime.now(_CURRENT_EPS_TIMEZONE).date().isoformat()


class SignalEPSLookup:
    """Signal-only EPS enrichment with explicit LIVE/REPLAY semantics."""

    DEFAULT_CSV_PATH = "us/signal_eps_pit.csv"

    @classmethod
    def _normalize_ticker(cls, code: object) -> str:
        return normalize_symbol(code)

    @classmethod
    def _normalize_date(cls, date_val: object) -> str:
        return date10(date_val)

    @classmethod
    def _is_truthy(cls, value: object) -> bool:
        if value is None:
            return False
        try:
            if pd.isna(value):
                return False
        except Exception:
            pass
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"true", "1", "1.0", "yes", "y"}

    @classmethod
    def clear_cache(cls) -> None:
        # Backward-compatible no-op. The durable PIT store is read directly,
        # so there is no process-global stale cache to clear anymore.
        return None

    @classmethod
    def get_record(
        cls,
        snapshot_date: object,
        code: object,
        csv_path: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        result = EPSPITStore(csv_path or cls.DEFAULT_CSV_PATH).get(snapshot_date, code)
        return result.to_record() if result is not None else None

    @classmethod
    def get_eps(
        cls,
        snapshot_date: object,
        code: object,
        csv_path: Optional[str] = None,
    ) -> Optional[float]:
        result = EPSPITStore(csv_path or cls.DEFAULT_CSV_PATH).get(snapshot_date, code)
        return result.eps_yoy_growth if result and result.is_resolved else None

    @classmethod
    def fetch_sec_yahoo_eps(
        cls,
        snapshot_date: object,
        codes: list[str],
        *,
        allow_current_yahoo: bool = False,
        observation_date: object | None = None,
        sec_cik_hints: dict[str, str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        snap = cls._normalize_date(snapshot_date)
        observation = cls._normalize_date(observation_date) if observation_date else ""
        symbols = sorted(
            {cls._normalize_ticker(code) for code in codes if cls._normalize_ticker(code)}
        )
        if not snap or not symbols:
            return {}
        if allow_current_yahoo and (not observation or observation != snap):
            raise ValueError(
                "Yahoo current observation may only be used for the observation snapshot"
            )

        provider = SECYahooEPSProvider()
        results: dict[str, dict[str, Any]] = {}
        for symbol in symbols:
            record, reason = provider.fetch_eps_yoy_detailed(
                symbol,
                snap,
                allow_current_yahoo=allow_current_yahoo,
                observation_date=observation if allow_current_yahoo else None,
                sec_cik_hint=(sec_cik_hints or {}).get(symbol),
            )
            if record is not None:
                results[symbol] = record
            else:
                results[symbol] = {
                    "missing_reason": reason or EPSMissingReason.NO_QUARTERLY_EPS
                }
        return results

    @classmethod
    def fetch_tradingview_eps(cls, codes: list[str]) -> dict[str, dict[str, Any]]:
        return TradingViewEPSProvider().fetch_eps_yoy(codes)

    @classmethod
    def _coerce_reason(
        cls,
        value: object,
        default: EPSMissingReason,
    ) -> EPSMissingReason:
        if isinstance(value, EPSMissingReason):
            return value
        if value is None:
            return default
        try:
            return EPSMissingReason(str(value))
        except ValueError:
            return default

    @classmethod
    def _resolved_from_record(
        cls,
        sym: str,
        snap: str,
        record: dict[str, Any],
    ) -> EPSResult | None:
        value = safe_float(record.get("eps_yoy_growth"))
        if value is None:
            return None
        source = str(record.get("source") or "UNKNOWN")
        effective = cls._normalize_date(record.get("effective_date")) or None
        if source in {"TV_DIRECT", "TV_STAGE2", "POOL_EXISTING"} and not effective:
            # Current/provider-reported observations are known to have been
            # available at least by the snapshot at which we captured them.
            effective = snap
        return EPSResult(
            code=sym,
            snapshot_date=snap,
            status=EPSStatus.RESOLVED,
            eps_yoy_growth=value,
            source=source,
            effective_date=effective,
            current_eps=safe_float(record.get("current_eps")),
            prior_year_eps=safe_float(record.get("prior_year_eps")),
            current_period=cls._normalize_date(record.get("current_period")) or None,
            prior_year_period=cls._normalize_date(record.get("prior_year_period")) or None,
            calculation_method=str(record.get("calculation_method") or "") or None,
            sec_cik=str(record.get("sec_cik") or "").strip() or None,
            source_record_id=str(record.get("source_record_id") or "") or None,
        )

    @classmethod
    def resolve_eps(
        cls,
        snapshot_date: object,
        code: object,
        *,
        mode: EPSResolveMode,
        csv_path: Optional[str] = None,
        allow_network: bool = True,
        _allow_live_current_provider: bool = True,
        _live_current_outcome: object = _UNSET,
        _live_current_error: Exception | None = None,
        observation_date: object | None = None,
    ) -> EPSResult:
        if not isinstance(mode, EPSResolveMode):
            raise TypeError("mode must be EPSResolveMode")
        snap = cls._normalize_date(snapshot_date)
        sym = cls._normalize_ticker(code)
        if not snap:
            raise ValueError(f"Invalid EPS snapshot_date: {snapshot_date!r}")
        if not sym:
            raise ValueError(f"Invalid EPS code: {code!r}")

        observed_on = cls._normalize_date(observation_date) if observation_date else ""
        if not observed_on:
            observed_on = current_eps_observation_date()
        if snap > observed_on:
            raise ValueError(
                f"EPS snapshot_date is in the future: {snap} > {observed_on}"
            )
        current_state_allowed = mode is EPSResolveMode.LIVE and snap == observed_on

        store = EPSPITStore(csv_path or cls.DEFAULT_CSV_PATH)
        cached = store.get(snap, sym)
        if cached is not None and cached.is_resolved:
            return cached

        if not allow_network:
            return EPSResult(
                code=sym,
                snapshot_date=snap,
                status=EPSStatus.NOT_ATTEMPTED,
                missing_reason=EPSMissingReason.REFRESH_DISABLED,
            )

        tv_reason: EPSMissingReason | None = None
        tv_error = _live_current_error
        if current_state_allowed:
            outcome = _live_current_outcome
            if outcome is _UNSET and _allow_live_current_provider and tv_error is None:
                try:
                    outcome = cls.fetch_tradingview_eps([sym]).get(sym)
                except Exception as exc:
                    tv_error = exc
                    logging.warning("Signal EPS TradingView provider error for %s: %s", sym, exc)
                    outcome = None
            if outcome is not _UNSET and isinstance(outcome, dict):
                resolved = cls._resolved_from_record(sym, snap, outcome)
                if resolved is not None:
                    store.upsert(resolved)
                    return resolved
                tv_reason = cls._coerce_reason(
                    outcome.get("missing_reason"), EPSMissingReason.TV_FIELD_NULL
                )
                if tv_reason is EPSMissingReason.PROVIDER_ERROR:
                    tv_error = RuntimeError(
                        "TradingView provider reported technical/identity failure"
                    )
            elif outcome is None and tv_error is None:
                tv_reason = EPSMissingReason.TV_NOT_FOUND

        pit_error: Exception | None = None
        pit_entry: dict[str, Any] | None = None
        try:
            sec_cik = store.get_sec_cik(sym)
            pit_kwargs: dict[str, Any] = {
                "allow_current_yahoo": current_state_allowed,
                "observation_date": observed_on if current_state_allowed else None,
            }
            if sec_cik:
                pit_kwargs["sec_cik_hints"] = {sym: sec_cik}
            pit_entry = cls.fetch_sec_yahoo_eps(
                snap,
                [sym],
                **pit_kwargs,
            ).get(sym)
        except Exception as exc:
            pit_error = exc
            logging.warning("Signal EPS PIT provider error for %s: %s", sym, exc)

        if pit_entry is not None:
            resolved = cls._resolved_from_record(sym, snap, pit_entry)
            if resolved is not None:
                store.upsert(resolved)
                return resolved
            pit_reason = cls._coerce_reason(
                pit_entry.get("missing_reason"), EPSMissingReason.NO_QUARTERLY_EPS
            )
            if pit_reason is EPSMissingReason.PROVIDER_ERROR:
                pit_error = RuntimeError("PIT provider reported technical failure")
        else:
            pit_reason = EPSMissingReason.NO_QUARTERLY_EPS

        # A provider error matters only if no other source resolved the value.
        # At that point completeness has not been established, so fail closed.
        if tv_error is not None or pit_error is not None:
            return EPSResult(
                code=sym,
                snapshot_date=snap,
                status=EPSStatus.PROVIDER_ERROR,
                missing_reason=EPSMissingReason.PROVIDER_ERROR,
            )

        if pit_entry is not None and pit_reason in {
            EPSMissingReason.PRIOR_YEAR_EPS_ZERO,
            EPSMissingReason.NO_PRIOR_YEAR_QUARTER,
            EPSMissingReason.NO_VERIFIED_YAHOO_RELEASE_DATE,
            EPSMissingReason.NO_QUARTERLY_EPS,
        }:
            reason = pit_reason
        elif tv_reason in {EPSMissingReason.TV_FIELD_NULL, EPSMissingReason.TV_NOT_FOUND}:
            reason = tv_reason
        else:
            reason = EPSMissingReason.NO_QUARTERLY_EPS

        return EPSResult(
            code=sym,
            snapshot_date=snap,
            status=EPSStatus.EXPECTED_UNAVAILABLE,
            missing_reason=reason,
        )

    @classmethod
    def enrich_pool(
        cls,
        pool_df: pd.DataFrame,
        snapshot_date: Optional[object] = None,
        csv_path: Optional[str] = None,
        stage2_path: Optional[str] = None,
        refresh_missing: bool = False,
        *,
        mode: EPSResolveMode = EPSResolveMode.LIVE,
        observation_date: object | None = None,
    ) -> pd.DataFrame:
        del stage2_path  # retained only for compatibility with current callers
        if not isinstance(mode, EPSResolveMode):
            raise TypeError("mode must be EPSResolveMode")
        if pool_df.empty:
            return pool_df.copy()

        df = pool_df.copy()
        for column in (
            "eps_yoy_growth",
            "eps_yoy_growth_source",
            "eps_yoy_growth_status",
            "eps_yoy_growth_missing_reason",
        ):
            if column not in df.columns:
                df[column] = pd.NA

        default_snap = cls._normalize_date(snapshot_date) if snapshot_date else ""
        observed_on = cls._normalize_date(observation_date) if observation_date else ""
        if not observed_on:
            observed_on = current_eps_observation_date()
        has_signal = "signal" in df.columns
        store = EPSPITStore(csv_path or cls.DEFAULT_CSV_PATH)

        # Validate all in-scope signal identities before any provider call or
        # PIT-store write so a malformed row cannot create a partial run.
        if has_signal:
            for _, row in df.loc[df["signal"].map(cls._is_truthy)].iterrows():
                snap = cls._normalize_date(row.get("snapshot_date")) or default_snap
                sym = cls._normalize_ticker(row.get("code"))
                if not snap:
                    raise ValueError(f"Signal row has invalid snapshot_date: {row.get('code')!r}")
                if not sym:
                    raise ValueError("Signal row has invalid code")
                if snap > observed_on:
                    raise ValueError(
                        f"Signal row snapshot_date is in the future: {sym} {snap} > {observed_on}"
                    )

        tv_batch_attempted = False
        tv_batch_error: Exception | None = None
        tv_results: dict[str, dict[str, Any]] = {}
        if mode is EPSResolveMode.LIVE and refresh_missing and has_signal and "code" in df.columns:
            signal_mask = df["signal"].map(cls._is_truthy)
            row_snaps = df.apply(
                lambda row: cls._normalize_date(row.get("snapshot_date")) or default_snap,
                axis=1,
            )
            current_snapshot_mask = row_snaps.eq(observed_on)
            missing_mask = (
                signal_mask
                & current_snapshot_mask
                & df["eps_yoy_growth"].map(safe_float).isna()
            )
            missing_codes = sorted(
                {
                    cls._normalize_ticker(code)
                    for code in df.loc[missing_mask, "code"]
                    if cls._normalize_ticker(code)
                }
            )
            if missing_codes:
                tv_batch_attempted = True
                try:
                    tv_results = cls.fetch_tradingview_eps(missing_codes)
                except Exception as exc:
                    tv_batch_error = exc
                    logging.warning("Signal EPS TradingView batch error: %s", exc)
                if tv_batch_error is None:
                    for _, row in df.loc[missing_mask].iterrows():
                        sym = cls._normalize_ticker(row.get("code"))
                        snap = cls._normalize_date(row.get("snapshot_date")) or default_snap
                        outcome = tv_results.get(sym)
                        if not isinstance(outcome, dict):
                            continue
                        resolved = cls._resolved_from_record(sym, snap, outcome)
                        if resolved is not None:
                            store.upsert(resolved)

        for idx, row in df.iterrows():
            if has_signal and not cls._is_truthy(row.get("signal")):
                continue

            snap = cls._normalize_date(row.get("snapshot_date")) or default_snap
            sym = cls._normalize_ticker(row.get("code"))
            existing_eps = safe_float(row.get("eps_yoy_growth"))
            current_state_allowed = mode is EPSResolveMode.LIVE and snap == observed_on
            source_raw = row.get("eps_yoy_growth_source")
            try:
                existing_source = None if pd.isna(source_raw) else str(source_raw).strip()
            except Exception:
                existing_source = str(source_raw).strip() if source_raw is not None else None

            replay_revalidation = (
                existing_eps is not None
                and mode is EPSResolveMode.REPLAY
            )
            stale_current_replacement = (
                existing_eps is not None
                and mode is EPSResolveMode.LIVE
                and not current_state_allowed
                and (not existing_source or existing_source in {
                    "TV_STAGE2",
                    "TV_DIRECT",
                    "POOL_EXISTING",
                })
            )
            if replay_revalidation or stale_current_replacement:
                # A current-state Stage2 value observed after this snapshot
                # cannot be backdated. Re-resolve the old snapshot from strict
                # PIT sources instead of silently stamping effective_date=snap.
                existing_eps = None
                df.at[idx, "eps_yoy_growth"] = pd.NA
                df.at[idx, "eps_yoy_growth_source"] = pd.NA
                df.at[idx, "eps_yoy_growth_status"] = pd.NA
                df.at[idx, "eps_yoy_growth_missing_reason"] = pd.NA

            if existing_eps is not None:
                source = existing_source
                if not source:
                    source = "TV_STAGE2" if mode is EPSResolveMode.LIVE else "POOL_EXISTING"
                df.at[idx, "eps_yoy_growth_status"] = EPSStatus.RESOLVED.value
                df.at[idx, "eps_yoy_growth_missing_reason"] = pd.NA
                df.at[idx, "eps_yoy_growth_source"] = source
                if snap and sym:
                    store.upsert(
                        EPSResult(
                            code=sym,
                            snapshot_date=snap,
                            status=EPSStatus.RESOLVED,
                            eps_yoy_growth=existing_eps,
                            source=source,
                            effective_date=snap,
                            calculation_method="provider_reported_yoy",
                        )
                    )
                continue

            if not snap or not sym:
                # Preserve compatibility for non-signal generic frames. Signal
                # rows were strictly validated before reaching this branch.
                continue

            live_outcome: object = _UNSET
            if current_state_allowed and tv_batch_attempted and tv_batch_error is None:
                live_outcome = tv_results.get(
                    sym, {"missing_reason": EPSMissingReason.TV_NOT_FOUND}
                )

            result = cls.resolve_eps(
                snap,
                sym,
                mode=mode,
                csv_path=csv_path,
                allow_network=(
                    refresh_missing
                    or stale_current_replacement
                    or replay_revalidation
                ),
                _allow_live_current_provider=not tv_batch_attempted,
                _live_current_outcome=live_outcome,
                _live_current_error=tv_batch_error if current_state_allowed else None,
                observation_date=observed_on,
            )
            df.at[idx, "eps_yoy_growth_status"] = result.status.value
            df.at[idx, "eps_yoy_growth_missing_reason"] = (
                result.missing_reason.value if result.missing_reason else pd.NA
            )
            if result.is_resolved:
                df.at[idx, "eps_yoy_growth"] = result.eps_yoy_growth
                df.at[idx, "eps_yoy_growth_source"] = result.source or "PIT"

        return df


def get_signal_eps(
    snapshot_date: object,
    code: object,
    csv_path: Optional[str] = None,
) -> Optional[float]:
    return SignalEPSLookup.get_eps(snapshot_date, code, csv_path=csv_path)


def resolve_signal_eps(
    snapshot_date: object,
    code: object,
    *,
    mode: EPSResolveMode,
    csv_path: Optional[str] = None,
    allow_network: bool = True,
    observation_date: object | None = None,
) -> EPSResult:
    return SignalEPSLookup.resolve_eps(
        snapshot_date,
        code,
        mode=mode,
        csv_path=csv_path,
        allow_network=allow_network,
        observation_date=observation_date,
    )


def enrich_pool_with_signal_eps(
    pool_df: pd.DataFrame,
    snapshot_date: Optional[object] = None,
    csv_path: Optional[str] = None,
    stage2_path: Optional[str] = None,
    refresh_missing: bool = False,
    *,
    mode: EPSResolveMode = EPSResolveMode.LIVE,
    observation_date: object | None = None,
) -> pd.DataFrame:
    return SignalEPSLookup.enrich_pool(
        pool_df,
        snapshot_date=snapshot_date,
        csv_path=csv_path,
        stage2_path=stage2_path,
        refresh_missing=refresh_missing,
        mode=mode,
        observation_date=observation_date,
    )
