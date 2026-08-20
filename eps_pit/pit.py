import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from eps_pit.fiscal_period import FiscalPeriodMatcher
from eps_pit.growth import EPSGrowthCalculator


class PITTimelineEngine:
    """Builds Point-in-Time events and performs backward-asof merging."""

    @staticmethod
    def build_growth_events(records: List[Dict[str, Any]]) -> pd.DataFrame:
        """Converts raw quarterly history records into PIT growth events."""
        if not records:
            return pd.DataFrame(columns=[
                "code", "report_period", "fiscal_year", "fiscal_quarter",
                "eps_current", "eps_prior_year", "eps_yoy_growth", "growth_status",
                "effective_at_conservative", "effective_at_release",
                "effective_date_method", "source", "source_record_id"
            ])

        # Group by code
        by_code = {}
        for r in records:
            c = r["code"]
            by_code.setdefault(c, []).append(r)

        event_rows = []
        for code, code_records in by_code.items():
            matched_pairs = FiscalPeriodMatcher.match_quarters(code_records)
            for curr, prior, match_status in matched_pairs:
                eps_curr = curr.get("eps_diluted")
                eps_prior = prior.get("eps_diluted") if prior else None

                growth_val, growth_status, is_calc = EPSGrowthCalculator.calculate(eps_curr, eps_prior)
                
                # Determine effective dates
                filing_date = curr.get("filing_date")
                release_date = curr.get("earnings_release_at") or filing_date

                # Conservative PIT uses filing date (when 10-Q was officially available)
                eff_conservative = filing_date or curr.get("report_period")
                # Release PIT uses earnings announcement date
                eff_release = release_date or eff_conservative

                # Format as YYYY-MM-DD string
                if isinstance(eff_conservative, str) and "T" in eff_conservative:
                    eff_conservative = eff_conservative.split("T")[0]
                if isinstance(eff_release, str) and "T" in eff_release:
                    eff_release = eff_release.split("T")[0]

                method = "SEC_FILING" if curr.get("source") == "SEC" else "EARNINGS_RELEASE"

                event_rows.append({
                    "code": code,
                    "report_period": curr.get("report_period"),
                    "fiscal_year": curr.get("fiscal_year"),
                    "fiscal_quarter": curr.get("fiscal_quarter"),
                    "eps_current": eps_curr,
                    "eps_prior_year": eps_prior,
                    "eps_yoy_growth": growth_val if is_calc else None,
                    "growth_status": growth_status if is_calc else match_status,
                    "effective_at_conservative": eff_conservative,
                    "effective_at_release": eff_release,
                    "effective_date_method": method,
                    "source": curr.get("source"),
                    "source_record_id": curr.get("source_record_id"),
                })

        df_events = pd.DataFrame(event_rows)
        if not df_events.empty:
            df_events = df_events.sort_values(
                by=["code", "effective_at_conservative", "report_period"]
            ).reset_index(drop=True)
            # When multiple events share the same (code, effective_at_conservative),
            # merge_asof picks the last matching row — make that deterministic by
            # keeping only the event with the latest report_period per group.
            df_events = df_events.drop_duplicates(
                subset=["code", "effective_at_conservative"], keep="last"
            ).reset_index(drop=True)
        return df_events

    @staticmethod
    def merge_asof_snapshot(
        snapshot_df: pd.DataFrame,
        events_df: pd.DataFrame,
        snapshot_date_col: str = "snapshot_date",
        pit_mode: str = "conservative"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Merges PIT events into snapshot dataframe using backward merge_asof.
        
        Args:
            snapshot_df: Weekly snapshot dataframe
            events_df: PIT growth events dataframe
            snapshot_date_col: Column name containing snapshot date string (YYYY-MM-DD)
            pit_mode: 'conservative' (effective_at_conservative) or 'release' (effective_at_release)
            
        Returns:
            (patched_df, provenance_df)
        """
        if snapshot_df.empty:
            return snapshot_df.copy(), pd.DataFrame()

        # Preserve original order and index
        snap_copy = snapshot_df.copy()
        orig_cols = list(snap_copy.columns)
        snap_copy["__orig_idx"] = np.arange(len(snap_copy))

        # Ensure snapshot date is datetime
        snap_copy["__snap_dt"] = pd.to_datetime(snap_copy[snapshot_date_col])
        snap_copy["__code_clean"] = snap_copy["code"].astype(str).str.strip().str.upper()

        if events_df.empty:
            snap_copy["eps_yoy_growth"] = np.nan
            return snap_copy[orig_cols], pd.DataFrame()

        eff_col = "effective_at_conservative" if pit_mode == "conservative" else "effective_at_release"
        
        ev_copy = events_df.dropna(subset=[eff_col]).copy()
        ev_copy["__eff_dt"] = pd.to_datetime(ev_copy[eff_col])
        ev_copy["__code_clean"] = ev_copy["code"].astype(str).str.strip().str.upper()

        # Sort for merge_asof
        snap_sorted = snap_copy.sort_values(by="__snap_dt")
        ev_sorted = ev_copy.sort_values(by="__eff_dt")

        # Perform merge_asof
        target_ev_cols = [
            "__code_clean", "__eff_dt", "eps_yoy_growth", "eps_current",
            "eps_prior_year", "report_period", "growth_status",
            "effective_at_conservative", "effective_at_release", "source"
        ]
        available_ev_cols = [c for c in target_ev_cols if c in ev_sorted.columns]

        merged = pd.merge_asof(
            snap_sorted,
            ev_sorted[available_ev_cols],
            left_on="__snap_dt",
            right_on="__eff_dt",
            by="__code_clean",
            direction="backward",
            suffixes=("", "_pit")
        )

        # Restore original order
        merged = merged.sort_values(by="__orig_idx").reset_index(drop=True)

        pit_col = "eps_yoy_growth_pit" if "eps_yoy_growth_pit" in merged.columns else "eps_yoy_growth"

        # Build provenance record
        prov_cols = [
            snapshot_date_col, "code", pit_col, "eps_current",
            "eps_prior_year", "report_period", "growth_status",
            "effective_at_conservative", "effective_at_release", "source"
        ]
        prov_df = merged[[c for c in prov_cols if c in merged.columns]].copy()
        if pit_col in prov_df.columns:
            prov_df = prov_df.rename(columns={pit_col: "eps_yoy_growth"})

        # Build patched df
        patched_df = merged[orig_cols].copy()
        if pit_col in merged.columns:
            patched_df["eps_yoy_growth"] = merged[pit_col]

        return patched_df, prov_df
