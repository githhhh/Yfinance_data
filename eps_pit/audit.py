import os
import glob
import json
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple


class ReplayPoolAuditor:
    """Audits input replay pool CSVs and generates inventory & coverage reports."""

    def __init__(self, base_dir: str = "backtest/ibd_skill_replay_pools", output_dir: str = "outputs/eps_pit_backfill"):
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.audit_dir = os.path.join(output_dir, "audit")
        os.makedirs(self.audit_dir, exist_ok=True)

    def scan_inventory(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """Scans all weekly replay pools, verifies snapshot_date, and builds ticker universe."""
        pattern = os.path.join(self.base_dir, "*", "breakout_follow_pool.csv")
        pool_files = sorted(glob.glob(pattern))

        inventory_records = []
        ticker_week_presence = {}

        for p in pool_files:
            folder_date = os.path.basename(os.path.dirname(p))
            try:
                df = pd.read_csv(p)
            except Exception as e:
                print(f"[Auditor] Error reading {p}: {e}")
                continue

            rows = len(df)
            codes = df["code"].dropna().astype(str).str.strip().str.upper().tolist() if "code" in df.columns else []
            unique_codes = set(codes)

            # Signal rows
            sig_codes = set()
            if "signal" in df.columns:
                sig_df = df[df["signal"] == True]
                sig_codes = set(sig_df["code"].dropna().astype(str).str.strip().str.upper().tolist())
            elif "breakout_signal" in df.columns:
                sig_df = df[df["breakout_signal"] == True]
                sig_codes = set(sig_df["code"].dropna().astype(str).str.strip().str.upper().tolist())

            # Snapshot date check
            csv_dates = set(df["snapshot_date"].dropna().astype(str).tolist()) if "snapshot_date" in df.columns else set()
            date_status = "PASS"
            if csv_dates and len(csv_dates) == 1:
                csv_date = list(csv_dates)[0]
                if csv_date != folder_date:
                    date_status = f"CONFLICT({folder_date} vs {csv_date})"
            elif len(csv_dates) > 1:
                date_status = f"MULTIPLE_DATES({list(csv_dates)})"

            has_eps = "eps_yoy_growth" in df.columns
            eps_missing = int(df["eps_yoy_growth"].isna().sum()) if has_eps else rows
            eps_existing = int(df["eps_yoy_growth"].notna().sum()) if has_eps else 0

            inventory_records.append({
                "file": p,
                "snapshot_date": folder_date,
                "date_status": date_status,
                "rows": rows,
                "unique_codes": len(unique_codes),
                "signal_rows": len(sig_df) if "signal" in df.columns or "breakout_signal" in df.columns else 0,
                "signal_unique_codes": len(sig_codes),
                "has_eps_yoy_growth": has_eps,
                "eps_missing_count": eps_missing,
                "eps_existing_count": eps_existing,
            })

            # Update ticker universe
            for c in unique_codes:
                if c not in ticker_week_presence:
                    ticker_week_presence[c] = {
                        "code": c,
                        "weeks_present": 0,
                        "first_seen": folder_date,
                        "last_seen": folder_date,
                        "row_count": 0,
                        "has_signal": False,
                    }
                ticker_week_presence[c]["weeks_present"] += 1
                ticker_week_presence[c]["last_seen"] = folder_date
                ticker_week_presence[c]["row_count"] += 1
                if c in sig_codes:
                    ticker_week_presence[c]["has_signal"] = True

        inv_df = pd.DataFrame(inventory_records)
        inv_path = os.path.join(self.audit_dir, "input_inventory.csv")
        inv_df.to_csv(inv_path, index=False)

        # Build universe dataframe
        univ_records = list(ticker_week_presence.values())
        univ_df = pd.DataFrame(univ_records)
        if not univ_df.empty:
            univ_df = univ_df.sort_values(by=["weeks_present", "row_count"], ascending=[False, False]).reset_index(drop=True)
        univ_path = os.path.join(self.audit_dir, "ticker_universe.csv")
        univ_df.to_csv(univ_path, index=False)

        summary = {
            "files_count": len(pool_files),
            "total_rows": int(inv_df["rows"].sum()) if not inv_df.empty else 0,
            "total_signal_rows": int(inv_df["signal_rows"].sum()) if not inv_df.empty else 0,
            "unique_codes_total": len(univ_df),
            "unique_signal_codes": int(univ_df["has_signal"].sum()) if not univ_df.empty else 0,
            "earliest_snapshot": inv_df["snapshot_date"].min() if not inv_df.empty else None,
            "latest_snapshot": inv_df["snapshot_date"].max() if not inv_df.empty else None,
        }

        return inv_df, univ_df, summary
