import os
import glob
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from eps_pit.pit import PITTimelineEngine


class ReplayPoolBackfiller:
    """Safely backfills weekly replay pool CSVs with PIT EPS data."""

    def __init__(
        self,
        base_dir: str = "backtest/ibd_skill_replay_pools",
        output_dir: str = "outputs/eps_pit_backfill",
        pit_mode: str = "conservative"
    ):
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.patched_dir = os.path.join(output_dir, "patched")
        self.audit_dir = os.path.join(output_dir, "audit")
        self.pit_mode = pit_mode

        os.makedirs(self.patched_dir, exist_ok=True)
        os.makedirs(self.audit_dir, exist_ok=True)

    def backfill_all(self, events_df: pd.DataFrame) -> Dict[str, Any]:
        """Executes backfill across all 32 weekly snapshot CSVs."""
        pattern = os.path.join(self.base_dir, "*", "breakout_follow_pool.csv")
        pool_files = sorted(glob.glob(pattern))

        all_provenance = []
        weekly_stats = []

        total_rows = 0
        total_filled = 0
        total_missing = 0

        for p in pool_files:
            folder_date = os.path.basename(os.path.dirname(p))
            try:
                orig_df = pd.read_csv(p)
            except Exception as e:
                print(f"[Backfiller] Failed reading {p}: {e}")
                continue

            # Ensure snapshot_date column exists
            if "snapshot_date" not in orig_df.columns:
                orig_df["snapshot_date"] = folder_date

            # Execute merge_asof
            patched_df, prov_df = PITTimelineEngine.merge_asof_snapshot(
                snapshot_df=orig_df,
                events_df=events_df,
                snapshot_date_col="snapshot_date",
                pit_mode=self.pit_mode
            )

            # Safety validations
            assert len(patched_df) == len(orig_df), f"Row count mismatch in {folder_date}: {len(patched_df)} vs {len(orig_df)}"
            assert list(patched_df["code"]) == list(orig_df["code"]), f"Code order mismatch in {folder_date}"

            # Save patched CSV
            target_week_dir = os.path.join(self.patched_dir, folder_date)
            os.makedirs(target_week_dir, exist_ok=True)
            target_csv = os.path.join(target_week_dir, "breakout_follow_pool.csv")
            patched_df.to_csv(target_csv, index=False)

            # Stats
            n_rows = len(patched_df)
            n_filled = int(patched_df["eps_yoy_growth"].notna().sum())
            n_missing = n_rows - n_filled
            cov_pct = round(n_filled / n_rows * 100.0, 2) if n_rows > 0 else 0.0

            total_rows += n_rows
            total_filled += n_filled
            total_missing += n_missing

            weekly_stats.append({
                "snapshot_date": folder_date,
                "rows": n_rows,
                "need_eps": n_rows,
                "filled": n_filled,
                "missing": n_missing,
                "coverage_pct": cov_pct,
            })

            if not prov_df.empty:
                all_provenance.append(prov_df)

        # Save weekly coverage
        df_weekly = pd.DataFrame(weekly_stats)
        weekly_cov_path = os.path.join(self.audit_dir, "coverage_by_week.csv")
        df_weekly.to_csv(weekly_cov_path, index=False)

        # Save provenance sidecar parquet
        if all_provenance:
            full_prov = pd.concat(all_provenance, ignore_index=True)
            prov_path = os.path.join(self.audit_dir, "weekly_eps_provenance.parquet")
            full_prov.to_parquet(prov_path, index=False)

        # Overall summary
        summary = {
            "weekly_files": len(pool_files),
            "total_rows": total_rows,
            "rows_need_eps": total_rows,
            "rows_filled": total_filled,
            "rows_unresolved": total_missing,
            "coverage_pct": round(total_filled / total_rows * 100.0, 2) if total_rows > 0 else 0.0,
            "pit_mode": self.pit_mode,
        }

        with open(os.path.join(self.audit_dir, "coverage_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        return summary
