from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from ..config import TRAIN_END, FEATURE_MANIFEST_PATH


def create_anonymized_discovery_dataset(
    panel_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    """Create strictly point-in-time, Train-only, feature-allowlisted, anonymized dataset for discovery."""
    # 1. Strict Train Horizon Filter
    train_df = panel_df[panel_df.snapshot_date.astype(str) <= str(TRAIN_END)].copy()

    # 2. Strict Feature Allowlist Filter
    with open(FEATURE_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    allowed_feats = [
        k for k, v in manifest["features"].items() if v.get("allowed_for_discovery") is True
    ]

    # Required metadata columns for identification (will be anonymized)
    keep_cols = ["code", "snapshot_date", "industry", "sector"] + [
        c for c in allowed_feats if c in train_df.columns and c not in ("code", "snapshot_date", "industry", "sector")
    ]
    train_sub = train_df[keep_cols].copy()

    # 3. Anonymize Tickers & Dates
    unique_codes = sorted(train_sub["code"].dropna().unique().tolist())
    code_map = {c: f"entity_{i+1:04d}" for i, c in enumerate(unique_codes)}
    reverse_code_map = {v: k for k, v in code_map.items()}

    unique_snaps = sorted(train_sub["snapshot_date"].astype(str).dropna().unique().tolist())
    snap_map = {s: f"snapshot_{i+1:03d}" for i, s in enumerate(unique_snaps)}
    reverse_snap_map = {v: k for k, v in snap_map.items()}

    train_sub["anon_code"] = train_sub["code"].map(code_map)
    train_sub["anon_snapshot_date"] = train_sub["snapshot_date"].astype(str).map(snap_map)

    # Drop real code & snapshot_date from the discovery view
    anon_view = train_sub.drop(columns=["code", "snapshot_date"]).rename(
        columns={"anon_code": "code", "anon_snapshot_date": "snapshot_date"}
    )

    # Reorder columns with code and snapshot_date first
    lead_cols = ["code", "snapshot_date", "industry", "sector"]
    other_cols = [c for c in anon_view.columns if c not in lead_cols]
    anon_view = anon_view[lead_cols + other_cols]

    return anon_view, code_map, snap_map
