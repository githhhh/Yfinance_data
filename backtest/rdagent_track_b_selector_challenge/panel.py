from __future__ import annotations
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from .config import *


def get_full_panel(copy_to_local: bool = True) -> pd.DataFrame:
    """Load the full candidate factor panel without restricting to b0_eligible."""
    if not PANEL_SOURCE.exists():
        raise FileNotFoundError(f"Candidate factor panel not found at {PANEL_SOURCE}")
    DATA.mkdir(parents=True, exist_ok=True)
    local_target = DATA / 'candidate_factor_panel.parquet'
    if copy_to_local and not local_target.exists():
        shutil.copy2(PANEL_SOURCE, local_target)
    panel_p = local_target if local_target.exists() else PANEL_SOURCE
    df = pd.read_parquet(panel_p)
    df['code'] = df['code'].astype(str).str.upper().str.strip()
    df['snapshot_date'] = df['snapshot_date'].astype(str)
    return df


def get_universe_panel(panel: pd.DataFrame, universe: str) -> pd.DataFrame:
    """Filter panel strictly by Universe S (signal) or Universe A (actionable).
    b0_eligible is NEVER used as a gate."""
    u = universe.lower().strip()
    sig = panel['signal'].astype(str).str.lower().isin({'true', '1', 'yes'}) if 'signal' in panel.columns else pd.Series(True, index=panel.index)
    if u == 'signal':
        # All valid signal rows
        return panel[sig].copy()
    elif u == 'actionable':
        # PIT ACTIONABLE state within valid signal candidates
        act_col = (pd.to_numeric(panel['is_actionable'], errors='coerce') == 1.0) if 'is_actionable' in panel.columns else pd.Series(False, index=panel.index)
        stat_col = (panel['ibd_entry_status'].astype(str).str.upper() == 'ACTIONABLE') if 'ibd_entry_status' in panel.columns else pd.Series(False, index=panel.index)
        return panel[sig & (act_col | stat_col)].copy()
    elif u == 'b0_eligible':
        # DIAGNOSTIC / COMPARISON ONLY - NEVER A CHALLENGER UNIVERSE
        if 'b0_eligible' not in panel.columns:
            return panel.iloc[0:0].copy()
        return panel[panel['b0_eligible'] == True].copy()
    else:
        raise ValueError(f"Unknown universe: {universe}. Must be 'signal' or 'actionable'")


def write_universe_manifest(panel: pd.DataFrame) -> dict:
    """Generate and write universe manifest documenting sample counts across horizons and splits."""
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        'total_rows': int(len(panel)),
        'total_weeks': int(panel['snapshot_date'].nunique()),
        'universes': {},
    }
    for u in (*UNIVERSES, 'b0_eligible'):
        up = get_universe_panel(panel, u)
        train_up = up[up.snapshot_date <= TRAIN_END]
        val_up = up[(up.snapshot_date >= CONTAM_VAL_START) & (up.snapshot_date <= CONTAM_VAL_END)]
        manifest['universes'][u] = {
            'total_rows': int(len(up)),
            'total_weeks': int(up['snapshot_date'].nunique()),
            'train_rows': int(len(train_up)),
            'train_weeks': int(train_up['snapshot_date'].nunique()),
            'val_rows': int(len(val_up)),
            'val_weeks': int(val_up['snapshot_date'].nunique()),
            'avg_candidates_per_week_train': float(round(len(train_up) / max(1, train_up['snapshot_date'].nunique()), 2)),
            'avg_candidates_per_week_val': float(round(len(val_up) / max(1, val_up['snapshot_date'].nunique()), 2)),
        }
    p = OUT / 'universe_manifest.json'
    p.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest
