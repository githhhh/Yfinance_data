from __future__ import annotations
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
from .config import *

LEAK_PREFIXES = ('w1_', 'w2_', 'w3_', 'w4_', 'ret_', 'fwd_')
LEAK_EXACT_COLS = {
    'is_b0', 'pick_order', 'period', 'is_valid_entry', 'entry_status',
    'entry_date', 'entry_open', 'stop_8_hit_ever', 'gap_stop', 'profit20_hit',
    'max_drawdown_to_asof_pct', 'b0_eligible', 'b0_rank'
}


@dataclass
class FactorAuditResult:
    factor_name: str
    origin: str  # 'rdagent_original', 'gemini_modified', 'gemini_authored'
    rdagent_run_id: str
    description: str
    formula: str
    code_path: str
    semantic_direction: str  # 'POSITIVE' or 'NEGATIVE'
    leakage_pass: bool
    redundancy_status: str  # 'DISTINCT', 'REDUNDANT', 'AFFINE_DUPLICATE'
    train_only_discovery: bool
    replay_pass: bool
    accepted: bool
    rejection_reason: str = ''
    redundant_with_feature: str = ''
    correlation_max: float = 0.0
    output_hash: str = ''


def check_code_leakage(code_text: str) -> tuple[bool, str]:
    """Inspect factor source code for forbidden terms, future labels, or networking."""
    for col in LEAK_EXACT_COLS:
        if re.search(r'\b' + re.escape(col) + r'\b', code_text):
            return False, f"Code references forbidden column: {col}"
    for pref in LEAK_PREFIXES:
        if re.search(r'\b' + re.escape(pref) + r'\w+', code_text):
            return False, f"Code references forbidden prefix: {pref}"
    if any(k in code_text for k in ['requests.', 'urllib.', 'socket.', 'http.', 'yfinance']):
        return False, "Code contains forbidden networking calls"
    return True, "Passed"


def audit_semantic_direction(
    factor_values: pd.Series,
    feature_name: str,
    description: str,
    base_panel: pd.DataFrame,
) -> tuple[bool, str]:
    """Verify that the factor sign aligns with the stated semantic description."""
    desc_lower = description.lower()
    # Check proximity to 52w high:
    if 'proximity' in desc_lower and '52' in desc_lower and 'high' in desc_lower:
        # dist_to_52w_high_pct is negative (e.g. -2% means closer than -20%).
        # Proximity should be positively correlated with dist_to_52w_high_pct (closer to 0 is better).
        if 'dist_to_52w_high_pct' in base_panel.columns:
            corr = factor_values.corr(base_panel['dist_to_52w_high_pct'])
            if corr < -0.5:
                return False, f"Factor description is proximity to 52w high, but correlation with dist_to_52w_high_pct is negative ({corr:.2f}). Direction inverted!"
    # Check volume / demand confirmation:
    if 'volume' in desc_lower and ('demand' in desc_lower or 'expansion' in desc_lower or 'ratio' in desc_lower):
        if 'ibd_entry_volume_ratio' in base_panel.columns:
            corr = factor_values.corr(base_panel['ibd_entry_volume_ratio'])
            if corr < -0.5:
                return False, f"Factor claims volume/demand confirmation but is strongly negatively correlated with entry volume ratio ({corr:.2f})"
    return True, "Semantic direction verified"


def audit_redundancy(
    factor_values: pd.Series,
    base_panel: pd.DataFrame,
    feature_cols: Sequence[str],
    threshold: float = 0.95,
) -> tuple[str, str, float]:
    """Check whether the factor is an exact duplicate, affine transform, or near-collinear with existing features."""
    max_corr = 0.0
    most_correlated_feat = ''
    fv = factor_values.dropna()
    
    for feat in feature_cols:
        if feat not in base_panel.columns:
            continue
        base_v = pd.to_numeric(base_panel.loc[fv.index, feat], errors='coerce')
        valid = pd.concat([fv, base_v], axis=1).dropna()
        if len(valid) < 20 or valid.iloc[:, 0].std() < 1e-8 or valid.iloc[:, 1].std() < 1e-8:
            continue
        
        # Pearson & Spearman correlation
        p_corr = abs(float(valid.iloc[:, 0].corr(valid.iloc[:, 1])))
        s_corr = abs(float(valid.iloc[:, 0].rank().corr(valid.iloc[:, 1].rank())))
        c = max(p_corr, s_corr)
        
        if c > max_corr:
            max_corr = c
            most_correlated_feat = feat
            
        if p_corr > 0.999:
            return 'AFFINE_DUPLICATE', feat, p_corr
        if c >= threshold:
            return 'REDUNDANT', feat, c
            
    return 'DISTINCT', most_correlated_feat, max_corr


def deterministic_replay_factor(
    name: str,
    factor_py_path: Path,
    source_h5_dir: Path,
    workspace_dir: Path,
) -> tuple[bool, str, pd.DataFrame | None, str]:
    """Replay factor.py in an isolated workspace against candidate_panel.h5."""
    run_dir = workspace_dir / f'replay_{name}'
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    for f in source_h5_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, run_dir / f.name)
    shutil.copy2(factor_py_path, run_dir / 'factor.py')
    
    proc = subprocess.run(
        [sys.executable, 'factor.py'],
        cwd=run_dir,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return False, f"Execution failed: {proc.stderr}\n{proc.stdout}", None, ""
        
    result_h5 = run_dir / 'result.h5'
    if not result_h5.exists():
        return False, "factor.py executed but result.h5 was not created", None, ""
        
    try:
        res = pd.read_hdf(result_h5)
    except Exception as e:
        return False, f"Failed to read result.h5: {e}", None, ""
        
    if isinstance(res, pd.Series):
        res = res.to_frame('value')
    if res.shape[1] != 1:
        return False, f"Expected 1 column in result.h5, got {res.shape[1]}", None, ""
    if not isinstance(res.index, pd.MultiIndex):
        return False, "Index must be MultiIndex(datetime, instrument)", None, ""
        
    # Compute output content hash
    out_bytes = res.to_numpy().tobytes()
    out_hash = hashlib.sha256(out_bytes).hexdigest()[:16]
    
    return True, "Replay success", res, out_hash
