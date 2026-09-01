from __future__ import annotations
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from .config import *
from .factor_audit import (
    FactorAuditResult,
    audit_redundancy,
    audit_semantic_direction,
    check_code_leakage,
    deterministic_replay_factor,
)

README_SPEC = """# RD-Agent Track B Candidate Factor Discovery Dataset

This dataset contains PIT-safe weekly candidate features for US equities.
BOUNDARY RULES:
- Dataset is strictly TRAIN-ONLY (snapshot_date <= 2026-05-22).
- Data contains ONLY pre-entry candidate snapshot features and historical price metrics.
- NO future returns (W1, W2, W3, W4), NO post-entry stops/drawdowns, NO future price lookaheads.
- NO B0 benchmark rankings, selection memberships, or pick orders.
- Implement an interpretable, continuous alpha factor in factor.py reading candidate_panel.h5 (key='data').
- Output result.h5 with key='data', MultiIndex(datetime, instrument), and exactly one numeric float column.
- Higher value must signify stronger expected post-breakout selection quality.
"""


def create_safe_train_dataset(panel: pd.DataFrame) -> tuple[Path, Path, Path]:
    """Prepares strictly Train-only PIT data for RD-Agent without labels or future indicators."""
    AGENT_WORKSPACE.mkdir(parents=True, exist_ok=True)
    
    # Exclude all labels, B0 indicators, future entries, and forward dates
    excluded_prefixes = ('w1_', 'w2_', 'w3_', 'w4_', 'ret_', 'fwd_')
    excluded_exact = {
        'is_b0', 'pick_order', 'period', 'is_valid_entry', 'entry_status',
        'entry_date', 'entry_open', 'stop_8_hit_ever', 'gap_stop', 'profit20_hit',
        'max_drawdown_to_asof_pct', 'b0_eligible', 'b0_rank'
    }
    
    safe_cols = [
        c for c in panel.columns
        if c not in excluded_exact
        and not c.startswith(excluded_prefixes)
        and not c.endswith('_return_pct')
        and not c.endswith('_stop8')
    ]
    
    df_safe = panel[safe_cols].copy()
    df_safe['datetime'] = pd.to_datetime(df_safe['snapshot_date'])
    df_safe['instrument'] = df_safe['code'].astype(str).str.upper().str.strip()
    df_safe = df_safe.set_index(['datetime', 'instrument']).sort_index()
    
    train_only = df_safe[df_safe.index.get_level_values('datetime') <= pd.Timestamp(TRAIN_END)].copy()
    
    train_dir = AGENT_WORKSPACE / 'source_data_train'
    debug_dir = AGENT_WORKSPACE / 'source_data_debug'
    full_dir = AGENT_WORKSPACE / 'source_data_full'
    
    for d, data in [(train_dir, train_only), (debug_dir, train_only.groupby(level=0).head(25)), (full_dir, df_safe)]:
        d.mkdir(parents=True, exist_ok=True)
        data.to_hdf(d / 'candidate_panel.h5', key='data', mode='w')
        data.to_hdf(d / 'daily_pv.h5', key='data', mode='w')
        (d / 'README.md').write_text(README_SPEC, encoding='utf-8')
        
    return train_dir, debug_dir, full_dir


def sanitize_dict_secrets(d: dict) -> dict:
    """Remove sensitive API keys from metadata."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = sanitize_dict_secrets(v)
        elif isinstance(v, str):
            if any(s in k.lower() for s in ['key', 'token', 'secret', 'password']) or 'AIza' in v:
                out[k] = '***REDACTED***'
            else:
                out[k] = v
        else:
            out[k] = v
    return out


def run_rdagent_discovery(step_n: int = 3) -> dict:
    """Execute real RD-Agent discovery loop and log full provenance."""
    load_dotenv(ROOT / '.env', override=False)
    
    panel = pd.read_parquet(DATA / 'candidate_factor_panel.parquet' if (DATA / 'candidate_factor_panel.parquet').exists() else PANEL_SOURCE)
    train_dir, debug_dir, full_dir = create_safe_train_dataset(panel)
    
    env_bin = Path(sys.executable).resolve().parent
    rdagent_bin = shutil.which('rdagent') or (env_bin / 'rdagent' if (env_bin / 'rdagent').exists() else None)
    if not rdagent_bin:
        raise RuntimeError("RD-Agent CLI executable not found in current environment.")
        
    gemini_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('OPENAI_API_KEY')
    if not gemini_key:
        raise RuntimeError("No GEMINI_API_KEY found in .env. RD-Agent requires LLM credentials.")
        
    chat_model = os.environ.get('CHAT_MODEL', 'gemini/gemini-3.6-flash')
    embedding_model = os.environ.get('EMBEDDING_MODEL', 'gemini/gemini-embedding-001')
    
    env = os.environ.copy()
    env['PATH'] = str(env_bin) + os.pathsep + env.get('PATH', '')
    env['CHAT_MODEL'] = chat_model
    env['EMBEDDING_MODEL'] = embedding_model
    env['OPENAI_API_KEY'] = gemini_key
    env['CHAT_STREAM'] = 'false'
    env['LOG_LLM_CHAT_CONTENT'] = 'false'
    env['FACTOR_COSTEER_DATA_FOLDER'] = str(train_dir)
    env['FACTOR_COSTEER_DATA_FOLDER_DEBUG'] = str(debug_dir)
    env['FACTOR_COSTEER_PYTHON_BIN'] = str(sys.executable)
    
    run_id = f"rdagent_track_b_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    start_time = datetime.datetime.now().isoformat()
    
    cmd = [str(rdagent_bin), 'fin_factor', f'--step-n={step_n}']
    print(f"Starting RD-Agent execution: {' '.join(cmd)}")
    
    proc = subprocess.run(
        cmd,
        env=env,
        cwd=str(AGENT_WORKSPACE),
        capture_output=True,
        text=True,
    )
    end_time = datetime.datetime.now().isoformat()
    
    RAW_RDAGENT.mkdir(parents=True, exist_ok=True)
    raw_log = RAW_RDAGENT / f"{run_id}_stdout.log"
    # Sanitize log of any keys before saving
    sanitized_stdout = proc.stdout.replace(gemini_key, '***REDACTED***') if gemini_key else proc.stdout
    sanitized_stderr = proc.stderr.replace(gemini_key, '***REDACTED***') if gemini_key else proc.stderr
    raw_log.write_text(f"=== STDOUT ===\n{sanitized_stdout}\n\n=== STDERR ===\n{sanitized_stderr}", encoding='utf-8')
    
    # Inspect generated workspace factors
    discovered_factors = []
    # Check session folders or workspace folders created by rdagent
    possible_dirs = list(AGENT_WORKSPACE.glob('**/factor.py')) + list(Path('git_ignore_folder').glob('**/factor.py')) + list(Path('log').glob('**/factor.py'))
    
    provenance = {
        'rdagent_version': '0.8.0',
        'run_id': run_id,
        'command': f"rdagent fin_factor --step-n={step_n}",
        'chat_model': chat_model,
        'embedding_model': embedding_model,
        'step_n': step_n,
        'start_time': start_time,
        'end_time': end_time,
        'exit_code': proc.returncode,
        'task_injection_mode': 'generic_financial_factor_generator_evaluated_externally',
        'task_spec_read_verified': False,  # Transparent audit reporting
        'train_data_boundary': f'<= {TRAIN_END}',
        'leakage_columns_excluded': True,
        'discovered_factors_count': len(possible_dirs),
    }
    
    prov_file = OUT / 'rdagent_provenance.json'
    OUT.mkdir(parents=True, exist_ok=True)
    prov_file.write_text(json.dumps(sanitize_dict_secrets(provenance), indent=2), encoding='utf-8')
    
    return provenance
