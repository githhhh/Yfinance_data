from __future__ import annotations
import datetime
import hashlib
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
    """Remove sensitive API keys and credentials from dictionary metadata."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = sanitize_dict_secrets(v)
        elif isinstance(v, list):
            out[k] = [sanitize_dict_secrets(x) if isinstance(x, dict) else x for x in v]
        elif isinstance(v, str):
            if any(s in k.lower() for s in ['key', 'token', 'secret', 'password', 'auth']) or v.startswith('sk-') or 'AIza' in v:
                out[k] = '***REDACTED***'
            else:
                out[k] = v
        else:
            out[k] = v
    return out


def file_sha256(path: Path) -> str:
    """Compute sha256 hash of a file."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def run_rdagent_discovery(step_n: int = 3, run_id: str | None = None) -> dict:
    """Execute a real RD-Agent discovery run in an isolated run directory with strict mechanical provenance."""
    load_dotenv(ROOT / '.env', override=True)
    
    panel = pd.read_parquet(DATA / 'candidate_factor_panel.parquet' if (DATA / 'candidate_factor_panel.parquet').exists() else PANEL_SOURCE)
    train_dir, debug_dir, full_dir = create_safe_train_dataset(panel)
    
    env_bin = Path(sys.executable).resolve().parent
    rdagent_bin = shutil.which('rdagent') or (env_bin / 'rdagent' if (env_bin / 'rdagent').exists() else None)
    if not rdagent_bin:
        raise RuntimeError("RD-Agent CLI executable not found in current environment.")
        
    # Check credentials
    deepseek_key = os.environ.get('DEEPSEEK_API_KEY')
    deepseek_base = os.environ.get('DEEPSEEK_API_BASE')
    gemini_key = os.environ.get('GEMINI_API_KEY')
    openai_key = os.environ.get('OPENAI_API_KEY')
    
    if not (deepseek_key or gemini_key or openai_key):
        raise RuntimeError("No LLM API keys found in .env. RD-Agent requires credentials.")
        
    chat_model = os.environ.get('CHAT_MODEL', 'deepseek/deepseek-v4-pro')
    embedding_model = os.environ.get('EMBEDDING_MODEL', 'deepseek/deepseek-v4-pro')
    
    if run_id is None:
        run_id = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    run_dir = RAW_RDAGENT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    factors_dir = run_dir / 'factors'
    proposals_dir = run_dir / 'proposals'
    factors_dir.mkdir(parents=True, exist_ok=True)
    proposals_dir.mkdir(parents=True, exist_ok=True)
    
    # Snapshot pre-existing workspace factor files before run
    pre_existing_files = {p.resolve(): file_sha256(p) for p in AGENT_WORKSPACE.glob('**/*.py')}
    
    # Copy .env to AGENT_WORKSPACE so rdagent loads it natively
    if (ROOT / '.env').exists():
        shutil.copy2(ROOT / '.env', AGENT_WORKSPACE / '.env')
        
    env = os.environ.copy()
    ws_bin = AGENT_WORKSPACE / 'bin'
    env['PATH'] = str(ws_bin) + os.pathsep + str(env_bin) + os.pathsep + env.get('PATH', '')
    env['PYTHONPATH'] = str(AGENT_WORKSPACE) + os.pathsep + env.get('PYTHONPATH', '')
    env['CHAT_MODEL'] = chat_model
    env['LITELLM_CHAT_MODEL'] = chat_model
    env['EMBEDDING_MODEL'] = embedding_model
    env['LITELLM_EMBEDDING_MODEL'] = embedding_model
    if deepseek_key:
        env['DEEPSEEK_API_KEY'] = deepseek_key
        env['OPENAI_API_KEY'] = deepseek_key
    if deepseek_base:
        env['DEEPSEEK_API_BASE'] = deepseek_base
        env['OPENAI_API_BASE'] = deepseek_base
    if gemini_key:
        env['GEMINI_API_KEY'] = gemini_key
    env['CHAT_STREAM'] = 'false'
    env['LOG_LLM_CHAT_CONTENT'] = 'false'
    env['FACTOR_COSTEER_DATA_FOLDER'] = str(train_dir)
    env['FACTOR_COSTEER_DATA_FOLDER_DEBUG'] = str(debug_dir)
    env['FACTOR_COSTEER_PYTHON_BIN'] = str(sys.executable)
    
    start_time = datetime.datetime.now().isoformat()
    cmd = [str(rdagent_bin), 'fin_factor', f'--step-n={step_n}']
    print(f"Starting RD-Agent execution for run_id={run_id}: {' '.join(cmd)}")
    
    proc = subprocess.run(
        cmd,
        env=env,
        cwd=str(AGENT_WORKSPACE),
        capture_output=True,
        text=True,
    )
    end_time = datetime.datetime.now().isoformat()
    
    # Sanitize and save logs
    raw_stdout = proc.stdout
    raw_stderr = proc.stderr
    for secret in [deepseek_key, gemini_key, openai_key]:
        if secret:
            raw_stdout = raw_stdout.replace(secret, '***REDACTED***')
            raw_stderr = raw_stderr.replace(secret, '***REDACTED***')
            
    (run_dir / 'stdout.log').write_text(raw_stdout, encoding='utf-8')
    (run_dir / 'stderr.log').write_text(raw_stderr, encoding='utf-8')
    
    # Discover ONLY newly created or modified factor files from this specific run
    captured_factors = []
    current_files = {p.resolve(): p for p in AGENT_WORKSPACE.glob('**/*.py')}
    for p_res, p_obj in current_files.items():
        if p_obj.name.startswith('__') or 'source_data' in str(p_obj) or 'replay' in str(p_obj):
            continue
        curr_hash = file_sha256(p_obj)
        # Check if new or modified during run
        if p_res not in pre_existing_files or pre_existing_files[p_res] != curr_hash:
            code_text = p_obj.read_text(encoding='utf-8')
            # Check if this is an executable factor script (e.g. calculates a factor and writes result.h5)
            if 'read_hdf' in code_text or 'to_hdf' in code_text or 'def calculate' in code_text or 'def factor' in code_text:
                target_fname = f"{p_obj.stem}_{curr_hash[:8]}.py" if p_obj.stem in {'factor', 'factor_experiment'} else f"{p_obj.stem}.py"
                target_path = factors_dir / target_fname
                shutil.copy2(p_obj, target_path)
                captured_hash = file_sha256(target_path)
                
                captured_factors.append({
                    'factor_name': target_path.stem,
                    'source_artifact_path': str(p_obj.relative_to(AGENT_WORKSPACE) if p_obj.is_relative_to(AGENT_WORKSPACE) else p_obj),
                    'source_artifact_hash': curr_hash,
                    'captured_factor_path': str(target_path.relative_to(CHALLENGE_ROOT)),
                    'captured_factor_hash': captured_hash,
                    'verified_match': bool(curr_hash == captured_hash),
                    'origin': 'rdagent_original',
                })
                
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
        'task_spec_read_verified': False,
        'train_data_boundary': f'<= {TRAIN_END}',
        'leakage_columns_excluded': True,
        'captured_factors': captured_factors,
    }
    
    (run_dir / 'provenance.json').write_text(json.dumps(sanitize_dict_secrets(provenance), indent=2), encoding='utf-8')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'rdagent_provenance.json').write_text(json.dumps(sanitize_dict_secrets(provenance), indent=2), encoding='utf-8')
    
    print(f"RD-Agent run completed with exit_code={proc.returncode}. Captured {len(captured_factors)} new factor(s).")
    return provenance

