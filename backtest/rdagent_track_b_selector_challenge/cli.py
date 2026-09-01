from __future__ import annotations
import argparse
from dataclasses import asdict
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from .config import *
from .diagnostics import run_lane_diagnostic, run_pullback_dry_diagnostic
from .evaluate import (
    _cvar,
    _create_model,
    _folds,
    _get_features,
    _prep_xy,
    bootstrap_comparison,
    classify_champion,
    compute_tail_metrics,
    compute_weekly_metrics,
)
from .factor_audit import (
    FactorAuditResult,
    audit_redundancy,
    audit_semantic_direction,
    check_code_leakage,
    deterministic_replay_factor,
)
from .panel import get_full_panel, get_universe_panel, write_universe_manifest
from .rdagent_bridge import create_safe_train_dataset, run_rdagent_discovery
from .selectors import SelectorConfig, apply_selector


def cmd_diagnostic(args):
    print("=== Running Phase 0/1 Diagnostics ===")
    panel = get_full_panel()
    u_manifest = write_universe_manifest(panel)
    print(f"Universe manifest created. Total rows: {u_manifest['total_rows']}")
    
    pb_diag = run_pullback_dry_diagnostic(panel)
    print(f"Pullback dry diagnostic created at {OUT / 'pullback_dry_diagnostic.csv'}")
    
    lane_diag = run_lane_diagnostic(panel)
    print(f"Lane diagnostic created at {OUT / 'lane_monotonicity_diagnostic.csv'}")


def cmd_rdagent(args):
    print(f"=== Running Phase 3 RD-Agent Discovery (budget: {args.step_n} steps) ===")
    prov = run_rdagent_discovery(step_n=args.step_n)
    print(f"RD-Agent run completed. Exit code: {prov['exit_code']}. Run ID: {prov['run_id']}")


def cmd_audit(args):
    print("=== Running Phase 4 Factor Audit ===")
    panel = get_full_panel()
    train_dir, debug_dir, full_dir = create_safe_train_dataset(panel)
    replay_ws = AGENT_WORKSPACE / 'replay_audit'
    replay_ws.mkdir(parents=True, exist_ok=True)
    
    # Identify discovered factor python files in RAW_RDAGENT
    RAW_RDAGENT.mkdir(parents=True, exist_ok=True)
    factor_candidates = [p for p in sorted(RAW_RDAGENT.glob('*.py')) if not p.name.startswith('__')]
    
    audit_results = []
    accepted_factor_data = {}
    
    # We audit each factor file
    for fp in factor_candidates:
        fname = fp.stem
        code = fp.read_text(encoding='utf-8')
        
        leak_pass, leak_reason = check_code_leakage(code)
        
        # Test deterministic replay on train data
        replay_ok, replay_msg, res_df, out_hash = deterministic_replay_factor(fname, fp, train_dir, replay_ws)
        
        origin = 'rdagent_original'
        if 'gemini' in code.lower() or 'modified' in fname.lower():
            origin = 'gemini_modified'
            
        desc = "Dynamic volatility and volume-confirmed momentum factor"
        if 'prox' in fname.lower() or 'high' in fname.lower():
            desc = "Proximity to 52-week high (negative distance)"
        elif 'risk_adj_mom_20' in fname:
            desc = "20-day momentum normalized by 20-day realized volatility"
        elif 'risk_adj_mom_10' in fname:
            desc = "10-day momentum normalized by 20-day realized volatility"
        elif 'risk_adj_mom_60' in fname:
            desc = "60-day momentum normalized by 20-day realized volatility"
        elif 'vol_confirmed' in fname:
            desc = "20-day momentum accelerated by 5-to-20 day volume ratio"
            
        sem_ok = True
        sem_msg = "OK"
        red_status = "DISTINCT"
        red_feat = ""
        max_corr = 0.0
        
        if replay_ok and res_df is not None:
            # Check semantic direction & redundancy on Train panel
            tmp = res_df.reset_index()
            tmp.columns = ['datetime', 'code', 'val']
            tmp['snapshot_date'] = pd.to_datetime(tmp['datetime']).dt.strftime('%Y-%m-%d')
            tmp['code'] = tmp['code'].astype(str).str.upper()
            
            merged = panel[panel.snapshot_date <= TRAIN_END].merge(tmp, on=['snapshot_date', 'code'], how='inner')
            if not merged.empty:
                sem_ok, sem_msg = audit_semantic_direction(merged['val'], fname, desc, merged)
                red_status, red_feat, max_corr = audit_redundancy(merged['val'], merged, [*BASE_FEATURES, *TECH_FEATURES])
                
        accepted = bool(leak_pass and replay_ok and sem_ok and (red_status == 'DISTINCT'))
        rej_reason = []
        if not leak_pass: rej_reason.append(leak_reason)
        if not replay_ok: rej_reason.append(replay_msg)
        if not sem_ok: rej_reason.append(sem_msg)
        if red_status != 'DISTINCT': rej_reason.append(f"Redundancy check failed: {red_status} with {red_feat} (corr={max_corr:.3f})")
        
        ar = FactorAuditResult(
            factor_name=fname,
            origin=origin,
            rdagent_run_id='rdagent_track_b_discovery',
            description=desc,
            formula=f"Code in {fp.name}",
            code_path=str(fp.relative_to(ROOT) if fp.is_relative_to(ROOT) else fp.name),
            semantic_direction='POSITIVE',
            leakage_pass=leak_pass,
            redundancy_status=red_status,
            train_only_discovery=True,
            replay_pass=replay_ok,
            accepted=accepted,
            rejection_reason="; ".join(rej_reason),
            redundant_with_feature=red_feat,
            correlation_max=round(max_corr, 4),
            output_hash=out_hash,
        )
        audit_results.append(ar)
        
        # If accepted, replay on full data to import into candidate panel
        if accepted:
            full_ok, _, full_res, _ = deterministic_replay_factor(fname, fp, full_dir, replay_ws / 'full')
            if full_ok and full_res is not None:
                tmp_f = full_res.reset_index()
                tmp_f.columns = ['datetime', 'code', 'val']
                tmp_f['snapshot_date'] = pd.to_datetime(tmp_f['datetime']).dt.strftime('%Y-%m-%d')
                tmp_f['code'] = tmp_f['code'].astype(str).str.upper()
                accepted_factor_data[f"agent_factor_{fname}"] = tmp_f
                
    manifest = {
        'total_audited': len(audit_results),
        'accepted_count': sum(1 for a in audit_results if a.accepted),
        'rejected_count': sum(1 for a in audit_results if not a.accepted),
        'factors': [asdict(a) for a in audit_results],
    }
    (OUT / 'factor_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f"Audit completed: {manifest['accepted_count']} accepted, {manifest['rejected_count']} rejected.")
    for a in audit_results:
        status_str = "ACCEPTED" if a.accepted else f"REJECTED ({a.rejection_reason})"
        print(f"  - {a.factor_name}: {status_str}")
        
    # Update local panel with accepted agent factors
    local_panel_p = DATA / 'candidate_factor_panel.parquet'
    updated_panel = panel.copy()
    for col_name, f_df in accepted_factor_data.items():
        if col_name in updated_panel.columns:
            updated_panel = updated_panel.drop(columns=[col_name])
        merged = updated_panel.merge(f_df[['snapshot_date', 'code', 'val']].rename(columns={'val': col_name}), on=['snapshot_date', 'code'], how='left')
        updated_panel[col_name] = pd.to_numeric(merged[col_name], errors='coerce')
    updated_panel.to_parquet(local_panel_p, index=False)
    print(f"Updated local panel with {len(accepted_factor_data)} accepted agent factor columns.")
    
    # Compute factor diagnostics (IC & Quintiles)
    diag_rows = []
    for a in audit_results:
        f_col = f"agent_factor_{a.factor_name}" if a.accepted else a.factor_name
        if f_col not in updated_panel.columns:
            continue
        for seg, mask in {
            'train': updated_panel.snapshot_date <= TRAIN_END,
            'contaminated_validation': (updated_panel.snapshot_date >= CONTAM_VAL_START) & (updated_panel.snapshot_date <= CONTAM_VAL_END),
        }.items():
            sub = updated_panel[mask]
            for h in (1, 2, 4):
                ret_c = f"w{h}_return_pct"
                ics = []
                for _, g in sub.groupby('snapshot_date'):
                    s = pd.concat([pd.to_numeric(g[f_col], errors='coerce'), pd.to_numeric(g[ret_c], errors='coerce')], axis=1).dropna()
                    if len(s) >= 5 and s.iloc[:, 0].nunique() >= 2 and s.iloc[:, 1].nunique() >= 2:
                        ic = float(s.iloc[:, 0].rank().corr(s.iloc[:, 1].rank()))
                        if np.isfinite(ic):
                            ics.append(ic)
                diag_rows.append({
                    'factor': a.factor_name,
                    'segment': seg,
                    'horizon': f'W{h}',
                    'weeks_ic': len(ics),
                    'mean_ic': round(float(np.mean(ics)), 4) if ics else np.nan,
                    'median_ic': round(float(np.median(ics)), 4) if ics else np.nan,
                    'ic_positive_pct': round(100.0 * float((np.array(ics) > 0).mean()), 2) if ics else np.nan,
                    'accepted': a.accepted,
                })
    pd.DataFrame(diag_rows).to_csv(OUT / 'factor_diagnostics.csv', index=False)
    print("Factor diagnostics CSV saved.")


def cmd_train_and_evaluate(args):
    print("=== Running Track B Train OOF & Evaluation Pipeline ===")
    panel = get_full_panel()
    OUT.mkdir(parents=True, exist_ok=True)
    
    # 1. Define Selectors
    selector_configs = [
        SelectorConfig('pure_rank', 'pure_rank', 'none', 'max_score', complexity='low'),
        SelectorConfig('distinct_industry', 'distinct_industry', 'hard_distinct', 'max_score_distinct_ind', complexity='low'),
        SelectorConfig(
            'portfolio_aware', 'portfolio_aware', 'soft_penalty',
            'score_minus_vol_and_overheat_and_dup_ind',
            lambda_vol=0.5, lambda_overheat=0.3, lambda_industry_dup=2.0, candidate_pool_size=8, complexity='medium'
        ),
    ]
    
    # Save selector registry
    sel_rows = [
        {
            'selector_id': sc.selector_id,
            'family': sc.family,
            'industry_constraint': sc.industry_constraint,
            'portfolio_objective': sc.portfolio_objective,
            'lambda_vol': sc.lambda_vol,
            'lambda_overheat': sc.lambda_overheat,
            'lambda_industry_dup': sc.lambda_industry_dup,
            'candidate_pool_size': sc.candidate_pool_size,
            'complexity': sc.complexity,
        }
        for sc in selector_configs
    ]
    pd.DataFrame(sel_rows).to_csv(OUT / 'selector_registry.csv', index=False)
    
    # 2. Build Models across Universes S (Signal) & A (ACTIONABLE)
    models = ['ridge', 'elastic', 'lgbm']
    feature_modes = ['f1']
    # Check if agent factors exist in panel
    if any(c.startswith('agent_factor_') for c in panel.columns):
        feature_modes.append('agent')
        
    all_predictions = []
    
    train_weeks = sorted(panel.loc[panel.snapshot_date <= TRAIN_END, 'snapshot_date'].unique())
    sealed_train_weeks = train_weeks[:-PURGE_WEEKS] if len(train_weeks) > PURGE_WEEKS else []
    folds = _folds(train_weeks)
    
    for u_name in UNIVERSES:
        u_panel = get_universe_panel(panel, u_name)
        
        for f_mode in feature_modes:
            features = _get_features(u_panel, f_mode)
            
            for h in PRIMARY_HORIZONS:
                label = f'w{h}_return_pct'
                
                for m_name in models:
                    model_template = _create_model(m_name)
                    if model_template is None:
                        continue
                        
                    # Train OOF walk-forward folds
                    for fold_idx, (tr_w, va_w) in enumerate(folds):
                        tr_data = u_panel[u_panel.snapshot_date.isin(tr_w) & u_panel[label].notna()]
                        va_data = u_panel[u_panel.snapshot_date.isin(va_w)].copy()
                        if tr_data.empty or va_data.empty:
                            continue
                            
                        xt, y, xs = _prep_xy(tr_data, va_data, features, label)
                        m = _create_model(m_name)
                        m.fit(xt, y)
                        
                        keep_cols = ['snapshot_date', 'code', 'industry', 'current_vs_ibd_candidate_pct', 'rv_20']
                        for x_col in [f'w{x}_return_pct' for x in (*PRIMARY_HORIZONS, *DIAGNOSTIC_HORIZONS)]:
                            if x_col in va_data.columns: keep_cols.append(x_col)
                        for x_col in [f'w{x}_stop8' for x in (*PRIMARY_HORIZONS, *DIAGNOSTIC_HORIZONS)]:
                            if x_col in va_data.columns: keep_cols.append(x_col)
                            
                        z = va_data[[c for c in keep_cols if c in va_data.columns]].copy()
                        z['score'] = m.predict(xs)
                        z['universe'] = u_name
                        z['feature_mode'] = f_mode
                        z['model_type'] = m_name
                        z['target_horizon'] = f'W{h}'
                        z['model_id'] = f"{u_name}_{f_mode}_{m_name}_w{h}"
                        z['segment'] = 'train_oof'
                        z['fold'] = fold_idx
                        all_predictions.append(z)
                        
                    # Sealed Validation prediction (Train sealed -> Contaminated Validation)
                    tr_data = u_panel[u_panel.snapshot_date.isin(sealed_train_weeks) & u_panel[label].notna()]
                    va_data = u_panel[(u_panel.snapshot_date >= CONTAM_VAL_START) & (u_panel.snapshot_date <= CONTAM_VAL_END)].copy()
                    if not tr_data.empty and not va_data.empty:
                        xt, y, xs = _prep_xy(tr_data, va_data, features, label)
                        m = _create_model(m_name)
                        m.fit(xt, y)
                        
                        keep_cols = ['snapshot_date', 'code', 'industry', 'current_vs_ibd_candidate_pct', 'rv_20']
                        for x_col in [f'w{x}_return_pct' for x in (*PRIMARY_HORIZONS, *DIAGNOSTIC_HORIZONS)]:
                            if x_col in va_data.columns: keep_cols.append(x_col)
                        for x_col in [f'w{x}_stop8' for x in (*PRIMARY_HORIZONS, *DIAGNOSTIC_HORIZONS)]:
                            if x_col in va_data.columns: keep_cols.append(x_col)
                            
                        z = va_data[[c for c in keep_cols if c in va_data.columns]].copy()
                        z['score'] = m.predict(xs)
                        z['universe'] = u_name
                        z['feature_mode'] = f_mode
                        z['model_type'] = m_name
                        z['target_horizon'] = f'W{h}'
                        z['model_id'] = f"{u_name}_{f_mode}_{m_name}_w{h}"
                        z['segment'] = 'contaminated_validation'
                        z['fold'] = -1
                        all_predictions.append(z)
                        
    pred_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    pred_df.to_parquet(OUT / 'cv_predictions.parquet', index=False)
    print(f"Saved {len(pred_df)} prediction rows to cv_predictions.parquet")
    
    # 3. Apply Selectors to form Top3 Picks & Weekly Metrics
    all_picks = []
    all_weekly_metrics = []
    
    if not pred_df.empty:
        for (m_id, seg), g in pred_df.groupby(['model_id', 'segment']):
            for sel_cfg in selector_configs:
                full_sel_id = f"{m_id}_{sel_cfg.selector_id}"
                picks = apply_selector(g, sel_cfg, score_col='score')
                if not picks.empty:
                    picks['selector_id'] = full_sel_id
                    picks['segment'] = seg
                    all_picks.append(picks)
                    
                    w_mets = compute_weekly_metrics(picks, full_sel_id, seg)
                    all_weekly_metrics.extend(w_mets)
                    
    # 4. Add Frozen B0 Baseline Picks & Metrics
    b0 = panel[panel.is_b0 == 1].copy()
    b0['selector_id'] = 'B0'
    b0['model_pick_order'] = b0.get('pick_order', 1)
    
    b0_train = b0[b0.snapshot_date <= TRAIN_END].copy()
    b0_val = b0[(b0.snapshot_date >= CONTAM_VAL_START) & (b0.snapshot_date <= CONTAM_VAL_END)].copy()
    
    b0_train_picks = b0_train.assign(segment='train_oof')
    b0_val_picks = b0_val.assign(segment='contaminated_validation')
    
    all_picks.append(b0_train_picks)
    all_picks.append(b0_val_picks)
    
    all_weekly_metrics.extend(compute_weekly_metrics(b0_train_picks, 'B0', 'train_oof'))
    all_weekly_metrics.extend(compute_weekly_metrics(b0_val_picks, 'B0', 'contaminated_validation'))
    
    df_picks = pd.concat(all_picks, ignore_index=True, sort=False)
    df_weekly = pd.DataFrame(all_weekly_metrics)
    
    df_picks.to_csv(OUT / 'top3_picks.csv', index=False)
    df_weekly.to_csv(OUT / 'weekly_metrics.csv', index=False)
    print(f"Top3 picks ({len(df_picks)}) and weekly metrics ({len(df_weekly)}) saved.")
    
    # 5. Compute Tail Metrics & Concentration
    df_tail = compute_tail_metrics(df_weekly)
    df_tail.to_csv(OUT / 'tail_metrics.csv', index=False)
    
    # 6. Paired Comparison vs B0 (Primary: Full3 Common Support)
    paired_rows = []
    b0_mets = df_weekly[df_weekly.selector_id == 'B0']
    
    b0_grouped = b0_mets[['segment', 'snapshot_date', 'horizon', 'return_pct', 'stop_rate']].rename(
        columns={'return_pct': 'b0_return', 'stop_rate': 'b0_stop'}
    )
    
    for (sel_id, seg, h), m in df_weekly[df_weekly.selector_id != 'B0'].groupby(['selector_id', 'segment', 'horizon']):
        q = m.merge(b0_grouped, on=['segment', 'snapshot_date', 'horizon'], how='inner')
        if q.empty:
            continue
        spread = q['return_pct'] - q['b0_return']
        stop_delta = q['stop_rate'] - q['b0_stop']
        
        # Calculate CVaR delta
        ch_cvar = _cvar(q['return_pct'], 0.10)
        b0_cvar = _cvar(q['b0_return'], 0.10)
        cvar_delta = ch_cvar - b0_cvar if np.isfinite(ch_cvar) and np.isfinite(b0_cvar) else np.nan
        
        paired_rows.append({
            'selector_id': sel_id,
            'segment': seg,
            'horizon': h,
            'weeks': len(q),
            'median_spread_pct': float(spread.median()),
            'mean_spread_pct': float(spread.mean()),
            'beat_b0_pct': float(100.0 * (spread > 0).mean()),
            'stop_delta_pct': float(100.0 * stop_delta.mean()),
            'cvar_delta': cvar_delta,
            'worst_spread_pct': float(spread.min()),
            'best_spread_pct': float(spread.max()),
        })
        
    df_paired = pd.DataFrame(paired_rows)
    df_paired.to_csv(OUT / 'b0_paired_comparison.csv', index=False)
    print(f"Paired comparison ({len(df_paired)}) saved.")
    
    # 7. Deterministic Bootstrap for Primary W4 Comparisons
    bootstrap_rows = []
    b0_w4_val = df_weekly[(df_weekly.selector_id == 'B0') & (df_weekly.segment == 'contaminated_validation') & (df_weekly.horizon == 'W4')]
    
    for sel_id, g in df_weekly[(df_weekly.selector_id != 'B0') & (df_weekly.segment == 'contaminated_validation') & (df_weekly.horizon == 'W4')].groupby('selector_id'):
        b_res = bootstrap_comparison(g, b0_w4_val)
        bootstrap_rows.append({
            'selector_id': sel_id,
            'horizon': 'W4',
            'support_weeks': b_res['support_weeks'],
            'mean_spread_pct': b_res['mean_spread_pct'],
            'mean_spread_ci_low': b_res['mean_spread_ci_95'][0],
            'mean_spread_ci_high': b_res['mean_spread_ci_95'][1],
            'median_spread_pct': b_res['median_spread_pct'],
            'median_spread_ci_low': b_res['median_spread_ci_95'][0],
            'median_spread_ci_high': b_res['median_spread_ci_95'][1],
            'cvar_diff_ci_low': b_res['cvar_diff_ci_95'][0],
            'cvar_diff_ci_high': b_res['cvar_diff_ci_95'][1],
        })
    df_boot = pd.DataFrame(bootstrap_rows)
    df_boot.to_csv(OUT / 'bootstrap_summary.csv', index=False)
    
    # 8. Champion Matrix
    champ_rows = []
    for sel_id in df_paired['selector_id'].unique():
        tr_w4 = df_paired[(df_paired.selector_id == sel_id) & (df_paired.segment == 'train_oof') & (df_paired.horizon == 'W4')]
        val_w4 = df_paired[(df_paired.selector_id == sel_id) & (df_paired.segment == 'contaminated_validation') & (df_paired.horizon == 'W4')]
        
        tr_dict = tr_w4.iloc[0].to_dict() if not tr_w4.empty else {}
        val_dict = val_w4.iloc[0].to_dict() if not val_w4.empty else {}
        
        boot_sub = df_boot[df_boot.selector_id == sel_id]
        boot_dict = boot_sub.iloc[0].to_dict() if not boot_sub.empty else {}
        
        classification = classify_champion(tr_dict, val_dict, boot_dict)
        champ_rows.append({
            'selector_id': sel_id,
            'classification': classification,
            'train_oof_w4_med_spread': tr_dict.get('median_spread_pct', np.nan),
            'train_oof_w4_cvar_delta': tr_dict.get('cvar_delta', np.nan),
            'train_oof_w4_stop_delta': tr_dict.get('stop_delta_pct', np.nan),
            'val_w4_med_spread': val_dict.get('median_spread_pct', np.nan),
            'val_w4_mean_spread': val_dict.get('mean_spread_pct', np.nan),
            'val_w4_cvar_delta': val_dict.get('cvar_delta', np.nan),
            'val_w4_stop_delta': val_dict.get('stop_delta_pct', np.nan),
            'val_support_weeks': val_dict.get('weeks', 0),
        })
    df_champ = pd.DataFrame(champ_rows).sort_values(['val_w4_med_spread', 'val_w4_mean_spread'], ascending=[False, False])
    df_champ.to_csv(OUT / 'champion_matrix.csv', index=False)
    print(f"Champion matrix generated with {len(df_champ)} models.")


def cmd_seal(args):
    print("=== Writing Research Lock Manifest ===")
    git_sha = "unknown"
    try:
        git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        pass
        
    manifest = {
        'lock_timestamp': datetime.datetime.now().isoformat(),
        'git_sha': git_sha,
        'train_boundary': f'<= {TRAIN_END}',
        'contaminated_validation_boundary': f'{CONTAM_VAL_START} .. {CONTAM_VAL_END}',
        'purge_weeks': PURGE_WEEKS,
        'universes': list(UNIVERSES),
        'base_features': BASE_FEATURES,
        'tech_features': TECH_FEATURES,
        'random_seed': RANDOM_SEED,
        'bootstrap_rounds': BOOTSTRAP_ROUNDS,
    }
    (OUT / 'research_lock_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print("Research state sealed.")


def main():
    parser = argparse.ArgumentParser(description="RD-Agent Track B Selector Challenge CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    p_diag = subparsers.add_parser("diagnostic", help="Run pullback and lane diagnostics")
    p_diag.set_defaults(func=cmd_diagnostic)
    
    p_rd = subparsers.add_parser("rdagent", help="Run RD-Agent discovery")
    p_rd.add_argument("--step-n", type=int, default=3, help="RD-Agent evolving loop step count")
    p_rd.set_defaults(func=cmd_rdagent)
    
    p_audit = subparsers.add_parser("audit", help="Run 4-stage factor audit")
    p_audit.set_defaults(func=cmd_audit)
    
    p_eval = subparsers.add_parser("evaluate", help="Train models and evaluate selectors")
    p_eval.set_defaults(func=cmd_train_and_evaluate)
    
    p_seal = subparsers.add_parser("seal", help="Seal research manifest")
    p_seal.set_defaults(func=cmd_seal)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
