#!/usr/bin/env python
"""CLI Tool for Point-in-Time EPS YoY Growth Backfill and Audit.

Usage:
    python tools/backfill_eps_pit.py audit
    python tools/backfill_eps_pit.py validate --sample-size 100
    python tools/backfill_eps_pit.py fetch
    python tools/backfill_eps_pit.py build-events
    python tools/backfill_eps_pit.py backfill --pit-mode conservative
    python tools/backfill_eps_pit.py report
    python tools/backfill_eps_pit.py all
"""

import os
import sys
import argparse
import json
import time
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eps_pit.audit import ReplayPoolAuditor
from eps_pit.mapping import TickerMapper
from eps_pit.providers.factory import ProviderFactory
from eps_pit.pit import PITTimelineEngine
from eps_pit.backfill import ReplayPoolBackfiller


def run_audit(args):
    print("=== [Phase 0] Scanning Replay Pools & Input Inventory ===")
    auditor = ReplayPoolAuditor(base_dir=args.pool_dir, output_dir=args.output_dir)
    inv_df, univ_df, summary = auditor.scan_inventory()
    
    # Build ticker mapping
    print("Building ticker mapping...")
    provider = ProviderFactory.get_provider("composite", cache_dir=os.path.join(args.output_dir, "cache", "raw"))
    mapper = TickerMapper(sec_provider=provider.sec)
    mapping_df = mapper.build_mapping_table(univ_df["code"].tolist())
    
    mapping_path = os.path.join(args.output_dir, "audit", "ticker_mapping.csv")
    mapping_df.to_csv(mapping_path, index=False)

    print("\n--- Phase 0 Summary ---")
    print(f"Weekly CSV Files: {summary['files_count']}")
    print(f"Total Rows: {summary['total_rows']}")
    print(f"Total Signal Rows: {summary['total_signal_rows']}")
    print(f"Total Unique Tickers: {summary['unique_codes_total']}")
    print(f"Signal Unique Tickers: {summary['unique_signal_codes']}")
    print(f"Snapshot Range: {summary['earliest_snapshot']} to {summary['latest_snapshot']}")
    print(f"Mapping Table: {len(mapping_df)} tickers mapped -> {mapping_path}")
    print(f"Inventory Table -> {os.path.join(args.output_dir, 'audit', 'input_inventory.csv')}")
    print(f"Universe Table -> {os.path.join(args.output_dir, 'audit', 'ticker_universe.csv')}")
    return summary


def select_validation_sample(univ_df: pd.DataFrame, sample_size: int = 100) -> List[str]:
    """Stratified sampling of tickers across frequencies, sectors, and special types."""
    # Specific landmark tickers
    priority_tickers = [
        "NVDA", "AAPL", "MSFT", "AMZN", "GOOG", "GOOGL", "META", "TSLA", "AVGO", "COST",
        "NFLX", "PFIS", "WULF", "MPC", "TMP", "ALL", "KO", "ROST", "BWFG", "OSBC",
        "ASML", "TSM", "ARM", "BABA", "PDD", "BRK.B", "BF.B", "CLMT", "URGN", "SPHR"
    ]
    
    sample_set = set()
    for t in priority_tickers:
        if t in univ_df["code"].values:
            sample_set.add(t)

    # Add top frequent tickers
    top_tickers = univ_df.head(40)["code"].tolist()
    sample_set.update(top_tickers)

    # Add tickers across different sectors if available
    if "sector" in univ_df.columns:
        for sector, grp in univ_df.groupby("sector"):
            sample_set.update(grp.head(3)["code"].tolist())

    # Fill remaining from universe
    for t in univ_df["code"].tolist():
        if len(sample_set) >= sample_size:
            break
        sample_set.add(t)

    return sorted(list(sample_set))[:sample_size]


def run_validate(args):
    print(f"=== [Phase 1] 100-Ticker Representative Sample Validation ===")
    auditor = ReplayPoolAuditor(base_dir=args.pool_dir, output_dir=args.output_dir)
    _, univ_df, _ = auditor.scan_inventory()

    sample_size = args.sample_size or 100
    sample_tickers = select_validation_sample(univ_df, sample_size=sample_size)
    print(f"Selected {len(sample_tickers)} validation tickers.")

    val_dir = os.path.join(args.output_dir, "validation")
    os.makedirs(val_dir, exist_ok=True)

    # Save sample list
    sample_df = univ_df[univ_df["code"].isin(sample_tickers)].copy()
    sample_df.to_csv(os.path.join(val_dir, "sample_100.csv"), index=False)

    # Fetch data for sample
    max_workers = args.workers or 8
    print(f"Fetching quarterly fundamentals for {len(sample_tickers)} sample tickers with {max_workers} threads...")
    provider = ProviderFactory.get_provider("composite", cache_dir=os.path.join(args.output_dir, "cache", "raw"))
    
    all_sample_records = []
    fetch_success = 0
    fetch_fail = 0

    def fetch_sample_one(sym):
        try:
            recs = provider.fetch_quarterly_history(sym)
            return sym, recs
        except Exception as e:
            return sym, []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_sample_one, t): t for t in sample_tickers}
        for fut in as_completed(futures):
            sym, recs = fut.result()
            if recs:
                all_sample_records.extend(recs)
                fetch_success += 1
            else:
                fetch_fail += 1

    print(f"Fetch completed: {fetch_success} succeeded, {fetch_fail} failed (Coverage: {fetch_success/len(sample_tickers)*100:.1f}%)")

    # Build events
    events_df = PITTimelineEngine.build_growth_events(all_sample_records)
    print(f"Built {len(events_df)} PIT growth events for sample.")

    # Load reference TradingView / Whitelist data
    ref_file = "us/stage2/stage2_whitelist.csv"
    ref_map = {}
    if os.path.exists(ref_file):
        try:
            df_ref = pd.read_csv(ref_file)
            if "code" in df_ref.columns and "eps_yoy_growth" in df_ref.columns:
                for _, r in df_ref.dropna(subset=["eps_yoy_growth"]).iterrows():
                    ref_map[str(r["code"]).upper().strip()] = float(r["eps_yoy_growth"])
        except Exception as e:
            print(f"Warning: could not load reference data: {e}")

    # Compare latest event against reference
    compare_records = []
    special_case_records = []

    for sym in sample_tickers:
        sym_events = events_df[events_df["code"] == sym]
        if sym_events.empty:
            compare_records.append({
                "code": sym,
                "our_eps_yoy_growth": None,
                "reference_eps_yoy_growth": ref_map.get(sym),
                "abs_diff": None,
                "eps_current": None,
                "eps_prior_year": None,
                "report_period": None,
                "effective_at_conservative": None,
                "effective_at_release": None,
                "growth_status": "MISSING_SOURCE",
                "status": "MISSING_SOURCE",
            })
            continue

        latest_ev = sym_events.iloc[-1]
        our_val = latest_ev["eps_yoy_growth"]
        ref_val = ref_map.get(sym)
        growth_status = latest_ev["growth_status"]

        status = "MATCH"
        abs_diff = None
        if our_val is not None and ref_val is not None:
            abs_diff = abs(our_val - ref_val)
            if abs_diff <= 0.05:
                status = "MATCH"
            elif abs_diff <= 2.0:
                status = "ROUNDING_DIFF"
            else:
                status = "FORMULA_OR_TIMING_DIFF"
        elif ref_val is None and our_val is not None:
            status = "NO_REFERENCE_VALUE"
        else:
            status = growth_status

        if growth_status != "NORMAL_POSITIVE":
            special_case_records.append({
                "code": sym,
                "report_period": latest_ev["report_period"],
                "eps_current": latest_ev["eps_current"],
                "eps_prior_year": latest_ev["eps_prior_year"],
                "eps_yoy_growth": our_val,
                "growth_status": growth_status,
            })

        compare_records.append({
            "code": sym,
            "our_eps_yoy_growth": our_val,
            "reference_eps_yoy_growth": ref_val,
            "abs_diff": abs_diff,
            "eps_current": latest_ev["eps_current"],
            "eps_prior_year": latest_ev["eps_prior_year"],
            "report_period": latest_ev["report_period"],
            "effective_at_conservative": latest_ev["effective_at_conservative"],
            "effective_at_release": latest_ev["effective_at_release"],
            "growth_status": growth_status,
            "status": status,
        })

    df_comp = pd.DataFrame(compare_records)
    df_comp.to_csv(os.path.join(val_dir, "tradingview_compare.csv"), index=False)

    df_special = pd.DataFrame(special_case_records)
    df_special.to_csv(os.path.join(val_dir, "special_eps_cases.csv"), index=False)

    # Validation Metrics
    normal_pos = df_comp[df_comp["growth_status"] == "NORMAL_POSITIVE"]
    matched_pos = normal_pos[normal_pos["status"].isin(["MATCH", "ROUNDING_DIFF", "NO_REFERENCE_VALUE"])]
    match_rate = len(matched_pos) / len(normal_pos) * 100 if len(normal_pos) > 0 else 0.0

    print("\n--- Phase 1 Validation Metrics ---")
    print(f"Sample Size: {len(sample_tickers)}")
    print(f"Data Source Fetch Coverage: {fetch_success}/{len(sample_tickers)} ({fetch_success/len(sample_tickers)*100:.1f}%)")
    print(f"Normal Positive Cases: {len(normal_pos)}")
    print(f"Normal Positive Match Rate: {len(matched_pos)}/{len(normal_pos)} ({match_rate:.1f}%)")
    print(f"Special Cases Count: {len(df_special)}")
    print(f"Comparison Table -> {os.path.join(val_dir, 'tradingview_compare.csv')}")
    print(f"Special Cases Table -> {os.path.join(val_dir, 'special_eps_cases.csv')}")

    return {
        "sample_size": len(sample_tickers),
        "fetch_coverage_pct": fetch_success / len(sample_tickers) * 100,
        "normal_pos_count": len(normal_pos),
        "normal_pos_match_rate_pct": match_rate,
        "special_cases_count": len(df_special),
    }


def run_fetch(args):
    print("=== [Phase 2] Fetching Fundamentals for All Tickers ===")
    auditor = ReplayPoolAuditor(base_dir=args.pool_dir, output_dir=args.output_dir)
    _, univ_df, _ = auditor.scan_inventory()

    tickers = univ_df["code"].tolist()
    print(f"Total tickers to fetch: {len(tickers)}")

    cache_dir = os.path.join(args.output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    provider = ProviderFactory.get_provider("composite", cache_dir=os.path.join(cache_dir, "raw"))

    status_file = os.path.join(cache_dir, "fetch_status.csv")
    fetch_status = {}
    if os.path.exists(status_file):
        try:
            df_st = pd.read_csv(status_file)
            for _, r in df_st.iterrows():
                fetch_status[r["code"]] = r.to_dict()
        except Exception:
            pass

    max_workers = args.workers or 8
    print(f"Starting multi-threaded fetch with {max_workers} workers...")

    def fetch_one(sym):
        try:
            recs = provider.fetch_quarterly_history(sym)
            return sym, True, len(recs), None
        except Exception as e:
            return sym, False, 0, str(e)

    to_fetch = [t for t in tickers if t not in fetch_status or fetch_status[t].get("status") != "SUCCESS"]
    print(f"Tickers remaining to fetch (resume mode): {len(to_fetch)} / {len(tickers)}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_one, t): t for t in to_fetch}
        done_count = 0
        for fut in as_completed(futures):
            sym, ok, count, err = fut.result()
            fetch_status[sym] = {
                "code": sym,
                "source_symbol": sym,
                "provider": "composite",
                "status": "SUCCESS" if ok and count > 0 else ("NO_DATA" if ok else "ERROR"),
                "records": count,
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "error": err,
            }
            done_count += 1
            if done_count % 100 == 0 or done_count == len(to_fetch):
                print(f"Progress: {done_count}/{len(to_fetch)} ({done_count/len(to_fetch)*100:.1f}%)")

    # Save status
    pd.DataFrame(list(fetch_status.values())).to_csv(status_file, index=False)
    print(f"Fetch status saved -> {status_file}")


def run_build_events(args):
    print("=== Building Standardized Fundamentals & PIT Events ===")
    cache_dir = os.path.join(args.output_dir, "cache")
    provider = ProviderFactory.get_provider("composite", cache_dir=os.path.join(cache_dir, "raw"))

    auditor = ReplayPoolAuditor(base_dir=args.pool_dir, output_dir=args.output_dir)
    _, univ_df, _ = auditor.scan_inventory()
    tickers = univ_df["code"].tolist()

    all_records = []
    max_workers = args.workers or 16
    print(f"Extracting quarterly records for {len(tickers)} tickers with {max_workers} threads...")

    def extract_one(sym):
        try:
            return provider.fetch_quarterly_history(sym)
        except Exception as e:
            return []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_one, t): t for t in tickers}
        for fut in as_completed(futures):
            recs = fut.result()
            if recs:
                all_records.extend(recs)

    # Save standardized quarterly history
    df_hist = pd.DataFrame(all_records)
    hist_path = os.path.join(cache_dir, "eps_quarterly_history.parquet")
    if not df_hist.empty:
        df_hist.to_parquet(hist_path, index=False)
    print(f"Standardized history saved: {len(df_hist)} records -> {hist_path}")

    # Build events
    print("Building PIT growth events...")
    events_df = PITTimelineEngine.build_growth_events(all_records)
    events_path = os.path.join(cache_dir, "eps_growth_events.parquet")
    if not events_df.empty:
        events_df.to_parquet(events_path, index=False)
    print(f"PIT Growth events saved: {len(events_df)} events -> {events_path}")
    return events_df


def run_backfill(args):
    print(f"=== [Phase 2] Safe Backfill 32 Weekly CSVs (Mode: {args.pit_mode}) ===")
    cache_dir = os.path.join(args.output_dir, "cache")
    events_path = os.path.join(cache_dir, "eps_growth_events.parquet")

    if not os.path.exists(events_path):
        print(f"Events file not found at {events_path}. Running build-events first...")
        events_df = run_build_events(args)
    else:
        events_df = pd.read_parquet(events_path)

    backfiller = ReplayPoolBackfiller(
        base_dir=args.pool_dir,
        output_dir=args.output_dir,
        pit_mode=args.pit_mode
    )
    summary = backfiller.backfill_all(events_df)
    print("\n--- Backfill Summary ---")
    print(f"Total Rows: {summary['total_rows']}")
    print(f"Rows Filled: {summary['rows_filled']}")
    print(f"Rows Missing/Unresolved: {summary['rows_unresolved']}")
    print(f"Overall Coverage: {summary['coverage_pct']}%")
    print(f"Patched CSVs -> {os.path.join(args.output_dir, 'patched')}")
    print(f"Audit Provenance -> {os.path.join(args.output_dir, 'audit', 'weekly_eps_provenance.parquet')}")
    return summary


def run_report(args):
    print("=== Generating Comprehensive Audit & Coverage Reports ===")
    audit_dir = os.path.join(args.output_dir, "audit")
    cache_dir = os.path.join(args.output_dir, "cache")
    
    # 1. Unresolved tickers
    status_file = os.path.join(cache_dir, "fetch_status.csv")
    events_file = os.path.join(cache_dir, "eps_growth_events.parquet")
    
    events_codes = set()
    if os.path.exists(events_file):
        df_ev = pd.read_parquet(events_file)
        events_codes = set(df_ev["code"].dropna().unique())

    auditor = ReplayPoolAuditor(base_dir=args.pool_dir, output_dir=args.output_dir)
    _, univ_df, _ = auditor.scan_inventory()

    unresolved = []
    source_errors = []
    
    if os.path.exists(status_file):
        df_st = pd.read_csv(status_file)
        for _, r in df_st.iterrows():
            sym = r["code"]
            st = r["status"]
            err = r.get("error")
            if st != "SUCCESS":
                source_errors.append({
                    "code": sym,
                    "provider": r.get("provider", "composite"),
                    "status": st,
                    "error": err,
                    "fetched_at": r.get("fetched_at"),
                })
            if sym not in events_codes:
                unresolved.append({
                    "code": sym,
                    "reason": "NO_FUNDAMENTALS_DATA" if st == "NO_DATA" else ("PROVIDER_ERROR" if err else "INSUFFICIENT_QUARTERS"),
                    "details": err or f"Status: {st}",
                })

    df_unres = pd.DataFrame(unresolved)
    df_unres.to_csv(os.path.join(audit_dir, "unresolved_tickers.csv"), index=False)

    df_err = pd.DataFrame(source_errors)
    df_err.to_csv(os.path.join(audit_dir, "source_errors.csv"), index=False)

    # 2. Special cases
    if os.path.exists(events_file):
        df_ev = pd.read_parquet(events_file)
        special_ev = df_ev[df_ev["growth_status"] != "NORMAL_POSITIVE"]
        special_ev.to_csv(os.path.join(audit_dir, "special_cases.csv"), index=False)

    cov_sum_file = os.path.join(audit_dir, "coverage_summary.json")
    if os.path.exists(cov_sum_file):
        with open(cov_sum_file, "r") as f:
            data = json.load(f)
            data["unresolved_count"] = len(df_unres)
            data["special_cases_count"] = len(special_ev) if os.path.exists(events_file) else 0
            data["source_errors_count"] = len(df_err)
            print(json.dumps(data, indent=2))
        with open(cov_sum_file, "w") as f:
            json.dump(data, f, indent=2)


def run_export_signal_eps(args):
    print("=== Exporting 32-Week Signal PIT EPS Dataset ===")
    import glob
    prov_path = os.path.join(args.output_dir, "audit", "weekly_eps_provenance.parquet")
    if not os.path.exists(prov_path):
        print(f"Provenance file not found at {prov_path}. Running backfill first...")
        run_backfill(args)

    df_prov = pd.read_parquet(prov_path)

    # Scan weekly pool files for signal rows
    pattern = os.path.join(args.pool_dir, "*", "breakout_follow_pool.csv")
    pool_files = sorted(glob.glob(pattern))
    signal_rows = []
    for f in pool_files:
        snap_date = os.path.basename(os.path.dirname(f))
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        sig_col = "signal" if "signal" in df.columns else ("breakout_signal" if "breakout_signal" in df.columns else None)
        if sig_col:
            sig_df = df[df[sig_col] == True]
            for _, r in sig_df.iterrows():
                code = str(r["code"]).strip().upper()
                signal_rows.append({"snapshot_date": snap_date, "code": code})

    df_sig = pd.DataFrame(signal_rows).drop_duplicates(subset=["snapshot_date", "code"])
    merged = df_sig.merge(df_prov, on=["snapshot_date", "code"], how="inner")
    merged = merged.sort_values(by=["snapshot_date", "code"]).reset_index(drop=True)

    target_csv = os.path.join(args.pool_dir, "signal_eps_pit.csv")
    merged.to_csv(target_csv, index=False)
    print(f"Exported {len(merged)} signal PIT EPS records -> {target_csv}")
    return merged


def main():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--pool-dir", default="backtest/ibd_skill_replay_pools", help="Path to weekly pool folders")
    parent_parser.add_argument("--output-dir", default="outputs/eps_pit_backfill", help="Output directory")
    parent_parser.add_argument("--workers", type=int, default=8, help="Number of concurrent download threads")
    parent_parser.add_argument("--sample-size", type=int, default=100, help="Validation sample size")
    parent_parser.add_argument("--pit-mode", default="conservative", choices=["conservative", "release"], help="PIT date mode")

    parser = argparse.ArgumentParser(description="Point-in-Time EPS YoY Growth Backfill & Audit Tool", parents=[parent_parser])
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to execute")
    
    subparsers.add_parser("audit", parents=[parent_parser], help="Phase 0: Run inventory audit")
    subparsers.add_parser("validate", parents=[parent_parser], help="Phase 1: Run 100-ticker validation")
    subparsers.add_parser("fetch", parents=[parent_parser], help="Phase 2: Fetch fundamentals data")
    subparsers.add_parser("build-events", parents=[parent_parser], help="Build PIT growth events")
    subparsers.add_parser("backfill", parents=[parent_parser], help="Backfill 32 weekly CSVs")
    subparsers.add_parser("export-signal-eps", parents=[parent_parser], help="Export 32-week signal PIT EPS dataset")
    subparsers.add_parser("report", parents=[parent_parser], help="Print audit reports")
    subparsers.add_parser("all", parents=[parent_parser], help="Run full pipeline end-to-end")

    args = parser.parse_args()

    if args.command == "audit":
        run_audit(args)
    elif args.command == "validate":
        run_audit(args)
        run_validate(args)
    elif args.command == "fetch":
        run_fetch(args)
    elif args.command == "build-events":
        run_build_events(args)
    elif args.command == "backfill":
        run_backfill(args)
        run_export_signal_eps(args)
    elif args.command == "export-signal-eps":
        run_export_signal_eps(args)
    elif args.command == "report":
        run_report(args)
    elif args.command == "all" or not args.command:
        # End to end
        run_audit(args)
        val_res = run_validate(args)
        # Validation gate
        if val_res["normal_pos_match_rate_pct"] < 90.0:
            print(f"Validation gate FAILED: match rate {val_res['normal_pos_match_rate_pct']:.1f}% < 90.0%")
            sys.exit(1)
        print("Validation gate PASSED! Proceeding to Phase 2...")
        run_fetch(args)
        events_df = run_build_events(args)
        run_backfill(args)
        run_export_signal_eps(args)
        run_report(args)


if __name__ == "__main__":
    main()
