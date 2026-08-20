import os
import glob
import json
import pytest
import pandas as pd


def test_patched_files_integrity():
    """Verify all 32 patched files exist and match original structure."""
    orig_files = sorted(glob.glob("backtest/ibd_skill_replay_pools/*/breakout_follow_pool.csv"))
    patched_files = sorted(glob.glob("outputs/eps_pit_backfill/patched/*/breakout_follow_pool.csv"))

    assert len(orig_files) == 32
    assert len(patched_files) == 32

    for o_path, p_path in zip(orig_files, patched_files):
        o_df = pd.read_csv(o_path)
        p_df = pd.read_csv(p_path)

        # Exact row count
        assert len(o_df) == len(p_df), f"Row count mismatch in {o_path}"
        # Exact code order
        assert list(o_df["code"]) == list(p_df["code"]), f"Code order mismatch in {o_path}"
        # Non-target columns
        other_cols = [c for c in o_df.columns if c != "eps_yoy_growth"]
        for col in other_cols:
            if o_df[col].dtype == float:
                diff = (o_df[col] - p_df[col]).abs().max()
                assert diff < 1e-5 or (o_df[col].isna() & p_df[col].isna()).all()
            else:
                assert o_df[col].fillna("").equals(p_df[col].fillna(""))


def test_zero_future_leakage_in_provenance():
    """Verify strictly no future data leakage across all 11,895 rows."""
    prov_path = "outputs/eps_pit_backfill/audit/weekly_eps_provenance.parquet"
    assert os.path.exists(prov_path)
    df_prov = pd.read_parquet(prov_path)

    filled = df_prov.dropna(subset=["eps_yoy_growth", "effective_at_conservative"])
    leakage = filled[filled["effective_at_conservative"] > filled["snapshot_date"]]
    assert len(leakage) == 0, f"Future leakage found: {len(leakage)} rows"


def test_coverage_and_audit_artifacts_exist():
    """Verify all required audit artifacts are generated."""
    audit_dir = "outputs/eps_pit_backfill/audit"
    assert os.path.exists(os.path.join(audit_dir, "input_inventory.csv"))
    assert os.path.exists(os.path.join(audit_dir, "ticker_universe.csv"))
    assert os.path.exists(os.path.join(audit_dir, "ticker_mapping.csv"))
    assert os.path.exists(os.path.join(audit_dir, "coverage_summary.json"))
    assert os.path.exists(os.path.join(audit_dir, "coverage_by_week.csv"))
    assert os.path.exists(os.path.join(audit_dir, "unresolved_tickers.csv"))
    assert os.path.exists(os.path.join(audit_dir, "special_cases.csv"))
    assert os.path.exists(os.path.join(audit_dir, "source_errors.csv"))

    with open(os.path.join(audit_dir, "coverage_summary.json")) as f:
        summary = json.load(f)
        assert summary["coverage_pct"] >= 90.0
        assert summary["rows_filled"] > 11000


def test_no_self_match_contamination():
    """Self-match rows (growth=0, eps_curr==eps_prior, non-zero) must be rare.

    A small number is expected (genuine flat EPS quarters like BFST Q2 $0.70→$0.70).
    Before the P0 fix, contamination was 38.9% (4,386 rows). After fix, only genuine
    cases remain (~1%). We assert < 2% as a regression guard.
    """
    prov_path = "outputs/eps_pit_backfill/audit/weekly_eps_provenance.parquet"
    assert os.path.exists(prov_path)
    df_prov = pd.read_parquet(prov_path)

    filled = df_prov.dropna(subset=["eps_yoy_growth"])
    self_match = filled[
        (filled["eps_yoy_growth"] == 0)
        & (filled["eps_current"] == filled["eps_prior_year"])
        & (filled["eps_current"] != 0)
    ]
    rate = len(self_match) / len(filled) * 100 if len(filled) > 0 else 0
    assert rate < 2.0, (
        f"Self-match rate {rate:.2f}% exceeds 2% threshold: "
        f"{len(self_match)} rows, {self_match['code'].nunique()} tickers"
    )


def test_events_no_phantom_duplicates():
    """Same (report_period, fiscal_quarter) must not have multiple fiscal_year entries."""
    events_path = "outputs/eps_pit_backfill/cache/eps_growth_events.parquet"
    assert os.path.exists(events_path)
    df_ev = pd.read_parquet(events_path)

    sec_ev = df_ev[df_ev["source"] == "SEC"]
    if sec_ev.empty:
        return
    dup = sec_ev.groupby(["code", "report_period", "fiscal_quarter"])["fiscal_year"].nunique()
    phantoms = dup[dup > 1]
    assert len(phantoms) == 0, (
        f"Phantom comparative-period events: {len(phantoms)} groups"
    )
