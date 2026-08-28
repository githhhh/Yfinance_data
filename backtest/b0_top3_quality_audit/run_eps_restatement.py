"""EPS-only research restatement orchestrator.

Run AFTER historical pools have been recalibrated with:
    PYTHONPATH=. python backtest/ibd_skill_replay_pools/recalibrate_eps_pit.py --reset-store

This runner:
- fails if any non-EPS historical pool fact differs from --baseline-ref;
- rebuilds EPS-bearing event data only;
- reuses frozen daily price cache and candidate_weekly_outcomes byte-for-byte;
- reruns B0, Matched-N random, Three-Tier, Rank/TopK, Layer-1 and contaminated validation;
- creates V2 golden/manifest without reselecting champions or changing production rules.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from backtest.b0_top3_quality_audit.baseline import run_b0_across_all_pools
from backtest.b0_top3_quality_audit.eligibility import to_bool, to_float
from backtest.b0_top3_quality_audit.generate_b0_quality_vs_matched_random_report import (
    run_report as run_b0_vs_random_report,
)
from backtest.b0_top3_quality_audit.generate_b0_rank_topk_audit import (
    run_b0_rank_topk_audit,
)
from backtest.b0_top3_quality_audit.generate_layer1_screening_ablation_audit import (
    run_layer1_screening_ablation_audit,
)
from backtest.b0_top3_quality_audit.historical_validation_verifier import (
    run_historical_validation_unblind,
)
from backtest.b0_top3_quality_audit.metrics import (
    compute_b0_vs_random_summary,
    compute_paired_pick_comparison,
    compute_pick_level_quality,
    compute_weekly_top3_quality,
)
from backtest.b0_top3_quality_audit.random_control import run_random_top3_benchmark
from backtest.b0_top3_quality_audit.research_windows import train_dates
from backtest.b0_top3_quality_audit.three_tier_baseline import (
    generate_three_tier_report,
    run_three_tier_baseline,
)
from backtest.b0_top3_quality_audit.universe import (
    build_review_universe_events,
    scan_replay_pools,
)

LOG = logging.getLogger("eps_restatement")
ROOT = Path("backtest/b0_top3_quality_audit")
POOL_ROOT = Path("backtest/ibd_skill_replay_pools")
DATA = ROOT / "data"
OUT = ROOT / "output"
GOLDEN = ROOT / "golden"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_show_bytes(ref: str, path: Path) -> bytes:
    return subprocess.check_output(["git", "show", f"{ref}:{path.as_posix()}"])


def git_show_csv(ref: str, path: Path) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(git_show_bytes(ref, path)), encoding="utf-8-sig")


def git_show_json(ref: str, path: Path) -> dict[str, Any]:
    return json.loads(git_show_bytes(ref, path).decode("utf-8"))


def _is_eps_column(name: str) -> bool:
    return name.startswith("eps_") or name == "effective_eps_yoy_growth"


def _assert_pool_non_eps_matches_baseline(
    baseline_ref: str,
    pool_paths: list[Path],
) -> None:
    failures: list[str] = []
    for path in pool_paths:
        old = git_show_csv(baseline_ref, path)
        new = pd.read_csv(path, encoding="utf-8-sig")
        old_cols = [c for c in old.columns if not _is_eps_column(str(c))]
        new_cols = [c for c in new.columns if not _is_eps_column(str(c))]
        if old_cols != new_cols:
            failures.append(f"{path}: non-EPS columns changed")
            continue
        try:
            pd.testing.assert_frame_equal(
                old.loc[:, old_cols].reset_index(drop=True),
                new.loc[:, new_cols].reset_index(drop=True),
                check_dtype=False,
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
            )
        except AssertionError as exc:
            failures.append(f"{path}: {str(exc).splitlines()[0]}")
    if failures:
        raise RuntimeError(
            "EPS-only restatement invariant failed; technical pool facts changed:\n"
            + "\n".join(failures[:30])
        )


def _restate_candidate_event_outcomes(
    new_events: pd.DataFrame,
    existing_outcomes_path: Path,
) -> pd.DataFrame:
    old = pd.read_parquet(existing_outcomes_path)
    key = "event_id"
    if key not in old.columns or key not in new_events.columns:
        raise RuntimeError("event_id is required for EPS-only event restatement")
    old_ids = set(old[key].astype(str))
    new_ids = set(new_events[key].astype(str))
    if old_ids != new_ids:
        raise RuntimeError(
            "Review-universe event identity changed; EPS-only restatement cannot proceed "
            f"(old={len(old_ids)}, new={len(new_ids)})"
        )

    new_by_id = new_events.set_index(key)
    result = old.copy()
    eps_cols = [c for c in new_events.columns if _is_eps_column(str(c))]
    for col in eps_cols:
        mapped = result[key].astype(str).map(new_by_id[col].to_dict())
        result[col] = mapped
    return result


def _build_b0_outcomes(
    events_df: pd.DataFrame,
    b0_events_df: pd.DataFrame,
) -> pd.DataFrame:
    picks = b0_events_df.loc[:, ["snapshot_date", "code", "pick_order"]].copy()
    picks["snapshot_date"] = picks["snapshot_date"].astype(str)
    picks["code"] = picks["code"].astype(str)
    events = events_df.copy()
    events["snapshot_date"] = events["snapshot_date"].astype(str)
    events["code"] = events["code"].astype(str)
    merged = picks.merge(
        events,
        on=["snapshot_date", "code"],
        how="left",
        validate="one_to_one",
    )
    if merged["entry_status"].isna().all() and not merged.empty:
        raise RuntimeError("B0 picks failed to join frozen candidate outcomes")
    return merged.sort_values(["snapshot_date", "pick_order"]).reset_index(drop=True)


def _write_v2_golden(
    b0_events_df: pd.DataFrame,
    invariant_df: pd.DataFrame,
) -> tuple[Path, Path]:
    GOLDEN.mkdir(parents=True, exist_ok=True)
    candidate_path = GOLDEN / "b0_top3_golden_reference_eps_recalibrated_v2.csv"
    weekly_path = GOLDEN / "b0_top3_golden_weekly_reference_eps_recalibrated_v2.csv"
    b0_events_df.to_csv(candidate_path, index=False, encoding="utf-8-sig")
    weekly = invariant_df.loc[
        :,
        [
            "snapshot_date",
            "pool_sha256",
            "replay_top3_count",
            "replay_codes",
            "selection_signature_sha256",
        ],
    ].rename(
        columns={
            "replay_top3_count": "expected_pick_count",
            "replay_codes": "expected_codes",
        }
    )
    weekly.to_csv(weekly_path, index=False, encoding="utf-8-sig")
    return candidate_path, weekly_path


def _canonical_manifest_hash(payload: dict[str, Any]) -> str:
    tmp = dict(payload)
    tmp.pop("manifest_sha256", None)
    raw = json.dumps(tmp, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _write_v2_manifest(baseline_ref: str) -> Path:
    old_path = OUT / "frozen_rules_manifest.json"
    manifest = git_show_json(baseline_ref, old_path)
    repo_root = Path.cwd()
    manifest["manifest_version"] = "2.1-eps-recalibrated-v2"
    manifest["source_base_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    manifest["freeze_type"] = "DATA_REVISION_ONLY"
    manifest["data_revision"] = "EPS_RECALIBRATED_V2"
    manifest["data_revision_reason"] = (
        "EPS PIT recalibration after LIVE/REPLAY resolver separation"
    )
    manifest["rule_change"] = False
    manifest["selector_change"] = False
    manifest["methodology_change"] = False
    manifest["price_data_change"] = False
    manifest["champions_reselected"] = False
    manifest["champion_selection_baseline_ref"] = baseline_ref
    manifest["historical_champion_metrics_semantics"] = (
        "PRESERVED_FROM_OLD_EPS_BASELINE; rule identities/params frozen; "
        "not used to reselect champions"
    )
    manifest["code_fingerprints"] = {
        "production_selector_sha256": sha256_file(
            repo_root / "dashboard/skill_industry_eps_known.py"
        ),
        "eligibility_predicate_sha256": sha256_file(
            ROOT / "eligibility.py"
        ),
        "skill_rule_engine_sha256": sha256_file(
            ROOT / "skill_rule_engine.py"
        ),
        "three_tier_baseline_sha256": sha256_file(
            ROOT / "three_tier_baseline.py"
        ),
        "evaluate_rule_signatures_sha256": sha256_file(
            ROOT / "evaluate_rule_signatures.py"
        ),
    }
    manifest["data_fingerprints"] = {
        "train_candidate_events_parquet_sha256": sha256_file(
            DATA / "frozen/train_candidate_event_outcomes.parquet"
        ),
        "train_candidate_weekly_outcomes_parquet_sha256": sha256_file(
            DATA / "frozen/train_candidate_weekly_outcomes.parquet"
        ),
    }
    manifest["manifest_sha256"] = _canonical_manifest_hash(manifest)
    path = OUT / "frozen_rules_manifest_eps_recalibrated_v2.json"
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def _explicit_eligible(row: pd.Series, eps: float | None) -> bool:
    if to_bool(row.get("signal")) is not True:
        return False
    if not str(row.get("ibd_candidate_rule", "") or "").strip():
        return False
    if str(row.get("ibd_entry_status", "") or "").strip().upper() != "ACTIONABLE":
        return False
    pos = to_float(row.get("ibd_entry_close_position"))
    rr = to_float(row.get("ibd_entry_breakout_range_ratio"))
    if rr is not None and rr <= 0:
        return False
    if pos is not None and pos < 0.65:
        return False
    cur = to_float(row.get("current_vs_ibd_candidate_pct"))
    if cur is None or cur < 0:
        return False
    if eps is None:
        return False
    if not str(row.get("industry", "") or "").strip():
        return False
    return True


def _augment_eps_impact(
    baseline_ref: str,
    new_events: pd.DataFrame,
    b0_events_df: pd.DataFrame,
) -> dict[str, Any]:
    impact_path = POOL_ROOT / "EPS_PIT_RECALIBRATION_IMPACT.csv"
    if not impact_path.exists():
        raise RuntimeError(
            "EPS_PIT_RECALIBRATION_IMPACT.csv missing; run recalibrate_eps_pit.py first"
        )
    impact = pd.read_csv(impact_path, dtype={"code": str})
    old_pools = []
    for path in scan_replay_pools(POOL_ROOT):
        old = git_show_csv(baseline_ref, path)
        old["snapshot_date"] = path.parent.name
        old["code"] = old["code"].astype(str)
        old_pools.append(old)
    old_all = pd.concat(old_pools, ignore_index=True)

    old_lookup = {
        (str(r["snapshot_date"]), str(r["code"])): r
        for _, r in old_all.iterrows()
    }
    new_lookup = {
        (str(r["snapshot_date"]), str(r["code"])): r
        for _, r in new_events.iterrows()
    }

    old_b0 = git_show_csv(baseline_ref, OUT / "b0_selection_events.csv")
    old_pick = {
        (str(r["snapshot_date"]), str(r["code"])): int(r["pick_order"])
        for _, r in old_b0.iterrows()
    }
    new_pick = {
        (str(r["snapshot_date"]), str(r["code"])): int(r["pick_order"])
        for _, r in b0_events_df.iterrows()
    }

    old_e0 = []
    new_e0 = []
    for _, row in impact.iterrows():
        key = (str(row["snapshot_date"]), str(row["code"]))
        old_row = old_lookup.get(key)
        new_row = new_lookup.get(key)
        old_eps = to_float(row.get("old_eps"))
        new_eps = to_float(row.get("new_eps"))
        old_e0.append(bool(old_row is not None and _explicit_eligible(old_row, old_eps)))
        new_e0.append(bool(new_row is not None and _explicit_eligible(new_row, new_eps)))

    impact["old_e0_eligible"] = old_e0
    impact["new_e0_eligible"] = new_e0
    impact["old_b0_selected"] = [
        (str(r.snapshot_date), str(r.code)) in old_pick for r in impact.itertuples()
    ]
    impact["new_b0_selected"] = [
        (str(r.snapshot_date), str(r.code)) in new_pick for r in impact.itertuples()
    ]
    impact["old_pick_order"] = [
        old_pick.get((str(r.snapshot_date), str(r.code))) for r in impact.itertuples()
    ]
    impact["new_pick_order"] = [
        new_pick.get((str(r.snapshot_date), str(r.code))) for r in impact.itertuples()
    ]
    impact.to_csv(impact_path, index=False, encoding="utf-8-sig")

    e0_changed = impact["old_e0_eligible"] != impact["new_e0_eligible"]
    old_selected_count = old_b0.groupby("snapshot_date").size().to_dict()
    new_selected_count = b0_events_df.groupby("snapshot_date").size().to_dict()
    all_weeks = sorted(set(old_selected_count) | set(new_selected_count))
    count_changed_weeks = [
        w for w in all_weeks
        if int(old_selected_count.get(w, 0)) != int(new_selected_count.get(w, 0))
    ]
    old_codes = (
        old_b0.sort_values(["snapshot_date", "pick_order"])
        .groupby("snapshot_date")["code"].apply(list).to_dict()
    )
    new_codes = (
        b0_events_df.sort_values(["snapshot_date", "pick_order"])
        .groupby("snapshot_date")["code"].apply(list).to_dict()
    )
    code_changed_weeks = [
        w for w in all_weeks if old_codes.get(w, []) != new_codes.get(w, [])
    ]
    order_only_weeks = [
        w for w in code_changed_weeks
        if set(old_codes.get(w, [])) == set(new_codes.get(w, []))
        and len(old_codes.get(w, [])) == len(new_codes.get(w, []))
    ]
    return {
        "e0_membership_changed_count": int(e0_changed.sum()),
        "e0_affected_weeks": int(
            impact.loc[e0_changed, "snapshot_date"].astype(str).nunique()
        ),
        "b0_selected_count_changed_weeks": len(count_changed_weeks),
        "b0_codes_changed_weeks": len(code_changed_weeks),
        "b0_order_only_changed_weeks": len(order_only_weeks),
    }


def _write_restatement_report(
    baseline_ref: str,
    impact_summary: dict[str, Any],
    frozen_hashes: dict[str, str],
) -> Path:
    lines = [
        "# EPS Recalibration Research Restatement",
        "",
        f"- Old EPS baseline ref: {baseline_ref}",
        "- Data revision: EPS_RECALIBRATED_V2",
        "- Rule change: NO",
        "- Production selector change: NO",
        "- Price data change: NO",
        "- Champion reselection: NO",
        "",
        "## Candidate / B0 impact",
        "",
        f"- E0 membership changed candidates: {impact_summary['e0_membership_changed_count']}",
        f"- E0 affected weeks: {impact_summary['e0_affected_weeks']}",
        f"- B0 selected-count changed weeks: {impact_summary['b0_selected_count_changed_weeks']}",
        f"- B0 code-set/order changed weeks: {impact_summary['b0_codes_changed_weeks']}",
        f"- B0 order-only changed weeks: {impact_summary['b0_order_only_changed_weeks']}",
        "",
        "## Frozen outcome invariants",
        "",
        f"- signal_daily_prices.parquet SHA256: {frozen_hashes['price']}",
        f"- candidate_weekly_outcomes.parquet SHA256: {frozen_hashes['weekly']}",
        f"- train_candidate_weekly_outcomes.parquet SHA256: {frozen_hashes['train_weekly']}",
        "",
        "## Regenerated fixed research outputs",
        "",
        "- B0 vs Matched-N Random",
        "- Three-Tier decomposition",
        "- Rank1/Rank2/Rank3 + TopK",
        "- Layer-1 eligibility / industry / ranking decomposition",
        "- Fixed-date contaminated historical validation",
        "",
        "Interpretation must compare the regenerated CSVs against the old baseline "
        "and label each prior conclusion RETAINED / WEAKENED / STRENGTHENED / "
        "REVERSED / NO_LONGER_IDENTIFIABLE. This runner intentionally does not "
        "search new rules or select new champions.",
    ]
    path = OUT / "EPS_RECALIBRATION_RESEARCH_RESTATEMENT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run(*, baseline_ref: str, n_draws: int = 1000, seed: int = 42) -> dict[str, Any]:
    pool_paths = scan_replay_pools(POOL_ROOT)
    if not pool_paths:
        raise RuntimeError("No replay pools found")
    _assert_pool_non_eps_matches_baseline(baseline_ref, pool_paths)

    price_path = DATA / "signal_daily_prices.parquet"
    weekly_path = DATA / "candidate_weekly_outcomes.parquet"
    train_weekly_path = DATA / "frozen/train_candidate_weekly_outcomes.parquet"
    selector_path = Path("dashboard/skill_industry_eps_known.py")
    frozen_before = {
        "price": sha256_file(price_path),
        "weekly": sha256_file(weekly_path),
        "train_weekly": sha256_file(train_weekly_path),
        "selector": sha256_file(selector_path),
    }

    # Rebuild PIT candidate facts only.
    events_path = DATA / "review_universe_events.parquet"
    new_events = build_review_universe_events(pool_paths=pool_paths, output_path=events_path)
    candidate_path = DATA / "candidate_event_outcomes.parquet"
    restated_events = _restate_candidate_event_outcomes(new_events, candidate_path)
    restated_events.to_parquet(candidate_path, index=False)

    train_set = train_dates(sorted(restated_events["snapshot_date"].astype(str).unique()))
    train_events = restated_events[
        restated_events["snapshot_date"].astype(str).isin(train_set)
    ].copy()
    train_events.to_parquet(
        DATA / "frozen/train_candidate_event_outcomes.parquet",
        index=False,
    )

    # Re-run selection against corrected EPS. Old golden is comparison-only.
    b0_events_df, invariant_df = run_b0_across_all_pools(
        pool_paths=pool_paths,
        output_events_csv=OUT / "b0_selection_events.csv",
        output_invariant_csv=OUT / "b0_production_invariant_audit.csv",
    )
    invariant_df["reference_semantics"] = "OLD_EPS_BASELINE_EXPECTED_TO_DRIFT"
    invariant_df.to_csv(
        OUT / "b0_production_invariant_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _write_v2_golden(b0_events_df, invariant_df)

    weekly_df = pd.read_parquet(weekly_path)
    b0_outcomes = _build_b0_outcomes(restated_events, b0_events_df)
    b0_outcomes.to_csv(
        OUT / "b0_path_quality_to_asof.csv", index=False, encoding="utf-8-sig"
    )
    pick_quality = compute_pick_level_quality(
        b0_outcomes, output_csv=OUT / "b0_pick_quality.csv"
    )
    paired = compute_paired_pick_comparison(
        b0_outcomes, output_csv=OUT / "b0_paired_pick_comparison.csv"
    )
    weekly_quality = compute_weekly_top3_quality(
        b0_outcomes,
        weekly_outcomes_df=weekly_df,
        output_csv=OUT / "b0_weekly_top3_quality.csv",
    )
    random_df = run_random_top3_benchmark(
        event_outcomes_df=restated_events,
        weekly_outcomes_df=weekly_df,
        b0_events_df=b0_outcomes,
        n_draws_per_week=n_draws,
        seed=seed,
        benchmark_mode="MATCHED_N",
        output_distribution_csv=OUT / "random_signal_top3_distribution.csv",
    )
    compute_b0_vs_random_summary(
        weekly_quality,
        random_df,
        output_csv=OUT / "b0_vs_random_summary.csv",
    )

    three_weekly, three_summary, three_meta = run_three_tier_baseline(
        restated_events,
        weekly_df,
        n_draws=n_draws,
        seed=seed,
    )
    three_weekly.to_csv(OUT / "three_tier_weekly_comparison.csv", index=False)
    three_summary.to_csv(OUT / "three_tier_alpha_summary.csv", index=False)
    generate_three_tier_report(
        three_weekly,
        three_summary,
        three_meta,
        OUT / "three_tier_alpha_report.md",
    )

    run_b0_vs_random_report()
    run_b0_rank_topk_audit()
    run_layer1_screening_ablation_audit(n_draws=n_draws, base_seed=seed)

    manifest_path = _write_v2_manifest(baseline_ref)
    run_historical_validation_unblind(
        manifest_path,
        restated_events,
        weekly_df,
        three_weekly,
        OUT,
    )

    impact_summary = _augment_eps_impact(baseline_ref, new_events, b0_events_df)
    summary_path = POOL_ROOT / "EPS_PIT_RECALIBRATION_SUMMARY.json"
    eps_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    eps_summary.update(impact_summary)
    summary_path.write_text(
        json.dumps(eps_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    frozen_after = {
        "price": sha256_file(price_path),
        "weekly": sha256_file(weekly_path),
        "train_weekly": sha256_file(train_weekly_path),
        "selector": sha256_file(selector_path),
    }
    if frozen_before != frozen_after:
        raise RuntimeError(
            f"Frozen data/selector changed during EPS restatement: "
            f"before={frozen_before}, after={frozen_after}"
        )

    report_path = _write_restatement_report(
        baseline_ref,
        impact_summary,
        {
            "price": frozen_after["price"],
            "weekly": frozen_after["weekly"],
            "train_weekly": frozen_after["train_weekly"],
        },
    )
    return {
        "baseline_ref": baseline_ref,
        "b0_picks": int(len(b0_events_df)),
        "e0_membership_changed_count": impact_summary["e0_membership_changed_count"],
        "b0_codes_changed_weeks": impact_summary["b0_codes_changed_weeks"],
        "v2_manifest": str(manifest_path),
        "restatement_report": str(report_path),
        "production_selector_sha256": frozen_after["selector"],
        "price_sha256": frozen_after["price"],
        "weekly_outcomes_sha256": frozen_after["weekly"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-ref",
        required=True,
        help="Git ref containing the old-EPS baseline pools and research outputs.",
    )
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run(
        baseline_ref=args.baseline_ref,
        n_draws=args.draws,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
