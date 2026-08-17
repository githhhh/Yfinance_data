from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from backtest.ibd_skill_iteration.reports import (
    build_reasoning_pick_metric_rows,
    find_quality_pullback_candidates,
)
from backtest.ibd_skill_oracle.core import (
    oracle_rows,
    rank_path_adjusted_oracle,
    summarize_group,
)
from backtest.ibd_skill_oracle.run_oracle_replay import _build_candidates, _candidate_metric_rows
from backtest.ibd_skill_replay.core import to_bool, to_float
from backtest.ibd_skill_replay.run_ytd_replay import _load_price_cache, _load_supplemental_prices


AUDIT_FEATURE_COLUMNS = [
    "ibd_entry_status",
    "ibd_candidate_price",
    "latest_close",
    "current_vs_ibd_candidate_pct",
    "ibd_entry_volume_ratio",
    "ibd_entry_close_position",
    "ibd_entry_breakout_range_ratio",
    "ibd_entry_close_vs_trigger_pct",
    "dist_to_52w_high_pct",
    "volume_ratio",
    "eps_yoy_growth",
    "base_depth_pct",
    "base_duration_weeks",
    "pullback_pct",
    "pullback_duration_weeks",
    "pullback_v_is_dry",
    "sector",
    "industry",
]

PRIMARY_SKILLS = [
    "reasoning_v2_priority_top3",
    "reasoning_v2_alpha_radar_top5",
    "reasoning_v2_pullback_radar_top5",
    "reasoning_v2_non_actionable_alpha_radar_top10",
    "reasoning_v3_priority_top3",
    "reasoning_v3_alpha_radar_top5",
    "reasoning_v3_pullback_radar_top5",
    "reasoning_v3_non_actionable_alpha_radar_top10",
    "reasoning_v3_pullback_scout_top10",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit optimized IBD reasoning skill over every repaired pool file.")
    parser.add_argument("--pools-dir", default="backtest/ibd_skill_replay_pools")
    parser.add_argument("--price-cache", default="results_pkl/stock_data_150826_1d.pkl")
    parser.add_argument("--supplemental-price-csv", default="backtest/ibd_skill_replay_audit/supplemental_price_bars.csv")
    parser.add_argument("--end", default="2026-08-14")
    parser.add_argument("--output-root", default="backtest")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    report_dir = output_root / "ibd_skill_full_repaired_audit_reports"
    audit_dir = output_root / "ibd_skill_full_repaired_audit"
    for directory in (report_dir, audit_dir):
        directory.mkdir(parents=True, exist_ok=True)

    prices = _load_price_cache(Path(args.price_cache))
    prices.update(_load_supplemental_prices(Path(args.supplemental_price_csv)))

    result = build_full_repaired_pool_audit(
        pools_dir=Path(args.pools_dir),
        prices=prices,
        end_date=args.end,
    )

    _write_outputs(result, report_dir=report_dir, audit_dir=audit_dir)
    manifest = {
        "pools_dir": args.pools_dir,
        "end": args.end,
        "pool_files": int(len(result["pool_scope"])),
        "outputs": {
            "pool_scope": str(audit_dir / "pool_scope.csv"),
            "pick_metrics": str(report_dir / "full_repaired_pool_pick_metrics.csv"),
            "skill_summary": str(report_dir / "full_repaired_pool_skill_summary.csv"),
            "review_hit_matrix": str(report_dir / "full_repaired_pool_review_hit_matrix.csv"),
            "review_hit_summary": str(report_dir / "full_repaired_pool_review_hit_summary.csv"),
            "actionable_hit_matrix": str(report_dir / "full_repaired_pool_actionable_hit_matrix.csv"),
            "actionable_hit_summary": str(report_dir / "full_repaired_pool_actionable_hit_summary.csv"),
            "review_oracle_top5": str(report_dir / "full_repaired_pool_review_oracle_top5.csv"),
            "actionable_oracle_top5": str(report_dir / "full_repaired_pool_actionable_oracle_top5.csv"),
            "non_actionable_hit_summary": str(report_dir / "full_repaired_pool_non_actionable_hit_summary.csv"),
            "quality_pullback_candidates": str(audit_dir / "quality_pullback_candidates.csv"),
            "quality_pullback_coverage": str(audit_dir / "quality_pullback_coverage.csv"),
            "aapl_audit": str(audit_dir / "aapl_audit.csv"),
            "summary": str(report_dir / "full_repaired_pool_audit_summary.md"),
        },
    }
    (report_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def build_full_repaired_pool_audit(
    *,
    pools_dir: Path,
    prices: dict[str, pd.DataFrame],
    end_date: str,
) -> dict[str, pd.DataFrame]:
    scope_rows = []
    review_oracle_rows = []
    actionable_oracle_rows = []
    review_metric_rows = []
    reasoning_rows = []
    aapl_rows = []

    for path in discover_repaired_pool_files(pools_dir):
        snapshot_date, commit, pool_id = pool_id_from_path(path)
        raw_pool = pd.read_csv(path, encoding="utf-8-sig")
        pool = repair_iteration_pool_schema(raw_pool)
        scope_rows.append(_scope_row(raw_pool, pool, path=path, pool_id=pool_id, snapshot_date=snapshot_date, commit=commit))
        aapl_rows.extend(_aapl_rows(pool, path=path, pool_id=pool_id, snapshot_date=snapshot_date, commit=commit, prices=prices, end_date=end_date))

        actionable_candidates, _ = _build_candidates(pool, prices, snapshot_date=snapshot_date, end_date=end_date)
        review_candidates, _ = _build_candidates(
            pool,
            prices,
            snapshot_date=snapshot_date,
            end_date=end_date,
            universe="review",
        )
        actionable_ranked = rank_path_adjusted_oracle(actionable_candidates)
        review_ranked = rank_path_adjusted_oracle(review_candidates)
        actionable_oracle_rows.extend(
            _tag_rows(oracle_rows(actionable_ranked, ranking_method="actionable_path_adjusted", limit=5), path=path, pool_id=pool_id, commit=commit)
        )
        review_oracle_rows.extend(
            _tag_rows(oracle_rows(review_ranked, ranking_method="review_path_adjusted", limit=5), path=path, pool_id=pool_id, commit=commit)
        )
        review_metric_rows.extend(
            _tag_rows(_candidate_metric_rows(review_candidates, include_entry_status=True), path=path, pool_id=pool_id, commit=commit)
        )
        for version in ["v1", "v2", "v3"]:
            rows = build_reasoning_pick_metric_rows(
                pool,
                prices,
                snapshot_date=snapshot_date,
                end_date=end_date,
                version=version,
            )
            reasoning_rows.extend(_tag_rows(rows, path=path, pool_id=pool_id, commit=commit))

    pool_scope = pd.DataFrame(scope_rows)
    review_oracle = pd.DataFrame(review_oracle_rows)
    actionable_oracle = pd.DataFrame(actionable_oracle_rows)
    review_metrics = pd.DataFrame(review_metric_rows)
    pick_metrics = pd.DataFrame(reasoning_rows)
    skill_summary = summarize_group(pick_metrics, group_cols=["skill"]) if not pick_metrics.empty else pd.DataFrame()
    review_hit_matrix = build_pool_hit_matrix(review_oracle, pick_metrics)
    review_hit_summary = summarize_hit_matrix(review_hit_matrix)
    actionable_hit_matrix = build_pool_hit_matrix(actionable_oracle, pick_metrics)
    actionable_hit_summary = summarize_hit_matrix(actionable_hit_matrix)
    non_actionable_hit_summary = build_pool_non_actionable_hit_summary(pick_metrics, review_oracle)
    quality_pullbacks = find_quality_pullback_candidates(review_metrics, limit=100) if not review_metrics.empty else pd.DataFrame()
    quality_pullback_coverage = build_quality_pullback_coverage(quality_pullbacks, pick_metrics)
    aapl_audit = pd.DataFrame(aapl_rows)

    return {
        "pool_scope": pool_scope,
        "review_oracle": review_oracle,
        "actionable_oracle": actionable_oracle,
        "review_metrics": review_metrics,
        "pick_metrics": pick_metrics,
        "skill_summary": skill_summary,
        "review_hit_matrix": review_hit_matrix,
        "review_hit_summary": review_hit_summary,
        "actionable_hit_matrix": actionable_hit_matrix,
        "actionable_hit_summary": actionable_hit_summary,
        "non_actionable_hit_summary": non_actionable_hit_summary,
        "quality_pullbacks": quality_pullbacks,
        "quality_pullback_coverage": quality_pullback_coverage,
        "aapl_audit": aapl_audit,
    }


def discover_repaired_pool_files(pools_dir: Path) -> list[Path]:
    return sorted(pools_dir.glob("*_pool.csv"), key=lambda path: pool_id_from_path(path))


def pool_id_from_path(path: str | Path) -> tuple[str, str, str]:
    name = Path(path).name
    match = re.fullmatch(r"(\d{4}-\d{2}-\d{2})_([^_]+)_pool\.csv", name)
    if not match:
        raise ValueError(f"Unexpected repaired pool filename: {name}")
    snapshot_date, commit = match.groups()
    return snapshot_date, commit, f"{snapshot_date}_{commit}"


def repair_iteration_pool_schema(pool: pd.DataFrame) -> pd.DataFrame:
    repaired = pool.copy()
    for column in AUDIT_FEATURE_COLUMNS:
        if column not in repaired.columns:
            repaired[column] = pd.NA

    if "ibd_entry_status_repair_method" not in repaired.columns:
        repaired["ibd_entry_status_repair_method"] = "original"
    if "eps_yoy_growth_repair_method" not in repaired.columns:
        repaired["eps_yoy_growth_repair_method"] = "original"

    if "eps_yoy_growth" in repaired.columns:
        eps_missing = repaired["eps_yoy_growth"].isna()
        repaired.loc[eps_missing, "eps_yoy_growth_repair_method"] = "missing_in_repaired_pool"

    for idx, row in repaired.iterrows():
        raw_status = row.get("ibd_entry_status", "")
        status = "" if pd.isna(raw_status) else str(raw_status).strip().upper()
        if status:
            repaired.at[idx, "ibd_entry_status"] = status
            repaired.at[idx, "ibd_entry_status_repair_method"] = "original"
            continue
        inferred = _infer_entry_status(row)
        repaired.at[idx, "ibd_entry_status"] = inferred
        repaired.at[idx, "ibd_entry_status_repair_method"] = _status_repair_method(row, inferred)
    return repaired


def build_pool_hit_matrix(oracle: pd.DataFrame, skill: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "pool_id",
        "snapshot_date",
        "commit",
        "skill",
        "skill_picks",
        "oracle_top3_hits",
        "oracle_top3_hit_codes",
        "oracle_top5_hits",
        "oracle_top5_hit_codes",
        "skill_top3_codes",
        "skill_top5_codes",
        "oracle_top3_codes",
        "oracle_top5_codes",
    ]
    if oracle.empty or skill.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for (pool_id, skill_name), picks in skill.groupby(["pool_id", "skill"], sort=True):
        oracle_pool = oracle[oracle["pool_id"].eq(pool_id)]
        oracle_top3 = set(oracle_pool[oracle_pool["oracle_rank"].le(3)]["code"].astype(str))
        oracle_top5 = set(oracle_pool[oracle_pool["oracle_rank"].le(5)]["code"].astype(str))
        skill_top3_rows = picks[picks["pick_order"].le(3)]
        skill_top5_rows = picks[picks["pick_order"].le(5)]
        skill_top3 = set(skill_top3_rows["code"].astype(str))
        skill_top5 = set(skill_top5_rows["code"].astype(str))
        first = picks.iloc[0]
        rows.append(
            {
                "pool_id": pool_id,
                "snapshot_date": first["snapshot_date"],
                "commit": first["commit"],
                "skill": skill_name,
                "skill_picks": len(picks),
                "oracle_top3_hits": len(skill_top3 & oracle_top3),
                "oracle_top3_hit_codes": ",".join(sorted(skill_top3 & oracle_top3)),
                "oracle_top5_hits": len(skill_top5 & oracle_top5),
                "oracle_top5_hit_codes": ",".join(sorted(skill_top5 & oracle_top5)),
                "skill_top3_codes": ",".join(skill_top3_rows["code"].astype(str)),
                "skill_top5_codes": ",".join(skill_top5_rows["code"].astype(str)),
                "oracle_top3_codes": ",".join(oracle_pool[oracle_pool["oracle_rank"].le(3)]["code"].astype(str)),
                "oracle_top5_codes": ",".join(oracle_pool[oracle_pool["oracle_rank"].le(5)]["code"].astype(str)),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def summarize_hit_matrix(hit_matrix: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "skill",
        "pools",
        "skill_picks",
        "oracle_top3_hits",
        "oracle_top5_hits",
        "pools_with_top3_hit",
        "pools_with_top5_hit",
        "top3_hits_per_pool",
        "top5_hits_per_pool",
    ]
    if hit_matrix.empty:
        return pd.DataFrame(columns=columns)
    grouped = hit_matrix.groupby("skill", sort=True)
    frame = grouped.agg(
        pools=("pool_id", "nunique"),
        skill_picks=("skill_picks", "sum"),
        oracle_top3_hits=("oracle_top3_hits", "sum"),
        oracle_top5_hits=("oracle_top5_hits", "sum"),
        pools_with_top3_hit=("oracle_top3_hits", lambda series: int((series > 0).sum())),
        pools_with_top5_hit=("oracle_top5_hits", lambda series: int((series > 0).sum())),
    ).reset_index()
    frame["top3_hits_per_pool"] = frame["oracle_top3_hits"] / frame["pools"]
    frame["top5_hits_per_pool"] = frame["oracle_top5_hits"] / frame["pools"]
    return frame[columns]


def build_pool_non_actionable_hit_summary(skill_metrics: pd.DataFrame, review_oracle: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "skill",
        "entry_status",
        "picks",
        "review_oracle_top3_hits",
        "review_oracle_top5_hits",
        "review_oracle_top3_hit_rate",
        "review_oracle_top5_hit_rate",
        "review_oracle_top3_hit_codes",
        "review_oracle_top5_hit_codes",
    ]
    if skill_metrics.empty:
        return pd.DataFrame(columns=columns)
    non_actionable = skill_metrics[skill_metrics["entry_status"].astype(str).ne("ACTIONABLE")].copy()
    if non_actionable.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for (skill, status), group in non_actionable.groupby(["skill", "entry_status"], sort=True):
        top3_hits = []
        top5_hits = []
        for _, row in group.iterrows():
            oracle = review_oracle[review_oracle["pool_id"].eq(row["pool_id"])]
            top3 = set(oracle[oracle["oracle_rank"].le(3)]["code"].astype(str))
            top5 = set(oracle[oracle["oracle_rank"].le(5)]["code"].astype(str))
            code = str(row["code"])
            if code in top3:
                top3_hits.append(code)
            if code in top5:
                top5_hits.append(code)
        picks = len(group)
        rows.append(
            {
                "skill": skill,
                "entry_status": status,
                "picks": picks,
                "review_oracle_top3_hits": len(top3_hits),
                "review_oracle_top5_hits": len(top5_hits),
                "review_oracle_top3_hit_rate": len(top3_hits) / picks if picks else 0.0,
                "review_oracle_top5_hit_rate": len(top5_hits) / picks if picks else 0.0,
                "review_oracle_top3_hit_codes": ",".join(top3_hits),
                "review_oracle_top5_hit_codes": ",".join(top5_hits),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_quality_pullback_coverage(pullbacks: pd.DataFrame, skill_metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "pool_id",
        "snapshot_date",
        "commit",
        "code",
        "skill",
        "covered",
        "pick_order",
        "latest_close_return_pct",
        "max_gain_pct",
        "max_drawdown_pct",
        "entry_status",
        "reason_codes",
        "risk_codes",
    ]
    if pullbacks.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for _, winner in pullbacks.iterrows():
        subset = skill_metrics[
            skill_metrics["pool_id"].eq(winner["pool_id"])
            & skill_metrics["code"].astype(str).eq(str(winner["code"]))
        ]
        if subset.empty:
            rows.append(
                {
                    "pool_id": winner["pool_id"],
                    "snapshot_date": winner["snapshot_date"],
                    "commit": winner["commit"],
                    "code": winner["code"],
                    "skill": "none",
                    "covered": False,
                    "pick_order": "",
                    "latest_close_return_pct": winner["latest_close_return_pct"],
                    "max_gain_pct": winner["max_gain_pct"],
                    "max_drawdown_pct": winner["max_drawdown_pct"],
                    "entry_status": winner.get("ibd_entry_status", ""),
                    "reason_codes": "",
                    "risk_codes": "",
                }
            )
            continue
        for _, row in subset.iterrows():
            rows.append(
                {
                    "pool_id": winner["pool_id"],
                    "snapshot_date": winner["snapshot_date"],
                    "commit": winner["commit"],
                    "code": winner["code"],
                    "skill": row["skill"],
                    "covered": True,
                    "pick_order": row["pick_order"],
                    "latest_close_return_pct": winner["latest_close_return_pct"],
                    "max_gain_pct": winner["max_gain_pct"],
                    "max_drawdown_pct": winner["max_drawdown_pct"],
                    "entry_status": row.get("entry_status", ""),
                    "reason_codes": row.get("reason_codes", ""),
                    "risk_codes": row.get("risk_codes", ""),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _write_outputs(result: dict[str, pd.DataFrame], *, report_dir: Path, audit_dir: Path) -> None:
    result["pool_scope"].to_csv(audit_dir / "pool_scope.csv", index=False)
    result["review_metrics"].to_csv(audit_dir / "review_universe_path_metrics.csv", index=False)
    result["quality_pullbacks"].to_csv(audit_dir / "quality_pullback_candidates.csv", index=False)
    result["quality_pullback_coverage"].to_csv(audit_dir / "quality_pullback_coverage.csv", index=False)
    result["aapl_audit"].to_csv(audit_dir / "aapl_audit.csv", index=False)
    result["pick_metrics"].to_csv(report_dir / "full_repaired_pool_pick_metrics.csv", index=False)
    result["skill_summary"].to_csv(report_dir / "full_repaired_pool_skill_summary.csv", index=False)
    result["review_hit_matrix"].to_csv(report_dir / "full_repaired_pool_review_hit_matrix.csv", index=False)
    result["review_hit_summary"].to_csv(report_dir / "full_repaired_pool_review_hit_summary.csv", index=False)
    result["actionable_hit_matrix"].to_csv(report_dir / "full_repaired_pool_actionable_hit_matrix.csv", index=False)
    result["actionable_hit_summary"].to_csv(report_dir / "full_repaired_pool_actionable_hit_summary.csv", index=False)
    result["review_oracle"].to_csv(report_dir / "full_repaired_pool_review_oracle_top5.csv", index=False)
    result["actionable_oracle"].to_csv(report_dir / "full_repaired_pool_actionable_oracle_top5.csv", index=False)
    result["non_actionable_hit_summary"].to_csv(
        report_dir / "full_repaired_pool_non_actionable_hit_summary.csv",
        index=False,
    )
    (report_dir / "full_repaired_pool_audit_summary.md").write_text(_render_summary(result), encoding="utf-8")


def _render_summary(result: dict[str, pd.DataFrame]) -> str:
    scope = result["pool_scope"]
    skill_summary = result["skill_summary"]
    review_hit_summary = result["review_hit_summary"]
    actionable_hit_summary = result["actionable_hit_summary"]
    non_actionable = result["non_actionable_hit_summary"]
    pullback_coverage = result["quality_pullback_coverage"]
    aapl = result["aapl_audit"]
    primary_quality = skill_summary[skill_summary["skill"].isin(PRIMARY_SKILLS)].copy() if not skill_summary.empty else pd.DataFrame()
    primary_review_hits = (
        review_hit_summary[review_hit_summary["skill"].isin(PRIMARY_SKILLS)].copy()
        if not review_hit_summary.empty
        else pd.DataFrame()
    )
    primary_actionable_hits = (
        actionable_hit_summary[actionable_hit_summary["skill"].isin(PRIMARY_SKILLS)].copy()
        if not actionable_hit_summary.empty
        else pd.DataFrame()
    )
    priority = _single_row(primary_quality, "reasoning_v2_priority_top3")
    priority_actionable = _single_row(primary_actionable_hits, "reasoning_v2_priority_top3")
    non_actionable_v2 = _single_row(primary_quality, "reasoning_v2_non_actionable_alpha_radar_top10")
    non_actionable_review = _single_row(primary_review_hits, "reasoning_v2_non_actionable_alpha_radar_top10")
    pullback = _single_row(primary_quality, "reasoning_v2_pullback_radar_top5")
    pullback_review = _single_row(primary_review_hits, "reasoning_v2_pullback_radar_top5")
    v3_pullback_scout = _single_row(primary_quality, "reasoning_v3_pullback_scout_top10")
    v3_pullback_scout_review = _single_row(primary_review_hits, "reasoning_v3_pullback_scout_top10")

    lines = [
        "# IBD Skill Full Repaired-Pool Audit",
        "",
        "## Core Conclusion",
        "",
        "This audit evaluates the v2 main reasoning skill plus the v3 non-ACTIONABLE pullback scout over every repaired historical pool file, including older non-Friday and non-comparable-schema snapshots. Each pool file is kept as its own audit unit, so duplicate snapshot dates with different commits are not averaged together or silently collapsed.",
        "",
        f"- Repaired pool files audited: `{len(scope)}`",
        f"- Distinct snapshot dates: `{scope['snapshot_date'].nunique() if not scope.empty else 0}`",
        f"- Files with inferred review status: `{int((scope['inferred_review_status_rows'] > 0).sum()) if not scope.empty else 0}`",
    ]
    if priority is not None and priority_actionable is not None:
        lines.append(
            f"- Immediate ACTIONABLE review quality is the strongest part of v2: `reasoning_v2_priority_top3` median latest `{_fmt(priority['median_latest_return_pct'])}`, worst latest `{_fmt(priority['worst_latest_return_pct'])}`, positive picks `{int(priority['positive_count'])}/{int(priority['picks'])}`, and ACTIONABLE oracle Top3/Top5 hits `{int(priority_actionable['oracle_top3_hits'])}/{int(priority_actionable['oracle_top5_hits'])}` across `{int(priority_actionable['pools'])}` pools."
        )
    if non_actionable_v2 is not None and non_actionable_review is not None:
        lines.append(
            f"- Review Universe winners are mainly found by the non-ACTIONABLE discovery layer: `reasoning_v2_non_actionable_alpha_radar_top10` median max gain `{_fmt(non_actionable_v2['median_max_gain_pct'])}`, Review oracle Top3/Top5 hits `{int(non_actionable_review['oracle_top3_hits'])}/{int(non_actionable_review['oracle_top5_hits'])}`, but stops `{int(non_actionable_v2['stop_8pct_count'])}` and worst latest `{_fmt(non_actionable_v2['worst_latest_return_pct'])}` keep it radar-only."
        )
    if pullback is not None and pullback_review is not None:
        lines.append(
            f"- Pullback discovery improves but remains the biggest gap: `reasoning_v2_pullback_radar_top5` median latest `{_fmt(pullback['median_latest_return_pct'])}` and Review oracle Top3/Top5 hits `{int(pullback_review['oracle_top3_hits'])}/{int(pullback_review['oracle_top5_hits'])}`, while many high-quality pullbacks remain uncovered below."
        )
    if v3_pullback_scout is not None and v3_pullback_scout_review is not None:
        lines.append(
            f"- v3 adds a pullback scout layer instead of upgrading non-ACTIONABLE rows into buy candidates: `reasoning_v3_pullback_scout_top10` median latest `{_fmt(v3_pullback_scout['median_latest_return_pct'])}`, Review oracle Top3/Top5 hits `{int(v3_pullback_scout_review['oracle_top3_hits'])}/{int(v3_pullback_scout_review['oracle_top5_hits'])}`, stops `{int(v3_pullback_scout['stop_8pct_count'])}`."
        )
        lines.append(
            "- Recommended operating mode: keep `reasoning_v2_priority_top3` as the first-recommendation engine; keep v2 non-ACTIONABLE Alpha Radar for broad Review Universe discovery; add v3 Pullback Scout as a labeled watch-only layer for AAPL-like unfinished pullbacks."
        )
    lines.extend(["", "## Optimized Skill Quality", ""])
    lines.extend(_markdown_table(primary_quality))
    lines.extend(["", "## Optimized Skill vs ACTIONABLE Oracle", ""])
    lines.extend(_markdown_table(primary_actionable_hits))
    lines.extend(["", "## Optimized Skill vs Review Universe Oracle", ""])
    lines.extend(_markdown_table(primary_review_hits))
    lines.extend(["", "## Non-ACTIONABLE Radar Hit Summary", ""])
    lines.extend(
        _markdown_table(
            non_actionable[
                non_actionable["skill"].isin(
                    [
                        "reasoning_v1_non_actionable_alpha_radar_top10",
                        "reasoning_v2_non_actionable_alpha_radar_top10",
                        "reasoning_v3_non_actionable_alpha_radar_top10",
                        "reasoning_v3_pullback_scout_top10",
                    ]
                )
            ]
            if not non_actionable.empty
            else non_actionable
        )
    )
    lines.extend(["", "## Quality Pullback Coverage", ""])
    if pullback_coverage.empty:
        lines.append("No quality pullback rows were available.")
    else:
        covered = pullback_coverage[pullback_coverage["covered"].eq(True)]
        covered_v2 = covered[covered["skill"].astype(str).str.startswith("reasoning_v2")]
        covered_v3 = covered[covered["skill"].astype(str).str.startswith("reasoning_v3")]
        covered_v3_scout = covered[covered["skill"].astype(str).eq("reasoning_v3_pullback_scout_top10")]
        lines.append(
            f"- Quality pullback candidates: `{pullback_coverage[['pool_id', 'code']].drop_duplicates().shape[0]}`"
        )
        lines.append(f"- Covered by any skill output: `{covered[['pool_id', 'code']].drop_duplicates().shape[0]}`")
        lines.append(f"- Covered by optimized v2 output: `{covered_v2[['pool_id', 'code']].drop_duplicates().shape[0]}`")
        lines.append(f"- Covered by v3 output: `{covered_v3[['pool_id', 'code']].drop_duplicates().shape[0]}`")
        lines.append(f"- Covered by v3 Pullback Scout: `{covered_v3_scout[['pool_id', 'code']].drop_duplicates().shape[0]}`")
        lines.extend(["", "Top uncovered pullbacks by latest return:", ""])
        uncovered = pullback_coverage[pullback_coverage["covered"].eq(False)].sort_values(
            ["latest_close_return_pct", "max_gain_pct"],
            ascending=[False, False],
        )
        lines.extend(_markdown_table(uncovered.head(15)))
    lines.extend(["", "## AAPL Audit", ""])
    if aapl.empty:
        lines.append("AAPL did not appear in audited repaired pool files.")
    else:
        lines.extend(_markdown_table(aapl.sort_values(["snapshot_date", "commit"]).head(50)))
    lines.extend(["", "## Data Files", ""])
    lines.extend(
        [
            "- `backtest/ibd_skill_full_repaired_audit_reports/full_repaired_pool_pick_metrics.csv`",
            "- `backtest/ibd_skill_full_repaired_audit_reports/full_repaired_pool_actionable_hit_matrix.csv`",
            "- `backtest/ibd_skill_full_repaired_audit_reports/full_repaired_pool_actionable_hit_summary.csv`",
            "- `backtest/ibd_skill_full_repaired_audit_reports/full_repaired_pool_review_hit_matrix.csv`",
            "- `backtest/ibd_skill_full_repaired_audit_reports/full_repaired_pool_review_hit_summary.csv`",
            "- `backtest/ibd_skill_full_repaired_audit_reports/full_repaired_pool_non_actionable_hit_summary.csv`",
            "- `backtest/ibd_skill_full_repaired_audit_reports/full_repaired_pool_review_oracle_top5.csv`",
            "- `backtest/ibd_skill_full_repaired_audit/pool_scope.csv`",
            "- `backtest/ibd_skill_full_repaired_audit/quality_pullback_candidates.csv`",
            "- `backtest/ibd_skill_full_repaired_audit/quality_pullback_coverage.csv`",
            "- `backtest/ibd_skill_full_repaired_audit/aapl_audit.csv`",
        ]
    )
    return "\n".join(lines)


def _scope_row(
    raw_pool: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    path: Path,
    pool_id: str,
    snapshot_date: str,
    commit: str,
) -> dict[str, object]:
    review = pool["signal"].map(to_bool).eq(True) & pool["ibd_candidate_rule"].astype(str).str.strip().ne("")
    actionable = review & pool["ibd_entry_status"].astype(str).str.upper().eq("ACTIONABLE")
    non_actionable = review & ~pool["ibd_entry_status"].astype(str).str.upper().eq("ACTIONABLE")
    inferred = pool["ibd_entry_status_repair_method"].astype(str).ne("original")
    inferred_review = review & inferred
    missing_eps = pool["eps_yoy_growth_repair_method"].astype(str).eq("missing_in_repaired_pool")
    return {
        "pool_id": pool_id,
        "snapshot_date": snapshot_date,
        "commit": commit,
        "pool_path": str(path),
        "row_count": len(pool),
        "raw_columns": len(raw_pool.columns),
        "review_universe_count": int(review.sum()),
        "actionable_count": int(actionable.sum()),
        "non_actionable_count": int(non_actionable.sum()),
        "inferred_status_rows": int(inferred.sum()),
        "inferred_review_status_rows": int(inferred_review.sum()),
        "missing_eps_rows": int(missing_eps.sum()),
    }


def _aapl_rows(
    pool: pd.DataFrame,
    *,
    path: Path,
    pool_id: str,
    snapshot_date: str,
    commit: str,
    prices: dict[str, pd.DataFrame],
    end_date: str,
) -> list[dict[str, object]]:
    rows = []
    for _, row in pool[pool["code"].astype(str).eq("AAPL")].iterrows():
        metrics = _path_metrics_for_row(row, prices, snapshot_date=snapshot_date, end_date=end_date)
        rows.append(
            {
                "pool_id": pool_id,
                "snapshot_date": snapshot_date,
                "commit": commit,
                "pool_path": str(path),
                "signal": row.get("signal"),
                "ibd_entry_status": row.get("ibd_entry_status"),
                "ibd_entry_status_repair_method": row.get("ibd_entry_status_repair_method"),
                "ibd_candidate_rule": row.get("ibd_candidate_rule"),
                "ibd_entry_reject_reason": row.get("ibd_entry_reject_reason"),
                "current_vs_ibd_candidate_pct": row.get("current_vs_ibd_candidate_pct"),
                "ibd_entry_volume_ratio": row.get("ibd_entry_volume_ratio"),
                "volume_ratio": row.get("volume_ratio"),
                "eps_yoy_growth": row.get("eps_yoy_growth"),
                "dist_to_52w_high_pct": row.get("dist_to_52w_high_pct"),
                "pullback_v_is_dry": row.get("pullback_v_is_dry"),
                "latest_close_return_pct": metrics.latest_close_return_pct,
                "max_gain_pct": metrics.max_gain_pct,
                "max_drawdown_pct": metrics.max_drawdown_pct,
                "hit_stop_8pct": metrics.hit_stop_8pct,
            }
        )
    return rows


def _path_metrics_for_row(row: pd.Series, prices: dict[str, pd.DataFrame], *, snapshot_date: str, end_date: str):
    from backtest.ibd_skill_replay.core import compute_path_metrics

    code = str(row.get("code", "")).strip()
    return compute_path_metrics(
        code=code,
        snapshot_date=snapshot_date,
        buy_price=to_float(row.get("ibd_candidate_price")),
        snapshot_close=to_float(row.get("latest_close")),
        price_bars=prices.get(code),
        end_date=end_date,
    )


def _tag_rows(rows: list[dict[str, object]], *, path: Path, pool_id: str, commit: str) -> list[dict[str, object]]:
    tagged = []
    for row in rows:
        tagged.append(
            {
                "pool_id": pool_id,
                "commit": commit,
                "pool_path": str(path),
                **row,
            }
        )
    return tagged


def _infer_entry_status(row: pd.Series) -> str:
    if not _is_review_row(row):
        return ""
    valid = to_bool(row.get("ibd_entry_valid"))
    cur = to_float(row.get("current_vs_ibd_candidate_pct"))
    if valid is True:
        if cur is not None and cur > 5:
            return "EXTENDED"
        return "ACTIONABLE"
    return "UNCONFIRMED"


def _status_repair_method(row: pd.Series, inferred: str) -> str:
    if not inferred:
        return "not_review_universe"
    valid = to_bool(row.get("ibd_entry_valid"))
    if valid is True:
        return "inferred_from_valid_entry"
    return "inferred_from_rejected_signal"


def _is_review_row(row: pd.Series) -> bool:
    return to_bool(row.get("signal")) is True and bool(str(row.get("ibd_candidate_rule", "") or "").strip())


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["No rows."]
    formatted = frame.copy()
    for column in formatted.columns:
        if column.endswith("_pct"):
            formatted[column] = formatted[column].map(_fmt)
        elif column.endswith("_rate") or column.endswith("_per_pool"):
            formatted[column] = formatted[column].map(_fmt_ratio)
    return formatted.to_markdown(index=False).splitlines()


def _single_row(frame: pd.DataFrame, skill: str) -> pd.Series | None:
    if frame.empty or "skill" not in frame.columns:
        return None
    rows = frame[frame["skill"].eq(skill)]
    if rows.empty:
        return None
    return rows.iloc[0]


def _fmt(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.2f}%"


def _fmt_ratio(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.2f}"


if __name__ == "__main__":
    main()
