from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from backtest.ibd_skill_iteration.reports import (
    build_non_actionable_hit_summary,
    build_reasoning_pick_metric_rows,
    find_quality_pullback_candidates,
)
from backtest.ibd_skill_oracle.core import (
    build_hit_matrix,
    oracle_rows,
    rank_path_adjusted_oracle,
    summarize_group,
)
from backtest.ibd_skill_oracle.run_oracle_replay import (
    _build_candidates,
    _candidate_metric_rows,
    _load_replay_pools,
    _skill_pick_rows,
)
from backtest.ibd_skill_replay.run_ytd_replay import _load_price_cache, _load_supplemental_prices


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay experimental reasoning IBD skill against historical pools.")
    parser.add_argument("--manifest", default="backtest/ibd_skill_replay_reports/manifest.json")
    parser.add_argument("--price-cache", default="results_pkl/stock_data_150826_1d.pkl")
    parser.add_argument("--supplemental-price-csv", default="backtest/ibd_skill_replay_audit/supplemental_price_bars.csv")
    parser.add_argument("--end", default="2026-08-14")
    parser.add_argument("--output-root", default="backtest")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    report_dir = output_root / "ibd_skill_iteration_reports"
    audit_dir = output_root / "ibd_skill_iteration_audit"
    for directory in (report_dir, audit_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    prices = _load_price_cache(Path(args.price_cache))
    prices.update(_load_supplemental_prices(Path(args.supplemental_price_csv)))
    pools = _load_replay_pools(manifest)

    actionable_oracle_rows = []
    review_oracle_rows = []
    review_metric_rows = []
    baseline_skill_rows = []
    reasoning_rows = []
    for meta in manifest["comparable_snapshots"]:
        snapshot_date = meta["snapshot_date"]
        pool = pools[snapshot_date]
        actionable_candidates, _ = _build_candidates(pool, prices, snapshot_date=snapshot_date, end_date=args.end)
        review_candidates, _ = _build_candidates(
            pool,
            prices,
            snapshot_date=snapshot_date,
            end_date=args.end,
            universe="review",
        )
        actionable_ranked = rank_path_adjusted_oracle(actionable_candidates)
        review_ranked = rank_path_adjusted_oracle(review_candidates)
        actionable_oracle_rows.extend(oracle_rows(actionable_ranked, ranking_method="actionable_path_adjusted", limit=5))
        review_oracle_rows.extend(oracle_rows(review_ranked, ranking_method="review_path_adjusted", limit=5))
        review_metric_rows.extend(_candidate_metric_rows(review_candidates, include_entry_status=True))
        baseline_skill_rows.extend(_skill_pick_rows(pool, prices, snapshot_date=snapshot_date, end_date=args.end))
        for version in ["v1", "v2"]:
            reasoning_rows.extend(
                build_reasoning_pick_metric_rows(
                    pool,
                    prices,
                    snapshot_date=snapshot_date,
                    end_date=args.end,
                    version=version,
                )
            )

    actionable_oracle = pd.DataFrame(actionable_oracle_rows)
    review_oracle = pd.DataFrame(review_oracle_rows)
    review_metrics = pd.DataFrame(review_metric_rows)
    baseline_skills = pd.DataFrame(baseline_skill_rows)
    reasoning_skills = pd.DataFrame(reasoning_rows)
    skill_metrics = pd.concat([baseline_skills, reasoning_skills], ignore_index=True, sort=False)

    actionable_hit_matrix = build_hit_matrix(actionable_oracle, skill_metrics)
    review_hit_matrix = build_hit_matrix(review_oracle, skill_metrics)
    skill_summary = summarize_group(skill_metrics, group_cols=["skill"])
    skill_order_summary = summarize_group(skill_metrics, group_cols=["skill", "pick_order"])
    quality_pullbacks = find_quality_pullback_candidates(review_metrics, limit=30)
    pullback_coverage = _build_pullback_coverage(quality_pullbacks, skill_metrics)
    non_actionable_alpha_metrics = skill_metrics[
        skill_metrics["skill"].astype(str).str.contains("_non_actionable_alpha_radar_top10", regex=False)
    ].copy()
    non_actionable_hit_summary = build_non_actionable_hit_summary(non_actionable_alpha_metrics, review_oracle)

    actionable_oracle.to_csv(report_dir / "actionable_oracle_top5.csv", index=False)
    review_oracle.to_csv(report_dir / "review_oracle_top5.csv", index=False)
    skill_metrics.to_csv(report_dir / "iteration_pick_metrics.csv", index=False)
    actionable_hit_matrix.to_csv(report_dir / "actionable_oracle_hit_matrix.csv", index=False)
    review_hit_matrix.to_csv(report_dir / "review_oracle_hit_matrix.csv", index=False)
    skill_summary.to_csv(report_dir / "skill_quality_summary.csv", index=False)
    skill_order_summary.to_csv(report_dir / "skill_order_quality_summary.csv", index=False)
    non_actionable_alpha_metrics.to_csv(report_dir / "non_actionable_alpha_pick_metrics.csv", index=False)
    non_actionable_hit_summary.to_csv(report_dir / "non_actionable_alpha_hit_summary.csv", index=False)
    quality_pullbacks.to_csv(audit_dir / "quality_pullback_candidates.csv", index=False)
    pullback_coverage.to_csv(audit_dir / "quality_pullback_coverage.csv", index=False)

    (report_dir / "iteration_summary.md").write_text(
        _render_iteration_summary(
            actionable_oracle,
            review_oracle,
            skill_metrics,
            actionable_hit_matrix,
            review_hit_matrix,
            skill_summary,
            skill_order_summary,
            non_actionable_hit_summary,
            quality_pullbacks,
            pullback_coverage,
            args,
        ),
        encoding="utf-8",
    )
    (report_dir / "reasoning_skill_v2_design.md").write_text(_render_skill_design(), encoding="utf-8")
    result_manifest = {
        "source_replay_manifest": args.manifest,
        "end": args.end,
        "outputs": {
            "iteration_pick_metrics": str(report_dir / "iteration_pick_metrics.csv"),
            "actionable_oracle_hit_matrix": str(report_dir / "actionable_oracle_hit_matrix.csv"),
            "review_oracle_hit_matrix": str(report_dir / "review_oracle_hit_matrix.csv"),
            "non_actionable_alpha_pick_metrics": str(report_dir / "non_actionable_alpha_pick_metrics.csv"),
            "non_actionable_alpha_hit_summary": str(report_dir / "non_actionable_alpha_hit_summary.csv"),
            "quality_pullback_candidates": str(audit_dir / "quality_pullback_candidates.csv"),
            "quality_pullback_coverage": str(audit_dir / "quality_pullback_coverage.csv"),
            "summary": str(report_dir / "iteration_summary.md"),
            "reasoning_skill_v2_design": str(report_dir / "reasoning_skill_v2_design.md"),
        },
    }
    (report_dir / "manifest.json").write_text(json.dumps(result_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result_manifest, indent=2, ensure_ascii=False))


def _build_pullback_coverage(pullbacks: pd.DataFrame, skill_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if pullbacks.empty:
        return pd.DataFrame(columns=["snapshot_date", "code", "skill", "covered", "pick_order"])
    for _, winner in pullbacks.iterrows():
        subset = skill_metrics[
            skill_metrics["snapshot_date"].eq(winner["snapshot_date"])
            & skill_metrics["code"].astype(str).eq(str(winner["code"]))
        ]
        if subset.empty:
            rows.append(
                {
                    "snapshot_date": winner["snapshot_date"],
                    "code": winner["code"],
                    "skill": "none",
                    "covered": False,
                    "pick_order": "",
                    "latest_close_return_pct": winner["latest_close_return_pct"],
                    "max_gain_pct": winner["max_gain_pct"],
                    "max_drawdown_pct": winner["max_drawdown_pct"],
                }
            )
            continue
        for _, row in subset.iterrows():
            rows.append(
                {
                    "snapshot_date": winner["snapshot_date"],
                    "code": winner["code"],
                    "skill": row["skill"],
                    "covered": True,
                    "pick_order": row["pick_order"],
                    "latest_close_return_pct": winner["latest_close_return_pct"],
                    "max_gain_pct": winner["max_gain_pct"],
                    "max_drawdown_pct": winner["max_drawdown_pct"],
                    "lane": row.get("lane", ""),
                    "final_group": row.get("final_group", ""),
                    "reason_codes": row.get("reason_codes", ""),
                    "risk_codes": row.get("risk_codes", ""),
                }
            )
    return pd.DataFrame(rows)


def _render_iteration_summary(
    actionable_oracle: pd.DataFrame,
    review_oracle: pd.DataFrame,
    skill_metrics: pd.DataFrame,
    actionable_hit_matrix: pd.DataFrame,
    review_hit_matrix: pd.DataFrame,
    skill_summary: pd.DataFrame,
    skill_order_summary: pd.DataFrame,
    non_actionable_hit_summary: pd.DataFrame,
    quality_pullbacks: pd.DataFrame,
    pullback_coverage: pd.DataFrame,
    args: argparse.Namespace,
) -> str:
    lines = [
        "# IBD Skill Iteration Replay",
        "",
        "## Core Conclusion",
        "",
        "Reasoning v2 tests a different job from the current live skill: it keeps ACTIONABLE priority review separate from Alpha Radar discovery, treats imperfect Geometry as path evidence rather than a default ranking penalty, and adds the old skill's useful fresh-demand reasoning without allowing clear technical failures to survive.",
        "",
        f"- Evaluation end: `{args.end}`",
        f"- Comparable complete-week snapshots: `{actionable_oracle['snapshot_date'].nunique() if not actionable_oracle.empty else 0}`",
        "",
        "## Recommended Iteration",
        "",
    ]
    lines.extend(
        _recommended_iteration_lines(
            skill_metrics,
            actionable_hit_matrix,
            pullback_coverage,
            non_actionable_hit_summary,
        )
    )
    lines.extend(
        [
            "",
            "## Skill Quality Summary",
            "",
        ]
    )
    lines.extend(_markdown_table(skill_summary))
    lines.extend(["", "## Skill Order Quality", ""])
    lines.extend(_markdown_table(skill_order_summary))
    lines.extend(["", "## ACTIONABLE Oracle Hit Matrix", ""])
    lines.extend(_markdown_table(actionable_hit_matrix))
    lines.extend(["", "## Review Universe Oracle Hit Matrix", ""])
    lines.extend(_markdown_table(review_hit_matrix))
    lines.extend(["", "## Non-ACTIONABLE Alpha Radar Hit Summary", ""])
    lines.extend(_markdown_table(non_actionable_hit_summary))
    lines.extend(
        [
            "",
            "## Review Universe Oracle Top5 By Week",
            "",
            "| Week | Rank | Code | Status | Rule | Latest Ret | Max Gain | Max Drawdown | -8% Stop |",
            "|---|---:|---|---|---|---:|---:|---:|---|",
        ]
    )
    for _, row in review_oracle.sort_values(["snapshot_date", "oracle_rank"]).iterrows():
        lines.append(
            f"| {row['snapshot_date']} | {int(row['oracle_rank'])} | {row['code']} | {row.get('ibd_entry_status', '')} | "
            f"{row.get('ibd_candidate_rule', '')} | {_fmt(row['latest_close_return_pct'])} | {_fmt(row['max_gain_pct'])} | "
            f"{_fmt(row['max_drawdown_pct'])} | {'yes' if row['hit_stop_8pct'] else 'no'} |"
        )
    lines.extend(["", "## Quality Pullback Candidates", ""])
    if quality_pullbacks.empty:
        lines.append("No quality pullback candidates matched the no-stop positive-path definition.")
    else:
        lines.extend(
            [
                "| Week | Code | Status | Rule | Latest Ret | Max Gain | Max Drawdown | Covered By |",
                "|---|---|---|---|---:|---:|---:|---|",
            ]
        )
        for _, row in quality_pullbacks.head(15).iterrows():
            covered = pullback_coverage[
                pullback_coverage["snapshot_date"].eq(row["snapshot_date"])
                & pullback_coverage["code"].astype(str).eq(str(row["code"]))
                & pullback_coverage["covered"].eq(True)
            ]
            covered_by = ", ".join(
                f"{cover.skill}#{int(cover.pick_order)}" for _, cover in covered.sort_values("skill").iterrows()
            )
            lines.append(
                f"| {row['snapshot_date']} | {row['code']} | {row.get('ibd_entry_status', '')} | {row.get('ibd_candidate_rule', '')} | "
                f"{_fmt(row['latest_close_return_pct'])} | {_fmt(row['max_gain_pct'])} | {_fmt(row['max_drawdown_pct'])} | {covered_by or 'none'} |"
            )
    lines.extend(["", "## Data Files", ""])
    lines.extend(
        [
            "- `backtest/ibd_skill_iteration_reports/iteration_pick_metrics.csv`",
            "- `backtest/ibd_skill_iteration_reports/skill_quality_summary.csv`",
            "- `backtest/ibd_skill_iteration_reports/actionable_oracle_hit_matrix.csv`",
            "- `backtest/ibd_skill_iteration_reports/review_oracle_hit_matrix.csv`",
            "- `backtest/ibd_skill_iteration_reports/non_actionable_alpha_pick_metrics.csv`",
            "- `backtest/ibd_skill_iteration_reports/non_actionable_alpha_hit_summary.csv`",
            "- `backtest/ibd_skill_iteration_audit/quality_pullback_candidates.csv`",
            "- `backtest/ibd_skill_iteration_audit/quality_pullback_coverage.csv`",
            "- `backtest/ibd_skill_iteration_reports/reasoning_skill_v2_design.md`",
        ]
    )
    return "\n".join(lines)


def _render_skill_design() -> str:
    return "\n".join(
        [
            "# Reasoning Skill v2 Draft",
            "",
            "## Intent",
            "",
            "Use the historical replay to improve the IBD pre-screen reasoning without turning replay outcomes into hard fitted thresholds.",
            "",
            "## Rules To Carry Forward",
            "",
            "- Keep Immediate Review Priority and Alpha Radar separate. ACTIONABLE remains the review-ready list; non-ACTIONABLE candidates can only be radar/watchlist evidence.",
            "- Give non-ACTIONABLE Alpha Radar independent capacity. Do not let ACTIONABLE leftovers consume the non-ACTIONABLE discovery list.",
            "- Audit non-ACTIONABLE radar by status against the Review Universe oracle. Missing breakout-day fields on non-ACTIONABLE rows are manual chart-confirmation needs, not automatic suppression when weekly/EPS/near-high/pullback evidence is strong.",
            "- Treat Geometry as one path-evidence block. Imperfect Geometry is a caution and a manual chart-confirmation prompt, not a default ranking penalty.",
            "- Penalize only clear technical failure: defensive/no breakout range, a materially weak close below the established breakout-quality zone, below candidate buy point, or other broken structure evidence.",
            "- Carry forward the old skill's useful Fresh Demand Alpha reasoning: new/near buy point, abnormal breakout demand, supportive weekly follow-through, constructive EPS context, and proximity/recovery near highs.",
            "- For pullbacks, ask whether the candidate is digesting normally after demand or failing. Dry pullback evidence supports radar; a non-dry pullback is a risk note rather than an automatic rejection unless combined with clear failure.",
            "- Rank evidence completeness before cosmetic perfection. A candidate with strong demand, follow-through, and constructive fundamental context can outrank a prettier breakout-day geometry candidate.",
            "",
            "## Anti-Overfit Guard",
            "",
            "Do not paste replay medians or sample-specific numeric ranges into the live skill. Use this replay to teach the reasoning order: evidence clusters first, clear invalidation second, exact historical return labels never.",
        ]
    )


def _recommended_iteration_lines(
    skill_metrics: pd.DataFrame,
    actionable_hit_matrix: pd.DataFrame,
    pullback_coverage: pd.DataFrame,
    non_actionable_hit_summary: pd.DataFrame,
) -> list[str]:
    lines = [
        "The recommended experimental rule is `reasoning_v2_priority_top3` for immediate review plus `reasoning_v2_alpha_radar_top5` / `reasoning_v2_pullback_radar_top5` for discovery.",
        "",
    ]
    summary = summarize_group(skill_metrics, group_cols=["skill"])
    key_skills = [
        "new_final_top3",
        "old_final_top3",
        "reasoning_v1_priority_top3",
        "reasoning_v2_priority_top3",
        "new_raw_top5",
        "reasoning_v2_actionable_raw_top5",
    ]
    for skill in key_skills:
        rowset = summary[summary["skill"].eq(skill)]
        if rowset.empty:
            continue
        row = rowset.iloc[0]
        hits = actionable_hit_matrix[actionable_hit_matrix["skill"].eq(skill)]
        top3_hits = int(hits["oracle_top3_hits"].sum()) if not hits.empty else 0
        top5_hits = int(hits["oracle_top5_hits"].sum()) if not hits.empty else 0
        lines.append(
            f"- `{skill}`: median latest `{_fmt(row['median_latest_return_pct'])}`, median max gain `{_fmt(row['median_max_gain_pct'])}`, "
            f"worst latest `{_fmt(row['worst_latest_return_pct'])}`, stops `{int(row['stop_8pct_count'])}`, "
            f"ACTIONABLE oracle hits Top3 `{top3_hits}`, Top5 `{top5_hits}`."
        )
    imax = skill_metrics[skill_metrics["code"].astype(str).eq("IMAX")]
    if not imax.empty:
        v2_imax = imax[imax["skill"].eq("reasoning_v2_priority_top3")]
        if not v2_imax.empty:
            row = v2_imax.sort_values("pick_order").iloc[0]
            lines.append(
                f"- `IMAX`: v2 ranks it priority #{int(row['pick_order'])}, latest `{_fmt(row['latest_close_return_pct'])}`, "
                f"max gain `{_fmt(row['max_gain_pct'])}`, max drawdown `{_fmt(row['max_drawdown_pct'])}`, stop `{'yes' if row['hit_stop_8pct'] else 'no'}`."
            )
    covered_pullbacks = pullback_coverage[
        pullback_coverage["covered"].eq(True)
        & pullback_coverage["skill"].astype(str).str.startswith("reasoning_v2")
    ]
    if not covered_pullbacks.empty:
        best = covered_pullbacks.sort_values("latest_close_return_pct", ascending=False).iloc[0]
        lines.append(
            f"- Pullback discovery: v2 surfaces `{best['code']}` via `{best['skill']}#{int(best['pick_order'])}`, "
            f"latest `{_fmt(best['latest_close_return_pct'])}`, max gain `{_fmt(best['max_gain_pct'])}`, "
            f"max drawdown `{_fmt(best['max_drawdown_pct'])}`."
        )
    v1_non_actionable = non_actionable_hit_summary[
        non_actionable_hit_summary["skill"].eq("reasoning_v1_non_actionable_alpha_radar_top10")
    ]
    v2_non_actionable = non_actionable_hit_summary[
        non_actionable_hit_summary["skill"].eq("reasoning_v2_non_actionable_alpha_radar_top10")
    ]
    if not v2_non_actionable.empty:
        v1_top3 = int(v1_non_actionable["review_oracle_top3_hits"].sum()) if not v1_non_actionable.empty else 0
        v1_top5 = int(v1_non_actionable["review_oracle_top5_hits"].sum()) if not v1_non_actionable.empty else 0
        v2_top3 = int(v2_non_actionable["review_oracle_top3_hits"].sum())
        v2_top5 = int(v2_non_actionable["review_oracle_top5_hits"].sum())
        v2_codes = ",".join(
            code
            for value in v2_non_actionable["review_oracle_top5_hit_codes"].dropna().astype(str)
            for code in value.split(",")
            if code
        )
        v2_risk = summary[summary["skill"].eq("reasoning_v2_non_actionable_alpha_radar_top10")]
        risk_text = ""
        if not v2_risk.empty:
            risk_row = v2_risk.iloc[0]
            risk_text = (
                f" Median max gain `{_fmt(risk_row['median_max_gain_pct'])}` but stops `{int(risk_row['stop_8pct_count'])}` "
                f"and worst latest `{_fmt(risk_row['worst_latest_return_pct'])}` keep this as radar-only."
            )
        lines.append(
            f"- Non-ACTIONABLE radar: v1 hit Review Universe oracle Top3/Top5 `{v1_top3}/{v1_top5}`; "
            f"v2 hit `{v2_top3}/{v2_top5}` by adding EXTENDED demand-continuation observation "
            f"(`{v2_codes or 'none'}`).{risk_text}"
        )
    lines.append(
        "- Interpretation: v2 improves the current skill's first-three ordering without importing the old skill's severe tail risk; v1 is retained as an audit trail showing why weaker Geometry failure handling was not enough."
    )
    return lines


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["No rows."]
    formatted = frame.copy()
    for column in formatted.columns:
        if column.endswith("_pct"):
            formatted[column] = formatted[column].map(_fmt)
        elif column.endswith("_rate"):
            formatted[column] = formatted[column].map(_fmt_ratio)
    return formatted.to_markdown(index=False).splitlines()


def _fmt(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.2f}%"


def _fmt_ratio(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.2f}%"


if __name__ == "__main__":
    main()
