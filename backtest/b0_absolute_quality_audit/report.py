from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _fmt(value: Any, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
    except Exception:
        pass
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except Exception:
        return str(value)


def _rate(value: Any) -> str:
    if value is None:
        return "N/A"
    return _fmt(float(value) * 100.0, 1, "%")


def write_report(
    path: Path,
    health: dict[str, Any],
    nonoverlap: pd.DataFrame,
    rank_buckets: pd.DataFrame,
    simple_summary: pd.DataFrame,
    rejection_reasons: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    absolute = health["absolute"]
    abs_stats = absolute["entry_aligned_w4"]
    abs_ci = absolute["entry_aligned_block_bootstrap"]
    eligible = health["eligible_random"]
    raw = health["raw_random_distinct1"]
    gate = health["eligibility"]
    ranking = health["ranking"]
    market = health["market"]

    lines = [
        "# Current Production B0 — Absolute Quality Health Check",
        "",
        "## Executive coordinates",
        "",
        "- No arbitrary PASS/FAIL score is used. The report exposes the raw coordinates, "
        "uncertainty and oracle headroom; final interpretation must use them jointly.",
        "",
        "This audit measures the current Production B0 as-is. B0 is never used to define the raw-signal "
        "benchmark universe. Current B0 eligibility/rank/Top3 are recomputed from Production source; "
        "the frozen panel's old b0_eligible/is_b0 helper columns are not trusted.",
        "",
        "Two outcome systems are deliberately separated:",
        "",
        "1. **Entry-aligned W4** — used for B0-eligible ranking/random/oracle comparisons.",
        "2. **Snapshot-close +28 calendar-day W4** — used for the raw signal universe, eligibility "
        "opportunity cost, simple baselines, and SPY/QQQ. This avoids pretending non-ACTIONABLE raw "
        "signals have a comparable Production entry event.",
        "",
        "## 1. Absolute B0 W4 cohort quality",
        "",
        "| Metric | Current B0 |",
        "| --- | ---: |",
        f"| Mature/cash weeks | {abs_stats['n']} |",
        f"| Mean capital-adjusted W4 | {_fmt(abs_stats['mean'], 2, '%')} |",
        f"| Median capital-adjusted W4 | {_fmt(abs_stats['median'], 2, '%')} |",
        f"| P10 | {_fmt(abs_stats['p10'], 2, '%')} |",
        f"| P25 | {_fmt(abs_stats['p25'], 2, '%')} |",
        f"| P75 | {_fmt(abs_stats['p75'], 2, '%')} |",
        f"| P90 | {_fmt(abs_stats['p90'], 2, '%')} |",
        f"| CVaR10 | {_fmt(abs_stats['cvar10'], 2, '%')} |",
        f"| Positive-week rate | {_rate(abs_stats['positive_rate'])} |",
        f"| Worst cohort | {_fmt(abs_stats['worst'], 2, '%')} |",
        f"| Best cohort | {_fmt(abs_stats['best'], 2, '%')} |",
        f"| Mean capital Stop8 | {_fmt(absolute['mean_capital_stop8_pct'], 2, '%')} |",
        f"| One-pick-ruin week rate | {_rate(absolute['one_pick_ruin_week_rate'])} |",
        f"| Mean slot coverage | {_rate(absolute['mean_slot_coverage'])} |",
        f"| Full Top3 rate | {_rate(absolute['full_top3_rate'])} |",
        f"| Zero-pick weeks | {absolute['zero_pick_weeks']} |",
        "",
        "Moving-block (4-week) bootstrap for the overlapping W4 cohorts:",
        "",
        f"- Mean 95% CI: **[{_fmt(abs_ci['mean_ci_low'], 2, '%')}, "
        f"{_fmt(abs_ci['mean_ci_high'], 2, '%')}]**",
        f"- Median 95% CI: **[{_fmt(abs_ci['median_ci_low'], 2, '%')}, "
        f"{_fmt(abs_ci['median_ci_high'], 2, '%')}]**",
        "",
        "These are cohort-selection diagnostics, not a tradable CAGR. Weekly W4 windows overlap.",
        "",
        "## 2. Does B0 ranking add value inside its own eligible universe?",
        "",
        f"- Strict common-maturity support: **{eligible['support_weeks']} weeks**",
        f"- Median weekly feasible-portfolio percentile: "
        f"**{_fmt(eligible['median_weekly_percentile'], 1, '%')}**",
        f"- Mean weekly feasible-portfolio percentile: "
        f"**{_fmt(eligible['mean_weekly_percentile'], 1, '%')}**",
    ]

    eedge = eligible.get("edge", {})
    if eedge:
        lines += [
            f"- Mean W4 edge vs exact eligible distinct-industry random: "
            f"**{_fmt(eedge.get('spread', {}).get('mean'), 2, '%')}**",
            f"- Median W4 edge vs random: "
            f"**{_fmt(eedge.get('spread', {}).get('median'), 2, '%')}**",
            f"- Beat-random-mean week rate: **{_rate(eedge.get('beat_rate'))}**",
            f"- Edge block-bootstrap mean CI: "
            f"**[{_fmt(eedge.get('spread_block_bootstrap', {}).get('mean_ci_low'), 2, '%')}, "
            f"{_fmt(eedge.get('spread_block_bootstrap', {}).get('mean_ci_high'), 2, '%')}]**",
        ]

    ecap = eligible.get("oracle_capture", {})
    lines += [
        f"- Eligible-universe aggregate oracle capture: "
        f"**{_rate(ecap.get('aggregate_capture_ratio'))}**",
        "",
        "Interpretation: this section isolates the ranking/selection layer after the current hard gates. "
        "A percentile near 50% means the detailed B0 ranking is not doing much once eligibility has "
        "already done the filtering.",
        "",
        "## 3. Does the whole B0 system beat the raw signal universe?",
        "",
        f"- Raw benchmark support with required price coverage: **{raw['support_weeks']} weeks**",
        f"- Median weekly percentile vs raw-signal **fixed-capacity** distinct-industry random: "
        f"**{_fmt(raw['median_weekly_percentile'], 1, '%')}**",
        f"- Mean weekly percentile: **{_fmt(raw['mean_weekly_percentile'], 1, '%')}**",
    ]

    redge = raw.get("edge", {})
    if redge:
        lines += [
            f"- Mean snapshot-W4 edge vs raw random: "
            f"**{_fmt(redge.get('spread', {}).get('mean'), 2, '%')}**",
            f"- Median snapshot-W4 edge: "
            f"**{_fmt(redge.get('spread', {}).get('median'), 2, '%')}**",
            f"- Beat-random-mean week rate: **{_rate(redge.get('beat_rate'))}**",
            f"- Edge block-bootstrap mean CI: "
            f"**[{_fmt(redge.get('spread_block_bootstrap', {}).get('mean_ci_low'), 2, '%')}, "
            f"{_fmt(redge.get('spread_block_bootstrap', {}).get('mean_ci_high'), 2, '%')}]**",
        ]

    rcap = raw.get("oracle_capture", {})
    lines += [
        f"- Raw-universe aggregate oracle capture: **{_rate(rcap.get('aggregate_capture_ratio'))}**",
        "",
        f"- Conditional Matched-N raw percentile median (name-selection only): "
        f"**{_fmt(raw.get('matched_n', {}).get('median_weekly_percentile'), 1, '%')}**",
        "",
        "This is the most important total-system coordinate: raw signal names are not pre-filtered by "
        "b0_eligible or Lane. The **primary fixed-capacity benchmark** mechanically fills up to three "
        "distinct-industry slots whenever the raw universe can do so, so B0 abstention/underfill is "
        "evaluated rather than copied into the benchmark. Matched-N is retained only as a conditional "
        "name-selection diagnostic.",
        "",
        "## 4. Eligibility gate: how many future winners does B0 retain or reject?",
        "",
        f"- Support weeks: **{gate['support_weeks']}**",
        f"- Mean raw price-outcome coverage: **{_rate(gate['mean_raw_price_coverage'])}**",
        f"- Top-20% future-winner retention by B0 eligibility: "
        f"**{_rate(gate['winner_retention_rate'])}**",
        f"- Top-20% future-winner capture by final B0 picks: "
        f"**{_rate(gate['b0_winner_capture_rate'])}**",
        f"- Winner rate among rejected candidates: **{_rate(gate['rejected_winner_rate'])}**",
        f"- Bottom-20% future-loser rejection rate: "
        f"**{_rate(gate['bottom_loser_rejection_rate'])}**",
        f"- >=20% big-winner retention: **{_rate(gate['big_winner_retention_rate'])}**",
        f"- Mean eligible-minus-rejected snapshot-W4 lift: "
        f"**{_fmt(gate['mean_gate_lift'], 2, '%')}**",
        f"- Median weekly gate lift: **{_fmt(gate['median_gate_lift'], 2, '%')}**",
        "",
        "### Hard-reject reason diagnostics",
        "",
    ]

    if rejection_reasons.empty:
        lines.append("No covered rejected-candidate reason rows.")
    else:
        lines += [
            "| Reject reason | Rows | Weeks | Mean W4 | Median W4 | Top20 winner rate | >=20% winner rate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for _, row in rejection_reasons.iterrows():
            lines.append(
                f"| {row['reason']} | {int(row['rejected_candidate_rows'])} | "
                f"{int(row['weeks'])} | {_fmt(row['mean_snapshot_w4'], 2, '%')} | "
                f"{_fmt(row['median_snapshot_w4'], 2, '%')} | "
                f"{_rate(row['top20_winner_rate'])} | {_rate(row['big_winner_rate'])} |"
            )

    lines += [
        "",
        "## 5. Fine-ranking information across the entire eligible universe",
        "",
        f"- Strict-maturity support: **{ranking['support_weeks']} weeks**",
        f"- Mean weekly Spearman (-eligible_rank vs W4): "
        f"**{_fmt(ranking['weekly_spearman_mean'], 3)}**",
        f"- Median weekly Spearman: **{_fmt(ranking['weekly_spearman_median'], 3)}**",
        f"- Positive-Spearman week rate: **{_rate(ranking['positive_spearman_week_rate'])}**",
        f"- B0 selected mean minus all-eligible mean W4: "
        f"**{_fmt(ranking['b0_minus_eligible_mean_mean'], 2, '%')}** mean / "
        f"**{_fmt(ranking['b0_minus_eligible_mean_median'], 2, '%')}** median",
        "",
        "Rank-bucket outcome profile:",
        "",
    ]

    if rank_buckets.empty:
        lines.append("No strict-maturity rank bucket rows.")
    else:
        lines += [
            "| Eligible rank bucket | Candidate rows | Weeks | Mean W4 | Median W4 | Positive | Stop8 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for _, row in rank_buckets.iterrows():
            lines.append(
                f"| {row['rank_bucket']} | {int(row['candidate_rows'])} | {int(row['weeks'])} | "
                f"{_fmt(row['mean_w4'], 2, '%')} | {_fmt(row['median_w4'], 2, '%')} | "
                f"{_rate(row['positive_rate'])} | {_rate(row['stop8_rate'])} |"
            )

    lines += [
        "",
        "## 6. Simple de-anchored rules from the raw signal universe",
        "",
        "These rules do not use b0_eligible, B0 Lane, B0 rank, reason_codes, or future outcomes. "
        "They use one raw PIT feature plus distinct-industry selection and a de-anchored fixed capacity "
        "of up to three positions. Therefore they also challenge B0 abstention/underfill instead of "
        "copying B0's weekly position count.",
        "",
    ]
    if simple_summary.empty:
        lines.append("No simple-baseline common support.")
    else:
        lines += [
            "| Baseline | Weeks | Mean W4 | Median W4 | Mean Δ vs B0 | Median Δ vs B0 | Beat B0 | Pick coverage |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for _, row in simple_summary.iterrows():
            lines.append(
                f"| {row['baseline']} | {int(row['support_weeks'])} | "
                f"{_fmt(row['mean_return'], 2, '%')} | {_fmt(row['median_return'], 2, '%')} | "
                f"{_fmt(row['mean_spread_vs_b0'], 2, '%')} | "
                f"{_fmt(row['median_spread_vs_b0'], 2, '%')} | "
                f"{_rate(row['beat_b0_rate'])} | {_rate(row['mean_pick_coverage'])} |"
            )

    spy = market["vs_spy_selection_quality"]
    spy_cap = market["vs_spy_exposure_matched"]
    spy_full = market["vs_spy_full_exposure"]
    qqq = market["vs_qqq_selection_quality"]
    qqq_full = market["vs_qqq_full_exposure"]
    lines += [
        "",
        "## 7. Market benchmark",
        "",
        f"- B0 active-pick snapshot-W4 mean spread vs SPY: "
        f"**{_fmt(spy.get('spread', {}).get('mean'), 2, '%')}** "
        f"({spy.get('support_weeks', 0)} weeks)",
        f"- B0 capital-adjusted spread vs exposure-matched SPY: "
        f"**{_fmt(spy_cap.get('spread', {}).get('mean'), 2, '%')}**",
        f"- B0 capital-adjusted spread vs fully invested SPY: "
        f"**{_fmt(spy_full.get('spread', {}).get('mean'), 2, '%')}**",
        f"- B0 active-pick snapshot-W4 mean spread vs QQQ: "
        f"**{_fmt(qqq.get('spread', {}).get('mean'), 2, '%')}** "
        f"({qqq.get('support_weeks', 0)} weeks)",
        f"- B0 capital-adjusted spread vs fully invested QQQ: "
        f"**{_fmt(qqq_full.get('spread', {}).get('mean'), 2, '%')}**",
        "",
        "## 8. Four-offset non-overlap stability",
        "",
        "Each row takes every fourth weekly cohort. This removes W4 horizon overlap within an offset.",
        "",
    ]
    if nonoverlap.empty:
        lines.append("No non-overlap rows.")
    else:
        lines += [
            "| Comparison | Offset | Weeks | Value mean | Value median | Benchmark mean | Spread mean | Spread median |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for _, row in nonoverlap.iterrows():
            lines.append(
                f"| {row['comparison']} | {int(row['offset'])} | {int(row['weeks'])} | "
                f"{_fmt(row.get('value_mean'), 2, '%')} | {_fmt(row.get('value_median'), 2, '%')} | "
                f"{_fmt(row.get('benchmark_mean'), 2, '%')} | {_fmt(row.get('spread_mean'), 2, '%')} | "
                f"{_fmt(row.get('spread_median'), 2, '%')} |"
            )

    lines += [
        "",
        "## 9. Evidence boundary",
        "",
        health["evidence_boundary"],
        "",
        "The most reliable interpretation hierarchy is:",
        "",
        "1. raw-signal percentile / edge = total B0 selection-system quality;",
        "2. eligible-random percentile / edge = incremental ranking quality after hard gates;",
        "3. eligibility winner retention = gate opportunity-cost quality;",
        "4. oracle capture = how much headroom remains;",
        "5. four-offset consistency = whether overlapping W4 labels exaggerate stability.",
        "",
        "No single mean return or p-value is treated as a sufficient verdict.",
        "",
        "## Provenance",
        "",
        f"- source_git_sha: {manifest['source_git_sha']}",
        f"- production_b0_hash: {manifest['production_b0_hash']}",
        f"- panel_hash: {manifest['panel_hash']}",
        f"- price_cache_hash: {manifest['price_cache_hash']}",
        f"- snapshots: **{manifest['snapshot_count']}** "
        f"({manifest['panel_min_snapshot']} .. {manifest['panel_max_snapshot']})",
        "",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
