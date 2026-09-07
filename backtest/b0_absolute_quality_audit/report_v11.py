from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _fmt(v: Any, d: int = 2, suffix: str = "") -> str:
    if v is None:
        return "N/A"
    try:
        if pd.isna(v):
            return "N/A"
    except Exception:
        pass
    try:
        return f"{float(v):.{d}f}{suffix}"
    except Exception:
        return str(v)


def _rate(v: Any) -> str:
    if v is None:
        return "N/A"
    return _fmt(float(v) * 100.0, 1, "%")


def write_v11_report(
    path: Path,
    *,
    health: dict[str, Any],
    raw_by_pick: pd.DataFrame,
    capacity: pd.DataFrame,
    capacity_pick_quality: pd.DataFrame,
    capacity_added_reason: pd.DataFrame,
    capacity_stop8: pd.DataFrame,
    underfill_causes: pd.DataFrame,
    support_calendar: pd.DataFrame,
    momentum_gate: pd.DataFrame,
    momentum_gate_reasons: pd.DataFrame,
    exclusive_reject: pd.DataFrame,
    overlap_reject: pd.DataFrame,
    reject_combos: pd.DataFrame,
    rank_buckets: pd.DataFrame,
    simple: pd.DataFrame,
    simple_stop8: pd.DataFrame,
    market: pd.DataFrame,
    nonoverlap: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    absolute = health["absolute"]
    ranking = health["eligible_ranking_active_choice"]
    raw = health["raw_fixed_capacity_next_open"]
    gate = health["gate"]
    info = health["ranking_information"]

    lines = [
        "# Current Production B0 — Absolute Quality Health Check v1.3",
        "",
        "## Executive coordinates",
        "",
        "No composite PASS/FAIL score is used. B0 is evaluated on two separate axes:",
        "",
        "- **Name selection quality**: compare B0 with random portfolios using the same weekly N.",
        "- **Capital utilization quality**: compare B0 with a de-anchored fixed-capacity 3-slot portfolio "
        "and with explicit Fill3 counterfactuals.",
        "",
        "This prevents B0's 0/1/2-stock weeks from being either unfairly punished in the ranking audit "
        "or automatically protected by forcing the benchmark to hold the same amount of cash.",
        "",
        "Raw-universe and market comparisons use a tradable outcome:",
        "",
        "**first trading-session open after the snapshot -> close at entry date + 28 calendar days**.",
        "",
        f"Yahoo supplementation and all benchmark outcomes are frozen at **{manifest['audit_as_of_date']}**.",
        "",
        "## 1. Absolute Production B0",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Mature entry-aligned weeks | {absolute['entry_aligned']['n']} |",
        f"| Mean W4 capital return | {_fmt(absolute['entry_aligned']['mean'], 2, '%')} |",
        f"| Median W4 capital return | {_fmt(absolute['entry_aligned']['median'], 2, '%')} |",
        f"| P10 | {_fmt(absolute['entry_aligned']['p10'], 2, '%')} |",
        f"| CVaR10 | {_fmt(absolute['entry_aligned']['cvar10'], 2, '%')} |",
        f"| Positive week rate | {_rate(absolute['entry_aligned']['positive_rate'])} |",
        f"| Mean slot coverage | {_rate(absolute['mean_slot_coverage_all_snapshots'])} |",
        f"| Full Top3 rate | {_rate(absolute['full3_rate_all_snapshots'])} |",
        "",
        f"Pick-count distribution across all snapshots: **{absolute['pick_count_distribution']}**.",
        "",
        "The absolute W4 cohorts overlap and must not be annualized as independent monthly trades.",
        "",
        "## 2. Ranking quality — only weeks where ranking actually had a choice",
        "",
        f"- Eligible/mature weeks: **{ranking['all_valid_weeks']}**",
        f"- Gate-locked / one-feasible-portfolio weeks: **{ranking['no_choice_weeks']}**",
        f"- **Active-choice weeks: {ranking['active_choice_weeks']}**",
        f"- Active-choice mean feasible percentile: **{_fmt(ranking['mean_percentile'], 1, '%')}**",
        f"- Active-choice median feasible percentile: **{_fmt(ranking['median_percentile'], 1, '%')}**",
    ]

    edge = ranking.get("edge", {})
    lines += [
        f"- Mean B0 edge vs eligible random: **{_fmt(edge.get('spread', {}).get('mean'), 2, 'pp')}**",
        f"- Median edge: **{_fmt(edge.get('spread', {}).get('median'), 2, 'pp')}**",
        f"- Beat-random week rate: **{_rate(edge.get('beat_rate'))}**",
        f"- 4-week block-bootstrap mean-edge CI: "
        f"**[{_fmt(edge.get('spread_block_bootstrap', {}).get('mean_ci_low'), 2, 'pp')}, "
        f"{_fmt(edge.get('spread_block_bootstrap', {}).get('mean_ci_high'), 2, 'pp')}]**",
        f"- Aggregate oracle capture on active-choice weeks: "
        f"**{_rate(ranking.get('oracle_capture', {}).get('aggregate_capture_ratio'))}**",
        "",
        "This is the cleanest estimate of B0's incremental ranking skill after hard eligibility. "
        "Weeks with only one feasible portfolio are excluded from the percentile headline because "
        "there was no ranking decision to evaluate.",
        "",
        "## 3. Whole-system quality vs raw signal universe — tradable next-open outcome",
        "",
        f"- Strict 100%-covered support: **{raw['support_weeks']} weeks** "
        f"({_rate(raw['support_fraction_of_snapshots'])} of all snapshots)",
        f"- Mean raw price coverage across all weeks: **{_rate(raw['mean_price_coverage_all_weeks'])}**",
        f"- Mean fixed-3 percentile: **{_fmt(raw['mean_percentile'], 1, '%')}**",
        f"- Median fixed-3 percentile: **{_fmt(raw['median_percentile'], 1, '%')}**",
    ]
    redge = raw.get("edge", {})
    lines += [
        f"- Mean capital spread vs raw fixed-3 random: "
        f"**{_fmt(redge.get('spread', {}).get('mean'), 2, 'pp')}**",
        f"- Median spread: **{_fmt(redge.get('spread', {}).get('median'), 2, 'pp')}**",
        f"- Beat fixed-3 random week rate: **{_rate(redge.get('beat_rate'))}**",
        f"- Mean-edge block-bootstrap CI: "
        f"**[{_fmt(redge.get('spread_block_bootstrap', {}).get('mean_ci_low'), 2, 'pp')}, "
        f"{_fmt(redge.get('spread_block_bootstrap', {}).get('mean_ci_high'), 2, 'pp')}]**",
        f"- Raw-universe aggregate oracle capture: "
        f"**{_rate(raw.get('oracle_capture', {}).get('aggregate_capture_ratio'))}**",
        "",
        "### Raw fixed-3 result split by B0's actual weekly position count",
        "",
    ]

    if raw_by_pick.empty:
        lines.append("No strict-support rows.")
    else:
        lines += [
            "| B0 picks | Weeks | B0 mean | Fixed-3 random mean | Mean Δ | Median Δ | Beat rate | Median percentile |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for _, row in raw_by_pick.iterrows():
            lines.append(
                f"| {int(row['pick_count'])} | {int(row['support_weeks'])} | "
                f"{_fmt(row['b0_mean'], 2, '%')} | {_fmt(row['random_mean'], 2, '%')} | "
                f"{_fmt(row['mean_spread'], 2, 'pp')} | {_fmt(row['median_spread'], 2, 'pp')} | "
                f"{_rate(row['beat_rate'])} | {_fmt(row['median_percentile'], 1, '%')} |"
            )

    matched = raw.get("matched_n_edge", {})
    lines += [
        "",
        "Matched-N is retained as a name-selection diagnostic only:",
        f"- Mean matched-N edge: **{_fmt(matched.get('spread', {}).get('mean'), 2, 'pp')}**",
        f"- Median matched-N edge: **{_fmt(matched.get('spread', {}).get('median'), 2, 'pp')}**",
        f"- Beat-rate: **{_rate(matched.get('beat_rate'))}**",
        "",
        "### Temporal support coverage",
        "",
        "Strict-support results are interpreted together with their calendar concentration.",
        "",
    ]

    if support_calendar.empty:
        lines.append("No support-calendar rows.")
    else:
        raw_support = support_calendar[
            support_calendar["comparison"].isin(
                ["raw_fixed3_primary", "simple_momentum_20"]
            )
        ]
        lines += [
            "| Comparison | Quarter | Support weeks | Total snapshots | Support rate |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
        for _, row in raw_support.iterrows():
            lines.append(
                f"| {row['comparison']} | {row['quarter']} | "
                f"{int(row['support_weeks'])} | {int(row['total_snapshots'])} | "
                f"{_rate(row['support_rate'])} |"
            )

    lines += [
        "",
        "## 4. Underfill / cash policy — Fill3 counterfactual ladder",
        "",
        "### Why B0 is underfilled",
        "",
    ]

    if underfill_causes.empty:
        lines.append("No underfill-cause rows.")
    else:
        lines += [
            "| Cause | Weeks | Mean original picks | Mature next-open weeks |",
            "| --- | ---: | ---: | ---: |",
        ]
        for _, row in underfill_causes.iterrows():
            lines.append(
                f"| {row['underfill_cause']} | {int(row['weeks'])} | "
                f"{_fmt(row['mean_original_pick_count'], 2)} | {int(row['mature_weeks'])} |"
            )

    lines += [
        "",
        "### Fill3 counterfactuals",
        "",
        "Every Fill3 policy preserves all original B0 picks. It may only fill empty slots; it never "
        "replaces an original pick. This isolates the value of cash/underfill rather than creating "
        "a new ranking system.",
        "",
        "- RELAX_INDUSTRY: only relax distinct-industry when already-eligible names remain.",
        "- EPS_ONLY: fill only with candidates rejected solely for EPS unknown; distinct1 remains.",
        "- SINGLE_REJECT: fill with the highest B0-ranked candidate that failed exactly one hard gate.",
        "- ANY_REJECT: diagnostic upper bound; any rejected known-industry candidate may fill.",
        "",
    ]

    if capacity.empty:
        lines.append("No mature capacity diagnostics.")
    else:
        lines += [
            "| Policy | Scope | Weeks | Mean Δ vs B0 | Median Δ | Beat B0 | Slot-stop exposure Δ | Any Stop/≤-8 week Δ | Full3 | Added-pick mean |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for _, row in capacity.iterrows():
            lines.append(
                f"| {row['policy_id']} | {row['scope']} | {int(row['support_weeks'])} | "
                f"{_fmt(row['mean_spread_vs_b0'], 2, 'pp')} | "
                f"{_fmt(row['median_spread_vs_b0'], 2, 'pp')} | "
                f"{_rate(row['beat_b0_rate'])} | "
                f"{_fmt(row['mean_slot_stop_exposure_delta_pp'], 2, 'pp')} | "
                f"{_fmt(row['any_stop_or_le8_week_delta_pp'], 2, 'pp')} | "
                f"{_rate(row['full3_rate'])} | "
                f"{_fmt(row['mean_added_pick_return'], 2, '%')} |"
            )

    lines += [
        "",
        "### Per-pick quality — removes the mechanical effect of holding more names",
        "",
    ]

    if capacity_pick_quality.empty:
        lines.append("No per-pick capacity diagnostics.")
    else:
        lines += [
            "| Policy | Cohort | Picks | Mean W4 | Median W4 | P10 | CVaR10 | Positive | Stop8/pick | Final W4≤-8 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for _, row in capacity_pick_quality.iterrows():
            lines.append(
                f"| {row['policy_id']} | {row['cohort']} | {int(row['picks'])} | "
                f"{_fmt(row['mean_w4'], 2, '%')} | {_fmt(row['median_w4'], 2, '%')} | "
                f"{_fmt(row['p10_w4'], 2, '%')} | {_fmt(row['cvar10_w4'], 2, '%')} | "
                f"{_rate(row['positive_rate'])} | {_rate(row['stop8_rate'])} | "
                f"{_rate(row['terminal_le_minus8_rate'])} |"
            )

    if not capacity_added_reason.empty:
        lines += [
            "",
            "Added fills by exact reject reason:",
            "",
            "| Policy | Reject reason | Picks | Mean W4 | Median W4 | Positive | Stop8/pick | Final W4≤-8 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for _, row in capacity_added_reason.iterrows():
            lines.append(
                f"| {row['policy_id']} | {row['reject_reason']} | {int(row['picks'])} | "
                f"{_fmt(row['mean_w4'], 2, '%')} | {_fmt(row['median_w4'], 2, '%')} | "
                f"{_rate(row['positive_rate'])} | {_rate(row['stop8_rate'])} | "
                f"{_rate(row['terminal_le_minus8_rate'])} |"
            )

    lines += [
        "",
        "### Idealized Stop8 execution scenario",
        "",
        "For every name that triggers Stop8, scenario return is set to exactly -8%; "
        "otherwise terminal W4 is used. This is optimistic no-slippage execution and "
        "should be read as a scenario, not realized fills.",
        "",
    ]

    if capacity_stop8.empty:
        lines.append("No Stop8 execution capacity diagnostics.")
    else:
        lines += [
            "| Policy | Scope | Weeks | Stop8-exec mean | Stop8-exec median | Mean Δ vs B0 | Median Δ | Beat B0 | 95% mean-edge CI | Worst | P10 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
        ]
        for _, row in capacity_stop8.iterrows():
            lines.append(
                f"| {row['policy_id']} | {row['scope']} | {int(row['support_weeks'])} | "
                f"{_fmt(row['mean_return'], 2, '%')} | {_fmt(row['median_return'], 2, '%')} | "
                f"{_fmt(row['mean_spread_vs_b0'], 2, 'pp')} | "
                f"{_fmt(row['median_spread_vs_b0'], 2, 'pp')} | "
                f"{_rate(row['beat_b0_rate'])} | "
                f"[{_fmt(row['spread_ci_low'], 2, 'pp')}, {_fmt(row['spread_ci_high'], 2, 'pp')}] | "
                f"{_fmt(row['worst_return'], 2, '%')} | {_fmt(row['p10_return'], 2, '%')} |"
            )

    lines += [
        "",
        "Portfolio-level any-stop/≤-8-week deltas rise mechanically when more positions are held; "
        "they are exposure diagnostics, not proof that each added name is intrinsically riskier. "
        "Per-pick Stop8 and terminal-left-tail rates above are the quality comparison.",
        "",
        "Interpretation rule: do not change Production to always-3 merely because fixed-3 random has a "
        "higher mean. The minimal-change candidate is B0_FILL3_SINGLE_REJECT; only a coherent gain "
        "on underfilled weeks without material per-pick or portfolio left-tail deterioration would justify future shadow testing.",
        "",
        "## 5. Eligibility gate quality",
        "",
        f"- Strict support: **{gate['support_weeks']} weeks**",
        f"- Raw candidate events: **{gate['raw_candidate_events']}**",
        f"- Eligible candidate events: **{gate['eligible_candidate_events']}**",
        f"- Acceptance rate: **{_rate(gate['accept_rate'])}**",
        f"- Top20 winner retention: **{_rate(gate['winner_retention_rate'])}**",
        f"- Winner enrichment vs random acceptance: "
        f"**{_fmt(gate['winner_enrichment_vs_random_selectivity'], 2)}x**",
        f"- Bottom20 loser retention: **{_rate(gate['bottom_loser_retention_rate'])}**",
        f"- Loser retention vs random acceptance: "
        f"**{_fmt(gate['loser_retention_vs_random_selectivity'], 2)}x**",
        f"- Final B0 Top3 winner-capture rate: **{_rate(gate['b0_winner_capture_rate'])}**",
        f"- Winner-capture enrichment vs Matched-N random: "
        f"**{_fmt(gate['b0_winner_capture_enrichment_vs_matched_n_random'], 2)}x**",
        f"- Winner-capture enrichment vs mechanical fixed-3 random: "
        f"**{_fmt(gate['b0_winner_capture_enrichment_vs_fixed3_random'], 2)}x**",
        f"- Mean eligible-minus-rejected W4 lift: **{_fmt(gate['mean_gate_lift'], 2, 'pp')}**",
        f"- Median weekly gate lift: **{_fmt(gate['median_gate_lift'], 2, 'pp')}**",
        "",
        "Low recall must therefore be interpreted together with the gate's acceptance rate. "
        "A gate accepting only a small fraction of the universe cannot be judged solely by the percentage "
        "of all future winners that it rejects.",
        "",
        "## 6. Reject-reason audit — exclusive attribution separated from overlap",
        "",
        "### Exclusive single-reason rejects",
        "",
    ]

    if exclusive_reject.empty:
        lines.append("No exclusive reject events.")
    else:
        lines += [
            "| Sole reject reason | Events | Weeks | Mean W4 | Median W4 | Positive | Stop8 | Top20 winner | >=20% winner |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for _, row in exclusive_reject.iterrows():
            lines.append(
                f"| {row['exclusive_reason']} | {int(row['candidate_events'])} | {int(row['weeks'])} | "
                f"{_fmt(row['mean_w4'], 2, '%')} | {_fmt(row['median_w4'], 2, '%')} | "
                f"{_rate(row['positive_rate'])} | {_rate(row['stop8_rate'])} | "
                f"{_rate(row['top20_winner_rate'])} | {_rate(row['big_winner_rate'])} |"
            )

    lines += [
        "",
        "### Overlapping reason labels — descriptive only, not causal",
        "",
    ]
    if overlap_reject.empty:
        lines.append("No overlapping reason labels.")
    else:
        lines += [
            "| Reason label | Events | Multi-reason share | Mean W4 | Median W4 | Top20 winner |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for _, row in overlap_reject.iterrows():
            lines.append(
                f"| {row['reason']} | {int(row['label_events'])} | "
                f"{_rate(row['multi_reason_rate'])} | {_fmt(row['mean_w4'], 2, '%')} | "
                f"{_fmt(row['median_w4'], 2, '%')} | {_rate(row['top20_winner_rate'])} |"
            )

    lines += [
        "",
        "Reason-combination counts are separately materialized; overlapping labels must never be used to "
        "claim that one gate caused the rejection outcome.",
        "",
        "## 7. Fine-ranking information",
        "",
        f"- Mature eligible-universe Spearman support: **{info['support_weeks']} weeks**",
        f"- Mean weekly Spearman (-rank vs W4): **{_fmt(info['weekly_spearman_mean'], 3)}**",
        f"- Median Spearman: **{_fmt(info['weekly_spearman_median'], 3)}**",
        f"- Positive-Spearman weeks: **{_rate(info['positive_spearman_week_rate'])}**",
        f"- Selected mean minus all-eligible mean: "
        f"**{_fmt(info['selected_minus_eligible_mean'], 2, 'pp')}** mean / "
        f"**{_fmt(info['selected_minus_eligible_median'], 2, 'pp')}** median",
        "",
        "Rank buckets:",
        "",
    ]

    if rank_buckets.empty:
        lines.append("No rank-bucket rows.")
    else:
        lines += [
            "| Rank bucket | Rows | Weeks | Mean W4 | Median W4 | Positive | Stop8 |",
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
        "A strong Top bucket does not imply globally monotonic fine ranking; the entire bucket curve and "
        "Spearman distribution must be considered.",
        "",
        "## 8. Simple raw-PIT baselines — tradable next-open outcome",
        "",
    ]
    if simple.empty:
        lines.append("No full-capacity common-support simple baselines.")
    else:
        lines += [
            "| Baseline | Weeks | Mean W4 | Median W4 | Mean Δ | Median Δ | Beat B0 | 95% mean-edge CI | Mean w/o best1 | Mean w/o best2 | Stop exposure Δ | Any Stop/≤-8 week Δ |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
        for _, row in simple.iterrows():
            lines.append(
                f"| {row['baseline']} | {int(row['support_weeks'])} | "
                f"{_fmt(row['mean_return'], 2, '%')} | {_fmt(row['median_return'], 2, '%')} | "
                f"{_fmt(row['mean_spread_vs_b0'], 2, 'pp')} | "
                f"{_fmt(row['median_spread_vs_b0'], 2, 'pp')} | "
                f"{_rate(row['beat_b0_rate'])} | "
                f"[{_fmt(row['spread_ci_low'], 2, 'pp')}, {_fmt(row['spread_ci_high'], 2, 'pp')}] | "
                f"{_fmt(row['mean_without_best1'], 2, 'pp')} | "
                f"{_fmt(row['mean_without_best2'], 2, 'pp')} | "
                f"{_fmt(row['stop8_exposure_delta_pp'], 2, 'pp')} | "
                f"{_fmt(row['any_stop_or_le8_week_delta_pp'], 2, 'pp')} |"
            )

    lines += [
        "",
        "### Idealized Stop8 execution scenario for simple baselines",
        "",
    ]

    if simple_stop8.empty:
        lines.append("No Stop8 execution simple-baseline diagnostics.")
    else:
        lines += [
            "| Baseline | Weeks | Stop8-exec mean | Stop8-exec median | Mean Δ vs B0 | Median Δ | Beat B0 | 95% mean-edge CI | Worst | P10 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
        ]
        for _, row in simple_stop8.iterrows():
            lines.append(
                f"| {row['baseline']} | {int(row['support_weeks'])} | "
                f"{_fmt(row['mean_return'], 2, '%')} | {_fmt(row['median_return'], 2, '%')} | "
                f"{_fmt(row['mean_spread_vs_b0'], 2, 'pp')} | "
                f"{_fmt(row['median_spread_vs_b0'], 2, 'pp')} | "
                f"{_rate(row['beat_b0_rate'])} | "
                f"[{_fmt(row['spread_ci_low'], 2, 'pp')}, {_fmt(row['spread_ci_high'], 2, 'pp')}] | "
                f"{_fmt(row['worst_return'], 2, '%')} | {_fmt(row['p10_return'], 2, '%')} |"
            )

    lines += [
        "",
        "Relative-SPY 20-session momentum is retained as a PIT diagnostic field, not as an "
        "independent Top3 baseline: subtracting the same SPY return from every candidate in "
        "one snapshot cannot change that snapshot's cross-sectional momentum ranking.",
        "",
        "Large mean gains with flat/negative medians or strong best-week dependence are treated as "
        "right-tail hypotheses, not as proof of stable superiority.",
        "",
        "### Momentum20 gate-location diagnostic",
        "",
        "This answers whether the raw momentum baseline's incremental names were already inside "
        "the B0 eligible universe or mainly outside the hard gates.",
        "",
    ]

    if momentum_gate.empty:
        lines.append("No momentum gate diagnostics.")
    else:
        lines += [
            "| Cohort | Picks | Share of momentum picks | Selected by B0 | Mean W4 | Median W4 | Positive | Stop8/pick | Final W4≤-8 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for _, row in momentum_gate.iterrows():
            lines.append(
                f"| {row['cohort']} | {int(row['picks'])} | "
                f"{_rate(row['share_of_momentum_picks'])} | "
                f"{_rate(row['selected_by_b0_rate'])} | "
                f"{_fmt(row['mean_w4'], 2, '%')} | {_fmt(row['median_w4'], 2, '%')} | "
                f"{_rate(row['positive_rate'])} | {_rate(row['stop8_rate'])} | "
                f"{_rate(row['terminal_le_minus8_rate'])} |"
            )

    if not momentum_gate_reasons.empty:
        lines += [
            "",
            "Gate-outside momentum picks by exact reject reason:",
            "",
            "| Reject reason | Picks | Share of gate-outside momentum | Mean W4 | Median W4 | Positive | Stop8/pick | Final W4≤-8 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for _, row in momentum_gate_reasons.iterrows():
            lines.append(
                f"| {row['reject_reason']} | {int(row['picks'])} | "
                f"{_rate(row['share_of_gate_outside_momentum'])} | "
                f"{_fmt(row['mean_w4'], 2, '%')} | {_fmt(row['median_w4'], 2, '%')} | "
                f"{_rate(row['positive_rate'])} | {_rate(row['stop8_rate'])} | "
                f"{_rate(row['terminal_le_minus8_rate'])} |"
            )

    lines += [
        "",
        "## 9. SPY / QQQ benchmark — Yahoo, same tradable clock",
        "",
    ]
    if market.empty:
        lines.append("No market benchmark support.")
    else:
        lines += [
            "| Benchmark | Weeks | Benchmark mean | B0 capital mean | B0 vs fully invested | B0 vs exposure-matched | Active-pick selection spread | Full-spread 95% CI |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for _, row in market.iterrows():
            lines.append(
                f"| {row['benchmark']} | {int(row.get('support_weeks', 0))} | "
                f"{_fmt(row.get('benchmark_mean'), 2, '%')} | "
                f"{_fmt(row.get('b0_capital_mean'), 2, '%')} | "
                f"{_fmt(row.get('full_exposure_mean_spread'), 2, 'pp')} | "
                f"{_fmt(row.get('exposure_matched_mean_spread'), 2, 'pp')} | "
                f"{_fmt(row.get('active_pick_selection_mean_spread'), 2, 'pp')} | "
                f"[{_fmt(row.get('full_spread_ci_low'), 2, 'pp')}, "
                f"{_fmt(row.get('full_spread_ci_high'), 2, 'pp')}] |"
            )

    lines += [
        "",
        "## 10. Non-overlap stability — ranking, raw fixed3, and momentum20",
        "",
    ]
    if nonoverlap.empty:
        lines.append("No non-overlap rows.")
    else:
        lines += [
            "| Comparison | Offset | Weeks | Value mean | Benchmark mean | Spread mean | Spread median |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for _, row in nonoverlap.iterrows():
            lines.append(
                f"| {row['comparison']} | {int(row['offset'])} | {int(row['weeks'])} | "
                f"{_fmt(row.get('value_mean'), 2, '%')} | "
                f"{_fmt(row.get('benchmark_mean'), 2, '%')} | "
                f"{_fmt(row.get('spread_mean'), 2, 'pp')} | "
                f"{_fmt(row.get('spread_median'), 2, 'pp')} |"
            )

    lines += [
        "",
        "## Evidence boundary",
        "",
        health["evidence_boundary"],
        "",
        "This report is a retrospective measurement instrument. It can identify where B0 appears strong "
        "or weak and which minimal counterfactual deserves future shadow testing; it cannot convert "
        "historical reuse into untouched OOS proof.",
        "",
        "## Provenance",
        "",
        f"- source_git_sha: {manifest['source_git_sha']}",
        f"- protocol_version: {manifest['protocol_version']}",
        f"- production_b0_hash: {manifest['production_b0_hash']}",
        f"- panel_hash: {manifest['panel_hash']}",
        f"- base_price_cache_hash: {manifest['base_price_cache_hash']}",
        f"- yahoo_supplement_hash: {manifest['yahoo_supplement_hash']}",
        f"- audit_as_of_date: {manifest['audit_as_of_date']}",
        "",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
