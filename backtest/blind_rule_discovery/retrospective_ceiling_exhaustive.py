"""R3 retrospective empirical-ceiling search with exhaustive two-condition interactions.

This is an explicitly coupled historical study. It is NOT a sealed-holdout or
canonical blind experiment. R1/R2 periods are known history here.

R3 fixes four limitations of R2:
- every generated quantile condition participates in exhaustive pair search;
- rules need enough evaluable quarters, not merely active quarters;
- fixed-rule drop-one-quarter sensitivity is separated from true held-quarter re-search;
- rolling summaries count zero-selection folds and report a pooled contemporaneous baseline.

Triples and DNF remain deterministic beam expansions after the exact pair layer.
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .dataset import DISCOVERY_FEATURE_ALLOWLIST, build_blind_dataset, load_replay_candidates
from .outcomes import OutcomeConfig, load_price_pickle, restrict_to_mature_outcome_quarters
from .pipeline_contract import validate_replay_preflight
from .retrospective_ceiling import (
    DEFAULT_QUANTILES,
    ROLLING_QUANTILES,
    PARETO_METRICS,
    SCORE_WEIGHTS,
    _apply_rule_to_frame,
    _canonical_rule,
    _condition_key,
    _condition_mask,
    _conditions_compatible,
    _dedupe_records,
    _public_table,
    _record_for_rule,
    _rule_conditions,
    _rule_key,
    _write_best_rule,
    generate_conditions,
    leave_one_quarter_out,
    pareto_frontier,
    score_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--daily-pkl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--spy-code", default="SPY")
    parser.add_argument("--beam-width", type=int, default=40)
    parser.add_argument("--max-conditions", type=int, default=3)
    parser.add_argument("--max-clauses", type=int, default=3)
    parser.add_argument("--dnf-clause-pool", type=int, default=20)
    parser.add_argument("--min-selected", type=int, default=40)
    parser.add_argument("--min-resolved", type=int, default=30)
    parser.add_argument("--min-active-quarters", type=int, default=5)
    parser.add_argument("--min-evaluated-quarters", type=int, default=5)
    parser.add_argument("--min-evaluated-fraction", type=float, default=0.50)
    parser.add_argument("--min-resolved-per-quarter", type=int, default=5)
    parser.add_argument("--rolling-min-train-quarters", type=int, default=6)
    parser.add_argument("--rolling-beam-width", type=int, default=24)
    parser.add_argument("--skip-rolling", action="store_true")
    parser.add_argument("--skip-loqo-research", action="store_true")
    return parser.parse_args()


def _finite_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)


@dataclass(frozen=True)
class EvaluationContext:
    row_count: int
    resolved: np.ndarray
    winner: np.ndarray
    excess_12w: np.ndarray
    mae_12w: np.ndarray
    mfe_12w: np.ndarray
    quarters: tuple[str, ...]
    quarter_masks: tuple[np.ndarray, ...]
    universe_winner_rate: float | None
    quarter_universe_winner_rates: tuple[float | None, ...]

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "EvaluationContext":
        primary = frame["Y_primary"].astype(str).to_numpy()
        resolved = np.isin(primary, ["winner", "loser"])
        winner = primary == "winner"
        quarter_values = frame["period_quarter"].astype(str).to_numpy()
        quarters = tuple(sorted(pd.unique(quarter_values)))
        quarter_masks = tuple(quarter_values == quarter for quarter in quarters)
        universe_wr = float(np.mean(winner[resolved])) if resolved.any() else None
        quarter_wr: list[float | None] = []
        for q_mask in quarter_masks:
            q_resolved = q_mask & resolved
            quarter_wr.append(float(np.mean(winner[q_resolved])) if q_resolved.any() else None)
        return cls(
            row_count=len(frame),
            resolved=resolved,
            winner=winner,
            excess_12w=_finite_values(frame, "Y_12w_excess"),
            mae_12w=_finite_values(frame, "Y_mae_12w"),
            mfe_12w=_finite_values(frame, "Y_mfe_12w"),
            quarters=quarters,
            quarter_masks=quarter_masks,
            universe_winner_rate=universe_wr,
            quarter_universe_winner_rates=tuple(quarter_wr),
        )

    def support(self, mask: np.ndarray, *, min_resolved_per_quarter: int) -> dict[str, int | float]:
        selected_n = int(np.count_nonzero(mask))
        resolved_selected = mask & self.resolved
        resolved_n = int(np.count_nonzero(resolved_selected))
        active_quarters = 0
        evaluated_quarters = 0
        for q_mask in self.quarter_masks:
            if np.any(mask & q_mask):
                active_quarters += 1
            if int(np.count_nonzero(resolved_selected & q_mask)) >= min_resolved_per_quarter:
                evaluated_quarters += 1
        fraction = evaluated_quarters / active_quarters if active_quarters else 0.0
        return {
            "selected_n": selected_n,
            "resolved_n": resolved_n,
            "active_quarters": active_quarters,
            "evaluated_quarters": evaluated_quarters,
            "evaluated_quarter_fraction": fraction,
        }

    def evaluate(self, mask: np.ndarray, *, min_resolved_per_quarter: int) -> dict[str, Any]:
        if len(mask) != self.row_count:
            raise ValueError("mask length differs from evaluation context")
        support = self.support(mask, min_resolved_per_quarter=min_resolved_per_quarter)
        selected_n = int(support["selected_n"])
        resolved_n = int(support["resolved_n"])
        resolved_selected = mask & self.resolved
        winner_n = int(np.count_nonzero(resolved_selected & self.winner))
        winner_rate = winner_n / resolved_n if resolved_n else None
        lift = (
            winner_rate - self.universe_winner_rate
            if winner_rate is not None and self.universe_winner_rate is not None
            else None
        )

        def q(values: np.ndarray, quantile: float) -> float | None:
            selected = values[mask]
            selected = selected[np.isfinite(selected)]
            return float(np.quantile(selected, quantile)) if selected.size else None

        quarter_lifts: list[float] = []
        outperform_quarters = 0
        for q_mask, universe_wr in zip(self.quarter_masks, self.quarter_universe_winner_rates):
            q_selected_resolved = resolved_selected & q_mask
            q_n = int(np.count_nonzero(q_selected_resolved))
            if q_n < min_resolved_per_quarter or universe_wr is None:
                continue
            selected_wr = float(np.mean(self.winner[q_selected_resolved]))
            q_lift = selected_wr - universe_wr
            quarter_lifts.append(q_lift)
            if q_lift > 0:
                outperform_quarters += 1

        evaluated = int(support["evaluated_quarters"])
        return {
            "selected_n": selected_n,
            "coverage": selected_n / self.row_count if self.row_count else 0.0,
            "resolved_n": resolved_n,
            "winner_n": winner_n,
            "resolved_winner_rate": winner_rate,
            "universe_resolved_winner_rate": self.universe_winner_rate,
            "winner_rate_lift": lift,
            "excess_12w_p25": q(self.excess_12w, 0.25),
            "excess_12w_p50": q(self.excess_12w, 0.50),
            "excess_12w_p75": q(self.excess_12w, 0.75),
            "mae_12w_p50": q(self.mae_12w, 0.50),
            "mfe_12w_p50": q(self.mfe_12w, 0.50),
            "active_quarters": int(support["active_quarters"]),
            "evaluated_quarters": evaluated,
            "evaluated_quarter_fraction": float(support["evaluated_quarter_fraction"]),
            "quarter_outperform_fraction": outperform_quarters / evaluated if evaluated else None,
            "median_quarter_lift": float(np.median(quarter_lifts)) if quarter_lifts else None,
            "worst_quarter_lift": min(quarter_lifts) if quarter_lifts else None,
        }


def _supported(
    metrics: Mapping[str, Any],
    *,
    min_selected: int,
    min_resolved: int,
    min_active_quarters: int,
    min_evaluated_quarters: int,
    min_evaluated_fraction: float,
) -> bool:
    active = int(metrics["active_quarters"])
    evaluated = int(metrics["evaluated_quarters"])
    fraction = evaluated / active if active else 0.0
    return (
        int(metrics["selected_n"]) >= min_selected
        and int(metrics["resolved_n"]) >= min_resolved
        and active >= min_active_quarters
        and evaluated >= min_evaluated_quarters
        and fraction >= min_evaluated_fraction
    )


def _quick_supported(
    context: EvaluationContext,
    mask: np.ndarray,
    *,
    min_selected: int,
    min_resolved: int,
    min_active_quarters: int,
    min_evaluated_quarters: int,
    min_evaluated_fraction: float,
    min_resolved_per_quarter: int,
) -> bool:
    support = context.support(mask, min_resolved_per_quarter=min_resolved_per_quarter)
    return _supported(
        support,
        min_selected=min_selected,
        min_resolved=min_resolved,
        min_active_quarters=min_active_quarters,
        min_evaluated_quarters=min_evaluated_quarters,
        min_evaluated_fraction=min_evaluated_fraction,
    )


def exhaustive_pair_search_rules(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    beam_width: int = 40,
    max_conditions: int = 3,
    max_clauses: int = 3,
    dnf_clause_pool: int = 20,
    min_selected: int = 40,
    min_resolved: int = 30,
    min_active_quarters: int = 5,
    min_evaluated_quarters: int = 5,
    min_evaluated_fraction: float = 0.50,
    min_resolved_per_quarter: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Exhaust every generated pair; use deterministic beam only for depth >= 3."""
    if frame.empty:
        raise ValueError("retrospective search frame is empty")
    if not 1 <= max_conditions <= 3:
        raise ValueError("max_conditions must be within 1..3")
    if not 1 <= max_clauses <= 3:
        raise ValueError("max_clauses must be within 1..3")
    if not 0.0 <= min_evaluated_fraction <= 1.0:
        raise ValueError("min_evaluated_fraction must be within 0..1")

    conditions = generate_conditions(frame, feature_columns, quantiles=quantiles)
    if not conditions:
        raise ValueError("no usable search conditions")
    condition_masks = {_condition_key(c): _condition_mask(frame, c) for c in conditions}
    context = EvaluationContext.from_frame(frame)

    def accepted(rule: dict[str, Any], mask: np.ndarray) -> dict[str, Any] | None:
        if not _quick_supported(
            context,
            mask,
            min_selected=min_selected,
            min_resolved=min_resolved,
            min_active_quarters=min_active_quarters,
            min_evaluated_quarters=min_evaluated_quarters,
            min_evaluated_fraction=min_evaluated_fraction,
            min_resolved_per_quarter=min_resolved_per_quarter,
        ):
            return None
        metrics = context.evaluate(mask, min_resolved_per_quarter=min_resolved_per_quarter)
        return _record_for_rule(rule, mask, metrics)

    single_records: list[dict[str, Any]] = []
    for condition in conditions:
        rule = _canonical_rule([[condition]])
        record = accepted(rule, condition_masks[_condition_key(condition)])
        if record is not None:
            single_records.append(record)
    single_scored = score_records(single_records)
    if single_scored.empty:
        raise ValueError("no supported single-condition rule")

    pair_records: list[dict[str, Any]] = []
    exact_pair_count = 0
    if max_conditions >= 2:
        for left, right in itertools.combinations(conditions, 2):
            merged = [left, right]
            if not _conditions_compatible(merged):
                continue
            exact_pair_count += 1
            mask = condition_masks[_condition_key(left)] & condition_masks[_condition_key(right)]
            rule = _canonical_rule([merged])
            record = accepted(rule, mask)
            if record is not None:
                pair_records.append(record)
    pair_scored = score_records(pair_records)
    if max_conditions >= 2 and pair_scored.empty:
        raise ValueError("exhaustive pair search produced no supported pair")

    triple_records: list[dict[str, Any]] = []
    if max_conditions >= 3 and not pair_scored.empty:
        seen: set[str] = set()
        for _, parent in pair_scored.head(beam_width).iterrows():
            parent_conditions = _rule_conditions(parent["_rule"])
            for condition in conditions:
                merged = [*parent_conditions, condition]
                if not _conditions_compatible(merged):
                    continue
                rule = _canonical_rule([merged])
                key = _rule_key(rule)
                if key in seen:
                    continue
                seen.add(key)
                mask = np.ones(len(frame), dtype=bool)
                for item in merged:
                    mask &= condition_masks[_condition_key(item)]
                record = accepted(rule, mask)
                if record is not None:
                    triple_records.append(record)
    triple_scored = score_records(triple_records)

    all_records: list[dict[str, Any]] = []
    all_records.extend(single_records)
    all_records.extend(pair_records)
    all_records.extend(triple_records)
    conjunctions = score_records(_dedupe_records(all_records))
    if conjunctions.empty:
        raise ValueError("R3 conjunction search produced no supported rule")

    dnf_records: list[dict[str, Any]] = []
    if max_clauses > 1 and dnf_clause_pool >= 2:
        clause_pool = conjunctions.loc[conjunctions["clause_count"] == 1].head(dnf_clause_pool)
        clauses = [row["_rule"]["clauses"][0]["all"] for _, row in clause_pool.iterrows()]
        for clause_count in range(2, max_clauses + 1):
            for combo in itertools.combinations(clauses, clause_count):
                if sum(len(clause) for clause in combo) > 6:
                    continue
                rule = _canonical_rule(combo)
                mask = _apply_rule_to_frame(rule, frame)
                record = accepted(rule, mask)
                if record is not None:
                    dnf_records.append(record)

    scored = score_records(_dedupe_records([*all_records, *dnf_records]))
    pareto = pareto_frontier(scored)
    audit = {
        "generated_condition_count": len(conditions),
        "exact_compatible_pair_count": exact_pair_count,
        "supported_pair_count": len(pair_records),
        "triple_candidate_count": len(triple_records),
        "dnf_candidate_count": len(dnf_records),
    }
    return scored, pareto, audit


def drop_one_quarter_sensitivity(
    frame: pd.DataFrame,
    scored: pd.DataFrame,
    *,
    top_n: int,
    min_resolved_per_quarter: int,
) -> pd.DataFrame:
    """Fixed-rule sensitivity only; deliberately not called LOQ validation."""
    return leave_one_quarter_out(
        frame,
        scored,
        top_n=top_n,
        min_resolved_per_quarter=min_resolved_per_quarter,
    )


def leave_one_quarter_out_research(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    beam_width: int = 16,
    min_selected: int = 30,
    min_resolved: int = 20,
    min_resolved_per_quarter: int = 4,
) -> pd.DataFrame:
    """Re-search on all other quarters, freeze the best rule, then test the held quarter."""
    quarters = sorted(frame["period_quarter"].astype(str).unique())
    rows: list[dict[str, Any]] = []
    for held_quarter in quarters:
        train = frame.loc[frame["period_quarter"].astype(str) != held_quarter].reset_index(drop=True)
        test = frame.loc[frame["period_quarter"].astype(str) == held_quarter].reset_index(drop=True)
        train_quarters = sorted(train["period_quarter"].astype(str).unique())
        min_active = min(4, max(2, len(train_quarters) // 2))
        scored, _, _ = exhaustive_pair_search_rules(
            train,
            feature_columns,
            quantiles=ROLLING_QUANTILES,
            beam_width=beam_width,
            max_conditions=2,
            max_clauses=1,
            dnf_clause_pool=0,
            min_selected=min_selected,
            min_resolved=min_resolved,
            min_active_quarters=min_active,
            min_evaluated_quarters=min_active,
            min_evaluated_fraction=0.50,
            min_resolved_per_quarter=min_resolved_per_quarter,
        )
        best = scored.iloc[0]
        rule = best["_rule"]
        test_context = EvaluationContext.from_frame(test)
        test_mask = _apply_rule_to_frame(rule, test)
        test_metrics = test_context.evaluate(test_mask, min_resolved_per_quarter=min_resolved_per_quarter)
        rows.append(
            {
                "held_quarter": held_quarter,
                "train_quarter_count": len(train_quarters),
                "train_first_quarter": train_quarters[0],
                "train_last_quarter": train_quarters[-1],
                "held_quarter_in_train": held_quarter in train_quarters,
                "rule_json": best["rule_json"],
                "train_robust_score": best["robust_score"],
                "train_winner_rate_lift": best["winner_rate_lift"],
                "train_excess_12w_p50": best["excess_12w_p50"],
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
        )
    return pd.DataFrame(rows)


def rolling_walk_forward_research(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    min_train_quarters: int = 6,
    beam_width: int = 24,
    min_selected: int = 30,
    min_resolved: int = 20,
    min_resolved_per_quarter: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expanding-window exact-pair re-search on past quarters, then next-quarter test."""
    quarters = sorted(frame["period_quarter"].astype(str).unique())
    if len(quarters) <= min_train_quarters:
        raise ValueError("not enough quarters for rolling walk-forward")
    rows: list[dict[str, Any]] = []
    usage: dict[str, int] = {}
    for test_index in range(min_train_quarters, len(quarters)):
        train_quarters = quarters[:test_index]
        test_quarter = quarters[test_index]
        train = frame.loc[frame["period_quarter"].astype(str).isin(train_quarters)].reset_index(drop=True)
        test = frame.loc[frame["period_quarter"].astype(str) == test_quarter].reset_index(drop=True)
        min_active = min(4, max(2, len(train_quarters) // 2))
        scored, _, _ = exhaustive_pair_search_rules(
            train,
            feature_columns,
            quantiles=ROLLING_QUANTILES,
            beam_width=beam_width,
            max_conditions=2,
            max_clauses=1,
            dnf_clause_pool=0,
            min_selected=min_selected,
            min_resolved=min_resolved,
            min_active_quarters=min_active,
            min_evaluated_quarters=min_active,
            min_evaluated_fraction=0.50,
            min_resolved_per_quarter=min_resolved_per_quarter,
        )
        best = scored.iloc[0]
        rule = best["_rule"]
        for condition in _rule_conditions(rule):
            feature = str(condition["feature"])
            usage[feature] = usage.get(feature, 0) + 1

        test_context = EvaluationContext.from_frame(test)
        test_mask = _apply_rule_to_frame(rule, test)
        test_metrics = test_context.evaluate(test_mask, min_resolved_per_quarter=min_resolved_per_quarter)
        baseline_metrics = test_context.evaluate(
            np.ones(len(test), dtype=bool),
            min_resolved_per_quarter=min_resolved_per_quarter,
        )
        rows.append(
            {
                "test_quarter": test_quarter,
                "train_start_quarter": train_quarters[0],
                "train_end_quarter": train_quarters[-1],
                "train_quarter_count": len(train_quarters),
                "test_quarter_in_train": test_quarter in train_quarters,
                "rule_json": best["rule_json"],
                "train_robust_score": best["robust_score"],
                "train_winner_rate_lift": best["winner_rate_lift"],
                "train_excess_12w_p50": best["excess_12w_p50"],
                "test_universe_resolved_n": baseline_metrics["resolved_n"],
                "test_universe_winner_n": baseline_metrics["winner_n"],
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
        )

    usage_frame = pd.DataFrame(
        [
            {"feature": feature, "rolling_fold_count": count, "fraction": count / len(rows)}
            for feature, count in usage.items()
        ]
    ).sort_values(["rolling_fold_count", "feature"], ascending=[False, True]).reset_index(drop=True)
    return pd.DataFrame(rows), usage_frame


def summarize_rolling(rolling: pd.DataFrame) -> dict[str, Any]:
    if rolling.empty:
        return {}
    selected = pd.to_numeric(rolling["test_selected_n"], errors="coerce").fillna(0)
    resolved = pd.to_numeric(rolling["test_resolved_n"], errors="coerce").fillna(0)
    winners = pd.to_numeric(rolling["test_winner_n"], errors="coerce").fillna(0)
    universe_resolved = pd.to_numeric(rolling["test_universe_resolved_n"], errors="coerce").fillna(0)
    universe_winners = pd.to_numeric(rolling["test_universe_winner_n"], errors="coerce").fillna(0)
    lift = pd.to_numeric(rolling["test_winner_rate_lift"], errors="coerce")
    finite_lift = lift.dropna()
    excess = pd.to_numeric(rolling["test_excess_12w_p50"], errors="coerce").dropna()
    resolved_total = float(resolved.sum())
    winner_total = float(winners.sum())
    universe_resolved_total = float(universe_resolved.sum())
    universe_winner_total = float(universe_winners.sum())
    selected_wr = winner_total / resolved_total if resolved_total else None
    universe_wr = universe_winner_total / universe_resolved_total if universe_resolved_total else None
    positive_count = int((finite_lift > 0).sum())
    zero_selection_count = int((selected == 0).sum())
    return {
        "folds": int(len(rolling)),
        "evaluable_lift_folds": int(len(finite_lift)),
        "zero_selection_fold_count": zero_selection_count,
        "zero_selection_fold_fraction": zero_selection_count / len(rolling),
        "selected_n_total": int(selected.sum()),
        "resolved_n_total": int(resolved_total),
        "winner_n_total": int(winner_total),
        "pooled_resolved_winner_rate": selected_wr,
        "pooled_universe_resolved_n": int(universe_resolved_total),
        "pooled_universe_winner_n": int(universe_winner_total),
        "pooled_universe_resolved_winner_rate": universe_wr,
        "pooled_winner_rate_lift": (
            selected_wr - universe_wr if selected_wr is not None and universe_wr is not None else None
        ),
        "positive_lift_evaluable_fraction": positive_count / len(finite_lift) if len(finite_lift) else None,
        "positive_lift_all_fold_fraction": positive_count / len(rolling),
        "winner_rate_lift_p50_evaluable": float(finite_lift.median()) if len(finite_lift) else None,
        "winner_rate_lift_min_evaluable": float(finite_lift.min()) if len(finite_lift) else None,
        "excess_12w_p50_across_nonempty_folds": float(excess.median()) if len(excess) else None,
    }


def main() -> int:
    args = parse_args()
    r1_root = Path("backtest/blind_rule_discovery/output/rd_agent_run_01").resolve()
    r2_root = Path("backtest/blind_rule_discovery/output/retrospective_ceiling_r2").resolve()
    if args.output_root.resolve() in {r1_root, r2_root}:
        raise RuntimeError("historical R1/R2 outputs are immutable and may not be reused")
    args.output_root.mkdir(parents=True, exist_ok=True)

    provenance = validate_replay_preflight(
        args.replay_root,
        daily_pkl=args.daily_pkl,
        required_quarters=12,
    )
    prices = load_price_pickle(args.daily_pkl, require_adjusted=True)
    if args.spy_code not in prices:
        raise KeyError(f"benchmark {args.spy_code!r} missing from daily price bundle")

    candidates_all = load_replay_candidates(args.replay_root)
    config = OutcomeConfig()
    candidates, immature_rows, maturity_cutoff = restrict_to_mature_outcome_quarters(
        candidates_all,
        prices[args.spy_code],
        minimum_sessions=config.minimum_sessions + config.entry_window_sessions,
    )
    agent_df, feature_map, reviewer = build_blind_dataset(
        candidates,
        prices,
        prices[args.spy_code],
        config=config,
    )
    frame = agent_df.rename(columns=feature_map).copy()
    feature_columns = [
        column
        for column in [*DISCOVERY_FEATURE_ALLOWLIST, *sorted(c for c in frame.columns if c.startswith("M_"))]
        if column in frame.columns
    ]
    if not feature_columns:
        raise ValueError("no retrospective search features available")

    scored, pareto, search_audit = exhaustive_pair_search_rules(
        frame,
        feature_columns,
        beam_width=args.beam_width,
        max_conditions=args.max_conditions,
        max_clauses=args.max_clauses,
        dnf_clause_pool=args.dnf_clause_pool,
        min_selected=args.min_selected,
        min_resolved=args.min_resolved,
        min_active_quarters=args.min_active_quarters,
        min_evaluated_quarters=args.min_evaluated_quarters,
        min_evaluated_fraction=args.min_evaluated_fraction,
        min_resolved_per_quarter=args.min_resolved_per_quarter,
    )
    _public_table(scored).to_csv(args.output_root / "candidate_rules.csv", index=False)
    _public_table(pareto).to_csv(args.output_root / "pareto_frontier.csv", index=False)
    _write_best_rule(args.output_root / "best_rule.json", scored.iloc[0])

    fixed_sensitivity = drop_one_quarter_sensitivity(
        frame,
        scored,
        top_n=20,
        min_resolved_per_quarter=args.min_resolved_per_quarter,
    )
    fixed_sensitivity.to_csv(args.output_root / "drop_one_quarter_sensitivity.csv", index=False)

    loqo_research = pd.DataFrame()
    if not args.skip_loqo_research:
        loqo_research = leave_one_quarter_out_research(
            frame,
            feature_columns,
            beam_width=max(12, min(args.rolling_beam_width, 20)),
            min_selected=max(20, min(args.min_selected, 30)),
            min_resolved=max(15, min(args.min_resolved, 20)),
            min_resolved_per_quarter=max(3, min(args.min_resolved_per_quarter, 4)),
        )
        loqo_research.to_csv(args.output_root / "leave_one_quarter_out_research.csv", index=False)

    rolling = pd.DataFrame()
    rolling_usage = pd.DataFrame()
    if not args.skip_rolling:
        rolling, rolling_usage = rolling_walk_forward_research(
            frame,
            feature_columns,
            min_train_quarters=args.rolling_min_train_quarters,
            beam_width=args.rolling_beam_width,
            min_selected=max(20, min(args.min_selected, 30)),
            min_resolved=max(15, min(args.min_resolved, 20)),
            min_resolved_per_quarter=max(3, min(args.min_resolved_per_quarter, 4)),
        )
        rolling.to_csv(args.output_root / "rolling_walk_forward.csv", index=False)
        rolling_usage.to_csv(args.output_root / "rolling_feature_frequency.csv", index=False)

    baseline_context = EvaluationContext.from_frame(frame)
    baseline = baseline_context.evaluate(
        np.ones(len(frame), dtype=bool),
        min_resolved_per_quarter=args.min_resolved_per_quarter,
    )
    censor_reasons = (
        reviewer.loc[~reviewer["usable"].fillna(False), "reason"]
        .fillna("unknown")
        .value_counts()
        .to_dict()
        if not reviewer.empty
        else {}
    )
    metadata = {
        "research_mode": "retrospective_empirical_ceiling_exhaustive_pair_r3",
        "canonical_blind_experiment": False,
        "unseen_holdout_claim_allowed": False,
        "r1_r2_periods_are_known_history": True,
        "purpose": "estimate exact two-condition historical interaction ceiling and robustness",
        "search_semantics": {
            "single_conditions": "all generated dense empirical-quantile conditions",
            "pair_conditions": "exhaustive over every compatible generated condition pair",
            "triple_conditions": "beam expansion from strongest exact pairs",
            "dnf": "OR combinations from strongest conjunction clauses",
            "fixed_drop_one_quarter": "sensitivity only; no re-search",
            "loqo_research": "re-search on all non-held quarters then evaluate held quarter",
            "rolling": "expanding-window re-search using past quarters only",
            "llm_used": False,
        },
        "replay_provenance_verified": True,
        "replay_dataset_sha256": provenance.get("replay_dataset_sha256"),
        "candidate_rows_before_maturity_filter": int(len(candidates_all)),
        "candidate_rows": int(len(candidates)),
        "excluded_immature_rows": int(len(immature_rows)),
        "outcome_maturity_cutoff": str(maturity_cutoff.date()),
        "usable_rows": int(len(frame)),
        "entry_quarters": sorted(frame["period_quarter"].astype(str).unique()),
        "censored_rows": int((~reviewer["usable"].fillna(False)).sum()) if not reviewer.empty else 0,
        "censor_reasons": censor_reasons,
        "features": feature_columns,
        "feature_count": len(feature_columns),
        "quantiles": list(DEFAULT_QUANTILES),
        "validation_quantiles": list(ROLLING_QUANTILES),
        "beam_width_after_exact_pairs": args.beam_width,
        "max_conditions": args.max_conditions,
        "max_clauses": args.max_clauses,
        "dnf_clause_pool": args.dnf_clause_pool,
        "support_constraints": {
            "min_selected": args.min_selected,
            "min_resolved": args.min_resolved,
            "min_active_quarters": args.min_active_quarters,
            "min_evaluated_quarters": args.min_evaluated_quarters,
            "min_evaluated_fraction": args.min_evaluated_fraction,
            "min_resolved_per_quarter": args.min_resolved_per_quarter,
        },
        "score_weights": SCORE_WEIGHTS,
        "pareto_metrics": list(PARETO_METRICS),
        "search_audit": search_audit,
        "candidate_rule_count": int(len(scored)),
        "pareto_rule_count": int(len(pareto)),
        "baseline": baseline,
        "best_rule_file": "best_rule.json",
        "drop_one_quarter_sensitivity_file": "drop_one_quarter_sensitivity.csv",
        "loqo_research_enabled": not args.skip_loqo_research,
        "loqo_research_rows": int(len(loqo_research)),
        "rolling_enabled": not args.skip_rolling,
        "rolling_summary": summarize_rolling(rolling),
    }
    (args.output_root / "retrospective_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
