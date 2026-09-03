from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

from .audit import (
    _oracle_value,
    _ranking_rows,
    _seed_for_snapshot,
    _simple_baseline_codes,
    raw_fixed_capacity_k,
)
from .capacity import capacity_policy_weekly
from .config import (
    AUDIT_AS_OF_DATE,
    BENCHMARK_CODES,
    BIG_WINNER_THRESHOLD_PCT,
    BLOCK_BOOTSTRAP_LEN,
    CAPACITY_POLICY_IDS,
    ELIGIBLE_ENTRY_COVERAGE_MIN_FOR_PRIMARY,
    LOSER_BOTTOM_FRAC,
    RAW_PRICE_COVERAGE_MIN_FOR_PRIMARY,
    SIMPLE_BASELINES,
    TOP_N,
    WINNER_TOP_FRAC,
    YAHOO_DOWNLOAD_AUDIT_CSV,
    YAHOO_SUPPLEMENT_PARQUET,
    PANEL_SOURCE,
    PRICE_CACHE,
    PRODUCTION_B0_PATH,
    PROTOCOL_VERSION,
)
from .data import (
    add_current_b0_state,
    git_sha,
    load_panel,
    load_price_cache,
    sha256_file,
)
from .diagnostics import (
    capacity_pick_quality,
    momentum_gate_diagnostics,
    momentum_nonoverlap,
    support_calendar_summary,
)
from .market_data import (
    build_next_open_forward_returns,
    download_yahoo_supplement,
    spy_momentum_asof,
)
from .metrics import (
    aggregate_oracle_capture,
    basic_distribution_stats,
    four_offset_nonoverlap,
    moving_block_bootstrap_ci,
    paired_edge_summary,
)
from .portfolio import (
    distribution_summary,
    portfolio_distribution,
    portfolio_from_codes,
)


def _selected_codes(s_df: pd.DataFrame) -> list[str]:
    selected = s_df[s_df["current_b0_selected"]].copy()
    if selected.empty:
        return []
    return (
        selected.sort_values("current_b0_pick_order")["code"]
        .astype(str)
        .tolist()
    )


def build_v11_frame() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    panel = add_current_b0_state(load_panel())
    base_prices = load_price_cache()
    supplement = download_yahoo_supplement(panel, base_prices)
    forward = build_next_open_forward_returns(panel, supplement.prices)

    candidate_forward = forward[~forward["code"].isin(BENCHMARK_CODES)].copy()
    merged = panel.merge(
        candidate_forward,
        on=["snapshot_date", "code"],
        how="left",
        validate="one_to_one",
    )
    spy_mom = spy_momentum_asof(
        supplement.prices,
        merged["snapshot_date"].astype(str).unique().tolist(),
        sessions=20,
    )
    merged = merged.merge(
        spy_mom,
        on="snapshot_date",
        how="left",
        validate="many_to_one",
    )
    merged["rel_spy_20"] = (
        pd.to_numeric(merged.get("mom_20"), errors="coerce")
        - pd.to_numeric(merged.get("spy_momentum"), errors="coerce")
    )
    benchmarks = forward[forward["code"].isin(BENCHMARK_CODES)].copy()

    manifest = {
        "source_git_sha": git_sha(),
        "protocol_version": PROTOCOL_VERSION,
        "audit_as_of_date": AUDIT_AS_OF_DATE,
        "panel_hash": sha256_file(PANEL_SOURCE),
        "production_b0_hash": sha256_file(PRODUCTION_B0_PATH),
        "base_price_cache_hash": sha256_file(PRICE_CACHE),
        "yahoo_supplement_hash": sha256_file(YAHOO_SUPPLEMENT_PARQUET),
        "yahoo_download_audit_hash": sha256_file(YAHOO_DOWNLOAD_AUDIT_CSV),
        "snapshot_count": int(merged["snapshot_date"].nunique()),
        "review_rows": int(len(merged)),
        "current_eligible_rows": int(merged["current_b0_eligible"].sum()),
        "current_selected_rows": int(merged["current_b0_selected"].sum()),
        "benchmark_codes": list(BENCHMARK_CODES),
        "relative_momentum_semantics": (
            "rel_spy_20 recomputed inside audit as candidate frozen mom_20 minus "
            "Yahoo-supplemented SPY 20-session momentum as of snapshot close."
        ),
        "raw_outcome_semantics": (
            "Tradable first-session open strictly after snapshot -> close at "
            "entry_date+28 calendar days, frozen at AUDIT_AS_OF_DATE."
        ),
        "entry_outcome_semantics": (
            "Existing frozen Production-style entry W4 return/stop; used only "
            "inside current B0-eligible ranking audit."
        ),
    }
    return merged, benchmarks, manifest


def _gate_weekly(s_df: pd.DataFrame, selected_codes: list[str]) -> dict[str, Any]:
    covered = s_df[s_df["next_open_price_valid"] == True].copy()
    raw_count = int(len(s_df))
    covered_count = int(len(covered))
    coverage = covered_count / float(raw_count) if raw_count else 0.0

    base = {
        "raw_count": raw_count,
        "covered_count": covered_count,
        "price_coverage": coverage,
        "primary_valid": bool(
            raw_count > 0 and coverage >= RAW_PRICE_COVERAGE_MIN_FOR_PRIMARY
        ),
    }
    if covered.empty:
        return base

    ordered = covered.sort_values(
        ["next_open_w4_return_pct", "code"],
        ascending=[False, True],
        kind="stable",
    )
    top_n = max(1, int(math.ceil(len(ordered) * WINNER_TOP_FRAC)))
    bottom_n = max(1, int(math.ceil(len(ordered) * LOSER_BOTTOM_FRAC)))
    winners = set(ordered.head(top_n)["code"].astype(str))
    losers = set(ordered.tail(bottom_n)["code"].astype(str))
    big_winners = set(
        ordered[
            pd.to_numeric(ordered["next_open_w4_return_pct"], errors="coerce")
            >= BIG_WINNER_THRESHOLD_PCT
        ]["code"].astype(str)
    )
    eligible = set(
        ordered[ordered["current_b0_eligible"]]["code"].astype(str)
    )
    selected = set(selected_codes) & set(ordered["code"].astype(str))
    rejected = set(ordered["code"].astype(str)) - eligible

    eligible_frame = ordered[ordered["code"].isin(eligible)]
    rejected_frame = ordered[ordered["code"].isin(rejected)]

    def ratio(a: float, b: float) -> float | None:
        return None if b <= 0 else float(a / b)

    base.update({
        "eligible_count": len(eligible),
        "accept_rate": ratio(len(eligible), len(ordered)),
        "winner_count": len(winners),
        "winner_retained_by_eligibility": len(winners & eligible),
        "winner_retention_rate": ratio(len(winners & eligible), len(winners)),
        "winner_captured_by_b0": len(winners & selected),
        "winner_capture_rate_b0": ratio(len(winners & selected), len(winners)),
        "rejected_count": len(rejected),
        "rejected_winner_count": len(winners & rejected),
        "rejected_winner_rate": ratio(len(winners & rejected), len(rejected)),
        "bottom_loser_count": len(losers),
        "bottom_loser_retained_count": len(losers & eligible),
        "bottom_loser_rejected_count": len(losers & rejected),
        "bottom_loser_rejection_rate": ratio(len(losers & rejected), len(losers)),
        "big_winner_count": len(big_winners),
        "big_winner_retained_count": len(big_winners & eligible),
        "big_winner_retention_rate": ratio(len(big_winners & eligible), len(big_winners)),
        "eligible_next_open_w4_mean": (
            None if eligible_frame.empty
            else float(eligible_frame["next_open_w4_return_pct"].mean())
        ),
        "rejected_next_open_w4_mean": (
            None if rejected_frame.empty
            else float(rejected_frame["next_open_w4_return_pct"].mean())
        ),
        "gate_mean_lift": (
            None
            if eligible_frame.empty or rejected_frame.empty
            else float(
                eligible_frame["next_open_w4_return_pct"].mean()
                - rejected_frame["next_open_w4_return_pct"].mean()
            )
        ),
        "selected_count": len(selected_codes),
        "expected_random_matched_n_winner_capture": (
            float(len(winners) * len(selected_codes) / len(ordered))
            if len(ordered) else 0.0
        ),
        "expected_random_fixed3_winner_capture": (
            float(len(winners) * min(TOP_N, len(ordered)) / len(ordered))
            if len(ordered) else 0.0
        ),
    })
    return base


def _rejection_event_rows(s_df: pd.DataFrame, gate: dict[str, Any]) -> list[dict[str, Any]]:
    if not gate.get("primary_valid"):
        return []

    covered = s_df[s_df["next_open_price_valid"] == True].copy()
    covered = covered.sort_values(
        ["next_open_w4_return_pct", "code"],
        ascending=[False, True],
        kind="stable",
    )
    top_n = max(1, int(math.ceil(len(covered) * WINNER_TOP_FRAC)))
    winners = set(covered.head(top_n)["code"].astype(str))

    rows: list[dict[str, Any]] = []
    for _, row in covered[~covered["current_b0_eligible"]].iterrows():
        reasons = [
            x
            for x in str(row.get("current_b0_reject_reasons", "") or "").split("|")
            if x
        ]
        rows.append({
            "snapshot_date": str(row["snapshot_date"]),
            "code": str(row["code"]),
            "reasons": "|".join(reasons),
            "reason_count": len(reasons),
            "exclusive_reason": reasons[0] if len(reasons) == 1 else "",
            "next_open_w4_return_pct": float(row["next_open_w4_return_pct"]),
            "next_open_w4_stop8": bool(row["next_open_w4_stop8"]),
            "is_top20_winner": str(row["code"]) in winners,
            "is_big_winner": float(row["next_open_w4_return_pct"]) >= BIG_WINNER_THRESHOLD_PCT,
        })
    return rows


def _rejection_summaries(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if events.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    exclusive = events[events["reason_count"] == 1].copy()
    exclusive_summary = (
        exclusive.groupby("exclusive_reason", sort=False)
        .agg(
            candidate_events=("code", "size"),
            weeks=("snapshot_date", "nunique"),
            mean_w4=("next_open_w4_return_pct", "mean"),
            median_w4=("next_open_w4_return_pct", "median"),
            positive_rate=("next_open_w4_return_pct", lambda s: float((s > 0).mean())),
            stop8_rate=("next_open_w4_stop8", "mean"),
            top20_winner_rate=("is_top20_winner", "mean"),
            big_winner_rate=("is_big_winner", "mean"),
        )
        .reset_index()
    )

    exploded_rows: list[dict[str, Any]] = []
    for _, row in events.iterrows():
        for reason in [x for x in str(row["reasons"]).split("|") if x]:
            exploded_rows.append({
                **row.to_dict(),
                "reason": reason,
            })
    exploded = pd.DataFrame(exploded_rows)
    overlap_summary = (
        exploded.groupby("reason", sort=False)
        .agg(
            label_events=("code", "size"),
            weeks=("snapshot_date", "nunique"),
            multi_reason_rate=("reason_count", lambda s: float((s > 1).mean())),
            mean_w4=("next_open_w4_return_pct", "mean"),
            median_w4=("next_open_w4_return_pct", "median"),
            top20_winner_rate=("is_top20_winner", "mean"),
            big_winner_rate=("is_big_winner", "mean"),
        )
        .reset_index()
    )

    combos = (
        events.groupby("reasons", dropna=False)
        .agg(
            candidate_events=("code", "size"),
            weeks=("snapshot_date", "nunique"),
            mean_w4=("next_open_w4_return_pct", "mean"),
            median_w4=("next_open_w4_return_pct", "median"),
            top20_winner_rate=("is_top20_winner", "mean"),
        )
        .reset_index()
        .sort_values(["candidate_events", "reasons"], ascending=[False, True])
    )
    return exclusive_summary, overlap_summary, combos


def _simple_summary(simple: pd.DataFrame, b0_weekly: pd.DataFrame) -> pd.DataFrame:
    if simple.empty:
        return pd.DataFrame()
    b0 = b0_weekly[
        [
            "snapshot_date",
            "next_open_capital_adjusted",
            "next_open_stop8_pct",
            "next_open_one_pick_ruined",
        ]
    ].rename(
        columns={
            "next_open_capital_adjusted": "b0_return",
            "next_open_stop8_pct": "b0_stop8_pct",
            "next_open_one_pick_ruined": "b0_any_stop_or_le8",
        }
    )
    rows: list[dict[str, Any]] = []

    for baseline, group in simple.groupby("baseline"):
        merged = group.merge(b0, on="snapshot_date", how="left")
        valid = merged[
            (merged["primary_valid"] == True)
            & merged["return"].notna()
            & merged["b0_return"].notna()
        ].copy()
        if valid.empty:
            rows.append({
                "baseline": baseline,
                "support_weeks": 0,
                "mean_return": None,
                "median_return": None,
                "mean_spread_vs_b0": None,
                "median_spread_vs_b0": None,
                "beat_b0_rate": None,
                "spread_ci_low": None,
                "spread_ci_high": None,
                "mean_without_best1": None,
                "mean_without_best2": None,
                "positive_edge_concentration": None,
                "mean_stop8_pct": None,
                "b0_mean_stop8_pct": None,
                "stop8_exposure_delta_pp": None,
                "any_stop_or_le8_week_rate": None,
                "b0_any_stop_or_le8_week_rate": None,
                "any_stop_or_le8_week_delta_pp": None,
            })
            continue

        spread = valid["return"].astype(float) - valid["b0_return"].astype(float)
        ci = moving_block_bootstrap_ci(spread.to_numpy())
        ordered = spread.sort_values(ascending=False).reset_index(drop=True)
        positives = spread[spread > 0]
        positive_sum = float(positives.sum())

        rows.append({
            "baseline": baseline,
            "support_weeks": int(len(valid)),
            "mean_return": float(valid["return"].mean()),
            "median_return": float(valid["return"].median()),
            "mean_spread_vs_b0": float(spread.mean()),
            "median_spread_vs_b0": float(spread.median()),
            "beat_b0_rate": float((spread > 0).mean()),
            "spread_ci_low": ci["mean_ci_low"],
            "spread_ci_high": ci["mean_ci_high"],
            "mean_without_best1": (
                None if len(ordered) <= 1 else float(ordered.iloc[1:].mean())
            ),
            "mean_without_best2": (
                None if len(ordered) <= 2 else float(ordered.iloc[2:].mean())
            ),
            "positive_edge_concentration": (
                None if positive_sum <= 0 else float(ordered.iloc[0] / positive_sum)
            ),
            "mean_stop8_pct": float(
                pd.to_numeric(valid["stop8_pct"], errors="coerce").mean()
            ),
            "b0_mean_stop8_pct": float(
                pd.to_numeric(valid["b0_stop8_pct"], errors="coerce").mean()
            ),
            "stop8_exposure_delta_pp": float(
                (
                    pd.to_numeric(valid["stop8_pct"], errors="coerce")
                    - pd.to_numeric(valid["b0_stop8_pct"], errors="coerce")
                ).mean()
            ),
            "any_stop_or_le8_week_rate": float(
                valid["one_pick_ruined"].astype(float).mean()
            ),
            "b0_any_stop_or_le8_week_rate": float(
                valid["b0_any_stop_or_le8"].astype(float).mean()
            ),
            "any_stop_or_le8_week_delta_pp": float(
                (
                    valid["one_pick_ruined"].astype(float)
                    - valid["b0_any_stop_or_le8"].astype(float)
                ).mean() * 100.0
            ),
        })
    return pd.DataFrame(rows)


def _capacity_summary(weekly: pd.DataFrame) -> pd.DataFrame:
    if weekly.empty:
        return pd.DataFrame()

    original = weekly[weekly["policy_id"] == "B0_ORIGINAL"][
        [
            "snapshot_date",
            "capital_adjusted_return",
            "capital_stop8_pct",
            "one_pick_ruined",
            "original_pick_count",
            "underfill_cause",
        ]
    ].rename(columns={
        "capital_adjusted_return": "b0_return",
        "capital_stop8_pct": "b0_stop8",
        "one_pick_ruined": "b0_ruin",
    })

    rows: list[dict[str, Any]] = []
    for policy_id in CAPACITY_POLICY_IDS:
        group = weekly[weekly["policy_id"] == policy_id].merge(
            original,
            on=["snapshot_date", "original_pick_count", "underfill_cause"],
            how="left",
        )
        for scope_name, scope in [
            ("all_mature", group),
            ("underfilled_only", group[group["original_pick_count"] < TOP_N]),
        ]:
            valid = scope[
                (scope["mature"] == True)
                & scope["capital_adjusted_return"].notna()
                & scope["b0_return"].notna()
            ].copy()
            spread = (
                valid["capital_adjusted_return"].astype(float)
                - valid["b0_return"].astype(float)
            )
            stop_delta = (
                pd.to_numeric(valid["capital_stop8_pct"], errors="coerce")
                - pd.to_numeric(valid["b0_stop8"], errors="coerce")
            )
            ruin_delta = (
                valid["one_pick_ruined"].astype(float)
                - valid["b0_ruin"].astype(float)
            )
            added_returns = pd.to_numeric(
                valid["added_selection_quality_return"],
                errors="coerce",
            ).dropna()

            rows.append({
                "policy_id": policy_id,
                "scope": scope_name,
                "support_weeks": int(len(valid)),
                "mean_return": None if valid.empty else float(valid["capital_adjusted_return"].mean()),
                "median_return": None if valid.empty else float(valid["capital_adjusted_return"].median()),
                "mean_spread_vs_b0": None if valid.empty else float(spread.mean()),
                "median_spread_vs_b0": None if valid.empty else float(spread.median()),
                "beat_b0_rate": None if valid.empty else float((spread > 0).mean()),
                "spread_ci_low": (
                    None if valid.empty else moving_block_bootstrap_ci(spread.to_numpy())["mean_ci_low"]
                ),
                "spread_ci_high": (
                    None if valid.empty else moving_block_bootstrap_ci(spread.to_numpy())["mean_ci_high"]
                ),
                "mean_slot_stop_exposure_delta_pp": (
                    None if valid.empty else float(stop_delta.mean())
                ),
                "any_stop_or_le8_week_delta_pp": (
                    None if valid.empty else float(ruin_delta.mean() * 100.0)
                ),
                "full3_rate": None if valid.empty else float(valid["full3"].mean()),
                "mean_added_pick_return": (
                    None if added_returns.empty else float(added_returns.mean())
                ),
                "median_added_pick_return": (
                    None if added_returns.empty else float(added_returns.median())
                ),
            })
    return pd.DataFrame(rows)


def materialize_v11_core() -> dict[str, Any]:
    panel, benchmarks, manifest = build_v11_frame()

    b0_rows: list[dict[str, Any]] = []
    eligible_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    rank_bucket_rows: list[dict[str, Any]] = []
    reject_events: list[dict[str, Any]] = []
    simple_rows: list[dict[str, Any]] = []

    benchmark_lookup = {
        (str(row["snapshot_date"]), str(row["code"])): row
        for _, row in benchmarks.iterrows()
    }

    for snapshot in sorted(panel["snapshot_date"].astype(str).unique().tolist()):
        s_df = panel[panel["snapshot_date"].astype(str) == snapshot].copy()
        selected_codes = _selected_codes(s_df)
        k = len(selected_codes)

        entry_port = portfolio_from_codes(
            s_df,
            selected_codes,
            return_col="w4_return_pct",
            stop_col="w4_stop8",
        )
        tradable_port = portfolio_from_codes(
            s_df,
            selected_codes,
            return_col="next_open_w4_return_pct",
            stop_col="next_open_w4_stop8",
        )

        spy = benchmark_lookup.get((snapshot, "SPY"))
        qqq = benchmark_lookup.get((snapshot, "QQQ"))
        spy_ret = (
            float(spy["next_open_w4_return_pct"])
            if spy is not None and bool(spy["next_open_price_valid"])
            else None
        )
        qqq_ret = (
            float(qqq["next_open_w4_return_pct"])
            if qqq is not None and bool(qqq["next_open_price_valid"])
            else None
        )

        b0_rows.append({
            "snapshot_date": snapshot,
            "pick_count": k,
            "selected_codes": json.dumps(selected_codes),
            "entry_mature": bool(entry_port["mature"]),
            "entry_capital_adjusted": (
                entry_port["capital_adjusted_return"] if entry_port["mature"] else None
            ),
            "entry_selection_quality": (
                entry_port["selection_quality_return"] if entry_port["mature"] else None
            ),
            "entry_stop8_pct": (
                entry_port["capital_adjusted_stop8"] if entry_port["mature"] else None
            ),
            "entry_one_pick_ruined": (
                entry_port["one_pick_ruined"] if entry_port["mature"] else None
            ),
            "next_open_mature": bool(tradable_port["mature"]),
            "next_open_capital_adjusted": (
                tradable_port["capital_adjusted_return"] if tradable_port["mature"] else None
            ),
            "next_open_selection_quality": (
                tradable_port["selection_quality_return"] if tradable_port["mature"] else None
            ),
            "next_open_stop8_pct": (
                tradable_port["capital_adjusted_stop8"] if tradable_port["mature"] else None
            ),
            "next_open_one_pick_ruined": (
                tradable_port["one_pick_ruined"] if tradable_port["mature"] else None
            ),
            "spy_next_open_w4": spy_ret,
            "spy_exposure_matched_w4": (
                None if spy_ret is None else spy_ret * k / float(TOP_N)
            ),
            "qqq_next_open_w4": qqq_ret,
            "qqq_exposure_matched_w4": (
                None if qqq_ret is None else qqq_ret * k / float(TOP_N)
            ),
        })

        # Entry-aligned ranking benchmark.
        eligible = s_df[s_df["current_b0_eligible"]].copy()
        eligible_mature = eligible[
            pd.to_numeric(eligible["w4_return_pct"], errors="coerce").notna()
            & eligible["w4_stop8"].notna()
        ].copy()
        eligible_cov = len(eligible_mature) / float(len(eligible)) if len(eligible) else 0.0
        e_record: dict[str, Any] = {
            "snapshot_date": snapshot,
            "pick_count": k,
            "eligible_count": int(len(eligible)),
            "eligible_mature_count": int(len(eligible_mature)),
            "eligible_coverage": eligible_cov,
            "primary_valid": False,
            "active_choice": False,
            "b0_return": (
                entry_port["capital_adjusted_return"] if entry_port["mature"] else None
            ),
        }
        if (
            k > 0
            and entry_port["mature"]
            and len(eligible) >= k
            and eligible_cov >= ELIGIBLE_ENTRY_COVERAGE_MIN_FOR_PRIMARY
        ):
            dist, meta = portfolio_distribution(
                eligible,
                k=k,
                return_col="w4_return_pct",
                distinct_industry=True,
                seed=_seed_for_snapshot(snapshot, 511),
            )
            oracle_codes, oracle = _oracle_value(
                eligible,
                k=k,
                return_col="w4_return_pct",
                distinct_industry=True,
            )
            if len(dist) > 0 and oracle is not None:
                e_record["primary_valid"] = True
                e_record["active_choice"] = len(dist) > 1
                e_record.update(
                    distribution_summary(
                        float(entry_port["capital_adjusted_return"]),
                        dist,
                        oracle,
                    )
                )
                e_record.update({
                    "oracle_codes": json.dumps(oracle_codes),
                    "feasible_portfolio_count": int(len(dist)),
                    **{f"distribution_{key}": value for key, value in meta.items()},
                })
        eligible_rows.append(e_record)

        # Tradable raw-universe benchmark.
        raw_covered = s_df[s_df["next_open_price_valid"] == True].copy()
        raw_cov = len(raw_covered) / float(len(s_df)) if len(s_df) else 0.0
        fixed_k = raw_fixed_capacity_k(raw_covered)
        raw_primary = bool(
            len(s_df) > 0
            and raw_cov >= RAW_PRICE_COVERAGE_MIN_FOR_PRIMARY
            and fixed_k > 0
            and tradable_port["mature"]
        )
        r_record: dict[str, Any] = {
            "snapshot_date": snapshot,
            "pick_count": k,
            "fixed_capacity_pick_count": fixed_k,
            "raw_count": int(len(s_df)),
            "covered_count": int(len(raw_covered)),
            "coverage": raw_cov,
            "primary_valid": raw_primary,
            "b0_return": (
                tradable_port["capital_adjusted_return"] if tradable_port["mature"] else None
            ),
            "b0_selection_quality": (
                tradable_port["selection_quality_return"] if tradable_port["mature"] else None
            ),
            "raw_equal_weight_mean": (
                None if raw_covered.empty else float(raw_covered["next_open_w4_return_pct"].mean())
            ),
        }
        if raw_primary:
            fixed_dist, fixed_meta = portfolio_distribution(
                raw_covered,
                k=fixed_k,
                return_col="next_open_w4_return_pct",
                distinct_industry=True,
                seed=_seed_for_snapshot(snapshot, 611),
            )
            fixed_oracle_codes, fixed_oracle = _oracle_value(
                raw_covered,
                k=fixed_k,
                return_col="next_open_w4_return_pct",
                distinct_industry=True,
            )
            if len(fixed_dist) > 0 and fixed_oracle is not None:
                fixed = distribution_summary(
                    float(tradable_port["capital_adjusted_return"]),
                    fixed_dist,
                    fixed_oracle,
                )
                r_record.update({f"fixed_{key}": value for key, value in fixed.items()})
                r_record["fixed_oracle_codes"] = json.dumps(fixed_oracle_codes)
                r_record.update({
                    f"fixed_distribution_{key}": value
                    for key, value in fixed_meta.items()
                })
            else:
                r_record["primary_valid"] = False

            if k > 0:
                matched_dist, matched_meta = portfolio_distribution(
                    raw_covered,
                    k=k,
                    return_col="next_open_w4_return_pct",
                    distinct_industry=True,
                    seed=_seed_for_snapshot(snapshot, 711),
                )
                matched_oracle_codes, matched_oracle = _oracle_value(
                    raw_covered,
                    k=k,
                    return_col="next_open_w4_return_pct",
                    distinct_industry=True,
                )
                if len(matched_dist) > 0 and matched_oracle is not None:
                    matched = distribution_summary(
                        float(tradable_port["capital_adjusted_return"]),
                        matched_dist,
                        matched_oracle,
                    )
                    r_record.update({
                        f"matched_{key}": value for key, value in matched.items()
                    })
                    r_record["matched_oracle_codes"] = json.dumps(matched_oracle_codes)
                    r_record.update({
                        f"matched_distribution_{key}": value
                        for key, value in matched_meta.items()
                    })
        raw_rows.append(r_record)

        gate = _gate_weekly(s_df, selected_codes)
        gate["snapshot_date"] = snapshot
        gate_rows.append(gate)
        reject_events.extend(_rejection_event_rows(s_df, gate))

        weekly_rank, bucket_rows = _ranking_rows(s_df, selected_codes)
        weekly_rank["snapshot_date"] = snapshot
        ranking_rows.append(weekly_rank)
        for row in bucket_rows:
            row["snapshot_date"] = snapshot
            rank_bucket_rows.append(row)

        for name, feature, direction in SIMPLE_BASELINES:
            codes = _simple_baseline_codes(
                s_df,
                k=fixed_k,
                feature=feature,
                direction=direction,
            )
            port = portfolio_from_codes(
                s_df,
                codes,
                return_col="next_open_w4_return_pct",
                stop_col="next_open_w4_stop8",
            )
            simple_rows.append({
                "snapshot_date": snapshot,
                "baseline": name,
                "feature": feature,
                "target_pick_count": fixed_k,
                "pick_count": len(codes),
                "codes": json.dumps(codes),
                "raw_week_primary_valid": raw_primary,
                "full_feature_capacity": len(codes) == fixed_k and fixed_k > 0,
                "mature": bool(port["mature"]),
                "primary_valid": bool(
                    raw_primary
                    and fixed_k > 0
                    and len(codes) == fixed_k
                    and port["mature"]
                ),
                "return": (
                    port["capital_adjusted_return"] if port["mature"] else None
                ),
                "stop8_pct": (
                    port["capital_adjusted_stop8"] if port["mature"] else None
                ),
                "one_pick_ruined": (
                    port["one_pick_ruined"] if port["mature"] else None
                ),
            })

    frames = {
        "panel": panel,
        "benchmarks": benchmarks,
        "b0_weekly": pd.DataFrame(b0_rows),
        "eligible_random_weekly": pd.DataFrame(eligible_rows),
        "raw_random_weekly": pd.DataFrame(raw_rows),
        "gate_weekly": pd.DataFrame(gate_rows),
        "ranking_weekly": pd.DataFrame(ranking_rows),
        "rank_bucket_rows": pd.DataFrame(rank_bucket_rows),
        "rejection_events": pd.DataFrame(reject_events),
        "simple_weekly": pd.DataFrame(simple_rows),
    }
    frames["capacity_weekly"] = capacity_policy_weekly(panel)
    return {"frames": frames, "manifest": manifest}


def active_choice_eligible_rows(eligible: pd.DataFrame) -> pd.DataFrame:
    """Ranking headline support: valid weeks with >1 feasible portfolio only."""
    if eligible.empty:
        return eligible.copy()
    return eligible[
        (eligible["primary_valid"] == True)
        & (eligible["active_choice"] == True)
        & (pd.to_numeric(eligible["feasible_portfolio_count"], errors="coerce") > 1)
    ].copy()


def summarize_v11(core: dict[str, Any]) -> dict[str, Any]:
    f = core["frames"]
    b0 = f["b0_weekly"]
    eligible = f["eligible_random_weekly"]
    raw = f["raw_random_weekly"]
    gate = f["gate_weekly"]
    ranking = f["ranking_weekly"]

    mature_entry = b0[b0["entry_mature"] == True].copy()
    entry_rets = pd.to_numeric(
        mature_entry["entry_capital_adjusted"], errors="coerce"
    ).dropna()
    absolute = {
        "entry_aligned": basic_distribution_stats(entry_rets.to_numpy()),
        "entry_block_bootstrap": moving_block_bootstrap_ci(entry_rets.to_numpy()),
        "mean_slot_coverage_all_snapshots": float(
            (b0["pick_count"] / float(TOP_N)).mean()
        ),
        "full3_rate_all_snapshots": float((b0["pick_count"] == TOP_N).mean()),
        "pick_count_distribution": {
            str(k): int((b0["pick_count"] == k).sum())
            for k in range(TOP_N + 1)
        },
    }

    elig_valid = eligible[eligible["primary_valid"] == True].copy()
    active = active_choice_eligible_rows(eligible)
    no_choice = elig_valid[elig_valid["active_choice"] != True].copy()
    active_pct = pd.to_numeric(active.get("b0_percentile"), errors="coerce").dropna()
    active_edge = (
        paired_edge_summary(active, "b0_return", "random_mean")
        if not active.empty else {}
    )
    active_capture = aggregate_oracle_capture(active) if not active.empty else {}

    raw_valid = raw[raw["primary_valid"] == True].copy()
    raw_pct = pd.to_numeric(
        raw_valid.get("fixed_b0_percentile", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    raw_edge = (
        paired_edge_summary(raw_valid, "b0_return", "fixed_random_mean")
        if not raw_valid.empty else {}
    )
    raw_capture_frame = pd.DataFrame()
    raw_capture = {}
    if not raw_valid.empty:
        cols = {
            "b0_return": "b0_return",
            "fixed_random_mean": "random_mean",
            "fixed_oracle": "oracle",
            "fixed_oracle_capture_ratio": "oracle_capture_ratio",
        }
        if all(col in raw_valid.columns for col in cols):
            raw_capture_frame = raw_valid[list(cols)].rename(columns=cols)
            raw_capture = aggregate_oracle_capture(raw_capture_frame)

    matched_valid = raw_valid[
        raw_valid.get("matched_random_mean", pd.Series(index=raw_valid.index)).notna()
    ].copy()
    matched_edge = (
        paired_edge_summary(matched_valid, "b0_return", "matched_random_mean")
        if not matched_valid.empty else {}
    )

    raw_by_pick_rows: list[dict[str, Any]] = []
    for k in range(TOP_N + 1):
        group = raw_valid[raw_valid["pick_count"] == k].copy()
        if group.empty:
            continue
        spread = group["b0_return"].astype(float) - group["fixed_random_mean"].astype(float)
        pct = pd.to_numeric(group["fixed_b0_percentile"], errors="coerce").dropna()
        raw_by_pick_rows.append({
            "pick_count": k,
            "support_weeks": int(len(group)),
            "b0_mean": float(group["b0_return"].mean()),
            "random_mean": float(group["fixed_random_mean"].mean()),
            "mean_spread": float(spread.mean()),
            "median_spread": float(spread.median()),
            "beat_rate": float((spread > 0).mean()),
            "median_percentile": None if pct.empty else float(pct.median()),
        })
    raw_by_pick = pd.DataFrame(raw_by_pick_rows)

    gate_valid = gate[gate["primary_valid"] == True].copy()

    def wsum(col: str) -> float:
        return float(pd.to_numeric(gate_valid[col], errors="coerce").fillna(0).sum())

    total_raw = wsum("raw_count")
    total_eligible = wsum("eligible_count")
    total_winners = wsum("winner_count")
    retained_winners = wsum("winner_retained_by_eligibility")
    captured_winners = wsum("winner_captured_by_b0")
    total_losers = wsum("bottom_loser_count")
    retained_losers = wsum("bottom_loser_retained_count")
    expected_matched = wsum("expected_random_matched_n_winner_capture")
    expected_fixed3 = wsum("expected_random_fixed3_winner_capture")

    accept_rate = None if total_raw <= 0 else total_eligible / total_raw
    winner_retention = None if total_winners <= 0 else retained_winners / total_winners
    loser_retention = None if total_losers <= 0 else retained_losers / total_losers
    b0_capture = None if total_winners <= 0 else captured_winners / total_winners

    gate_summary = {
        "support_weeks": int(len(gate_valid)),
        "mean_price_coverage_all_weeks": float(gate["price_coverage"].mean()),
        "raw_candidate_events": int(total_raw),
        "eligible_candidate_events": int(total_eligible),
        "accept_rate": accept_rate,
        "winner_retention_rate": winner_retention,
        "winner_enrichment_vs_random_selectivity": (
            None
            if accept_rate in {None, 0} or winner_retention is None
            else winner_retention / accept_rate
        ),
        "bottom_loser_retention_rate": loser_retention,
        "loser_retention_vs_random_selectivity": (
            None
            if accept_rate in {None, 0} or loser_retention is None
            else loser_retention / accept_rate
        ),
        "b0_winner_capture_rate": b0_capture,
        "b0_winner_capture_enrichment_vs_matched_n_random": (
            None if expected_matched <= 0 else captured_winners / expected_matched
        ),
        "b0_winner_capture_enrichment_vs_fixed3_random": (
            None if expected_fixed3 <= 0 else captured_winners / expected_fixed3
        ),
        "mean_gate_lift": (
            None if gate_valid.empty
            else float(pd.to_numeric(gate_valid["gate_mean_lift"], errors="coerce").mean())
        ),
        "median_gate_lift": (
            None if gate_valid.empty
            else float(pd.to_numeric(gate_valid["gate_mean_lift"], errors="coerce").median())
        ),
    }

    exclusive_reject, overlap_reject, reject_combos = _rejection_summaries(
        f["rejection_events"]
    )

    ranking_valid = ranking[ranking["primary_valid"] == True].copy()
    spearman = pd.to_numeric(ranking_valid["weekly_spearman"], errors="coerce").dropna()
    rank_lift = pd.to_numeric(
        ranking_valid["b0_minus_eligible_mean"], errors="coerce"
    ).dropna()
    ranking_summary = {
        "support_weeks": int(len(ranking_valid)),
        "weekly_spearman_mean": None if spearman.empty else float(spearman.mean()),
        "weekly_spearman_median": None if spearman.empty else float(spearman.median()),
        "positive_spearman_week_rate": (
            None if spearman.empty else float((spearman > 0).mean())
        ),
        "selected_minus_eligible_mean": (
            None if rank_lift.empty else float(rank_lift.mean())
        ),
        "selected_minus_eligible_median": (
            None if rank_lift.empty else float(rank_lift.median())
        ),
    }

    rank_bucket_summary = (
        f["rank_bucket_rows"].groupby("rank_bucket", sort=False)
        .agg(
            candidate_rows=("code", "size"),
            weeks=("snapshot_date", "nunique"),
            mean_w4=("w4_return_pct", "mean"),
            median_w4=("w4_return_pct", "median"),
            positive_rate=("w4_return_pct", lambda s: float((s > 0).mean())),
            stop8_rate=("w4_stop8", "mean"),
        )
        .reset_index()
        if not f["rank_bucket_rows"].empty
        else pd.DataFrame()
    )

    simple_summary = _simple_summary(f["simple_weekly"], b0)
    capacity_summary = _capacity_summary(f["capacity_weekly"])
    original_capacity = f["capacity_weekly"][
        f["capacity_weekly"]["policy_id"] == "B0_ORIGINAL"
    ].copy()
    underfill_cause_summary = (
        original_capacity.groupby("underfill_cause", sort=False)
        .agg(
            weeks=("snapshot_date", "nunique"),
            mean_original_pick_count=("original_pick_count", "mean"),
            mature_weeks=("mature", "sum"),
        )
        .reset_index()
        if not original_capacity.empty
        else pd.DataFrame()
    )
    capacity_pick_quality_summary, capacity_added_reason_summary = capacity_pick_quality(
        f["panel"], f["capacity_weekly"]
    )
    support_calendar = support_calendar_summary(
        sorted(f["panel"]["snapshot_date"].astype(str).unique().tolist()),
        f["raw_random_weekly"],
        f["simple_weekly"],
    )
    momentum_gate_summary, momentum_gate_reason_summary = momentum_gate_diagnostics(
        f["panel"], f["simple_weekly"]
    )

    bmark = f["benchmarks"]
    market_rows: list[dict[str, Any]] = []
    for code in BENCHMARK_CODES:
        bench = bmark[
            (bmark["code"] == code)
            & (bmark["next_open_price_valid"] == True)
        ][["snapshot_date", "next_open_w4_return_pct"]].copy()
        merged = b0.merge(bench, on="snapshot_date", how="inner")
        merged = merged[
            (merged["next_open_mature"] == True)
            & merged["next_open_capital_adjusted"].notna()
        ].copy()
        if merged.empty:
            market_rows.append({"benchmark": code, "support_weeks": 0})
            continue
        full_spread = (
            merged["next_open_capital_adjusted"].astype(float)
            - merged["next_open_w4_return_pct"].astype(float)
        )
        exposure_bench = (
            merged["next_open_w4_return_pct"].astype(float)
            * merged["pick_count"].astype(float)
            / float(TOP_N)
        )
        exposure_spread = (
            merged["next_open_capital_adjusted"].astype(float) - exposure_bench
        )
        selection = merged[
            merged["pick_count"] > 0
        ].copy()
        selection_spread = (
            selection["next_open_selection_quality"].astype(float)
            - selection["next_open_w4_return_pct"].astype(float)
        )
        market_rows.append({
            "benchmark": code,
            "support_weeks": int(len(merged)),
            "benchmark_mean": float(merged["next_open_w4_return_pct"].mean()),
            "b0_capital_mean": float(merged["next_open_capital_adjusted"].mean()),
            "full_exposure_mean_spread": float(full_spread.mean()),
            "full_exposure_median_spread": float(full_spread.median()),
            "exposure_matched_mean_spread": float(exposure_spread.mean()),
            "selection_quality_support": int(len(selection)),
            "active_pick_selection_mean_spread": (
                None if selection.empty else float(selection_spread.mean())
            ),
            "full_spread_ci_low": moving_block_bootstrap_ci(
                full_spread.to_numpy()
            )["mean_ci_low"],
            "full_spread_ci_high": moving_block_bootstrap_ci(
                full_spread.to_numpy()
            )["mean_ci_high"],
        })
    market_summary = pd.DataFrame(market_rows)

    offsets: list[pd.DataFrame] = []
    if not active.empty:
        off = four_offset_nonoverlap(
            active,
            value_col="b0_return",
            benchmark_col="random_mean",
        )
        off["comparison"] = "active_choice_b0_vs_eligible_random"
        offsets.append(off)
    if not raw_valid.empty:
        off = four_offset_nonoverlap(
            raw_valid,
            value_col="b0_return",
            benchmark_col="fixed_random_mean",
        )
        off["comparison"] = "b0_vs_raw_fixed3_next_open"
        offsets.append(off)
    mom_off = momentum_nonoverlap(f["simple_weekly"], b0)
    if not mom_off.empty:
        offsets.append(mom_off)
    nonoverlap = (
        pd.concat(offsets, ignore_index=True) if offsets else pd.DataFrame()
    )

    health = {
        "summary_policy": "NO_ARBITRARY_PASS_FAIL_THRESHOLD",
        "absolute": absolute,
        "eligible_ranking_active_choice": {
            "all_valid_weeks": int(len(elig_valid)),
            "no_choice_weeks": int(len(no_choice)),
            "active_choice_weeks": int(len(active)),
            "mean_percentile": (
                None if active_pct.empty else float(active_pct.mean())
            ),
            "median_percentile": (
                None if active_pct.empty else float(active_pct.median())
            ),
            "edge": active_edge,
            "oracle_capture": active_capture,
        },
        "raw_fixed_capacity_next_open": {
            "support_weeks": int(len(raw_valid)),
            "support_fraction_of_snapshots": (
                float(len(raw_valid) / max(1, b0["snapshot_date"].nunique()))
            ),
            "mean_price_coverage_all_weeks": float(raw["coverage"].mean()),
            "mean_percentile": None if raw_pct.empty else float(raw_pct.mean()),
            "median_percentile": None if raw_pct.empty else float(raw_pct.median()),
            "edge": raw_edge,
            "oracle_capture": raw_capture,
            "matched_n_edge": matched_edge,
            "support_dates": raw_valid["snapshot_date"].astype(str).tolist(),
        },
        "gate": gate_summary,
        "momentum_gate": (
            {}
            if momentum_gate_summary.empty
            else {
                row["cohort"]: {
                    key: row[key]
                    for key in momentum_gate_summary.columns
                    if key != "cohort"
                }
                for _, row in momentum_gate_summary.iterrows()
            }
        ),
        "ranking_information": ranking_summary,
        "evidence_boundary": (
            "Retrospective audit. B0 was developed with visibility into this history. "
            "Yahoo supplementation is frozen at 2026-09-02 and used only for outcome "
            "completion / benchmarks; no future bar after the frozen as-of date is allowed."
        ),
    }

    return {
        "health": health,
        "raw_by_pick_count": raw_by_pick,
        "exclusive_rejection_summary": exclusive_reject,
        "overlap_rejection_summary": overlap_reject,
        "rejection_combinations": reject_combos,
        "rank_bucket_summary": rank_bucket_summary,
        "simple_summary": simple_summary,
        "capacity_summary": capacity_summary,
        "capacity_pick_quality_summary": capacity_pick_quality_summary,
        "capacity_added_reason_summary": capacity_added_reason_summary,
        "underfill_cause_summary": underfill_cause_summary,
        "support_calendar_summary": support_calendar,
        "momentum_gate_summary": momentum_gate_summary,
        "momentum_gate_reason_summary": momentum_gate_reason_summary,
        "market_summary": market_summary,
        "nonoverlap": nonoverlap,
    }
