from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    BIG_WINNER_THRESHOLD_PCT,
    ELIGIBLE_ENTRY_COVERAGE_MIN_FOR_PRIMARY,
    LOSER_BOTTOM_FRAC,
    RANDOM_SEED,
    RAW_PRICE_COVERAGE_MIN_FOR_PRIMARY,
    SIMPLE_BASELINES,
    TOP_N,
    WINNER_TOP_FRAC,
)
from .data import build_audit_frame, source_manifest
from .metrics import (
    aggregate_oracle_capture,
    basic_distribution_stats,
    four_offset_nonoverlap,
    moving_block_bootstrap_ci,
    paired_edge_summary,
    safe_spearman,
)
from .portfolio import (
    distribution_summary,
    greedy_oracle_codes,
    industry_key,
    portfolio_distribution,
    portfolio_from_codes,
)


def _seed_for_snapshot(snapshot: str, salt: int = 0) -> int:
    digits = int(pd.Timestamp(snapshot).strftime("%y%m%d"))
    return RANDOM_SEED + digits + salt


def raw_fixed_capacity_k(raw_covered: pd.DataFrame) -> int:
    """De-anchored weekly capacity: fill up to 3 distinct-industry slots.

    Unknown/blank industries are intentionally treated as unique-by-code by
    industry_key(); raw-system evaluation must not silently inherit B0's
    industry-metadata eligibility gate.
    """
    if raw_covered.empty:
        return 0
    distinct_industries = {
        industry_key(row) for _, row in raw_covered.iterrows()
    }
    return min(TOP_N, len(raw_covered), len(distinct_industries))


def _selected_codes(s_df: pd.DataFrame) -> list[str]:
    selected = s_df[s_df["current_b0_selected"]].copy()
    if selected.empty:
        return []
    selected = selected.sort_values("current_b0_pick_order")
    return selected["code"].astype(str).tolist()


def _oracle_value(
    candidates: pd.DataFrame,
    *,
    k: int,
    return_col: str,
    distinct_industry: bool,
) -> tuple[list[str], float | None]:
    codes = greedy_oracle_codes(
        candidates,
        k=k,
        return_col=return_col,
        distinct_industry=distinct_industry,
    )
    if len(codes) != k:
        return codes, None
    p = portfolio_from_codes(candidates, codes, return_col=return_col)
    return codes, float(p["capital_adjusted_return"]) if p["mature"] else None


def _simple_baseline_codes(
    s_df: pd.DataFrame,
    *,
    k: int,
    feature: str,
    direction: str,
) -> list[str]:
    """Raw-universe PIT-only baseline. It never reads B0 state or outcomes."""
    if k <= 0 or feature not in s_df.columns:
        return []

    work = s_df[["code", "industry", feature]].copy()
    vals = pd.to_numeric(work[feature], errors="coerce")
    if direction == "abs_asc":
        work["_score"] = vals.abs()
        work = work[work["_score"].notna()].sort_values(
            ["_score", "code"], ascending=[True, True], kind="stable"
        )
    elif direction == "desc":
        work["_score"] = vals
        work = work[work["_score"].notna()].sort_values(
            ["_score", "code"], ascending=[False, True], kind="stable"
        )
    else:
        raise RuntimeError(f"Unknown simple baseline direction: {direction}")

    selected: list[str] = []
    used_industries: set[str] = set()
    for _, row in work.iterrows():
        if len(selected) >= k:
            break
        ind = industry_key(row)
        if ind in used_industries:
            continue
        selected.append(str(row["code"]))
        used_industries.add(ind)
    return selected


def _winner_gate_row(s_df: pd.DataFrame, selected_codes: list[str]) -> dict[str, Any]:
    covered = s_df[s_df["snapshot_price_valid"] == True].copy()
    n_raw = len(s_df)
    n_covered = len(covered)
    coverage = n_covered / float(n_raw) if n_raw else 0.0

    if n_covered == 0:
        return {
            "raw_count": n_raw,
            "price_covered_count": 0,
            "price_coverage": coverage,
            "primary_valid": False,
        }

    covered = covered.sort_values(
        ["snapshot_w4_return_pct", "code"],
        ascending=[False, True],
        kind="stable",
    )
    top_n = max(1, int(math.ceil(len(covered) * WINNER_TOP_FRAC)))
    bottom_n = max(1, int(math.ceil(len(covered) * LOSER_BOTTOM_FRAC)))
    winners = set(covered.head(top_n)["code"].astype(str))
    losers = set(covered.tail(bottom_n)["code"].astype(str))
    big_winners = set(
        covered[
            pd.to_numeric(covered["snapshot_w4_return_pct"], errors="coerce")
            >= BIG_WINNER_THRESHOLD_PCT
        ]["code"].astype(str)
    )

    eligible = set(
        covered[covered["current_b0_eligible"]]["code"].astype(str)
    )
    selected = set(selected_codes) & set(covered["code"].astype(str))
    rejected = set(covered["code"].astype(str)) - eligible

    eligible_frame = covered[covered["code"].isin(eligible)]
    rejected_frame = covered[covered["code"].isin(rejected)]

    def ratio(num: int, den: int) -> float | None:
        return None if den <= 0 else float(num / den)

    return {
        "raw_count": n_raw,
        "price_covered_count": n_covered,
        "price_coverage": round(coverage, 6),
        "primary_valid": coverage >= RAW_PRICE_COVERAGE_MIN_FOR_PRIMARY,
        "winner_count": len(winners),
        "winner_retained_by_eligibility": len(winners & eligible),
        "winner_retention_rate": ratio(len(winners & eligible), len(winners)),
        "winner_captured_by_b0": len(winners & selected),
        "winner_capture_rate_b0": ratio(len(winners & selected), len(winners)),
        "eligible_winner_precision": ratio(len(winners & eligible), len(eligible)),
        "b0_winner_precision": ratio(len(winners & selected), len(selected)),
        "rejected_count": len(rejected),
        "rejected_winner_count": len(winners & rejected),
        "rejected_winner_rate": ratio(len(winners & rejected), len(rejected)),
        "bottom_loser_count": len(losers),
        "bottom_loser_rejected_count": len(losers & rejected),
        "bottom_loser_rejection_rate": ratio(len(losers & rejected), len(losers)),
        "big_winner_count": len(big_winners),
        "big_winner_retained_count": len(big_winners & eligible),
        "big_winner_retention_rate": ratio(len(big_winners & eligible), len(big_winners)),
        "eligible_snapshot_w4_mean": (
            None if eligible_frame.empty
            else float(pd.to_numeric(eligible_frame["snapshot_w4_return_pct"], errors="coerce").mean())
        ),
        "rejected_snapshot_w4_mean": (
            None if rejected_frame.empty
            else float(pd.to_numeric(rejected_frame["snapshot_w4_return_pct"], errors="coerce").mean())
        ),
        "gate_mean_lift": (
            None
            if eligible_frame.empty or rejected_frame.empty
            else float(
                pd.to_numeric(eligible_frame["snapshot_w4_return_pct"], errors="coerce").mean()
                - pd.to_numeric(rejected_frame["snapshot_w4_return_pct"], errors="coerce").mean()
            )
        ),
    }


def _ranking_rows(s_df: pd.DataFrame, selected_codes: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    eligible = s_df[s_df["current_b0_eligible"]].copy()
    eligible = eligible.sort_values(["current_b0_raw_rank", "code"], kind="stable")
    eligible["eligible_rank"] = range(1, len(eligible) + 1)
    mature = eligible[
        pd.to_numeric(eligible["w4_return_pct"], errors="coerce").notna()
        & eligible["w4_stop8"].notna()
    ].copy()

    coverage = len(mature) / float(len(eligible)) if len(eligible) else 0.0
    corr = None
    if len(eligible) > 0 and coverage >= ELIGIBLE_ENTRY_COVERAGE_MIN_FOR_PRIMARY:
        corr = safe_spearman(eligible["eligible_rank"], eligible["w4_return_pct"])

    selected_port = portfolio_from_codes(
        s_df,
        selected_codes,
        return_col="w4_return_pct",
        stop_col="w4_stop8",
    )
    eligible_mean = (
        None
        if len(eligible) == 0 or coverage < ELIGIBLE_ENTRY_COVERAGE_MIN_FOR_PRIMARY
        else float(pd.to_numeric(eligible["w4_return_pct"], errors="coerce").mean())
    )

    weekly = {
        "eligible_count": int(len(eligible)),
        "eligible_mature_count": int(len(mature)),
        "eligible_entry_coverage": round(coverage, 6),
        "primary_valid": bool(
            len(eligible) > 0
            and coverage >= ELIGIBLE_ENTRY_COVERAGE_MIN_FOR_PRIMARY
            and selected_port["mature"]
        ),
        "weekly_spearman": corr,
        "eligible_mean_w4": eligible_mean,
        "b0_selection_quality_w4": (
            selected_port["selection_quality_return"] if selected_port["mature"] else None
        ),
        "b0_minus_eligible_mean": (
            None
            if eligible_mean is None or not selected_port["mature"] or len(selected_codes) == 0
            else float(selected_port["selection_quality_return"] - eligible_mean)
        ),
    }

    buckets: list[dict[str, Any]] = []
    if coverage >= ELIGIBLE_ENTRY_COVERAGE_MIN_FOR_PRIMARY:
        for _, row in eligible.iterrows():
            rank = int(row["eligible_rank"])
            if rank <= 3:
                bucket = "rank_1_3"
            elif rank <= 6:
                bucket = "rank_4_6"
            elif rank <= 10:
                bucket = "rank_7_10"
            else:
                bucket = "rank_11_plus"
            buckets.append({
                "code": str(row["code"]),
                "eligible_rank": rank,
                "rank_bucket": bucket,
                "w4_return_pct": float(row["w4_return_pct"]),
                "w4_stop8": bool(row["w4_stop8"]),
                "selected": str(row["code"]) in set(selected_codes),
            })
    return weekly, buckets


def materialize_core() -> dict[str, Any]:
    panel, market_benchmarks, _ = build_audit_frame()
    manifest = source_manifest(panel)

    b0_weekly: list[dict[str, Any]] = []
    eligible_random_weekly: list[dict[str, Any]] = []
    raw_random_weekly: list[dict[str, Any]] = []
    eligibility_weekly: list[dict[str, Any]] = []
    ranking_weekly: list[dict[str, Any]] = []
    rank_bucket_rows: list[dict[str, Any]] = []
    simple_rows: list[dict[str, Any]] = []
    rejection_reason_rows: list[dict[str, Any]] = []

    market_lookup = {
        (str(row["snapshot_date"]), str(row["code"])): row
        for _, row in market_benchmarks.iterrows()
    }

    for snapshot in sorted(panel["snapshot_date"].unique().tolist()):
        s_df = panel[panel["snapshot_date"] == snapshot].copy()
        selected_codes = _selected_codes(s_df)
        k = len(selected_codes)

        entry_port = portfolio_from_codes(
            s_df,
            selected_codes,
            return_col="w4_return_pct",
            stop_col="w4_stop8",
        )
        snap_port = portfolio_from_codes(
            s_df,
            selected_codes,
            return_col="snapshot_w4_return_pct",
            stop_col="snapshot_w4_stop8",
        )

        spy_row = market_lookup.get((snapshot, "SPY"))
        qqq_row = market_lookup.get((snapshot, "QQQ"))
        spy_ret = (
            None
            if spy_row is None or not bool(spy_row.get("snapshot_price_valid"))
            else float(spy_row["snapshot_w4_return_pct"])
        )
        qqq_ret = (
            None
            if qqq_row is None or not bool(qqq_row.get("snapshot_price_valid"))
            else float(qqq_row["snapshot_w4_return_pct"])
        )

        b0_weekly.append({
            "snapshot_date": snapshot,
            "pick_count": k,
            "selected_codes": json.dumps(selected_codes),
            "entry_w4_mature": bool(entry_port["mature"]),
            "entry_w4_selection_quality": (
                entry_port["selection_quality_return"] if entry_port["mature"] else None
            ),
            "entry_w4_capital_adjusted": (
                entry_port["capital_adjusted_return"] if entry_port["mature"] else None
            ),
            "entry_w4_capital_stop8": (
                entry_port["capital_adjusted_stop8"] if entry_port["mature"] else None
            ),
            "entry_w4_one_pick_ruined": (
                entry_port["one_pick_ruined"] if entry_port["mature"] else None
            ),
            "snapshot_w4_mature": bool(snap_port["mature"]),
            "snapshot_w4_selection_quality": (
                snap_port["selection_quality_return"] if snap_port["mature"] else None
            ),
            "snapshot_w4_capital_adjusted": (
                snap_port["capital_adjusted_return"] if snap_port["mature"] else None
            ),
            "spy_w4": spy_ret,
            "spy_exposure_matched_w4": None if spy_ret is None else spy_ret * k / float(TOP_N),
            "qqq_w4": qqq_ret,
            "qqq_exposure_matched_w4": None if qqq_ret is None else qqq_ret * k / float(TOP_N),
        })

        # Current eligible universe: strict full-maturity primary benchmark.
        eligible = s_df[s_df["current_b0_eligible"]].copy()
        eligible_mature = eligible[
            pd.to_numeric(eligible["w4_return_pct"], errors="coerce").notna()
            & eligible["w4_stop8"].notna()
        ].copy()
        eligible_cov = (
            len(eligible_mature) / float(len(eligible))
            if len(eligible) else 0.0
        )
        eligible_primary = bool(
            k > 0
            and entry_port["mature"]
            and len(eligible) >= k
            and eligible_cov >= ELIGIBLE_ENTRY_COVERAGE_MIN_FOR_PRIMARY
        )

        elig_record: dict[str, Any] = {
            "snapshot_date": snapshot,
            "pick_count": k,
            "eligible_count": int(len(eligible)),
            "eligible_mature_count": int(len(eligible_mature)),
            "eligible_entry_coverage": round(eligible_cov, 6),
            "primary_valid": eligible_primary,
            "b0_return": (
                float(entry_port["capital_adjusted_return"])
                if entry_port["mature"] else None
            ),
        }
        if eligible_primary:
            dist, dist_meta = portfolio_distribution(
                eligible,
                k=k,
                return_col="w4_return_pct",
                distinct_industry=True,
                seed=_seed_for_snapshot(snapshot, 101),
            )
            oracle_codes, oracle_val = _oracle_value(
                eligible,
                k=k,
                return_col="w4_return_pct",
                distinct_industry=True,
            )
            if len(dist) == 0 or oracle_val is None:
                elig_record["primary_valid"] = False
            else:
                elig_record.update(distribution_summary(
                    float(entry_port["capital_adjusted_return"]),
                    dist,
                    oracle_val,
                ))
                elig_record.update({
                    "oracle_codes": json.dumps(oracle_codes),
                    **{f"distribution_{name}": value for name, value in dist_meta.items()},
                })

                dist_u, meta_u = portfolio_distribution(
                    eligible,
                    k=k,
                    return_col="w4_return_pct",
                    distinct_industry=False,
                    seed=_seed_for_snapshot(snapshot, 151),
                )
                oracle_u_codes, oracle_u = _oracle_value(
                    eligible,
                    k=k,
                    return_col="w4_return_pct",
                    distinct_industry=False,
                )
                if len(dist_u) > 0 and oracle_u is not None:
                    u = distribution_summary(
                        float(entry_port["capital_adjusted_return"]),
                        dist_u,
                        oracle_u,
                    )
                    elig_record.update({f"unconstrained_{key}": val for key, val in u.items()})
                    elig_record["unconstrained_oracle_codes"] = json.dumps(oracle_u_codes)
                    elig_record.update({
                        f"unconstrained_distribution_{name}": value
                        for name, value in meta_u.items()
                    })
        eligible_random_weekly.append(elig_record)

        # Raw signal universe: common snapshot-close outcome. Distinct1 is primary
        # random benchmark; unconstrained is a secondary total-system diagnostic.
        raw_covered = s_df[s_df["snapshot_price_valid"] == True].copy()
        raw_cov = len(raw_covered) / float(len(s_df)) if len(s_df) else 0.0
        fixed_k = raw_fixed_capacity_k(raw_covered)
        raw_primary = bool(
            snap_port["mature"]
            and raw_cov >= RAW_PRICE_COVERAGE_MIN_FOR_PRIMARY
            and fixed_k > 0
        )
        raw_record: dict[str, Any] = {
            "snapshot_date": snapshot,
            "pick_count": k,
            "fixed_capacity_pick_count": fixed_k,
            "raw_count": int(len(s_df)),
            "price_covered_count": int(len(raw_covered)),
            "price_coverage": round(raw_cov, 6),
            "primary_valid": raw_primary,
            "b0_return": (
                float(snap_port["capital_adjusted_return"])
                if snap_port["mature"] else None
            ),
            "b0_selection_quality_return": (
                float(snap_port["selection_quality_return"])
                if snap_port["mature"] else None
            ),
            "raw_equal_weight_mean": (
                None if raw_covered.empty
                else float(pd.to_numeric(raw_covered["snapshot_w4_return_pct"], errors="coerce").mean())
            ),
        }

        if raw_primary and k > 0:
            dist_d1, meta_d1 = portfolio_distribution(
                raw_covered,
                k=k,
                return_col="snapshot_w4_return_pct",
                distinct_industry=True,
                seed=_seed_for_snapshot(snapshot, 201),
            )
            oracle_d1_codes, oracle_d1 = _oracle_value(
                raw_covered,
                k=k,
                return_col="snapshot_w4_return_pct",
                distinct_industry=True,
            )
            if len(dist_d1) > 0 and oracle_d1 is not None:
                d1 = distribution_summary(
                    float(snap_port["capital_adjusted_return"]),
                    dist_d1,
                    oracle_d1,
                )
                raw_record.update({f"distinct1_{key}": val for key, val in d1.items()})
                raw_record["distinct1_oracle_codes"] = json.dumps(oracle_d1_codes)
                raw_record.update({
                    f"distinct1_distribution_{key}": val
                    for key, val in meta_d1.items()
                })
            else:
                raw_record["primary_valid"] = False

            dist_u, meta_u = portfolio_distribution(
                raw_covered,
                k=k,
                return_col="snapshot_w4_return_pct",
                distinct_industry=False,
                seed=_seed_for_snapshot(snapshot, 301),
            )
            oracle_u_codes, oracle_u = _oracle_value(
                raw_covered,
                k=k,
                return_col="snapshot_w4_return_pct",
                distinct_industry=False,
            )
            if len(dist_u) > 0 and oracle_u is not None:
                u = distribution_summary(
                    float(snap_port["capital_adjusted_return"]),
                    dist_u,
                    oracle_u,
                )
                raw_record.update({f"unconstrained_{key}": val for key, val in u.items()})
                raw_record["unconstrained_oracle_codes"] = json.dumps(oracle_u_codes)
                raw_record.update({
                    f"unconstrained_distribution_{key}": val
                    for key, val in meta_u.items()
                })

        if raw_primary and fixed_k > 0:
            fixed_dist, fixed_meta = portfolio_distribution(
                raw_covered,
                k=fixed_k,
                return_col="snapshot_w4_return_pct",
                distinct_industry=True,
                seed=_seed_for_snapshot(snapshot, 401),
            )
            fixed_oracle_codes, fixed_oracle = _oracle_value(
                raw_covered,
                k=fixed_k,
                return_col="snapshot_w4_return_pct",
                distinct_industry=True,
            )
            if len(fixed_dist) > 0 and fixed_oracle is not None:
                fixed = distribution_summary(
                    float(snap_port["capital_adjusted_return"]),
                    fixed_dist,
                    fixed_oracle,
                )
                raw_record.update({f"fixed_capacity_{key}": val for key, val in fixed.items()})
                raw_record["fixed_capacity_oracle_codes"] = json.dumps(fixed_oracle_codes)
                raw_record.update({
                    f"fixed_capacity_distribution_{key}": val
                    for key, val in fixed_meta.items()
                })
            else:
                raw_record["primary_valid"] = False

        raw_random_weekly.append(raw_record)

        gate = _winner_gate_row(s_df, selected_codes)
        gate["snapshot_date"] = snapshot
        eligibility_weekly.append(gate)

        covered_gate = s_df[s_df["snapshot_price_valid"] == True].copy()
        if gate.get("primary_valid") and not covered_gate.empty:
            covered_gate = covered_gate.sort_values(
                ["snapshot_w4_return_pct", "code"],
                ascending=[False, True],
                kind="stable",
            )
            top_n_gate = max(1, int(math.ceil(len(covered_gate) * WINNER_TOP_FRAC)))
            winner_codes_gate = set(covered_gate.head(top_n_gate)["code"].astype(str))
            for _, rr in covered_gate[~covered_gate["current_b0_eligible"]].iterrows():
                reasons = [
                    x for x in str(rr.get("current_b0_reject_reasons", "")).split("|")
                    if x
                ]
                for reason in reasons:
                    rejection_reason_rows.append({
                        "snapshot_date": snapshot,
                        "code": str(rr["code"]),
                        "reason": reason,
                        "snapshot_w4_return_pct": float(rr["snapshot_w4_return_pct"]),
                        "is_top20_winner": str(rr["code"]) in winner_codes_gate,
                        "is_big_winner": float(rr["snapshot_w4_return_pct"]) >= BIG_WINNER_THRESHOLD_PCT,
                    })

        ranking, bucket_rows = _ranking_rows(s_df, selected_codes)
        ranking["snapshot_date"] = snapshot
        ranking_weekly.append(ranking)
        for row in bucket_rows:
            row["snapshot_date"] = snapshot
            rank_bucket_rows.append(row)

        for baseline_name, feature, direction in SIMPLE_BASELINES:
            codes = _simple_baseline_codes(
                s_df,
                k=fixed_k,
                feature=feature,
                direction=direction,
            )
            p = portfolio_from_codes(
                s_df,
                codes,
                return_col="snapshot_w4_return_pct",
                stop_col="snapshot_w4_stop8",
            )
            simple_rows.append({
                "snapshot_date": snapshot,
                "raw_primary_valid": raw_primary,
                "baseline": baseline_name,
                "feature": feature,
                "pick_count_target": fixed_k,
                "pick_count": len(codes),
                "codes": json.dumps(codes),
                "mature": bool(p["mature"]),
                "capital_adjusted_return": (
                    p["capital_adjusted_return"] if p["mature"] else None
                ),
                "selection_quality_return": (
                    p["selection_quality_return"] if p["mature"] else None
                ),
            })

    frames = {
        "panel": panel,
        "b0_weekly": pd.DataFrame(b0_weekly),
        "eligible_random_weekly": pd.DataFrame(eligible_random_weekly),
        "raw_random_weekly": pd.DataFrame(raw_random_weekly),
        "eligibility_weekly": pd.DataFrame(eligibility_weekly),
        "ranking_weekly": pd.DataFrame(ranking_weekly),
        "rank_bucket_rows": pd.DataFrame(rank_bucket_rows),
        "simple_baseline_weekly": pd.DataFrame(simple_rows),
        "rejection_reason_rows": pd.DataFrame(rejection_reason_rows),
    }
    return {"frames": frames, "manifest": manifest}


def _rank_bucket_summary(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    out = (
        rows.groupby("rank_bucket", sort=False)
        .agg(
            candidate_rows=("code", "size"),
            weeks=("snapshot_date", "nunique"),
            mean_w4=("w4_return_pct", "mean"),
            median_w4=("w4_return_pct", "median"),
            positive_rate=("w4_return_pct", lambda s: float((s > 0).mean())),
            stop8_rate=("w4_stop8", "mean"),
            selected_rate=("selected", "mean"),
        )
        .reset_index()
    )
    return out


def _rejection_reason_summary(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    return (
        rows.groupby("reason", sort=False)
        .agg(
            rejected_candidate_rows=("code", "size"),
            weeks=("snapshot_date", "nunique"),
            mean_snapshot_w4=("snapshot_w4_return_pct", "mean"),
            median_snapshot_w4=("snapshot_w4_return_pct", "median"),
            top20_winner_rows=("is_top20_winner", "sum"),
            top20_winner_rate=("is_top20_winner", "mean"),
            big_winner_rows=("is_big_winner", "sum"),
            big_winner_rate=("is_big_winner", "mean"),
        )
        .reset_index()
    )


def _simple_summary(simple: pd.DataFrame, b0_weekly: pd.DataFrame) -> pd.DataFrame:
    if simple.empty:
        return pd.DataFrame()
    simple = simple[simple["raw_primary_valid"] == True].copy()
    if simple.empty:
        return pd.DataFrame()
    b0 = b0_weekly[
        ["snapshot_date", "snapshot_w4_capital_adjusted"]
    ].rename(columns={"snapshot_w4_capital_adjusted": "b0_return"})

    rows: list[dict[str, Any]] = []
    for baseline, group in simple.groupby("baseline"):
        merged = group.merge(b0, on="snapshot_date", how="left")
        merged = merged[
            (merged["mature"] == True)
            & merged["capital_adjusted_return"].notna()
            & merged["b0_return"].notna()
        ].copy()
        spread = (
            merged["capital_adjusted_return"].astype(float)
            - merged["b0_return"].astype(float)
        )
        rows.append({
            "baseline": baseline,
            "support_weeks": int(len(merged)),
            "mean_return": None if merged.empty else float(merged["capital_adjusted_return"].mean()),
            "median_return": None if merged.empty else float(merged["capital_adjusted_return"].median()),
            "mean_spread_vs_b0": None if merged.empty else float(spread.mean()),
            "median_spread_vs_b0": None if merged.empty else float(spread.median()),
            "beat_b0_rate": None if merged.empty else float((spread > 0).mean()),
            "mean_pick_coverage": (
                None if merged.empty
                else float((merged["pick_count"] / merged["pick_count_target"].replace(0, np.nan)).mean())
            ),
        })
    return pd.DataFrame(rows)


def summarize_core(core: dict[str, Any]) -> dict[str, Any]:
    f = core["frames"]
    b0 = f["b0_weekly"]
    eligible = f["eligible_random_weekly"]
    raw = f["raw_random_weekly"]
    gate = f["eligibility_weekly"]
    ranking = f["ranking_weekly"]

    mature_b0 = b0[b0["entry_w4_mature"] == True].copy()
    b0_rets = pd.to_numeric(
        mature_b0["entry_w4_capital_adjusted"], errors="coerce"
    ).dropna().to_numpy(dtype=float)

    absolute = {
        "entry_aligned_w4": basic_distribution_stats(b0_rets),
        "entry_aligned_block_bootstrap": moving_block_bootstrap_ci(b0_rets),
        "mean_capital_stop8_pct": (
            None if mature_b0.empty
            else float(pd.to_numeric(mature_b0["entry_w4_capital_stop8"], errors="coerce").mean())
        ),
        "one_pick_ruin_week_rate": (
            None if mature_b0.empty
            else float(mature_b0["entry_w4_one_pick_ruined"].astype(bool).mean())
        ),
        "mean_slot_coverage": (
            float((b0["pick_count"] / float(TOP_N)).mean()) if not b0.empty else None
        ),
        "full_top3_rate": (
            float((b0["pick_count"] == TOP_N).mean()) if not b0.empty else None
        ),
        "zero_pick_weeks": int((b0["pick_count"] == 0).sum()) if not b0.empty else 0,
    }

    elig_valid = eligible[eligible["primary_valid"] == True].copy()
    eligible_edge = paired_edge_summary(
        elig_valid,
        "b0_return",
        "random_mean",
    ) if not elig_valid.empty and "random_mean" in elig_valid.columns else {}
    eligible_percentiles = pd.to_numeric(
        elig_valid.get("b0_percentile", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    eligible_capture = aggregate_oracle_capture(elig_valid) if not elig_valid.empty else {}

    eligible_unconstrained_edge = {}
    if (
        not elig_valid.empty
        and "unconstrained_random_mean" in elig_valid.columns
    ):
        eligible_unconstrained_edge = paired_edge_summary(
            elig_valid,
            "b0_return",
            "unconstrained_random_mean",
        )

    raw_valid = raw[raw["primary_valid"] == True].copy()
    raw_edge = paired_edge_summary(
        raw_valid,
        "b0_return",
        "fixed_capacity_random_mean",
    ) if not raw_valid.empty and "fixed_capacity_random_mean" in raw_valid.columns else {}
    raw_percentiles = pd.to_numeric(
        raw_valid.get("fixed_capacity_b0_percentile", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    raw_matched_edge = (
        paired_edge_summary(raw_valid, "b0_return", "distinct1_random_mean")
        if not raw_valid.empty and "distinct1_random_mean" in raw_valid.columns
        else {}
    )
    raw_matched_percentiles = pd.to_numeric(
        raw_valid.get("distinct1_b0_percentile", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    raw_capture_frame = pd.DataFrame()
    raw_capture = {}
    raw_unconstrained_edge = {}
    if not raw_valid.empty:
        cols = {
            "b0_return": "b0_return",
            "fixed_capacity_random_mean": "random_mean",
            "fixed_capacity_oracle": "oracle",
            "fixed_capacity_oracle_capture_ratio": "oracle_capture_ratio",
        }
        if all(c in raw_valid.columns for c in cols):
            raw_capture_frame = raw_valid[list(cols)].rename(columns=cols)
            raw_capture = aggregate_oracle_capture(raw_capture_frame)
        if "unconstrained_random_mean" in raw_valid.columns:
            raw_unconstrained_edge = paired_edge_summary(
                raw_valid,
                "b0_return",
                "unconstrained_random_mean",
            )

    gate_valid = gate[gate["primary_valid"] == True].copy()
    def weighted_ratio(num_col: str, den_col: str) -> float | None:
        if gate_valid.empty:
            return None
        num = pd.to_numeric(gate_valid[num_col], errors="coerce").sum()
        den = pd.to_numeric(gate_valid[den_col], errors="coerce").sum()
        return None if den <= 0 else float(num / den)

    eligibility_summary = {
        "support_weeks": int(len(gate_valid)),
        "mean_raw_price_coverage": (
            None if gate.empty else float(gate["price_coverage"].mean())
        ),
        "winner_retention_rate": weighted_ratio(
            "winner_retained_by_eligibility", "winner_count"
        ),
        "b0_winner_capture_rate": weighted_ratio(
            "winner_captured_by_b0", "winner_count"
        ),
        "rejected_winner_rate": weighted_ratio(
            "rejected_winner_count", "rejected_count"
        ),
        "bottom_loser_rejection_rate": weighted_ratio(
            "bottom_loser_rejected_count", "bottom_loser_count"
        ),
        "big_winner_retention_rate": weighted_ratio(
            "big_winner_retained_count", "big_winner_count"
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

    ranking_valid = ranking[ranking["primary_valid"] == True].copy()
    spearman = pd.to_numeric(
        ranking_valid["weekly_spearman"], errors="coerce"
    ).dropna()
    ranking_lift = pd.to_numeric(
        ranking_valid["b0_minus_eligible_mean"], errors="coerce"
    ).dropna()
    ranking_summary = {
        "support_weeks": int(len(ranking_valid)),
        "weekly_spearman_mean": None if spearman.empty else float(spearman.mean()),
        "weekly_spearman_median": None if spearman.empty else float(spearman.median()),
        "positive_spearman_week_rate": None if spearman.empty else float((spearman > 0).mean()),
        "b0_minus_eligible_mean_mean": None if ranking_lift.empty else float(ranking_lift.mean()),
        "b0_minus_eligible_mean_median": None if ranking_lift.empty else float(ranking_lift.median()),
        "ranking_lift_block_bootstrap": moving_block_bootstrap_ci(ranking_lift.to_numpy()),
    }

    market = b0[
        [
            "snapshot_date",
            "snapshot_w4_selection_quality",
            "snapshot_w4_capital_adjusted",
            "spy_w4",
            "spy_exposure_matched_w4",
            "qqq_w4",
            "qqq_exposure_matched_w4",
        ]
    ].copy()
    market_summary = {
        "vs_spy_selection_quality": paired_edge_summary(
            market, "snapshot_w4_selection_quality", "spy_w4"
        ),
        "vs_spy_exposure_matched": paired_edge_summary(
            market, "snapshot_w4_capital_adjusted", "spy_exposure_matched_w4"
        ),
        "vs_spy_full_exposure": paired_edge_summary(
            market, "snapshot_w4_capital_adjusted", "spy_w4"
        ),
        "vs_qqq_selection_quality": paired_edge_summary(
            market, "snapshot_w4_selection_quality", "qqq_w4"
        ),
        "vs_qqq_full_exposure": paired_edge_summary(
            market, "snapshot_w4_capital_adjusted", "qqq_w4"
        ),
    }

    eligible_pct_median = (
        None if eligible_percentiles.empty else float(eligible_percentiles.median())
    )
    raw_pct_median = (
        None if raw_percentiles.empty else float(raw_percentiles.median())
    )
    raw_agg_capture = raw_capture.get("aggregate_capture_ratio") if raw_capture else None

    offsets = []
    off_abs = four_offset_nonoverlap(
        mature_b0,
        value_col="entry_w4_capital_adjusted",
    )
    off_abs["comparison"] = "b0_absolute_entry_w4"
    offsets.append(off_abs)

    if not elig_valid.empty and "random_mean" in elig_valid.columns:
        off = four_offset_nonoverlap(
            elig_valid,
            value_col="b0_return",
            benchmark_col="random_mean",
        )
        off["comparison"] = "b0_vs_eligible_random"
        offsets.append(off)

    if not raw_valid.empty and "fixed_capacity_random_mean" in raw_valid.columns:
        off = four_offset_nonoverlap(
            raw_valid,
            value_col="b0_return",
            benchmark_col="fixed_capacity_random_mean",
        )
        off["comparison"] = "b0_vs_raw_random_fixed_capacity"
        offsets.append(off)

    nonoverlap = pd.concat(offsets, ignore_index=True) if offsets else pd.DataFrame()

    rank_bucket_summary = _rank_bucket_summary(f["rank_bucket_rows"])
    simple_summary = _simple_summary(f["simple_baseline_weekly"], b0)
    rejection_reason_summary = _rejection_reason_summary(f["rejection_reason_rows"])

    health = {
        "summary_policy": "NO_ARBITRARY_PASS_FAIL_THRESHOLD",
        "absolute": absolute,
        "eligible_random": {
            "support_weeks": int(len(elig_valid)),
            "median_weekly_percentile": eligible_pct_median,
            "mean_weekly_percentile": (
                None if eligible_percentiles.empty else float(eligible_percentiles.mean())
            ),
            "edge": eligible_edge,
            "oracle_capture": eligible_capture,
            "unconstrained_edge": eligible_unconstrained_edge,
        },
        "raw_random_distinct1": {
            "support_weeks": int(len(raw_valid)),
            "mode": "fixed_capacity_up_to_3",
            "median_weekly_percentile": raw_pct_median,
            "mean_weekly_percentile": (
                None if raw_percentiles.empty else float(raw_percentiles.mean())
            ),
            "edge": raw_edge,
            "oracle_capture": raw_capture,
            "matched_n": {
                "median_weekly_percentile": (
                    None if raw_matched_percentiles.empty else float(raw_matched_percentiles.median())
                ),
                "mean_weekly_percentile": (
                    None if raw_matched_percentiles.empty else float(raw_matched_percentiles.mean())
                ),
                "edge": raw_matched_edge,
            },
            "unconstrained_edge": raw_unconstrained_edge,
        },
        "eligibility": eligibility_summary,
        "ranking": ranking_summary,
        "market": market_summary,
        "evidence_boundary": (
            "Retrospective diagnostic only. Current B0 and its components were developed "
            "with substantial visibility into this historical period; no p-value or CI "
            "is treated as virgin OOS proof."
        ),
    }

    return {
        "health": health,
        "nonoverlap_offsets": nonoverlap,
        "rank_bucket_summary": rank_bucket_summary,
        "simple_baseline_summary": simple_summary,
        "rejection_reason_summary": rejection_reason_summary,
    }
