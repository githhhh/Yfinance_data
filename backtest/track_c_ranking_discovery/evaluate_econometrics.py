from __future__ import annotations
import dataclasses
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from .config import (
    BLOCK_BOOTSTRAP_BLOCK_LEN,
    BOOTSTRAP_ROUNDS,
    LOWO_MAX_POSITIVE_EDGE_CONCENTRATION,
    LOWO_MIN_SIGN_STABILITY,
    RANDOM_SEED,
)
from .protocol import WeeklyPortfolioOutcome


@dataclass
class LowoFragilityResult:
    """Standardized Leave-One-Week-Out (LOWO) Fragility Analysis Result."""
    full_mean_spread: float
    min_lowo_mean: float
    max_lowo_mean: float
    positive_edge_concentration: float
    sign_stability: float
    largest_abs_influence: float
    is_fragile_overfit: bool


@dataclass
class BlockBootstrapResult:
    """Moving Block Bootstrap (block_len=4) CI Result for overlapping return inference."""
    mean_spread_ci_low: float
    mean_spread_ci_high: float
    median_spread_ci_low: float
    median_spread_ci_high: float


@dataclass
class PairedEvaluationSummary:
    """Complete multi-dimensional evaluation summary for a challenger vs B0."""
    selector_id: str
    family: str
    segment: str
    horizon: str
    support_weeks: int
    challenger_mean: float
    b0_mean: float
    mean_spread: float
    challenger_median: float
    b0_median: float
    median_spread: float
    challenger_cvar10: float
    b0_cvar10: float
    cvar_delta: float
    challenger_p10: float
    b0_p10: float
    challenger_stop8_pct: float
    b0_stop8_pct: float
    stop_delta_pct: float
    challenger_one_pick_ruins_pct: float
    b0_one_pick_ruins_pct: float
    one_pick_ruins_delta_pct: float
    slot_coverage_pct: float
    full_top3_rate_pct: float
    top3_membership_jaccard_vs_b0: float
    lowo: LowoFragilityResult | None = None
    bootstrap: BlockBootstrapResult | None = None
    pareto_score: float = 0.0
    classification: str = ""


def compute_lowo_fragility(
    challenger_rets: np.ndarray,
    b0_rets: np.ndarray,
) -> LowoFragilityResult:
    """Compute Leave-One-Week-Out (LOWO) sensitivity and positive edge concentration."""
    spreads = challenger_rets - b0_rets
    T = len(spreads)
    if T <= 2:
        return LowoFragilityResult(
            full_mean_spread=float(np.mean(spreads)) if T > 0 else 0.0,
            min_lowo_mean=0.0,
            max_lowo_mean=0.0,
            positive_edge_concentration=0.0,
            sign_stability=1.0,
            largest_abs_influence=0.0,
            is_fragile_overfit=False,
        )

    full_mean = float(np.mean(spreads))
    lowo_means = np.zeros(T, dtype=float)
    for t in range(T):
        sub = np.delete(spreads, t)
        lowo_means[t] = float(np.mean(sub))

    min_l = float(np.min(lowo_means))
    max_l = float(np.max(lowo_means))
    sign_stab = float(np.mean(lowo_means > 0.0))
    largest_abs_inf = float(np.max(np.abs(full_mean - lowo_means)))

    # Positive Edge Concentration
    pos_spreads = np.maximum(spreads, 0.0)
    sum_pos = float(np.sum(pos_spreads))
    if full_mean > 0.0 and sum_pos > 1e-6:
        pos_edge_conc = float(np.max(pos_spreads) / sum_pos)
    else:
        pos_edge_conc = 0.0

    # Strictly Pre-Registered Fragility Rule
    is_fragile = bool(
        pos_edge_conc > LOWO_MAX_POSITIVE_EDGE_CONCENTRATION and
        sign_stab < LOWO_MIN_SIGN_STABILITY
    )

    return LowoFragilityResult(
        full_mean_spread=round(full_mean, 4),
        min_lowo_mean=round(min_l, 4),
        max_lowo_mean=round(max_l, 4),
        positive_edge_concentration=round(pos_edge_conc, 4),
        sign_stability=round(sign_stab, 4),
        largest_abs_influence=round(largest_abs_inf, 4),
        is_fragile_overfit=is_fragile,
    )


def compute_moving_block_bootstrap(
    challenger_rets: np.ndarray,
    b0_rets: np.ndarray,
    block_len: int = BLOCK_BOOTSTRAP_BLOCK_LEN,
    n_rounds: int = BOOTSTRAP_ROUNDS,
    seed: int = RANDOM_SEED,
) -> BlockBootstrapResult:
    """Compute 4-week Moving Block Bootstrap 95% Confidence Intervals."""
    spreads = challenger_rets - b0_rets
    T = len(spreads)
    if T < block_len:
        return BlockBootstrapResult(
            mean_spread_ci_low=round(float(np.mean(spreads)), 4),
            mean_spread_ci_high=round(float(np.mean(spreads)), 4),
            median_spread_ci_low=round(float(np.median(spreads)), 4),
            median_spread_ci_high=round(float(np.median(spreads)), 4),
        )

    rng = np.random.default_rng(seed)
    num_blocks = int(np.ceil(T / float(block_len)))

    # All possible starting indices for blocks
    block_starts = list(range(T - block_len + 1))
    means = np.zeros(n_rounds, dtype=float)
    medians = np.zeros(n_rounds, dtype=float)

    for r in range(n_rounds):
        chosen_starts = rng.choice(block_starts, size=num_blocks, replace=True)
        resampled = []
        for st in chosen_starts:
            resampled.extend(spreads[st : st + block_len])
        resampled_arr = np.array(resampled[:T])
        means[r] = float(np.mean(resampled_arr))
        medians[r] = float(np.median(resampled_arr))

    return BlockBootstrapResult(
        mean_spread_ci_low=round(float(np.percentile(means, 2.5)), 4),
        mean_spread_ci_high=round(float(np.percentile(means, 97.5)), 4),
        median_spread_ci_low=round(float(np.percentile(medians, 2.5)), 4),
        median_spread_ci_high=round(float(np.percentile(medians, 97.5)), 4),
    )


def evaluate_paired_challenger(
    ch_outcomes: list[WeeklyPortfolioOutcome],
    b0_outcomes: list[WeeklyPortfolioOutcome],
    selector_id: str,
    family: str,
    segment: str,
    horizon: str = "W4",
) -> PairedEvaluationSummary:
    """Evaluate paired portfolio performance against B0 strictly on identical mature support."""
    ch_map = {o.snapshot_date: o for o in ch_outcomes}
    b0_map = {o.snapshot_date: o for o in b0_outcomes}

    # Find identical common support of mature weeks
    common_snaps = sorted(
        set(s for s, o in ch_map.items() if o.is_mature) &
        set(s for s, o in b0_map.items() if o.is_mature)
    )

    if not common_snaps:
        return PairedEvaluationSummary(
            selector_id=selector_id,
            family=family,
            segment=segment,
            horizon=horizon,
            support_weeks=0,
            challenger_mean=0.0,
            b0_mean=0.0,
            mean_spread=0.0,
            challenger_median=0.0,
            b0_median=0.0,
            median_spread=0.0,
            challenger_cvar10=0.0,
            b0_cvar10=0.0,
            cvar_delta=0.0,
            challenger_p10=0.0,
            b0_p10=0.0,
            challenger_stop8_pct=0.0,
            b0_stop8_pct=0.0,
            stop_delta_pct=0.0,
            challenger_one_pick_ruins_pct=0.0,
            b0_one_pick_ruins_pct=0.0,
            one_pick_ruins_delta_pct=0.0,
            slot_coverage_pct=0.0,
            full_top3_rate_pct=0.0,
            top3_membership_jaccard_vs_b0=0.0,
        )

    ch_rets = np.array([ch_map[s].capital_adjusted_return for s in common_snaps])
    b0_rets = np.array([b0_map[s].capital_adjusted_return for s in common_snaps])

    ch_stops = np.array([ch_map[s].capital_adjusted_stop8 for s in common_snaps])
    b0_stops = np.array([b0_map[s].capital_adjusted_stop8 for s in common_snaps])

    ch_ruins = np.array([ch_map[s].one_pick_ruined for s in common_snaps])
    b0_ruins = np.array([b0_map[s].one_pick_ruined for s in common_snaps])

    ch_mean = float(np.mean(ch_rets))
    b0_mean = float(np.mean(b0_rets))
    ch_med = float(np.median(ch_rets))
    b0_med = float(np.median(b0_rets))

    # CVaR 10%
    n_tail = max(1, int(len(common_snaps) * 0.1))
    ch_cvar = float(np.mean(np.sort(ch_rets)[:n_tail]))
    b0_cvar = float(np.mean(np.sort(b0_rets)[:n_tail]))
    ch_p10 = float(np.percentile(ch_rets, 10))
    b0_p10 = float(np.percentile(b0_rets, 10))

    # Top3 Membership Jaccard similarity vs B0
    jaccards = []
    for s in common_snaps:
        set_ch = set(ch_map[s].selected_codes)
        set_b0 = set(b0_map[s].selected_codes)
        union = set_ch | set_b0
        if not union:
            jaccards.append(1.0)
        else:
            jaccards.append(len(set_ch & set_b0) / float(len(union)))
    avg_jaccard = float(np.mean(jaccards))

    # Coverage & Abstention metrics
    cov_pct = float(np.mean([ch_map[s].slot_coverage for s in common_snaps]) * 100.0)
    full_top3_pct = float(np.mean([ch_map[s].full_top3 for s in common_snaps]) * 100.0)

    # Econometric Fragility & Block Bootstrap
    lowo_res = compute_lowo_fragility(ch_rets, b0_rets)
    boot_res = compute_moving_block_bootstrap(ch_rets, b0_rets)

    # Pareto Score: Return Spread + 0.5 * CVaR Spread - 0.2 * Stop Spread - Penalty if Fragile
    pareto = (ch_med - b0_med) + 0.5 * (ch_cvar - b0_cvar) - 0.2 * (float(np.mean(ch_stops)) - float(np.mean(b0_stops)))
    if lowo_res.is_fragile_overfit:
        pareto -= 50.0  # Heavy penalty for fragile overfit

    return PairedEvaluationSummary(
        selector_id=selector_id,
        family=family,
        segment=segment,
        horizon=horizon,
        support_weeks=len(common_snaps),
        challenger_mean=round(ch_mean, 4),
        b0_mean=round(b0_mean, 4),
        mean_spread=round(ch_mean - b0_mean, 4),
        challenger_median=round(ch_med, 4),
        b0_median=round(b0_med, 4),
        median_spread=round(ch_med - b0_med, 4),
        challenger_cvar10=round(ch_cvar, 4),
        b0_cvar10=round(b0_cvar, 4),
        cvar_delta=round(ch_cvar - b0_cvar, 4),
        challenger_p10=round(ch_p10, 4),
        b0_p10=round(b0_p10, 4),
        challenger_stop8_pct=round(float(np.mean(ch_stops)), 2),
        b0_stop8_pct=round(float(np.mean(b0_stops)), 2),
        stop_delta_pct=round(float(np.mean(ch_stops) - np.mean(b0_stops)), 2),
        challenger_one_pick_ruins_pct=round(float(np.mean(ch_ruins) * 100.0), 2),
        b0_one_pick_ruins_pct=round(float(np.mean(b0_ruins) * 100.0), 2),
        one_pick_ruins_delta_pct=round(float((np.mean(ch_ruins) - np.mean(b0_ruins)) * 100.0), 2),
        slot_coverage_pct=round(cov_pct, 2),
        full_top3_rate_pct=round(full_top3_pct, 2),
        top3_membership_jaccard_vs_b0=round(avg_jaccard, 4),
        lowo=lowo_res,
        bootstrap=boot_res,
        pareto_score=round(pareto, 4),
    )


def classify_champion_track_c(
    train_summary: PairedEvaluationSummary,
    val_summary: PairedEvaluationSummary,
) -> str:
    """Classify champion into 3 pre-registered exit states."""
    if val_summary.support_weeks == 0:
        return "INSUFFICIENT VALIDATION DATA"

    # Immateriality threshold
    is_immaterial = (
        abs(val_summary.median_spread) <= 0.05 and
        abs(val_summary.mean_spread) <= 0.05 and
        abs(val_summary.cvar_delta) <= 0.5 and
        abs(val_summary.stop_delta_pct) <= 0.5
    )
    if is_immaterial:
        return "EQUIVALENT TO B0"

    # State A: Robust Candidate (Promoted to FORWARD SHADOW CANDIDATE)
    has_material_win = (
        val_summary.median_spread > 0.05 or
        val_summary.mean_spread > 0.05 or
        val_summary.cvar_delta > 0.5 or
        val_summary.stop_delta_pct < -0.5
    )
    has_no_degradation = (
        val_summary.median_spread >= -0.05 and
        val_summary.mean_spread >= -0.05 and
        val_summary.cvar_delta >= -0.5 and
        val_summary.stop_delta_pct <= 0.5
    )

    if has_material_win and has_no_degradation:
        # Check Train fragility and block bootstrap CI low
        if train_summary.lowo and train_summary.lowo.is_fragile_overfit:
            return "UNSTABLE (FRAGILE ON TRAIN)"
        if val_summary.bootstrap and val_summary.bootstrap.mean_spread_ci_low < -2.0:
            return "INSUFFICIENT EVIDENCE (CI LOW < -2.0)"
        return "FORWARD SHADOW CANDIDATE"

    # State B: Trade-offs or Unstable
    if val_summary.mean_spread < -1.0 or val_summary.median_spread < -1.0:
        return "UNSTABLE (NOT ROBUST ON OBSERVED VALIDATION)"

    return "NO ROBUST REPLACEMENT FOR B0 FOUND (RETAIN B0 OPERATIONALLY)"
