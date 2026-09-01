from __future__ import annotations
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol
import numpy as np
import pandas as pd
from .config import TOP_N


@dataclass
class PolicyProposal:
    """Pre-registered Blind Policy Proposal specification."""
    policy_id: str
    family: str
    description: str
    ranker_type: str  # 'rule_lexicographic', 'continuous_linear', 'tree_model', 'pairwise_ltr', 'custom_heuristic'
    allocator_type: str  # 'greedy_distinct', 'pure_score', 'max_k_per_ind', 'industry_breadth_first', 'portfolio_utility'
    picker_type: str  # 'highest_rank_in_quota', 'max_score_in_quota', 'utility_optimizer'
    spec_params: dict[str, Any] = field(default_factory=dict)
    custom_code: str | None = None
    spec_hash: str = ""
    fitted_state_hash: str = "none"


class ChallengerProtocol(Protocol):
    """Protocol defining the 3 decoupled modular layers of Track C."""
    policy_id: str
    family: str
    spec_hash: str
    fitted_state_hash: str

    def score_candidates(self, snapshot_df: pd.DataFrame) -> pd.DataFrame:
        """Layer 1: Individual candidate ranking & scoring."""
        ...

    def allocate_industries(self, scored_df: pd.DataFrame) -> dict[str, int]:
        """Layer 2: Industry allocation quotas (returns mapping of industry -> integer quota, sum <= TOP_N)."""
        ...

    def pick_stocks(self, scored_df: pd.DataFrame, industry_quotas: dict[str, int]) -> list[str]:
        """Layer 3: Within-industry selection (returns list of 0..3 selected ticker codes)."""
        ...


@dataclass
class WeeklyPortfolioOutcome:
    """Standardized weekly portfolio result with 3-slot capital accounting."""
    snapshot_date: str
    selector_id: str
    horizon: str
    pick_count: int
    slot_coverage: float  # pick_count / 3.0
    active_week: bool  # pick_count > 0
    full_top3: bool  # pick_count == 3
    selected_codes: list[str]
    selection_quality_return: float  # Mean of actual selected returns (NaN if pick_count == 0)
    capital_adjusted_return: float  # (sum of selected returns + (3 - pick_count) * 0.0) / 3.0
    selection_quality_stop8: float  # Stop rate of active picks
    capital_adjusted_stop8: float  # sum(stop8) / 3.0
    one_pick_ruined: bool  # True if any active pick suffered >= 8% loss
    is_mature: bool  # True if all selected picks have non-null return outcome (or pick_count == 0)


def compute_3slot_portfolio_weekly(
    picks_df: pd.DataFrame,
    all_snapshots: list[str],
    selector_id: str,
    horizon: str = "W4",
) -> list[WeeklyPortfolioOutcome]:
    """Compute exact 3-slot capital accounting and selection quality metrics across all snapshots."""
    ret_col = f"{horizon.lower()}_return_pct"
    stop_col = f"{horizon.lower()}_stop8"

    outcomes = []
    # Group picks by snapshot
    picks_by_snap = {}
    if not picks_df.empty:
        for s_date, g in picks_df.groupby("snapshot_date"):
            picks_by_snap[str(s_date)] = g

    for s_date in sorted(all_snapshots):
        g = picks_by_snap.get(str(s_date))
        if g is None or g.empty:
            # 0 picks: Capital return = 0.0, Quality return = NaN
            outcomes.append(
                WeeklyPortfolioOutcome(
                    snapshot_date=s_date,
                    selector_id=selector_id,
                    horizon=horizon,
                    pick_count=0,
                    slot_coverage=0.0,
                    active_week=False,
                    full_top3=False,
                    selected_codes=[],
                    selection_quality_return=np.nan,
                    capital_adjusted_return=0.0,
                    selection_quality_stop8=np.nan,
                    capital_adjusted_stop8=0.0,
                    one_pick_ruined=False,
                    is_mature=True,
                )
            )
            continue

        # Active picks (1 <= k <= 3)
        k = len(g)
        codes = g["code"].astype(str).tolist()
        rets = pd.to_numeric(g[ret_col], errors="coerce") if ret_col in g.columns else pd.Series([np.nan] * k)
        stops = g[stop_col].astype(bool) if stop_col in g.columns else pd.Series([False] * k)

        is_mature = bool(rets.notna().all())
        if not is_mature:
            # Not fully mature yet
            outcomes.append(
                WeeklyPortfolioOutcome(
                    snapshot_date=s_date,
                    selector_id=selector_id,
                    horizon=horizon,
                    pick_count=k,
                    slot_coverage=round(k / float(TOP_N), 4),
                    active_week=True,
                    full_top3=(k == TOP_N),
                    selected_codes=codes,
                    selection_quality_return=np.nan,
                    capital_adjusted_return=np.nan,
                    selection_quality_stop8=np.nan,
                    capital_adjusted_stop8=np.nan,
                    one_pick_ruined=False,
                    is_mature=False,
                )
            )
            continue

        valid_rets = rets.values
        valid_stops = stops.values

        sel_qual_ret = float(np.mean(valid_rets))
        cap_adj_ret = float(np.sum(valid_rets) / float(TOP_N))  # 3-slot capital allocation

        sel_qual_stop = float(np.mean(valid_stops) * 100.0)
        cap_adj_stop = float((np.sum(valid_stops) / float(TOP_N)) * 100.0)

        any_ruined = bool(np.any(valid_rets <= -8.0) or np.any(valid_stops))

        outcomes.append(
            WeeklyPortfolioOutcome(
                snapshot_date=s_date,
                selector_id=selector_id,
                horizon=horizon,
                pick_count=k,
                slot_coverage=round(k / float(TOP_N), 4),
                active_week=True,
                full_top3=(k == TOP_N),
                selected_codes=codes,
                selection_quality_return=round(sel_qual_ret, 4),
                capital_adjusted_return=round(cap_adj_ret, 4),
                selection_quality_stop8=round(sel_qual_stop, 2),
                capital_adjusted_stop8=round(cap_adj_stop, 2),
                one_pick_ruined=any_ruined,
                is_mature=True,
            )
        )

    return outcomes
