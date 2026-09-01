from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from ..config import (
    DISCOVERY_BUDGET,
    FAMILY_BUDGETS,
    FEATURE_MANIFEST_PATH,
    PROPOSALS_DIR,
    RANDOM_SEED,
    TOP_N,
    TRAIN_END,
)
from ..protocol import ChallengerProtocol, PolicyProposal
from .anonymizer import create_anonymized_discovery_dataset
from .behavioral_dedup import deduplicate_proposals_behaviorally


# ---------------------------------------------------------
# Family 1: Industry Breadth First Challengers
# ---------------------------------------------------------
class IndustryBreadthChallenger:
    """Industry-First Allocator ranking industries by breadth and assigning dynamic quotas (e.g. 2+1 vs 1+1+1)."""

    def __init__(
        self,
        policy_id: str,
        breadth_metric: str = "actionable_count",
        allow_dynamic_2_plus_1: bool = True,
        min_breadth_for_2: int = 2,
        within_ind_sort: str = "eps_and_volume",
    ):
        self.policy_id = f"IND_BREADTH__{policy_id}"
        self.family = "industry_breadth"
        self.breadth_metric = breadth_metric
        self.allow_dynamic_2_plus_1 = allow_dynamic_2_plus_1
        self.min_breadth_for_2 = min_breadth_for_2
        self.within_ind_sort = within_ind_sort
        self.spec_hash = hashlib.sha256(f"{self.policy_id}_{breadth_metric}_{allow_dynamic_2_plus_1}_{min_breadth_for_2}_{within_ind_sort}".encode()).hexdigest()
        self.fitted_state_hash = "none"

    def score_candidates(self, snapshot_df: pd.DataFrame) -> pd.DataFrame:
        if snapshot_df.empty:
            return pd.DataFrame()

        df = snapshot_df.copy()
        # Compute candidate level within-industry score
        scores = np.zeros(len(df), dtype=float)
        if "is_actionable" in df.columns:
            scores += df["is_actionable"].fillna(0).astype(float) * 10.0
        if "dist_to_52w_high_pct" in df.columns:
            scores += np.clip(df["dist_to_52w_high_pct"].fillna(-50).astype(float) / 10.0, -5.0, 5.0)
        if "eps_yoy_growth" in df.columns:
            scores += np.clip(df["eps_yoy_growth"].fillna(0).astype(float) / 25.0, -2.0, 5.0)
        if "ibd_entry_volume_ratio" in df.columns:
            scores += np.clip(df["ibd_entry_volume_ratio"].fillna(1.0).astype(float), 0.0, 5.0)

        df["candidate_score"] = scores
        df["raw_rank"] = df["candidate_score"].rank(ascending=False, method="min")
        df["has_geom_failure"] = (df["clear_geometry_failure"] == 1).astype(int) if "clear_geometry_failure" in df.columns else np.zeros(len(df), dtype=int)
        df["below_buy_point"] = (df["current_vs_ibd_candidate_pct"] < 0).astype(int) if "current_vs_ibd_candidate_pct" in df.columns else np.zeros(len(df), dtype=int)
        return df

    def allocate_industries(self, scored_df: pd.DataFrame) -> dict[str, int]:
        if scored_df.empty:
            return {}

        eligible = scored_df[
            (scored_df.get("is_actionable", 1) == 1) &
            (scored_df.get("has_geom_failure", 0) == 0) &
            (scored_df.get("below_buy_point", 0) == 0)
        ].copy()

        if eligible.empty:
            return {}

        # Compute Industry Breadth Metrics
        ind_stats = {}
        for ind, g in eligible.groupby("industry"):
            act_cnt = len(g)
            avg_score = float(g["candidate_score"].mean()) if "candidate_score" in g.columns else 0.0
            vol_breadth = float(g["ibd_entry_volume_ratio"].mean()) if "ibd_entry_volume_ratio" in g.columns else 1.0

            if self.breadth_metric == "actionable_count":
                ind_rank_score = act_cnt * 10.0 + avg_score
            elif self.breadth_metric == "volume_breadth":
                ind_rank_score = vol_breadth * 10.0 + act_cnt
            elif self.breadth_metric == "quality_and_count":
                ind_rank_score = avg_score * 2.0 + act_cnt * 5.0
            else:
                ind_rank_score = act_cnt

            ind_stats[ind] = {
                "breadth_score": ind_rank_score,
                "candidate_count": act_cnt,
            }

        sorted_inds = sorted(ind_stats.keys(), key=lambda x: ind_stats[x]["breadth_score"], reverse=True)
        quotas: dict[str, int] = {}

        if self.allow_dynamic_2_plus_1 and len(sorted_inds) >= 1 and ind_stats[sorted_inds[0]]["candidate_count"] >= self.min_breadth_for_2:
            # Top industry gets 2, next gets 1
            quotas[sorted_inds[0]] = 2
            if len(sorted_inds) >= 2:
                quotas[sorted_inds[1]] = 1
        else:
            # 1 per industry
            for ind in sorted_inds[:TOP_N]:
                quotas[ind] = 1

        return quotas

    def pick_stocks(self, scored_df: pd.DataFrame, industry_quotas: dict[str, int]) -> list[str]:
        if scored_df.empty or not industry_quotas:
            return []

        eligible = scored_df[
            (scored_df.get("is_actionable", 1) == 1) &
            (scored_df.get("has_geom_failure", 0) == 0) &
            (scored_df.get("below_buy_point", 0) == 0)
        ].sort_values("candidate_score", ascending=False)

        selected = []
        ind_counts: dict[str, int] = {}

        for _, r in eligible.iterrows():
            if len(selected) >= TOP_N:
                break
            code = str(r["code"])
            ind = str(r["industry"])
            quota = industry_quotas.get(ind, 0)
            cur = ind_counts.get(ind, 0)

            if cur < quota:
                selected.append(code)
                ind_counts[ind] = cur + 1

        return selected


# ---------------------------------------------------------
# Family 2: Continuous Multi-Factor Scoring Challengers
# ---------------------------------------------------------
class ContinuousScoreChallenger:
    """Multi-factor continuous scoring ranker with distinct or max_2 industry allocator."""

    def __init__(
        self,
        policy_id: str,
        weights: dict[str, float],
        selector_mode: str = "distinct_1",
    ):
        self.policy_id = f"CONT_SCORE__{policy_id}"
        self.family = "continuous"
        self.weights = weights
        self.selector_mode = selector_mode
        self.spec_hash = hashlib.sha256(f"{self.policy_id}_{json.dumps(weights, sort_keys=True)}_{selector_mode}".encode()).hexdigest()
        self.fitted_state_hash = "none"

    def score_candidates(self, snapshot_df: pd.DataFrame) -> pd.DataFrame:
        if snapshot_df.empty:
            return pd.DataFrame()

        df = snapshot_df.copy()
        score = np.zeros(len(df), dtype=float)

        for col, w in self.weights.items():
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0).values
                # Standardize with clip
                std = np.std(vals)
                norm_vals = (vals - np.mean(vals)) / (std if std > 1e-6 else 1.0)
                score += w * np.clip(norm_vals, -3.0, 3.0)

        df["candidate_score"] = score
        df["raw_rank"] = df["candidate_score"].rank(ascending=False, method="min")
        df["has_geom_failure"] = (df["clear_geometry_failure"] == 1).astype(int) if "clear_geometry_failure" in df.columns else np.zeros(len(df), dtype=int)
        df["below_buy_point"] = (df["current_vs_ibd_candidate_pct"] < 0).astype(int) if "current_vs_ibd_candidate_pct" in df.columns else np.zeros(len(df), dtype=int)
        return df

    def allocate_industries(self, scored_df: pd.DataFrame) -> dict[str, int]:
        if scored_df.empty:
            return {}
        eligible = scored_df[
            (scored_df.get("is_actionable", 1) == 1) &
            (scored_df.get("has_geom_failure", 0) == 0) &
            (scored_df.get("below_buy_point", 0) == 0)
        ]
        inds = eligible["industry"].dropna().unique()
        quota = 1 if self.selector_mode == "distinct_1" else (2 if self.selector_mode == "max_2_per_ind" else TOP_N)
        return {ind: quota for ind in inds}

    def pick_stocks(self, scored_df: pd.DataFrame, industry_quotas: dict[str, int]) -> list[str]:
        if scored_df.empty or not industry_quotas:
            return []
        eligible = scored_df[
            (scored_df.get("is_actionable", 1) == 1) &
            (scored_df.get("has_geom_failure", 0) == 0) &
            (scored_df.get("below_buy_point", 0) == 0)
        ].sort_values("candidate_score", ascending=False)

        selected = []
        ind_counts: dict[str, int] = {}
        for _, r in eligible.iterrows():
            if len(selected) >= TOP_N:
                break
            code = str(r["code"])
            ind = str(r["industry"])
            quota = industry_quotas.get(ind, 0)
            cur = ind_counts.get(ind, 0)
            if cur < quota:
                selected.append(code)
                ind_counts[ind] = cur + 1
        return selected


# ---------------------------------------------------------
# Family 3: Multi-Feature Linear Ranking (Deterministic)
# ---------------------------------------------------------
class MultiFeatureLinearChallenger:
    """Deterministic multi-feature linear ranking challenger with standardized feature weights."""

    def __init__(
        self,
        policy_id: str,
        feature_subset: list[str],
        regularization: float = 1.0,
        selector_mode: str = "distinct_1",
    ):
        self.policy_id = f"LINEAR_RANK__{policy_id}"
        self.family = "linear_ranking"
        self.feature_subset = feature_subset
        self.regularization = regularization
        self.selector_mode = selector_mode
        self.spec_hash = hashlib.sha256(f"{self.policy_id}_{json.dumps(feature_subset)}_{regularization}_{selector_mode}".encode()).hexdigest()
        self.fitted_state_hash = "none"

    def score_candidates(self, snapshot_df: pd.DataFrame) -> pd.DataFrame:
        if snapshot_df.empty:
            return pd.DataFrame()
        df = snapshot_df.copy()

        score = np.zeros(len(df), dtype=float)
        for col in self.feature_subset:
            if col in df.columns:
                v = pd.to_numeric(df[col], errors="coerce").fillna(0.0).values
                score += (v / (np.std(v) + 1e-6)) / float(self.regularization)

        df["candidate_score"] = score
        df["raw_rank"] = df["candidate_score"].rank(ascending=False, method="min")
        df["has_geom_failure"] = (df["clear_geometry_failure"] == 1).astype(int) if "clear_geometry_failure" in df.columns else np.zeros(len(df), dtype=int)
        df["below_buy_point"] = (df["current_vs_ibd_candidate_pct"] < 0).astype(int) if "current_vs_ibd_candidate_pct" in df.columns else np.zeros(len(df), dtype=int)
        return df

    def allocate_industries(self, scored_df: pd.DataFrame) -> dict[str, int]:
        if scored_df.empty:
            return {}
        eligible = scored_df[
            (scored_df.get("is_actionable", 1) == 1) &
            (scored_df.get("has_geom_failure", 0) == 0) &
            (scored_df.get("below_buy_point", 0) == 0)
        ]
        inds = eligible["industry"].dropna().unique()
        quota = 1 if self.selector_mode == "distinct_1" else (2 if self.selector_mode == "max_2_per_ind" else TOP_N)
        return {ind: quota for ind in inds}

    def pick_stocks(self, scored_df: pd.DataFrame, industry_quotas: dict[str, int]) -> list[str]:
        if scored_df.empty or not industry_quotas:
            return []
        eligible = scored_df[
            (scored_df.get("is_actionable", 1) == 1) &
            (scored_df.get("has_geom_failure", 0) == 0) &
            (scored_df.get("below_buy_point", 0) == 0)
        ].sort_values("candidate_score", ascending=False)

        selected = []
        ind_counts: dict[str, int] = {}
        for _, r in eligible.iterrows():
            if len(selected) >= TOP_N:
                break
            code = str(r["code"])
            ind = str(r["industry"])
            quota = industry_quotas.get(ind, 0)
            cur = ind_counts.get(ind, 0)
            if cur < quota:
                selected.append(code)
                ind_counts[ind] = cur + 1
        return selected


# ---------------------------------------------------------
# Family 4: Portfolio Utility Optimizer Challengers
# ---------------------------------------------------------
class PortfolioUtilityChallenger:
    """Portfolio Utility Challenger maximizing stock score minus dynamic industry concentration penalty."""

    def __init__(
        self,
        policy_id: str,
        concentration_lambda: float = 1.0,
        stock_quality_metric: str = "balanced",
    ):
        self.policy_id = f"PORT_UTIL__{policy_id}"
        self.family = "portfolio"
        self.concentration_lambda = concentration_lambda
        self.stock_quality_metric = stock_quality_metric
        self.spec_hash = hashlib.sha256(f"{self.policy_id}_{concentration_lambda}_{stock_quality_metric}".encode()).hexdigest()
        self.fitted_state_hash = "none"

    def score_candidates(self, snapshot_df: pd.DataFrame) -> pd.DataFrame:
        if snapshot_df.empty:
            return pd.DataFrame()
        df = snapshot_df.copy()
        score = np.zeros(len(df), dtype=float)

        if self.stock_quality_metric == "balanced":
            if "eps_yoy_growth" in df.columns:
                score += np.clip(pd.to_numeric(df["eps_yoy_growth"], errors="coerce").fillna(0.0) / 20.0, -2, 5)
            if "mom_20" in df.columns:
                score += np.clip(pd.to_numeric(df["mom_20"], errors="coerce").fillna(0.0) / 10.0, -2, 5)
            if "ibd_entry_volume_ratio" in df.columns:
                score += np.clip(pd.to_numeric(df["ibd_entry_volume_ratio"], errors="coerce").fillna(1.0), 0, 5)
        elif self.stock_quality_metric == "momentum_first":
            if "mom_60" in df.columns:
                score += np.clip(pd.to_numeric(df["mom_60"], errors="coerce").fillna(0.0) / 15.0, -2, 5)
            if "dist_to_52w_high_pct" in df.columns:
                score += np.clip(pd.to_numeric(df["dist_to_52w_high_pct"], errors="coerce").fillna(-50) / 10.0, -5, 5)

        df["candidate_score"] = score
        df["raw_rank"] = df["candidate_score"].rank(ascending=False, method="min")
        df["has_geom_failure"] = (df["clear_geometry_failure"] == 1).astype(int) if "clear_geometry_failure" in df.columns else np.zeros(len(df), dtype=int)
        df["below_buy_point"] = (df["current_vs_ibd_candidate_pct"] < 0).astype(int) if "current_vs_ibd_candidate_pct" in df.columns else np.zeros(len(df), dtype=int)
        return df

    def allocate_industries(self, scored_df: pd.DataFrame) -> dict[str, int]:
        if scored_df.empty:
            return {}
        eligible = scored_df[
            (scored_df.get("is_actionable", 1) == 1) &
            (scored_df.get("has_geom_failure", 0) == 0) &
            (scored_df.get("below_buy_point", 0) == 0)
        ]
        inds = eligible["industry"].dropna().unique()
        return {ind: TOP_N for ind in inds}

    def pick_stocks(self, scored_df: pd.DataFrame, industry_quotas: dict[str, int]) -> list[str]:
        if scored_df.empty:
            return []
        eligible = scored_df[
            (scored_df.get("is_actionable", 1) == 1) &
            (scored_df.get("has_geom_failure", 0) == 0) &
            (scored_df.get("below_buy_point", 0) == 0)
        ].copy()

        if eligible.empty:
            return []

        candidates = eligible.to_dict(orient="records")
        selected_records: list[dict] = []
        ind_counts: dict[str, int] = {}

        for _ in range(TOP_N):
            best_cand = None
            best_gain = -999999.0

            for c in candidates:
                if c["code"] in [s["code"] for s in selected_records]:
                    continue
                ind = str(c["industry"])
                cur_ind_cnt = ind_counts.get(ind, 0)
                penalty_delta = self.concentration_lambda * ((cur_ind_cnt + 1) ** 2 - cur_ind_cnt ** 2)
                gain = c["candidate_score"] - penalty_delta

                if gain > best_gain:
                    best_gain = gain
                    best_cand = c

            if best_cand is not None:
                selected_records.append(best_cand)
                ind = str(best_cand["industry"])
                ind_counts[ind] = ind_counts.get(ind, 0) + 1
            else:
                break

        return [s["code"] for s in selected_records]


# ---------------------------------------------------------
# Family 5: Novel Heuristic Policy Challengers
# ---------------------------------------------------------
class NovelHeuristicChallenger:
    """Domain-grounded heuristic combining volume dry-up, base structure, and actionable status in non-linear rules."""

    def __init__(
        self,
        policy_id: str,
        dry_weight: float = 2.0,
        base_depth_penalty: float = 1.0,
        volume_spike_bonus: float = 2.0,
        selector_mode: str = "distinct_1",
    ):
        self.policy_id = f"NOVEL_HEURISTIC__{policy_id}"
        self.family = "novel_heuristic"
        self.dry_weight = dry_weight
        self.base_depth_penalty = base_depth_penalty
        self.volume_spike_bonus = volume_spike_bonus
        self.selector_mode = selector_mode
        self.spec_hash = hashlib.sha256(f"{self.policy_id}_{dry_weight}_{base_depth_penalty}_{volume_spike_bonus}_{selector_mode}".encode()).hexdigest()
        self.fitted_state_hash = "none"

    def score_candidates(self, snapshot_df: pd.DataFrame) -> pd.DataFrame:
        if snapshot_df.empty:
            return pd.DataFrame()
        df = snapshot_df.copy()
        score = np.zeros(len(df), dtype=float)

        if "is_actionable" in df.columns:
            score += df["is_actionable"].fillna(0).astype(float) * 5.0
        if "pullback_v_is_dry" in df.columns:
            is_dry = df["pullback_v_is_dry"].fillna(False).astype(bool)
            score += np.where(is_dry, self.dry_weight, 0.0)
        if "base_depth_pct" in df.columns:
            depth = pd.to_numeric(df["base_depth_pct"], errors="coerce").fillna(20.0).values
            score -= np.where(depth > 35.0, self.base_depth_penalty * (depth - 35.0) / 10.0, 0.0)
        if "ibd_entry_volume_ratio" in df.columns:
            vol = pd.to_numeric(df["ibd_entry_volume_ratio"], errors="coerce").fillna(1.0).values
            score += np.where(vol >= 1.5, self.volume_spike_bonus, 0.0)

        df["candidate_score"] = score
        df["raw_rank"] = df["candidate_score"].rank(ascending=False, method="min")
        df["has_geom_failure"] = (df["clear_geometry_failure"] == 1).astype(int) if "clear_geometry_failure" in df.columns else np.zeros(len(df), dtype=int)
        df["below_buy_point"] = (df["current_vs_ibd_candidate_pct"] < 0).astype(int) if "current_vs_ibd_candidate_pct" in df.columns else np.zeros(len(df), dtype=int)
        return df

    def allocate_industries(self, scored_df: pd.DataFrame) -> dict[str, int]:
        if scored_df.empty:
            return {}
        eligible = scored_df[
            (scored_df.get("is_actionable", 1) == 1) &
            (scored_df.get("has_geom_failure", 0) == 0) &
            (scored_df.get("below_buy_point", 0) == 0)
        ]
        inds = eligible["industry"].dropna().unique()
        quota = 1 if self.selector_mode == "distinct_1" else (2 if self.selector_mode == "max_2_per_ind" else TOP_N)
        return {ind: quota for ind in inds}

    def pick_stocks(self, scored_df: pd.DataFrame, industry_quotas: dict[str, int]) -> list[str]:
        if scored_df.empty or not industry_quotas:
            return []
        eligible = scored_df[
            (scored_df.get("is_actionable", 1) == 1) &
            (scored_df.get("has_geom_failure", 0) == 0) &
            (scored_df.get("below_buy_point", 0) == 0)
        ].sort_values("candidate_score", ascending=False)

        selected = []
        ind_counts: dict[str, int] = {}
        for _, r in eligible.iterrows():
            if len(selected) >= TOP_N:
                break
            code = str(r["code"])
            ind = str(r["industry"])
            quota = industry_quotas.get(ind, 0)
            cur = ind_counts.get(ind, 0)
            if cur < quota:
                selected.append(code)
                ind_counts[ind] = cur + 1
        return selected


# ---------------------------------------------------------
# Proposal Generation Factory within Search Budgets
# ---------------------------------------------------------
def generate_all_discovery_proposals() -> list[ChallengerProtocol]:
    """Generate pre-registered candidate discovery proposals adhering strictly to pre-registered budgets."""
    proposals: list[ChallengerProtocol] = []

    # 1. Family: Industry Breadth (Quota: 11)
    b_metrics = ["actionable_count", "volume_breadth", "quality_and_count"]
    dyn_configs = [(True, 2), (True, 3), (False, 2)]
    ind_count = 0
    for bm in b_metrics:
        for dyn, min_b in dyn_configs:
            if ind_count >= FAMILY_BUDGETS["industry_breadth"]:
                break
            p_id = f"breadth_{bm}_dyn{dyn}_min{min_b}"
            proposals.append(IndustryBreadthChallenger(p_id, breadth_metric=bm, allow_dynamic_2_plus_1=dyn, min_breadth_for_2=min_b))
            ind_count += 1

    # 2. Family: Continuous Scoring (Quota: 11)
    cont_configs = [
        ("eps_mom_vol", {"eps_yoy_growth": 2.0, "mom_20": 1.5, "ibd_entry_volume_ratio": 1.5, "dist_to_52w_high_pct": 1.0}, "distinct_1"),
        ("mom_heavy", {"mom_60": 3.0, "mom_20": 2.0, "rel_spy_60": 2.0, "dist_to_52w_high_pct": 1.5}, "distinct_1"),
        ("eps_heavy", {"eps_yoy_growth": 4.0, "ibd_entry_volume_ratio": 2.0, "dist_to_52w_high_pct": 1.0}, "distinct_1"),
        ("vol_confirmed", {"ibd_entry_volume_ratio": 3.0, "volume_ratio": 2.0, "mom_20": 1.5, "dist_to_52w_high_pct": 1.0}, "distinct_1"),
        ("quality_52w", {"dist_to_52w_high_pct": 3.0, "eps_yoy_growth": 2.0, "mom_20": 1.0}, "distinct_1"),
        ("eps_mom_max2", {"eps_yoy_growth": 2.0, "mom_20": 1.5, "ibd_entry_volume_ratio": 1.5}, "max_2_per_ind"),
        ("mom_heavy_max2", {"mom_60": 3.0, "mom_20": 2.0, "rel_spy_60": 2.0}, "max_2_per_ind"),
        ("vol_confirmed_max2", {"ibd_entry_volume_ratio": 3.0, "volume_ratio": 2.0}, "max_2_per_ind"),
        ("tight_structure", {"dist_to_52w_high_pct": 2.0, "rv_20": -1.5, "atr_14_pct": -1.0, "eps_yoy_growth": 2.0}, "distinct_1"),
        ("balanced_all", {"eps_yoy_growth": 1.0, "mom_20": 1.0, "volume_ratio": 1.0, "dist_to_52w_high_pct": 1.0}, "distinct_1"),
        ("pure_breakout", {"ibd_entry_volume_ratio": 3.0, "dist_to_52w_high_pct": 2.0}, "distinct_1"),
    ]
    for p_id, w_dict, sm in cont_configs[:FAMILY_BUDGETS["continuous"]]:
        proposals.append(ContinuousScoreChallenger(p_id, w_dict, selector_mode=sm))

    # 3. Family: Linear Multi-Feature Ranking (Quota: 11)
    lin_configs = [
        ("linear_core", ["eps_yoy_growth", "mom_20", "ibd_entry_volume_ratio", "dist_to_52w_high_pct"], 1.0, "distinct_1"),
        ("linear_reg_high", ["eps_yoy_growth", "mom_20", "ibd_entry_volume_ratio", "dist_to_52w_high_pct"], 2.5, "distinct_1"),
        ("linear_momentum", ["mom_5", "mom_10", "mom_20", "mom_60", "rel_spy_20", "rel_spy_60"], 1.0, "distinct_1"),
        ("linear_volume", ["ibd_entry_volume_ratio", "volume_ratio", "vol_ratio_5_20"], 1.0, "distinct_1"),
        ("linear_geom", ["dist_to_52w_high_pct", "base_depth_pct", "base_duration_weeks"], 1.0, "distinct_1"),
        ("linear_core_max2", ["eps_yoy_growth", "mom_20", "ibd_entry_volume_ratio", "dist_to_52w_high_pct"], 1.0, "max_2_per_ind"),
        ("linear_mom_max2", ["mom_20", "mom_60", "rel_spy_60"], 1.0, "max_2_per_ind"),
        ("linear_slope", ["ma10_slope_5", "ma20_slope_5", "ma50_slope_10", "dist_to_52w_high_pct"], 1.0, "distinct_1"),
        ("linear_full", ["eps_yoy_growth", "mom_20", "ibd_entry_volume_ratio", "vol_ratio_5_20", "dist_to_52w_high_pct", "rv_20"], 1.5, "distinct_1"),
        ("linear_tight", ["rv_20", "atr_14_pct", "base_depth_pct", "eps_yoy_growth"], 1.0, "distinct_1"),
        ("linear_fast_mom", ["mom_5", "mom_10", "ibd_entry_volume_ratio"], 1.0, "distinct_1"),
    ]
    for p_id, feats, reg, sm in lin_configs[:FAMILY_BUDGETS["linear_ranking"]]:
        proposals.append(MultiFeatureLinearChallenger(p_id, feats, regularization=reg, selector_mode=sm))

    # 4. Family: Portfolio Utility (Quota: 10)
    port_configs = [
        ("lambda_0_5_bal", 0.5, "balanced"),
        ("lambda_1_0_bal", 1.0, "balanced"),
        ("lambda_2_0_bal", 2.0, "balanced"),
        ("lambda_5_0_bal", 5.0, "balanced"),
        ("lambda_0_5_mom", 0.5, "momentum_first"),
        ("lambda_1_0_mom", 1.0, "momentum_first"),
        ("lambda_2_0_mom", 2.0, "momentum_first"),
        ("lambda_5_0_mom", 5.0, "momentum_first"),
        ("lambda_0_2_bal", 0.2, "balanced"),
        ("lambda_3_0_bal", 3.0, "balanced"),
    ]
    for p_id, lam, qm in port_configs[:FAMILY_BUDGETS["portfolio"]]:
        proposals.append(PortfolioUtilityChallenger(p_id, concentration_lambda=lam, stock_quality_metric=qm))

    # 5. Family: Novel Heuristics (Quota: 11)
    novel_configs = [
        ("dry_heavy", 3.0, 1.0, 2.0, "distinct_1"),
        ("dry_moderate", 1.5, 1.0, 1.5, "distinct_1"),
        ("shallow_base_focus", 2.0, 3.0, 2.0, "distinct_1"),
        ("vol_spike_focus", 2.0, 1.0, 4.0, "distinct_1"),
        ("dry_heavy_max2", 3.0, 1.0, 2.0, "max_2_per_ind"),
        ("vol_spike_max2", 2.0, 1.0, 4.0, "max_2_per_ind"),
        ("conservative_geom", 1.0, 2.5, 1.0, "distinct_1"),
        ("aggressive_breakout", 2.5, 0.5, 3.0, "distinct_1"),
        ("tight_dry_pullback", 4.0, 2.0, 1.0, "distinct_1"),
        ("balanced_novel", 2.0, 2.0, 2.0, "distinct_1"),
        ("pure_volume_dry", 3.5, 0.0, 3.5, "distinct_1"),
    ]
    for p_id, dw, bdp, vsb, sm in novel_configs[:FAMILY_BUDGETS["novel_heuristic"]]:
        proposals.append(NovelHeuristicChallenger(p_id, dry_weight=dw, base_depth_penalty=bdp, volume_spike_bonus=vsb, selector_mode=sm))

    return proposals



# ---------------------------------------------------------
# Frozen RD-Agent proposal spec validation / instantiation
# ---------------------------------------------------------
def _allowed_discovery_feature_types() -> dict[str, str]:
    with open(FEATURE_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return {
        k: str(v.get("data_type", ""))
        for k, v in manifest["features"].items()
        if v.get("allowed_for_discovery") is True
    }


def _require_selector(value: object) -> str:
    selector = str(value or "distinct_1")
    if selector not in {"distinct_1", "max_2_per_ind", "pure_top3"}:
        raise ValueError(f"Unsupported selector_mode: {selector}")
    return selector


def _bounded_float(value: object, lo: float, hi: float, name: str) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(x) or x < lo or x > hi:
        raise ValueError(f"{name} must be within [{lo}, {hi}], got {x}")
    return x


def instantiate_discovery_proposal(
    record: dict[str, Any],
) -> tuple[ChallengerProtocol, dict[str, Any]]:
    """Validate one frozen RD-Agent spec and create its executable policy object."""
    family = str(record.get("family") or "").strip()
    name = str(record.get("name") or "").strip()
    params = record.get("params")
    if not name or not isinstance(params, dict):
        raise ValueError("RD-Agent proposal requires non-empty name and params object")

    feature_types = _allowed_discovery_feature_types()
    numeric_allowed = {
        k for k, typ in feature_types.items()
        if typ in {"float", "int", "bool"}
    }

    if family == "industry_breadth":
        metric = str(params.get("breadth_metric") or "")
        if metric not in {"actionable_count", "volume_breadth", "quality_and_count"}:
            raise ValueError(f"Unsupported breadth_metric: {metric}")
        dynamic = bool(params.get("allow_dynamic_2_plus_1", True))
        min_b = int(params.get("min_breadth_for_2", 2))
        if min_b not in {2, 3}:
            raise ValueError("min_breadth_for_2 must be 2 or 3")
        normalized_params = {
            "breadth_metric": metric,
            "allow_dynamic_2_plus_1": dynamic,
            "min_breadth_for_2": min_b,
        }
        policy = IndustryBreadthChallenger(
            name,
            breadth_metric=metric,
            allow_dynamic_2_plus_1=dynamic,
            min_breadth_for_2=min_b,
        )

    elif family == "continuous":
        weights_raw = params.get("weights")
        if not isinstance(weights_raw, dict) or not (2 <= len(weights_raw) <= 8):
            raise ValueError("continuous.weights must contain 2..8 features")
        weights: dict[str, float] = {}
        for feature, weight in weights_raw.items():
            if feature not in numeric_allowed:
                raise ValueError(f"Feature {feature!r} is not an allowed numeric PIT feature")
            w = _bounded_float(weight, -8.0, 8.0, f"weight[{feature}]")
            if abs(w) < 1e-12:
                raise ValueError(f"weight[{feature}] must be non-zero")
            weights[str(feature)] = w
        selector = _require_selector(params.get("selector_mode"))
        normalized_params = {"weights": weights, "selector_mode": selector}
        policy = ContinuousScoreChallenger(name, weights, selector_mode=selector)

    elif family == "linear_ranking":
        feats_raw = params.get("feature_subset")
        if not isinstance(feats_raw, list):
            raise ValueError("linear_ranking.feature_subset must be an array")
        features = list(dict.fromkeys(str(x) for x in feats_raw))
        if not (2 <= len(features) <= 8):
            raise ValueError("linear_ranking.feature_subset must contain 2..8 unique features")
        invalid = [x for x in features if x not in numeric_allowed]
        if invalid:
            raise ValueError(f"Non-PIT or non-numeric linear features: {invalid}")
        regularization = _bounded_float(params.get("regularization", 1.0), 0.25, 5.0, "regularization")
        selector = _require_selector(params.get("selector_mode"))
        normalized_params = {
            "feature_subset": features,
            "regularization": regularization,
            "selector_mode": selector,
        }
        policy = MultiFeatureLinearChallenger(
            name,
            features,
            regularization=regularization,
            selector_mode=selector,
        )

    elif family == "portfolio":
        lam = _bounded_float(params.get("concentration_lambda", 1.0), 0.0, 5.0, "concentration_lambda")
        metric = str(params.get("stock_quality_metric") or "balanced")
        if metric not in {"balanced", "momentum_first"}:
            raise ValueError(f"Unsupported stock_quality_metric: {metric}")
        normalized_params = {
            "concentration_lambda": lam,
            "stock_quality_metric": metric,
        }
        policy = PortfolioUtilityChallenger(
            name,
            concentration_lambda=lam,
            stock_quality_metric=metric,
        )

    elif family == "novel_heuristic":
        dry = _bounded_float(params.get("dry_weight", 2.0), 0.0, 6.0, "dry_weight")
        depth = _bounded_float(params.get("base_depth_penalty", 1.0), 0.0, 6.0, "base_depth_penalty")
        vol = _bounded_float(params.get("volume_spike_bonus", 2.0), 0.0, 6.0, "volume_spike_bonus")
        selector = _require_selector(params.get("selector_mode"))
        normalized_params = {
            "dry_weight": dry,
            "base_depth_penalty": depth,
            "volume_spike_bonus": vol,
            "selector_mode": selector,
        }
        policy = NovelHeuristicChallenger(
            name,
            dry_weight=dry,
            base_depth_penalty=depth,
            volume_spike_bonus=vol,
            selector_mode=selector,
        )

    else:
        raise ValueError(f"Unsupported RD-Agent discovery family: {family!r}")

    normalized_record = {
        "policy_id": policy.policy_id,
        "family": policy.family,
        "name": name,
        "hypothesis": str(record.get("hypothesis") or "").strip(),
        "spec_params": normalized_params,
        "spec_hash": policy.spec_hash,
        "fitted_state_hash": policy.fitted_state_hash,
        "source_response_hash": str(record.get("source_response_hash") or ""),
        "source_response_path": str(record.get("source_response_path") or ""),
        "source_model": str(record.get("source_model") or ""),
        "proposal_engine": "rdagent_model",
    }
    return policy, normalized_record


def normalize_discovery_records(
    raw_records: list[dict[str, Any]],
) -> tuple[list[ChallengerProtocol], list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate blind model proposals without letting one malformed item kill the family.

    Invalid proposals are rejected individually and preserved in the audit ledger.
    The run still fails closed if any family represented in the RD-Agent response
    has zero executable proposals after schema validation.
    """
    policies: list[ChallengerProtocol] = []
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    family_counts: dict[str, int] = {}

    attempted_families = {
        str(raw.get("family") or "").strip()
        for raw in raw_records
        if str(raw.get("family") or "").strip()
    }

    def reject(raw: dict[str, Any], reason: str) -> None:
        rejected.append({
            "family": str(raw.get("family") or "").strip(),
            "name": str(raw.get("name") or "").strip(),
            "reason": str(reason),
            "source_response_hash": str(raw.get("source_response_hash") or ""),
            "source_response_path": str(raw.get("source_response_path") or ""),
            "source_model": str(raw.get("source_model") or ""),
        })

    for raw in raw_records:
        try:
            policy, rec = instantiate_discovery_proposal(raw)
        except (TypeError, ValueError) as exc:
            reject(raw, f"schema_validation: {exc}")
            continue

        if policy.policy_id in seen_ids:
            reject(raw, f"duplicate_policy_id: {policy.policy_id}")
            continue

        next_count = family_counts.get(policy.family, 0) + 1
        if next_count > int(FAMILY_BUDGETS[policy.family]):
            reject(raw, f"family_budget_exceeded: {policy.family}")
            continue

        family_counts[policy.family] = next_count
        seen_ids.add(policy.policy_id)
        policies.append(policy)
        records.append(rec)

    if not policies:
        raise RuntimeError("RD-Agent discovery produced zero executable proposals")

    empty_families = sorted(
        family for family in attempted_families
        if family_counts.get(family, 0) == 0
    )
    if empty_families:
        raise RuntimeError(
            "RD-Agent family produced zero schema-valid executable proposals: "
            + ", ".join(empty_families)
        )

    return policies, records, rejected


def instantiate_discovery_proposals(
    frozen_records: list[dict[str, Any]],
) -> list[ChallengerProtocol]:
    """Instantiate only from sealed normalized records; never regenerate discovery hypotheses."""
    policies: list[ChallengerProtocol] = []
    for rec in frozen_records:
        raw = {
            "family": rec["family"],
            "name": rec["name"],
            "hypothesis": rec.get("hypothesis", ""),
            "params": rec["spec_params"],
            "source_response_hash": rec.get("source_response_hash", ""),
            "source_response_path": rec.get("source_response_path", ""),
            "source_model": rec.get("source_model", ""),
        }
        policy, normalized = instantiate_discovery_proposal(raw)
        if policy.policy_id != rec.get("policy_id"):
            raise RuntimeError(f"Frozen policy_id mismatch for {rec.get('policy_id')}")
        if policy.spec_hash != rec.get("spec_hash"):
            raise RuntimeError(f"Frozen spec_hash mismatch for {policy.policy_id}")
        policies.append(policy)
    return policies
