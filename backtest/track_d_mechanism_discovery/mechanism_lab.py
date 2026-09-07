from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from typing import Any

import pandas as pd

from dashboard.skill_industry_eps_known import is_pullback_rule, is_review_universe, to_bool
from backtest.track_c_ranking_discovery.b0_ablation_grid import reasoned_item_variant
from backtest.track_c_ranking_discovery.discovery_sandbox.discovery_runner import controlled_eligible
from backtest.track_c_ranking_discovery.evaluate_econometrics import evaluate_paired_challenger
from backtest.track_c_ranking_discovery.protocol import compute_3slot_portfolio_weekly

from .config import (
    B0_COMPONENTS,
    INTERACTION_PAIRS,
    MAX_MINIMAL_B0_REMOVALS,
    PRIMARY_HORIZON,
    TOP_N,
)


@dataclass(frozen=True)
class MechanismSpec:
    policy_id: str
    experiment_kind: str
    components: tuple[str, ...]
    dry_false_penalty: bool = True
    standalone_dry_term: bool = False
    selector_mode: str = "distinct_1"
    capacity_mode: str = "fixed"
    max_positions: int = TOP_N
    comparator_id: str = "B0_FULL"


def full_b0_spec() -> MechanismSpec:
    return MechanismSpec(
        policy_id="B0_FULL",
        experiment_kind="baseline",
        components=tuple(x for x in B0_COMPONENTS if x != "distinct1"),
        dry_false_penalty=True,
        selector_mode="distinct_1",
        comparator_id="B0_FULL",
    )


def neutral_spec() -> MechanismSpec:
    return MechanismSpec(
        policy_id="NEUTRAL_CORE",
        experiment_kind="neutral",
        components=(),
        dry_false_penalty=False,
        selector_mode="unconstrained",
        comparator_id="NEUTRAL_CORE",
    )


def _remove_component(spec: MechanismSpec, component: str, policy_id: str) -> MechanismSpec:
    comps = tuple(x for x in spec.components if x != component)
    dry = spec.dry_false_penalty
    selector = spec.selector_mode
    if component == "dry_false_penalty":
        dry = False
    if component == "distinct1":
        selector = "unconstrained"
    return replace(
        spec,
        policy_id=policy_id,
        experiment_kind="knockout",
        components=comps,
        dry_false_penalty=dry,
        selector_mode=selector,
        comparator_id="B0_FULL",
    )


def _rescue_component(component: str) -> MechanismSpec:
    comps: tuple[str, ...] = ()
    dry = False
    standalone_dry = False
    selector = "unconstrained"

    if component == "dry_false_penalty":
        standalone_dry = True
    elif component == "distinct1":
        selector = "distinct_1"
    else:
        comps = (component,)

    return MechanismSpec(
        policy_id=f"RESCUE__{component}",
        experiment_kind="rescue",
        components=comps,
        dry_false_penalty=dry,
        standalone_dry_term=standalone_dry,
        selector_mode=selector,
        comparator_id="NEUTRAL_CORE",
    )


def generate_mechanism_specs() -> list[MechanismSpec]:
    full = full_b0_spec()
    specs = [full, neutral_spec()]

    for comp in B0_COMPONENTS:
        specs.append(_remove_component(full, comp, f"KNOCKOUT__{comp}"))
        specs.append(_rescue_component(comp))

    for a, b in INTERACTION_PAIRS:
        comps = tuple(x for x in (a, b) if x not in {"dry_false_penalty", "distinct1", "capacity"})
        specs.append(MechanismSpec(
            policy_id=f"INTERACTION__{a}__X__{b}",
            experiment_kind="interaction",
            components=comps,
            dry_false_penalty=False,
            standalone_dry_term=("dry_false_penalty" in {a, b}),
            selector_mode="distinct_1" if "distinct1" in {a, b} else "unconstrained",
            capacity_mode="evidence_positive" if "capacity" in {a, b} else "fixed",
            max_positions=TOP_N,
            comparator_id="NEUTRAL_CORE",
        ))

    specs.extend([
        replace(
            full,
            policy_id="CAPACITY__TOP1",
            experiment_kind="capacity",
            max_positions=1,
            comparator_id="B0_FULL",
        ),
        replace(
            full,
            policy_id="CAPACITY__TOP2",
            experiment_kind="capacity",
            max_positions=2,
            comparator_id="B0_FULL",
        ),
        replace(
            full,
            policy_id="CAPACITY__EVIDENCE_POSITIVE",
            experiment_kind="capacity",
            capacity_mode="evidence_positive",
            max_positions=3,
            comparator_id="B0_FULL",
        ),
    ])
    return specs


def generate_minimal_b0_specs() -> list[MechanismSpec]:
    """Pre-register a bounded subset lattice for Minimal-B0 compression.

    Only removal is allowed; no thresholds are fit to outcomes. Distinct1 removal
    becomes unconstrained selection. dry_false_penalty removal preserves dry=True
    reward but removes the dry=False risk via reward_only semantics.
    """
    full = full_b0_spec()
    out: list[MechanismSpec] = []
    for remove_n in range(1, MAX_MINIMAL_B0_REMOVALS + 1):
        for removed in itertools.combinations(B0_COMPONENTS, remove_n):
            spec = full
            for comp in removed:
                spec = _remove_component(spec, comp, spec.policy_id)
            out.append(replace(
                spec,
                policy_id="MINIMAL_B0__DROP__" + "__".join(removed),
                experiment_kind="minimal_b0",
                comparator_id="B0_FULL",
            ))
    return out


class MechanismPolicy:
    """Deterministic B0 component ablation/rescue policy on the common B0 universe."""

    def __init__(self, spec: MechanismSpec):
        self.spec = spec
        self.policy_id = f"TRACK_D_MECH__{spec.policy_id}"
        self.family = "track_d_mechanism"
        self.spec_hash = self.policy_id + "::" + "::".join(spec.components)
        self.fitted_state_hash = "none"

    def score_candidates(self, snapshot_df: pd.DataFrame) -> pd.DataFrame:
        if snapshot_df.empty:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        dry_policy = "symmetric" if self.spec.dry_false_penalty else "reward_only"

        for row_idx, (_, row) in enumerate(snapshot_df.iterrows()):
            if not is_review_universe(row):
                continue
            item = reasoned_item_variant(row, row_idx, dry_policy=dry_policy, lane_policy="B0_LANE")
            base_key = item.sort_key

            key_parts: list[Any] = [base_key[0]]
            if "lane" in self.spec.components:
                key_parts.append(base_key[1])
            key_parts.append(base_key[2])

            if "evidence_risk" in self.spec.components:
                key_parts.extend([base_key[3], base_key[4]])

            rule = str(row.get("ibd_candidate_rule", "") or "").strip()
            dry_false = bool(
                is_pullback_rule(rule)
                and to_bool(row.get("pullback_v_is_dry")) is False
            )
            if self.spec.standalone_dry_term:
                key_parts.append(1 if dry_false else 0)

            if "freshness" in self.spec.components:
                key_parts.append(base_key[5])
            if "eps_preference" in self.spec.components:
                key_parts.append(base_key[6])
            if "weekly_volume" in self.spec.components:
                key_parts.append(base_key[7])
            if "entry_volume" in self.spec.components:
                key_parts.append(base_key[8])

            key_parts.extend([item.code, row_idx])

            evidence_count = sum(
                x in item.reason_codes
                for x in [
                    "near_buy_point", "volume_confirms_breakout", "eps_acceleration_support",
                    "weekly_volume_follow_through", "near_52w_high", "dry_pullback",
                ]
            )
            risk_count = sum(
                x in item.risk_codes
                for x in [
                    "non_actionable_radar_only", "freshness_missing", "below_candidate_buy_point",
                    "extended_from_buy_point", "entry_volume_missing",
                    "entry_volume_below_standard", "pullback_not_dry",
                ]
            )
            raw_eligible = row.get("b0_eligible", False)
            if pd.isna(raw_eligible):
                eligible = False
            elif isinstance(raw_eligible, bool):
                eligible = raw_eligible
            elif isinstance(raw_eligible, (int, float)):
                eligible = float(raw_eligible) == 1.0
            else:
                eligible = str(raw_eligible).strip().lower() in {"true", "1", "1.0", "yes"}
            rows.append({
                "code": item.code,
                "industry": item.industry,
                "industry_key": str(item.industry or "").strip().lower(),
                "b0_eligible": eligible,
                "candidate_sort_key": tuple(key_parts),
                "evidence_balance": float(evidence_count - risk_count),
                "dry_false": dry_false,
            })

        scored = pd.DataFrame(rows)
        if scored.empty:
            return scored
        scored = scored.sort_values("candidate_sort_key", kind="stable").reset_index(drop=True)
        scored["raw_rank"] = range(1, len(scored) + 1)
        return scored

    def allocate_industries(self, scored_df: pd.DataFrame) -> dict[str, int]:
        if scored_df.empty:
            return {}
        eligible = controlled_eligible(scored_df)
        inds = eligible["industry_key"].fillna("").astype(str).str.strip().str.lower().unique()
        if self.spec.selector_mode == "distinct_1":
            quota = 1
        elif self.spec.selector_mode == "max_2_per_ind":
            quota = 2
        else:
            quota = TOP_N
        return {ind: quota for ind in inds if ind}

    def pick_stocks(self, scored_df: pd.DataFrame, industry_quotas: dict[str, int]) -> list[str]:
        if scored_df.empty or not industry_quotas:
            return []
        eligible = controlled_eligible(scored_df).sort_values("raw_rank")
        if self.spec.capacity_mode == "evidence_positive":
            eligible = eligible[eligible["evidence_balance"] > 0]

        selected: list[str] = []
        counts: dict[str, int] = {}
        for _, row in eligible.iterrows():
            if len(selected) >= int(self.spec.max_positions):
                break
            ind = str(row["industry_key"]).strip().lower()
            if counts.get(ind, 0) >= int(industry_quotas.get(ind, 0)):
                continue
            selected.append(str(row["code"]))
            counts[ind] = counts.get(ind, 0) + 1
        return selected


def run_policy(
    policy: Any,
    panel_df: pd.DataFrame,
    snapshots: list[str],
    selector_id: str | None = None,
    horizon: str = PRIMARY_HORIZON,
):
    picks_rows = []
    for snap in snapshots:
        s_df = panel_df[panel_df["snapshot_date"].astype(str) == str(snap)].copy()
        scored = policy.score_candidates(s_df)
        quotas = policy.allocate_industries(scored)
        picks = policy.pick_stocks(scored, quotas)
        for code in picks:
            match = s_df[s_df["code"].astype(str) == str(code)]
            if match.empty:
                raise RuntimeError(f"Policy selected code outside snapshot panel: {snap}/{code}")
            picks_rows.append(match.iloc[0])

    picks_df = pd.DataFrame(picks_rows) if picks_rows else pd.DataFrame()
    return compute_3slot_portfolio_weekly(
        picks_df,
        snapshots,
        selector_id or policy.policy_id,
        horizon,
    )


def run_mechanism_lab(panel_df: pd.DataFrame, discovery_snapshots: list[str]) -> pd.DataFrame:
    specs = generate_mechanism_specs()
    policies = {spec.policy_id: MechanismPolicy(spec) for spec in specs}
    outcomes = {
        spec.policy_id: run_policy(policies[spec.policy_id], panel_df, discovery_snapshots, spec.policy_id)
        for spec in specs
    }

    rows: list[dict[str, Any]] = []
    for spec in specs:
        if spec.experiment_kind in {"baseline", "neutral"}:
            continue
        comparator = outcomes[spec.comparator_id]
        summary = evaluate_paired_challenger(
            outcomes[spec.policy_id],
            comparator,
            spec.policy_id,
            "track_d_mechanism",
            "discovery_train",
            PRIMARY_HORIZON,
        )
        rows.append({
            "policy_id": spec.policy_id,
            "experiment_kind": spec.experiment_kind,
            "components": "|".join(spec.components),
            "comparator_id": spec.comparator_id,
            "support_weeks": summary.support_weeks,
            "mean_spread": summary.mean_spread,
            "median_spread": summary.median_spread,
            "cvar_delta": summary.cvar_delta,
            "stop_delta_pct": summary.stop_delta_pct,
            "one_pick_ruins_delta_pct": summary.one_pick_ruins_delta_pct,
            "slot_coverage_pct": summary.slot_coverage_pct,
            "jaccard_vs_comparator": summary.top3_membership_jaccard_vs_b0,
            "ci_low": summary.bootstrap.mean_spread_ci_low if summary.bootstrap else 0.0,
            "ci_high": summary.bootstrap.mean_spread_ci_high if summary.bootstrap else 0.0,
        })
    return pd.DataFrame(rows)
