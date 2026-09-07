from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from dashboard.skill_industry_eps_known import select_skill_industry_eps_known
from backtest.track_d_mechanism_discovery.config import (
    HISTORICAL_END,
    OUTER_BLOCKS,
    OUTER_TEST_SNAPSHOTS,
    SCREENING_BLOCKS,
    CONFIRMATION_BLOCKS,
    B0_COMPONENTS,
    PANEL_SOURCE,
    POLICY_REVIEW_CALLS,
    POLICY_SYNTHESIS_CALLS,
    FINAL_INTERPRETATION_CALLS,
    REQUEST_HARD_LIMIT,
    RESEARCH_ROLE_SEQUENCE,
)
from backtest.track_d_mechanism_discovery.mechanism_lab import (
    MechanismPolicy,
    full_b0_spec,
    generate_mechanism_specs,
    generate_minimal_b0_specs,
)
from backtest.track_d_mechanism_discovery.policy_dsl import (
    DSLPolicy,
    validate_policy_spec,
)
from backtest.track_d_mechanism_discovery.research_questions import build_question_plan
from backtest.track_d_mechanism_discovery.request_budget import (
    DuplicateResearchRequest,
    RequestBudgetExceeded,
    RequestBudgetLedger,
)
from backtest.track_d_mechanism_discovery.walk_forward import build_locked_forward_split


def _historical_panel() -> pd.DataFrame:
    df = pd.read_parquet(PANEL_SOURCE)
    return df[df["snapshot_date"].astype(str) <= HISTORICAL_END].copy()


def test_deep_research_plan_is_unique_and_bounded():
    plan = build_question_plan()
    assert len(plan) == 78
    assert len({q.fingerprint for q in plan}) == len(plan)
    mechanism_text = " ".join(
        q.question for q in plan if q.direction == "mechanism_falsification"
    )
    assert all(component in mechanism_text for component in B0_COMPONENTS)
    planned = (
        len(plan) * len(RESEARCH_ROLE_SEQUENCE)
        + POLICY_SYNTHESIS_CALLS
        + POLICY_REVIEW_CALLS
        + FINAL_INTERPRETATION_CALLS
    )
    assert planned == 335
    assert planned < REQUEST_HARD_LIMIT


def test_focused_question_budget_by_direction():
    from collections import Counter

    plan = build_question_plan()
    counts = Counter(q.direction for q in plan)
    assert counts == {
        "mechanism_falsification": 20,
        "failure_archaeology": 22,
        "capacity_abstention": 10,
        "lane_mechanism": 8,
        "nonlinear_b1": 12,
        "adversarial_review": 6,
    }


def test_track_d_default_retry_cap_is_four(monkeypatch):
    from backtest.track_d_mechanism_discovery.llm_client import load_model_config

    monkeypatch.setenv("TRACK_D_RDAGENT_MODEL", "deepseek/deepseek-v4-pro")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_API_BASE", "https://example.invalid/v1")
    monkeypatch.setenv("MAX_RETRY", "15")
    monkeypatch.delenv("TRACK_D_MAX_RETRY", raising=False)

    cfg = load_model_config()
    assert cfg["max_retry"] == 4


def test_request_budget_counts_attempts_and_blocks_duplicate_success(tmp_path):
    ledger = RequestBudgetLedger(tmp_path / "requests.json", hard_limit=2)
    h = ledger.prompt_hash("system", "unique prompt")
    ledger.reserve_attempt("p1", h, 1)
    ledger.mark_success("p1", h, "response-hash")

    with pytest.raises(DuplicateResearchRequest):
        ledger.reserve_attempt("different-purpose", h, 1)

    h2 = ledger.prompt_hash("system", "second prompt")
    ledger.reserve_attempt("p2", h2, 1)
    assert ledger.remaining == 0
    with pytest.raises(RequestBudgetExceeded):
        ledger.reserve_attempt("p3", ledger.prompt_hash("s", "third"), 1)


def test_policy_dsl_blocks_outcomes_and_invalid_transform():
    with pytest.raises(ValueError, match="not numeric PIT-allowed"):
        validate_policy_spec({
            "policy_id": "leaky",
            "base": "zero",
            "terms": [{"type": "linear", "feature": "w4_return_pct", "weight": 1.0}],
            "selector": {"industry_mode": "distinct_1", "capacity": {"mode": "fixed", "max_positions": 3}},
        })

    with pytest.raises(ValueError, match="interaction transform"):
        validate_policy_spec({
            "policy_id": "bad_transform",
            "base": "zero",
            "terms": [{
                "type": "interaction",
                "left": "mom_20",
                "right": "eps_yoy_growth",
                "transform": "magic",
                "weight": 1.0,
            }],
            "selector": {"industry_mode": "distinct_1", "capacity": {"mode": "fixed", "max_positions": 3}},
        })


def test_dsl_policy_never_escapes_b0_eligible_common_universe():
    panel = _historical_panel()
    snap = sorted(panel["snapshot_date"].astype(str).unique())[0]
    s_df = panel[panel["snapshot_date"].astype(str) == snap].copy()

    policy = DSLPolicy({
        "policy_id": "common_universe_probe",
        "base": "b0_rank",
        "terms": [
            {"type": "linear", "feature": "mom_20", "transform": "zscore", "weight": 2.0},
            {"type": "linear", "feature": "eps_yoy_growth", "transform": "rank_pct", "weight": 1.0},
        ],
        "selector": {
            "industry_mode": "unconstrained",
            "capacity": {"mode": "fixed", "max_positions": 3},
        },
    })
    scored = policy.score_candidates(s_df)
    quotas = policy.allocate_industries(scored)
    picks = policy.pick_stocks(scored, quotas)

    eligible_codes = set(
        s_df[s_df["b0_eligible"].fillna(False).astype(bool)]["code"].astype(str)
    )
    assert set(picks).issubset(eligible_codes)
    assert len(picks) <= 3


def test_full_mechanism_policy_matches_production_b0_on_sample():
    panel = _historical_panel()
    snaps = sorted(panel["snapshot_date"].astype(str).unique())[:8]
    policy = MechanismPolicy(full_b0_spec())

    for snap in snaps:
        s_df = panel[panel["snapshot_date"].astype(str) == snap].copy()
        expected = [x.code for x in select_skill_industry_eps_known(s_df, limit=3)]
        scored = policy.score_candidates(s_df)
        quotas = policy.allocate_industries(scored)
        actual = policy.pick_stocks(scored, quotas)
        assert actual == expected, f"{snap}: {actual} != {expected}"


def test_mechanism_and_minimal_b0_search_are_pre_registered_and_bounded():
    specs = generate_mechanism_specs()
    ids = [x.policy_id for x in specs]
    assert len(ids) == len(set(ids))
    assert len(specs) == 28  # B0 + neutral + 8x(knockout,rescue) + 7 interactions + 3 capacity probes

    minimal = generate_minimal_b0_specs()
    assert len(minimal) == 162  # sum C(8,k), k=1..4
    assert all(x.experiment_kind == "minimal_b0" for x in minimal)


def test_locked_forward_split_is_non_overlapping_and_exact():
    panel = _historical_panel()
    snaps = sorted(panel["snapshot_date"].astype(str).unique())
    split = build_locked_forward_split(snaps)

    discovery = set(split["discovery_train"])
    purge = set(split["purge"])
    blocks = [set(x["snapshots"]) for x in split["forward_blocks"]]
    stages = [x["stage"] for x in split["forward_blocks"]]

    assert len(blocks) == OUTER_BLOCKS
    assert stages.count("screening") == SCREENING_BLOCKS
    assert stages.count("confirmation") == CONFIRMATION_BLOCKS
    assert len(split["screening_blocks"]) == SCREENING_BLOCKS
    assert len(split["confirmation_blocks"]) == CONFIRMATION_BLOCKS
    assert all(len(x) == OUTER_TEST_SNAPSHOTS for x in blocks)
    assert not discovery & purge
    assert all(not discovery & b for b in blocks)
    assert all(not purge & b for b in blocks)
    for i, a in enumerate(blocks):
        for b in blocks[i + 1:]:
            assert not a & b



def test_policy_dsl_rejects_duplicate_policy_ids_with_different_specs():
    from backtest.track_d_mechanism_discovery.policy_dsl import deduplicate_policy_specs

    base = {
        "policy_id": "same_id",
        "base": "zero",
        "selector": {
            "industry_mode": "distinct_1",
            "capacity": {"mode": "fixed", "max_positions": 3},
        },
    }
    kept, dropped = deduplicate_policy_specs([
        {
            **base,
            "terms": [
                {"type": "linear", "feature": "mom_20", "transform": "zscore", "weight": 1.0}
            ],
        },
        {
            **base,
            "terms": [
                {"type": "linear", "feature": "mom_60", "transform": "zscore", "weight": 1.0}
            ],
        },
    ])
    assert len(kept) == 1
    assert len(dropped) == 1
    assert dropped[0]["reason"] == "duplicate_policy_id_with_different_spec"



def test_track_d_evidence_bundle_handles_boolean_pit_quantiles():
    from backtest.track_d_mechanism_discovery.research_loop import build_evidence_bundle

    discovery = pd.DataFrame({
        "pullback_v_is_dry": [True, False, True, True],
        "mom_20": [0.10, 0.20, 0.30, 0.40],
    })
    mechanism = pd.DataFrame([{"policy_id": "probe", "mean_spread": 0.0}])
    evidence = build_evidence_bundle(
        mechanism,
        {"case_count": 0, "labels": {}, "numeric_contrasts": []},
        discovery,
    )

    dry = evidence["pit_distributions"]["pullback_v_is_dry"]
    assert dry["p50"] == pytest.approx(1.0)
    assert evidence["pit_distributions"]["mom_20"]["p50"] == pytest.approx(0.25)
