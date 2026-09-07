from __future__ import annotations

import json

import pandas as pd

from backtest.blind_rule_discovery.stop_risk_validation_r5 import (
    RISK_FAMILIES,
    build_risk_family_rule,
    rank_stop_risk_rules,
    rolling_risk_families,
    rolling_stop_risk_research,
    summarize_family_rolling,
    summarize_risk_rolling,
)
from backtest.blind_rule_discovery.trigger_path_characterization_r4_search import search_stock_interactions_fast


def risk_frame(quarters: int = 8, rows_per_quarter: int = 60) -> pd.DataFrame:
    rows = []
    for q_index in range(quarters):
        quarter = f"{2023 + q_index // 4}Q{q_index % 4 + 1}"
        period = pd.Period(quarter, freq="Q")
        snapshot = period.start_time + pd.Timedelta(days=10)
        for i in range(rows_per_quarter):
            risky = i < rows_per_quarter // 3
            stop = int(risky and i % 5 != 0)
            fast = int((not risky) and i % 7 == 0)
            unresolved = int(not stop and not fast)
            exit_w3 = snapshot + pd.Timedelta(days=20)
            if i >= rows_per_quarter - 2:
                exit_w3 = period.end_time.normalize() + pd.Timedelta(days=5)
            rows.append(
                {
                    "snapshot_date": snapshot + pd.Timedelta(days=i % 3),
                    "entry_quarter": quarter,
                    "exit_date_w3": exit_w3,
                    "r3_favorable_regime": 1,
                    "pullback_pct": -22.0 if risky else -6.0 + (i % 4),
                    "current_vs_ibd_candidate_pct": 5.0 if risky else 0.5 + (i % 3) * 0.2,
                    "pct_above_ceiling": 30.0 if risky else 4.0 + (i % 5),
                    "entry_extension_pct": 0.04 if risky else 0.003 + (i % 3) * 0.001,
                    "volume_ratio": 1.6 if risky else 0.9,
                    "ambiguous_3w": 0,
                    "fast_winner_3w": fast,
                    "stop_first_3w": stop,
                    "unresolved_3w": unresolved,
                    "stop_first_then_winner_12w": int(stop and i % 4 == 0),
                    "return_w1": -0.03 if risky else 0.01,
                    "return_w2": -0.05 if risky else 0.02,
                    "return_w3": -0.07 if risky else 0.03,
                    "return_w4": -0.08 if risky else 0.04,
                    "excess_w1": -0.04 if risky else 0.01,
                    "excess_w2": -0.06 if risky else 0.02,
                    "excess_w3": -0.08 if risky else 0.03,
                    "excess_w4": -0.09 if risky else 0.04,
                    "mae_3w": -0.12 if risky else -0.03,
                    "mfe_3w": 0.05 if risky else 0.10,
                    "mae_4w": -0.13 if risky else -0.04,
                    "mfe_4w": 0.06 if risky else 0.12,
                }
            )
    return pd.DataFrame(rows)


def test_risk_ranking_selects_consistent_high_stop_rule():
    data = risk_frame(quarters=6)
    features = [
        "pullback_pct",
        "current_vs_ibd_candidate_pct",
        "pct_above_ceiling",
        "entry_extension_pct",
        "volume_ratio",
    ]
    scored, _ = search_stock_interactions_fast(
        data,
        features,
        min_selected=40,
        min_evaluable=30,
        min_quarter_n=5,
        min_evaluated_quarters=4,
    )
    baseline_persistent = data["stop_first_3w"].sub(data["stop_first_then_winner_12w"]).clip(lower=0).mean()
    ranked = rank_stop_risk_rules(scored, baseline_persistent_rate=float(baseline_persistent))
    assert not ranked.empty
    best = ranked.iloc[0]
    assert best["higher_stop_risk_quarter_fraction"] == 1.0
    assert best["stop_first_lift"] > 0.25
    assert best["persistent_stop_first_lift"] > 0.15


def test_r5_rolling_preserves_every_fold_and_generalizes_synthetic_risk():
    data = risk_frame()
    features = [
        "pullback_pct",
        "current_vs_ibd_candidate_pct",
        "pct_above_ceiling",
        "entry_extension_pct",
        "volume_ratio",
    ]
    rolling = rolling_stop_risk_research(data, features, min_train_quarters=4, min_quarter_n=5)
    assert len(rolling) == 8  # four chronological folds for each of two scopes
    assert (~rolling["test_quarter_in_train"].astype(bool)).all()
    assert (~rolling["w3_label_overlap_after_purge"].astype(bool)).all()
    assert (rolling["train_rows_purged_for_w3_overlap"] > 0).any()
    assert (pd.to_numeric(rolling["test_stop_first_lift"], errors="coerce").dropna() > 0).all()
    assert (pd.to_numeric(rolling["test_persistent_stop_first_lift"], errors="coerce").dropna() > 0).all()
    assert (pd.to_numeric(rolling["test_matched_stop_first_lift_p50"], errors="coerce").dropna() > 0).all()
    for rule in rolling["rule_json"].dropna():
        assert "M_" not in str(rule)


def test_post_r4_family_thresholds_are_training_only_quantiles():
    train = risk_frame(quarters=4)
    rule = build_risk_family_rule(train, "deep_pullback_and_extended")
    assert rule is not None
    conditions = {feature: (op, threshold) for feature, op, threshold in rule["conditions"]}
    expected_pullback = pd.to_numeric(train["pullback_pct"]).quantile(0.20)
    expected_extension = pd.to_numeric(train["current_vs_ibd_candidate_pct"]).quantile(0.80)
    assert conditions["pullback_pct"][0] == "<="
    assert conditions["pullback_pct"][1] == expected_pullback
    assert conditions["current_vs_ibd_candidate_pct"][0] == ">="
    assert conditions["current_vs_ibd_candidate_pct"][1] == expected_extension
    assert set(RISK_FAMILIES) == {
        "deep_pullback",
        "extended_vs_candidate",
        "deep_pullback_and_extended",
        "high_pct_above_ceiling",
        "high_entry_extension",
    }


def test_risk_family_rolling_keeps_all_families_folds_and_matched_risk():
    data = risk_frame()
    families = rolling_risk_families(data, min_train_quarters=4)
    assert len(families) == 2 * 4 * len(RISK_FAMILIES)
    assert (~families["test_quarter_in_train"].astype(bool)).all()
    assert (~families["w3_label_overlap_after_purge"].astype(bool)).all()
    deep_extended = families.loc[families["family"] == "deep_pullback_and_extended"]
    assert (pd.to_numeric(deep_extended["test_stop_first_lift"], errors="coerce").dropna() > 0).all()
    assert (pd.to_numeric(deep_extended["test_persistent_stop_first_lift"], errors="coerce").dropna() > 0).all()
    assert (pd.to_numeric(deep_extended["test_matched_stop_first_lift_p50"], errors="coerce").dropna() > 0).all()
    for payload in deep_extended["family_rule_json"].dropna():
        decoded = json.loads(payload)
        assert decoded["family"] == "deep_pullback_and_extended"


def test_r5_summaries_count_all_folds_and_pooled_risk():
    data = risk_frame()
    features = [
        "pullback_pct",
        "current_vs_ibd_candidate_pct",
        "pct_above_ceiling",
        "entry_extension_pct",
        "volume_ratio",
    ]
    rolling = rolling_stop_risk_research(data, features, min_train_quarters=4, min_quarter_n=5)
    summary = summarize_risk_rolling(rolling)
    for scope in ("all", "r3_favorable"):
        item = summary[scope]
        assert item["folds"] == 4
        assert item["post_purge_overlap_fold_count"] == 0
        assert item["positive_stop_lift_all_fold_fraction"] == 1.0
        assert item["pooled_selected_stop_first_lift"] > 0
        assert item["pooled_selected_persistent_stop_lift"] > 0
        assert item["matched_positive_stop_lift_fraction"] == 1.0

    families = rolling_risk_families(data, min_train_quarters=4)
    family_summary = summarize_family_rolling(families)
    assert len(family_summary) == 2 * len(RISK_FAMILIES)
    target = [
        row for row in family_summary
        if row["scope"] == "all" and row["family"] == "deep_pullback_and_extended"
    ][0]
    assert target["folds"] == 4
    assert target["positive_stop_lift_all_fold_fraction"] == 1.0
    assert target["pooled_selected_stop_first_lift"] > 0
    assert target["pooled_selected_persistent_stop_lift"] > 0
    assert target["matched_positive_stop_lift_fraction"] == 1.0
    assert target["matched_positive_persistent_lift_fraction"] == 1.0
