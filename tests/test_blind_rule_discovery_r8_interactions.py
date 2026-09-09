from pathlib import Path

import pandas as pd

from backtest.blind_rule_discovery.r6_stability import canonical
from backtest.blind_rule_discovery.r8_atlas import AtlasConfig
from backtest.blind_rule_discovery.r8_interactions import (
    PROFILE_SCHEMA,
    compact_feedback,
    compact_past_profile,
    discover_interactions_compact,
)


def synthetic_feature_frame(periods=8, weeks=4, feature_count=19):
    rows = []
    features = [f"feature_{i:02d}" for i in range(feature_count)]
    for q_idx, quarter in enumerate(pd.period_range("2023Q1", periods=periods, freq="Q")):
        for week in range(weeks):
            day = quarter.start_time + pd.Timedelta(days=(4-quarter.start_time.dayofweek) % 7 + 7*week)
            for label_idx, label in enumerate(("fast_winner_3w", "stop_first_3w", "unresolved_3w", "ambiguous_3w")):
                for sample in range(6):
                    row = {
                        "code": f"{q_idx}-{week}-{label_idx}-{sample}",
                        "snapshot_date": day,
                        "entry_date": day + pd.Timedelta(days=1),
                        "exit_date_w3": day + pd.Timedelta(days=21),
                        "fast_winner_3w": int(label == "fast_winner_3w"),
                        "stop_first_3w": int(label == "stop_first_3w"),
                        "unresolved_3w": int(label == "unresolved_3w"),
                        "ambiguous_3w": int(label == "ambiguous_3w"),
                    }
                    for i, feature in enumerate(features):
                        row[feature] = q_idx + week/10 + sample/100 + label_idx/1000 + i/10000
                    rows.append(row)
    return pd.DataFrame(rows), features


def test_compact_profile_keeps_every_feature_with_fixed_short_schema():
    frame, features = synthetic_feature_frame()
    profile = compact_past_profile(frame, features)
    assert profile["schema"] == list(PROFILE_SCHEMA)
    assert [row[0] for row in profile["rows"]] == features
    assert len(profile["rows"]) == 19
    assert len(canonical(profile)) < 5000


def test_compact_feedback_does_not_echo_full_inner_fold_history():
    proposal = {"name": "x", "hypothesis": "h", "expression": {}, "target": "winner", "tail": "high", "quantile": .8}
    evidence = {
        "supported_quarters": 4,
        "evaluated_quarters": 5,
        "positive_fraction_all_quarters": .8,
        "median_matched_target_lift": .123456,
        "median_capture_minus_other_loss": .045678,
        "folds": [{"quarter": "2024Q1", "very_large": "x" * 1000}],
    }
    compact = compact_feedback(proposal, evidence)
    assert "folds" not in compact["evidence"]
    assert compact["evidence"]["median_matched_target_lift"] == .1235


def test_discovery_first_prompt_is_compact_and_contains_all_features(tmp_path: Path):
    frame, features = synthetic_feature_frame(periods=8)
    calendar = [str(q) for q in pd.period_range("2023Q1", periods=8, freq="Q")]
    payloads = []

    def proposer(payload):
        payloads.append(payload)
        return {"proposals": []}

    output = tmp_path / "interactions"
    output.mkdir()
    discover_interactions_compact(frame, features, calendar, proposer, AtlasConfig(), output)
    assert payloads
    first = payloads[0]
    assert first["contract"] == "R8_COMPACT_INTERACTION_V1"
    assert "univariate_profile" not in first
    assert [row[0] for row in first["profile"]["rows"]] == features
    assert len(canonical(first)) < 6000
    assert (output / "interaction_frozen.json").exists()
    assert (output / "discovery_trace.jsonl").exists()
