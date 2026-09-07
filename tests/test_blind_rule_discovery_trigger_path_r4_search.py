from __future__ import annotations

import pandas as pd

from backtest.blind_rule_discovery.trigger_path_characterization_r4 import (
    search_stock_interactions,
)
from backtest.blind_rule_discovery.trigger_path_characterization_r4_search import (
    enrich_full_metrics,
    search_stock_interactions_fast,
)


def frame(quarters: int = 6, rows_per_quarter: int = 40) -> pd.DataFrame:
    rows = []
    for q_index in range(quarters):
        quarter = f"{2023 + q_index // 4}Q{q_index % 4 + 1}"
        snapshot = pd.Period(quarter, freq="Q").start_time + pd.Timedelta(days=10)
        for i in range(rows_per_quarter):
            high = i >= rows_per_quarter // 2
            fast = int(high and i % 4 != 0)
            stop = int((not high) and i % 4 != 0)
            unresolved = int(not fast and not stop)
            rows.append(
                {
                    "snapshot_date": snapshot + pd.Timedelta(days=i % 2),
                    "entry_quarter": quarter,
                    "r3_favorable_regime": 1,
                    "A": 1.0 if high else 0.0,
                    "B": i / rows_per_quarter,
                    "ambiguous_3w": 0,
                    "fast_winner_3w": fast,
                    "stop_first_3w": stop,
                    "unresolved_3w": unresolved,
                    "stop_first_then_winner_12w": 0,
                    "return_w1": 0.06 if high else -0.03,
                    "return_w2": 0.09 if high else -0.05,
                    "return_w3": 0.13 if high else -0.07,
                    "return_w4": 0.15 if high else -0.06,
                    "excess_w1": 0.05 if high else -0.04,
                    "excess_w2": 0.07 if high else -0.06,
                    "excess_w3": 0.11 if high else -0.08,
                    "excess_w4": 0.12 if high else -0.07,
                    "mae_3w": -0.03 if high else -0.10,
                    "mfe_3w": 0.22 if high else 0.05,
                    "mae_4w": -0.04 if high else -0.11,
                    "mfe_4w": 0.25 if high else 0.07,
                }
            )
    return pd.DataFrame(rows)


def test_vectorized_search_preserves_full_search_top_rule_and_pair_count():
    data = frame()
    kwargs = dict(
        min_selected=20,
        min_evaluable=20,
        min_quarter_n=5,
        min_evaluated_quarters=4,
    )
    full, full_audit = search_stock_interactions(data, ["A", "B"], **kwargs)
    fast, fast_audit = search_stock_interactions_fast(data, ["A", "B"], **kwargs)
    assert not full.empty and not fast.empty
    assert fast.iloc[0]["rule_json"] == full.iloc[0]["rule_json"]
    assert fast_audit["distinct_feature_pair_count_tested"] == full_audit["distinct_feature_pair_count_tested"]
    assert fast_audit["candidate_rule_count"] == full_audit["candidate_rule_count"]


def test_full_enrichment_adds_w1_w4_distribution_only_after_search():
    data = frame()
    compact, _ = search_stock_interactions_fast(
        data,
        ["A", "B"],
        min_selected=20,
        min_evaluable=20,
        min_quarter_n=5,
        min_evaluated_quarters=4,
    )
    assert "return_w1_p25" not in compact.columns
    enriched = enrich_full_metrics(data, compact, top_n=5)
    assert len(enriched) == min(5, len(compact))
    for week in ("w1", "w2", "w3", "w4"):
        for q in ("p25", "p50", "p75"):
            assert f"return_{week}_{q}" in enriched.columns
            assert f"excess_{week}_{q}" in enriched.columns
