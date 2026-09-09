import pandas as pd

from backtest.blind_rule_discovery.r8_runner import nonoverlap_w3_frame


def test_nonoverlap_w3_is_shared_outcome_independent_issuer_schedule():
    frame = pd.DataFrame({
        "code": ["A", "A", "A", "A", "B"],
        "snapshot_date": pd.to_datetime([
            "2025-01-03", "2025-01-10", "2025-01-24", "2025-01-31", "2025-01-10"
        ]),
        "entry_date": pd.to_datetime([
            "2025-01-06", "2025-01-13", "2025-01-27", "2025-02-03", "2025-01-13"
        ]),
        "exit_date_w3": pd.to_datetime([
            "2025-01-27", "2025-02-03", "2025-02-17", "2025-02-24", "2025-02-03"
        ]),
        "fast_winner_3w": [1, 0, 0, 1, 0],
        "stop_first_3w": [0, 1, 1, 0, 1],
        "unresolved_3w": [0, 0, 0, 0, 0],
        "ambiguous_3w": [0, 0, 0, 0, 0],
        "pullback_pct": [-5.0, -20.0, -40.0, -2.0, -10.0],
    })

    admitted = nonoverlap_w3_frame(frame)
    # A: first entry admitted; overlapping entry excluded; same-day re-entry on
    # the first W3 close excluded; the later entry is admitted. B is independent.
    assert admitted.index.tolist() == [0, 1, 2]
    assert admitted.code.tolist() == ["A", "A", "B"]
    assert admitted.entry_date.tolist() == [
        pd.Timestamp("2025-01-06"), pd.Timestamp("2025-02-03"), pd.Timestamp("2025-01-13")
    ]

    changed = frame.copy()
    changed[["fast_winner_3w", "stop_first_3w"]] = changed[["stop_first_3w", "fast_winner_3w"]].to_numpy()
    changed["pullback_pct"] *= 1000
    rerun = nonoverlap_w3_frame(changed)
    assert rerun[["code", "snapshot_date", "entry_date", "exit_date_w3"]].equals(
        admitted[["code", "snapshot_date", "entry_date", "exit_date_w3"]]
    )
