from __future__ import annotations

import inspect
from types import SimpleNamespace

import pandas as pd
import pytest

import yfinance_data
from eps_pit.lookup import SignalEPSLookup


def _valid_complete_signal(code: str, *, rank: int) -> dict[str, object]:
    return {
        "code": code,
        "snapshot_date": "2026-07-24",
        "signal": True,
        "latest_close": 101.0,
        "ibd_candidate_price": 100.0,
        "ibd_entry_valid": 1,
        "ibd_entry_status": "ACTIONABLE",
        "current_vs_ibd_candidate_pct": 1.0,
        "ibd_candidate_rule": "pivot",
        "ibd_entry_volume_ratio": 2.0,
        "ibd_entry_reject_reason": None,
        "volume_ratio": 1.2,
        "rank_C_continuous": rank,
        "C_continuous": float(rank),
    }


def test_quant_trade_import_path_and_zero_argument_contract_are_available():
    pool_type = yfinance_data.BreakoutFollowPoolRun

    assert pool_type.weekend().name == "weekend"
    assert pool_type.midweek().name == "midweek"
    assert list(inspect.signature(pool_type.weekend).parameters) == []
    assert list(inspect.signature(pool_type.midweek).parameters) == []
    assert list(inspect.signature(pool_type.load_actionable_codes).parameters) == ["self"]
    assert list(inspect.signature(pool_type.commit).parameters) == ["self"]


def test_weekend_pool_run_returns_reverse_csv_order_actionable_list(tmp_path, monkeypatch):
    pool_path = tmp_path / "breakout_follow_pool.csv"
    monkeypatch.setattr(yfinance_data, "BREAKOUT_FOLLOW_POOL_PATH", str(pool_path))
    snapshot = pd.DataFrame(
        [
            {"code": "FIRST", "signal": True, "ibd_entry_valid": 1, "ibd_entry_status": "ACTIONABLE"},
            {"code": "WAIT", "signal": True, "ibd_entry_valid": 0, "ibd_entry_status": "UNCONFIRMED"},
            {"code": "SECOND", "signal": True, "ibd_entry_valid": 1, "ibd_entry_status": "ACTIONABLE"},
            {"code": "POOL", "signal": False, "ibd_entry_valid": None, "ibd_entry_status": None},
        ]
    )
    pool_run = yfinance_data.BreakoutFollowPoolRun.weekend()
    pool_run.save_snapshot(snapshot)

    assert pool_run.load_actionable_codes() == ["SECOND", "FIRST"]


def test_pool_run_supplements_signal_eps_before_publishing(tmp_path, monkeypatch):
    pool_path = tmp_path / "breakout_follow_pool.csv"
    pit_path = tmp_path / "signal_eps_pit.csv"
    stage2_path = tmp_path / "stage2_whitelist.csv"

    pd.DataFrame(
        [{"snapshot_date": "2026-08-14", "code": "PIT", "eps_yoy_growth": 31.5}]
    ).to_csv(pit_path, index=False)
    pd.DataFrame([{"code": "STAGE", "eps_yoy_growth": 42.0}]).to_csv(stage2_path, index=False)

    SignalEPSLookup.clear_cache()
    monkeypatch.setattr(SignalEPSLookup, "DEFAULT_CSV_PATH", str(pit_path))
    monkeypatch.setattr(SignalEPSLookup, "DEFAULT_STAGE2_PATH", str(stage2_path))
    monkeypatch.setattr(yfinance_data, "BREAKOUT_FOLLOW_POOL_PATH", str(pool_path))

    pool_run = yfinance_data.BreakoutFollowPoolRun.weekend()
    pool_run.save_snapshot(
        pd.DataFrame(
            [
                {"code": "PIT", "snapshot_date": "2026-08-14", "signal": True, "eps_yoy_growth": pd.NA},
                {"code": "STAGE", "snapshot_date": "2026-08-14", "signal": True, "eps_yoy_growth": pd.NA},
                {"code": "QUIET", "snapshot_date": "2026-08-14", "signal": False, "eps_yoy_growth": pd.NA},
            ]
        )
    )

    saved = pd.read_csv(pool_path)
    assert saved.loc[saved["code"].eq("PIT"), "eps_yoy_growth"].item() == 31.5
    assert saved.loc[saved["code"].eq("STAGE"), "eps_yoy_growth"].item() == 42.0
    assert pd.isna(saved.loc[saved["code"].eq("QUIET"), "eps_yoy_growth"].item())


def test_midweek_pool_run_uses_unified_projection_and_matches_quant_fixture(tmp_path, monkeypatch):
    complete_path = tmp_path / "breakout_follow_pool.csv"
    midweek_path = tmp_path / "breakout_follow_pool_midweek.csv"
    pd.DataFrame(
        [
            _valid_complete_signal("CARRY_ACTION", rank=1),
            _valid_complete_signal("CARRY_EXTENDED", rank=2),
            _valid_complete_signal("EXITED", rank=3),
            _valid_complete_signal("CURRENT_OVERRIDE", rank=4),
        ]
    ).to_csv(complete_path, index=False)
    current = pd.DataFrame(
        [
            {"code": "NEW_ACTION", "snapshot_date": "2026-07-29", "signal": True, "ibd_entry_valid": 1, "ibd_candidate_price": 50.0, "latest_close": 51.0, "ibd_entry_status": "ACTIONABLE"},
            {"code": "CARRY_ACTION", "snapshot_date": "2026-07-29", "signal": False, "ibd_entry_valid": None, "ibd_candidate_price": None, "latest_close": 104.0, "ibd_entry_status": None},
            {"code": "CARRY_EXTENDED", "snapshot_date": "2026-07-29", "signal": False, "ibd_entry_valid": None, "ibd_candidate_price": None, "latest_close": 106.0, "ibd_entry_status": None},
            {"code": "CURRENT_OVERRIDE", "snapshot_date": "2026-07-29", "signal": True, "ibd_entry_valid": 0, "ibd_candidate_price": 100.0, "latest_close": 101.0, "ibd_entry_status": "UNCONFIRMED"},
        ]
    )
    monkeypatch.setattr(yfinance_data, "BREAKOUT_FOLLOW_POOL_PATH", str(complete_path))
    monkeypatch.setattr(yfinance_data, "BREAKOUT_FOLLOW_POOL_MIDWEEK_PATH", str(midweek_path))
    pool_run = yfinance_data.BreakoutFollowPoolRun.midweek()
    pool_run.save_snapshot(current)

    assert pool_run.load_actionable_codes() == ["CARRY_ACTION", "NEW_ACTION"]


def test_pool_run_rejects_unpublished_and_replaced_snapshots(tmp_path, monkeypatch):
    pool_path = tmp_path / "breakout_follow_pool.csv"
    monkeypatch.setattr(yfinance_data, "BREAKOUT_FOLLOW_POOL_PATH", str(pool_path))
    unpublished = yfinance_data.BreakoutFollowPoolRun.weekend()

    with pytest.raises(RuntimeError, match="本轮快照尚未成功写入"):
        unpublished.load_actionable_codes()

    published = yfinance_data.BreakoutFollowPoolRun.weekend()
    published.save_snapshot(pd.DataFrame({"code": ["CURRENT"]}))
    pd.DataFrame({"code": ["STALE"]}).to_csv(pool_path, index=False)

    with pytest.raises(ValueError, match="与本轮快照不一致"):
        published.load_actionable_codes()
    with pytest.raises(ValueError, match="与本轮快照不一致"):
        published.commit()


def test_pool_run_rejects_same_codes_with_changed_snapshot_content(tmp_path, monkeypatch):
    pool_path = tmp_path / "breakout_follow_pool.csv"
    monkeypatch.setattr(yfinance_data, "BREAKOUT_FOLLOW_POOL_PATH", str(pool_path))
    published = yfinance_data.BreakoutFollowPoolRun.weekend()
    published.save_snapshot(
        pd.DataFrame(
            [
                {
                    "code": "SAME",
                    "signal": True,
                    "ibd_entry_valid": 1,
                    "ibd_entry_status": "ACTIONABLE",
                }
            ]
        )
    )
    pd.DataFrame(
        [
            {
                "code": "SAME",
                "signal": False,
                "ibd_entry_valid": 1,
                "ibd_entry_status": "ACTIONABLE",
            }
        ]
    ).to_csv(pool_path, index=False)

    with pytest.raises(ValueError, match="与本轮快照不一致"):
        published.ensure_current_snapshot()


def test_pool_run_rejects_duplicate_codes_before_publishing(tmp_path, monkeypatch):
    pool_path = tmp_path / "breakout_follow_pool.csv"
    monkeypatch.setattr(yfinance_data, "BREAKOUT_FOLLOW_POOL_PATH", str(pool_path))
    pool_run = yfinance_data.BreakoutFollowPoolRun.weekend()

    with pytest.raises(ValueError, match="code 重复"):
        pool_run.save_snapshot(pd.DataFrame({"code": ["DUP", "DUP"]}))

    assert not pool_path.exists()


@pytest.mark.parametrize(
    "baseline_rows",
    [
        None,
        [
            {
                "code": "OLD_CARRY",
                "snapshot_date": "2026-07-17",
                "signal": True,
                "ibd_entry_valid": 1,
                "ibd_candidate_price": 100.0,
                "ibd_entry_status": "ACTIONABLE",
            }
        ],
        [
            {
                "code": "OLD_CARRY",
                "snapshot_date": "2026-08-03",
                "signal": True,
                "ibd_entry_valid": 1,
                "ibd_candidate_price": 100.0,
                "ibd_entry_status": "ACTIONABLE",
            }
        ],
        [
            {
                "code": "OLD_CARRY",
                "snapshot_date": "not-a-date",
                "signal": True,
                "ibd_entry_valid": 1,
                "ibd_candidate_price": 100.0,
                "ibd_entry_status": "ACTIONABLE",
            }
        ],
    ],
    ids=["missing", "stale", "newer", "malformed"],
)
def test_midweek_public_contract_never_carries_from_an_invalid_baseline(
    tmp_path,
    monkeypatch,
    baseline_rows,
):
    complete_path = tmp_path / "breakout_follow_pool.csv"
    midweek_path = tmp_path / "breakout_follow_pool_midweek.csv"
    if baseline_rows is not None:
        pd.DataFrame(baseline_rows).to_csv(complete_path, index=False)
    monkeypatch.setattr(yfinance_data, "BREAKOUT_FOLLOW_POOL_PATH", str(complete_path))
    monkeypatch.setattr(yfinance_data, "BREAKOUT_FOLLOW_POOL_MIDWEEK_PATH", str(midweek_path))
    current = pd.DataFrame(
        [
            {
                "code": "CURRENT",
                "snapshot_date": "2026-07-29",
                "signal": True,
                "ibd_entry_valid": 1,
                "ibd_candidate_price": 50.0,
                "latest_close": 51.0,
                "ibd_entry_status": "ACTIONABLE",
            },
            {
                "code": "OLD_CARRY",
                "snapshot_date": "2026-07-29",
                "signal": False,
                "ibd_entry_valid": None,
                "ibd_candidate_price": None,
                "latest_close": 104.0,
                "ibd_entry_status": None,
            },
        ]
    )
    pool_run = yfinance_data.BreakoutFollowPoolRun.midweek()
    pool_run.save_snapshot(current)

    assert pool_run.load_actionable_codes() == ["CURRENT"]


def test_midweek_public_contract_ignores_an_unreadable_baseline_file(tmp_path, monkeypatch):
    complete_path = tmp_path / "breakout_follow_pool.csv"
    midweek_path = tmp_path / "breakout_follow_pool_midweek.csv"
    complete_path.write_text('code,snapshot_date,signal\n"unterminated', encoding="utf-8")
    monkeypatch.setattr(yfinance_data, "BREAKOUT_FOLLOW_POOL_PATH", str(complete_path))
    monkeypatch.setattr(yfinance_data, "BREAKOUT_FOLLOW_POOL_MIDWEEK_PATH", str(midweek_path))
    pool_run = yfinance_data.BreakoutFollowPoolRun.midweek()
    pool_run.save_snapshot(
        pd.DataFrame(
            [
                {
                    "code": "CURRENT",
                    "snapshot_date": "2026-07-29",
                    "signal": True,
                    "ibd_entry_valid": 1,
                    "ibd_candidate_price": 50.0,
                    "latest_close": 51.0,
                    "ibd_entry_status": "ACTIONABLE",
                }
            ]
        )
    )

    assert pool_run.load_actionable_codes() == ["CURRENT"]


def test_pool_run_fails_closed_when_ibd_enrichment_is_incomplete(tmp_path, monkeypatch):
    pool_path = tmp_path / "breakout_follow_pool.csv"
    monkeypatch.setattr(yfinance_data, "BREAKOUT_FOLLOW_POOL_PATH", str(pool_path))
    pool_run = yfinance_data.BreakoutFollowPoolRun.weekend()
    pool_run.save_snapshot(
        pd.DataFrame(
            [{"code": "CURRENT", "signal": True, "ibd_entry_valid": None, "ibd_entry_status": None}]
        )
    )

    with pytest.raises(ValueError, match="IBD enrichment"):
        pool_run.load_actionable_codes()


def test_pool_run_commit_targets_only_the_run_file(tmp_path, monkeypatch):
    complete_path = tmp_path / "breakout_follow_pool.csv"
    midweek_path = tmp_path / "breakout_follow_pool_midweek.csv"
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(yfinance_data, "BREAKOUT_FOLLOW_POOL_PATH", str(complete_path))
    monkeypatch.setattr(yfinance_data, "BREAKOUT_FOLLOW_POOL_MIDWEEK_PATH", str(midweek_path))
    monkeypatch.setattr(yfinance_data.subprocess, "run", fake_run)
    pool_run = yfinance_data.BreakoutFollowPoolRun.midweek()
    pool_run.save_snapshot(pd.DataFrame({"code": ["MID"]}))

    pool_run.commit()

    diff_paths = [args[-1] for args in calls if args[:3] == ["git", "diff", "--quiet"]]
    assert diff_paths == [str(midweek_path)]
