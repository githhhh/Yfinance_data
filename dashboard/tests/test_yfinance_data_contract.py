from __future__ import annotations

import inspect
from types import SimpleNamespace

import pandas as pd
import pytest

import yfinance_data
from eps_pit.lookup import SignalEPSLookup
from eps_pit.models import EPS_RESOLVER_VERSION


def _passthrough_resolved_eps(pool: pd.DataFrame) -> pd.DataFrame:
    df = pool.copy()
    if "signal" not in df.columns:
        return df
    for column in (
        "eps_yoy_growth",
        "eps_yoy_growth_source",
        "eps_yoy_growth_status",
        "eps_yoy_growth_missing_reason",
    ):
        if column not in df.columns:
            df[column] = pd.NA
    signal_mask = df["signal"].fillna(False).astype(bool)
    missing_eps = signal_mask & df["eps_yoy_growth"].isna()
    df.loc[missing_eps, "eps_yoy_growth"] = 1.0
    df.loc[signal_mask, "eps_yoy_growth_source"] = "PIT"
    df.loc[signal_mask, "eps_yoy_growth_status"] = "resolved"
    df.loc[signal_mask, "eps_yoy_growth_missing_reason"] = pd.NA
    return df


@pytest.fixture(autouse=True)
def _default_expected_unavailable_eps(tmp_path, monkeypatch):
    SignalEPSLookup.clear_cache()
    monkeypatch.setattr(
        SignalEPSLookup,
        "DEFAULT_CSV_PATH",
        str(tmp_path / "default_signal_eps_pit.csv"),
    )
    monkeypatch.setattr(
        SignalEPSLookup,
        "fetch_tradingview_eps",
        staticmethod(lambda codes: {}),
    )
    monkeypatch.setattr(
        SignalEPSLookup,
        "fetch_sec_yahoo_eps",
        staticmethod(
            lambda snapshot, codes, **kwargs: {
                code: {"missing_reason": "NO_PRIOR_YEAR_QUARTER"}
                for code in codes
            }
        ),
    )


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
    monkeypatch.setattr(yfinance_data, "_enrich_signal_eps", _passthrough_resolved_eps)
    pool_path = tmp_path / "breakout_follow_pool.csv"
    monkeypatch.setattr(yfinance_data, "BREAKOUT_FOLLOW_POOL_PATH", str(pool_path))
    snapshot = pd.DataFrame(
        [
            {"code": "FIRST", "snapshot_date": "2026-08-21", "signal": True, "ibd_entry_valid": 1, "ibd_entry_status": "ACTIONABLE"},
            {"code": "WAIT", "snapshot_date": "2026-08-21", "signal": True, "ibd_entry_valid": 0, "ibd_entry_status": "UNCONFIRMED"},
            {"code": "SECOND", "snapshot_date": "2026-08-21", "signal": True, "ibd_entry_valid": 1, "ibd_entry_status": "ACTIONABLE"},
            {"code": "POOL", "snapshot_date": "2026-08-21", "signal": False, "ibd_entry_valid": None, "ibd_entry_status": None},
        ]
    )
    pool_run = yfinance_data.BreakoutFollowPoolRun.weekend()
    pool_run.save_snapshot(snapshot)

    assert pool_run.load_actionable_codes() == ["SECOND", "FIRST"]


def test_pool_run_supplements_signal_eps_before_publishing(tmp_path, monkeypatch):
    pool_path = tmp_path / "breakout_follow_pool.csv"
    pit_path = tmp_path / "signal_eps_pit.csv"

    pd.DataFrame(
        [{
            "snapshot_date": "2026-08-14",
            "code": "PIT",
            "eps_yoy_growth": 31.5,
            "status": "resolved",
            "resolver_version": EPS_RESOLVER_VERSION,
        }]
    ).to_csv(pit_path, index=False)

    SignalEPSLookup.clear_cache()
    monkeypatch.setattr(SignalEPSLookup, "DEFAULT_CSV_PATH", str(pit_path))
    monkeypatch.setattr(
        SignalEPSLookup,
        "fetch_sec_yahoo_eps",
        staticmethod(
            lambda snapshot, codes, **kwargs: {
                "SECY": {
                    "eps_yoy_growth": 42.0,
                    "source": "SEC",
                    "effective_date": "2026-08-01",
                }
            }
        ),
    )
    monkeypatch.setattr(yfinance_data, "BREAKOUT_FOLLOW_POOL_PATH", str(pool_path))

    pool_run = yfinance_data.BreakoutFollowPoolRun.weekend()
    pool_run.save_snapshot(
        pd.DataFrame(
            [
                {"code": "PIT", "snapshot_date": "2026-08-14", "signal": True, "eps_yoy_growth": pd.NA},
                {"code": "SECY", "snapshot_date": "2026-08-14", "signal": True, "eps_yoy_growth": pd.NA},
                {"code": "QUIET", "snapshot_date": "2026-08-14", "signal": False, "eps_yoy_growth": pd.NA},
            ]
        )
    )

    saved = pd.read_csv(pool_path)
    assert saved.loc[saved["code"].eq("PIT"), "eps_yoy_growth"].item() == 31.5
    assert saved.loc[saved["code"].eq("PIT"), "eps_yoy_growth_source"].item() == "PIT"
    assert saved.loc[saved["code"].eq("SECY"), "eps_yoy_growth"].item() == 42.0
    assert saved.loc[saved["code"].eq("SECY"), "eps_yoy_growth_source"].item() == "SEC"
    assert pd.isna(saved.loc[saved["code"].eq("QUIET"), "eps_yoy_growth"].item())
    assert "eps_yoy_growth_repair_method" not in saved.columns
    assert "eps_yoy_growth_effective_date" not in saved.columns
    assert "eps_yoy_growth_current_eps" not in saved.columns
    assert "eps_yoy_growth_prior_year_eps" not in saved.columns


def test_pool_run_logs_unresolved_signal_eps_codes(tmp_path, monkeypatch, caplog):
    pool_path = tmp_path / "breakout_follow_pool.csv"

    SignalEPSLookup.clear_cache()
    monkeypatch.setattr(SignalEPSLookup, "DEFAULT_CSV_PATH", str(tmp_path / "missing_pit.csv"))
    monkeypatch.setattr(
        SignalEPSLookup,
        "fetch_sec_yahoo_eps",
        staticmethod(
            lambda snapshot, codes, **kwargs: {
                "FILLED": {
                    "eps_yoy_growth": 42.0,
                    "source": "Yahoo",
                    "effective_date": "2026-08-01",
                }
            }
        ),
    )
    monkeypatch.setattr(yfinance_data, "BREAKOUT_FOLLOW_POOL_PATH", str(pool_path))

    pool_run = yfinance_data.BreakoutFollowPoolRun.weekend()
    with caplog.at_level("WARNING"):
        pool_run.save_snapshot(
            pd.DataFrame(
                [
                    {"code": "FILLED", "snapshot_date": "2026-08-14", "signal": True, "eps_yoy_growth": pd.NA},
                    {"code": "MISS", "snapshot_date": "2026-08-14", "signal": True, "eps_yoy_growth": pd.NA},
                    {"code": "QUIET", "snapshot_date": "2026-08-14", "signal": False, "eps_yoy_growth": pd.NA},
                ]
            )
        )

    saved = pd.read_csv(pool_path)
    assert saved.loc[saved["code"].eq("FILLED"), "eps_yoy_growth"].item() == 42.0
    assert pd.isna(saved.loc[saved["code"].eq("MISS"), "eps_yoy_growth"].item())
    assert "BF Pool signal EPS unresolved codes: MISS" in caplog.text
    assert "QUIET" not in caplog.text


def test_pool_run_logs_signal_eps_pit_summary(tmp_path, monkeypatch, caplog):
    pool_path = tmp_path / "breakout_follow_pool.csv"

    SignalEPSLookup.clear_cache()
    monkeypatch.setattr(SignalEPSLookup, "DEFAULT_CSV_PATH", str(tmp_path / "missing_pit.csv"))
    monkeypatch.setattr(
        SignalEPSLookup,
        "fetch_sec_yahoo_eps",
        staticmethod(
            lambda snapshot, codes, **kwargs: {
                "FILLED": {
                    "eps_yoy_growth": 42.0,
                    "source": "Yahoo",
                    "effective_date": "2026-08-01",
                },
                "MISS": {
                    "missing_reason": "NO_PRIOR_YEAR_QUARTER",
                },
            }
        ),
    )
    monkeypatch.setattr(yfinance_data, "BREAKOUT_FOLLOW_POOL_PATH", str(pool_path))

    pool_run = yfinance_data.BreakoutFollowPoolRun.weekend()
    with caplog.at_level("INFO"):
        pool_run.save_snapshot(
            pd.DataFrame(
                [
                    {"code": "FILLED", "snapshot_date": "2026-08-14", "signal": True, "eps_yoy_growth": pd.NA},
                    {"code": "MISS", "snapshot_date": "2026-08-14", "signal": True, "eps_yoy_growth": pd.NA},
                    {"code": "QUIET", "snapshot_date": "2026-08-14", "signal": False, "eps_yoy_growth": pd.NA},
                ]
            )
        )

    assert "BF Pool signal EPS PIT summary:" in caplog.text
    assert "resolved=1 [FILLED]" in caplog.text
    assert "expected_unavailable=1 [MISS(NO_PRIOR_YEAR_QUARTER)]" in caplog.text
    assert "provider_error=0 [none]" in caplog.text
    assert "QUIET" not in caplog.text


def test_supplement_latest_pool_eps_updates_only_latest_snapshot_pool(tmp_path, monkeypatch):
    complete_path = tmp_path / "breakout_follow_pool.csv"
    midweek_path = tmp_path / "breakout_follow_pool_midweek.csv"
    complete = pd.DataFrame(
        [
            {"code": "COMPLETE", "snapshot_date": "2026-08-21", "signal": True, "eps_yoy_growth": pd.NA},
        ]
    )
    midweek = pd.DataFrame(
        [
            {"code": "MIDWEEK", "snapshot_date": "2026-08-19", "signal": True, "eps_yoy_growth": pd.NA},
        ]
    )
    complete.to_csv(complete_path, index=False)
    midweek.to_csv(midweek_path, index=False)

    monkeypatch.setattr(yfinance_data, "BREAKOUT_FOLLOW_POOL_PATH", str(complete_path))
    monkeypatch.setattr(yfinance_data, "BREAKOUT_FOLLOW_POOL_MIDWEEK_PATH", str(midweek_path))
    monkeypatch.setattr(SignalEPSLookup, "DEFAULT_CSV_PATH", str(tmp_path / "missing_pit.csv"))
    monkeypatch.setattr(
        SignalEPSLookup,
        "fetch_sec_yahoo_eps",
        staticmethod(
            lambda snapshot, codes, **kwargs: {
                "COMPLETE": {
                    "eps_yoy_growth": 31.0,
                    "source": "SEC",
                    "effective_date": snapshot,
                },
                "MIDWEEK": {
                    "eps_yoy_growth": 99.0,
                    "source": "Yahoo",
                    "effective_date": snapshot,
                },
            }
        ),
    )

    result = yfinance_data.supplement_latest_pool_signal_eps()

    saved_complete = pd.read_csv(complete_path)
    saved_midweek = pd.read_csv(midweek_path)
    assert result["path"] == str(complete_path)
    assert result["snapshot_date"] == "2026-08-21"
    assert result["repaired"] == 1
    assert saved_complete.loc[0, "eps_yoy_growth"] == 31.0
    assert pd.isna(saved_midweek.loc[0, "eps_yoy_growth"])


def test_supplement_latest_pool_eps_uses_midweek_when_it_has_newer_snapshot(tmp_path, monkeypatch):
    complete_path = tmp_path / "breakout_follow_pool.csv"
    midweek_path = tmp_path / "breakout_follow_pool_midweek.csv"
    pd.DataFrame(
        [{"code": "COMPLETE", "snapshot_date": "2026-08-21", "signal": True, "eps_yoy_growth": pd.NA}]
    ).to_csv(complete_path, index=False)
    pd.DataFrame(
        [{"code": "MIDWEEK", "snapshot_date": "2026-08-24", "signal": True, "eps_yoy_growth": pd.NA}]
    ).to_csv(midweek_path, index=False)

    monkeypatch.setattr(yfinance_data, "BREAKOUT_FOLLOW_POOL_PATH", str(complete_path))
    monkeypatch.setattr(yfinance_data, "BREAKOUT_FOLLOW_POOL_MIDWEEK_PATH", str(midweek_path))
    monkeypatch.setattr(SignalEPSLookup, "DEFAULT_CSV_PATH", str(tmp_path / "missing_pit.csv"))
    monkeypatch.setattr(
        SignalEPSLookup,
        "fetch_sec_yahoo_eps",
        staticmethod(
            lambda snapshot, codes, **kwargs: {
                "COMPLETE": {
                    "eps_yoy_growth": 31.0,
                    "source": "SEC",
                    "effective_date": snapshot,
                },
                "MIDWEEK": {
                    "eps_yoy_growth": 99.0,
                    "source": "Yahoo",
                    "effective_date": snapshot,
                },
            }
        ),
    )

    result = yfinance_data.supplement_latest_pool_signal_eps()

    saved_complete = pd.read_csv(complete_path)
    saved_midweek = pd.read_csv(midweek_path)
    assert result["path"] == str(midweek_path)
    assert result["snapshot_date"] == "2026-08-24"
    assert result["repaired"] == 1
    assert pd.isna(saved_complete.loc[0, "eps_yoy_growth"])
    assert saved_midweek.loc[0, "eps_yoy_growth"] == 99.0


def test_midweek_pool_run_uses_unified_projection_and_matches_quant_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(yfinance_data, "_enrich_signal_eps", _passthrough_resolved_eps)
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
    monkeypatch.setattr(yfinance_data, "_enrich_signal_eps", _passthrough_resolved_eps)
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
    monkeypatch.setattr(yfinance_data, "_enrich_signal_eps", _passthrough_resolved_eps)
    pool_path = tmp_path / "breakout_follow_pool.csv"
    monkeypatch.setattr(yfinance_data, "BREAKOUT_FOLLOW_POOL_PATH", str(pool_path))
    published = yfinance_data.BreakoutFollowPoolRun.weekend()
    published.save_snapshot(
        pd.DataFrame(
            [
                {
                    "code": "SAME",
                    "snapshot_date": "2026-08-21",
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
                "snapshot_date": "2026-08-21",
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
    monkeypatch.setattr(yfinance_data, "_enrich_signal_eps", _passthrough_resolved_eps)
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
    monkeypatch.setattr(yfinance_data, "_enrich_signal_eps", _passthrough_resolved_eps)
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
    monkeypatch.setattr(yfinance_data, "_enrich_signal_eps", _passthrough_resolved_eps)
    pool_path = tmp_path / "breakout_follow_pool.csv"
    monkeypatch.setattr(yfinance_data, "BREAKOUT_FOLLOW_POOL_PATH", str(pool_path))
    pool_run = yfinance_data.BreakoutFollowPoolRun.weekend()
    pool_run.save_snapshot(
        pd.DataFrame(
            [{"code": "CURRENT", "snapshot_date": "2026-08-21", "signal": True, "ibd_entry_valid": None, "ibd_entry_status": None}]
        )
    )

    with pytest.raises(ValueError, match="IBD enrichment"):
        pool_run.load_actionable_codes()


def test_pool_run_commit_targets_only_the_run_file(tmp_path, monkeypatch):
    monkeypatch.setattr(yfinance_data, "_enrich_signal_eps", _passthrough_resolved_eps)
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

    diff_calls = [
        args
        for args in calls
        if args[:4] == ["git", "diff", "--cached", "--quiet"]
    ]
    assert len(diff_calls) == 1
    assert str(midweek_path) in diff_calls[0]
