from pathlib import Path

import pandas as pd

from backtest.latest_quant_trade_replay import (
    ReplayPoolSink,
    audit_pool_schema,
    apply_replay_strategy_env,
    clip_price_data_asof,
    enumerate_complete_snapshot_weeks,
)


def _frame(dates):
    return pd.DataFrame(
        {
            "Open": [10.0 + i for i, _ in enumerate(dates)],
            "High": [11.0 + i for i, _ in enumerate(dates)],
            "Low": [9.0 + i for i, _ in enumerate(dates)],
            "Close": [10.5 + i for i, _ in enumerate(dates)],
            "Volume": [1000 + i for i, _ in enumerate(dates)],
        },
        index=pd.to_datetime(dates),
    )


def test_enumerates_complete_weeks_before_latest_production_week():
    weeks = enumerate_complete_snapshot_weeks(
        start_date="2026-01-01",
        exclude_week_ending="2026-08-14",
    )

    assert weeks[0].snapshot_date == "2026-01-02"
    assert weeks[-1].snapshot_date == "2026-08-07"
    assert "2026-08-14" not in [week.snapshot_date for week in weeks]


def test_clips_price_data_before_quant_trade_receives_it():
    raw = {
        "AAA": _frame(["2026-08-06", "2026-08-07", "2026-08-10"]),
        "BBB": _frame(["2026-08-05", "2026-08-07"]),
    }

    result = clip_price_data_asof(raw, "2026-08-07")

    assert result.has_future_data_before_clip is True
    assert result.max_date_before_clip == "2026-08-10"
    assert result.max_date_after_clip == "2026-08-07"
    assert list(result.data["AAA"].index.strftime("%Y-%m-%d")) == ["2026-08-06", "2026-08-07"]


def test_replay_pool_sink_writes_only_to_replay_directory(tmp_path):
    sink = ReplayPoolSink(tmp_path / "2026-08-07" / "breakout_follow_pool.csv")
    pool = pd.DataFrame({"code": ["IMAX"], "signal": [True], "latest_close": [28.0]})

    sink.save_snapshot(pool)

    assert sink.path == str(tmp_path / "2026-08-07" / "breakout_follow_pool.csv")
    assert Path(sink.path).exists()
    assert pd.read_csv(sink.path)["code"].tolist() == ["IMAX"]


def test_replay_pool_sink_rejects_publish_and_commit_side_effects(tmp_path):
    sink = ReplayPoolSink(tmp_path / "pool.csv")
    sink.save_snapshot(pd.DataFrame({"code": ["IMAX"]}))

    for method_name in ("publish", "commit", "load_actionable_codes"):
        try:
            getattr(sink, method_name)()
        except RuntimeError as exc:
            assert "disabled for replay" in str(exc)
        else:
            raise AssertionError(f"{method_name} should be disabled")


def test_schema_audit_repairs_industry_but_fails_missing_core_price():
    pool = pd.DataFrame(
        {
            "code": ["IMAX", "AAPL"],
            "snapshot_date": ["2026-08-07", "2026-08-07"],
            "signal": [True, False],
            "signal_source": ["ceiling_breakout", "pivot"],
            "latest_close": [28.0, pd.NA],
            "volume_ratio": [2.1, 1.2],
            "ceiling": [26.0, 200.0],
        }
    )

    audit = audit_pool_schema(pool)

    assert audit.schema_validation_status == "failed_critical_schema"
    assert "latest_close" in audit.missing_critical_fields
    assert audit.repaired_fields == ["industry", "sector"]
    assert "industry" in audit.missing_repairable_fields


def test_replay_strategy_env_sets_only_non_side_effect_parameters(monkeypatch, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "STRATEGY_RECENT_N=5",
                "BOX_PERIOD_L=12",
                "BOX_PERIOD_M=6",
                "BOX_PERIOD_S=3",
                "LAST_PERIOD_L_BREAKOUT_RESISTANCE_COUNT=2",
                "LAST_PERIOD_M_BREAKOUT_RESISTANCE_COUNT=3",
                "STRATEGY_TYPE=1",
                "TELEGRAM_BOT_TOKEN=secret",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "secret")
    applied = apply_replay_strategy_env(env_path)

    assert applied["STRATEGY_RECENT_N"] == "5"
    assert applied["BOX_PERIOD_L"] == "12"
    assert applied["STRATEGY_TYPE"] == "1"
    assert "TELEGRAM_BOT_TOKEN" not in applied


def test_schema_audit_allows_blank_signal_source_on_non_signal_rows():
    pool = pd.DataFrame(
        {
            "code": ["IMAX", "AAPL"],
            "snapshot_date": ["2026-08-07", "2026-08-07"],
            "signal": [True, False],
            "signal_source": ["ceiling_breakout", pd.NA],
            "latest_close": [28.0, 210.0],
            "volume_ratio": [2.1, 1.2],
            "ceiling": [26.0, 200.0],
            "industry": ["Entertainment", "Hardware"],
            "sector": ["Consumer Services", "Technology"],
        }
    )

    audit = audit_pool_schema(pool)

    assert audit.schema_validation_status == "passed_with_repairs_or_optional_gaps"
    assert "signal_source" not in audit.missing_critical_fields
