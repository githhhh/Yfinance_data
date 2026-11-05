from pathlib import Path

import pandas as pd
import pytest

import DataStore


def _write_codes(root: Path, filename: str, codes: list[str]) -> None:
    pd.DataFrame({"code": codes}).to_csv(root / filename, index=False)


def test_read_stock_list_preserves_the_named_business_input_union(tmp_path):
    stock_list_dir = tmp_path / "us"
    stock_list_dir.mkdir()
    _write_codes(stock_list_dir, "52wk_new_high_results.csv", ["HIGH", "DOT.NAME"])
    _write_codes(stock_list_dir, "breakout_follow_pool.csv", ["POOL", "HIGH"])
    _write_codes(stock_list_dir, "breakout_follow_pool_midweek.csv", ["MID"])
    _write_codes(stock_list_dir, "eps_growth_screener_results.csv", ["EPS"])
    _write_codes(stock_list_dir, "weekly_vol_screener_results.csv", ["VOL"])

    assert set(DataStore.read_stock_list(str(stock_list_dir))) == {
        "DOT-NAME",
        "EPS",
        "HIGH",
        "MID",
        "POOL",
        "VOL",
    }


def test_manual_download_exclusions_preserve_default_behavior():
    tickers = ["^GSPC", "^IXIC", "^DJI", "LXP", "AAPL"]
    assert DataStore.apply_manual_download_exclusions(tickers, "") is tickers
    assert DataStore.apply_manual_download_exclusions(tickers, "  ") is tickers


def test_manual_download_exclusions_apply_after_all_symbol_additions(capsys):
    # The final list already contains Watch-restored codes and required indices.
    tickers = ["^GSPC", "^IXIC", "^DJI", "LXP", "WATCH", "BRK-B", "AAPL"]
    result = DataStore.apply_manual_download_exclusions(tickers, " lxp, brk.b, WATCH, LXP ")
    assert result == ["^GSPC", "^IXIC", "^DJI", "AAPL"]
    assert tickers == ["^GSPC", "^IXIC", "^DJI", "LXP", "WATCH", "BRK-B", "AAPL"]
    assert "Explicitly skipped 3" in capsys.readouterr().out


def test_manual_download_exclusions_ignore_symbols_absent_from_one_period(capsys):
    tickers = ["^GSPC", "^IXIC", "^DJI", "AAPL"]
    assert DataStore.apply_manual_download_exclusions(tickers, "LXP") == tickers
    assert "Not in this run's universe: ['LXP']" in capsys.readouterr().out


@pytest.mark.parametrize("symbol", ["^GSPC", "^IXIC", "^DJI"])
def test_manual_download_exclusions_protect_market_indices(symbol):
    with pytest.raises(ValueError, match="Cannot exclude required market indices"):
        DataStore.apply_manual_download_exclusions(
            ["^GSPC", "^IXIC", "^DJI", "LXP"], symbol
        )


def test_manual_download_exclusions_cannot_remove_every_equity():
    with pytest.raises(ValueError, match="Cannot exclude all equity tickers"):
        DataStore.apply_manual_download_exclusions(
            ["^GSPC", "^IXIC", "^DJI", "LXP"], "LXP"
        )


@pytest.mark.parametrize("workflow,download_steps", [
    ("data-update-1d.yml", 1),
    ("data-update-1wk.yml", 1),
    ("data-update.yml", 2),
])
def test_manual_workflows_forward_optional_exclusions(workflow, download_steps):
    path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / workflow
    content = path.read_text(encoding="utf-8")
    assert "workflow_dispatch:" in content
    assert "      exclude_tickers:" in content
    assert "  schedule:" not in content
    assert content.count("EXCLUDE_TICKERS: ${{ inputs.exclude_tickers || '' }}") == download_steps
    assert content.count('"${EXCLUDE_ARGS[@]}"') == download_steps

