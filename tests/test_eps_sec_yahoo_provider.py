import pandas as pd

from eps_pit.providers import sec_yahoo_provider
from eps_pit.providers.sec_yahoo_provider import YahooFundamentalsProvider, calculate_latest_eps_yoy


def test_calculate_latest_eps_yoy_uses_sec_same_quarter_before_snapshot():
    records = [
        {
            "code": "TEST",
            "fiscal_year": 2024,
            "fiscal_quarter": "Q2",
            "report_period": "2024-06-30",
            "eps_diluted": 0.80,
            "filing_date": "2024-08-01",
            "source": "SEC",
        },
        {
            "code": "TEST",
            "fiscal_year": 2025,
            "fiscal_quarter": "Q1",
            "report_period": "2025-03-31",
            "eps_diluted": 0.90,
            "filing_date": "2025-05-01",
            "source": "SEC",
        },
        {
            "code": "TEST",
            "fiscal_year": 2025,
            "fiscal_quarter": "Q2",
            "report_period": "2025-06-30",
            "eps_diluted": 1.00,
            "filing_date": "2025-08-01",
            "source": "SEC",
        },
        {
            "code": "TEST",
            "fiscal_year": 2026,
            "fiscal_quarter": "Q1",
            "report_period": "2026-03-31",
            "eps_diluted": 1.20,
            "filing_date": "2026-05-01",
            "source": "SEC",
        },
        {
            "code": "TEST",
            "fiscal_year": 2026,
            "fiscal_quarter": "Q2",
            "report_period": "2026-06-30",
            "eps_diluted": 1.50,
            "filing_date": "2026-08-01",
            "source": "SEC",
        },
    ]

    result = calculate_latest_eps_yoy(records, "2026-08-21")

    assert result["eps_yoy_growth"] == 50.0
    assert result["source"] == "SEC"
    assert result["effective_date"] == "2026-08-01"
    assert result["current_eps"] == 1.50
    assert result["prior_year_eps"] == 1.00


def test_calculate_latest_eps_yoy_prefers_report_period_prior_year_over_fiscal_year_label():
    records = [
        {
            "code": "DK",
            "fiscal_year": 2025,
            "fiscal_quarter": "Q2",
            "report_period": "2024-06-30",
            "eps_diluted": -0.58,
            "filing_date": "2025-08-06",
            "source": "SEC",
        },
        {
            "code": "DK",
            "fiscal_year": 2026,
            "fiscal_quarter": "Q2",
            "report_period": "2025-06-30",
            "eps_diluted": -1.76,
            "filing_date": "2026-08-05",
            "source": "SEC",
        },
        {
            "code": "DK",
            "fiscal_year": 2025,
            "fiscal_quarter": "Q3",
            "report_period": "2025-09-30",
            "eps_diluted": 2.93,
            "filing_date": "2025-11-07",
            "source": "SEC",
        },
        {
            "code": "DK",
            "fiscal_year": 2026,
            "fiscal_quarter": "Q1",
            "report_period": "2026-03-31",
            "eps_diluted": -3.34,
            "filing_date": "2026-04-29",
            "source": "SEC",
        },
        {
            "code": "DK",
            "fiscal_year": 2026,
            "fiscal_quarter": "Q2",
            "report_period": "2026-06-30",
            "eps_diluted": 2.71,
            "filing_date": "2026-08-05",
            "source": "SEC",
        },
    ]

    result = calculate_latest_eps_yoy(records, "2026-08-21")

    assert round(result["eps_yoy_growth"], 2) == 253.98
    assert result["prior_year_eps"] == -1.76
    assert result["prior_year_period"] == "2025-06-30"


def test_calculate_latest_eps_yoy_requires_enough_quarter_history():
    records = [
        {
            "code": "SSMR",
            "fiscal_year": 2026,
            "fiscal_quarter": "Q2",
            "report_period": "2025-06-30",
            "eps_diluted": -0.08,
            "filing_date": "2026-08-12",
            "source": "SEC",
        },
        {
            "code": "SSMR",
            "fiscal_year": 2026,
            "fiscal_quarter": "Q2",
            "report_period": "2026-06-30",
            "eps_diluted": -0.13,
            "filing_date": "2026-08-12",
            "source": "SEC",
        },
    ]

    assert calculate_latest_eps_yoy(records, "2026-08-21") is None


def test_calculate_latest_eps_yoy_does_not_use_future_records():
    records = [
        {
            "code": "TEST",
            "fiscal_year": 2025,
            "fiscal_quarter": "Q2",
            "report_period": "2025-06-30",
            "eps_diluted": 1.00,
            "filing_date": "2025-08-01",
            "source": "SEC",
        },
        {
            "code": "TEST",
            "fiscal_year": 2026,
            "fiscal_quarter": "Q2",
            "report_period": "2026-06-30",
            "eps_diluted": 1.50,
            "filing_date": "2026-08-01",
            "source": "SEC",
        },
    ]

    assert calculate_latest_eps_yoy(records, "2026-07-31") is None


def test_calculate_latest_eps_yoy_can_use_yahoo_chronological_quarters():
    records = [
        {"code": "ADR", "report_period": "2025-05-01", "eps_diluted": 0.50, "earnings_release_at": "2025-05-01", "source": "Yahoo"},
        {"code": "ADR", "report_period": "2025-08-01", "eps_diluted": 0.60, "earnings_release_at": "2025-08-01", "source": "Yahoo"},
        {"code": "ADR", "report_period": "2025-11-01", "eps_diluted": 0.70, "earnings_release_at": "2025-11-01", "source": "Yahoo"},
        {"code": "ADR", "report_period": "2026-02-01", "eps_diluted": 0.80, "earnings_release_at": "2026-02-01", "source": "Yahoo"},
        {"code": "ADR", "report_period": "2026-05-01", "eps_diluted": 1.00, "earnings_release_at": "2026-05-01", "source": "Yahoo"},
    ]

    result = calculate_latest_eps_yoy(records, "2026-08-21")

    assert result["eps_yoy_growth"] == 100.0
    assert result["source"] == "Yahoo"
    assert result["effective_date"] == "2026-05-01"


def test_yahoo_provider_does_not_use_event_reported_eps_when_income_statement_is_stale(
    monkeypatch,
    tmp_path,
):
    class FakeTicker:
        def get_earnings_dates(self, limit=32):
            index = pd.to_datetime(["2026-08-13", "2026-05-07", "2025-08-07", "2025-05-01"])
            return pd.DataFrame(
                {"Reported EPS": [0.90, 0.27, -0.64, -1.58]},
                index=index,
            )

        @property
        def quarterly_income_stmt(self):
            return pd.DataFrame(
                {
                    pd.Timestamp("2026-03-31"): [9.75],
                    pd.Timestamp("2025-03-31"): [-1.58],
                },
                index=["Diluted EPS"],
            )

    monkeypatch.setattr(sec_yahoo_provider.yf, "Ticker", lambda symbol: FakeTicker())

    records = YahooFundamentalsProvider(tmp_path).fetch_quarterly_history("ASND")

    assert records == []
