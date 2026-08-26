from __future__ import annotations

from typing import Any

import pandas as pd

from eps_pit.models import EPSMissingReason
from eps_pit.providers.pit_provider import normalize_symbol, safe_float

EPS_FIELD = "earnings_per_share_diluted_yoy_growth_fq"


class TradingViewEPSProvider:
    """Current-state TradingView EPS provider. Never use for historical replay."""

    def fetch_eps_yoy(self, codes: list[str]) -> dict[str, dict[str, Any]]:
        symbols = sorted({normalize_symbol(code) for code in codes if normalize_symbol(code)})
        if not symbols:
            return {}

        from tradingview_screener import Query, col

        _, frame = (
            Query()
            .select("name", "exchange", EPS_FIELD)
            .where(
                col("exchange").isin(["AMEX", "CBOE", "NASDAQ", "NYSE"]),
                col("active_symbol") == True,
                col("name").isin(symbols),
            )
            .limit(max(50, len(symbols) * 4))
            .set_markets("america")
            .get_scanner_data()
        )

        outcomes: dict[str, dict[str, Any]] = {
            symbol: {"missing_reason": EPSMissingReason.TV_NOT_FOUND}
            for symbol in symbols
        }
        if frame is None or frame.empty:
            return outcomes

        name_col = "name" if "name" in frame.columns else "ticker"
        if name_col not in frame.columns or EPS_FIELD not in frame.columns:
            raise RuntimeError("TradingView EPS response schema changed")

        rows_by_code: dict[str, list[pd.Series]] = {}
        for _, row in frame.iterrows():
            code = normalize_symbol(row.get(name_col))
            if code in outcomes:
                rows_by_code.setdefault(code, []).append(row)

        for code in symbols:
            rows = rows_by_code.get(code, [])
            if not rows:
                continue
            # Bare symbols can theoretically collide across exchanges. Never
            # silently choose one if the provider returns multiple identities.
            identities = {
                (normalize_symbol(row.get(name_col)), str(row.get("exchange") or ""))
                for row in rows
            }
            if len(identities) > 1:
                outcomes[code] = {"missing_reason": EPSMissingReason.PROVIDER_ERROR}
                continue

            value = safe_float(rows[0].get(EPS_FIELD))
            if value is None:
                outcomes[code] = {"missing_reason": EPSMissingReason.TV_FIELD_NULL}
                continue
            outcomes[code] = {
                "eps_yoy_growth": value,
                "source": "TV_DIRECT",
                "calculation_method": "provider_reported_yoy",
            }
        return outcomes
