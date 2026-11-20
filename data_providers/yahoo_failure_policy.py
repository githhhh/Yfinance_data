"""Pure, fail-closed classification of Yahoo download exceptions.

No network calls, retries, global state, or knowledge of Pool/Watch.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from yfinance.exceptions import YFException


class YahooFailureAction(str, Enum):
    RETRY = "retry"
    QUARANTINE = "quarantine"


@dataclass(frozen=True)
class YahooFailureDecision:
    action: YahooFailureAction
    reason_code: str = ""


# Match the known yfinance safe_merge_dfs exception, not a generic substring.
_DIVIDEND_OUT_OF_RANGE = re.compile(
    r"\AThe following 'Dividends' events are out-of-range, "
    r"did not expect with interval (1d|1wk): DatetimeIndex\("
)


def classify_yahoo_exception(
    exception: Exception, *, interval: str
) -> YahooFailureDecision:
    """Quarantine only a known action-merge failure, never a transport error.

    The policy does not claim the OHLCV is valid; the symbol is omitted.
    Unknown exceptions remain subject to the provider's existing retry/fail gate.
    """
    if isinstance(exception, YFException) and interval in {"1d", "1wk"}:
        match = _DIVIDEND_OUT_OF_RANGE.match(str(exception))
        if match and match.group(1) == interval:
            return YahooFailureDecision(
                YahooFailureAction.QUARANTINE, "DIVIDEND_EVENT_OUT_OF_RANGE"
            )
    return YahooFailureDecision(YahooFailureAction.RETRY)
