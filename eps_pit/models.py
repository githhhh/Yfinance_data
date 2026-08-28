from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


EPS_RESOLVER_VERSION = "eps_pit_v3"


class EPSResolveMode(Enum):
    LIVE = "live"
    REPLAY = "replay"


class EPSStatus(Enum):
    RESOLVED = "resolved"
    EXPECTED_UNAVAILABLE = "expected_unavailable"
    PROVIDER_ERROR = "provider_error"
    NOT_ATTEMPTED = "not_attempted"


class EPSMissingReason(Enum):
    NO_PRIOR_YEAR_QUARTER = "NO_PRIOR_YEAR_QUARTER"
    PRIOR_YEAR_EPS_ZERO = "PRIOR_YEAR_EPS_ZERO"
    NO_QUARTERLY_EPS = "NO_QUARTERLY_EPS"
    NO_VERIFIED_YAHOO_RELEASE_DATE = "NO_VERIFIED_YAHOO_RELEASE_DATE"
    TV_NOT_FOUND = "TV_NOT_FOUND"
    TV_FIELD_NULL = "TV_FIELD_NULL"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    REFRESH_DISABLED = "REFRESH_DISABLED"


@dataclass(frozen=True)
class EPSResult:
    code: str
    snapshot_date: str
    status: EPSStatus
    eps_yoy_growth: float | None = None
    source: str | None = None
    effective_date: str | None = None
    current_eps: float | None = None
    prior_year_eps: float | None = None
    current_period: str | None = None
    prior_year_period: str | None = None
    calculation_method: str | None = None
    missing_reason: EPSMissingReason | None = None
    sec_cik: str | None = None
    source_record_id: str | None = None
    resolver_version: str = EPS_RESOLVER_VERSION

    @property
    def is_resolved(self) -> bool:
        return self.status is EPSStatus.RESOLVED and self.eps_yoy_growth is not None

    def to_record(self) -> dict[str, Any]:
        return {
            "snapshot_date": self.snapshot_date,
            "code": self.code,
            "eps_yoy_growth": self.eps_yoy_growth,
            "source": self.source,
            "effective_date": self.effective_date,
            "current_eps": self.current_eps,
            "prior_year_eps": self.prior_year_eps,
            "current_period": self.current_period,
            "prior_year_period": self.prior_year_period,
            "calculation_method": self.calculation_method,
            "status": self.status.value,
            "missing_reason": self.missing_reason.value if self.missing_reason else None,
            "sec_cik": self.sec_cik,
            "source_record_id": self.source_record_id,
            "resolver_version": self.resolver_version,
        }
