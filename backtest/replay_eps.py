"""Research-only EPS accessors that force historical REPLAY semantics.

Production code keeps its normal LIVE defaults. Historical/backtest callers use
this module so a missing EPS value can never fall back to the LIVE PIT store.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import eps_pit.lookup as eps_lookup
from eps_pit import EPSResolveMode, resolve_signal_eps


def get_replay_signal_eps(
    snapshot_date: object,
    code: object,
    csv_path: str | None = None,
    *,
    allow_network: bool = False,
) -> float | None:
    """Resolve/read one EPS value explicitly in REPLAY mode."""
    result = resolve_signal_eps(
        snapshot_date,
        code,
        mode=EPSResolveMode.REPLAY,
        csv_path=csv_path,
        allow_network=allow_network,
    )
    return result.eps_yoy_growth if result.is_resolved else None


@contextmanager
def replay_signal_eps_lookup(
    *,
    allow_network: bool = False,
    csv_path: str | None = None,
) -> Iterator[None]:
    """Temporarily route legacy get_signal_eps() calls through REPLAY mode.

    Some frozen production-selector functions import get_signal_eps lazily.
    Backtests keep those production files byte-for-byte unchanged and bind the
    lookup only for the duration of the historical selection call.
    """

    original = eps_lookup.get_signal_eps
    default_csv_path = csv_path

    def _replay_get(
        snapshot_date: object,
        code: object,
        csv_path: str | None = None,
    ) -> float | None:
        return get_replay_signal_eps(
            snapshot_date,
            code,
            csv_path=csv_path or default_csv_path,
            allow_network=allow_network,
        )

    eps_lookup.get_signal_eps = _replay_get
    try:
        yield
    finally:
        eps_lookup.get_signal_eps = original
