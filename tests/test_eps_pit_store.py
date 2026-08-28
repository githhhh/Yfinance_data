import pandas as pd
import pytest

from eps_pit import EPSMissingReason, EPSResult, EPSStatus
from eps_pit.models import EPS_RESOLVER_VERSION
from eps_pit.store import EPSPITStore, EPSPITStoreError


def test_store_persists_resolved_only(tmp_path):
    path = tmp_path / "signal_eps_pit.csv"
    store = EPSPITStore(str(path))

    store.upsert(
        EPSResult(
            code="MISS",
            snapshot_date="2026-08-21",
            status=EPSStatus.EXPECTED_UNAVAILABLE,
            missing_reason=EPSMissingReason.NO_QUARTERLY_EPS,
        )
    )
    assert not path.exists()

    store.upsert(
        EPSResult(
            code="ABC",
            snapshot_date="2026-08-21",
            status=EPSStatus.RESOLVED,
            eps_yoy_growth=25.0,
            source="SEC",
            effective_date="2026-08-01",
        )
    )
    stored = store.get("2026-08-21", "ABC")
    assert stored is not None
    assert stored.eps_yoy_growth == 25.0


def test_store_rejects_future_effective_date_on_write(tmp_path):
    store = EPSPITStore(str(tmp_path / "signal_eps_pit.csv"))
    with pytest.raises(EPSPITStoreError, match="effective_date exceeds snapshot_date"):
        store.upsert(
            EPSResult(
                code="ABC",
                snapshot_date="2026-08-21",
                status=EPSStatus.RESOLVED,
                eps_yoy_growth=25.0,
                source="SEC",
                effective_date="2026-09-01",
            )
        )


def test_store_rejects_future_effective_date_on_read(tmp_path):
    path = tmp_path / "signal_eps_pit.csv"
    pd.DataFrame(
        [
            {
                "snapshot_date": "2026-08-21",
                "code": "ABC",
                "eps_yoy_growth": 25.0,
                "status": "resolved",
                "effective_date": "2026-09-01",
                "resolver_version": EPS_RESOLVER_VERSION,
            }
        ]
    ).to_csv(path, index=False)

    with pytest.raises(EPSPITStoreError, match="leaks future data"):
        EPSPITStore(str(path)).get("2026-08-21", "ABC")


def test_store_corruption_is_not_silently_overwritten(tmp_path):
    path = tmp_path / "signal_eps_pit.csv"
    path.write_text("garbage\n1\n")
    original = path.read_text()

    with pytest.raises(EPSPITStoreError):
        EPSPITStore(str(path)).upsert(
            EPSResult(
                code="ABC",
                snapshot_date="2026-08-21",
                status=EPSStatus.RESOLVED,
                eps_yoy_growth=25.0,
            )
        )

    assert path.read_text() == original


def test_store_rejects_duplicate_logical_keys(tmp_path):
    path = tmp_path / "signal_eps_pit.csv"
    pd.DataFrame(
        [
            {"snapshot_date": "2026-08-21", "code": "ABC", "eps_yoy_growth": 10.0},
            {"snapshot_date": "2026-08-21", "code": "ABC", "eps_yoy_growth": 20.0},
        ]
    ).to_csv(path, index=False)

    with pytest.raises(EPSPITStoreError, match="duplicate snapshot/code"):
        EPSPITStore(str(path)).get("2026-08-21", "ABC")


def test_store_rejects_value_with_non_resolved_status(tmp_path):
    path = tmp_path / "signal_eps_pit.csv"
    pd.DataFrame(
        [
            {
                "snapshot_date": "2026-08-21",
                "code": "ABC",
                "eps_yoy_growth": 25.0,
                "status": "provider_error",
                "resolver_version": EPS_RESOLVER_VERSION,
            }
        ]
    ).to_csv(path, index=False)

    with pytest.raises(EPSPITStoreError, match="non-resolved status"):
        EPSPITStore(str(path)).get("2026-08-21", "ABC")


def test_store_ignores_legacy_resolver_version(tmp_path):
    path = tmp_path / "signal_eps_pit.csv"
    pd.DataFrame(
        [
            {
                "snapshot_date": "2026-08-21",
                "code": "ABC",
                "eps_yoy_growth": 25.0,
                "status": "resolved",
                "effective_date": "2026-08-01",
                "resolver_version": "eps_pit_v2",
            }
        ]
    ).to_csv(path, index=False)

    assert EPSPITStore(str(path)).get("2026-08-21", "ABC") is None


def test_store_persists_and_reuses_single_sec_cik_binding(tmp_path):
    path = tmp_path / "signal_eps_pit.csv"
    store = EPSPITStore(str(path))
    store.upsert(
        EPSResult(
            code="ABC",
            snapshot_date="2026-08-21",
            status=EPSStatus.RESOLVED,
            eps_yoy_growth=25.0,
            source="SEC",
            effective_date="2026-08-01",
            sec_cik="0000123456",
        )
    )
    store.upsert(
        EPSResult(
            code="ABC",
            snapshot_date="2026-08-28",
            status=EPSStatus.RESOLVED,
            eps_yoy_growth=30.0,
            source="YahooLiveObserved",
            effective_date="2026-08-28",
        )
    )

    assert store.get_sec_cik("ABC") == "0000123456"
