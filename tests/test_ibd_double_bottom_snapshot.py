import os

import pandas as pd
import pytest

import yfinance_data as yd


def _snapshot(date="2026-09-18"):
    return pd.DataFrame(
        [
            {
                "code": "AAA",
                "snapshot_date": date,
                "detection_path": "double_bottom",
                "signal_type": "Early Watch",
                "selection_eligible": True,
            },
            {
                "code": "BBB",
                "snapshot_date": date,
                "detection_path": "ibd_double_bottom",
                "signal_type": "Buy Zone",
                "selection_eligible": True,
            },
        ]
    )


@pytest.mark.parametrize(
    ("factory", "filename"),
    [
        (yd.IbdDoubleBottomSnapshotRun.complete, "ibd_double_bottom_snapshot.csv"),
        (yd.IbdDoubleBottomSnapshotRun.midweek, "ibd_double_bottom_snapshot_midweek.csv"),
    ],
)
def test_double_bottom_snapshot_publish_is_atomic_and_run_scoped(
    tmp_path, monkeypatch, factory, filename
):
    target = tmp_path / filename
    if filename.endswith("_midweek.csv"):
        monkeypatch.setattr(yd, "IBD_DOUBLE_BOTTOM_SNAPSHOT_MIDWEEK_PATH", str(target))
    else:
        monkeypatch.setattr(yd, "IBD_DOUBLE_BOTTOM_SNAPSHOT_PATH", str(target))

    run = factory()
    run.save_snapshot(_snapshot())

    assert target.is_file()
    loaded = run.ensure_current_snapshot()
    assert loaded["code"].tolist() == ["AAA", "BBB"]
    assert loaded["snapshot_date"].tolist() == ["2026-09-18", "2026-09-18"]
    assert not list(tmp_path.glob("*.pending"))


def test_double_bottom_snapshot_rejects_duplicate_codes(tmp_path, monkeypatch):
    target = tmp_path / "ibd_double_bottom_snapshot.csv"
    monkeypatch.setattr(yd, "IBD_DOUBLE_BOTTOM_SNAPSHOT_PATH", str(target))
    snapshot = _snapshot()
    snapshot.loc[1, "code"] = "AAA"

    with pytest.raises(ValueError, match="code"):
        yd.IbdDoubleBottomSnapshotRun.complete().save_snapshot(snapshot)

    assert not target.exists()


def test_double_bottom_snapshot_rejects_mixed_snapshot_dates(tmp_path, monkeypatch):
    target = tmp_path / "ibd_double_bottom_snapshot.csv"
    monkeypatch.setattr(yd, "IBD_DOUBLE_BOTTOM_SNAPSHOT_PATH", str(target))
    snapshot = _snapshot()
    snapshot.loc[1, "snapshot_date"] = "2026-09-19"

    with pytest.raises(ValueError, match="snapshot_date"):
        yd.IbdDoubleBottomSnapshotRun.complete().save_snapshot(snapshot)

    assert not target.exists()


def test_empty_double_bottom_snapshot_is_valid_when_schema_is_present(
    tmp_path, monkeypatch
):
    target = tmp_path / "ibd_double_bottom_snapshot.csv"
    monkeypatch.setattr(yd, "IBD_DOUBLE_BOTTOM_SNAPSHOT_PATH", str(target))
    empty = _snapshot().iloc[0:0]

    run = yd.IbdDoubleBottomSnapshotRun.complete()
    run.save_snapshot(empty)

    assert target.is_file()
    loaded = run.ensure_current_snapshot()
    assert loaded.empty
    assert set(yd.IbdDoubleBottomSnapshotRun.REQUIRED_COLUMNS).issubset(loaded.columns)


def test_double_bottom_snapshot_commit_is_explicit(monkeypatch, tmp_path):
    target = tmp_path / "ibd_double_bottom_snapshot.csv"
    monkeypatch.setattr(yd, "IBD_DOUBLE_BOTTOM_SNAPSHOT_PATH", str(target))
    calls = []
    monkeypatch.setattr(yd, "_commit_managed_csv", lambda path, message: calls.append((path, message)))

    run = yd.IbdDoubleBottomSnapshotRun.complete()
    run.save_snapshot(_snapshot())
    assert calls == []

    run.commit()
    assert calls == [(str(target), "Update IBD double bottom snapshot")]
