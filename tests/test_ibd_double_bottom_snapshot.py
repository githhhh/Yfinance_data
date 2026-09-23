import json

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


def _patch_paths(tmp_path, monkeypatch, *, midweek=False):
    complete = tmp_path / "ibd_double_bottom_snapshot.csv"
    mid = tmp_path / "ibd_double_bottom_snapshot_midweek.csv"
    state = tmp_path / "ibd_double_bottom_snapshot_state.json"
    monkeypatch.setattr(yd, "IBD_DOUBLE_BOTTOM_SNAPSHOT_PATH", str(complete))
    monkeypatch.setattr(yd, "IBD_DOUBLE_BOTTOM_SNAPSHOT_MIDWEEK_PATH", str(mid))
    monkeypatch.setattr(yd, "IBD_DOUBLE_BOTTOM_SNAPSHOT_STATE_PATH", str(state))
    return (mid if midweek else complete), state


@pytest.mark.parametrize(
    ("factory", "midweek", "kind"),
    [
        (yd.IbdDoubleBottomSnapshotRun.complete, False, "complete"),
        (yd.IbdDoubleBottomSnapshotRun.midweek, True, "midweek"),
    ],
)
def test_double_bottom_snapshot_publish_is_atomic_and_run_scoped(
    tmp_path, monkeypatch, factory, midweek, kind
):
    target, state = _patch_paths(tmp_path, monkeypatch, midweek=midweek)

    run = factory()
    run.save_snapshot(_snapshot(), snapshot_date="2026-09-18")

    assert target.is_file()
    assert state.is_file()
    loaded = run.ensure_current_snapshot()
    assert loaded["code"].tolist() == ["AAA", "BBB"]
    assert loaded["snapshot_date"].tolist() == ["2026-09-18", "2026-09-18"]
    assert json.loads(state.read_text(encoding="utf-8")) == {
        "kind": kind,
        "snapshot_date": "2026-09-18",
    }
    assert not list(tmp_path.glob("*.pending"))


def test_double_bottom_snapshot_rejects_duplicate_codes(tmp_path, monkeypatch):
    target, state = _patch_paths(tmp_path, monkeypatch)
    snapshot = _snapshot()
    snapshot.loc[1, "code"] = "AAA"

    with pytest.raises(ValueError, match="code"):
        yd.IbdDoubleBottomSnapshotRun.complete().save_snapshot(
            snapshot,
            snapshot_date="2026-09-18",
        )

    assert not target.exists()
    assert not state.exists()


def test_double_bottom_snapshot_rejects_invalid_explicit_date(tmp_path, monkeypatch):
    target, state = _patch_paths(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="snapshot_date"):
        yd.IbdDoubleBottomSnapshotRun.complete().save_snapshot(
            _snapshot(),
            snapshot_date="not-a-date",
        )

    assert not target.exists()
    assert not state.exists()


def test_explicit_snapshot_date_is_authoritative(tmp_path, monkeypatch):
    target, state = _patch_paths(tmp_path, monkeypatch)
    snapshot = _snapshot("2026-09-17")

    run = yd.IbdDoubleBottomSnapshotRun.complete()
    run.save_snapshot(snapshot, snapshot_date="2026-09-18")

    loaded = pd.read_csv(target, encoding="utf-8-sig")
    assert loaded["snapshot_date"].tolist() == ["2026-09-18", "2026-09-18"]
    assert json.loads(state.read_text(encoding="utf-8"))["snapshot_date"] == "2026-09-18"


def test_empty_double_bottom_snapshot_persists_authoritative_date(
    tmp_path, monkeypatch
):
    target, state = _patch_paths(tmp_path, monkeypatch)
    empty = _snapshot().iloc[0:0]

    run = yd.IbdDoubleBottomSnapshotRun.complete()
    run.save_snapshot(empty, snapshot_date="2026-09-18")

    assert target.is_file()
    loaded = run.ensure_current_snapshot()
    assert loaded.empty
    assert set(yd.IbdDoubleBottomSnapshotRun.REQUIRED_COLUMNS).issubset(loaded.columns)
    assert json.loads(state.read_text(encoding="utf-8")) == {
        "kind": "complete",
        "snapshot_date": "2026-09-18",
    }


def test_double_bottom_snapshot_commit_is_explicit(monkeypatch, tmp_path):
    target, state = _patch_paths(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(
        yd,
        "_commit_managed_files",
        lambda paths, message: calls.append((list(paths), message)),
    )

    run = yd.IbdDoubleBottomSnapshotRun.complete()
    run.save_snapshot(_snapshot(), snapshot_date="2026-09-18")
    assert calls == []

    run.commit()
    assert calls == [
        (
            [str(target), str(state)],
            "Update IBD double bottom snapshot",
        )
    ]
