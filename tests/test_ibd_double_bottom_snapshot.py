import pandas as pd
import pytest

import yfinance_data as yd


def _snapshot():
    return pd.DataFrame(
        [
            {
                "code": "AAA",
                "detection_path": "double_bottom",
                "signal_type": "Early Watch",
                "selection_eligible": True,
            },
            {
                "code": "BBB",
                "detection_path": "ibd_double_bottom",
                "signal_type": "Buy Zone",
                "selection_eligible": True,
            },
        ]
    )


@pytest.mark.parametrize(
    ("factory", "midweek", "filename"),
    [
        (
            yd.IbdDoubleBottomSnapshotRun.weekend,
            False,
            "ibd_double_bottom_snapshot.csv",
        ),
        (
            yd.IbdDoubleBottomSnapshotRun.midweek,
            True,
            "ibd_double_bottom_snapshot_midweek.csv",
        ),
    ],
)
def test_double_bottom_snapshot_publish_is_atomic_and_run_scoped(
    tmp_path,
    monkeypatch,
    factory,
    midweek,
    filename,
):
    complete = tmp_path / "ibd_double_bottom_snapshot.csv"
    mid = tmp_path / "ibd_double_bottom_snapshot_midweek.csv"
    monkeypatch.setattr(yd, "IBD_DOUBLE_BOTTOM_SNAPSHOT_PATH", str(complete))
    monkeypatch.setattr(yd, "IBD_DOUBLE_BOTTOM_SNAPSHOT_MIDWEEK_PATH", str(mid))

    run = factory()
    run.save_snapshot(_snapshot(), snapshot_date="2026-09-18")

    target = mid if midweek else complete
    assert target.name == filename
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
        yd.IbdDoubleBottomSnapshotRun.weekend().save_snapshot(
            snapshot,
            snapshot_date="2026-09-18",
        )

    assert not target.exists()


def test_double_bottom_snapshot_rejects_invalid_explicit_date(tmp_path, monkeypatch):
    target = tmp_path / "ibd_double_bottom_snapshot.csv"
    monkeypatch.setattr(yd, "IBD_DOUBLE_BOTTOM_SNAPSHOT_PATH", str(target))

    with pytest.raises(ValueError, match="snapshot_date"):
        yd.IbdDoubleBottomSnapshotRun.weekend().save_snapshot(
            _snapshot(),
            snapshot_date="not-a-date",
        )

    assert not target.exists()


def test_empty_double_bottom_snapshot_is_valid(tmp_path, monkeypatch):
    target = tmp_path / "ibd_double_bottom_snapshot.csv"
    monkeypatch.setattr(yd, "IBD_DOUBLE_BOTTOM_SNAPSHOT_PATH", str(target))
    empty = _snapshot().iloc[0:0]

    run = yd.IbdDoubleBottomSnapshotRun.weekend()
    run.save_snapshot(empty, snapshot_date="2026-09-18")

    assert target.is_file()
    loaded = run.ensure_current_snapshot()
    assert loaded.empty
    assert set(yd.IbdDoubleBottomSnapshotRun.REQUIRED_COLUMNS).issubset(loaded.columns)


def test_double_bottom_snapshot_commit_is_explicit(monkeypatch, tmp_path):
    target = tmp_path / "ibd_double_bottom_snapshot.csv"
    monkeypatch.setattr(yd, "IBD_DOUBLE_BOTTOM_SNAPSHOT_PATH", str(target))
    calls = []
    monkeypatch.setattr(
        yd,
        "_commit_managed_csv",
        lambda path, message: calls.append((path, message)),
    )

    run = yd.IbdDoubleBottomSnapshotRun.weekend()
    run.save_snapshot(_snapshot(), snapshot_date="2026-09-18")
    assert calls == []

    run.commit()
    assert calls == [
        (
            str(target),
            "Update IBD double bottom snapshot",
        )
    ]


def test_double_bottom_weekend_preserves_complete_name_compatibility():
    weekend = yd.IbdDoubleBottomSnapshotRun.weekend()
    complete = yd.IbdDoubleBottomSnapshotRun.complete()

    assert weekend.name == "complete"
    assert complete.name == "complete"
    assert weekend.path == yd.IBD_DOUBLE_BOTTOM_SNAPSHOT_PATH
    assert complete.path == yd.IBD_DOUBLE_BOTTOM_SNAPSHOT_PATH


def test_breakout_follow_commit_reuses_managed_csv(monkeypatch, tmp_path):
    pool_path = tmp_path / "breakout_follow_pool.csv"
    pit_path = tmp_path / "signal_eps_pit.csv"
    pit_path.write_text("code\nAAA\n", encoding="utf-8")
    calls = []

    monkeypatch.setattr(yd, "_pit_store_path", lambda: str(pit_path))
    monkeypatch.setattr(
        yd,
        "_commit_managed_csv",
        lambda paths, message: calls.append((list(paths), message)),
    )

    yd._commit_pool(str(pool_path))

    assert calls == [
        (
            [str(pool_path), str(pit_path)],
            "Update breakout follow pool",
        )
    ]


def test_commit_managed_csv_pushes_even_when_no_new_diff(monkeypatch, tmp_path):
    calls = []

    class Result:
        def __init__(self, returncode=0):
            self.returncode = returncode
            self.args = []

    def fake_run(args, cwd=None, check=False):
        calls.append(list(args))
        if args[:4] == ["git", "diff", "--cached", "--quiet"]:
            return Result(0)
        return Result(0)

    monkeypatch.setattr(yd, "DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(yd.subprocess, "run", fake_run)

    yd._commit_managed_csv("us/example.csv", message="Update example")

    assert ["git", "commit", "-m", "Update example"] not in calls
    assert ["git", "push"] in calls


def test_commit_managed_csv_commits_then_pushes_when_diff_is_staged(
    monkeypatch,
    tmp_path,
):
    calls = []

    class Result:
        def __init__(self, returncode=0):
            self.returncode = returncode
            self.args = []

    def fake_run(args, cwd=None, check=False):
        calls.append(list(args))
        if args[:4] == ["git", "diff", "--cached", "--quiet"]:
            return Result(1)
        return Result(0)

    monkeypatch.setattr(yd, "DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(yd.subprocess, "run", fake_run)

    yd._commit_managed_csv("us/example.csv", message="Update example")

    commit_idx = calls.index(["git", "commit", "-m", "Update example"])
    push_idx = calls.index(["git", "push"])
    assert commit_idx < push_idx
