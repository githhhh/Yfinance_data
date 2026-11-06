"""Focused adversarial tests for manual BF Watch download holds."""
from __future__ import annotations
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import pool_exit_watch_store as store


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def official_repo(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    git(root, "config", "user.email", "hold@example.invalid")
    git(root, "config", "user.name", "Watch test")
    git(root, "branch", "-M", "main")
    (root / "us").mkdir()
    frame = pd.DataFrame([{"code": "A", "snapshot_date": "2026-10-30"}])
    frame.to_csv(root / "us" / "breakout_follow_pool.csv", index=False)
    source, digest = store.pool_identity(frame)
    store.watch_frame([{"code": "A", "watch_cycle_id": "A:2026-10-26",
                        "phase": "OBSERVING", "events": []}],
                      snapshot_date=source, pool_codes_hash=digest).to_csv(
                          root / store.WATCH_COMPLETE, index=False, encoding="utf-8-sig")
    git(root, "add", "us")
    git(root, "commit", "-qm", f"paired\n\nWatch-Source-Date: {source}\nWatch-Codes-SHA256: {digest}")
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
    git(root, "remote", "add", "origin", str(remote))
    git(root, "push", "-u", "origin", "main")
    return root


def test_only_strict_24_columns_and_one_time_migration():
    assert len(store.COLUMNS) == 24
    frame = store.watch_frame([{"code": "A", "phase": "OBSERVING",
                                "watch_cycle_id": "A:2026-10-26"}],
                              snapshot_date="2026-10-30", pool_codes_hash="digest")
    assert list(frame.columns) == list(store.COLUMNS)
    with pytest.raises(ValueError, match="24-column"):
        store.parse_watch_frame(frame.assign(source_pool_date="2026-10-30"))


def test_active_same_cycle_inherits_manual_clear_and_ended_or_new_does_not():
    published = [
        {"code": "A", "phase": "OBSERVING", "watch_cycle_id": "A:old",
         "download_hold_reason": "REQUEST_ERROR"},
        {"code": "B", "phase": "AWAITING_DATA", "watch_cycle_id": "B:old",
         "download_hold_reason": ""},
    ]
    draft = store.watch_frame([
        {"code": "A", "phase": "OBSERVING", "watch_cycle_id": "A:old",
         "download_hold_reason": ""},
        {"code": "B", "phase": "OBSERVING", "watch_cycle_id": "B:old",
         "download_hold_reason": "STALE_HOLD"},
        {"code": "C", "phase": "OBSERVING", "watch_cycle_id": "C:new",
         "download_hold_reason": "STALE_HOLD"},
        {"code": "D", "phase": "RETURNED", "watch_cycle_id": "D:old",
         "download_hold_reason": "STALE_HOLD"},
        {"code": "E", "phase": "PROVISIONAL_OUT", "watch_cycle_id": "",
         "download_hold_reason": "STALE_HOLD"},
    ], snapshot_date="2026-10-30", pool_codes_hash="digest")
    merged = store.inherit_download_holds(draft, published)
    got = dict(zip(merged.code, merged.download_hold_reason))
    assert got == {"A": "REQUEST_ERROR", "B": "", "C": "", "D": "", "E": ""}
    store.parse_watch_frame(merged)


def test_hold_excludes_even_if_independently_selected_and_preserves_normal_list(tmp_path):
    from DataStore import apply_final_download_filter
    frame = store.watch_frame([
        {"code": "BRK.B", "phase": "OBSERVING",
         "watch_cycle_id": "BRK.B:2026-10-26", "download_hold_reason": "REQUEST_ERROR"},
        {"code": "A", "phase": "OBSERVING", "watch_cycle_id": "A:2026-10-26"},
    ], snapshot_date="2026-10-30", pool_codes_hash="digest")
    folder = tmp_path / "us"
    folder.mkdir()
    frame.to_csv(folder / "bf_pool_exit_watch.csv", index=False)
    assert store.held_watch_codes(tmp_path) == ["BRK-B"]
    all_sources = ["A", "BRK-B", "LXP", "OTHER"]
    assert apply_final_download_filter(all_sources, held_codes=[])[0] == all_sources
    filtered, excluded = apply_final_download_filter(
        all_sources + ["BRK.B"], held_codes=store.held_watch_codes(tmp_path),
        manual_csv="LXP")
    assert filtered == ["A", "OTHER"]
    assert excluded == {"BRK-B", "LXP"}


def test_manual_watch_only_commit_has_pair_trailers_and_does_not_change_pool(tmp_path, monkeypatch):
    import yfinance_data as authority
    root = official_repo(tmp_path)
    monkeypatch.setattr(authority, "DATA_ROOT", str(root))
    pool_before = (root / "us/breakout_follow_pool.csv").read_bytes()
    watch_before = store.parse_watch_csv(root / store.WATCH_COMPLETE)[0]
    authority.set_download_hold("A", "A:2026-10-26", "REQUEST_ERROR")
    watch_after = store.parse_watch_csv(root / store.WATCH_COMPLETE)[0]
    assert {k: v for k, v in watch_before.items() if k != "download_hold_reason"} == {
        k: v for k, v in watch_after.items() if k != "download_hold_reason"}
    current = authority.get_download_hold("A")
    assert current["download_hold_reason"] == "REQUEST_ERROR"
    assert (root / "us/breakout_follow_pool.csv").read_bytes() == pool_before
    assert git(root, "show", "--format=", "--name-only", "HEAD") == store.WATCH_COMPLETE
    message = git(root, "show", "-s", "--format=%B", "HEAD")
    _, digest = store.pool_identity(pd.read_csv(root / "us/breakout_follow_pool.csv"))
    assert "Watch-Source-Date: 2026-10-30" in message
    assert f"Watch-Codes-SHA256: {digest}" in message
    assert store.official_watch_for_week(root, "2026-10-26")[0]["download_hold_reason"] == "REQUEST_ERROR"
    authority.set_download_hold("A", "A:2026-10-26", "")
    assert authority.get_download_hold("A")["download_hold_reason"] == ""
    with pytest.raises(ValueError, match="identity"):
        authority.set_download_hold("A", "A:old-cycle", "INVALID")
    with pytest.raises(ValueError, match="identity"):
        authority.set_download_hold("LXP", "LXP:2026-10-26", "REQUEST_ERROR")


def test_stale_watch_vintage_rejected_before_replace(tmp_path, monkeypatch):
    import yfinance_data as authority
    root = official_repo(tmp_path)
    monkeypatch.setattr(authority, "DATA_ROOT", str(root))
    # A complete-week run has already staged a fresh Pool outside this local
    # official file; register_companion must read the REMOTE prior Watch.
    new_pool = pd.DataFrame([{"code": "A", "snapshot_date": "2026-11-06"}])
    source, digest = store.pool_identity(new_pool)
    obj = authority.BreakoutFollowPoolRun.weekend()
    monkeypatch.setattr(obj, "ensure_current_snapshot", lambda: new_pool)
    stage = root / "us/.new-watch.pending"
    store.watch_frame([{"code": "A", "watch_cycle_id": "A:2026-10-26",
                        "phase": "OBSERVING", "events": []}],
                      snapshot_date=source, pool_codes_hash=digest).to_csv(
                          stage, index=False, encoding="utf-8-sig")
    target = root / store.WATCH_COMPLETE
    obj.register_companion(target_path=str(target), staged_path=str(stage),
                           source_pool_date=source, source_pool_codes_hash=digest)
    assert obj._watch_companion["watch_version"]
    # An independent manual writer advances the same official Watch.
    authority.set_download_hold("A", "A:2026-10-26", "REQUEST_ERROR")
    now = target.read_bytes()
    def simulate_locked_publish(path, *, extra_paths=None, before_stage=None):
        assert before_stage is not None
        before_stage()
    monkeypatch.setattr(authority, "_commit_pool", simulate_locked_publish)
    with pytest.raises(RuntimeError, match="changed since preparation"):
        obj.commit()
    assert target.read_bytes() == now
    assert stage.exists()



def test_midweek_projection_never_carries_manual_hold(tmp_path, monkeypatch):
    import yfinance_data as authority
    root = tmp_path / "no_git"
    (root / "us").mkdir(parents=True)
    monkeypatch.setattr(authority, "DATA_ROOT", str(root))
    pool = pd.DataFrame([{"code": "A", "snapshot_date": "2026-11-04"}])
    when, digest = store.pool_identity(pool)
    staged = root / "us/.midweek.pending"
    store.watch_frame([
        {"code": "A", "watch_cycle_id": "A:2026-10-26",
         "phase": "OBSERVING", "events": [],
         "download_hold_reason": "STALE_HOLD"},
    ], snapshot_date=when, pool_codes_hash=digest).to_csv(
        staged, index=False, encoding="utf-8-sig")
    target = root / store.WATCH_MIDWEEK
    run = authority.BreakoutFollowPoolRun.midweek()
    monkeypatch.setattr(run, "ensure_current_snapshot", lambda: pool)
    run.register_companion(target_path=str(target), staged_path=str(staged),
                           source_pool_date=when, source_pool_codes_hash=digest)
    assert store.parse_watch_csv(staged)[0]["download_hold_reason"] == ""
    assert run._watch_companion["watch_version"] is None



def test_real_managed_git_lock_rejects_stale_companion_before_replace(tmp_path, monkeypatch):
    """Exercise the real managed writer, not a mocked before_stage callback."""
    import yfinance_data as authority
    root = official_repo(tmp_path)
    monkeypatch.setattr(authority, "DATA_ROOT", str(root))

    next_pool = pd.DataFrame([{"code": "A", "snapshot_date": "2026-11-06"}])
    when, digest = store.pool_identity(next_pool)
    pending = root / "us/.stale-companion.pending"
    store.watch_frame([{
        "code": "A", "watch_cycle_id": "A:2026-10-26",
        "phase": "OBSERVING", "events": [],
    }], snapshot_date=when, pool_codes_hash=digest).to_csv(
        pending, index=False, encoding="utf-8-sig")

    run = authority.BreakoutFollowPoolRun.weekend()
    monkeypatch.setattr(run, "ensure_current_snapshot", lambda: next_pool)
    target = root / store.WATCH_COMPLETE
    run.register_companion(
        target_path=str(target), staged_path=str(pending),
        source_pool_date=when, source_pool_codes_hash=digest)
    staged_digest = run._watch_companion["sha"]

    authority.set_download_hold("A", "A:2026-10-26", "REQUEST_ERROR")
    published_head = git(root, "rev-parse", "origin/main")
    published_watch = target.read_bytes()

    def use_real_git_writer(_pool_path, *, extra_paths=None, before_stage=None):
        authority._commit_managed_csv(
            extra_paths or [], message="stale run should never commit",
            before_stage=before_stage)

    monkeypatch.setattr(authority, "_commit_pool", use_real_git_writer)
    with pytest.raises(RuntimeError, match="changed since preparation"):
        run.commit()
    assert git(root, "rev-parse", "origin/main") == published_head
    assert target.read_bytes() == published_watch
    assert pending.exists()
    from hashlib import sha256
    assert sha256(pending.read_bytes()).hexdigest() == staged_digest
