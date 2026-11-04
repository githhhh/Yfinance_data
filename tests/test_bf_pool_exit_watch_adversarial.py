"""Adversarial Watch publication and historical baseline tests."""
from __future__ import annotations
import io
import subprocess
from pathlib import Path
import pandas as pd
import pytest
import pool_exit_watch_store as store


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def setup_repo(tmp_path):
    root=tmp_path/"history"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    git(root,"config","user.email","watch-test@example.invalid")
    git(root,"config","user.name","watch-test")
    (root/"us").mkdir()
    return root


def commit_pool(root, d, symbols, *, paired=False):
    frame = pd.DataFrame([{"code":code,"snapshot_date":d,"ceiling":100,"signal":False}
                          for code in symbols])
    frame.to_csv(root/"us"/"breakout_follow_pool.csv",index=False)
    git(root,"add","us/breakout_follow_pool.csv")
    message = "pool"
    if paired:
        _, digest = store.pool_identity(frame)
        message += f"\n\nWatch-Source-Date: {d}\nWatch-Codes-SHA256: {digest}"
    git(root,"commit","-qm",message)
    git(root,"update-ref","refs/remotes/origin/main","HEAD")


def test_published_pool_history_and_valid_empty_watch(tmp_path):
    root=setup_repo(tmp_path)
    commit_pool(root,"2026-10-16",["A","B"])
    commit_pool(root,"2026-10-23",["B"])
    pools=store.published_complete_pools(root,limit=2)
    assert [p["snapshot_date"] for p in pools]==["2026-10-23","2026-10-16"]
    assert store.official_watch_for_week(root,"2026-10-19") is None
    _, digest=store.pool_identity(pd.DataFrame(pools[0]["rows"]))
    frame=store.watch_frame([],snapshot_date="2026-10-23",pool_codes_hash=digest)
    frame.to_csv(root/"us"/"bf_pool_exit_watch.csv",index=False)
    git(root,"add","us/bf_pool_exit_watch.csv")
    git(root,"commit","-qm",
        f"bootstrap empty\n\nWatch-Source-Date: 2026-10-23\nWatch-Codes-SHA256: {digest}")
    git(root,"update-ref","refs/remotes/origin/main","HEAD")
    assert store.official_watch_for_week(root,"2026-10-20") is None
    assert store.official_watch_for_week(root,"2026-10-19")==[]
    # An empty Watch has identical CSV bytes for a second week; history must
    # be sourced from the paired Pool commit, not Watch-only file change log.
    commit_pool(root, "2026-10-30", ["B"], paired=True)
    assert store.official_watch_for_week(root, "2026-10-26") == []
    # A non-Watch writer changing the Pool must not be accepted as a new baseline.
    commit_pool(root, "2026-11-06", ["C"])
    assert store.official_watch_for_week(root, "2026-11-02") is None


def test_owned_git_commit_excludes_unrelated_already_staged_file(tmp_path):
    root=setup_repo(tmp_path)
    (root/"owner.txt").write_text("old")
    (root/"foreign.txt").write_text("old")
    git(root,"add",".")
    git(root,"commit","-qm","initial")
    (root/"foreign.txt").write_text("unrelated staged")
    git(root,"add","foreign.txt")
    (root/"owner.txt").write_text("new")
    git(root,"add","owner.txt")
    git(root,"commit","--only","-qm","owned update","--","owner.txt")
    assert git(root,"show","HEAD:foreign.txt")=="old"
    assert git(root,"show","HEAD:owner.txt")=="new"
    assert "foreign.txt" in git(root,"diff","--cached","--name-only")


def test_companion_rejects_other_pool_source(tmp_path,monkeypatch):
    import yfinance_data
    pool=pd.DataFrame([{"code":"A","snapshot_date":"2026-10-30"}])
    obj=yfinance_data.BreakoutFollowPoolRun.weekend()
    monkeypatch.setattr(obj,"ensure_current_snapshot",lambda:pool)
    monkeypatch.setattr(yfinance_data,"DATA_ROOT",str(tmp_path))
    path=tmp_path/store.WATCH_COMPLETE
    path.parent.mkdir(parents=True)
    staged=tmp_path/"us"/".test.pending"
    store.watch_frame([],snapshot_date="2026-10-30",pool_codes_hash="wrong").to_csv(staged,index=False)
    with pytest.raises(ValueError,match="source identity"):
        obj.register_companion(target_path=str(path),staged_path=str(staged),
                               source_pool_date="2026-10-30",
                               source_pool_codes_hash="wrong")



def test_companion_survives_until_commit_and_rejects_pool_mutation(tmp_path,monkeypatch):
    import yfinance_data as authority
    import pool_exit_watch_store as store
    monkeypatch.setattr(authority,"DATA_ROOT",str(tmp_path))
    path=tmp_path/store.WATCH_COMPLETE
    path.parent.mkdir(parents=True)
    old_pool=pd.DataFrame([{"code":"A","snapshot_date":"2026-10-30"}])
    d,h=store.pool_identity(old_pool)
    staged=tmp_path/"us"/".watch.pending"
    store.watch_frame([],snapshot_date=d,pool_codes_hash=h).to_csv(staged,index=False)
    obj=authority.BreakoutFollowPoolRun.weekend()
    pool=[old_pool]
    monkeypatch.setattr(obj,"ensure_current_snapshot",lambda:pool[0])
    obj.register_companion(target_path=str(path),staged_path=str(staged),
                           source_pool_date=d,source_pool_codes_hash=h)
    assert staged.exists() and not path.exists()
    pool[0]=pd.DataFrame([{"code":"B","snapshot_date":"2026-10-30"}])
    with pytest.raises(RuntimeError,match="Pool changed"):
        obj.commit()
    assert staged.exists() and not path.exists()
    pool[0]=old_pool
    published=[]
    monkeypatch.setattr(authority,"_commit_pool",
                       lambda path,extra_paths=None: published.extend(extra_paths or []))
    obj.commit()
    assert path.exists() and not staged.exists()
    assert str(path) in published
    assert obj._watch_companion is None


def test_no_only_pool_commit_when_watch_enabled(tmp_path,monkeypatch):
    import yfinance_data as authority
    import pool_exit_watch_store as store
    monkeypatch.setattr(authority,"DATA_ROOT",str(tmp_path))
    path=tmp_path/store.WATCH_COMPLETE
    path.parent.mkdir(parents=True)
    store.watch_frame([],snapshot_date="2026-10-30",pool_codes_hash="x").to_csv(path,index=False)
    obj=authority.BreakoutFollowPoolRun.weekend()
    monkeypatch.setattr(obj,"ensure_current_snapshot",
                        lambda:pd.DataFrame([{"code":"A","snapshot_date":"2026-10-30"}]))
    with pytest.raises(RuntimeError,match="requires companion"):
        obj.commit()


def test_watch_download_source_must_be_remote_published(tmp_path):
    root = setup_repo(tmp_path)
    commit_pool(root, "2026-10-23", ["A"])
    pool = pd.read_csv(root / "us" / "breakout_follow_pool.csv")
    when, digest = store.pool_identity(pool)
    target = root / store.WATCH_COMPLETE
    store.watch_frame([], snapshot_date=when, pool_codes_hash=digest).to_csv(
        target, index=False)
    # Locally prepared but never pushed to origin/main is NOT a valid source.
    with pytest.raises(RuntimeError, match="not confirmed"):
        store.active_watch_codes(root)
    git(root, "add", store.WATCH_COMPLETE)
    git(root, "commit", "-qm",
        f"bootstrap\n\nWatch-Source-Date: {when}\nWatch-Codes-SHA256: {digest}")
    git(root, "update-ref", "refs/remotes/origin/main", "HEAD")
    assert store.active_watch_codes(root) == []
    target.unlink()
    with pytest.raises(RuntimeError, match="missing"):
        store.active_watch_codes(root)
    target.write_text("malformed local Watch")
    with pytest.raises(RuntimeError, match="differs"):
        store.active_watch_codes(root)


def test_deleted_published_watch_blocks_pool_only_commit(tmp_path, monkeypatch):
    import yfinance_data as authority
    root = setup_repo(tmp_path)
    monkeypatch.setattr(authority, "DATA_ROOT", str(root))
    commit_pool(root, "2026-10-23", ["A"])
    pool = pd.DataFrame([{"code":"A","snapshot_date":"2026-10-23"}])
    when, digest = store.pool_identity(pool)
    target = root / store.WATCH_COMPLETE
    store.watch_frame([], snapshot_date=when, pool_codes_hash=digest).to_csv(
        target, index=False)
    git(root, "add", store.WATCH_COMPLETE)
    git(root, "commit", "-qm", "initialize")
    git(root, "update-ref", "refs/remotes/origin/main", "HEAD")
    target.unlink()
    monkeypatch.setattr(authority, "DATA_ROOT", str(root))
    obj = authority.BreakoutFollowPoolRun.weekend()
    monkeypatch.setattr(obj, "ensure_current_snapshot", lambda: pool)
    with pytest.raises(RuntimeError, match="requires companion"):
        obj.commit()


def test_managed_pair_push_rejection_then_retry_keeps_foreign_staging(tmp_path, monkeypatch):
    """A rejected remote push leaves a single local pair, never an extra file."""
    import yfinance_data as authority
    root = setup_repo(tmp_path)
    monkeypatch.setattr(authority, "DATA_ROOT", str(root))
    commit_pool(root, "2026-10-23", ["A"])
    git(root, "branch", "-M", "main")
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
    git(root, "remote", "add", "origin", str(remote))
    git(root, "push", "-u", "origin", "main")
    git(root, "update-ref", "refs/remotes/origin/main", "HEAD")

    pool_path = root / "us" / "breakout_follow_pool.csv"
    pool = pd.DataFrame([{"code": "B", "snapshot_date": "2026-10-30"}])
    pool.to_csv(pool_path, index=False)
    when, digest = store.pool_identity(pool)
    watch_path = root / store.WATCH_COMPLETE
    store.watch_frame([], snapshot_date=when, pool_codes_hash=digest).to_csv(
        watch_path, index=False)
    foreign = root / "unrelated.txt"
    foreign.write_text("should remain staged")
    git(root, "add", "unrelated.txt")

    # Real local bare-repo receive hook fails the network publication step.
    hook = remote / "hooks" / "pre-receive"
    hook.write_text("#!/bin/sh\nexit 1\n")
    hook.chmod(0o755)
    monkeypatch.setattr(authority.time, "sleep", lambda *_: None)
    try:
        with pytest.raises(subprocess.CalledProcessError):
            authority._commit_managed_csv(
                [str(pool_path), str(watch_path)], message="Update test Pool Watch pair")
        committed = git(root, "rev-parse", "HEAD")
        assert committed != git(root, "rev-parse", "@{u}")
        assert "unrelated.txt" not in git(root, "show", "--pretty=format:",
                                           "--name-only", "HEAD").splitlines()
        assert "unrelated.txt" in git(root, "diff", "--cached", "--name-only")
    finally:
        hook.unlink(missing_ok=True)

    # With no new tracked file changes, the authority retries the unpushed
    # managed commit; the foreign staged file must still not be published.
    authority._commit_managed_csv(
        [str(pool_path), str(watch_path)], message="Update test Pool Watch pair")
    assert git(root, "rev-parse", "HEAD") == committed
    assert git(root, "rev-parse", "@{u}") == committed
    assert "unrelated.txt" not in git(root, "ls-tree", "--name-only", "HEAD").splitlines()


def setup_bare_remote(root, tmp_path):
    remote = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
    git(root, "branch", "-M", "main")
    git(root, "remote", "add", "origin", str(remote))
    git(root, "push", "-u", "origin", "main")
    subprocess.run(["git", "--git-dir", str(remote), "symbolic-ref",
                    "HEAD", "refs/heads/main"], check=True)
    return remote


def test_managed_writer_does_not_push_older_foreign_local_commits(tmp_path, monkeypatch):
    import yfinance_data as authority
    root = setup_repo(tmp_path)
    (root / "owned.txt").write_text("initial")
    (root / "foreign.txt").write_text("initial")
    git(root, "add", ".")
    git(root, "commit", "-qm", "initial")
    remote = setup_bare_remote(root, tmp_path)

    (root / "foreign.txt").write_text("unpublished")
    git(root, "add", "foreign.txt")
    git(root, "commit", "-qm", "foreign local commit")
    (root / "owned.txt").write_text("new")
    monkeypatch.setattr(authority, "DATA_ROOT", str(root))
    with pytest.raises(RuntimeError, match="foreign committed"):
        authority._commit_managed_csv(str(root / "owned.txt"), message="owned")
    assert git(root, "rev-parse", "@{u}") != git(root, "rev-parse", "HEAD")
    assert subprocess.check_output(["git", "--git-dir", str(remote), "show",
                                    "main:foreign.txt"], text=True).strip() == "initial"


def test_external_writer_refuses_dirty_pool_watch(tmp_path, monkeypatch):
    import yfinance_data as authority
    root = setup_repo(tmp_path)
    (root / "owner.txt").write_text("old")
    (root / "us" / "breakout_follow_pool.csv").write_text("old Pool")
    git(root, "add", ".")
    git(root, "commit", "-qm", "initial")
    setup_bare_remote(root, tmp_path)
    (root / "us" / "breakout_follow_pool.csv").write_text("dirty Pool")
    (root / "owner.txt").write_text("new")
    monkeypatch.setattr(authority, "DATA_ROOT", str(root))
    with pytest.raises(RuntimeError, match="dirty Pool/Watch"):
        authority._commit_managed_csv(str(root / "owner.txt"), message="data")
    assert git(root, "show", "HEAD:owner.txt") == "old"


def test_non_fast_forward_fails_before_managed_commit(tmp_path, monkeypatch):
    import yfinance_data as authority
    root = setup_repo(tmp_path)
    (root / "owned.txt").write_text("initial")
    git(root, "add", ".")
    git(root, "commit", "-qm", "initial")
    remote = setup_bare_remote(root, tmp_path)

    # Independent writer advances remote/main after this checkout was made.
    peer = tmp_path / "peer"
    subprocess.run(["git", "clone", "-q", str(remote), str(peer)], check=True)
    git(peer, "config", "user.email", "peer@example.invalid")
    git(peer, "config", "user.name", "peer")
    (peer / "other.txt").write_text("remote advanced")
    git(peer, "add", "other.txt")
    git(peer, "commit", "-qm", "peer update")
    git(peer, "push", "origin", "HEAD:main")

    (root / "owned.txt").write_text("new")
    monkeypatch.setattr(authority, "DATA_ROOT", str(root))
    with pytest.raises(subprocess.CalledProcessError):
        authority._commit_managed_csv(str(root / "owned.txt"), message="owned")
    assert git(root, "show", "HEAD:owned.txt") == "initial"
    assert subprocess.check_output(["git", "--git-dir", str(remote), "show",
                                    "main:other.txt"], text=True).strip() == "remote advanced"


def test_all_data_writers_share_single_owned_publisher():
    root = Path(__file__).parents[1]
    files = (
        ".github/workflows/data-update.yml",
        ".github/workflows/data-update-1d.yml",
        ".github/workflows/data-update-1wk.yml",
    )
    for name in files:
        source = (root / name).read_text(encoding="utf-8")
        assert "python scripts/publish_owned.py" in source
        assert "group: bf-stock-data-main-writers" in source
        assert "git commit --only" not in source
        assert "git push origin HEAD:main" not in source
    local = (root / "scripts/run_local_data_update.sh").read_text()
    assert "python scripts/publish_owned.py" in local
    assert "git push origin HEAD:main" not in local



def test_bootstrap_publishes_once_and_unblocks_download(tmp_path, monkeypatch):
    import yfinance_data as authority
    root = setup_repo(tmp_path)
    commit_pool(root, "2026-10-16", ["A", "B"])
    commit_pool(root, "2026-10-23", ["B"])
    setup_bare_remote(root, tmp_path)
    monkeypatch.setattr(authority, "DATA_ROOT", str(root))
    last_two = store.published_complete_pools(root, limit=2)
    recent = last_two[0]
    rows = [{"code": "A", "phase": "AWAITING_DATA",
             "original_pool_date": "2026-10-16",
             "watch_cycle_id": "A:2026-10-19", "events": []}]
    with pytest.raises(RuntimeError, match="remotely bootstrapped"):
        store.active_watch_codes(root, required=True)
    authority.bootstrap_bf_exit_watch(rows, recent)
    assert store.active_watch_codes(root, required=True) == ["A"]
    assert store.watch_was_ever_published(root)
    assert store.official_watch_for_week(root, "2026-10-19")[0]["code"] == "A"
    with pytest.raises(RuntimeError, match="already enabled"):
        authority.bootstrap_bf_exit_watch(rows, recent)


@pytest.mark.parametrize(("broken_field", "bad_value"), [
    ("phase", "INVALID_PHASE"),
    ("events", "{not-json}"),
    ("events", '{"not": "a list"}'),
    ("events", '[{"type":"POOL_EXIT","week":"2026-10-26"}]'),
    ("events", '[{"id":"BAD","type":"POOL_EXIT","week":"2026-10-26"}]'),
    ("events", '[{"id":"POOL_EXIT:2026-10-26","type":"POOL_EXIT","week":"2026-10-26"},{"id":"POOL_EXIT:2026-10-26","type":"POOL_EXIT","week":"2026-10-26"}]'),
    ("snapshot_date", "2026-10-23"),
    ("phase", "PROVISIONAL_OUT"),
])
def test_authority_rejects_unreadable_watch_before_any_publication(
    tmp_path, monkeypatch, broken_field, bad_value,
):
    import yfinance_data as authority
    root = tmp_path / "authority"
    (root / "us").mkdir(parents=True)
    monkeypatch.setattr(authority, "DATA_ROOT", str(root))
    pool = pd.DataFrame([{"code": "A", "snapshot_date": "2026-10-30"}])
    when, digest = store.pool_identity(pool)
    rows = [{"code": "A", "phase": "OBSERVING", "events": []}]
    companion = store.watch_frame(rows, snapshot_date=when, pool_codes_hash=digest)
    companion.loc[0, broken_field] = bad_value
    staged = root / "us" / ".bf_pool_exit_watch.test.pending"
    companion.to_csv(staged, index=False, encoding="utf-8-sig")
    target = root / store.WATCH_COMPLETE

    run = authority.BreakoutFollowPoolRun.weekend()
    monkeypatch.setattr(run, "ensure_current_snapshot", lambda: pool)
    with pytest.raises(ValueError):
        run.register_companion(
            target_path=str(target), staged_path=str(staged),
            source_pool_date=when, source_pool_codes_hash=digest)
    assert run._watch_companion is None
    assert not target.exists()


@pytest.mark.parametrize("corruption", [
    "missing_exit", "foreign_code", "wrong_source_identity",
    "provisional_phase", "bad_event",
])
def test_bootstrap_rejects_forged_or_unreadable_initial_watch(
    tmp_path, monkeypatch, corruption,
):
    import yfinance_data as authority
    root = setup_repo(tmp_path)
    commit_pool(root, "2026-10-16", ["A", "B"])
    commit_pool(root, "2026-10-23", ["B"])
    setup_bare_remote(root, tmp_path)
    monkeypatch.setattr(authority, "DATA_ROOT", str(root))
    recent = store.published_complete_pools(root, limit=2)[0]
    rows = [{"code": "A", "phase": "AWAITING_DATA", "events": []}]
    if corruption == "missing_exit":
        rows = []
    elif corruption == "foreign_code":
        rows[0]["code"] = "C"
    elif corruption == "wrong_source_identity":
        recent = {**recent, "rows": [
            {"code": "C", "snapshot_date": recent["snapshot_date"]}]}
    elif corruption == "provisional_phase":
        rows[0]["phase"] = "PROVISIONAL_OUT"
    elif corruption == "bad_event":
        rows[0]["events"] = [{"type": "POOL_EXIT", "week": "2026-10-19"}]

    initial = git(root, "rev-parse", "origin/main")
    with pytest.raises((ValueError, RuntimeError)):
        authority.bootstrap_bf_exit_watch(rows, recent)
    assert git(root, "rev-parse", "origin/main") == initial
    assert not (root / store.WATCH_COMPLETE).exists()
    assert not list((root / "us").glob("*.pending"))



def test_download_rejects_remote_pool_only_advance_despite_identical_watch_blob(tmp_path):
    """A published Watch file is NOT authentic without the latest Pool pair."""
    root = setup_repo(tmp_path)
    commit_pool(root, "2026-10-23", ["A", "B"])
    old = pd.read_csv(root / "us" / "breakout_follow_pool.csv")
    when, digest = store.pool_identity(old)
    target = root / store.WATCH_COMPLETE
    store.watch_frame([{"code": "X", "phase": "OBSERVING", "events": []}],
                      snapshot_date=when, pool_codes_hash=digest).to_csv(target, index=False)
    git(root, "add", store.WATCH_COMPLETE)
    git(root, "commit", "-qm",
        f"bootstrap\n\nWatch-Source-Date: {when}\nWatch-Codes-SHA256: {digest}")
    git(root, "update-ref", "refs/remotes/origin/main", "HEAD")
    assert store.active_watch_codes(root, required=True) == ["X"]

    # Legacy or foreign publisher updates complete Pool without companion.
    commit_pool(root, "2026-10-30", ["B"])
    assert target.read_bytes() == subprocess.check_output([
        "git", "-C", str(root), "show", f"origin/main:{store.WATCH_COMPLETE}"])
    with pytest.raises(RuntimeError, match="not paired"):
        store.active_watch_codes(root, required=True)


def test_download_rejects_forged_pool_trailer_with_unchanged_nonempty_watch(tmp_path):
    root = setup_repo(tmp_path)
    commit_pool(root, "2026-10-23", ["A", "B"])
    old = pd.read_csv(root / "us" / "breakout_follow_pool.csv")
    when, digest = store.pool_identity(old)
    target = root / store.WATCH_COMPLETE
    store.watch_frame([{"code": "A", "phase": "OBSERVING", "events": []}],
                      snapshot_date=when, pool_codes_hash=digest).to_csv(target, index=False)
    git(root, "add", store.WATCH_COMPLETE)
    git(root, "commit", "-qm",
        f"bootstrap\n\nWatch-Source-Date: {when}\nWatch-Codes-SHA256: {digest}")
    git(root, "update-ref", "refs/remotes/origin/main", "HEAD")
    commit_pool(root, "2026-10-30", ["B"], paired=True)
    # New Pool claims a Watch companion but its rows identify an older Pool.
    with pytest.raises(RuntimeError, match="not paired"):
        store.active_watch_codes(root)


def test_download_accepts_unchanged_empty_watch_when_new_pool_pair_is_signed(tmp_path):
    root = setup_repo(tmp_path)
    commit_pool(root, "2026-10-23", ["B"])
    old = pd.read_csv(root / "us" / "breakout_follow_pool.csv")
    when, digest = store.pool_identity(old)
    target = root / store.WATCH_COMPLETE
    store.watch_frame([], snapshot_date=when, pool_codes_hash=digest).to_csv(target, index=False)
    git(root, "add", store.WATCH_COMPLETE)
    git(root, "commit", "-qm",
        f"bootstrap\n\nWatch-Source-Date: {when}\nWatch-Codes-SHA256: {digest}")
    git(root, "update-ref", "refs/remotes/origin/main", "HEAD")
    commit_pool(root, "2026-10-30", ["B"], paired=True)
    assert store.active_watch_codes(root, required=True) == []
