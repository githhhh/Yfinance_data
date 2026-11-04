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


def commit_pool(root, d, symbols):
    pd.DataFrame([{"code":code,"snapshot_date":d,"ceiling":100,"signal":False}
                  for code in symbols]).to_csv(root/"us"/"breakout_follow_pool.csv",index=False)
    git(root,"add","us/breakout_follow_pool.csv")
    git(root,"commit","-qm","pool")
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
    git(root,"commit","-qm","bootstrap empty")
    git(root,"update-ref","refs/remotes/origin/main","HEAD")
    assert store.official_watch_for_week(root,"2026-10-20") is None
    assert store.official_watch_for_week(root,"2026-10-19")==[]


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
