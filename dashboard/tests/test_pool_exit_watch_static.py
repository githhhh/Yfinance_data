from pathlib import Path

def test_watch_html_uses_separate_payload_and_has_core_labels():
    path=Path(__file__).parents[1]/"pool-exit-watch.html"
    source=path.read_text(encoding="utf-8")
    assert "pool_exit_watch.json" in source
    assert "CEILING" in source and "MA" in source
    assert "events" in source


def test_details_include_current_and_frozen_ceiling():
    root=Path(__file__).parents[1]
    page=(root/'pool-exit-watch.html').read_text(encoding='utf-8')
    builder=(root/'build_static.py').read_text(encoding='utf-8')
    assert 'Old Ceiling:' in page and 'Current Ceiling:' in page
    assert '"current_ceiling"' in builder



def _source_pool(root, *, midweek: bool, when: str, code: str = "SOURCE") -> str:
    """Provide a real Pool identity; Watch rows are not authoritative dates."""
    import pandas as pd
    from pool_exit_watch_store import pool_identity

    frame = pd.DataFrame([{"code": code, "snapshot_date": when}])
    name = "breakout_follow_pool_midweek.csv" if midweek else "breakout_follow_pool.csv"
    frame.to_csv(root / "us" / name, index=False)
    return pool_identity(frame)[1]


def test_midweek_stale_or_corrupt_uses_latest_valid_complete_snapshot(tmp_path):
    from dashboard.build_static import build_watch_payload
    from pool_exit_watch_store import WATCH_COMPLETE, WATCH_MIDWEEK, watch_frame
    root = tmp_path
    (root / "us").mkdir()
    complete = root / WATCH_COMPLETE
    midweek = root / WATCH_MIDWEEK
    complete_hash = _source_pool(root, midweek=False, when="2026-10-30")
    old_hash = _source_pool(root, midweek=True, when="2026-10-28")
    watch_frame([{"code": "A", "phase": "OBSERVING", "events": []}],
                snapshot_date="2026-10-30", pool_codes_hash=complete_hash).to_csv(
                    complete, index=False)
    watch_frame([{"code": "B", "phase": "PROVISIONAL_OUT", "events": []}],
                snapshot_date="2026-10-28", pool_codes_hash=old_hash).to_csv(
                    midweek, index=False)
    result = build_watch_payload(root)
    assert result["available"] is True
    assert result["midweek_fallback"] is True
    assert result["complete"][0]["code"] == "A"

    midweek.write_text("bad,columns\\n1,2\\n", encoding="utf-8")
    result = build_watch_payload(root)
    assert result["available"] is True
    assert result["midweek_fallback"] is True
    assert result["errors"]["midweek"] == "ValueError"
    assert result["complete"][0]["code"] == "A"

    fresh_hash = _source_pool(root, midweek=True, when="2026-11-04")
    watch_frame([{"code": "C", "phase": "PROVISIONAL_OUT", "events": []}],
                snapshot_date="2026-11-04", pool_codes_hash=fresh_hash).to_csv(
                    midweek, index=False)
    fresh = build_watch_payload(root)
    assert fresh["midweek_fallback"] is False
    assert fresh["midweek"][0]["code"] == "C"

    # Missing/invalid complete authority may not elevate Midweek to a
    # standalone official observation view.
    complete.write_text("broken,columns\\n1,2\\n", encoding="utf-8")
    unavailable = build_watch_payload(root)
    assert unavailable["available"] is False
    assert unavailable["midweek"] == []
    assert unavailable["errors"]["complete"] == "ValueError"


def test_empty_complete_watch_does_not_resurrect_old_midweek(tmp_path):
    from dashboard.build_static import build_watch_payload
    from pool_exit_watch_store import WATCH_COMPLETE, WATCH_MIDWEEK, watch_frame

    (tmp_path / "us").mkdir()
    complete_hash = _source_pool(tmp_path, midweek=False, when="2026-11-06")
    midweek_hash = _source_pool(tmp_path, midweek=True, when="2026-11-04")
    watch_frame([], snapshot_date="2026-11-06", pool_codes_hash=complete_hash).to_csv(
        tmp_path / WATCH_COMPLETE, index=False)
    watch_frame([{"code": "OLD", "phase": "PROVISIONAL_OUT", "events": []}],
                snapshot_date="2026-11-04", pool_codes_hash=midweek_hash).to_csv(
                    tmp_path / WATCH_MIDWEEK, index=False)
    result = build_watch_payload(tmp_path)
    assert result["available"] is True
    assert result["complete"] == []
    assert result["dates"]["complete"] == "2026-11-06"
    assert result["midweek_fallback"] is True


def test_current_empty_midweek_keeps_its_pool_date_without_fallback(tmp_path):
    from dashboard.build_static import build_watch_payload
    from pool_exit_watch_store import WATCH_COMPLETE, WATCH_MIDWEEK, watch_frame

    (tmp_path / "us").mkdir()
    complete_hash = _source_pool(tmp_path, midweek=False, when="2026-11-06")
    midweek_hash = _source_pool(tmp_path, midweek=True, when="2026-11-11")
    watch_frame([{"code": "A", "phase": "OBSERVING", "events": []}],
                snapshot_date="2026-11-06", pool_codes_hash=complete_hash).to_csv(
                    tmp_path / WATCH_COMPLETE, index=False)
    watch_frame([], snapshot_date="2026-11-11", pool_codes_hash=midweek_hash).to_csv(
        tmp_path / WATCH_MIDWEEK, index=False)
    result = build_watch_payload(tmp_path)
    assert result["available"] is True
    assert result["dates"] == {"complete": "2026-11-06", "midweek": "2026-11-11"}
    assert result["midweek"] == []
    assert result["midweek_fallback"] is False


def test_watch_midweek_source_must_match_actual_pool(tmp_path):
    from dashboard.build_static import build_watch_payload
    from pool_exit_watch_store import WATCH_COMPLETE, WATCH_MIDWEEK, watch_frame

    (tmp_path / "us").mkdir()
    complete_hash = _source_pool(tmp_path, midweek=False, when="2026-10-30")
    _source_pool(tmp_path, midweek=True, when="2026-11-04")
    watch_frame([], snapshot_date="2026-10-30", pool_codes_hash=complete_hash).to_csv(
        tmp_path / WATCH_COMPLETE, index=False)
    watch_frame([{"code": "STALE", "phase": "PROVISIONAL_OUT", "events": []}],
                snapshot_date="2026-11-04", pool_codes_hash="wrong-digest").to_csv(
                    tmp_path / WATCH_MIDWEEK, index=False)
    result = build_watch_payload(tmp_path)
    assert result["available"] is True
    assert result["midweek_fallback"] is True
    assert result["errors"]["midweek"] == "ValueError"


def test_thursday_complete_on_good_friday_week_is_valid_baseline(tmp_path):
    from dashboard.build_static import build_watch_payload
    from pool_exit_watch_store import WATCH_COMPLETE, WATCH_MIDWEEK, watch_frame

    (tmp_path / "us").mkdir()
    complete_hash = _source_pool(tmp_path, midweek=False, when="2026-04-02")
    midweek_hash = _source_pool(tmp_path, midweek=True, when="2026-04-06")
    watch_frame([], snapshot_date="2026-04-02", pool_codes_hash=complete_hash).to_csv(
        tmp_path / WATCH_COMPLETE, index=False)
    watch_frame([], snapshot_date="2026-04-06", pool_codes_hash=midweek_hash).to_csv(
        tmp_path / WATCH_MIDWEEK, index=False)
    result = build_watch_payload(tmp_path)
    assert result["available"] is True
    assert result["dates"]["complete"] == "2026-04-02"
    assert result["midweek_fallback"] is False


def test_midweek_page_explicitly_labels_complete_fallback():
    root = Path(__file__).parents[1]
    source = (root / "pool-exit-watch.html").read_text(encoding="utf-8")
    assert "midweek_fallback" in source
    assert "Midweek unavailable/stale" in source



def test_static_watch_page_does_not_show_stale_published_watch_after_pool_only_commit(tmp_path):
    import subprocess
    import pandas as pd
    from dashboard.build_static import build_watch_payload
    from pool_exit_watch_store import WATCH_COMPLETE, pool_identity, watch_frame

    root = tmp_path / "checkout"
    (root / "us").mkdir(parents=True)
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True).strip()
    git("init", "-b", "main")
    git("config", "user.email", "watch-page@example.invalid")
    git("config", "user.name", "Watch Page CI")
    first = pd.DataFrame([{"code": "ABC", "snapshot_date": "2026-10-23"}])
    when, digest = pool_identity(first)
    first.to_csv(root / "us" / "breakout_follow_pool.csv", index=False)
    watch_frame([{"code": "X", "phase": "OBSERVING", "events": []}],
                snapshot_date=when, pool_codes_hash=digest).to_csv(
                    root / WATCH_COMPLETE, index=False)
    git("add", ".")
    git("commit", "-m",
        f"initialize\n\nWatch-Source-Date: {when}\nWatch-Codes-SHA256: {digest}")
    git("update-ref", "refs/remotes/origin/main", "HEAD")
    assert build_watch_payload(root)["available"] is True

    newer = pd.DataFrame([{"code": "ABC", "snapshot_date": "2026-10-30"}])
    newer.to_csv(root / "us" / "breakout_follow_pool.csv", index=False)
    git("add", "us/breakout_follow_pool.csv")
    git("commit", "-m", "unpaired Pool-only update")
    git("update-ref", "refs/remotes/origin/main", "HEAD")

    result = build_watch_payload(root)
    assert result["available"] is False
    assert result["complete"] == []
    assert result["midweek"] == []
    assert result["errors"]["complete"] == "RuntimeError"
