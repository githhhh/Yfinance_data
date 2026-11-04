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



def test_midweek_stale_or_corrupt_uses_latest_valid_complete_snapshot(tmp_path):
    from dashboard.build_static import build_watch_payload
    from pool_exit_watch_store import WATCH_COMPLETE, WATCH_MIDWEEK, watch_frame
    root = tmp_path
    (root / "us").mkdir()
    complete = root / WATCH_COMPLETE
    midweek = root / WATCH_MIDWEEK
    watch_frame([{"code": "A", "phase": "OBSERVING", "events": []}],
                snapshot_date="2026-10-30", pool_codes_hash="complete").to_csv(
                    complete, index=False)
    watch_frame([{"code": "B", "phase": "PROVISIONAL_OUT", "events": []}],
                snapshot_date="2026-10-28", pool_codes_hash="midweek").to_csv(
                    midweek, index=False)
    result = build_watch_payload(root)
    assert result["available"] is True
    assert result["midweek_fallback"] is True
    assert result["complete"][0]["code"] == "A"

    midweek.write_text("bad,columns\n1,2\n", encoding="utf-8")
    result = build_watch_payload(root)
    assert result["available"] is True
    assert result["midweek_fallback"] is True
    assert result["errors"]["midweek"] == "ValueError"
    assert result["complete"][0]["code"] == "A"

    watch_frame([{"code": "C", "phase": "PROVISIONAL_OUT", "events": []}],
                snapshot_date="2026-11-04", pool_codes_hash="midweek").to_csv(
                    midweek, index=False)
    fresh = build_watch_payload(root)
    assert fresh["midweek_fallback"] is False
    assert fresh["midweek"][0]["code"] == "C"

    # Missing/invalid complete authority may not elevate Midweek to a
    # standalone official observation view.
    complete.write_text("broken,columns\n1,2\n", encoding="utf-8")
    unavailable = build_watch_payload(root)
    assert unavailable["available"] is False
    assert unavailable["midweek"] == []
    assert unavailable["errors"]["complete"] == "ValueError"


def test_midweek_page_explicitly_labels_complete_fallback():
    root = Path(__file__).parents[1]
    source = (root / "pool-exit-watch.html").read_text(encoding="utf-8")
    assert "midweek_fallback" in source
    assert "Midweek unavailable/stale" in source
