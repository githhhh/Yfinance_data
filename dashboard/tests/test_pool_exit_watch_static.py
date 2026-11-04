from pathlib import Path

def test_watch_html_uses_separate_payload_and_has_core_labels():
    path=Path(__file__).parents[1]/"pool-exit-watch.html"
    source=path.read_text(encoding="utf-8")
    assert "pool_exit_watch.json" in source
    assert "CEILING" in source and "MA" in source
    assert "events" in source
