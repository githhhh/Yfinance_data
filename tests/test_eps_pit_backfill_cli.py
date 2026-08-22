import json

from tools.backfill_eps_pit import clean_output_dir


def test_clean_output_dir_removes_stale_eps_artifacts_and_writes_log(tmp_path):
    output = tmp_path / "eps_output"
    stale = output / "cache" / "old.parquet"
    stale.parent.mkdir(parents=True)
    stale.write_text("stale", encoding="utf-8")

    log = clean_output_dir(str(output), "test clean rebuild")

    assert not stale.exists()
    assert log["removed_path_count"] >= 1
    assert "cache/old.parquet" in log["removed_paths"]

    log_path = output / "audit" / "clean_rebuild_log.json"
    written = json.loads(log_path.read_text(encoding="utf-8"))
    assert written["reason"] == "test clean rebuild"
