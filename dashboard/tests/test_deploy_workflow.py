from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = PROJECT_ROOT / ".github" / "workflows" / "deploy-review-dashboard.yml"


def test_dashboard_has_post_rs_refresh_schedule() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    # Pool pushes remain the authority. The scheduled run only rebuilds the
    # already-published Pool after the external RS publication window.
    assert "schedule:" in text
    assert "cron: '0 3 * * 4,6'" in text
    assert "python dashboard/build_static.py --output _site" in text
    assert "RS_GITHUB_TOKEN" not in text
