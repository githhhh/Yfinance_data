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


def test_scheduled_rs_refresh_cannot_replace_pages_with_stale_or_missing_data() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    # The actual eligibility logic is unit-tested in test_rs_reference.py;
    # workflow wiring must call it and gate both artifact upload and deploy.
    assert "scheduled_refresh_publishable" in text
    assert "latest_completed_market_date" in text
    assert "steps.static_build.outputs.publish == 'true'" in text
    assert "needs.build.outputs.publish == 'true'" in text
