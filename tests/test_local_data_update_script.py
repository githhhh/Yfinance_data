import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "run_local_data_update.sh"


def test_local_data_update_runner_has_valid_shell_syntax():
    subprocess.run(["/bin/bash", "-n", str(RUNNER)], check=True)


def test_local_data_update_runner_mirrors_existing_update_flow():
    source = RUNNER.read_text(encoding="utf-8")

    assert "git pull --ff-only origin main" in source
    assert 'DataStore.py --screener-only --min-eps-growth=150' in source
    assert 'DataStore.py --provider="$PROVIDER" --period=2y --interval=1d --skip-screener' in source
    assert 'DataStore.py --provider="$PROVIDER" --period=5y --interval=1wk --skip-screener' in source
    assert 'if [ "$PROVIDER" = "yahoo" ]; then' in source
    assert 'find results_pkl -name "*.pkl" ! -name "*${TODAY}*" -delete' in source
    assert "git add results_pkl/ us/" in source
    assert 'git commit -m "Update stock data [skip ci]"' in source
    assert "git push origin HEAD:main" in source


def test_local_data_update_syncs_first_and_publishes_only_after_validated_cleanup():
    source = RUNNER.read_text(encoding="utf-8")

    sync = source.index("git pull --ff-only origin main")
    screener = source.index("DataStore.py --screener-only --min-eps-growth=150")
    daily = source.index('DataStore.py --provider="$PROVIDER" --period=2y --interval=1d --skip-screener')
    weekly = source.index('DataStore.py --provider="$PROVIDER" --period=5y --interval=1wk --skip-screener')
    cleanup = source.index('find results_pkl -name "*.pkl" ! -name "*${TODAY}*" -delete')
    publish = source.index("git add results_pkl/ us/")

    assert sync < screener < daily < weekly < cleanup < publish
