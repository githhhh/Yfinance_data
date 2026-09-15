import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "run_local_data_update.sh"


def test_local_data_update_runner_has_valid_shell_syntax():
    subprocess.run(["/bin/bash", "-n", str(RUNNER)], check=True)


def test_local_data_update_runner_keeps_schwab_tokens_and_outputs_local():
    source = RUNNER.read_text(encoding="utf-8")

    assert 'DataStore.py --provider="$PROVIDER" --period=2y --interval=1d' in source
    assert 'DataStore.py --provider="$PROVIDER" --period=5y --interval=1wk --skip-screener' in source
    assert "git push" not in source
    assert "git commit" not in source
