from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local breakout pool dashboard.")
    default_csv = Path(__file__).resolve().parents[1] / "us" / "breakout_follow_pool.csv"
    parser.add_argument("--csv", default=str(default_csv))
    parser.add_argument("--server-port", default="8501")
    args = parser.parse_args()

    app_dir = Path(__file__).resolve().parent
    app_path = app_dir / "app.py"
    csv_path = Path(args.csv).expanduser()
    if not csv_path.is_absolute():
        csv_path = (Path.cwd() / csv_path).resolve()
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path.name),
        "--server.port",
        str(args.server_port),
        "--",
        "--csv",
        str(csv_path),
    ]
    return subprocess.call(command, cwd=app_dir)


if __name__ == "__main__":
    raise SystemExit(main())
