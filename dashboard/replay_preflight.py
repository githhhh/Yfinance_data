"""Read-only diagnostics for BF Observation Replay V2 in the LIVE checkout.

Usage:
  python dashboard/replay_preflight.py --as-of 2026-10-15
  python dashboard/replay_preflight.py --as-of 2026-10-15 --root /path/to/Yfinance_data
"""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from bf_snapshot import is_valid_complete_baseline
from dashboard.services.bf_observation import load_observation_history


def report(root: Path, as_of: date) -> dict:
    complete = root / "us" / "breakout_follow_pool.csv"
    if not complete.is_file():
        return {"usable": False, "reason": "complete Pool not found", "as_of": as_of.isoformat()}
    try:
        frame = pd.read_csv(complete, dtype={"code": str})
        dates = pd.to_datetime(frame["snapshot_date"], errors="coerce").dropna().dt.date.unique()
        if len(dates) != 1 or not is_valid_complete_baseline(dates[0], as_of):
            return {"usable": False, "reason": "invalid complete-week baseline",
                    "as_of": as_of.isoformat()}
        snapshots, diagnostics = load_observation_history(root, as_of=as_of)
        return {"as_of": as_of.isoformat(), "complete_date": dates[0].isoformat(),
                "previous_midweek_dates": [x["date"] for x in snapshots],
                **diagnostics}
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        return {"usable": False, "reason": str(exc), "as_of": as_of.isoformat()}


def main() -> int:
    p = argparse.ArgumentParser(description="BF Replay V2 local read-only preflight")
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--as-of", type=date.fromisoformat, required=True,
                   help="US EOD market date, YYYY-MM-DD")
    args = p.parse_args()
    result = report(args.root.resolve(), args.as_of)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result.get("usable") else 2


if __name__ == "__main__":
    raise SystemExit(main())
