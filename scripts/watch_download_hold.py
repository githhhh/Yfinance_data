"""Manual entry point: query, hold or resume one published BF Watch Cycle."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yfinance_data import get_download_hold, set_download_hold


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage one official BF Watch download hold")
    parser.add_argument("--code", required=True)
    parser.add_argument("--cycle-id", help="Required expected Cycle identity for writes")
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--query", action="store_true")
    actions.add_argument("--hold", metavar="REASON")
    actions.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.query:
        print(json.dumps(get_download_hold(args.code), ensure_ascii=False))
        return
    if not args.cycle_id:
        parser.error("--cycle-id is required for hold/resume")
    set_download_hold(args.code, args.cycle_id, "" if args.resume else args.hold)
    print(json.dumps(get_download_hold(args.code), ensure_ascii=False))


if __name__ == "__main__":
    main()
