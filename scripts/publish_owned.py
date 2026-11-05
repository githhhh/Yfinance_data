"""Publish explicit data artifacts through the shared locked Git writer.

Never run a direct git add/commit/push from data-update workflows: it can push
an earlier unverified Pool/Watch commit along with unrelated new data.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yfinance_data import _commit_managed_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--message", required=True)
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    _commit_managed_csv(args.paths, message=args.message)


if __name__ == "__main__":
    main()
