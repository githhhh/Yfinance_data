#!/usr/bin/env bash
# Run the same screener + daily + weekly cache generation as data-update.yml,
# but obtain OHLCV locally from the selected provider.  Schwab OAuth material
# stays local; this entry never publishes a branch or writes Git history.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PROVIDER="${1:-}"

case "$PROVIDER" in
    yahoo|schwab) ;;
    *)
        echo "Usage: $0 {yahoo|schwab}" >&2
        exit 2
        ;;
esac

if [ "${YFINANCE_DATA_LOCAL_UPDATE_SOURCE_ONLY:-}" = "1" ]; then
    return 0 2>/dev/null || exit 0
fi

cd "$DATA_REPO_ROOT"

echo "[LocalDataUpdate] provider=$PROVIDER: running screener + 2Y daily download"
python DataStore.py --provider="$PROVIDER" --period=2y --interval=1d

echo "[LocalDataUpdate] provider=$PROVIDER: running 5Y weekly download"
python DataStore.py --provider="$PROVIDER" --period=5y --interval=1wk --skip-screener

echo "[LocalDataUpdate] validated local PKL snapshots are ready for the strategy run"
