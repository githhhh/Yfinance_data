#!/usr/bin/env bash
# Mirror the existing data-update.yml flow locally while allowing the OHLCV
# provider to be selected. Provider-specific throttling/auth stays in the
# provider implementation; validated data artifacts follow the same Git
# publication lifecycle as the existing Yahoo workflow.

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

# Formal local updates must run from a checkout that can fast-forward to the
# authoritative main branch. Local developer changes/divergence fail closed.
git pull --ff-only origin main
mkdir -p results_pkl

echo "[LocalDataUpdate] running screener and merge"
if ! python DataStore.py --screener-only --min-eps-growth=150; then
    echo "[LocalDataUpdate] warning: screener failed; preserving existing data-update.yml continue-on-error semantics" >&2
fi

echo "[LocalDataUpdate] provider=$PROVIDER: running 2Y daily download"
python DataStore.py --provider="$PROVIDER" --period=2y --interval=1d --skip-screener

# The remote Yahoo workflow has an explicit 180~300s cooldown between daily and
# weekly downloads. Keep that behavior only when this local runner is explicitly
# used with Yahoo; Schwab pacing is owned by SchwabDataProvider.
if [ "$PROVIDER" = "yahoo" ]; then
    SLEEP_TIME=$((RANDOM % 121 + 180))
    echo "[LocalDataUpdate] waiting ${SLEEP_TIME}s for Yahoo Finance rate-limit cooldown"
    sleep "$SLEEP_TIME"
fi

echo "[LocalDataUpdate] provider=$PROVIDER: running 5Y weekly download"
python DataStore.py --provider="$PROVIDER" --period=5y --interval=1wk --skip-screener

TODAY=$(date +"%d%m%y")
echo "[LocalDataUpdate] keeping PKL files with suffix: $TODAY"
find results_pkl -name "*.pkl" ! -name "*${TODAY}*" -delete

echo "[LocalDataUpdate] publishing validated data artifacts"
git add results_pkl/ us/
if git diff --cached --quiet; then
    echo "[LocalDataUpdate] no data changes to commit"
else
    git commit -m "Update stock data [skip ci]"
    git push origin HEAD:main
fi

echo "[LocalDataUpdate] validated PKL snapshots are ready for the strategy run"
