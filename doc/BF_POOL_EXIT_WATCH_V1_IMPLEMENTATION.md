# BF Pool Exit Watch V1 — download hold extension

Official complete Watch is the sole authority for download hold decisions. Watch only
adds codes when an official Cycle is AWAITING_DATA or OBSERVING. The final 1D/1W
provider list subtracts all codes with nonempty download_hold_reason, even when
other universe sources or a Stage2 fallback independently include that code.

The published CSV is exactly 24 columns. Its four deleted columns are
source_pool_date, source_pool_codes_hash, breakout_evidence and exit_count.
The added column is download_hold_reason. Both tracked bootstrap CSVs are
migrated once; **old 27-column CSV is intentionally unsupported**. Historical Git
objects are not rewritten. The Pool/Watch pair still uses the existing
Watch-Source-Date and Watch-Codes-SHA256 trailers on each authoritative Watch
commit (including Watch-only manual edits).

Manual CLI: \`python scripts/watch_download_hold.py --code ABC --query\`, then
\`--code ABC --cycle-id ABC:2026-10-26 --hold REQUEST_ERROR\` or \`--resume\`.
Only an existing official active Cycle can be modified. Neither the engine
nor the midweek projection edits manual holds.

At complete Watch companion registration, the data authority reads the latest
published paired Git Watch and merges only the hold field for matching active
code/cycle identities (empty restore values override stale draft values). It
records the published Watch file version alongside the existing staged digest.
At commit, under the existing managed Git lock, the remote is refreshed and
the Watch version is compared **before os.replace**. Changed versions fail with
an explicit rerun requirement. No automatic rebase, merge or repair.

Only manual workflow_dispatch downloads may supply a temporary
\`exclude_tickers\` list to skip an unavailable ticker before its first formal
Watch Cycle. Use this for LXP only if it is still absent from the official
complete Watch. Remove it after the next valid complete Pool diff creates a
real LXP Cycle and then set its manual hold. Never synthesize a Watch Cycle.

### Merge / rollout prerequisite

The CSV migration commit itself must preserve complete Pool trailer identity
so it remains recognizable to official_watch_for_week(). When merging this
PR, use **rebase merge**, preserving that commit's trailers; an ordinary merge
commit or squash commit without these trailers will invalidate latest Watch
pairing. Verify current published Pool identity again before merging. Do not
deploy the strict 24-column reader against an un-migrated 27-column Watch.
