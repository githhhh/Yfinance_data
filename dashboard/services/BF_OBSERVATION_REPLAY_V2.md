# BF Observation Replay V2 — implementation contract

Scope: BF Review/Telegram observation and the read-only Midweek projection.
The selected price is the sole buy-point identity in both
`bf_transition._project` and V2 replay. Strategy algorithms, Pool CSV writer,
Futu trade eligibility and universe selection are not changed.

## Source of truth and timeframe

- Weekend A: the preceding successful COMPLETE-week `signal=True` candidates.
  Its fixed price tracking and event comparison remain independent of B.
- Midweek B: **current** `signal=True` selected buy-point **price** distinct from A. It is never an
  automatically persistent tracking universe.
- Git provides earlier committed **full MIDWEEK Pool snapshots** for the same
  review week. The capture of the previous file is accepted as a published
  comparison boundary only if it matches the newest dated committed snapshot.
- The COMPLETE Pool is a preceding-week observation even when its ticker
  is absent. Absence does not create an invented signal. Daily 1D PKL may fill
  price evidence, but never a historical signal or WATCH.
- No git fetch, network requests, database writes or side effects. If Git is
  shallow, prior publication is inconsistent, or a market date has conflicting
  revisions: B V2 fails closed, A v1 remains available.
- `run_cutoff` is timezone-aware. Commits published after the replay cutoff
  are ignored. Snapshot trading day and commit publication timestamp are
  different facts; later corrections must not contaminate earlier replays.
- `FIRST_OBSERVED` means first sighting **among available verified published
  snapshots**, not the real-world first market appearance.

## Four independent facts

`dashboard/services/bf_observation_replay.py` contains pure functions:

1. `identity_v2`: selected-price identity `price|<trigger>`.
   Strategy/rule, S_BOX/M_BOX, resistance date and other metadata remain
   explanatory evidence, never identity or a prerequisite for buy-point review.
   The Pool provides one selected price per ticker/snapshot; equality is exact.
2. `price_replay`: published-boundary `BELOW/ZONE/EXT` comparison.
   `TRACKED_ANCHOR` when the Anchor was observed historically;
   `RETROSPECTIVE` for a newly observed Anchor projected against earlier
   real EOD prices. Daily path is separate from notification delta.
   Trigger revision that changes the classification of either endpoint
   suppresses the supposed price-only crossing.
3. `watch_provenance`: preserves dated earlier `bf_watch_active=True`
   observations, distinguishing Weekend and Midweek sources.
   `active_prior` (ever seen earlier) differs from `active_at_price_boundary`
   (Watch active on the prior price observation date); only the latter
   authorizes `BELOW/EXT • WATCH → ZONE`. `TICKER_CONTEXT` **does not claim
   causal same-structure confirmation**.
4. `entry_confirmation`: ACT iff current resolver validity is true,
   status is ACTIONABLE and current price is in ZONE. Missing data is not
   equivalent to UNCONFIRMED.

`resolve_midweek_facts()` returns independent historical facts even when the
current B is outside ZONE. `midweek_opportunity_policy()` alone decides
Telegram eligibility; the compatibility façade only composes them. The legacy
A `_price_path` remains untouched; both A and B use authoritative
`bf_transition._price_zone` rounding through their adapters. This is a staged compatibility boundary, not a claim that A's whole timeline
is reimplemented. `UNVERIFIED` is not an Entry Status: valid selected prices
are never blocked by unclear strategy or structural metadata.

Facts combine into at most one B opportunity: `ZONE_REENTRY` for a
previously observed selected-price B crossing BELOW/EXT → ZONE;
`FIRST_OBSERVED_BP` for a new-to-published-history selected price in ZONE.
The same price reappearing is not NEW. ACT upgrade alone does not create an
event. V2 retains legacy JSON keys `anchor_key_v2` / `identity` for consumer
compatibility, but the key's value is now derived exclusively from price.

The schema-v2 event contains `source=MIDWEEK`, `anchor_key_v2`, `identity`,
`price`, `watch`, `entry`, `primary`, `modifiers`, `evidence`,
`event_key` (SHA256 over JSON tuple) and `schema_version=2`.
`event_key` is deterministic **event identity**, not exactly-once Telegram
delivery.

## Consumer and backward compatibility

The `quant_trade` bridge emits existing `observation_events` unchanged for
Weekend A, plus `observation_events_v2` for B and
`observation_replay_preflight` diagnostics. V2 input takes precedence for B.
If V2 exists but returns no B, never fall back to the legacy B NEW alert.
An older data repo without V2 still uses the previous v1 formatter.

Telegram displays full `Weekend Buy Point` and `Midweek Buy Point` headings,
no emojis, +xx.xx%, descending RS and max 3 per group. Its Weekend A display
is unchanged. Only an established-price ZONE_REENTRY B is labeled as an existing Anchor's
BELOW/EXT → ZONE price crossing. FIRST_OBSERVED_BP instead displays
WATCH → BUY POINT or NEW BUY POINT; its RETROSPECTIVE daily price path is
evidence, not a historical B crossing. A reentry without boundary crossing is
suppressed, not mislabeled as NEW. ACT remains independently confirmed;
neither WATCH nor ZONE implies ACT.

## P0: local LIVE preflight

From a trusted local checkout (Git history must be present):

```sh
python dashboard/replay_preflight.py --as-of YYYY-MM-DD --root /absolute/path/to/Yfinance_data
```

Return code 0 means Git and COMPLETE-baseline evidence are sufficient for
that `as_of`; code 2 indicates failed preflight. Standalone preflight passes
`require_previous=False` because it has no run-scoped Previous capture. The
actual LIVE bridge retains strict `require_previous=True` verification. A
preflight passing must never be misreported as proof of LIVE capture integrity. For the **first** midweek
session, zero earlier MIDWEEK commits is legitimate (the complete-week Pool
is the baseline). This check does not prove future LIVE executions will
always have complete history or exactly-once Telegram delivery.

## Required audits

- Identity: ANDG reappearance; KO differing prices; CBLL same price across
  M/S; AMN changed Trigger with unchanged structure; ambiguous dual-box
  metadata must not invalidate a valid selected price.
- Price: AOUT retrospective BELOW→ZONE, old B EXT→ZONE,
  current Trigger revision without invented crossing, 0/5% boundaries,
  PKL-vs-Pool mismatch, date clipping.
- Watch: active only; Weekend/Midweek source, time before B, cross-rule
  only TICKER_CONTEXT.
- Compatibility: independently tracked Weekend A price/status,
  Dashboard regression; quant_trade v1/v2 formatter and bridge.
- No merge until data repo CI and quant_trade contract CI both succeed.
  Promote data repo first; quant_trade second. Production P0 preflight
  is an explicit deployment gate, not a claim already completed here.
