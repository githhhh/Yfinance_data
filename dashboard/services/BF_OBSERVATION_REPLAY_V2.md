# BF Observation Replay V2 — implementation contract

Scope: read-only BF Review/Telegram observation. Existing `bf_transition._project`,
Pool CSV writer, strategy algorithms, Futu and universe selection are **not**
consumers and must never be modified by Replay.

## Source of truth and timeframe

- Weekend A: the preceding successful COMPLETE-week `signal=True` candidates.
  Existing `_price_path`, `build_observation_view` and
  `build_observation_attention` remain byte-for-byte unchanged.
- Midweek B: **current** `signal=True` Anchor distinct from A. It is never an
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

1. `identity_v2`: exact Pivot structure
   `pivot|resistance_date|selected_BOX_type`; selected candidate must
   agree with Trigger, or one unambiguous matching candidate must exist.
   Other rules reuse the existing dated v1 resolver where supported.
   Price is a Trigger **version**, not an identity. Ambiguity rejects B.
2. `price_replay`: published-boundary `BELOW/ZONE/EXT` comparison.
   `TRACKED_ANCHOR` when the Anchor was observed historically;
   `RETROSPECTIVE` for a newly observed Anchor projected against earlier
   real EOD prices. Daily path is separate from notification delta.
   Trigger revision that changes the classification of either endpoint
   suppresses the supposed price-only crossing.
3. `watch_provenance`: preserves dated earlier `bf_watch_active=True`
   observations, distinguishing Weekend and Midweek sources.
   `TICKER_CONTEXT` **does not claim causal same-structure confirmation**.
4. `entry_confirmation`: ACT iff current resolver validity is true,
   status is ACTIONABLE and current price is in ZONE. Missing data is not
   equivalent to UNCONFIRMED.

Facts combine into exactly one B opportunity: `ZONE_REENTRY` for a verified
existing B Anchor crossing BELOW/EXT → ZONE; `FIRST_OBSERVED_BP` for a
new-to-published-history B in ZONE. A reappearing identity is not NEW. ACT
upgrade is recorded but cannot alone trigger a Telegram event.

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
is unchanged; B second line is the price crossing, optional WATCH, and
optional ACT. Neither WATCH nor ZONE implies ACT.

## P0: local LIVE preflight

From a trusted local checkout (Git history must be present):

```sh
python dashboard/replay_preflight.py --as-of YYYY-MM-DD --root /absolute/path/to/Yfinance_data
```

Return code 0 means Git and COMPLETE-baseline evidence are sufficient for
that `as_of`; code 2 indicates failed preflight. For the **first** midweek
session, zero earlier MIDWEEK commits is legitimate (the complete-week Pool
is the baseline). This check does not prove future LIVE executions will
always have complete history or exactly-once Telegram delivery.

## Required audits

- Identity: ANDG reappearance; KO S_BOX→M_BOX despite shared date; CBLL
  selected M among M/S; AMN same M with minor Trigger update; ambiguous
  dual-box must fail closed.
- Price: AOUT retrospective BELOW→ZONE, old B EXT→ZONE,
  current Trigger revision without invented crossing, 0/5% boundaries,
  PKL-vs-Pool mismatch, date clipping.
- Watch: active only; Weekend/Midweek source, time before B, cross-rule
  only TICKER_CONTEXT.
- Compatibility: AST-equivalence of all preexisting A functions vs main;
  existing Dashboard regression; quant_trade v1/v2 formatter and bridge.
- No merge until data repo CI and quant_trade contract CI both succeed.
  Promote data repo first; quant_trade second. Production P0 preflight
  is an explicit deployment gate, not a claim already completed here.
