# BreakoutFollow Transition Consumer Semantics

This module has three consumers with intentionally different time semantics. Keep these boundaries stable when changing transition logic.

> **Dashboard 固定看“本周累计变化”；Push 看“今天新发生的变化”；Futu 看“现在可操作的标的”。**

## Dashboard

Dashboard comparison is always:

```text
complete-week pool -> current midweek pool
```

It answers: **what has changed so far this review week relative to the complete-week baseline?**

`previous_midweek` must never replace the complete-week baseline for Dashboard rows, review groups, labels, priority, or summary.

## Push / analysis repo

Push comparison answers: **what changed since the last valid observation?**

The baseline is selected by snapshot chronology:

- no previous midweek -> `complete -> current`
- `complete.snapshot_date >= previous_midweek.snapshot_date` -> `complete -> current` (new review-week reset)
- `previous_midweek.snapshot_date > complete.snapshot_date` -> `previous_midweek -> current`

When previous midweek is selected, compare **effective states**, not raw CSV `ibd_entry_status`. Previous and current midweek pools are both resolved through the same complete-week Carry semantics before comparison.

If snapshot chronology is missing, ambiguous, or current is not newer than the selected baseline, Push should fail closed and emit no events.

## Futu

Futu does not use comparison history. It consumes only the current effective ACTIONABLE set:

```text
current effective state -> actionable_codes
```

Therefore Futu answers only: **what is actionable now?**

## Design invariant

Do not collapse these three semantics into one timeline:

```text
Dashboard = complete -> current      # weekly cumulative change
Push      = selected baseline -> current  # new daily change
Futu      = current only             # current actionable state
```

The comparison algorithm may be shared, but the consumer baseline semantics must remain separate.
