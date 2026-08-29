# EPS Recalibration Research Restatement

- Old EPS baseline ref: 593bd333181da4fe301b3f61397c7bc95ac86ced
- Data revision: EPS_RECALIBRATED_V2
- Rule change: NO
- Production selector change: NO
- Price data change: NO
- Champion reselection: NO

## Candidate / B0 impact

- E0 membership changed candidates: 22
- E0 affected weeks: 10
- B0 selected-count changed weeks: 0
- B0 code-set/order changed weeks: 6
- B0 order-only changed weeks: 2

## Frozen outcome invariants

- signal_daily_prices.parquet SHA256: 1dfced4c23c639478dd42f0188648823f1c2cafea796fc1a043c56d87d55eb4f
- candidate_weekly_outcomes.parquet SHA256: 0d417cc276edf35293f4e2ea8a1dd723839c141d03ca006c276076c1ff006f83
- train_candidate_weekly_outcomes.parquet SHA256: 029f54e725d15046a5ab2902b2b009c6b6d43204823a35c32bcd2b03396dd35c

## Regenerated fixed research outputs

- B0 vs Matched-N Random
- Three-Tier decomposition
- Rank1/Rank2/Rank3 + TopK
- Layer-1 eligibility / industry / ranking decomposition
- Fixed-date contaminated historical validation

## Attribution caveat

All-historical comparisons isolate the EPS data revision most directly. Train/contaminated-validation old-to-new deltas are not attributable solely to EPS recalibration: the legacy baseline used positional week slicing while V2 uses the fixed calendar (Train 2025-10-10..2026-05-22; contaminated validation 2026-05-29..2026-08-07).

## Prior-conclusion restatement

| Prior conclusion | Verdict | Corrected interpretation |
| :--- | :--- | :--- |
| Pure Eligibility | RETAINED | Directionally positive; independent proof remains not demonstrated. |
| ACTIONABLE | RETAINED | Operationally critical gate. |
| Geometry | RETAINED | Quality/sanity filter; independent return alpha not demonstrated. |
| EPS Known | RETAINED | PIT data-quality/completeness gate; independent return alpha not demonstrated. |
| Industry Diversity | RETAINED | Portfolio construction only; robust independent advantage not demonstrated. |
| B0 W4 quality | STRENGTHENED | Promising historical medium-horizon signal, pending virgin forward validation. |
| Fine-rank monotonicity | RETAINED | Not demonstrated; B0 remains a non-monotonic top-bucket selector. |
| R3 vs R2 | WEAKENED | Prior statistical support largely disappeared after recalibration. |
| Top3 vs Top2 / MC3 | WEAKENED | Median W4 contribution remains directional but less stable. |
| Layer2 | RETAINED | No Layer2 rule qualifies. |

The audit intentionally does not search new rules, change selectors, or reselect champions.
