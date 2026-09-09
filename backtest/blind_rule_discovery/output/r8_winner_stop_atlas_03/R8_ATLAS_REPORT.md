# R8A Winner / Stop Deterministic Feature Atlas

Known-history retrospective research. NOT AN UNTOUCHED HOLDOUT.
This stage is fully deterministic and makes zero RD-Agent/provider calls.
Primary panel: nonoverlap_w3; all_entries is retained as a sensitivity panel.
Winner, Stop, Unresolved and Ambiguous remain distinct path classes.

## Bound Population

- Samples: 8983; nonoverlap_w3 rows: 6740.
- Snapshot-PIT features: 19.
- Input SHA256: `8bfd4411a7751177955927d8b145f15bdd077f49d26c7ddc6f9d6ef44fa6e83e`.

## Stable Winner / Stop Features — nonoverlap_w3

| feature | label | supported_q | consistency | median matched pct gap | median Cliff delta |
|---|---|---:|---:|---:|---:|
| base_depth_pct | CONSISTENT_STOP_HIGH | 8 | 0.750 | -0.0668 | -0.1717 |
| base_mbox_count | CONSISTENT_STOP_HIGH | 8 | 0.875 | -0.0442 | -0.1408 |
| dist_to_52w_high_pct | CONSISTENT_STOP_HIGH | 8 | 0.750 | -0.1046 | -0.2492 |
| pullback_pct | CONSISTENT_STOP_HIGH | 7 | 0.857 | -0.0843 | -0.2366 |

Labels require >=6 supported outer quarters, >=75% same matched-week direction, |median Cliff delta|>=0.10, aligned median effects and leave-one-quarter sign stability.
They are descriptive facts, not alpha certification or production weights.

## Outputs

class_profiles.csv and quintile_surfaces.csv contain both all_entries and nonoverlap_w3 panels.
quarter_feature_contrasts.csv and feature_stability.csv likewise retain both panels.
No feature was prefiltered or omitted because of weak results.

KEEP PRODUCTION FROZEN
