# Derived Dimension Proposals

Only deterministic, PIT-safe dimensions were used:

| Dimension | Formula | Unit | Applicability | Rationale |
|---|---|---:|---|---|
| `trigger_pos` | `ibd_entry_close_position - ibd_entry_breakout_range_ratio` | K-line fraction | rows with both fields known | Avoids repeated geometry guessing and checks trigger location explicitly. |
| `fresh_distance_score` | piecewise decay from `current_vs_ibd_candidate_pct`, best inside 0-2%, penalty above 5% or below 0 | score | all signal rows | Tests Fresh Zone as continuous risk rather than fixed hard band. |
| `entry_volume_saturation_score` | `log1p(volume_ratio - 1.5) + 1`, capped | score | rows with entry volume | Tests volume as non-linear evidence; avoids assuming more volume is always linearly better. |
| `base_pullback_context_score` | small bounded score from base depth/mbox and pullback depth | score | route-applicable only | Tests whether unused schema context adds value without becoming a hidden hard gate. |
