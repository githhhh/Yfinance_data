# B0 vs Recommended Rule Diff

No production Skill edit is recommended from this run.

Reason: the evaluator now marks incomplete 8-week labels as censored, and the sealed holdout has no complete paired 8-week labels with the current price cache. In addition, 12 Yahoo PIT EPS rows have unverified availability because their `effective_date` equals the fiscal period end.

Current machine decision: `Insufficient evidence` for every tested rule family. Keep B0 unchanged.

Future research may continue testing soft handling for `pullback_v_is_dry`, EPS, fresh distance, volume saturation and Geometry, but none should enter production Skill from this run.
