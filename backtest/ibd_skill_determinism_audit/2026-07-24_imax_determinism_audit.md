# 2026-07-24 IMAX Determinism Audit

## Core Conclusion

Skill 的“标准”可以定，但不能只靠自然语言 Skill 文档让不同模型各自推理。只要模型负责重新排序，就会在证据权重、报告展示顺序、最终分组解释上漂移。可复现方案是先生成确定性 artifact，再让模型只解释 artifact：

```text
pool CSV -> deterministic_prescreen artifact -> model renders explanation without reordering
```

`deterministic_prescreen` artifact 中的 `priority_top3`、`actionable_raw_top5`、`alpha_radar_top5`、`non_actionable_alpha_radar_top10` 和 `pullback_scout_top10` 是权威顺序。模型不得替换名单或重新排序，只能解释 `reason_codes`、`risk_codes` 和来源字段。

## Why Gemini / Claude Put IMAX Second

Gemini 和 Claude 都把 IMAX 放第二，说明它们执行的是旧/v1 式排序权重，或接近旧逻辑的自然语言权衡，而不是当前 v2/v3 deterministic replay。2026-07-24 有 `37c8c6a` 与 `6efb5d4` 两个修复 pool commit 版本；两者在 v1 与 v3 的优先顺序一致。

旧/v1 artifact 对 `2026-07-24_37c8c6a` 的优先复核顺序是：

| Rank | Code | Key Evidence |
|---:|---|---|
| 1 | OVV | near buy point, volume confirms breakout, EPS support, weekly volume, near 52w high |
| 2 | IMAX | near buy point, volume confirms breakout, EPS support, weekly volume |
| 3 | BLFS | near buy point, volume confirms breakout, EPS missing, weekly volume |

IMAX 旧逻辑排第二的决定性差异：OVV 有 `near_52w_high`，IMAX 没有。IMAX 的 `dist_to_52w_high_pct=-5.448997384481258`，没有通过 `> -5.0` 的近 52 周高点证据；OVV 为 `-2.2906670793994666`，通过。旧/v1 sort key 在这一层让 OVV 压过 IMAX。

当前 v3 artifact 对同一 pool 的优先复核顺序是：

| Rank | Code | Key Evidence |
|---:|---|---|
| 1 | IMAX | near buy point, volume confirms breakout, EPS support, weekly volume |
| 2 | NWFL | near buy point, volume confirms breakout, EPS support, near 52w high |
| 3 | PKG | near buy point, volume confirms breakout, weekly volume, near 52w high |

v3 继承 v2 主排序：先看 Fresh Demand Alpha 的完整证据链，强调 IMAX 型“近买点 + 强突破日放量 + EPS 支撑 + 周线跟进”，不让单个 `near_52w_high` 证据压过需求扩张。

## Saved Evidence

- v1 deterministic artifact: `backtest/ibd_skill_determinism_audit/2026-07-24_37c8c6a_v1_artifact.json`
- v1 markdown: `backtest/ibd_skill_determinism_audit/2026-07-24_37c8c6a_v1_artifact.md`
- v3 deterministic artifact: `backtest/ibd_skill_determinism_audit/2026-07-24_37c8c6a_v3_artifact.json`
- v3 markdown: `backtest/ibd_skill_determinism_audit/2026-07-24_37c8c6a_v3_artifact.md`
- second commit v1 artifact: `backtest/ibd_skill_determinism_audit/2026-07-24_6efb5d4_v1_artifact.json`
- second commit v1 markdown: `backtest/ibd_skill_determinism_audit/2026-07-24_6efb5d4_v1_artifact.md`
- second commit v3 artifact: `backtest/ibd_skill_determinism_audit/2026-07-24_6efb5d4_v3_artifact.json`
- second commit v3 markdown: `backtest/ibd_skill_determinism_audit/2026-07-24_6efb5d4_v3_artifact.md`

## Skill Improvement

The skill now requires deterministic artifact generation before report rendering. This changes the model's job:

- before: parse CSV, infer facts, rank candidates, write report;
- after: use artifact as authoritative ranking contract, then explain fixed rows.

This is the part that makes cross-model output stable. It does not require overfitting a new numeric threshold; it removes model discretion from list construction.
