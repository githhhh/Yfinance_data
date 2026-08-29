# EPS PIT Recalibration — Final Analysis Report

## Executive Summary

本轮工作完成了历史 BreakoutFollow / B0 研究链路的 **EPS PIT 数据校准（EPS_RECALIBRATED_V2）**，并在严格冻结价格、历史 outcome、生产 selector 与研究规则的前提下，重新陈述了 B0 / Matched-N Random / Three-Tier / Layer-1 / Rank / Historical Validation 结果。

最终结论：

> **KEEP PRODUCTION FROZEN**

本轮校准证明了两件同时成立的事情：

1. **旧 EPS 数据确实足以 materially 影响历史候选事实与部分选择结果**：2,738 条 signal rows 中有 553 条 EPS 数值变化，103 条从 unknown 变 known，16 条从 known 变 unknown，239 条 EPS>=25% 状态发生变化。
2. **生产 B0 的核心研究结论没有被推翻**：候选资格有变化，但每周 B0 选股数量没有发生变化；B0 相对 Matched-N Random 的中期 W4 质量反而增强，而内部 Rank1/2/3 的精细排序证据明显减弱。

因此，本次最重要的认知不是“应该重新调参”，而是：

> **B0 更适合被理解为 Top-Bucket Selector，而不是 Fine Ranker。**
>
> EPS PIT 校准提高了研究事实的可信度，但没有提供足够证据改变生产 selector。

---

## 1. Scope and Governance

### Data revision

- Revision: `EPS_RECALIBRATED_V2`
- Old EPS baseline ref: `593bd333181da4fe301b3f61397c7bc95ac86ced`
- Recalibration mode: `EPSResolveMode.REPLAY`
- Historical pools: 43
- Historical signal rows: 2,738

### Frozen boundaries

本轮明确冻结：

- production selector
- historical price cache
- candidate weekly outcomes
- train weekly outcomes
- production rule semantics
- champion identities / parameters
- forward-shadow registration

本轮是：

> **DATA CORRECTION + RESEARCH RESTATEMENT**

不是：

> rule optimization / parameter tuning / champion reselection

---

## 2. EPS PIT Data Quality Impact

| Metric | Result |
|---|---:|
| Old resolved rows | 2,380 |
| New resolved rows | 2,467 |
| Old resolved coverage | 86.92% |
| New resolved coverage | 90.10% |
| Coverage improvement | +3.18 pp |
| EPS value changed | 553 |
| Unknown -> known | 103 |
| Known -> unknown | 16 |
| Source changed | 138 |
| EPS >=25% state changed | 239 |
| Provider errors after recalibration | 0 |
| Future-leakage violations | 0 |

这说明校准不是 cosmetic refresh。约五分之一的 signal rows 出现了 EPS 数值修订，且 EPS25 状态变化达到 239 条，足以影响 EPS 相关 eligibility / ranking diagnostics。

同时，REPLAY store 经审计满足：

> `effective_date <= snapshot_date`

不存在未来 EPS 泄漏。

### Provider semantics

本轮同时修复了一个独立的 SEC provider 语义问题：

- SEC companyfacts 明确返回 HTTP 404 时，视为 **clean unavailable history**；
- 403 / 429 / 5xx / ticker-map technical failure 仍保持 provider-error / fail-close 语义。

因此没有为了让历史数据“更完整”而放松技术错误的 fail-close 原则。

---

## 3. Candidate Universe and B0 Selection Impact

EPS 修订确实改变了 eligibility 层：

- E0 membership changed candidates: **22**
- affected weeks: **10**

但生产选择结果的结构变化有限：

- B0 selected-count changed weeks: **0**
- B0 code-set/order changed weeks: **6**
- order-only changed weeks: **2**

关键含义是：

> EPS PIT correction 会改变候选资格与部分 Top3 成员/顺序，但没有改变生产系统每周的仓位数量行为。

这也是为什么本轮应被解释为 **data restatement**，而不是策略规则失效。

---

## 4. B0 vs Matched-N Random

校准后，B0 相对等仓位 Matched-N Random 的表现整体增强，尤其 W4。

### Primary horizons

| Horizon | Paired median spread Old | New | Paired mean spread Old | New | Beat Random Old | New |
|---|---:|---:|---:|---:|---:|---:|
| W1 | +0.4513% | +0.5798% | +1.1313% | +1.2391% | 55.00% | 57.50% |
| W2 | +1.5435% | +1.6820% | +2.1240% | +2.4141% | 57.50% | 60.00% |
| W4 | +3.3309% | +3.6791% | +2.6325% | +3.5630% | 63.16% | 65.79% |

### Random percentile

| Horizon | Median percentile Old | New |
|---|---:|---:|
| W1 | 54.98 | 57.58 |
| W2 | 64.09 | 68.90 |
| W4 | 65.52 | 72.57 |

### Interpretation

最有价值的变化发生在 W4：

- median spread 增加约 **+0.35 pp**
- mean spread 增加约 **+0.93 pp**
- beat-random rate 增加约 **+2.63 pp**
- median random percentile 从 **65.52** 提升到 **72.57**

因此旧结论：

> B0 在 medium horizon 存在值得继续前向观察的历史 selection-quality signal

应从 **RETAINED** 上调为：

> **STRENGTHENED**

但这仍然是历史样本证据，不是 virgin OOS proof。

---

## 5. Eligibility and Gate Interpretation

### Pure Eligibility

结论：

> **RETAINED**

校准后 Pure Eligibility 仍呈方向性正贡献，W4 最明显，但独立统计证据不足以称为 proven alpha。

因此 eligibility 层仍有合理性，但不应根据本次校准新增门槛。

### ACTIONABLE

结论：

> **RETAINED**

ACTIONABLE 仍是最重要的 operational compression gate。它的核心价值不仅是历史收益差异，更是把原始 signal universe 压缩为可执行候选。

### EPS Known

结论：

> **RETAINED as a PIT data-quality / completeness gate**

本轮最大变化发生在 EPS 数据本身，但 EPS Known gate 的独立 return alpha 仍然：

> **NOT DEMONSTRATED**

因此应区分：

- **EPS 数据质量改善：明确成立**
- **EPS Known 独立收益 alpha 增强：不成立**

不能因为 PIT coverage 提升就把 EPS Known 升级为更强的收益筛选器。

### Geometry

结论：

> **RETAINED — independent return alpha NOT DEMONSTRATED**

继续作为 sanity / quality gate，而不是新增权重来源。

### Industry Diversity

结论：

> **RETAINED**

继续定位于 portfolio construction，而非 eligibility alpha。

---

## 6. Fine Ranking Evidence Weakened

这是本轮校准后最重要的负向修订。

### R3 vs R2

W2：

| Metric | Old | New |
|---|---:|---:|
| Median R3-R2 spread | +2.0812% | +0.0662% |
| R3 > R2 win rate | 68.0% | 52.0% |
| Wilcoxon p | 0.0236 | 0.2872 |

旧样本中 W2 曾表现出较强 R3 > R2 结构，但校准后几乎收敛到中性。

W4：

- median spread: **+3.8469% -> +3.0804%**
- win rate: **73.91% -> 60.87%**
- p-value: **0.0522 -> 0.8462**

因此：

> **R3 vs R2 = WEAKENED**

不再存在足以支持修改 rank logic 的统计依据。

---

## 7. Top3 vs Top2 / MC3 Also Weakened

W4 Rank3 marginal contribution（MC3）：

| Metric | Old | New |
|---|---:|---:|
| Median MC3 | +0.9683% | +0.1951% |
| Mean MC3 | -0.1078% | -0.8874% |
| K3 > K2 win rate | 56.52% | 52.17% |
| Wilcoxon p | 0.6869 | 0.8229 |

所以旧的“Top3 比 Top2 可能有方向性优势”仍没有完全反转，但证据明显更弱。

结论：

> **Top3 vs Top2 / MC3 = WEAKENED**

这同样不能被解释成“应该删除第三只”。

目前证据只支持：

> 第三只的独立边际贡献尚未被证明。

生产端仍应保持最大 Top3 的既有结构，等待 forward shadow。

---

## 8. Rank1 / Rank2 / Rank3: No Monotonic Fine Ranking

校准后 All Common-Support Weeks 的 median return：

| Horizon | Rank1 | Rank2 | Rank3 |
|---|---:|---:|---:|
| W1 | +0.32% | +0.05% | +0.34% |
| W2 | +1.96% | -0.66% | +0.20% |
| W4 | +1.58% | +2.77% | +3.77% |

Pooled Spearman 在 W1/W2/W4 均接近 0。

因此：

> **Fine-rank monotonicity remains NOT DEMONSTRATED.**

这进一步强化 B0 的正确产品定位：

> **Top-Bucket Equal-Weighted Selector**

而不是：

> Rank1 必然优于 Rank2，Rank2 必然优于 Rank3 的精细评分系统。

---

## 9. EPS25 Tightening Probe

EPS25 是本次最敏感的 tightening probe，因为 239 条记录发生了 EPS>=25% 状态变化。

校准后：

- W1 median spread: 0
- W2 median spread: 0
- W4 median spread: 0
- W1/W2 mean spread 为负
- W4 mean spread略正但不稳定
- Wilcoxon 均不支持独立显著性

最终仍为：

> **MIXED / NOT YET DEMONSTRATED**

所以：

> **NO LAYER2 RULE QUALIFIES**

本轮 EPS 校准没有产生一个可直接升级到 production 的 EPS25 hard gate。

---

## 10. Historical Validation Boundary

Contaminated Historical Validation 固定日历为：

> **2026-05-29 through 2026-08-07**

共：

> **11 snapshot weeks**

这一段因为此前研发已接触过，必须继续标记为：

> **CONTAMINATED HISTORICAL VALIDATION**

不能称为 OOS。

此外，旧 baseline 使用 positional slicing，而 V2 使用 fixed-calendar windows，因此：

> Train / contaminated-validation 的 Old -> New delta 不能完全归因于 EPS correction。

最干净的 EPS revision 对比应优先看：

> **All-historical Old -> New**

真正的无偏验证从预注册的：

> **2026-08-28 Forward Shadow**

开始。

---

## 11. What the Recalibration Changed — and What It Did Not

### Changed

- 历史 EPS coverage 提高
- 553 条 EPS 数值修订
- 22 个 E0 membership 变化
- 6 周 B0 code/order 发生变化
- B0 W4 historical quality evidence 增强
- R3 vs R2 historical evidence 明显减弱
- Top3 vs Top2 historical marginal evidence减弱

### Did not change

- production selector
- price history
- weekly outcome facts
- stop / return calculation semantics
- eligibility rule definitions
- B0 rank rule
- champion identities
- forward-shadow registry
- production position-count behavior

这正是一个合格 data restatement 应该表现出的边界。

---

## 12. Final Restatement of Prior Conclusions

| Prior conclusion | Final verdict | Interpretation |
|---|---|---|
| Pure Eligibility | **RETAINED** | Directionally positive; independent proof not demonstrated |
| ACTIONABLE | **RETAINED** | Operationally critical compression gate |
| Geometry | **RETAINED** | Quality/sanity gate; independent alpha not demonstrated |
| EPS Known | **RETAINED** | PIT completeness gate; independent return alpha not demonstrated |
| Industry Diversity | **RETAINED** | Portfolio-construction role |
| B0 W4 quality | **STRENGTHENED** | Medium-horizon historical selection quality improved |
| Fine-rank monotonicity | **RETAINED: NOT DEMONSTRATED** | B0 is not a fine ranker |
| R3 vs R2 | **WEAKENED** | Prior statistical support largely disappeared |
| Top3 vs Top2 / MC3 | **WEAKENED** | Directional evidence remains weak and unstable |
| EPS25 Layer2 | **RETAINED: NOT QUALIFIED** | No production tightening rule justified |

---

## 13. Production Decision

### Decision

> **KEEP PRODUCTION FROZEN**

No production selector change is justified by this recalibration.

原因不是“新结果没有变化”，恰恰相反——EPS 数据变化很大；但重新计算后显示：

- 能稳定保留的是 **Top-bucket selection quality**
- 被削弱的是 **内部 fine ranking / Rank3-vs-Rank2 等细粒度假设**
- EPS-specific tightening 仍没有达到 production gate 标准

如果现在根据历史变化继续调整排序或新增 EPS25 gate，反而会把一次数据纠错变成 hindsight optimization。

---

## 14. Baseline Governance Going Forward

从本报告起：

> `EPS_RECALIBRATED_V2` 应视为历史 EPS 研究的冻结基线。

后续工作应遵循：

1. 不再回写或“美化”这套 historical baseline；
2. 新的 EPS/selector/ranking 假设作为独立实验；
3. 价格口径（例如 adjusted vs raw）如需重新研究，必须作为独立 data revision，不得与 EPS revision 混合归因；
4. 生产变更只接受 forward shadow 或新的明确实验设计支持；
5. 2026-08-28 之后的数据才承担真正 virgin forward evidence 的职责。

---

## 15. Audit Status

本轮最终审计状态：

- EPS PIT correctness: **PASS**
- Future leakage: **PASS / 0 violations**
- Non-EPS pool invariant: **PASS**
- Frozen price/outcome invariant: **PASS**
- Production selector invariant: **PASS**
- Matched-N protocol: **PASS**
- Fixed-calendar validation semantics: **PASS**
- Restatement report regression coverage: **PASS**
- Production decision: **KEEP PRODUCTION FROZEN**

Research closeout commit before this report:

`b111c2841a9d9d27c95df21e327a1fffcee7a58f`

---

## Final Conclusion

本轮 EPS PIT 修订的价值，不是产生了一个“更漂亮”的回测，而是提高了研究结果的可相信程度。

校准后的结果更清晰地表明：

> **B0 的主要价值在于把信号池压缩成一个质量更高的 Top3 bucket；它目前没有证明自己能可靠地区分 bucket 内部第 1、2、3 名的真实质量顺序。**

同时，中期 W4 相对 Matched-N Random 的表现增强，说明核心 selector 的历史质量信号没有因 EPS 数据纠错而消失。

因此本轮最合理的研究决策是：

> **接受数据修订，修正对 fine ranking 的认知，保持生产规则冻结，并把下一阶段证据要求交给 Forward Shadow。**
