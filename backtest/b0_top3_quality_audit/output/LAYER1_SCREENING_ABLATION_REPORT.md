# Layer-1 Eligibility Screening Decomposition & Ablation Audit Report

> **Diagnostic & Research Notice:** This report is an analytical diagnostic audit on frozen Phase 1/2 screening mechanics. It does not modify production selector rules, RuleSpec, eligibility definitions, or forward shadow pre-registrations.
> 
> **Horizon Classification:** W1, W2, and W4 are the Frozen Primary Endpoints. W3 is diagnostic only and is not a registered primary metric.

---

## Executive Summary: Answers to the 8 Core Questions

### Q1. Pure Eligibility (`E0 - L0`) 到底有没有 Screening Alpha？
- **结论：DIRECTIONALLY POSITIVE (全样本呈现正向利差，W4 最明显，但未达独立统计显著)**
- **核心数据（动态提取）：** 在全部 `40` 周样本中，E0（纯生产准入池，无行业去重、无排序）相对 L0（粗筛信号池）：
  - **W1 (n=40):** 配对中位数利差 = `+0.15%` (均值 `+0.85%`), 周胜率 = `55.0%`, Wilcoxon $p = 0.6085$
  - **W2 (n=40):** 配对中位数利差 = `+0.58%` (均值 `+1.77%`), 周胜率 = `55.0%`, Wilcoxon $p = 0.2163$
  - **W4 (n=38):** 配对中位数利差 = `+1.37%` (均值 `+2.01%`), 周胜率 = `60.5%`, Wilcoxon $p = 0.1718$
  - **定性定界：** 生产准入规则方向为正，且利差随持有期放大（W4 中位数 `+1.37%`）；但在统计检验上尚未达到独立 Alpha 显著性，且验证段存在阶段分化。

### Q2. Industry Diversity (`E1 - E0`) 到底提高收益、降低风险、两者都有还是无效果？
- **结论：NOT DEMONSTRATED (收益利差与风险削减在历史数据中均未呈现统计显著性)**
- **核心数据（动态提取）：**
  - **收益维度：** W1 利差 `+0.00%` (均值 `+0.26%`, $p=0.1297$), W2 利差 `+0.00%` (均值 `+0.25%`, $p=0.0599$), W4 利差 `+0.00%` (均值 `+0.42%`, $p=0.1591$)。
  - **风险维度：** 全止损率（All-Stopped）差异均值 `-0.36%`，配对检验 $p=0.5995$，未见统计显著差异。
  - **定性定界：** 行业去重属于理论上的组合构建约束（Portfolio Construction Constraint），但在历史样本中未证明独立收益或风险 Alpha。

### Q3. 当前 Layer-1 每一个 hard gate 的必要性评级
| Gate 门槛 | 证据评级 | 历史消融表现与理由 |
| :--- | :---: | :--- |
| **1. ACTIONABLE Status** | **DIRECTIONALLY SUPPORTED & OPERATIONALLY CRITICAL** | 剔除后候选池急剧膨胀至 2011 个（膨胀 5.1x），W1/W2/W4 收益均呈现正向利差（W2 `+0.58%`, W4 `+1.34%`）。属于主力候选规模压缩与业务第一道防线。 |
| **2. Geometry Failure Gate** | **NOT DEMONSTRATED (RETURN SPREAD NEUTRAL)** | 剔除后候选池膨胀至 648 个（+65%），过滤 256 只破位候选，但配对收益利差中位数在 W1/W2/W4 均为 `+0.06%` (p >= 0.27)。 |
| **3. Effective EPS Known Gate** | **NOT DEMONSTRATED (DATA-QUALITY CONSTRAINT)** | 剔除后额外引入 50 个 EPS 缺失标的，配对收益利差中位数为 `+0.00%` (p >= 0.62)，属于基本面数据完整性约束。 |
| **4. BuyPoint Proximity >= 0** | **NOT DEMONSTRATED (100% EMBEDDED IN ACTIONABLE)** | 在通过 ACTIONABLE 的标的中，100% 的标的均已满足 `0 <= current_vs <= 5%`，消融后新增 0 个候选。门槛在逻辑上必要但在 ACTIONABLE 后属于冗余防护。 |
| **5. Industry Known Gate** | **NOT DEMONSTRATED (DATA COMPLETENESS)** | 数据集中所有 ACTIONABLE 标的均具备有效行业字段，消融后新增 0 个候选。 |

### Q4. 当前 Layer-1 是否可以视为合理的“最小有效筛选集”？
- **结论：DEFENSIBLE BASELINE (合理可防御的基线，但不可宣称全局最优)**
- **理由：** ACTIONABLE 提供了必要的操作性候选池压缩，Geometry 与 EPS Known 提供了形态与数据质量兜底；没有单因素 tightening probe 能在覆盖度与收益两方面稳定超越当前基线。

### Q5. 哪个 Gate 的历史边际价值最大？
- **结论：`ibd_entry_status == ACTIONABLE` (压倒性主力规模压缩门槛)**
- **证据：** ACTIONABLE 单个门槛直接过滤了 73% 的信号池噪声标的（从 2738 压缩至 733），提供了主要的超额收益利差方向。

### Q6. Industry de-duplication 是否应该继续作为 Eligibility 还是 Portfolio Construction？
- **结论：PORTFOLIO CONSTRUCTION CONSTRAINT (组合构建约束)**
- **理由：** 标的个体即使同行业也是合法的 Eligible 资产，行业去重是生成 Top3 Portfolio 时的分散化约束。

### Q7. 5 个 Tightening Probes 中是否存在值得作为 Layer-2 Quality Confirmation 的候选？
- **结论：NONE QUALIFY AS PROVEN CANDIDATES (当前证据不足以确立第二层质量确认规则)**
- **证据：**
  - `T_FRESH_5` 与 `T_ENTRY_VOLUME_15`: 与 E0 完全重合（0 候选过滤，利差为 0）；
  - `T_EPS25`: 配对收益利差中位数在 W1/W2/W4 均为 `+0.00%` (p >= 0.30)，评为 **MIXED / NOT YET DEMONSTRATED**；
  - `T_FRESH_2`: 导致可用周数急剧下降至 22 周（覆盖度崩塌），评为 **UNFAVORABLE COVERAGE TRADEOFF**；
  - `T_WEEKLY_VOLUME_13`: 配对利差中位数为 0，覆盖度下降至 30 周，评为 **MIXED**。

### Q8. 是否应该现在修改 B0 / Layer-1 production？
- **结论：NO — KEEP PRODUCTION FROZEN**
- **治理原则：** 当前基线稳健，Phase 1/2 与 B0 生产选择器继续保持 100% 冻结，不作任何调整，直接切入 2026-08-28 Forward Shadow 跟踪。

---

## 一、Alpha 三层解耦全景表 (Alpha Decomposition)

| Horizon | 比较层级 | 样本周数 | 配对中位数利差 (%) | 配对均值利差 (%) | 周胜率 (%) | Wilcoxon p-value | 95% Bootstrap CI | 属性定性 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| W1 | Pure Eligibility Alpha (E0 - L0) | 40 | +0.15% | +0.85% | 55.0% | 0.6085 | [-0.53%, +2.47%] | Primary Alpha Direction |
| W2 | Pure Eligibility Alpha (E0 - L0) | 40 | +0.58% | +1.77% | 55.0% | 0.2163 | [-0.09%, +3.88%] | Primary Alpha Direction |
| W3 | Pure Eligibility Alpha (E0 - L0) | 39 | +1.17% | +2.35% | 56.4% | 0.2551 | [-0.32%, +5.39%] | Primary Alpha Direction |
| W4 | Pure Eligibility Alpha (E0 - L0) | 38 | +1.37% | +2.01% | 60.5% | 0.1718 | [-0.32%, +4.48%] | Primary Alpha Direction |
| W1 | Industry Diversity Alpha (E1 - E0) | 40 | +0.00% | +0.26% | 27.5% | 0.1297 | [+0.00%, +0.66%] | Risk Constraint |
| W2 | Industry Diversity Alpha (E1 - E0) | 40 | +0.00% | +0.25% | 32.5% | 0.0599 | [+0.04%, +0.50%] | Risk Constraint |
| W3 | Industry Diversity Alpha (E1 - E0) | 39 | +0.00% | +0.23% | 25.6% | 0.0984 | [+0.01%, +0.48%] | Risk Constraint |
| W4 | Industry Diversity Alpha (E1 - E0) | 38 | +0.00% | +0.42% | 26.3% | 0.1591 | [+0.03%, +0.89%] | Risk Constraint |
| W1 | Ranking Alpha (B0 - E1) | 40 | +0.00% | +0.19% | 27.5% | 0.6091 | [-0.31%, +0.71%] | Top-Bucket Selection |
| W2 | Ranking Alpha (B0 - E1) | 40 | +0.00% | +0.41% | 27.5% | 0.4749 | [-0.52%, +1.37%] | Top-Bucket Selection |
| W3 | Ranking Alpha (B0 - E1) | 39 | +0.00% | +0.86% | 33.3% | 0.1893 | [-0.19%, +2.08%] | Top-Bucket Selection |
| W4 | Ranking Alpha (B0 - E1) | 38 | +0.00% | +1.40% | 36.8% | 0.1042 | [-0.14%, +3.58%] | Top-Bucket Selection |
| W1 | Combined Diversity + Ranking Alpha (B0 - E0) | 40 | +0.00% | +0.45% | 32.5% | 0.2902 | [-0.14%, +1.13%] | Risk Constraint |
| W2 | Combined Diversity + Ranking Alpha (B0 - E0) | 40 | +0.00% | +0.66% | 30.0% | 0.2877 | [-0.34%, +1.71%] | Risk Constraint |
| W3 | Combined Diversity + Ranking Alpha (B0 - E0) | 39 | +0.00% | +1.09% | 33.3% | 0.0973 | [-0.02%, +2.37%] | Risk Constraint |
| W4 | Combined Diversity + Ranking Alpha (B0 - E0) | 38 | +0.00% | +1.82% | 36.8% | 0.0230 | [+0.25%, +4.01%] | Risk Constraint |
| W1 | Total Strategy Alpha (B0 - L0) | 40 | +0.53% | +1.30% | 57.5% | 0.2064 | [-0.24%, +3.02%] | Top-Bucket Selection |
| W2 | Total Strategy Alpha (B0 - L0) | 40 | +1.59% | +2.43% | 60.0% | 0.0548 | [+0.26%, +4.73%] | Top-Bucket Selection |
| W3 | Total Strategy Alpha (B0 - L0) | 39 | +3.55% | +3.44% | 64.1% | 0.0767 | [+0.58%, +6.63%] | Top-Bucket Selection |
| W4 | Total Strategy Alpha (B0 - L0) | 38 | +3.16% | +3.83% | 65.8% | 0.0253 | [+0.95%, +6.88%] | Top-Bucket Selection |

---

## 二、Industry Diversity 深度诊断 (E1 vs E0)

### 1. 行业重复客观画像
- **E0 候选池中存在同行业重复且产生实际组合影响的周数：** W1 共有 18 周 (45.0%), W2 共有 18 周 (45.0%), W4 共有 16 周 (42.1%);
- **行业去重活跃周 (Non-zero impact subset, 严格动态计算):**
  - **W1 (n=18):** 利差中位数 `+0.22%` (均值 `+0.58%`, 胜率 `61.1%`)
  - **W2 (n=18):** 利差中位数 `+0.25%` (均值 `+0.56%`, 胜率 `72.2%`)
  - **W4 (n=16):** 利差中位数 `+0.40%` (均值 `+1.00%`, 胜率 `62.5%`)

### 2. 全样本收益与风险统计对比

| Horizon | E0 Median (Mean) | E1 Median (Mean) | E1 - E0 利差 (p) | E0 All-Stopped | E1 All-Stopped | All-Stopped 差异 (p) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| W1 | +0.15% (+0.97%) | +0.43% (+1.23%) | +0.00% (0.1297) | 31.1% | 30.8% | -0.36% (0.5995) |
| W2 | +0.61% (+2.01%) | +0.65% (+2.26%) | +0.00% (0.0599) | 31.1% | 30.8% | -0.36% (0.5995) |
| W3 | +1.40% (+2.58%) | +1.40% (+2.81%) | +0.00% (0.0984) | 31.9% | 31.5% | -0.36% (0.5830) |
| W4 | +2.34% (+3.09%) | +1.83% (+3.51%) | +0.00% (0.1591) | 32.7% | 32.4% | -0.37% (0.5830) |

---

## 三、Leave-One-Gate-Out 消融实验 (Ablation Audit)

| 剔除门槛 (Removed Gate) | 候选池变化 (Med / Total) | W1 利差 (p) | W2 利差 (p) | W4 利差 (p) | Stop8 变动 | 门槛证据评级 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. ACTIONABLE Status** | 30.5 (总 2056) | +0.36% (0.4105) | +0.58% (0.3154) | +1.34% (0.3728) | -2.18% | **DIRECTIONALLY SUPPORTED & OPERATIONALLY CRITICAL GATE** |
| **2. Geometry Gate** | 7.5 (总 672) | +0.00% (0.3794) | +0.06% (0.5043) | +0.00% (0.2584) | -0.63% | **NOT DEMONSTRATED (RETURN SPREAD NEUTRAL; QUALITY FILTER)** |
| **3. BuyPoint Proximity** | N/A | +0.00% (nan) | +0.00% (nan) | +0.00% (nan) | +0.00% | **NOT DEMONSTRATED (100% EMBEDDED IN ACTIONABLE / DATA COMPLETENESS)** |
| **4. EPS Known Gate** | 5.0 (总 444) | +0.00% (0.8926) | +0.00% (1.0000) | +0.00% (0.7615) | +0.24% | **NOT DEMONSTRATED (RETURN SPREAD NEUTRAL; QUALITY FILTER)** |
| **5. Industry Known** | 5.0 (总 412) | +0.00% (nan) | +0.00% (nan) | +0.00% (nan) | +0.00% | **NOT DEMONSTRATED (100% EMBEDDED IN ACTIONABLE / DATA COMPLETENESS)** |

---

## 四、Add-Back 漏斗逐层递进分析 (Pipeline Decompression)

| 递进步骤 (Step) | 引入门槛 | 总候选数 | 中位数池大小 | W1 Median P50 | W2 Median P50 | W4 Median P50 | 边际收益增量 (W2) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Step 0: Review Universe** | Signal & Rule | 2738 | 38.0 | +0.37% | +0.09% | +0.50% | +nan% |
| **Step 1: + ACTIONABLE** | + ACTIONABLE Status | 733 | 9.0 | +0.26% | +0.05% | +1.36% | -0.04% |
| **Step 2: + Geometry** | + No Clear Geometry Failure | 442 | 5.0 | +0.26% | +0.55% | +2.04% | +0.50% |
| **Step 3: + BuyPoint Proximity** | + BuyPoint >= 0 | 442 | 5.0 | +0.26% | +0.55% | +2.04% | +0.00% |
| **Step 4: + EPS Known** | + Effective PIT EPS Known | 414 | 5.0 | +0.15% | +0.61% | +2.34% | +0.06% |
| **Step 5: + Industry Known (= E0_BASE)** | + Valid Industry String | 414 | 5.0 | +0.15% | +0.61% | +2.34% | +0.00% |

---

## 五、Pre-registered Tightening Probes (单因素强化探针)

| 探针名称 | 探针过滤规则 | 可行周数 | W1 利差 (p) | W2 利差 (p) | W4 利差 (p) | 探针定性评级 |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **T_FRESH_5** | `0 <= current_vs <= 5.0` | 40/40 | +0.00% (nan) | +0.00% (nan) | +0.00% (nan) | **NOT DEMONSTRATED (100% SUBSET OF BASELINE; SPREAD=0)** |
| **T_FRESH_2** | `0 <= current_vs <= 2.0` | 22/40 | +0.00% (0.1956) | +0.00% (0.8317) | +0.13% (0.2312) | **UNFAVORABLE COVERAGE TRADEOFF (SEVERE COVERAGE COLLAPSE)** |
| **T_EPS25** | `effective_eps >= 25.0` | 30/40 | +0.00% (0.5791) | +0.00% (0.3038) | +0.00% (0.9341) | **MIXED / NOT YET DEMONSTRATED** |
| **T_ENTRY_VOLUME_15** | `entry_volume_ratio >= 1.5` | 40/40 | +0.00% (nan) | +0.00% (nan) | +0.00% (nan) | **NOT DEMONSTRATED (100% SUBSET OF BASELINE; SPREAD=0)** |
| **T_WEEKLY_VOLUME_13** | `volume_ratio >= 1.3` | 31/40 | +0.00% (0.7119) | +0.00% (0.6705) | +0.00% (0.2979) | **MIXED / NOT YET DEMONSTRATED** |

---

## 六、分阶段稳定性 (Train-era 1~30 vs Contaminated Validation 31~40)

| 比较层级 | 阶段 | W1 Median Spread | W2 Median Spread | W4 Median Spread | 稳定性观察 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| Pure Eligibility Alpha (E0 - L0) | Train-era weeks 1-30 | `+0.01%` | `+1.16%` | `+1.24%` | 稳健正向 |
| Pure Eligibility Alpha (E0 - L0) | Contaminated validation weeks 31-40 | `+0.20%` | `-0.10%` | `+1.51%` | 阶段分化 |
| Industry Diversity Alpha (E1 - E0) | Train-era weeks 1-30 | `+0.00%` | `+0.00%` | `+0.00%` | 中性平滑 |
| Industry Diversity Alpha (E1 - E0) | Contaminated validation weeks 31-40 | `+0.22%` | `+0.00%` | `-0.33%` | 中性平滑 |
| Ranking Alpha (B0 - E1) | Train-era weeks 1-30 | `+0.00%` | `+0.00%` | `+0.00%` | 阶段分化 |
| Ranking Alpha (B0 - E1) | Contaminated validation weeks 31-40 | `-0.03%` | `-0.74%` | `+1.26%` | 阶段分化 |

---

## 七、最终治理总结与前向跟踪建议

1. **保持生产选择器 100% 冻结：** 本次审计未发现任何足以修改生产基线的统计证据，生产基线保持完全冻结；
2. **认知定性校准：** Pure Eligibility 呈现全样本正向方向性（W4 最明显），但独立统计显著性尚未达到；Industry Diversity 在历史样本中未证明超额收益或风险削减；
3. **禁止引入新规则：** 预注册探针均未表现出支配性增益，不引入任何第二层复杂规则，直接进入 2026-08-28 Forward Shadow。
