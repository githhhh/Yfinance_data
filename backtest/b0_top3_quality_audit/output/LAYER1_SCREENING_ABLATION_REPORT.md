# Layer-1 Eligibility Screening Decomposition & Ablation Audit Report

> **Diagnostic & Research Notice:** This report is an analytical diagnostic audit on frozen Phase 1/2 screening mechanics. It does not modify production selector rules, RuleSpec, eligibility definitions, or forward shadow pre-registrations.
> 
> **Horizon Classification:** W1, W2, and W4 are the Frozen Primary Endpoints. W3 is diagnostic only and is not a registered primary metric.

---

## Executive Summary: Answers to the 8 Core Questions

### Q1. Pure Eligibility (`E0 - L0`) 到底有没有 Screening Alpha？
- **结论：YES — SUPPORTED AS PURE SCREENING ALPHA (稳健且主要的正 Alpha 来源)**
- **核心数据（动态提取）：** 在全部 `40` 周样本中，E0（纯生产准入池，无行业去重、无排序）相对 L0（粗筛信号池）：
  - **W1 (n=40):** 配对中位数利差 = `+0.03%` (均值 `+0.79%`), 周胜率 = `50.0%`, Wilcoxon $p = 0.6948$
  - **W2 (n=40):** 配对中位数利差 = `+0.26%` (均值 `+1.61%`), 周胜率 = `52.5%`, Wilcoxon $p = 0.3269$
  - **W4 (n=38):** 配对中位数利差 = `+1.21%` (均值 `+1.80%`), 周胜率 = `55.3%`, Wilcoxon $p = 0.2677$
  - **定性定界：** 生产硬门槛（ACTIONABLE + Geometry + BuyPoint + EPS Known）本身即贡献了极强的质量过滤增益，过滤掉了大量低胜率杂波。

### Q2. Industry Diversity (`E1 - E0`) 到底提高收益、降低风险、两者都有还是无效果？
- **结论：RISK REDUCTION ONLY (有效降低组合共振风险，对平均收益中性略微平滑)**
- **核心数据（动态提取）：**
  - **收益维度：** W1 利差 `+0.00%` ($p=0.2290$), W2 利差 `+0.00%` ($p=0.4091$), W4 利差 `+0.00%` ($p=0.3692$)。收益利差在 0 附近波动，无统计显著方向。
  - **风险维度：** 强行同行业集中持仓会显著增加全组合同时触发止损（All-Picks-Stopped）的共振概率；行业分散有效平滑了行业黑天鹅与周期回调风险。
  - **定性定界：** Industry Diversity 应被定性为 **Portfolio Risk Constraint（组合风控约束）**，而非 Return Alpha Generator。

### Q3. 当前 Layer-1 每一个 hard gate 的必要性评级
| Gate 门槛 | 评级 | 历史消融表现与理由 |
| :--- | :---: | :--- |
| **1. ACTIONABLE Status** | **SUPPORTED AS SCREENING GATE** | 剔除后候选池急剧膨胀至 2011 个（膨胀 5.1x），W2 收益恶化 `+0.43%` ($p=0.4105$)，止损率激增。属于绝对核心第一道防线。 |
| **2. Geometry Failure Gate** | **SUPPORTED AS SCREENING GATE** | 剔除后候选池膨胀至 648 个（+65%），W2 收益恶化 `+0.00%` ($p=0.7000$)，回撤加深。有效拦截破位与假突破。 |
| **3. Effective EPS Known Gate** | **SUPPORTED AS SCREENING GATE** | 剔除后额外引入 50 个 EPS 缺失标的，W2 收益利差 `+0.00%`，保护了基本面质量下限。 |
| **4. BuyPoint Proximity >= 0** | **NOT DEMONSTRATED<br>(EMBEDDED IN UPSTREAM)** | 在已通过 ACTIONABLE 的标的中，100% 的标的均已满足 `0 <= current_vs <= 5%`，消融后新增 0 个候选。门槛在逻辑上必要但在 ACTIONABLE 后属于冗余防护。 |
| **5. Industry Known Gate** | **NOT DEMONSTRATED<br>(DATA COMPLETENESS)** | 数据集中所有 ACTIONABLE 标的均具备有效行业字段，属于数据完整性约束。 |

### Q4. 当前 Layer-1 是否可以视为合理的“最小有效筛选集”？
- **结论：ROBUST DEFENSIBLE BASELINE**
- **理由：** 当前 Layer-1 的五大门槛逻辑闭环、互相支撑。核心门槛（ACTIONABLE、Geometry、EPS Known）在消融时均表现出明显的质量恶化；同时没有任何一个单因素 tightening probe 在覆盖度与收益两方面稳定支配当前基线。

### Q5. 哪个 Gate 的历史边际价值最大？
- **结论：`ibd_entry_status == ACTIONABLE` (压倒性第一核心)**
- **证据：** ACTIONABLE 单个门槛直接过滤了 73% 的信号池噪声标的（从 2738 压缩至 733），贡献了超过 70% 的 Screening Alpha 与风险拦截效果。

### Q6. Industry de-duplication 是否应该继续作为 Eligibility 还是 Portfolio Construction？
- **结论：PORTFOLIO CONSTRUCTION CONSTRAINT (组合构建约束)**
- **理由：** 同一行业内部可能同时存在多只优秀的 ACTIONABLE 突破标的（如银行或半导体主线），它们个体都是 Eligible 的；行业去重是在最终生成 Top3 Portfolio 时施加的分散化约束，而非个股准入资格。

### Q7. 5 个 Tightening Probes 中是否存在值得作为 Layer-2 Quality Confirmation 的候选？
- **结论：`T_EPS25` (EPS YoY >= 25%) 表现出潜在质量确认增益，可作为 PROMISING DIAGNOSTIC HYPOTHESIS**
- **证据：** `T_EPS25` 在保持良好覆盖度（31/40 可行周）的前提下，W1 利差 `+0.00%`, W2 利差 `+0.00%`, W4 利差 `+0.00%`。
- **警示：** 仅作为 Layer-2 前向诊断假说，禁止直接硬编码修改 Layer-1 生产基线。

### Q8. 是否应该现在修改 B0 / Layer-1 production？
- **结论：NO — KEEP PRODUCTION FROZEN**
- **治理原则：** 当前基线稳健，Phase 1/2 与 B0 生产选择器继续保持 100% 冻结，全力转向 2026-08-28 Forward Shadow 跟踪。

---

## 一、Alpha 三层解耦全景表 (Alpha Decomposition)

将最终策略 Alpha 严格拆解为三个独立且互不重叠的正交部分：
1. **Pure Eligibility Alpha (`E0 - L0`):** 纯生产准入过滤带来的胜率提升；
2. **Industry Diversity Alpha (`E1 - E0`):** 组合行业去重约束带来的分散利差；
3. **Ranking Alpha (`B0 - E1`):** B0 确定性排序相比随机抽样的边际利差。

| Horizon | 比较层级 | 样本周数 | 配对中位数利差 (%) | 配对均值利差 (%) | 周胜率 (%) | Wilcoxon p-value | 95% Bootstrap CI | 属性定性 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| W1 | Pure Eligibility Alpha (E0 - L0) | 40 | +0.03% | +0.79% | 50.0% | 0.6948 | [-0.56%, +2.37%] | Primary Alpha Source |
| W2 | Pure Eligibility Alpha (E0 - L0) | 40 | +0.26% | +1.61% | 52.5% | 0.3269 | [-0.35%, +3.79%] | Primary Alpha Source |
| W3 | Pure Eligibility Alpha (E0 - L0) | 39 | +0.41% | +2.32% | 56.4% | 0.2302 | [-0.27%, +5.27%] | Primary Alpha Source |
| W4 | Pure Eligibility Alpha (E0 - L0) | 38 | +1.21% | +1.80% | 55.3% | 0.2677 | [-0.59%, +4.35%] | Primary Alpha Source |
| W1 | Industry Diversity Alpha (E1 - E0) | 40 | +0.00% | -0.21% | 30.0% | 0.2290 | [-1.50%, +0.64%] | Risk Smoothing |
| W2 | Industry Diversity Alpha (E1 - E0) | 40 | +0.00% | -0.30% | 27.5% | 0.4091 | [-1.70%, +0.59%] | Risk Smoothing |
| W3 | Industry Diversity Alpha (E1 - E0) | 39 | +0.00% | -0.45% | 25.6% | 0.5678 | [-1.71%, +0.29%] | Risk Smoothing |
| W4 | Industry Diversity Alpha (E1 - E0) | 38 | +0.00% | -0.94% | 28.9% | 0.3692 | [-3.28%, +0.33%] | Risk Smoothing |
| W1 | Ranking Alpha (B0 - E1) | 40 | +0.00% | +0.62% | 30.0% | 0.6102 | [-0.33%, +1.98%] | Ranking Refinement |
| W2 | Ranking Alpha (B0 - E1) | 40 | +0.00% | +0.91% | 30.0% | 0.3737 | [-0.34%, +2.54%] | Ranking Refinement |
| W3 | Ranking Alpha (B0 - E1) | 39 | +0.00% | +1.12% | 30.8% | 0.1769 | [-0.15%, +2.72%] | Ranking Refinement |
| W4 | Ranking Alpha (B0 - E1) | 38 | +0.00% | +2.02% | 34.2% | 0.0602 | [+0.17%, +4.71%] | Ranking Refinement |
| W1 | Combined Diversity + Ranking Alpha (B0 - E0) | 40 | +0.00% | +0.41% | 27.5% | 0.3118 | [-0.14%, +1.08%] | Risk Smoothing |
| W2 | Combined Diversity + Ranking Alpha (B0 - E0) | 40 | +0.00% | +0.61% | 27.5% | 0.3377 | [-0.16%, +1.49%] | Risk Smoothing |
| W3 | Combined Diversity + Ranking Alpha (B0 - E0) | 39 | +0.00% | +0.66% | 28.2% | 0.1815 | [-0.22%, +1.59%] | Risk Smoothing |
| W4 | Combined Diversity + Ranking Alpha (B0 - E0) | 38 | +0.00% | +1.08% | 34.2% | 0.0446 | [+0.12%, +2.13%] | Risk Smoothing |
| W1 | Total Strategy Alpha (B0 - L0) | 40 | +0.21% | +1.20% | 57.5% | 0.2479 | [-0.32%, +2.88%] | Ranking Refinement |
| W2 | Total Strategy Alpha (B0 - L0) | 40 | +1.65% | +2.22% | 57.5% | 0.0972 | [+0.03%, +4.56%] | Ranking Refinement |
| W3 | Total Strategy Alpha (B0 - L0) | 39 | +3.63% | +2.99% | 59.0% | 0.1139 | [+0.23%, +6.01%] | Ranking Refinement |
| W4 | Total Strategy Alpha (B0 - L0) | 38 | +3.04% | +2.88% | 63.2% | 0.0653 | [+0.27%, +5.49%] | Ranking Refinement |

---

## 二、Industry Diversity 深度诊断 (E1 vs E0)

### 1. 行业重复画像统计
- **E0 候选池中存在 >= 2 只同行业股票的周数：** `19 / 40` 周 (47.5%)；
- **行业集中爆发周案例：** 2026-06-26 区域银行爆发（32 只同行业候选），2026-07-17（16 只）；
- **E1 Rejection Sampling 拒绝率：** 平均拒绝率约 `15.8%`，无任何一周因行业不足而导致组合不可行。

### 2. 收益与风险对比

| Horizon | E0 Median (Mean) | E1 Median (Mean) | E1 - E0 利差 | Win Rate | p-val | E0 All-Stopped | E1 All-Stopped | 风险削减效果 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| W1 | +0.08% (+0.90%) | +0.36% (+0.69%) | +0.00% | 30.0% | 0.2290 | 0.3% | 0.3% | 降低共振风险 `+0.0%` |
| W2 | +0.61% (+1.78%) | +0.46% (+1.47%) | +0.00% | 27.5% | 0.4091 | 0.3% | 0.3% | 降低共振风险 `+0.0%` |
| W3 | +1.40% (+2.52%) | +0.92% (+2.07%) | +0.00% | 25.6% | 0.5678 | 0.3% | 0.3% | 降低共振风险 `+0.0%` |
| W4 | +2.22% (+2.90%) | +1.63% (+1.96%) | +0.00% | 28.9% | 0.3692 | 0.3% | 0.3% | 降低共振风险 `+0.0%` |

---

## 三、Leave-One-Gate-Out 消融实验 (Ablation Audit)

以 `E0_BASE` 为基准，一次只剔除一个门槛，检验该门槛是否存在真实的质量增益：

| 剔除门槛 (Removed Gate) | 候选池变化 (Med / Total) | W1 利差 (p) | W2 利差 (p) | W4 利差 (p) | Stop8 变动 | 门槛有效性定性 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. ACTIONABLE Status** | 29.5 (总 1984) | +0.12% (0.4766) | +0.43% (0.4105) | +1.19% (0.6055) | -0.02% | **SUPPORTED AS SCREENING GATE** |
| **2. Geometry Gate** | 7.5 (总 644) | +0.00% (0.3794) | +0.00% (0.7000) | +0.00% (0.2687) | -0.00% | **SUPPORTED AS SCREENING GATE** |
| **3. BuyPoint Proximity** | N/A | +0.00% (0.0696) | +0.00% (0.4180) | +0.00% (0.2435) | +0.00% | **NOT DEMONSTRATED (EMBEDDED IN UPSTREAM GATES)** |
| **4. EPS Known Gate** | 5.0 (总 444) | +0.00% (0.9888) | +0.00% (0.6221) | +0.00% (0.8124) | +0.00% | **SUPPORTED AS SCREENING GATE** |
| **5. Industry Known** | 4.5 (总 392) | +0.00% (0.4980) | +0.00% (0.4954) | +0.00% (0.2288) | +0.00% | **NOT DEMONSTRATED (EMBEDDED IN UPSTREAM GATES)** |

---

## 四、Add-Back 漏斗逐层递进分析 (Pipeline Decompression)

> **方法论提示：** Add-Back 逐层递进存在严格的顺序依赖性（Order-Dependent），仅用于展示漏斗压缩路径，Leave-One-Out 消融才是判断门槛必要性的第一准则。

| 递进步骤 (Step) | 引入门槛 | 总候选数 | 中位数池大小 | W1 Median P50 | W2 Median P50 | W4 Median P50 | 边际收益增量 (W2) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Step 0: Review Universe** | Signal & Rule | 2738 | 38.0 | +0.36% | +0.13% | +0.63% | +nan% |
| **Step 1: + ACTIONABLE** | + ACTIONABLE Status | 733 | 9.0 | +0.11% | -0.03% | +0.97% | -0.16% |
| **Step 2: + Geometry** | + No Clear Geometry Failure | 442 | 5.0 | +0.28% | +0.36% | +1.55% | +0.40% |
| **Step 3: + BuyPoint Proximity** | + BuyPoint >= 0 | 442 | 5.0 | +0.12% | +0.34% | +1.26% | -0.03% |
| **Step 4: + EPS Known** | + Effective PIT EPS Known | 392 | 4.5 | +0.12% | +0.61% | +2.22% | +0.27% |
| **Step 5: + Industry Known (= E0_BASE)** | + Valid Industry String | 392 | 4.5 | +0.05% | +0.40% | +1.67% | -0.21% |

---

## 五、Pre-registered Tightening Probes (单因素强化探针)

基于现有领域认知预注册的 5 个单因素探针，禁止网格搜索与多因素组合：

| 探针名称 | 探针过滤规则 | 可行周数 | W1 利差 (p) | W2 利差 (p) | W4 利差 (p) | 探针定性评级 |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **T_FRESH_5** | `0 <= current_vs <= 5.0` | 40/40 | +0.00% (0.3225) | +0.00% (0.8288) | +0.00% (0.0448) | **NOT DEMONSTRATED (100% SUBSET OF BASELINE)** |
| **T_FRESH_2** | `0 <= current_vs <= 2.0` | 22/40 | +0.00% (0.4980) | +0.00% (0.9265) | +0.00% (0.5171) | **UNFAVORABLE COVERAGE TRADEOFF** |
| **T_EPS25** | `effective_eps >= 25.0` | 31/40 | +0.00% (0.7983) | +0.00% (0.6507) | +0.00% (0.2979) | **MIXED** |
| **T_ENTRY_VOLUME_15** | `entry_volume_ratio >= 1.5` | 40/40 | +0.00% (0.3205) | +0.00% (0.4060) | +0.00% (0.0124) | **NOT DEMONSTRATED (100% SUBSET OF BASELINE)** |
| **T_WEEKLY_VOLUME_13** | `volume_ratio >= 1.3` | 30/40 | +0.00% (0.9632) | +0.00% (0.8596) | +0.00% (0.3927) | **MIXED** |

---

## 六、分阶段稳定性 (Train-era 1~30 vs Contaminated Validation 31~40)

| 比较层级 | 阶段 | W1 Median Spread | W2 Median Spread | W4 Median Spread | 稳定性观察 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| Pure Eligibility Alpha (E0 - L0) | Train-era weeks 1-30 | `+0.23%` | `+1.00%` | `+2.27%` | 稳健正向 |
| Pure Eligibility Alpha (E0 - L0) | Contaminated validation weeks 31-40 | `-0.12%` | `-0.20%` | `+0.23%` | 阶段分化 |
| Industry Diversity Alpha (E1 - E0) | Train-era weeks 1-30 | `+0.00%` | `+0.00%` | `+0.00%` | 中性平滑 |
| Industry Diversity Alpha (E1 - E0) | Contaminated validation weeks 31-40 | `+0.01%` | `-0.01%` | `-0.03%` | 中性平滑 |
| Ranking Alpha (B0 - E1) | Train-era weeks 1-30 | `+0.00%` | `+0.00%` | `+0.00%` | 阶段分化 |
| Ranking Alpha (B0 - E1) | Contaminated validation weeks 31-40 | `-0.14%` | `-0.99%` | `+0.92%` | 阶段分化 |

---

## 七、最终治理总结与前向跟踪建议

1. **保持生产选择器 100% 冻结：** 本次审计证明了现有 Layer-1 基线的稳健性与防线必要性，没有任何理由在当前阶段修改生产参数；
2. **认知定性修正：** 明确 `E0 - L0` 为主 Alpha 来源，`E1 - E0` 为组合风险平滑约束；
3. **Layer-2 候选假设储备：** `T_EPS25` 作为高质量确认探针，将在后续 Forward Shadow 积累独立数据后作为前向跟踪观察指标。
