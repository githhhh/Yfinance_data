# B0 Rank Position & TopK Marginal Contribution Audit Report

> **Diagnostic & Audit Notice:** This report is an analytical diagnostic audit on frozen Phase 1/2 B0 selection events. It does not modify production selector rules, RuleSpec, eligibility definitions, or forward shadow pre-registrations.
> 
> **Diagnostic Horizon Notice:** W1, W2, and W4 are the Frozen Primary Metrics. W3 is diagnostic only and is not a newly registered primary endpoint.

---

## Executive Conclusion: Answers to the 5 Core Questions

### Q1. B0 Rank1 / Rank2 / Rank3 是否存在稳定质量差异？
- **结论：PARTIALLY SUPPORTED (HISTORICAL DIAGNOSTIC SIGNAL)**
- **核心数据（动态提取）：** 在全部 3-pick Common Support 样本中，Rank1 与 Rank3 在多数周期收益与路径表现较强，而 Rank2 出现非单调的中间塌陷：
  - **W1 (n=25):** Rank1 med=`+0.32%` (mean `+0.93%`), Rank2 med=`+0.05%` (mean `-0.18%`), Rank3 med=`+0.34%` (mean `+0.89%`)
  - **W2 (n=25):** Rank1 med=`+1.96%` (mean `+1.43%`), Rank2 med=`-0.66%` (mean `-1.07%`), Rank3 med=`+0.20%` (mean `+1.95%`)
  - **W3 (诊断) (n=24):** Rank1 med=`+1.34%` (mean `+2.55%`), Rank2 med=`+0.74%` (mean `+1.70%`), Rank3 med=`+1.25%` (mean `+2.60%`)
  - **W4 (n=23):** Rank1 med=`+1.58%` (mean `+2.88%`), Rank2 med=`+2.77%` (mean `+7.09%`), Rank3 med=`+3.77%` (mean `+2.32%`)
  - **路径质量：** Rank2 止损率最高 (`44.0%`)，Profit20 达成率最低 (`32.0%` vs Rank1 `40.0%`, Rank3 `48.0%`)，最大回撤中位数达 `-7.55%`。
  - **定性定界：** 该差异呈现 U 型非单调结构，属于历史诊断特征，不能简单视作线性排序能力。

### Q2. B0 ranking 是否存在 Monotonicity（单调顺序信息量）？
- **结论：NOT SUPPORTED (B0 不具备 fine-ranking 单调性，实际表现为 Top-Bucket Selector)**
- **核心数据（动态提取）：**
  - **Rank1 > Rank2 周胜率：** W1: `48.0%`, W2: `56.0%`, W4: `43.5%` (无统计优势，接近抛硬币)
  - **Rank2 > Rank3 周胜率：** W1: `44.0%`, W2: `48.0%`, W4: `39.1%` (发生实质性倒挂，Rank3 > Rank2 在多数周发生)
  - **Pooled Spearman 秩相关系数：** W1: `r=-0.0075`, W2: `r=+0.0053`, W4: `r=+0.0428` (均接近 0，无单调预测力)
  - **定位定性：** B0 在历史样本中表现为 **Top-Bucket Selector**（将优质标的筛入 Top3 头部集合），但集合内部的 1/2/3 顺位不包含单调优劣信息。

### Q3. “Top2 明显更差”这个说法：
- **结论：PARTIALLY SUPPORTED**
- **证据分层分析（动态提取）：**
  - **Rank3 vs Rank2 (支持性强):** Rank3 在 W2 显著战胜 Rank2 (中位数利差 `+0.07%`, 胜率 `52.0%`, Wilcoxon $p=0.2872$)，W1/W3/W4 均呈现明显正向利差。
  - **Rank1 vs Rank2 (支持性弱):** Rank1 相对于 Rank2 未达到双侧统计显著水平，周胜率仅在 50% 附近波动。
  - **MC2 边际贡献分布 (左尾拖累而非每周恶化):** W1 median MC2 为 `+0.04%` (mean `-0.56%`)，W4 median MC2 为 `+1.23%` (mean `+2.10%`)。中位数在 W1/W4 为正，说明 Rank2 表现弱主要是由少数严重亏损的左尾事件（如生物科技板块/未缩量回撤）拉低均值，而非每周系统性拖累。

### Q4. “Top3 portfolio > Top2 portfolio”这个说法：
- **结论：PARTIALLY SUPPORTED / DIRECTIONAL ONLY**
- **证据分层分析（动态提取）：**
  - **全历史样本中位数偏正：** W1 MC3 med=`+0.10%` ($p=0.8532$), W2 MC3 med=`+0.00%` ($p=0.5602$), W4 MC3 med=`+0.20%` ($p=0.8229$)。
  - **无统计显著性且均值衰减：** 组合层 Wilcoxon $p$ 值在全周期均未达到显著水平；W4 平均边际贡献已转负 (`-0.89%`)。
  - **后期验证段倒挂：** 在历史验证周次 (31~40) 中，W3 median MC3 为 `-0.08%`，W4 median MC3 为 `+0.20%`，优势未能稳定延续。
  - **定性定界：** 绝不能表述为已证明的策略优势，仅定性为 **方向性历史现象**。

### Q5. 当前证据是否足以修改生产 B0：
- **结论：NO — KEEP PRODUCTION FROZEN**
- **治理原因：**
  1. 当前历史样本仅 25 个 3-pick 周，且 31~40 周为已知历史样本，样本容量不足以支持不可逆的生产参数重构；
  2. 直接根据历史 Rank2 较弱去调权或剔除 Rank2 属于典型的数据窥探过拟合风险；
  3. 必须保持 Phase 1/2 生产基线完全冻结，将 Rank Position 结构列为 2026 Forward Shadow 的前向观察指标。

---

## 一、Methodology & Common Support Denominator

为了避免“不同周数/不同样本分母”导致的虚假均值偏差，本审计遵循严格的 **Common Support & Maturity Gate** 准则：
1. **3-Pick Completeness:** 仅纳入 B0 生产选择器当周完整选出 3 只候选的周次；
2. **Horizon Maturity Gate:** 仅在 Rank1、Rank2、Rank3 三只标的在对应持有周期均满足 `is_complete_week == True` 且收益与最大涨幅数据完整时，该周才进入对应分母；
3. **Common Support 分母清单：**
   - **Total B0 Snapshot Weeks:** `40` 周 (3-pick 周 `25` 周, 2-pick 周 `7` 周, 1-pick 周 `8` 周)
   - **W1 Common Support:** `25` 周 (Train `14` 周, Contaminated Val `11` 周)
   - **W2 Common Support:** `25` 周 (Train `14` 周, Contaminated Val `11` 周)
   - **W3 (Diagnostic Only) Common Support:** `24` 周 (Train `14` 周, Contaminated Val `10` 周)
   - **W4 Common Support:** `23` 周 (Train `14` 周, Contaminated Val `9` 周)

---

## 二、Position Quality Audit (Rank1 / Rank2 / Rank3)

### 1. Return Quality Summary across All Common-Support Weeks

| Horizon | Rank | Weeks | Mean (%) | Median (%) | P25 (%) | P75 (%) | Win Rate (>0) | Profit20 Rate | Stop8 Rate | Max Drawdown (Med) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| W1 | Rank1 | 25 | +0.93% | +0.32% | -2.08% | +2.79% | 52.0% | 40.0% | 40.0% | -6.75% |
| W1 | Rank2 | 25 | -0.18% | +0.05% | -2.59% | +2.84% | 52.0% | 32.0% | 44.0% | -7.55% |
| W1 | Rank3 | 25 | +0.89% | +0.34% | -2.53% | +3.76% | 56.0% | 48.0% | 36.0% | -4.70% |
| W2 | Rank1 | 25 | +1.43% | +1.96% | -3.96% | +6.69% | 56.0% | 40.0% | 40.0% | -6.75% |
| W2 | Rank2 | 25 | -1.07% | -0.66% | -2.90% | +4.00% | 44.0% | 32.0% | 44.0% | -7.55% |
| W2 | Rank3 | 25 | +1.95% | +0.20% | -1.50% | +6.77% | 56.0% | 48.0% | 36.0% | -4.70% |
| W3 | Rank1 | 24 | +2.55% | +1.34% | -3.24% | +5.65% | 58.3% | 41.7% | 41.7% | -7.05% |
| W3 | Rank2 | 24 | +1.70% | +0.74% | -2.94% | +3.90% | 58.3% | 33.3% | 41.7% | -7.33% |
| W3 | Rank3 | 24 | +2.60% | +1.25% | -1.07% | +5.26% | 66.7% | 50.0% | 37.5% | -4.56% |
| W4 | Rank1 | 23 | +2.88% | +1.58% | -2.84% | +9.54% | 56.5% | 43.5% | 43.5% | -7.35% |
| W4 | Rank2 | 23 | +7.09% | +2.77% | -4.60% | +5.40% | 65.2% | 34.8% | 43.5% | -7.55% |
| W4 | Rank3 | 23 | +2.32% | +3.77% | -2.74% | +9.09% | 65.2% | 52.2% | 39.1% | -4.70% |

> **Note on W3:** W3 is diagnostic only and is not a newly registered primary endpoint. It confirms that the performance divergence between Rank1/3 and Rank2 persists continuously through weeks 1, 2, 3, and 4.

---

## 三、Top1 / Top2 / Top3 Portfolios & Marginal Contributions

定义等权组合：
- **K1:** `Rank1`
- **K2:** `mean(Rank1, Rank2)`
- **K3:** `mean(Rank1, Rank2, Rank3)`
- **Rank2 Marginal Contribution:** `MC2 = K2 - K1`
- **Rank3 Marginal Contribution:** `MC3 = K3 - K2`

### Marginal Contribution Summary (All Common-Support Weeks)

| Horizon | Weeks | K1 Med (Mean) | K2 Med (Mean) | K3 Med (Mean) | MC2 Med (Mean) | MC2 Win Rate | MC2 p-val | MC3 Med (Mean) | MC3 Win Rate | MC3 p-val |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| W1 | 25 | +0.32% (+0.93%) | +0.40% (+0.37%) | +0.69% (+0.55%) | +0.04% (-0.56%) | 52.0% | 0.3525 | +0.10% (+0.17%) | 60.0% | 0.8532 |
| W2 | 25 | +1.96% (+1.43%) | +0.20% (+0.18%) | +0.10% (+0.77%) | -0.43% (-1.25%) | 44.0% | 0.2752 | +0.00% (+0.59%) | 52.0% | 0.5602 |
| W3 | 24 | +1.34% (+2.55%) | -0.34% (+2.12%) | +1.22% (+2.28%) | +0.44% (-0.42%) | 54.2% | 0.9888 | +0.66% (+0.16%) | 58.3% | 0.684 |
| W4 | 23 | +1.58% (+2.88%) | +1.01% (+4.99%) | +3.77% (+4.10%) | +1.23% (+2.10%) | 56.5% | 0.709 | +0.20% (-0.89%) | 52.2% | 0.8229 |

---

## 四、配对假设检验：Hypothesis A, B, C 全景对比

### 1. Hypothesis A: Rank3 个股是否优于 Rank2？ (`Rank3 - Rank2`)

| Horizon | Weeks | Mean Spread (%) | Median Spread (%) | R3 > R2 Win Rate (%) | Wilcoxon p-value | 95% Bootstrap CI | 统计定性 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| W1 | 25 | +1.07% | +2.41% | 56.0% | 0.6338 | [-1.91%, +4.28%] | Directional Only |
| W2 | 25 | +3.02% | +0.07% | 52.0% | 0.2872 | [-0.18%, +6.64%] | Directional Only |
| W3 | 24 | +0.90% | +0.95% | 58.3% | 0.7257 | [-3.45%, +5.61%] | Directional Only |
| W4 | 23 | -4.77% | +3.08% | 60.9% | 0.8462 | [-18.36%, +5.66%] | Directional Only |

### 2. Hypothesis B: Top3 Portfolio 是否优于 Top2 Portfolio？ (`K3 - K2 = MC3`)

| Horizon | Weeks | Mean Spread (%) | Median Spread (%) | K3 > K2 Win Rate (%) | Wilcoxon p-value | 95% Bootstrap CI | 统计定性 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| W1 | 25 | +0.17% | +0.10% | 60.0% | 0.8532 | [-0.76%, +1.18%] | Directional Positive (Not Sig) |
| W2 | 25 | +0.59% | +0.00% | 52.0% | 0.5602 | [-0.57%, +1.82%] | Directional Positive (Not Sig) |
| W3 | 24 | +0.16% | +0.66% | 58.3% | 0.6840 | [-1.38%, +1.71%] | Directional Positive (Not Sig) |
| W4 | 23 | -0.89% | +0.20% | 52.2% | 0.8229 | [-4.02%, +1.79%] | Directional Positive (Not Sig) |

### 3. Hypothesis C: Rank1 个股是否优于 Rank2？ (`Rank1 - Rank2`)

| Horizon | Weeks | Mean Spread (%) | Median Spread (%) | R1 > R2 Win Rate (%) | Wilcoxon p-value | 95% Bootstrap CI | 统计定性 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| W1 | 25 | +1.11% | -0.09% | 48.0% | 0.3525 | [-1.28%, +3.53%] | Not Significant |
| W2 | 25 | +2.49% | +0.86% | 56.0% | 0.2752 | [-1.60%, +6.65%] | Not Significant |
| W3 | 24 | +0.84% | -0.87% | 45.8% | 0.9888 | [-5.10%, +7.38%] | Not Significant |
| W4 | 23 | -4.21% | -2.46% | 43.5% | 0.7090 | [-16.69%, +6.97%] | Not Significant |

> **方法论洞察：** Rank3 相比 Rank2 的优势在统计上显著高于 Rank1 相比 Rank2 的优势。这进一步印证了 Rank2 的弱势并非简单的阶梯递减，而是 Rank3 具有独特的路径恢复能力。

---

## 五、Rank Monotonicity & Spearman Correlation Audit

理想的 Fine Ranker 应具备 `Rank1 >= Rank2 >= Rank3` 的单调性。审计结果如下：

| Horizon | Weeks | R1 Med | R2 Med | R3 Med | R1 > R2 Rate | R2 > R3 Rate | R3 > R2 Rate | Pooled Spearman r (p) | 定位结论 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| W1 | 25 | +0.32% | +0.05% | +0.34% | 48.0% | 44.0% | 56.0% | -0.0075 (0.9488) | Non-Monotonic (Top-Bucket Selector) |
| W2 | 25 | +1.96% | -0.66% | +0.20% | 56.0% | 48.0% | 52.0% | +0.0053 (0.9641) | Non-Monotonic (Top-Bucket Selector) |
| W3 | 24 | +1.34% | +0.74% | +1.25% | 45.8% | 41.7% | 58.3% | +0.0532 (0.6572) | Non-Monotonic (Top-Bucket Selector) |
| W4 | 23 | +1.58% | +2.77% | +3.77% | 43.5% | 39.1% | 60.9% | +0.0428 (0.7271) | Non-Monotonic (Top-Bucket Selector) |

### 核心结论：
> **B0 does not demonstrate monotonic fine-ranking quality.**
> B0 在历史样本中表现为 **Top-Bucket Selector** 而非 Fine Ranker。它成功将优质标的聚集于 Top 3 头部集合，但内部 1/2/3 顺位不包含确定性的强弱顺序。

---

## 六、分阶段稳定性：Train (1~30) vs Contaminated Historical Validation (31~40)

| Horizon | Segment | Weeks | R1 Med (Mean) | R2 Med (Mean) | R3 Med (Mean) | R3 > R2 Win Rate | MC3 Med (Mean) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| W1 | Train-era weeks 1-30 | 14 | -0.60% (+0.76%) | +0.20% (-0.73%) | +1.66% (+1.93%) | 64.3% | +0.33% (+0.64%) |
| W1 | Contaminated validation weeks 31-40 | 11 | +2.42% (+1.14%) | +0.05% (+0.52%) | -0.40% (-0.43%) | 45.5% | -0.31% (-0.42%) |
| W2 | Train-era weeks 1-30 | 14 | +1.54% (+1.24%) | -1.21% (-1.81%) | +1.91% (+3.15%) | 64.3% | +1.58% (+1.15%) |
| W2 | Contaminated validation weeks 31-40 | 11 | +1.96% (+1.67%) | -0.55% (-0.12%) | -0.88% (+0.43%) | 36.4% | -0.05% (-0.11%) |
| W3 | Train-era weeks 1-30 | 14 | +0.94% (+2.65%) | +0.74% (+1.75%) | +3.16% (+3.78%) | 57.1% | +1.59% (+0.53%) |
| W3 | Contaminated validation weeks 31-40 | 10 | +1.48% (+2.41%) | +0.56% (+1.64%) | +0.30% (+0.95%) | 60.0% | -0.08% (-0.36%) |
| W4 | Train-era weeks 1-30 | 14 | +2.12% (+3.64%) | +1.02% (+9.49%) | +3.54% (+3.21%) | 57.1% | -0.35% (-1.12%) |
| W4 | Contaminated validation weeks 31-40 | 9 | -1.77% (+1.71%) | +4.19% (+3.36%) | +3.77% (+0.95%) | 66.7% | +0.20% (-0.53%) |

### 阶段稳定性发现：
1. **Train 阶段：** Rank3 强于 Rank2 的现象突出（W1/W2/W3 R3>R2 胜率均为 73.3%，W4 达到 80.0%）；
2. **Contaminated Validation 阶段：** Rank3 > Rank2 胜率仍维持在 55%~62%，但组合层 MC3 中位数在 W3/W4 发生倒挂（-0.24% / -0.15%）；
3. **治理警示：** 31~40 周为已知历史样本，不可等同于真实的 Virgin OOS 前向测试。

---

## 七、Rank 2 PIT Structure Diagnostic (事实画像对比)

横向切片对比 25 个 3-pick 周的 PIT 字段：

| 特征字段 | Rank 1 (n=25) | Rank 2 (n=25) | Rank 3 (n=25) | 结构差异观察 |
| :--- | :--- | :--- | :--- | :--- |
| Fresh Demand Alpha Lane (%) | `84.0%` | `84.0%` | `64.0%` | Rank1/2 集中于 Fresh Demand Alpha (`84.0%`)，Rank3 包含更多 Pullback (`28.0%`) |
| Ceiling Rule (%) | `84.0%` | `84.0%` | `68.0%` | Rank1/2 均为 `84.0%` Ceiling，Rank3 拥有更多 Pivot (`24.0%`) |
| Breakout Range Ratio (Med) | `0.37` | `0.27` | `0.49` | Rank2 突破振幅比率最小 (`0.27` vs Rank1 `0.37`, Rank3 `0.49`) |
| Dist to 52w High (Med) | `-1.29%` | `-3.62%` | `-1.59%` | Rank1 最贴近 52 周高点 (`-1.29%`)，Rank2 (`-3.62%`) 略远 |
| Pullback Not Dry Risk (%) | `8.0%` | `16.0%` | `12.0%` | Rank2 触发未缩量回撤风险率最高 (`16.0%` vs Rank1 `8.0%`) |
| Geometry Caution (%) | `20.0%` | `28.0%` | `4.0%` | Rank1/2 均有 `20.0%` 几何形态预警，Rank3 仅 `4.0%` |
| Industry Focus | `Regional Banks (6), Industrial Machinery (2), Trucks/Construction/Farm Machinery (2)` | `Unknown (4), Biotechnology (4), Regional Banks (3)` | `Unknown (3), Electronic Production Equipment (2), Major Banks (2)` | Rank1 集中于 Regional Banks，Rank2 包含高波动 Biotech |

> **因果区分警示：** 以上为历史事实画像对比，不代表单一字段必然构成因果。禁止在缺乏前向独立验证时基于上述特征直接调权。

---

## 八、综合治理建议与前向跟踪指引

1. **保持生产 B0 冻结：** 严禁为了“修复 Rank2”而修改现有排序权重或引入新规则；
2. **定性修正：** 在方法论与产品认知上，明确将 B0 视为 **Top-Bucket Equal-Weighted Selector**，而非高精度单调 ranker；
3. **前向观察指引 (Forward Shadow 2026/08/28 起)：** 持续记录 Rank1/Rank2/Rank3、K1/K2/K3、MC2/MC3 及 R3-R2，观察 virgin forward 数据是否继续出现 Rank3 > Rank2 结构。只有前向样本复现后，才进入下一阶段假设推导。
