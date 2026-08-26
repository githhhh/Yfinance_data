# B0 Rank Position & TopK Marginal Contribution Audit Report

> **Diagnostic & Audit Notice:** This report is an analytical diagnostic audit on frozen Phase 1/2 B0 selection events. It does not modify production selector rules, RuleSpec, eligibility definitions, or forward shadow pre-registrations.
> 
> **Diagnostic Horizon Notice:** W1, W2, and W4 are the Frozen Primary Metrics. W3 is diagnostic only and is not a newly registered primary endpoint.

---

## Executive Conclusion: Answers to the 5 Core Questions

### Q1. B0 Rank1 / Rank2 / Rank3 是否存在稳定质量差异？
- **结论：PARTIALLY SUPPORTED (HISTORICAL DIAGNOSTIC SIGNAL)**
- **核心数据（动态提取）：** 在全部 3-pick Common Support 样本中，Rank1 与 Rank3 在多数周期收益与路径表现较强，而 Rank2 出现非单调的中间塌陷：
  - **W1 (n=25):** Rank1 med=`+0.32%` (mean `+1.02%`), Rank2 med=`-0.05%` (mean `-1.07%`), Rank3 med=`+0.56%` (mean `+1.17%`)
  - **W2 (n=25):** Rank1 med=`+0.41%` (mean `+1.30%`), Rank2 med=`-0.66%` (mean `-2.58%`), Rank3 med=`+0.60%` (mean `+2.20%`)
  - **W3 (诊断) (n=24):** Rank1 med=`+1.25%` (mean `+2.93%`), Rank2 med=`-0.04%` (mean `-1.08%`), Rank3 med=`+0.88%` (mean `+2.64%`)
  - **W4 (n=23):** Rank1 med=`+2.81%` (mean `+3.32%`), Rank2 med=`+0.92%` (mean `+2.02%`), Rank3 med=`+3.81%` (mean `+2.35%`)
  - **路径质量：** Rank2 止损率最高 (`44.0%`)，Profit20 达成率最低 (`24.0%` vs Rank1 `36.0%`, Rank3 `48.0%`)，最大回撤中位数达 `-7.55%`。
  - **定性定界：** 该差异呈现 U 型非单调结构，属于历史诊断特征，不能简单视作线性排序能力。

### Q2. B0 ranking 是否存在 Monotonicity（单调顺序信息量）？
- **结论：NOT SUPPORTED (B0 不具备 fine-ranking 单调性，实际表现为 Top-Bucket Selector)**
- **核心数据（动态提取）：**
  - **Rank1 > Rank2 周胜率：** W1: `48.0%`, W2: `56.0%`, W4: `47.8%` (无统计优势，接近抛硬币)
  - **Rank2 > Rank3 周胜率：** W1: `32.0%`, W2: `32.0%`, W4: `26.1%` (发生实质性倒挂，Rank3 > Rank2 在多数周发生)
  - **Pooled Spearman 秩相关系数：** W1: `r=+0.0362`, W2: `r=+0.0566`, W4: `r=+0.0294` (均接近 0，无单调预测力)
  - **定位定性：** B0 在历史样本中表现为 **Top-Bucket Selector**（将优质标的筛入 Top3 头部集合），但集合内部的 1/2/3 顺位不包含单调优劣信息。

### Q3. “Top2 明显更差”这个说法：
- **结论：PARTIALLY SUPPORTED**
- **证据分层分析（动态提取）：**
  - **Rank3 vs Rank2 (支持性强):** Rank3 在 W2 显著战胜 Rank2 (中位数利差 `+2.08%`, 胜率 `68.0%`, Wilcoxon $p=0.0236$)，W1/W3/W4 均呈现明显正向利差。
  - **Rank1 vs Rank2 (支持性弱):** Rank1 相对于 Rank2 未达到双侧统计显著水平，周胜率仅在 50% 附近波动。
  - **MC2 边际贡献分布 (左尾拖累而非每周恶化):** W1 median MC2 为 `+0.04%` (mean `-1.05%`)，W4 median MC2 为 `+0.40%` (mean `-0.65%`)。中位数在 W1/W4 为正，说明 Rank2 表现弱主要是由少数严重亏损的左尾事件（如生物科技板块/未缩量回撤）拉低均值，而非每周系统性拖累。

### Q4. “Top3 portfolio > Top2 portfolio”这个说法：
- **结论：PARTIALLY SUPPORTED / DIRECTIONAL ONLY**
- **证据分层分析（动态提取）：**
  - **全历史样本中位数偏正：** W1 MC3 med=`+0.14%` ($p=0.4742$), W2 MC3 med=`+0.50%` ($p=0.2411$), W4 MC3 med=`+0.97%` ($p=0.6869$)。
  - **无统计显著性且均值衰减：** 组合层 Wilcoxon $p$ 值在全周期均未达到显著水平；W4 平均边际贡献已转负 (`-0.11%`)。
  - **后期验证段倒挂：** 在历史验证周次 (31~40) 中，W3 median MC3 为 `-0.24%`，W4 median MC3 为 `-0.15%`，优势未能稳定延续。
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
   - **W1 Common Support:** `25` 周 (Train `15` 周, Contaminated Val `10` 周)
   - **W2 Common Support:** `25` 周 (Train `15` 周, Contaminated Val `10` 周)
   - **W3 (Diagnostic Only) Common Support:** `24` 周 (Train `15` 周, Contaminated Val `9` 周)
   - **W4 Common Support:** `23` 周 (Train `15` 周, Contaminated Val `8` 周)

---

## 二、Position Quality Audit (Rank1 / Rank2 / Rank3)

### 1. Return Quality Summary across All Common-Support Weeks

| Horizon | Rank | Weeks | Mean (%) | Median (%) | P25 (%) | P75 (%) | Win Rate (>0) | Profit20 Rate | Stop8 Rate | Max Drawdown (Med) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| W1 | Rank1 | 25 | +1.02% | +0.32% | -2.08% | +2.79% | 52.0% | 36.0% | 40.0% | -6.75% |
| W1 | Rank2 | 25 | -1.07% | -0.05% | -3.25% | +2.28% | 44.0% | 24.0% | 44.0% | -7.55% |
| W1 | Rank3 | 25 | +1.17% | +0.56% | -2.09% | +3.76% | 60.0% | 48.0% | 36.0% | -4.70% |
| W2 | Rank1 | 25 | +1.30% | +0.41% | -3.96% | +6.44% | 52.0% | 36.0% | 40.0% | -6.75% |
| W2 | Rank2 | 25 | -2.58% | -0.66% | -4.93% | +1.42% | 44.0% | 24.0% | 44.0% | -7.55% |
| W2 | Rank3 | 25 | +2.20% | +0.60% | -1.50% | +6.77% | 60.0% | 48.0% | 36.0% | -4.70% |
| W3 | Rank1 | 24 | +2.93% | +1.25% | -3.24% | +5.65% | 58.3% | 37.5% | 41.7% | -7.05% |
| W3 | Rank2 | 24 | -1.08% | -0.04% | -4.50% | +2.67% | 45.8% | 25.0% | 41.7% | -7.33% |
| W3 | Rank3 | 24 | +2.64% | +0.88% | -1.33% | +5.87% | 62.5% | 50.0% | 37.5% | -4.56% |
| W4 | Rank1 | 23 | +3.32% | +2.81% | -2.55% | +9.54% | 60.9% | 39.1% | 43.5% | -7.35% |
| W4 | Rank2 | 23 | +2.02% | +0.92% | -7.38% | +3.13% | 56.5% | 26.1% | 43.5% | -7.55% |
| W4 | Rank3 | 23 | +2.35% | +3.81% | -2.31% | +7.67% | 69.6% | 52.2% | 39.1% | -4.70% |

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
| W1 | 25 | +0.32% (+1.02%) | +0.10% (-0.02%) | +0.62% (+0.37%) | +0.04% (-1.05%) | 52.0% | 0.2411 | +0.14% (+0.40%) | 64.0% | 0.4742 |
| W2 | 25 | +0.41% (+1.30%) | -0.56% (-0.64%) | +0.10% (+0.31%) | -0.43% (-1.94%) | 44.0% | 0.1485 | +0.50% (+0.95%) | 56.0% | 0.2411 |
| W3 | 24 | +1.25% (+2.93%) | -1.23% (+0.92%) | +1.01% (+1.50%) | -1.22% (-2.00%) | 45.8% | 0.3305 | +0.96% (+0.57%) | 58.3% | 0.4389 |
| W4 | 23 | +2.81% (+3.32%) | +0.99% (+2.67%) | +1.80% (+2.56%) | +0.40% (-0.65%) | 52.2% | 0.5202 | +0.97% (-0.11%) | 56.5% | 0.6869 |

---

## 四、配对假设检验：Hypothesis A, B, C 全景对比

### 1. Hypothesis A: Rank3 个股是否优于 Rank2？ (`Rank3 - Rank2`)

| Horizon | Weeks | Mean Spread (%) | Median Spread (%) | R3 > R2 Win Rate (%) | Wilcoxon p-value | 95% Bootstrap CI | 统计定性 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| W1 | 25 | +2.24% | +2.71% | 68.0% | 0.0903 | [-0.32%, +4.87%] | Marginally Sig (p < 0.10) |
| W2 | 25 | +4.78% | +2.08% | 68.0% | 0.0236 | [+1.54%, +8.40%] | Significant (p < 0.05) |
| W3 | 24 | +3.72% | +3.06% | 66.7% | 0.0691 | [-0.51%, +8.12%] | Marginally Sig (p < 0.10) |
| W4 | 23 | +0.32% | +3.85% | 73.9% | 0.0522 | [-13.08%, +10.05%] | Marginally Sig (p < 0.10) |

### 2. Hypothesis B: Top3 Portfolio 是否优于 Top2 Portfolio？ (`K3 - K2 = MC3`)

| Horizon | Weeks | Mean Spread (%) | Median Spread (%) | K3 > K2 Win Rate (%) | Wilcoxon p-value | 95% Bootstrap CI | 统计定性 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| W1 | 25 | +0.40% | +0.14% | 64.0% | 0.4742 | [-0.50%, +1.37%] | Directional Positive (Not Sig) |
| W2 | 25 | +0.95% | +0.50% | 56.0% | 0.2411 | [-0.20%, +2.16%] | Directional Positive (Not Sig) |
| W3 | 24 | +0.57% | +0.96% | 58.3% | 0.4389 | [-1.00%, +2.13%] | Directional Positive (Not Sig) |
| W4 | 23 | -0.11% | +0.97% | 56.5% | 0.6869 | [-3.33%, +2.62%] | Directional Positive (Not Sig) |

### 3. Hypothesis C: Rank1 个股是否优于 Rank2？ (`Rank1 - Rank2`)

| Horizon | Weeks | Mean Spread (%) | Median Spread (%) | R1 > R2 Win Rate (%) | Wilcoxon p-value | 95% Bootstrap CI | 统计定性 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| W1 | 25 | +2.10% | -0.09% | 48.0% | 0.2411 | [-0.20%, +4.56%] | Not Significant |
| W2 | 25 | +3.88% | +0.86% | 56.0% | 0.1485 | [-0.09%, +8.12%] | Not Significant |
| W3 | 24 | +4.01% | +2.45% | 54.2% | 0.3305 | [-1.97%, +10.48%] | Not Significant |
| W4 | 23 | +1.30% | -0.80% | 47.8% | 0.5202 | [-10.02%, +11.05%] | Not Significant |

> **方法论洞察：** Rank3 相比 Rank2 的优势在统计上显著高于 Rank1 相比 Rank2 的优势。这进一步印证了 Rank2 的弱势并非简单的阶梯递减，而是 Rank3 具有独特的路径恢复能力。

---

## 五、Rank Monotonicity & Spearman Correlation Audit

理想的 Fine Ranker 应具备 `Rank1 >= Rank2 >= Rank3` 的单调性。审计结果如下：

| Horizon | Weeks | R1 Med | R2 Med | R3 Med | R1 > R2 Rate | R2 > R3 Rate | R3 > R2 Rate | Pooled Spearman r (p) | 定位结论 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| W1 | 25 | +0.32% | -0.05% | +0.56% | 48.0% | 32.0% | 68.0% | +0.0362 (0.7578) | Non-Monotonic (Top-Bucket Selector) |
| W2 | 25 | +0.41% | -0.66% | +0.60% | 56.0% | 32.0% | 68.0% | +0.0566 (0.6297) | Non-Monotonic (Top-Bucket Selector) |
| W3 | 24 | +1.25% | -0.04% | +0.88% | 54.2% | 33.3% | 66.7% | +0.0581 (0.6278) | Non-Monotonic (Top-Bucket Selector) |
| W4 | 23 | +2.81% | +0.92% | +3.81% | 47.8% | 26.1% | 73.9% | +0.0294 (0.8104) | Non-Monotonic (Top-Bucket Selector) |

### 核心结论：
> **B0 does not demonstrate monotonic fine-ranking quality.**
> B0 在历史样本中表现为 **Top-Bucket Selector** 而非 Fine Ranker。它成功将优质标的聚集于 Top 3 头部集合，但内部 1/2/3 顺位不包含确定性的强弱顺序。

---

## 六、分阶段稳定性：Train (1~30) vs Contaminated Historical Validation (31~40)

| Horizon | Segment | Weeks | R1 Med (Mean) | R2 Med (Mean) | R3 Med (Mean) | R3 > R2 Win Rate | MC3 Med (Mean) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| W1 | Train-era weeks 1-30 | 15 | +0.73% (+1.38%) | -0.03% (-1.02%) | +1.23% (+2.04%) | 73.3% | +0.53% (+0.62%) |
| W1 | Contaminated validation weeks 31-40 | 10 | +0.14% (+0.50%) | -0.49% (-1.15%) | -0.25% (-0.13%) | 60.0% | +0.11% (+0.07%) |
| W2 | Train-era weeks 1-30 | 15 | +0.41% (+2.15%) | +0.25% (-2.67%) | +1.15% (+3.07%) | 73.3% | +1.47% (+1.11%) |
| W2 | Contaminated validation weeks 31-40 | 10 | +0.85% (+0.02%) | -0.76% (-2.45%) | -0.94% (+0.90%) | 60.0% | +0.25% (+0.71%) |
| W3 | Train-era weeks 1-30 | 15 | +1.23% (+2.94%) | +0.21% (-0.50%) | +2.67% (+3.46%) | 73.3% | +1.40% (+0.75%) |
| W3 | Contaminated validation weeks 31-40 | 9 | +1.42% (+2.91%) | -0.34% (-2.04%) | -0.91% (+1.28%) | 55.6% | -0.24% (+0.28%) |
| W4 | Train-era weeks 1-30 | 15 | +2.66% (+3.61%) | +0.92% (+4.15%) | +4.21% (+4.03%) | 80.0% | +1.44% (+0.05%) |
| W4 | Contaminated validation weeks 31-40 | 8 | +3.50% (+2.77%) | +0.58% (-1.97%) | +2.98% (-0.81%) | 62.5% | -0.15% (-0.40%) |

### 阶段稳定性发现：
1. **Train 阶段：** Rank3 强于 Rank2 的现象突出（W1/W2/W3 R3>R2 胜率均为 73.3%，W4 达到 80.0%）；
2. **Contaminated Validation 阶段：** Rank3 > Rank2 胜率仍维持在 55%~62%，但组合层 MC3 中位数在 W3/W4 发生倒挂（-0.24% / -0.15%）；
3. **治理警示：** 31~40 周为已知历史样本，不可等同于真实的 Virgin OOS 前向测试。

---

## 七、Rank 2 PIT Structure Diagnostic (事实画像对比)

横向切片对比 25 个 3-pick 周的 PIT 字段：

| 特征字段 | Rank 1 (n=25) | Rank 2 (n=25) | Rank 3 (n=25) | 结构差异观察 |
| :--- | :--- | :--- | :--- | :--- |
| Fresh Demand Alpha Lane (%) | `84.0%` | `84.0%` | `60.0%` | Rank1/2 集中于 Fresh Demand Alpha (`84.0%`)，Rank3 包含更多 Pullback (`32.0%`) |
| Ceiling Rule (%) | `84.0%` | `84.0%` | `64.0%` | Rank1/2 均为 `84.0%` Ceiling，Rank3 拥有更多 Pivot (`24.0%`) |
| Breakout Range Ratio (Med) | `0.39` | `0.25` | `0.49` | Rank2 突破振幅比率最小 (`0.25` vs Rank1 `0.39`, Rank3 `0.49`) |
| Dist to 52w High (Med) | `-1.29%` | `-2.56%` | `-1.68%` | Rank1 最贴近 52 周高点 (`-1.29%`)，Rank2 (`-2.56%`) 略远 |
| Pullback Not Dry Risk (%) | `8.0%` | `16.0%` | `12.0%` | Rank2 触发未缩量回撤风险率最高 (`16.0%` vs Rank1 `8.0%`) |
| Geometry Caution (%) | `24.0%` | `24.0%` | `4.0%` | Rank1/2 均有 `24.0%` 几何形态预警，Rank3 仅 `4.0%` |
| Industry Focus | `Regional Banks (8), Industrial Machinery (2), Trucks/Construction/Farm Machinery (2)` | `Biotechnology (4), Unknown (3), Semiconductors (2)` | `Unknown (2), Medical Specialties (2), Industrial Machinery (2)` | Rank1 集中于 Regional Banks，Rank2 包含高波动 Biotech |

> **因果区分警示：** 以上为历史事实画像对比，不代表单一字段必然构成因果。禁止在缺乏前向独立验证时基于上述特征直接调权。

---

## 八、综合治理建议与前向跟踪指引

1. **保持生产 B0 冻结：** 严禁为了“修复 Rank2”而修改现有排序权重或引入新规则；
2. **定性修正：** 在方法论与产品认知上，明确将 B0 视为 **Top-Bucket Equal-Weighted Selector**，而非高精度单调 ranker；
3. **前向观察指引 (Forward Shadow 2026/08/28 起)：** 持续记录 Rank1/Rank2/Rank3、K1/K2/K3、MC2/MC3 及 R3-R2，观察 virgin forward 数据是否继续出现 Rank3 > Rank2 结构。只有前向样本复现后，才进入下一阶段假设推导。
