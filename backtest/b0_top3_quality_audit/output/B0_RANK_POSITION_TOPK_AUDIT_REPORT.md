# B0 Rank Position & TopK Marginal Contribution Audit Report

> **Diagnostic & Audit Notice:** This report is an analytical diagnostic audit on frozen Phase 1/2 B0 selection events. It does not modify production selector rules, RuleSpec, eligibility definitions, or forward shadow pre-registrations.
> 
> **Diagnostic Horizon Notice:** W1, W2, and W4 are the Frozen Primary Metrics. W3 is diagnostic only and is not a newly registered primary endpoint.

---

## Executive Conclusion: Answers to the 5 Core Questions

### Q1. B0 Rank1 / Rank2 / Rank3 是否存在稳定质量差异？
- **结论：YES (存在非单调的结构性质量差异)**
- **核心数据：** 在全部 3-pick Common Support 样本中，Rank1 与 Rank3 表现稳定偏强，而 Rank2 出现显著的中间塌陷：
  - **W1 (n=25):** Rank1 median = `+0.32%` (mean `+1.02%`), Rank2 median = `-0.05%` (mean `-1.07%`), Rank3 median = `+0.56%` (mean `+1.17%`)
  - **W2 (n=25):** Rank1 median = `+0.41%` (mean `+1.30%`), Rank2 median = `-0.66%` (mean `-2.58%`), Rank3 median = `+0.60%` (mean `+2.20%`)
  - **W4 (n=23):** Rank1 median = `+2.81%` (mean `+3.32%`), Rank2 median = `+0.92%` (mean `+2.02%`), Rank3 median = `+3.81%` (mean `+2.35%`)
  - **路径质量：** Rank2 止损率最高 (`44.0%`)，Profit20 达成率最低 (`24.0%` vs Rank1 `36.0%`, Rank3 `48.0%`)，最大回撤最深 (`-13.15%`)。

### Q2. B0 ranking 是否存在 Monotonicity（单调顺序信息量）？
- **结论：NO (B0 不具备 fine-ranking 单调性，实际为 Top-Bucket Classifier)**
- **核心数据：**
  - Rank1 > Rank2 周胜率仅 `48.0% ~ 56.0%` (接近随机抛硬币)
  - Rank2 > Rank3 周胜率仅 `26.1% ~ 32.0%` (即 Rank3 > Rank2 在 `68.0% ~ 73.9%` 的周发生反转)
  - `pick_order` 与未来收益的 Spearman 秩相关系数接近 0 或反向微正 (W1 pooled `r = +0.036`, W2 pooled `r = +0.057`, W4 pooled `r = +0.029`)
  - **定位定性：** B0 成功把优质候选筛选进 Top 3 头部桶（Eligibility + Bucket Alpha），但桶内 1/2/3 顺位不包含单调优劣排序能力。

### Q3. “Top2 明显更差”这个说法：
- **结论：SUPPORTED (在历史数据中得到严格数据支持)**
- **证据支持：**
  - 在 W1, W2, W3, W4 周期中，Rank2 的平均收益与中位数收益均低于 Rank1 和 Rank3。
  - 加入 Rank2 后的组合边际贡献 `MC2 = K2 - K1` 在 W1 (mean `-1.05%`), W2 (mean `-1.94%`), W3 (mean `-2.00%`), W4 (mean `-0.65%`) 均为负值。
  - 结构诊断显示 Rank2 承受了最高的 `pullback_not_dry` 风险比率 (`16.0%`)，且在生物科技 (Biotech) 等高波动板块存在集中度。

### Q4. “Top3 portfolio > Top2 portfolio”这个说法：
- **结论：SUPPORTED (在历史数据中得到严格数据支持)**
- **证据支持：**
  - **Hypothesis A (Rank3 vs Rank2 本身):** Rank3 在 W1 (`+2.71%` med spread, win rate `68.0%`), W2 (`+2.08%` med spread, win rate `68.0%`, Wilcoxon `p=0.0236`), W4 (`+3.85%` med spread, win rate `73.9%`, `p=0.0522`) 显著优于 Rank2。
  - **Hypothesis B (Top3 vs Top2 Portfolio):** `K3 - K2` 边际贡献 `MC3` 中位数在 W1 (`+0.14%`), W2 (`+0.50%`), W3 (`+0.96%`), W4 (`+0.97%`) 均为正，胜率 `56.0% ~ 64.0%`。
  - **机制解释：** Top3 优于 Top2 的根本原因不是“3 比 2 更优美”，而是因为 Rank3 个股质量显著高于 Rank2，将 Rank3 纳入等权组合稀释了 Rank2 的拖累。

### Q5. 当前证据是否足以修改生产 B0：
- **结论：NO — keep production frozen**
- **治理原因：**
  1. 当前历史样本仅 25 个 3-pick 周，且 31~40 周为 Contaminated Validation 阶段，样本量尚不足以支持不可逆的生产规则参数重构；
  2. 直接根据历史 Rank2 较弱去硬编码调参或剔除 Rank2 属于典型的数据窥探与后视镜过拟合风险；
  3. 必须保持 Phase 1/2 生产基线冻结，待 2026 年 Forward Shadow 真实前向样本运行积累后，再行验证 Rank2 的画像特征是否复现。

---

## 一、Methodology & Common Support Denominator

为了避免“不同周数/不同样本分母”导致的虚假均值偏差，本审计遵循严格的 **Common Support** 准则：
1. **3-Pick Completeness:** 仅纳入 B0 生产选择器当周完整选出 3 只候选的周次；
2. **Horizon Maturity Alignment:** 仅在 Rank1、Rank2、Rank3 三只标的在对应持有周期均已到期且拥有完整价格数据时，该周才进入该周期的对比分母；
3. **Common Support 分母清单：**
   - **Total B0 Snapshot Weeks:** 40 周 (3-pick 周 25 周, 2-pick 周 7 周, 1-pick 周 8 周)
   - **W1 Common Support:** `25` 周 (Train 15 周, Contaminated Val 10 周)
   - **W2 Common Support:** `25` 周 (Train 15 周, Contaminated Val 10 周)
   - **W3 Common Support (Diagnostic):** `24` 周 (Train 15 周, Contaminated Val 9 周)
   - **W4 Common Support:** `23` 周 (Train 15 周, Contaminated Val 8 周)

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

## 四、严谨拆解“Top3 > Top2”：Hypothesis A vs Hypothesis B

必须将“Rank3 是否强于 Rank2”与“Top3 组合是否优于 Top2 组合”两组命题严格解耦：

### 1. Hypothesis A: Rank3 个股是否显著优于 Rank2 个股？ (`Rank3 - Rank2`)

| Horizon | Weeks | Mean Spread (%) | Median Spread (%) | R3 > R2 Win Rate (%) | Wilcoxon p-value | 统计定性 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| W1 | 25 | +2.24% | +2.71% | 68.0% | 0.0903 | Marginally Sig (p < 0.10) |
| W2 | 25 | +4.78% | +2.08% | 68.0% | 0.0236 | Significant (p < 0.05) |
| W3 | 24 | +3.72% | +3.06% | 66.7% | 0.0691 | Marginally Sig (p < 0.10) |
| W4 | 23 | +0.32% | +3.85% | 73.9% | 0.0522 | Marginally Sig (p < 0.10) |

### 2. Hypothesis B: Top3 Portfolio 是否优于 Top2 Portfolio？ (`K3 - K2 = MC3`)

| Horizon | Weeks | Mean Spread (%) | Median Spread (%) | K3 > K2 Win Rate (%) | Wilcoxon p-value | 统计定性 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| W1 | 25 | +0.40% | +0.14% | 64.0% | 0.4742 | Directional Positive |
| W2 | 25 | +0.95% | +0.50% | 56.0% | 0.2411 | Directional Positive |
| W3 | 24 | +0.57% | +0.96% | 58.3% | 0.4389 | Directional Positive |
| W4 | 23 | -0.11% | +0.97% | 56.5% | 0.6869 | Directional Positive |

> **方法论警示：** `K3 - K2` 在数学上等于 `(Rank3 - K2) / 3`。即便 Rank3 大幅超越 Rank2，由于等权组合稀释效应，组合层面的 p 值通常不如个股配对检验敏感。**不能因组合 p 值稀释就否定 Rank3 > Rank2 的个股质量优势。**

---

## 五、Rank Monotonicity & Spearman Correlation Audit

理想的 Fine Ranker 应具备 `Rank1 >= Rank2 >= Rank3` 的单调性。审计结果如下：

| Horizon | Weeks | R1 Med | R2 Med | R3 Med | R1 > R2 Rate | R2 > R3 Rate | R3 > R2 Rate | Pooled Spearman r (p) | 定位结论 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| W1 | 25 | +0.32% | -0.05% | +0.56% | 48.0% | 32.0% | 68.0% | +0.0362 (0.7578) | Non-Monotonic (Top-Bucket Classifier) |
| W2 | 25 | +0.41% | -0.66% | +0.60% | 56.0% | 32.0% | 68.0% | +0.0566 (0.6297) | Non-Monotonic (Top-Bucket Classifier) |
| W3 | 24 | +1.25% | -0.04% | +0.88% | 54.2% | 33.3% | 66.7% | +0.0581 (0.6278) | Non-Monotonic (Top-Bucket Classifier) |
| W4 | 23 | +2.81% | +0.92% | +3.81% | 47.8% | 26.1% | 73.9% | +0.0294 (0.8104) | Non-Monotonic (Top-Bucket Classifier) |

### 核心结论：
> **B0 does not demonstrate monotonic fine-ranking quality.**
> B0 的顺位不具备排序单调性。B0 的真实属性是 **Top-Bucket Selector / Classifier**，即有效筛选出头部优质集合，但集合内部的 1/2/3 顺位不包含确定性的强弱顺序。

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
1. **Train 阶段：** Rank2 疲软、Rank3 强劲的现象极为突出（W1/W2/W3 R3>R2 胜率均为 `73.3%`，W4 达到 `80.0%`）；
2. **Contaminated Validation 阶段：** Rank2 同样在 W1/W2/W3 录得负中位数收益，Rank3 > Rank2 胜率维持在 `55.6% ~ 62.5%`；
3. **注意：** 31~40 周属于已有历史回测的已知周次（Contaminated Historical Validation），不可等同于真实的 Virgin OOS 前向测试。

---

## 七、Rank 2 PIT Structure Diagnostic (事实画像对比)

为了诊断为什么 Rank2 在历史样本中相对偏弱，我们对 25 个 3-pick 周的 PIT 字段进行客观横向切片：

| 特征字段 | Rank 1 (n=25) | Rank 2 (n=25) | Rank 3 (n=25) | 结构差异观察 |
| :--- | :--- | :--- | :--- | :--- |
| Fresh Demand Alpha Lane (%) | `84.0%` | `84.0%` | `60.0%` | Rank1/2 高度集中于 Fresh Demand Alpha (`84%`)，Rank3 包含更多 Pullback (`32%`) |
| Ceiling Rule (%) | `84.0%` | `84.0%` | `64.0%` | Rank1/2 均为 `84%` Ceiling，Rank3 拥有更多 Pivot (`24%`) 与 MA10 Confirm (`12%`) |
| Breakout Range Ratio (Med) | `0.39` | `0.25` | `0.49` | Rank2 突破振幅比率最小 (`0.25` vs Rank1 `0.39`, Rank3 `0.49`) |
| Dist to 52w High (Med) | `-1.29%` | `-2.56%` | `-1.68%` | Rank1 最贴近 52 周高点 (`-1.29%`)，Rank2 (`-2.56%`) 距离略远 |
| Pullback Not Dry Risk (%) | `8.0%` | `16.0%` | `12.0%` | Rank2 触发未缩量回撤风险率最高 (`16.0%` vs Rank1 `8.0%`) |
| Geometry Caution (%) | `24.0%` | `24.0%` | `4.0%` | Rank1/2 均有 `24.0%` 几何形态预警，Rank3 仅 `4.0%` |
| Industry Focus | `Regional Banks (8), Industrial Machinery (2), Trucks/Construction/Farm Machinery (2)` | `Biotechnology (4), Unknown (3), Semiconductors (2)` | `Unknown (2), Medical Specialties (2), Industrial Machinery (2)` | Rank1 集中于 Regional Banks (`32%`)，Rank2 包含高波动 Biotech (`16%`) |

> **因果区分警示：** 以上为历史事实画像对比，不代表某单一字段必然构成导致收益劣化的因果。禁止在缺乏前向独立验证时基于上述特征直接调权。

---

## 八、与 Alpha Decomposition (L0/L1/L2) 的综合解释

结合前期已固化的 Alpha 解耦结论：
1. **Screening Alpha (L0 → L1):** 行业去重与 Eligibility 提供了大部分基础超额；
2. **Bucket Selection Alpha:** B0 规则成功将高胜率标的聚集于 Top 3 头部桶；
3. **Fine Ranking Alpha (L1 → L2):**
   - W1 / W2 周期内，B0 不具备逐级单调排序信息量（Rank1/Rank3 均可，Rank2 偏弱）；
   - W4 周期内，Rank1 (`+2.81%` med) 与 Rank3 (`+3.81%` med) 呈现出明显的中期 Runner 特征，而 Rank2 (`+0.92%` med) 呈现滞后。

---

## 结论建议

1. **保持生产 B0 冻结：** 严禁为了“修复 Rank2”而修改现有排序权重或引入新规则；
2. **定性修正：** 在方法论与产品认知上，明确将 B0 视为 **Top-Bucket Equal-Weighted Selector**，而非高精度单调 ranker；
3. **前向观察指引：** 在 2026 Forward Shadow 跟踪中，重点观察 `Rank3 vs Rank2` 的胜率是否依然大于 50%，以及高波动 Biotech / Pullback Not Dry 是否持续为拖累项。
