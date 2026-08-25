# Phase 2 Step 4: 四维 Pareto 优胜规则矩阵报告 (Train 集基线完全对齐版)

**决策样本**：严格锚定 Train 探索集 (第 1~30 周，2025-10-10 至 2026-05-22)  
**历史验证样本**：第 31~40 周 (2026-05-29 至 2026-08-07，受污染历史验证集，由 Historical Verifier 独立单向出表)  
**前瞻 OOS 起点**：2026-08-28 (代码冻结后首个真实未来周)  
**基线对齐**：`B0_BASELINE` 在 Train 集 W1 收益中位数 **+0.99%**  
**生产状态**：生产基线 `dashboard/skill_industry_eps_known.py` **100% 保持冻结零修改**  

---

## 一、四维 Champion 优胜矩阵总表 (Train 1~30 周)

| Champion 角色 | 优胜规则 ID | 规则复杂度 $C$ | 规则逻辑简述 | Train W1 收益中位 | Train 活跃周vsL1胜率 | Train 止损发生率 | Train 3-Block 稳定性 | 角色与实盘定位 |
|:---|:---|:---:|:---|:---:|:---:|:---:|:---:|:---|
| **生产基线 (Baseline)** | `B0_BASELINE` | $C=3$ | 新鲜度分桶 → 放量 → 收盘位置 | **+0.99%** | **54.5%** | **52.8%** | **-1.1425** | **生产唯一在役基准 (保持冻结)** |
| **🏆 RETURN_WINNER** | `B0_BASELINE` | $C=3$ | Production B0 Baseline (Exact 11-tuple sort_key) | **+0.99%** | **54.5%** | **52.8%** | **-1.1425** | **Train收益冠军 (生产B0继续胜出)** |
| **🛡️ LOWEST_STOP** | `SIMPLER_PURE_CLOSE_POS` | $C=1$ | Pure Close Position Sort (Highest intraday close position) | **+0.58%** | **63.6%** | **52.2%** | **-0.3241** | 观察储备 |
| **✂️ SIMPLER_EQUIV** | `SIMPLER_PURE_CLOSE_POS` | $C=1$ | Pure Close Position Sort (Highest intraday close position) | **+0.58%** | **63.6%** | **52.2%** | **-0.3241** | **极简研究候选 (Shadow)** |
| **⚖️ PARETO_BALANCED** | `SIMPLER_PURE_CLOSE_POS` | $C=1$ | Pure Close Position Sort (Highest intraday close position) | **+0.58%** | **63.6%** | **52.2%** | **-0.3241** | **综合平衡研究候选 (Shadow)** |

---

## 二、关键量化发现与风控洞察

1. **B0 标尺在搜索空间中的稳固地位**：
   * 在使用固定 W1/W2/W4 评价体系与真实 Censored Protocol 后，`B0_BASELINE` 在 Train 集以 **+0.99%** 的 W1 中位收益成为 `HISTORICAL_RETURN_WINNER`；
   * 在当前 85 个候选规则中，B0 在绝对 W1 收益上名列第一；但需客观指出，在活跃排序周其 W1 排序利差中位数仍为 0.00%（Wilcoxon $p=0.433$），尚未在统计上证实存在稳定的 W1 排序 Alpha。
2. **W4 跨周期信号展望**：
   * 在 38 个成熟周与 19 个活跃排序周中，B0 在 W4 显现出具有潜力的排序超额（配对利差中位 **+2.08%**，Wilcoxon $p=0.0299$），定性为 **Promising W4 Signal**，值得在后续前瞻影子账本中持续验证。
3. **生产零改动决策**：
   * 生产 selector 继续 100% 保持冻结；
   * 简化候选（Freshness 与 Close Position）作为 2026-08-28 起的前瞻影子账本（Forward Shadow）观察标的。
