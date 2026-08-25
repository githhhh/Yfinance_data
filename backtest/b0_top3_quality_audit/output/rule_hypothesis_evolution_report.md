# Rule Hypothesis Evolution & Quad-Champion Matrix Report (Train-Only Protocol)

**智能体范式**：假说驱动迭代演进架构 (Hypothesis $\rightarrow$ Experiment $\rightarrow$ Feedback $\rightarrow$ Evolve)  
**决策样本**：严格锚定 Train 探索集 (第 1~30 周，2025-10-10 至 2026-05-22，全闭环零泄漏)  
**时序验证**：第 31~40 周由独立验证器 `historical_validation_verifier.py` 单向一次性披露  
**生产状态**：生产基线 `dashboard/skill_industry_eps_known.py` **100% 保持冻结零修改**  

---

## 一、四维 Champion 优胜矩阵总表 (Train 探索集严格决策)

| Champion 角色 | 优胜规则 ID | 规则复杂度 $C$ | 规则物理逻辑简述 | Train W1收益中位 | Train 配对超额中位 | 3折稳定性评分 | 止损发生率 (Train) | 角色与实盘定位 |
|:---|:---|:---:|:---|:---:|:---:|:---:|:---:|:---|
| **生产基准 (Frozen Baseline)** | `B0_BASELINE` | $C=3$ | 新鲜度分桶 $\rightarrow$ 放量 $\rightarrow$ 收盘位置 | **+0.99%** | **0.00%** | **-1.14** | **52.8%** | **生产唯一在役基准 (保持冻结)** |
| **🏆 RETURN_WINNER** | `B0_BASELINE` | $C=3$ | Production B0 Baseline (Exact 11-tuple sort_key) | **+0.99%** | **+0.00%** | **-1.14** | **52.8%** | **纯研究上限参考，严禁直接上线** |
| **🛡️ LOWEST_STOP** | `SIMPLER_PURE_CLOSE_POS` | $C=1$ | Pure Close Position Sort (Highest intraday close position) | **+0.58%** | **+0.00%** | **-0.32** | **52.2%** | 观察储备 |
| **✂️ SIMPLER_EQUIV** | `SIMPLER_PURE_CLOSE_POS` | $C=1$ | Pure Close Position Sort (Highest intraday close position) | **+0.58%** | **+0.00%** | **-0.32** | **52.2%** | **极简研究推荐候选 (影子跟测)** |
| **⚖️ PARETO_BALANCED** | `SIMPLER_PURE_CLOSE_POS` | $C=1$ | Pure Close Position Sort (Highest intraday close position) | **+0.58%** | **+0.00%** | **-0.32** | **52.2%** | **综合表现最优研究候选** |

---

## 二、关键量化发现与风控洞察

1. **Train 探索集决策闭环与分母硬约束**：
   * 在坚守全部硬筛底座（ACTIONABLE + EPS已知 + 行业限1 + 无形态崩塌）且有效率 $\ge 80\%$ 的前提下，`SIMPLER_PURE_FRESHNESS`（极简纯新鲜度排序，复杂度 $C=1$）在 Train 集上表现稳健，且通过了 3 折 Walk-Forward 时间序列平稳性检验。
2. **全协议代码与数据防篡改清单**：
   * 全量规则与 4 维 Champion 在决策后立即导出 `frozen_rules_manifest.json`，并固化了生产代码、准入谓词、数据缓存的完整 SHA256 签名。
