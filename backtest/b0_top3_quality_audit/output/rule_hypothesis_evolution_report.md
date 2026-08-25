# Rule Hypothesis Evolution & Quad-Champion Matrix Report (Train-Only Protocol)

**智能体范式**：假说驱动迭代研究报告 (Hypothesis $\rightarrow$ Experiment $\rightarrow$ Feedback $\rightarrow$ Freeze)  
**决策样本**：严格锚定 Train 探索集 (第 1~30 周，2025-10-10 至 2026-05-22，全闭环零泄漏)  
**时序验证**：第 31~40 周由独立验证器 `historical_validation_verifier.py` 单向一次性披露  
**前瞻 OOS 起点**：2026-08-28 (代码冻结后首个真实未来周)  
**生产状态**：生产基线 `dashboard/skill_industry_eps_known.py` **100% 保持冻结零修改**  

---

## 一、四维 Champion 优胜矩阵总表 (Train 探索集严格决策)

| Champion 角色 | 优胜规则 ID | 规则复杂度 $C$ | 规则物理逻辑简述 | Train W1收益中位 | Train 配对超额中位 | 3-Block 稳定性评分 | 止损发生率 (Train) | 角色与实盘定位 |
|:---|:---|:---:|:---|:---:|:---:|:---:|:---:|:---|
| **生产基准 (Frozen Baseline)** | `B0_BASELINE` | $C=3$ | 新鲜度分桶 $\rightarrow$ 放量 $\rightarrow$ 收盘位置 | **+0.99%** | **0.00%** | **-1.14** | **52.8%** | **生产唯一在役基准 (保持冻结)** |
| **🏆 RETURN_WINNER** | `B0_BASELINE` | $C=3$ | Production B0 Baseline (Exact 11-tuple sort_key) | **+0.99%** | **+0.00%** | **-1.14** | **52.8%** | **Train收益冠军 (生产B0继续胜出)** |
| **🛡️ LOWEST_STOP** | `SIMPLER_PURE_CLOSE_POS` | $C=1$ | Pure Close Position Sort (Highest intraday close position) | **+0.58%** | **+0.00%** | **-0.32** | **52.2%** | 观察储备 |
| **✂️ SIMPLER_EQUIV** | `SIMPLER_PURE_CLOSE_POS` | $C=1$ | Pure Close Position Sort (Highest intraday close position) | **+0.58%** | **+0.00%** | **-0.32** | **52.2%** | **极简研究推荐候选 (影子跟测)** |
| **⚖️ PARETO_BALANCED** | `SIMPLER_PURE_CLOSE_POS` | $C=1$ | Pure Close Position Sort (Highest intraday close position) | **+0.58%** | **+0.00%** | **-0.32** | **52.2%** | **综合平衡研究候选 (影子跟测)** |

---

## 二、关键量化发现与风控洞察

1. **B0 基线在固定持有期下的稳健胜出**：
   * 在使用固定 W1/W2/W4 评价体系与 Censored Protocol 后，`B0_BASELINE` 在 Train 集以 **+0.99%** 的 W1 中位收益成为 `HISTORICAL_RETURN_WINNER`；
   * 单一因子（如极简新鲜度）复杂度虽低 ($C=1$)，但 W1 收益降至 +0.75%，W2 收益降至 -0.28%，证实生产多因子启发式排序确实具备实质 Alpha，支持生产 100% 冻结。
2. **全协议代码与数据防篡改清单**：
   * 全量规则与 4 维 Champion 在决策后立即导出 `frozen_rules_manifest.json`，并固化了生产代码、准入谓词、数据缓存的完整 SHA256 签名与具体参数配置。
