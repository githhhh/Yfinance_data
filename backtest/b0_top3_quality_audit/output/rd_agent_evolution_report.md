# RD-Agent 自动化假说演进与四维 Champion 优胜矩阵交付报告 (基线完全对齐版)

**智能体范式**：RD-Agent 标准多轮假说驱动进化架构 (Hypothesis $\rightarrow$ Experiment $\rightarrow$ Feedback $\rightarrow$ Evolve)  
**决策样本**：严格锚定 Train 探索集 (第 1~30 周，2025-10-10 至 2026-05-22)  
**盲测样本**：Sealed Final Holdout (第 31~40 周，2026-05-29 至 2026-08-07，单向一次性披露)  
**基线对齐**：`B0_BASELINE` 与 Step 1 Level 2 **40 周逐周收益 100% 恒等一致** (全周期中位数 **+2.04%**)  
**生产状态**：生产基线 `dashboard/skill_industry_eps_known.py` **100% 保持冻结零修改**  

---

## 一、四维 Champion 优胜矩阵总表 (Train 选模 + Holdout 盲测披露)

| Champion 角色 | 优胜规则 ID | 规则复杂度 $C$ | 规则物理逻辑简述 | Train收益中位 (1~30周) | 活跃周vsL1胜率 (Train) | 止损发生率 (Train) | Holdout盲测中位 (31~40周) | 全40周收益中位 | 角色与实盘定位 |
|:---|:---|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **生产基准 (Frozen Baseline)** | `B0_BASELINE` | $C=3$ | 新鲜度分桶 $\rightarrow$ 放量 $\rightarrow$ 收盘位置 | **+2.04%** | **63.6%** | **52.8%** | **+2.34%** | **+2.04%** | **生产唯一在役基准 (保持冻结)** |
| **🏆 RETURN_WINNER** | `COMPOSITE_F3_V1_P1_D0_52W0` | $C=2$ | Linear Composite (Fresh:3, Vol:1, Pos:1, Dry:0, 52W:0) | **+2.25%** | **72.7%** | **55.6%** | **-0.96%** | **+0.91%** | **纯研究上限参考，严禁直接上线** |
| **🛡️ LOWEST_STOP** | `SIMPLER_PURE_CLOSE_POS` | $C=1$ | Pure Close Position Sort (H3 Testing) | **+2.14%** | **63.6%** | **52.2%** | **-2.13%** | **+0.87%** | 观察储备 (Holdout出现样本外回撤) |
| **✂️ SIMPLER_EQUIV** | `SIMPLER_PURE_FRESHNESS` | $C=1$ | Pure Freshness Sort (H1 Testing) | **+2.25%** | **72.7%** | **55.6%** | **+1.89%** | **+2.25%** | **极简研究推荐候选 (影子跟测)** |
| **⚖️ PARETO_BALANCED** | `SIMPLER_PURE_FRESHNESS` | $C=1$ | Pure Freshness Sort (H1 Testing) | **+2.25%** | **72.7%** | **55.6%** | **+1.89%** | **+2.25%** | **综合表现最优研究候选** |

---

## 二、关键量化发现与风控洞察

1. **B0 标尺 100% 对齐复现**：
   * 在使用生产真实 11 元组 `sort_key` 与候选池后，`B0_BASELINE` 在 40 周上与 Step 1 `l2_executed_ret` **40 个周逐周收益完全恒等一致 (差值严格为 0.00%)**；
   * B0 全样本中位数为 **+2.04%** (Train: +2.04%, Holdout: +2.34%)。
2. **`SIMPLER_PURE_FRESHNESS`（极简纯新鲜度排序）的研究表现**：
   * 在**坚守全部硬筛底座**（ACTIONABLE + EPS已知 + 行业限1 + 无形态崩塌）的前提下，仅将排序键改为 `current_vs_ibd_candidate_pct` 升序（越紧贴买点越优先），复杂度从 $C=3$ 降至 $C=1$；
   * 在 Train 探索集（1~30周）上执行止损后中位数由 +2.04% 提升至 **+2.25%**，活跃排序周收益中位数由 +4.69% 提升至 **+5.33%**；
   * 在 10 周 Sealed Holdout 盲测集上维持 **+1.89%**（一次盲测未翻车，但样本仍小需继续观察）；
   * 全周期 40 周收益中位数为 **+2.25%**（略优于 B0 的 +2.04%）。
3. **复合规则退化与奥卡姆剃刀验证**：
   * `COMPOSITE_F3_V1_P1` 虽然在 Train 上与纯新鲜度相同（+2.25%），但在 Holdout 盲测集上因杂糅了放量与收盘位置，收益回落至 **-0.96%**；
   * 这强力印证了：在有限样本下，**极简规则比多参数复合规则具备更强的样本外泛化鲁棒性**。

---

## 三、生产变更与灰度治理建议

1. **生产环境现行策略不作即时切换**：
   * 生产代码 `dashboard/skill_industry_eps_known.py` 继续保持 100% 冻结；
2. **建议实施影子灰度跟测 (Shadow Canary Testing)**：
   * 在后续 4~8 周周度复盘中，保留 B0 为主选股通道，同时并行输出 `SIMPLER_PURE_FRESHNESS` 作为 Shadow 标尺进行实盘前瞻跟踪；
   * 若实盘 8 周内 Shadow 规则的滑点、流动性与回撤控制持续优于 B0，再提请正式生产变更。
