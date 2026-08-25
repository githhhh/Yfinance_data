# Contaminated Historical Validation Audit Report (Weeks 31~40)

**测试性质**：历史验证集单向出表 (One-Way Historical Validation Disclosure)  
**样本窗口**：第 31~40 周 (2026-05-29 至 2026-08-07，共 10 周)  
**输入来源**：`frozen_rules_manifest.json` (SHA256: `9554c84d368b1c6b30366c794254eae62623986a3e826999a6c97a939bd2417f`)  
**隔离原则**：本报告仅单向输出审计数据，**严禁反馈回演进引擎进行调参或规则重构**。

> [!WARNING]
> **方法论定位说明（Methodology Caveat）**：
> 鉴于前期研发探索已接触过第 31~40 周数据，本报告中的表现严格定性为 **“已被污染的历史验证集（Contaminated Historical Validation）”**，不能作为纯粹的盲测样本外证据。
> 真正的无偏样本外验证严格建立在 **2026-08-28 之后的实时前瞻 Shadow 跟测账本**。

---

## 一、历史验证集 (Weeks 31~40) 表现总表

| 角色 / 规则 | 规则 ID | 复杂度 $C$ | W1 收益中位 | W2 收益中位 | W4 收益中位 | 全周期收益中位 | 止损发生率 | vs L1 胜率 | 定位与建议 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **PRODUCTION_BASELINE** | `B0_BASELINE` | $C=3$ | **+0.14%** | +0.03% | +0.23% | **+2.34%** | 30.0% | 50.0% | 生产继续保持 100% 冻结 |
| **HISTORICAL_RETURN_WINNER** | `B0_BASELINE` | $C=3$ | **+0.14%** | +0.03% | +0.23% | **+2.34%** | 30.0% | 50.0% | 生产继续保持 100% 冻结 |
| **LOWEST_STOP_CANDIDATE** | `SIMPLER_PURE_CLOSE_POS` | $C=1$ | **-1.93%** | -1.71% | -2.72% | **-2.13%** | 40.0% | 20.0% | 研究观察候选 (Shadow) |
| **SIMPLER_EQUIVALENT** | `SIMPLER_PURE_CLOSE_POS` | $C=1$ | **-1.93%** | -1.71% | -2.72% | **-2.13%** | 40.0% | 20.0% | 研究观察候选 (Shadow) |
| **PARETO_BALANCED_RULE** | `SIMPLER_PURE_CLOSE_POS` | $C=1$ | **-1.93%** | -1.71% | -2.72% | **-2.13%** | 40.0% | 20.0% | 研究观察候选 (Shadow) |

---

## 二、审计结论与后续行动

1. **生产基准零修改**：`dashboard/skill_industry_eps_known.py` 继续 100% 冻结不变。
2. **时序分期与影子跟测账本 (Forward Shadow Ledger)**：
   * **Pre-Freeze Replay (2026-08-14 与 2026-08-21)**：作为规则冻结前的回放测试周；
   * **Forward Shadow Kickoff (2026-08-28 起)**：正式启动纯净前瞻影子账本，并行跟踪冻结清单中预注册的 `B0_BASELINE`、`SIMPLER_PURE_FRESHNESS` 与 `SIMPLER_PURE_CLOSE_POS`。
