# Breakout Quality 三方标准对比审计与建议

## 概要结论

> **我建议采用 Codex 提出的二维矩阵方案**，同时对 SKILL.md 的 Geometry 描述做一次对齐修正。Codex 方案在数学上严谨地修复了现有实现的两个真实 Bug，且 SKILL.md 原有的几何描述与 Codex 方案本质上是兼容的——SKILL.md 只需要把"Gap Breakout"的判定条件从 `range_ratio > 1.0` 更新为 `trigger_pos <= 0` 即可。

---

## 一、三方标准全景比对

### 1.1 方案总览

| 维度 | 现有实现 (Current) | SKILL.md (Geometry) | Codex 提案 (Matrix) |
|:---|:---|:---|:---|
| **定义来源** | [`data_utils.py:197-217`](file:///Users/dev/Documents/Yfinance_data/dashboard/data_utils.py#L197-L217) | [`SKILL.md:68-78`](file:///Users/dev/Documents/Yfinance_data/.agents/skills/ibd-candidate-prescreen/SKILL.md#L68-L78) | [`breakout-price-quality-final-review.md:61-106`](file:///Users/dev/Documents/Yfinance_data/breakout-price-quality-final-review.md#L61-L106) |
| **判定结构** | 串联 if-elif 瀑布 | 定性分类（非代码） | 二维评分矩阵 |
| **使用字段** | `pos`, `range_ratio`, `close_vs_trigger_pct` | `pos`, `range_ratio`, 派生 `trigger_pos` | `pos`, `range_ratio`, 派生 `trigger_pos` |
| **最高档条件** | `range_ratio > 1.0`（不检查 pos） | `range_ratio > 1.0` (Gap Breakout) | `pos >= 0.80 AND trigger_pos <= 0` |
| **pos < 0.65 处理** | 仅当 range_ratio ≤ 1.0 时才判 Weak | 定义为"冲高回落/上影线抛压" | **一票否决 → Weak Close** |
| **五档名称** | Powerful / Strong Close / Constructive (Tight) / Constructive / Weak | Gap / Strong Finish / Squat（3类） | Powerful / Strong / Constructive / Marginal / Weak |

### 1.2 分档映射对照表

| Codex 矩阵档位 | 二维组合 | 对应现有实现 | 对应 SKILL.md 分类 |
|:---|:---|:---|:---|
| **Powerful Breakout** | High Close + Full-range | ⚠️ 被 `range_ratio > 1.0` 部分覆盖（不检查 pos） | Gap Breakout（部分重叠） |
| **Strong Breakout** | High Close + Clear，或 Constructive Close + Full-range | ⚠️ 部分映射到 Strong Close；Full-range 中 pos < 0.80 的部分被错判为最高档 | Strong Finish（仅覆盖高半区） |
| **Constructive Breakout** | High Close + Near，或 Constructive Close + Clear | 现有 "Constructive Close (Tight)" 仅覆盖 High+Near 部分 | 无直接对应 |
| **Marginal Breakout** | Constructive Close + Near | 现有 "Constructive Close" 覆盖 | 无直接对应 |
| **Weak Close** | pos < 0.65 | ⚠️ 可被 range_ratio > 1.0 覆盖为 Powerful | Squat / Upper Shadow |

---

## 二、现有实现的两个确认 Bug

### Bug 1: 最高档 `range_ratio > 1.0` 不检查 pos — 冲高回落被错判

```python
# 现有代码 — 第一个判断分支完全不看 pos
if rr > 1.0:
    return "Powerful Breakout"   # pos=0.30 也会命中！
```

**反例（来自 Codex 文档 §2.4）**：

```
pos = 0.30, range_ratio = 1.20, trigger_pos = -0.90
```

这根 K 线虽然全天在 Trigger 上方（Gap-up），但收盘仅在日内区间 30%，上影线高达 70%。这是典型的 **Gap-up 后冲高回落**，在 IBD 框架中绝对不应被判为最高质量。

**SKILL.md 的 Geometry 部分也存在同样的描述疏漏**：SKILL.md 定义 Gap Breakout 的条件为 `range_ratio > 1.0`，未叠加 pos 约束。这意味着 SKILL.md 在做 Prescreen 时，也可能把一个 Gap-up 后冲高回落的 K 线当作"跳空缺口突破"而给予过高评价。

> **判定：Codex 的 `pos >= 0.80 AND trigger_pos <= 0` 修复是正确的。**

### Bug 2: Full-range 条件用固定阈值 `> 1.0` 不够精确

```
pos = 0.85, range_ratio = 0.90
→ trigger_pos = 0.85 - 0.90 = -0.05  （Trigger 已低于 Low）
```

数学上 Trigger 已经在 Low 之下（$\frac{Trigger - Low}{High - Low} = -0.05 < 0$），整根 K 线都站在 Trigger 之上。但因为 `range_ratio = 0.90 < 1.0`，现有实现会跳过最高档，落入 Strong Close（pos >= 0.80 且 range_ratio >= 0.50）。

准确的"整根 K 线站在 Trigger 上方"的数学等价条件是：

$$trigger\_pos \le 0 \iff range\_ratio \ge pos$$

**不是**固定的 `range_ratio > 1.0`。SKILL.md 在 §71 的公式推导中其实已经正确写出了 $trigger\_pos = pos - range\_ratio$，但在 §75 的分类中却用了 `> 1.0` 这个不精确的条件。

> **判定：Codex 使用 `trigger_pos <= 0`（等价于 `range_ratio >= pos`）是数学上更精确的边界。**

---

## 三、SKILL.md Geometry 的兼容性分析

SKILL.md 定义的 Breakout Geometry 是**定性分类**，用于 Prescreen 中的文字描述，不是仪表盘用的五档评分。两者有不同的职责：

| 项目 | SKILL.md Geometry | Dashboard Breakout Quality |
|:---|:---|:---|
| **职责** | Prescreen 报告中描述 K 线形态 | 仪表盘表格中的结构化分级 |
| **输出** | 文字描述（Gap Breakout / Strong Finish / Squat） | 五档枚举 + 颜色梯度 |
| **消费者** | AI Agent 做 Checklist 判定 | 人类快速扫表 |

**关键发现**：SKILL.md 的 3 类分类和 Codex 的 5 档矩阵在本质上是兼容的：

```
SKILL.md "Gap Breakout" (range_ratio > 1.0)
  ≈ Codex "Full-range" 维度中的一个子集

SKILL.md "Strong Finish" (pos >= 0.80 且 range_ratio >= 0.50)
  = Codex 矩阵中 "High Close + Clear" 格位 = Strong Breakout

SKILL.md "Squat / Upper Shadow" (pos < 0.65)
  = Codex 的 Weak Close 一票否决
```

唯一需要对齐的是：SKILL.md 应将 Gap Breakout 条件从 `range_ratio > 1.0` 更新为 `trigger_pos <= 0`（等价于 `range_ratio >= pos`），并补充 pos >= 0.80 的前置条件。**这不改变 SKILL.md 的 Checklist 判定逻辑**，只是让描述更精确。

---

## 四、逐项审计 Codex 方案

| 审计项 | 结论 | 理由 |
|:---|:---|:---|
| **数学正确性** | ✅ 正确 | `trigger_pos = pos - range_ratio` 推导与 K 线几何吻合 |
| **Full-range 边界** | ✅ 修复 | `trigger_pos <= 0` 比 `> 1.0` 更精确，消除漏判 |
| **pos < 0.65 一票否决** | ✅ 合理 | 上影线 > 35% 是 IBD 框架中公认的弱势收盘信号，不应被 Gap 覆盖 |
| **二维独立性** | ✅ 改善 | 消除现有实现中 pos 与 range_ratio 维度纠缠的问题 |
| **单调性** | ✅ 保持 | 任一维度提升不会导致综合档位下降 |
| **"Tight" 命名** | ✅ 移除正确 | `range_ratio < 0.50` 只能说明近 Trigger，无法证明 K 线或 Base Tight |
| **成交量不并入** | ✅ 正确 | 成交量由独立字段（Volume Ratio / W Vol）评估，口径未对齐前不应混入 |
| **阈值选择** | ⚠️ 经验假设 | `0.80 / 0.65 / 0.50` 均为启发式阈值，需未来回测验证 |

---

## 五、推荐方案

### 5.1 Dashboard `_compute_breakout_quality` → 采用 Codex 二维矩阵

```python
def classify_breakout_quality(pos: float, range_ratio: float) -> str:
    if pos < 0.65:
        return "Weak Close"

    close_score = 2 if pos >= 0.80 else 1
    trigger_pos = pos - range_ratio

    if trigger_pos <= 0:
        clearance_score = 2
    elif range_ratio >= 0.50:
        clearance_score = 1
    else:
        clearance_score = 0

    return {4: "Powerful Breakout", 3: "Strong Breakout",
            2: "Constructive Breakout", 1: "Marginal Breakout"}[close_score + clearance_score]
```

### 5.2 SKILL.md Geometry 描述 → 同步修正

SKILL.md §74-78 的 Geometry 分类建议更新为：

```diff
 ### 核心分类与防御规则
-* **跳空缺口突破 (Gap Breakout)**：`range_ratio > 1.0`（$trigger\_pos < 0$）
-* **光头强突破 (Strong Finish)**：`pos >= 0.80` 且 `range_ratio >= 0.50`
+* **Full-range 突破 (Gap/Full-range Breakout)**：`trigger_pos <= 0`（即 `range_ratio >= pos`），且 `pos >= 0.80`
+* **强劲收盘 (Strong Finish)**：`pos >= 0.80` 且 `range_ratio >= 0.50`（但 `trigger_pos > 0`）
 * **冲高回落/上影线抛压 (Squat / Upper Shadow)**：`pos < 0.65`（上影线 $> 35\%$）
 * **防御规则 (Defensive Rule)**：若 $range\_ratio \le 0$ ($Close \le Trigger$)，触发防守断路器，直接判定破位失败。
+* **一票否决**：`pos < 0.65` 时，无论 `range_ratio` 多大，均判定为 Squat/弱势收盘，不因 Gap 升级。
```

### 5.3 不建议修改的部分

- SKILL.md 的 10 项 Checklist（#1-#10）保持不变——它们评估的是整体突破质量（含成交量、基本面等），与 Dashboard 纯价格五档不冲突
- 上游 Trigger 有效性判断不改动
- 成交量独立展示，不合并进五档

---

## 六、关于"阈值是否准确"的边界说明

> [!IMPORTANT]
> Codex 方案自己也坦承（§2.10）：0.80 / 0.65 / 0.50 这些分界线是**逻辑自洽的启发式阈值**，不是回测优化后的最优参数。如果未来要证明这些阈值真的最有效，需要按二维矩阵的 9 个格位统计样本的突破成功率与后续最大有利/不利波动。

但在没有回测数据之前，Codex 方案相对现有实现的改进是**确定性的**：
1. 修复了 Bug（pos < 0.65 被 range_ratio > 1.0 覆盖）
2. 消除了数学上不精确的 Full-range 阈值
3. 建立了可解释、可验证的二维评分框架

这是一个从"有明确缺陷"到"逻辑正确的默认启发式"的升级，不需要回测数据就能确认其改进方向。
