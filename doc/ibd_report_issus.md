# IBD 候选预筛报告一致性问题

- [Claude Opus 4.6 复盘报告](./ibd_prescreen_report.md)

> **现象**：同一份候选池数据 + 同一份 Skill 规范，三种模型给出了三种不同的推荐结果。
>
> **日期**：2026-08-04
> **数据快照**：`us/breakout_follow_pool.csv`（2026-07-31）

---

## 问题定位

差异的根因 **不是模型主观判断不同**，而是数据解析基础设施存在系统性脆弱点，导致不同模型拿到的"事实数据"就不一样。按影响程度从大到小排列：

### 根因 #1：CSV 嵌套 JSON 导致字段错位（致命）

`breakout_follow_pool.csv` 中 `ibd_candidate_extra` 列包含嵌套 JSON：

```
"{""pivot_candidates"": [{""box_type"": ""S_BOX"", ""price"": 84.14, ""resistance_date"": ""2026-06-29""}]}"
```

JSON 内部含有大量逗号。简单按逗号切分 CSV（`awk -F','`、手工逐字段读取等）会导致后续所有字段全部错位——`base_depth_pct`、`eps_yoy_growth`、`pullback_v_is_dry`、`sector` 等关键 Checklist 判定字段的值张冠李戴。

**直接后果**：

- `ceiling` 类行没有嵌套 JSON（`ibd_candidate_extra` 为空），字段位置正确
- `pivot``ma10_touch_confirm``ceiling_pullback``three_weeks_tight` 类行含有不等长 JSON，各行偏移量不同
- 同一只股票在不同模型眼中的 EPS 可能是 +133% 或 -28%，直接翻转 Major #8 PASS/FAIL
- `pullback_v_is_dry` 可能读到日期字符串、`sector` 可能读到数字，所有下游判定均不可靠

**影响范围**：候选池中约 70% 以上的行含有嵌套 JSON，仅 `ceiling` 类（约 20%）不受影响。

### 根因 #2：Python 环境故障迫使即兴解析（放大器）

`quant_env` conda 环境不可用——`python3` 启动时报 `Failed to import encodings module`（`PYTHONHOME`/`PYTHONPATH` 环境变量与实际安装路径冲突）。

这意味着所有模型均无法运行 `pandas.read_csv()` 来正确解析含引号嵌套的 CSV。每个模型被迫采用各自的 fallback 策略：

| 策略 | 嵌套 JSON 处理能力 | 典型模型行为 |
|------|---------------------|-------------|
| `cat` + 手工逐行阅读 | ⚠️ 依赖模型识别引号边界 | 可能正确但不稳定 |
| `awk -F','` 切分 | ❌ 完全错位 | 字段全部偏移 |
| `grep` + 正则 | ⚠️ 部分字段可提取 | 依赖正则质量 |
| `head` / `view_file` 手工解读 | ⚠️ 逐行可行但易遗漏 | 人工精力有限 |

不同 fallback 策略对嵌套 JSON 的处理能力天差地别，**同一份数据产生了三份不同的"事实"**。

### 根因 #3：Checklist 排序存在合理弹性空间（次要）

即使数据解析完全一致，从 14~20 只通过 Critical 的标的中精选 3 只时，Skill 规范在 Major/Minor 层面允许权衡，例如：

- EPS 25% 边界值（如 THG 25.04%）算 PASS 还是边界 FAIL？
- Finance 拥挤限推 1 只时，多只 Finance 候选如何抉择？
- 缩量缺失 vs EPS 不达标，哪个对排序影响更大？

但这个弹性远小于根因 #1 / #2 造成的数据差异。**在数据一致的前提下**，这层弹性是可接受的，也是 Skill 设计中"经验辅助"的预期行为。

---

## 影响评估

| 影响维度 | 严重程度 | 说明 |
|----------|----------|------|
| Critical 淘汰准确性 | 🔴 高 | `close_position` 字段错位可能导致本应淘汰的标的通过、或本应通过的被误淘汰 |
| Major 判定准确性 | 🔴 高 | `eps_yoy_growth`、`pullback_v_is_dry`、`base_depth_pct` 错位直接影响 #4/#6/#8 |
| 板块拥挤判定 | 🟡 中 | `sector` 列错位可能导致拥挤风控计算偏差 |
| Geometry 分类 | 🟢 低 | `close_position` 和 `range_ratio` 位于 JSON 列之前，`ceiling` 行不受影响；但 JSON 行仍可能受影响 |
| 最终排序 | 🟡 中 | 即使数据正确，仍存在合理弹性（根因 #3） |

---

## 解决方案

### 方案 A：修复 Python 环境（短期止血）

修复 `quant_env` 的 `PYTHONHOME` / `PYTHONPATH` 冲突，使 Skill 执行时可用 `pandas.read_csv()` 精确解析 CSV。

- **优点**：改动最小
- **缺点**：仍依赖模型"选择用 Python"；不同模型可能仍走不同路径

### 方案 B：Skill 内嵌预处理脚本（长期正解）

在 `ibd-candidate-prescreen` Skill 中增加一个 `scripts/parse_pool.py` 预处理脚本。Skill 执行流程变为：

```
Phase 0: 运行 parse_pool.py → 输出标准化 JSON
Phase 1-4: AI 基于标准化 JSON 执行规则判定
```

脚本职责：
1. 用 `pandas.read_csv()` 或 Python `csv` 模块正确解析嵌套 JSON
2. 对每只 ACTIONABLE 标的输出扁平化 JSON，包含所有 Checklist 所需字段
3. 预计算 Breakout Geometry 分类（`trigger_pos`、几何类型）
4. 标注字段适用性（哪些字段对当前 signal 不适用）

**效果**：
- 消除根因 #1 + #2，不同模型拿到完全相同的结构化数据
- AI 只负责规则判定和排序（其擅长的部分）
- 残留弹性仅为根因 #3（合理且可接受）

### 方案 C：CSV 导出时消除嵌套 JSON（从源头治理）

修改上游 `breakout_follow_pool.csv` 的导出逻辑，将 `ibd_candidate_extra` 中的嵌套 JSON 拆分为独立列或移至单独文件。

- **优点**：从源头消除问题
- **缺点**：需改动上游 pipeline，影响面较大

### 推荐路径

**短期**：方案 A（修复环境）+ 方案 B（内嵌脚本），双管齐下。
**长期**：评估方案 C，在下一次 pipeline 重构时纳入。

---

## 验证方法

修复后，可用以下方式验证一致性：

1. 同一份 CSV 分别用 3 个模型执行 Skill
2. 对比三份报告的 Critical 淘汰名单是否一致
3. 对比通过 Critical 的标的的 Geometry 分类是否一致
4. 允许最终排序存在 1~2 只差异（根因 #3 的合理弹性）
