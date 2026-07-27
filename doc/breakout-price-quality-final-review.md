# Breakout Price Quality 最终判定标准与视觉梯度 Review

## 1. 目标

重新审计 Dashboard 表格中 `Breakout Quality` 的五档判定标准，并在判定语义正确后优化视觉层级：

1. Powerful Breakout
2. Strong Breakout
3. Constructive Breakout
4. Marginal Breakout
5. Weak Close

本次优先解决判定标准是否合理，再处理视觉呈现。上游突破有效性判断保持不变，但允许修改本列内部的五档分级条件与第三档名称。

---

## 最终档位逻辑（本版本唯一实现准则）

### 前置条件

- 上游已经确认 `close >= trigger`；
- `pos`、`range_ratio` 均为有效有限数值；
- `high == low`、`None`、`NaN` 沿用现有缺失值处理，不进入五档；
- 本列只评估价格行为，不包含成交量确认。

### 派生字段

```python
trigger_pos = pos - range_ratio
```

含义：

```text
trigger_pos <= 0
    Trigger 位于当天 Low 或更低位置
    整根 K 线位于 Trigger 之上或从 Trigger 起步

trigger_pos > 0
    Trigger 位于当天 K 线内部
```

因此 Full-range 的准确条件是：

```python
trigger_pos <= 0
```

等价于：

```python
range_ratio >= pos
```

不是固定的：

```python
range_ratio > 1.0
```

### 最终二维矩阵

| Close Quality | Full-range：`trigger_pos <= 0` | Clear：`trigger_pos > 0 and range_ratio >= 0.50` | Near：`0 <= range_ratio < 0.50` |
|---|---|---|---|
| High：`pos >= 0.80` | **Powerful Breakout** | **Strong Breakout** | **Constructive Breakout** |
| Constructive：`0.65 <= pos < 0.80` | **Strong Breakout** | **Constructive Breakout** | **Marginal Breakout** |
| Weak：`pos < 0.65` | **Weak Close** | **Weak Close** | **Weak Close** |

### Codex 应实现的唯一分类函数

```python
QUALITY_BY_MATRIX_SCORE = {
    4: "Powerful Breakout",
    3: "Strong Breakout",
    2: "Constructive Breakout",
    1: "Marginal Breakout",
}

QUALITY_SORT_RANK = {
    "Powerful Breakout": 5,
    "Strong Breakout": 4,
    "Constructive Breakout": 3,
    "Marginal Breakout": 2,
    "Weak Close": 1,
}


def classify_breakout_quality(pos: float, range_ratio: float) -> str:
    # 上游已保证 close >= trigger。
    # 缺失值和 high == low 继续使用项目现有策略，不在这里归入 Weak。

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

    return QUALITY_BY_MATRIX_SCORE[close_score + clearance_score]
```

任何后续章节、Tooltip、排序映射、视觉 Meta 和测试都必须以本节为唯一标准。

---

## 2. 已确认的业务前提

### 2.1 上游已经完成突破有效性判断

进入 Dashboard 和 `Breakout Quality` 分级的数据，已经经过上游 Trigger 判断。

因此本列不负责处理：

- Close 低于 Trigger
- Failed Breakout
- Below Trigger
- 无效突破状态

不要在 `field_config.py` 的 `QUALITY_META` 或 Cell Style 中新增上述状态，也不要重复实现上游判断。

### 2.2 Weak Close 仍然是有效突破

`Weak Close` 表示有效突破中收盘位置相对较弱，不代表突破无效或负面信号。

因此：

- 保留全绿色质量体系；
- 不要把 Weak Close 改成红色或橙色；
- 通过降低绿色背景强度和字重表达“相对较弱”。

### 2.3 两个指标分别表达什么

假设字段定义为：

```python
pos = (close - low) / (high - low)
range_ratio = (close - trigger) / (high - low)
trigger_pos = pos - range_ratio
```

那么：

- `pos` 衡量 Close 在当天 K 线中的位置，是收盘质量与上影线压力的指标；
- `range_ratio` 衡量 Close 超过 Trigger 的幅度，占当天振幅的比例；
- `trigger_pos` 表示 Trigger 在当天 K 线中的相对位置。

两者不是同一个维度：

| 指标 | 回答的问题 | 不能单独证明 |
|---|---|---|
| `pos` | 当天是否收在高位 | 突破 Trigger 的幅度是否充足 |
| `range_ratio` | Close 超过 Trigger 多远 | 是否冲高回落、是否收在日内低位 |

### 2.4 当前最高档存在两个逻辑漏洞

当前规则：

```python
if range_ratio > 1.0:
    quality = "Powerful Breakout"
```

第一个问题是它没有判断 `pos`，可能把明显冲高回落判为最高档。

第二个问题是：如果它想表达“Trigger 位于当天 Low 下方、整根 K 线站在 Trigger 上方”，那么 `range_ratio > 1.0` 并不是准确边界。

由：

```python
trigger_pos = pos - range_ratio
```

可知准确关系是：

```python
trigger_pos <= 0
<=> pos - range_ratio <= 0
<=> range_ratio >= pos
```

也就是说：

- `range_ratio > 1.0` 是 Trigger 位于 Low 下方的充分条件，但不是必要条件；
- 真正的 Full-range 条件是 `range_ratio >= pos`；
- 当 `range_ratio > pos` 时，Trigger 严格低于 Low；
- 当 `range_ratio == pos` 时，Trigger 正好位于 Low。

漏判示例：

```text
pos = 0.85
range_ratio = 0.90
trigger_pos = -0.05
```

虽然 `range_ratio < 1.0`，但 Trigger 已经位于 Low 下方，整根 K 线都在 Trigger 之上。原有 `>1.0` 条件会漏掉这种真正的 Full-range Breakout。

同时，即使满足 Full-range，也不能忽略收盘质量：

反例：

```text
pos = 0.30
range_ratio = 1.20
trigger_pos = -0.90
```

这根 K 线虽然整日位于 Trigger 上方，但 Close 只处于日内区间的 30%，上影线约为 70%。它更接近“Gap-up 后冲高回落”，不应因为 `range_ratio > 1.0` 被直接覆盖成最高档。

因此，`Powerful Breakout` 的准确价格行为条件应同时满足：

```python
pos >= 0.80 and trigger_pos <= 0
```

等价写法：

```python
pos >= 0.80 and range_ratio >= pos
```

最高档代表“高位收盘 + Trigger 位于当天 Low 或更低位置”，而不是使用固定的 `range_ratio > 1.0`。

### 2.5 为什么 High Close (Near Trigger) 不能直接排在 Constructive Close 前面

以下两个样本分别在一个维度更强：

```text
A: pos = 0.85, range_ratio = 0.10
   高位收盘，但仅略高于 Trigger

B: pos = 0.75, range_ratio = 0.80
   收盘尚可，但整根 K 线位于 Trigger 上方
```

A 的收盘位置更强，B 的突破幅度与结构更强。两者属于二维上的交叉组合，不存在天然的单向大小关系。

因此：

- `High Close (Near Trigger)` 是一个事实描述；
- `Constructive Close` 也是一个事实描述；
- 把前者设为第三档、后者设为第四档，实际上偷偷加入了“Close 位置绝对优先”的偏好；
- 如果没有明确策略依据或回测证据，这种固定优先级不能称为准确。

### 2.6 推荐原则：二维评分，弱收盘一票否决

先将两个维度各分成三级：

#### Close Quality

| Close 等级 | 条件 | 分值 |
|---|---|---:|
| High Close | `pos >= 0.80` | 2 |
| Constructive Close | `0.65 <= pos < 0.80` | 1 |
| Weak Close | `pos < 0.65` | 0 |

#### Trigger Clearance

| 突破幅度等级 | 条件 | 分值 |
|---|---|---:|
| Full-range At/Above Trigger | `trigger_pos <= 0`，即 `range_ratio >= pos` | 2 |
| Clear Above Trigger | `trigger_pos > 0 and range_ratio >= 0.50` | 1 |
| Near Trigger | `0 <= range_ratio < 0.50` | 0 |

整体等级采用两个维度相加，但保留一条风险约束：

```text
pos < 0.65 时始终为 Weak Close
```

原因是上影线超过约 35% 已经表达明显的收盘压力，不应仅靠较大的 Gap 或 Trigger 距离完全抵消。

### 2.7 推荐五档标准

| 档位 | 推荐名称 | 组合 | 含义 |
|---:|---|---|---|
| 5 | **Powerful Breakout** | High Close + Full-range | 两个维度同时最强 |
| 4 | **Strong Breakout** | High Close + Clear，或 Constructive Close + Full-range | 一个维度最强，另一个至少良好 |
| 3 | **Constructive Breakout** | High Close + Near Trigger，或 Constructive Close + Clear | 一个维度强、另一个偏弱，或两个维度都良好 |
| 2 | **Marginal Breakout** | Constructive Close + Near Trigger | 突破有效，但两个维度都没有形成明显优势 |
| 1 | **Weak Close** | `pos < 0.65` | 收盘压力明显，不因 Gap 或突破幅度升级 |

推荐实现：

```python
QUALITY_BY_MATRIX_SCORE = {
    4: "Powerful Breakout",
    3: "Strong Breakout",
    2: "Constructive Breakout",
    1: "Marginal Breakout",
}

QUALITY_SORT_RANK = {
    "Powerful Breakout": 5,
    "Strong Breakout": 4,
    "Constructive Breakout": 3,
    "Marginal Breakout": 2,
    "Weak Close": 1,
}


def classify_breakout_quality(pos: float, range_ratio: float) -> str:
    # 上游已保证 close >= trigger，因此此处不重复判断突破有效性。
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

    return QUALITY_BY_MATRIX_SCORE[close_score + clearance_score]
```

### 2.8 描述原因与主档位分开

`High Close (Near Trigger)` 与 `Constructive Close` 有关系，但它们不应该作为两个相邻主档位互相比较。

更合理的关系是：

```text
High Close + Near Trigger
    → Constructive Breakout

Constructive Close + Clear Above Trigger
    → Constructive Breakout
```

两个组合虽然形成原因不同，但综合得分相同。主单元格显示：

```text
Constructive Breakout
```

Tooltip 或详情显示具体原因：

```text
High Close · Near Trigger
```

或者：

```text
Constructive Close · Clear Above Trigger
```

这样既保留原始信息，又不会假装两个交叉维度存在绝对顺序。

### 2.9 Tight 名称仍应移除

原名称 `Constructive Close (Tight)` 仍不准确。`range_ratio < 0.50` 不能证明 K 线、Base 或多日结构 Tight，只能证明 Close 相对靠近 Trigger。

如果作为原因描述，使用：

```text
High Close · Near Trigger
```

不要继续使用 `Tight`。

### 2.10 “准确”的边界

上述二维矩阵可以确认的是：

- 逻辑自洽；
- 档位单调；
- 不会由单个极强指标掩盖明显弱收盘；
- High Close (Near Trigger) 与 Constructive Close 的交叉关系得到合理处理；
- 每个档位都能追溯到明确的二维组合。

但它仍然不能仅凭逻辑证明“预测效果最优”。以下设定仍属于需要数据验证的策略假设：

- `0.80` 和 `0.65` 是否是最有效的 Close 分界；
- `0.50` 是否是最有效的 Clear / Near Trigger 分界；
- Close Quality 与 Trigger Clearance 是否应该等权；
- `pos < 0.65` 是否应当一票否决；
- 五档是否能在真实样本中形成单调的后续表现。

因此本方案应定义为：

```text
逻辑正确、可解释的默认启发式
```

而不是：

```text
已经被历史数据证明的最优标准
```

如果要确认策略有效性，应按二维矩阵的九个组合统计样本量、突破失败率、后续最大有利波动和最大不利波动。只有当结果大体随档位单调改善时，才能把它升级为经验验证后的正式标准。

---

## 3. 判定矩阵

| Close 位置 | Full-range：`trigger_pos <= 0` | Clear：`trigger_pos > 0 and range_ratio >= 0.50` | Near Trigger：`0 <= range_ratio < 0.50` |
|---|---|---|---|
| High Close：`pos >= 0.80` | Powerful Breakout | Strong Breakout | Constructive Breakout |
| Constructive Close：`0.65 <= pos < 0.80` | Strong Breakout | Constructive Breakout | Marginal Breakout |
| `pos < 0.65` | Weak Close | Weak Close | Weak Close |

这张矩阵是本次 Review 的核心。它同时解决：

- Full-range 使用准确的 `trigger_pos <= 0`，不再错误使用固定 `range_ratio > 1.0`；
- High Close (Near Trigger) 与 Constructive Close 不再被强行排成固定先后；
- 两个维度各提升一级时，综合档位只提升一级；
- 档位具备对称、可解释的单调关系。

---

## 4. 字段语义与成交量边界

### 4.1 仅凭 pos 与 range_ratio 不能代表完整 Breakout Quality

当前五档只使用：

- Close 在日内区间的位置；
- Close 相对 Trigger 的突破幅度。

这两个字段能够评估价格行为，但不能评估突破背后的成交量确认。

IBD 对典型增长股突破强调强成交量，通常至少高于平均量约 40%；Schwab 的技术分析说明也指出，突破伴随高于平均的成交量通常更具显著性，低量突破可能缺乏参与热情：

- [IBD：What Is A Stock Breakout?](https://www.investors.com/how-to-invest/investors-corner/what-is-stock-breakout/)
- [Charles Schwab：Trading Volume as a Market Indicator](https://www.schwab.com/learn/story/trading-volume-as-market-indicator)

因此，当前五档更准确的语义是：

```text
Breakout Price Quality
```

而不是包含成交量、Base、市场环境和基本面的完整突破质量。

### 4.2 推荐字段命名

优先建议将表头改为：

```text
Breakout Price Quality
```

如果列宽需要更短，可以使用：

```text
Price Quality
```

Tooltip 必须说明：

```text
Price-action quality based on close position and trigger clearance.
Volume confirmation is evaluated separately.
```

这样用户不会把漂亮的绿色最高档误解为“所有突破条件均已确认”。

### 4.3 成交量暂不并入这五档

截图中的 `W Vol` 应继续独立展示，除非已经确认它满足以下条件：

- 确实对应本次 Breakout 的发生周期；
- 分母是稳定、明确的平均成交量；
- 与 IBD Entry 使用的日线或周线口径一致；
- 缺失值与半周数据不会制造虚假低量；
- 已有足够样本验证加入综合评分后的单调性。

在这些条件没有确认前，不要直接把 `W Vol >= 1.4` 写进五档规则。否则可能把“周成交量”“突破日成交量”和“当前周未完成成交量”混为一谈。

正确的职责划分是：

```text
Breakout Price Quality
    = Close Quality 档位 + Trigger Clearance 档位

Volume Confirmation
    = 独立字段、独立口径
```

未来如需形成 Overall Breakout Quality，应再明确组合规则，而不是在当前 Cell Style 中悄悄加入 Volume 条件。

---

## 5. 当前视觉实现 Review

当前实现已经具备明确的设计方向：

- 使用绿色背景强度表示突破质量；
- 使用左侧 Accent Border 强化等级；
- 使用字重形成第二层级；
- 标签文本使含义不依赖颜色，具备基本可访问性。

但从实际 Dashboard 截图观察，仍存在以下体验问题。

### 5.1 Breakout Quality 列视觉面积过大

当前该列接近 500px。连续多个 Powerful Breakout 时会形成大面积高饱和绿色块，视觉上压过：

- Entry Status
- Vs Candidate
- Entry / Reason
- Latest

质量列应帮助扫描，不应成为整个表格的主背景。

### 5.2 Powerful 与 Strong 的背景差异过小

当前 Alpha：

```text
Powerful: 0.34 → 0.16
Strong:   0.30 → 0.12
```

两组数值过于接近，在真实深色表格中几乎表现为同一档绿色。虽然参数不同，但用户难以快速感知。

### 5.3 动态边框宽度没有形成有效识别

当前使用：

```text
5px → 4px → 3px → 2px → 1px
```

实际显示中，用户主要只能看到一条绿色竖线，很难依靠 1–2px 的差异判断等级。同时，不同宽度会造成单元格文字起始位置轻微不齐。

### 5.4 背景、边框和字重同时五档变化，视觉编码偏重

三种视觉变量同时变化并没有带来三倍识别力，反而增加了表格噪音。

建议：

- 背景 Alpha：承担主要等级表达；
- 左边框：只作为统一的位置锚点；
- 字重：只区分最强两档和普通档。

### 5.5 AG Grid 状态可能被背景覆盖

需要确认 Cell 的 `backgroundImage` 不会使以下状态难以辨认：

- Row hover
- Selected row
- Keyboard focused row
- Cell focus outline
- Pinned / unpinned column边界

---

## 6. 建议视觉方案

### 6.1 列宽

建议：

```text
preferred width: 260px
min width:       220px
max width:       300px
```

如果当前使用 `flex`，不要让 `Breakout Quality` 吞掉所有剩余宽度。优先使用受约束的 `minWidth / maxWidth`。

### 6.2 左边框

五个等级统一：

```python
borderLeftWidth = "3px"
borderLeftStyle = "solid"
```

等级差异通过 `borderColor` 强度表达，不再通过宽度表达。

统一宽度可以：

- 保持文字左侧对齐；
- 避免 1px 边框在高分屏上过弱；
- 让用户把边框理解为等级锚点，而不是另一套独立量表。

### 6.3 推荐视觉参数

| 等级 | 文字颜色 | 左边框颜色 | 背景渐变 Alpha | 字重 |
|---|---|---|---|---:|
| Powerful Breakout | `#86efac` | `#22c55e` | `0.28 → 0.12` | `700` |
| Strong Breakout | `#4ade80` | `rgba(34, 197, 94, 0.78)` | `0.17 → 0.06` | `600` |
| Constructive Breakout | `#22c55e` | `rgba(74, 222, 128, 0.58)` | `0.09 → 0.03` | `500` |
| Marginal Breakout | `#86efac` | `rgba(134, 239, 172, 0.38)` | `0.04 → 0.012` | `500` |
| Weak Close | `#bbf7d0` | `rgba(187, 247, 208, 0.22)` | `0.01 → 0.00` | `400` |

说明：

- 保留当前文字色方向，降低本次修改范围；
- Powerful 与 Strong 的起始 Alpha 差距从当前的 `0.04` 扩大到 `0.11`；
- 五档起始 Alpha 采用明显的非线性衰减：`0.28 → 0.17 → 0.09 → 0.04 → 0.01`；
- 边框宽度统一，但边框透明度同步按 `100% → 78% → 58% → 38% → 22%` 衰减；
- Weak 仍属于绿色体系，但基本不形成填充色块，仅保留文字和极弱边框提示。

### 6.4 推荐 `QUALITY_META`

```python
QUALITY_META = {
    "Powerful Breakout": {
        "color": "#86efac",
        "borderColor": "#22c55e",
        "borderWidth": "3px",
        "backgroundImage": (
            "linear-gradient(90deg, "
            "rgba(34, 197, 94, 0.28), "
            "rgba(34, 197, 94, 0.12))"
        ),
        "fontWeight": "700",
    },
    "Strong Breakout": {
        "color": "#4ade80",
        "borderColor": "rgba(34, 197, 94, 0.78)",
        "borderWidth": "3px",
        "backgroundImage": (
            "linear-gradient(90deg, "
            "rgba(34, 197, 94, 0.17), "
            "rgba(34, 197, 94, 0.06))"
        ),
        "fontWeight": "600",
    },
    "Constructive Breakout": {
        "color": "#22c55e",
        "borderColor": "rgba(74, 222, 128, 0.58)",
        "borderWidth": "3px",
        "backgroundImage": (
            "linear-gradient(90deg, "
            "rgba(34, 197, 94, 0.09), "
            "rgba(34, 197, 94, 0.03))"
        ),
        "fontWeight": "500",
    },
    "Marginal Breakout": {
        "color": "#86efac",
        "borderColor": "rgba(134, 239, 172, 0.38)",
        "borderWidth": "3px",
        "backgroundImage": (
            "linear-gradient(90deg, "
            "rgba(34, 197, 94, 0.04), "
            "rgba(34, 197, 94, 0.012))"
        ),
        "fontWeight": "500",
    },
    "Weak Close": {
        "color": "#bbf7d0",
        "borderColor": "rgba(187, 247, 208, 0.22)",
        "borderWidth": "3px",
        "backgroundImage": (
            "linear-gradient(90deg, "
            "rgba(34, 197, 94, 0.01), "
            "rgba(34, 197, 94, 0.00))"
        ),
        "fontWeight": "400",
    },
}
```

如果现有代码使用 `borderLeft` 而不是拆分属性，统一生成：

```python
"borderLeft": f"3px solid {meta['borderColor']}"
```

---

## 7. 不在本次范围内

Codex 不应顺手修改以下内容：

- 上游 Trigger 有效性判断；
- Entry Status 的颜色或判定；
- Dashboard 排序逻辑；
- 其他字段的 Cell Style；
- 将 Weak Close 改成风险色。

---

## 8. 实现要求

1. 先定位真正生成 Breakout Quality label 的函数，不要只修改 `QUALITY_META`。
2. 按第 2.7 节的二维评分规则修改五档判定。
3. 将原有五档 label 更新为 `Powerful / Strong / Constructive / Marginal / Weak`，并同步更新：
   - `QUALITY_META`；
   - Tooltip；
   - 排序映射；
   - 测试期望；
   - 可能存在的筛选选项。
4. 将表头改为 `Breakout Price Quality`，或至少在 Tooltip 中明确该等级只包含价格行为，不包含 Volume。
5. 上游 Trigger 有效性判断保持不变，不在 Dashboard 层重复判断。
6. 保持 `W Vol` 独立；未确认数据口径前，不将成交量写入五档判定。
7. 完成规则、命名和 Tooltip 后，再调整样式 Meta、列宽约束和必要的 Cell Style 组合。
8. 避免新增重复的硬编码映射；label、排序权重和样式应有单一来源或明确映射。
9. 若 Cell Style 返回新字典，确认不会意外覆盖：
   - 对齐；
   - padding；
   - row selected 状态；
   - focus outline。
10. 如果 AG Grid 的 inline `backgroundImage` 会遮盖 selected/hover，使用 CSS class 或 CSS variable 合并状态，不要直接取消选中反馈。

---

## 9. 测试与验收

### 9.1 判定边界测试

至少覆盖以下边界和反例：

| `pos` | `range_ratio` | 预期 |
|---:|---:|---|
| `0.80` | `0.81` | Powerful Breakout |
| `0.80` | `0.80` | Powerful Breakout |
| `0.80` | `0.7999` | Strong Breakout |
| `0.7999` | `0.80` | Strong Breakout |
| `0.30` | `1.20` | Weak Close |
| `0.80` | `0.50` | Strong Breakout |
| `0.80` | `0.4999` | Constructive Breakout |
| `0.65` | `0.66` | Strong Breakout |
| `0.65` | `0.65` | Strong Breakout |
| `0.65` | `0.6499` | Constructive Breakout |
| `0.65` | `0.50` | Constructive Breakout |
| `0.65` | `0.20` | Marginal Breakout |
| `0.6499` | `1.20` | Weak Close |

必须专门保留以下反例测试，防止以后再次出现 `range_ratio` 覆盖弱收盘：

```python
assert classify_breakout_quality(0.30, 1.20) == "Weak Close"
```

### 9.2 性质测试

除具体边界样本外，建议增加参数化或性质测试：

1. `pos < 0.65` 时，无论有效范围内的 `range_ratio` 多大，结果都必须是 Weak Close。
2. `pos >= 0.65` 时，提高 `pos` 档位或 `range_ratio` 档位都不能导致综合等级下降。
3. High Close + Near Trigger 与 Constructive Close + Clear 必须得到相同的 Constructive Breakout。
4. Constructive Close + Full-range 必须是 Strong，而不是 Powerful 或普通 Constructive。
5. 当 `range_ratio` 从略小于 `pos` 增长到等于或大于 `pos` 时，Clear 必须升级为 Full-range。
6. `range_ratio > 1.0` 不是 Powerful 的必要条件；例如 `pos=0.80, range_ratio=0.81` 必须是 Powerful。
7. `None`、`NaN`、`high == low` 的处理沿用现有缺失值策略，不要无意归入 Weak Close。

### 9.3 语义验收

- 表头或 Tooltip 明确这是 Breakout Price Quality；
- Tooltip 明确两个输入维度是 Close Position 与 Trigger Clearance；
- Tooltip 不宣称 Volume Confirmed；
- `W Vol` 与 Price Quality 不重复表达；
- 用户可以从 Tooltip 看到同一主档位的具体形成原因。

### 9.4 视觉验收

- Powerful 能第一眼识别，但不会形成大面积高饱和绿色墙；
- Powerful 与 Strong 无需逐字阅读即可区分；
- Constructive、Marginal、Weak 呈连续衰减；
- 五档文字左边缘对齐；
- Weak 仍能看出属于绿色质量体系；
- Entry Status、Code、Vs Candidate 的视觉优先级没有被质量列压制；
- 260–300px 列宽下完整显示最长标签；
- 横向滚动、Pin Column 后无布局抖动。

### 9.5 交互验收

逐项检查：

- 鼠标 hover 行仍清晰；
- 点击选中行仍清晰；
- 使用 ↑ / ↓ 键切换行时焦点仍清晰；
- Selected Row Detail 与当前行一致；
- 表头筛选图标与列标题没有因列宽变化而挤压；
- 深色主题下文字可读；
- 如果项目支持浅色主题，浅色主题不得出现低对比度文字。

---

## 10. Codex Goal

> 先修正 Breakout Price Quality 的判断标准，再处理视觉。保持上游 Trigger 有效性判断不变，将 `pos` 离散为 Close Quality，并以 `trigger_pos = pos - range_ratio` 计算 Trigger Clearance。Full-range 的准确条件必须使用 `trigger_pos <= 0`（等价于 `range_ratio >= pos`），不得继续用固定的 `range_ratio > 1.0` 代替。通过二维矩阵形成 Powerful、Strong、Constructive、Marginal、Weak 五档，并对 `pos < 0.65` 保留 Weak Close 一票否决。High Close (Near Trigger) 与 Constructive Close 不得强行设为固定先后。明确该列只评估价格行为，成交量继续独立展示且不在口径未确认时并入评分。完成规则、命名、原因、排序映射、Tooltip 和测试后，最后才应用视觉梯度。

## 11. 完成标准

只有同时满足以下条件才算完成：

- `Powerful Breakout` 必须同时满足 `pos >= 0.80` 与 `trigger_pos <= 0`；
- Full-range 必须使用 `range_ratio >= pos`，不能继续使用固定 `range_ratio > 1.0`；
- `pos < 0.65 and range_ratio > 1.0` 必须保持 Weak Close；
- High Close + Near Trigger 与 Constructive Close + Clear 必须同为 Constructive Breakout；
- 五档名称统一为 Powerful、Strong、Constructive、Marginal、Weak，不再使用未经字段验证的 `Tight`；
- 表头或 Tooltip 明确该列为 Price Quality，不宣称已经包含 Volume Confirmation；
- 未确认 `W Vol` 数据口径前，成交量不得加入五档条件；
- 五档判断、Tooltip、排序映射和测试保持一致；
- 五档左边框统一为 3px；
- 使用推荐 Alpha 梯度或提供有截图依据的微调值；
- Breakout Quality 列最大宽度不超过 300px；
- Powerful 与 Strong 肉眼可区分；
- Weak 保持绿色体系但明显弱化；
- hover、selected、keyboard focus 可见；
- 自动测试通过；
- 无其他 Dashboard 字段或规则的非必要修改。
