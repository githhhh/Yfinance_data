# 突破池量化改进：执行规格书

> 本文档为 Codex 可直接执行的改动规格。按优先级排列。

---

## P0：Stage2 字典扩字段 — EPS + 52W High 注入 Pool

### 目标
在 `stage2_whitelist.csv` 源头新增 `eps_yoy_growth` 和 `price_52_week_high`，下游策略引擎 Left Join 注入 pool，赋能 Dashboard 多维筛选。

### 改动 1：`stage2_screener.py` — 扩展 `select()` 字段

**文件**：[stage2_screener.py](file:///Users/dev/Documents/Yfinance_data/stage2_screener.py)

Pass 1（L59）和 Pass 2（L80）的 `.select()` 均需扩展：

```python
# Before
.select('name', 'close', 'SMA10|1W', 'SMA40|1W', 'sector', 'industry')

# After
.select('name', 'close', 'SMA10|1W', 'SMA40|1W', 'sector', 'industry',
        'earnings_per_share_diluted_yoy_growth_fq', 'price_52_week_high')
```

`rename` 部分（L107）扩展：

```python
# Before
df_all = df_all.rename(columns={'name': 'code'})

# After
df_all = df_all.rename(columns={
    'name': 'code',
    'earnings_per_share_diluted_yoy_growth_fq': 'eps_yoy_growth',
})
```

> **零网络成本**：TradingView Query 是服务端聚合，加 select 字段不增加 RTT。

### 改动 2：下游策略引擎 — Pool Left Join（在 `quant_trade` 仓库）

在生成 `breakout_follow_pool.csv` 时，基于 `code` 做 merge：

```python
whitelist = pd.read_csv('us/stage2/stage2_whitelist.csv')[['code', 'eps_yoy_growth', 'price_52_week_high']]
pool = pool.merge(whitelist, on='code', how='left')
pool['dist_to_52w_high_pct'] = (pool['ibd_entry_price'] - pool['price_52_week_high']) / pool['price_52_week_high'] * 100
pool['is_52w_new_high'] = pool['ibd_entry_price'] >= pool['price_52_week_high']
```

### 新增 Pool 字段清单

| 字段 | 类型 | 语义 |
|:--|:--|:--|
| `eps_yoy_growth` | float | 最新季 EPS 稀释同比增速 %（来自 TradingView） |
| `price_52_week_high` | float | 52 周最高价（来自 TradingView） |
| `dist_to_52w_high_pct` | float | 入场价距 52W 高点百分比（派生，负值=在高点下方） |
| `is_52w_new_high` | bool | 入场时是否突破 52 周新高（派生） |

### 不变项
- `52wk_new_high_results.csv`：保持现有逻辑不变，继续作为独立 screener 输出 + ticker 贡献源
- `eps_screener.py`：保持现有逻辑不变
- `stage2_screener_filter.csv`：仍为仅含 `code` 列的 Fallback 快照，不扩展

---

## P1：5-Pattern 分类替换四象限

### 目标
用基于白皮书物理边界的 5 态分类**完全替换**现有四象限（Q1/Q2/Q3/Q4）体系。不新增 pool 存储字段，Dashboard 层实时派生。

### 分类规则（基于现有 pool 字段 `pos` + `range_ratio`）

```python
def classify_breakout_pattern(row):
    """基于白皮书 5 大机构标准模型的派生分类。"""
    pos = row['ibd_entry_close_position']
    rr = row['ibd_entry_breakout_range_ratio']
    if pos < 0.5:
        return 'BULL_TRAP'        # 冲高回落，收在 K 线下半区
    if rr > 1.0:
        return 'GAP_UP'           # 跳空高开：全天最低价 > 触发价
    if rr >= 0.4 and pos >= 0.7:
        return 'SOLID_BREAKOUT'   # 坚决长阳突破
    if rr >= 0.15:
        return 'MODERATE_BREAKOUT'  # 稳健有效穿透
    return 'MARGINAL_BREAKOUT'    # 刚蹭过中枢边缘
```

### Dashboard 筛选器 FilterSpec 映射

| 选项 | 过滤条件 |
|:--|:--|
| `GAP_UP` | `range_ratio > 1.0 AND pos >= 0.5` |
| `SOLID_BREAKOUT` | `0.4 <= range_ratio <= 1.0 AND pos >= 0.7` |
| `MODERATE_BREAKOUT` | `0.15 <= range_ratio < 0.4 AND pos >= 0.5` |
| `MARGINAL_BREAKOUT` | `0.0 <= range_ratio < 0.15 AND pos >= 0.5` |
| `BULL_TRAP` | `pos < 0.5` |

### 需改动文件清单

#### 1. `dashboard/data_utils.py`
- **函数** `_build_breakout_quadrant_data()`（L419-475）→ 重命名为 `_build_breakout_pattern_data()`
- `canonical_quadrants` → 替换为 `canonical_patterns = ["GAP_UP", "SOLID_BREAKOUT", "MODERATE_BREAKOUT", "MARGINAL_BREAKOUT", "BULL_TRAP"]`
- `assign_quadrant()` → 替换为上述 `classify_breakout_pattern()` 逻辑
- 返回 DataFrame 的 `"quadrant"` 列名 → `"pattern"`
- L147 的 chart key `"breakout_quadrant"` → `"breakout_pattern"`

#### 2. `dashboard/app.py`
- **筛选器 UI**（L196-246）：`quadrant_options` → 替换为 5 Pattern 选项 + 对应的 FilterSpec 组合
- **图表渲染**（L392-420）：`charts["breakout_quadrant"]` → `charts["breakout_pattern"]`，列名 `"quadrant"` → `"pattern"`
- 所有 UI 文案 `"Breakout Quadrant"` → `"Breakout Pattern"`

#### 3. `dashboard/tests/test_charts.py`
- `test_breakout_quadrant_profile_structure()`（L151-156）→ 重命名，`assert len(profile) == 4` → `== 5`，列名对齐

#### 4. `dashboard/tests/test_filters.py`
- `test_apply_filters_supports_gt_and_lt_for_quadrants()`（L305-318）→ 更新为 5-Pattern 对应的 FilterSpec 测试

#### 5. `dashboard/self_check.py`
- L41 `"chart: Sector Concentration aggregation"` 附近的 quadrant 引用 → 对齐新命名

---

## P2（Backlog）：`breakout_strength` 字段 — 暂缓

### 语义
$$\text{breakout\_strength} = \frac{\ln(\text{Close} / \text{trigger})}{\text{ATR}_{log}}$$

含义：收盘价穿透 trigger 的对数距离，以过去 20 日常态 K 线高度为标尺的倍数。`= 2.0` 表示穿透距离相当于两根标准日 K 线。

### 暂缓原因
- 现有 `ibd_entry_close_vs_trigger_pct` + `ibd_entry_volume_ratio` 组合已覆盖"强突破"筛选
- 实现需在策略引擎中额外计算 20 日对数 ATR，引入窗口期状态依赖
- 待 Dashboard 用户反馈"无法区分强弱突破"时再评估引入

### 现有字段不变
- `ibd_entry_close_position`（pos）：K 线收盘百分位，保留
- `ibd_entry_breakout_range_ratio`（range_ratio）：日内形态穿透比，保留
- `ibd_entry_close_vs_trigger_pct`：简单百分比穿透距离，保留
