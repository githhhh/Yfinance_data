# 突破强度字段量化语义评估与四象限分类体系改进探讨

本文档基于第一性原理，深入探讨定量交易体系中**突破中枢强度字段的语义完备性与改进方案**，以及 **Dashboard 四象限切分与白皮书标准模型的对照评估**。

---

## 议题一：当前突破强度字段能满足量化语义吗？及其改进讨论

### 1. 现状评估：字段自身的物理含义与量化盲区

当前用于衡量“日线突破中枢强度”的核心字段为：
- **`ibd_entry_close_position`（收盘位置 `pos`）**：
  $$\text{pos} = \frac{\text{Close} - \text{Low}}{\text{High} - \text{Low}}$$
- **`ibd_entry_breakout_range_ratio`（穿透比例 `range_ratio`）**：
  $$\text{range\_ratio} = \frac{\text{Close} - \text{trigger\_price}}{\text{High} - \text{Low}}$$

#### 评估结论：满足“K 线内部形态比例”语义，但不满足“突破推进力度”语义
假设 `trigger_price` 中枢点位完全准确：
1. **收盘强度 `pos`**：语义清晰精确，准确反映多空日内博弈最终结果（收在 K 线哪个百分位）。
2. **穿透比例 `range_ratio` 的量化盲区**：
   - 分母归一化为**当日 K 线总振幅**，它回答的是“在今天这一根 K 线内部，突破中枢的部分占了百分之几”。
   - **力度失真**：当股票在一个极端窄幅波动日（如全天振幅 1%）刚好微幅收于中枢上方 0.7% 时，其 `range_ratio = 0.70`；而一根振幅 10% 的巨大阳线，穿透中枢 7% 时，计算出的 `range_ratio` 同样是 `0.70`。
   - 这反映了现有字段把 **高动能长阳突破** 与 **窄幅微弱过线** 在指标层等同化的局限。

---

### 2. 改进讨论：人眼对数视觉感知（Log-Scale Visual Perception）与正交解耦

在专业对数坐标图表（Log-Scale Chart）中，交易者人眼感知的“K 线高度”和“突破远近”不是绝对美元价差，而是**对数相对比例百分比 $\ln(P_2 / P_1)$**。低价股（\$20）与高价股（\$200）同等百分比波动在图表上展现的是完全相同的几何长短。

为了模拟“人眼观测”的准确直觉，同时解决“形态比例”与“绝对推进力度”混同的问题，推导完整升级方案如下：

#### 改进公式定义（对数空间视觉转换）

| 指标维度 | 改进计算公式 | 物理与视觉含义 |
| :--- | :--- | :--- |
| **视觉 K 线高度** | $\text{candle\_height}_{log} = \ln\!\left(\frac{\text{High}}{\text{Low}}\right)$ | 该根 K 线在对数 K 线图上呈现的实际视觉上下跨度。 |
| **视觉形状穿透比** | $\text{range\_ratio}_{log} = \frac{\ln(\text{Close} / \text{trigger})}{\ln(\text{High} / \text{Low})}$ | 突破区段占当日图表 K 线整根高度的视觉百分比。 |
| **对数常态波动标尺** | $\text{ATR}_{log} = \text{SMA}\!\left(\ln\!\left(\frac{\text{High}}{\text{Low}}\right), 20\right)$ | 该股票过去 20 个交易日中，一根“正常标准 K 线”的视觉高度。 |

#### 核心升级：评估是否新增“绝对视觉推进力度”字段 `breakout_strength`

将突破评价由单一形状拆解为**“形状形态” + “绝对强度”**两个独立维度：

$$\text{breakout\_strength} = \frac{\ln(\text{Close} / \text{trigger})}{\text{ATR}_{log}}$$

> **量化语义诠释**：
> - `breakout_strength = 2.0` 代表：收盘价穿透阻力线的实际视觉距离，相当于这只股票**平时两根标准日 K 线的总高度**。
> - 它完全剔除了分母受当日意外上下影线变动的噪声干扰，真正让“不同标价、不同常规波动率”的股票在突破力度上完全横向平权。

#### 字段落地边界（讨论确认）

本议题的目标是判断字段必要性，不是把所有中间计算量都直接加入 `breakout_follow_pool.csv`：

- **保留现有基础字段**：`ibd_entry_close_position` 与 `ibd_entry_breakout_range_ratio` 继续作为 K 线形态比例与 5 类模型判断的基础字段。
- **不默认新增中间诊断字段**：`range_ratio_log` 与 `ATR_log_20` 主要服务于内部计算和公式解释，除非后续存在明确审计或前台展示需求，否则不建议进入 pool。
- **只优先评估一个核心字段**：若要补足“绝对推进力度”语义，优先新增 `ibd_entry_breakout_strength`。它是可直接筛选的原子决策字段，语义为“收盘价穿透 trigger 的对数距离，相当于过去 20 日常态 K 线高度的倍数”。
- **旧字段不替换**：`ibd_entry_close_vs_trigger_pct` 仍保留为简单百分比诊断字段；`ibd_entry_breakout_strength` 若落地，是对其做波动率标准化补充，而不是替代现有字段。

#### 建议落地组合体系
- **形态判定 (Shape)**：继续优先使用现有 `pos` + `range_ratio` 基础字段，避免为展示分类重复存储派生标签；
- **力度判定 (Intensity)**：用 **`breakout_strength >= 1.5`** 确保实际过顶空间不仅是一层薄纸，而是真实的放量实质性跨离。

---

## 议题二：当前象限维度划分准确吗？及其改进讨论

### 1. 现状评估：白皮书 5 大物理模型 vs Dashboard 四象限

白皮书（`BREAKOUT_FOLLOW_POOL_SCHEMA.md`）对突破逻辑采用物理形态映射，划分为 **5 大机构标准模型**：

```text
range_ratio (穿透深度比)
  ^
1.0 ┼ - - - - - - - - - - - - - - - - - - 🚀 Gap Up (跳空高开突破: 全天最低价 > 触发价)
    │
0.4 ┼ - - - - - - - - - - - - - - - - - - 🌟 Solid Breakout (坚决长阳突破: 主力资金强力拉升)
    │
0.15┼ - - - - - - - - - - - - - - - - - - ✅ Moderate Breakout (稳健有效穿透)
    │
0.0 ┼ - - - - - - - - - - - - - - - - - - ⚠️ Marginal Breakout (刚蹭过中枢边缘)
    ├────────────────────────────────────
    │                                     ❌ Bull Trap (冲高回落被砸入下半区: pos < 0.5)
    └────────────────────────────────────> pos (收盘分界线 0.5)
```

然而，Dashboard 现行的四象限划分使用 `range_ratio = 1.5` 与 `pos = 0.70` 进行正交四等分：
```text
range_ratio = 1.5  ┬─────────────────┬─────────────────
                   │   Q2: Trap      │   Q1: Power     
                   ├─────────────────┼─────────────────
range_ratio = 0.0  │   Q3: Noise     │   Q4: Stealth   
                   └─────────────────┴─────────────────
                         pos < 0.70       pos >= 0.70
```

#### 准确性评估：严重失真与降维
当前四象限体系不仅未能精确反映白皮书的分类逻辑，反而造成明显的表达失真：
1. **Q4 Stealth 成为内部标准割裂的“垃圾桶象限”**：
   - 因为横切线定在高达 `1.5`，导致只要没超过 `1.5` 的所有有效突破都被放入 Q4。
   - **后果**：白皮书中具有优秀延续爆发力的 **🌟 Solid Breakout (0.4~1.0)**、正常的 **✅ Moderate Breakout (0.15~0.4)** 与极易夭折的 **⚠️ Marginal Breakout (0.0~0.15)** 全部混杂在同一个 Q4 象限中。筛选结果失去了优劣区分度。
2. **切分线 `range_ratio = 1.5` 缺乏量化与物理边界意义**：
   - 白皮书的核心跃迁分水岭为 **`1.0`**（当 `range_ratio > 1.0` 意味着当日最低价在阻力点之上，为纯正的跳空缺口突破）。
   - `1.5` 仅仅是处于大缺口当中的某个任意分段，并不标志交易行为或供需力量的本质突变。

---

### 2. 改进讨论：分类逻辑优化与重构方案

针对现有四象限体系对优质信号的稀释问题，提出以下两个改进方向：

#### 方案 A（极简修复版）：将分界线从 1.5 下移对齐白皮书跳空边界 1.0
保持四象限布局形式不变，但使象限边缘契合真实供需跃迁规律：
- **Q1 Power** (`range_ratio >= 1.0`, `pos >= 0.70`)：精准映射 **Gap Up 跳空高开真突破**。
- **Q4 Stealth** (`0.0 <= range_ratio < 1.0`, `pos >= 0.70`)：映射 **有效非跳空盘中突破**。
  - *配合建议*：在该象限中，建议再增加前述讨论的 **`breakout_strength >= 1.0~1.5`** 门槛，把刚蹭破阻力皮毛（Marginal）的噪音直接滤除。

#### 方案 B（推荐精准版）：用“标准 5 大模型派生视图”替代传统四象限 UI
不新增 `breakout_pattern` 存储字段，而是在 Dashboard 层基于现有两个基础字段实时派生 5 态分类筛选条件：
1. `GAP_UP`（跳空高开强攻）
2. `SOLID_BREAKOUT`（长阳实破）
3. `MODERATE_BREAKOUT`（稳健中破）
4. `MARGINAL_BREAKOUT`（弱势边破）
5. `BULL_TRAP`（冲高反落诱多）

派生规则直接复用白皮书基础字段口径：

| Dashboard 选项 | 派生过滤条件 | 说明 |
| :--- | :--- | :--- |
| `GAP_UP` | `range_ratio > 1.0 AND pos >= 0.5` | 跳空越过阻力且收在中高位。 |
| `SOLID_BREAKOUT` | `0.4 <= range_ratio <= 1.0 AND pos >= 0.7` | 突破占比充实，收盘强势。 |
| `MODERATE_BREAKOUT` | `0.15 <= range_ratio < 0.4 AND pos >= 0.5` | 标准有效穿透，尾盘稳固。 |
| `MARGINAL_BREAKOUT` | `0.0 <= range_ratio < 0.15 AND pos >= 0.5` | 仅轻微越过阻力，质量偏弱。 |
| `BULL_TRAP` | `pos < 0.5` | 冲高回落，收在 K 线下半区。 |

此方案保留 `ibd_entry_close_position` 与 `ibd_entry_breakout_range_ratio` 作为唯一真实数据字段，避免在 pool 中持久化可由基础字段确定性计算出的派生枚举。Dashboard 可以将现有 `Breakout Quadrant` 控件替换为 `Breakout Pattern` 控件，但底层仍生成字段组合过滤条件。

---

## 议题三：多维特征扩充与仪表盘筛选能力赋能（EPS 业绩与 52 周新高指标）

为了将 Dashboard 仪表盘从单纯的“K 线几何形态观察器”跃升为真正的 **“量化 CANSLIM 机构牛股发现引擎”**，针对突破池缺失的 **EPS 业绩** 与 **52 周新高** 维度，结合现有工程架构（[stage2_screener.py](file:///Users/dev/Documents/Yfinance_data/stage2_screener.py) 与 [52_wk_new_high_screener.py](file:///Users/dev/Documents/Yfinance_data/52_wk_new_high_screener.py)）展开升级架构评估。

本议题只补齐 pool 尚缺失的决策元数据。当前 `breakout_follow_pool.csv` 已具备：
- **行业字段**：`sector` / `industry` 已由 `quant_trade/yfinance_data.load_industry_lookup()` 以 `stage2_whitelist.csv` 为最高优先级注入 pool。
- **周线量比字段**：`volume_ratio` 已在 pool 中表达当前周线相对成交量，不属于本次新增字段范围。

### 1. `52wk_new_high_results.csv` 的现行作用与进阶协同

#### 现行工程角色
目前系统通过 [52_wk_new_high_screener.py](file:///Users/dev/Documents/Yfinance_data/52_wk_new_high_screener.py) 运行 `TradingView Query` 生成 `us/52wk_new_high_results.csv`。它具备双重核心作用：
1. **动量新高池（Leader Pool）**：快速筛选全美股当日最高价触及或超越 52 周新高的强势动量标的；
2. **基本面与高点特征富集源**：其查询语法（`lines 41-56`）中**天然原生查询了** `earnings_per_share_diluted_yoy_growth_fq`（季度稀释 EPS 同比增速）与 `price_52_week_high`（52周最高价）。

---

### 2. 在 `stage2_whitelist` 源头直接注入 EPS 与 52 周新高：可行性与最佳落地架构

完全可行，且具备**极佳的架构优越性与零网络耗时优势**：

#### 核心优势：基于 `TradingView Screener` 的原生零耗时查询
不同于调用缓慢且受限流的 `yfinance.Ticker().info`，[stage2_screener.py](file:///Users/dev/Documents/Yfinance_data/stage2_screener.py) 底层基于 `tradingview_screener.Query()` 构建。只需在其 `select(...)` 字段列表中**直接扩展添加以下两个字段**：
- `earnings_per_share_diluted_yoy_growth_fq`（最新季度 EPS 稀释同比增速 %）
- `price_52_week_high`（最近 52 周最高价）

#### 字典表复用升级：统一“行业 + EPS + 52W 位置字典”
目前的 `stage2_whitelist.csv` 已经是行业分类映射的最高优先级来源。通过字段扩充后，它应进一步演进为全栈共享的元数据查询字典 (Metadata Dictionary)：
- **上游源头一步到位**：Screener 查询阶段通过 TradingView 服务端即时返回全量标的的 EPS 增速与 52 周高点，无耗时；
- **下游策略无缝注入**：`breakout_follow` 在生成最终输出快照池 `breakout_follow_pool.csv` 时，基于 `code` 做内存 Left Join，自动附加：
  - `eps_yoy_growth`
  - `price_52_week_high`
  - `dist_to_52w_high_pct = (Close - price_52_week_high) / price_52_week_high`
  - `is_52w_new_high = (Close >= price_52_week_high)`
- **补充数据源只做兜底**：`eps_growth_screener_results.csv`、`52wk_new_high_results.csv`、`weekly_vol_screener_results.csv` 可继续作为补充 lookup 来源，但不应覆盖 `stage2_whitelist.csv` 中同 code 的权威元数据。
- **不重复写入已有字段**：`sector`、`industry`、`volume_ratio` 已在 pool 中存在，本次不重新定义字段口径。

---

### 3. 赋能仪表盘页面（Dashboard UI）的多维立体漏斗能力

通过这一轻量且原生的数据源头扩展，Dashboard 将形成贯穿 **基本面 (CANSLIM) + 长期阻力真空度 (52W High) + 突破形态强度 (Log Shape & Intensity)** 的四维漏斗体系：

```text
                  企业级全方位漏斗筛选体系 (Dashboard 升级)
                                  │
    ┌─────────────────────────────┼─────────────────────────────┐
    ▼                             ▼                             ▼
【第一层：基本面背书】          【第二层：空间阻力真空度】     【第三层：中枢突破质量】
 EPS YoY Growth >= 25%       dist_to_52w_high >= -5%       pos >= 0.70 (收盘结实)
 (来自 Stage2 字典快照)       OR is_52w_new_high == True    breakout_strength >= 1.5
```

- **高预期选股直觉验证**：  
  用户可在仪表盘一键筛选出：**“处在 Stage 2 长牛通道中 + 最新季 EPS 增速 ≥ 25% + 处于 52 周新高阻力真空区 + 当日呈现突破绝对视觉强度 ≥ 1.5 倍 ATR 的实质长阳标的”**，从机制上将技术面形态与企业基本面景气度完美共振。
