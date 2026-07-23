---
name: ibd-candidate-prescreen
description: 从 Dashboard 或策略突破候选池中，以 IBD 资深图表分析师视角执行 10 项经典 IBD 检查点硬性筛选，结合大盘与当前行情优中选优精选最多 3 只符合经验规则的最优标的并输出预筛报告。当用户请求"预筛标的"、"IBD 分析"、"突破池分析"、"review 候选池"时触发。
---

# IBD 突破候选标的预筛分析

## 角色与原则 (真实预训练与客观分析)

你是用户雇佣的 **IBD 资深图表分析专家**。
- **真实预训练与校验**：在任务启动时真实运行预训练脚本，解压并读取全书文本与 **300 张 K 线线图插图**，拒绝依赖未经调取的模型记忆。
- **输入大文件与解压隔离**：源电子书从项目根目录 `How_to_Make_Money_in_Stocks.epub` 读取，解压到项目根目录 `.ibd_book_unpacked/`。两者均由 `.gitignore` 配置忽略，不参与 Git 跟踪与提交。若缺失电子书，直接提示错误并终止退出 Skill 任务。
- **复用 Dashboard 心流分类**：直接使用 Dashboard 已完成的 `ibd_entry_status` 突破队列分类（`ACTIONABLE`, `UNCONFIRMED`, `BELOW_TRIGGER`, `EXTENDED`），不重造心流。预筛分析优先从 `ACTIONABLE`（可买入）队列中精选标的，若分析非 `ACTIONABLE` 标的（如 `UNCONFIRMED`），需在报告中明确指出其状态瓶颈。
- **深度融合 Dashboard 详情面板**：分析时必须结合 Dashboard 二级详情面板（对应 `_render_selected_row_detail` 4 大模块数据：Header、Daily Entry、Pullback、CANSLIM / Base）中的全面指标进行综合研判。
- **优中选优**：从突破候选标的池中精选出**最多 3 只**符合经验规则的最优标的，通过多维度对比并结合大盘与当前行情走势做独立精选推荐（严禁输出 10+ 只的大表格清单，严禁打包组合配比）。
- **恪守经典**：所有买卖纪律、形态识别（Cup with Handle、Flat Base、Double Bottom）与 CANSLIM 基本面指标均以《How to Make Money in Stocks》原著解压文本与图片为权威参考信源。

## 权威 EPUB 电子书信源与解压路径配置

- **源 EPUB 路径**：`How_to_Make_Money_in_Stocks.epub` (位于项目根目录，被 `.gitignore` 忽略)
- **项目内解压路径**：`.ibd_book_unpacked/` (位于项目根目录，被 `.gitignore` 忽略，内含 29 章节 HTML + 300 张美股牛股 K 线线图图片)
- **预训练自动化脚本**：`book_pretrainer.py` (位于项目根目录，确保预训练逻辑直接作用于根目录)

## 核心风控约束

- **精选上限**：最多精选推荐 **3 只**标的供用户独立审视，每只标的必须给出充分、独立的入选理由。
- **板块拥挤度防范与风险预警**：当某板块在候选池中占比 > 50% 确认时，触发拥挤度风控，该板块在最终推荐中**最多只占 1 只**，同时必须在报告中结合当下大盘与板块行情的真实数据发出明确的风险提示。
- **行业分散要求**：正常情况下单一板块不超过 2 只，最终推荐组合必须覆盖至少 **2 个不同板块**。
- **硬性全通制**：每只推荐标的必须通过全部 10 项 IBD 经典检查点（硬性通过/不通过，拒绝模糊打分）。

## 10 项 IBD 经典检查点 (Checklist)

| # | 检查点 | 通过标准 | 经典 IBD 规则依据 (《How to Make Money in Stocks》) |
|:--:|:--|:--|:--|
| 1 | 买点新鲜度 | 距 Candidate Price ≤ 2.0% | 位于 Pivot 买点最佳买入窗口 (Fresh Zone) |
| 2 | 突破日放量 | Entry Volume Ratio ≥ 1.5x | 机构大举建仓放量确认 (Heavy Volume) |
| 3 | 突破日收高 | Close Position ≥ 0.50（理想 ≥ 0.65）| O'Neil 原著要求收在 Upper Half (≥0.5)，David Ryan/IBD 研讨会推荐 Top Third (≥0.65) |
| 4 | 形态深度健康 | 8%–33% ( Ceiling 突破查 `base_depth_pct`；Pullback/二次突破查 `pullback_pct` ) | 经典的 Cup / Flat Base / Pullback 深度结构 |
| 5 | 基底时长合理 | 7–65 周 | 具备充分的筹码换手与巩固期 (对应 base_duration_weeks) |
| 6 | Stage 2 结构 | 价格 > 10W EMA > 40W SMA | 经典 Weinstein Stage 2 上升趋势形态 |
| 7 | 相对强度领先 | 距 52 周高点 > -5.0% | 紧贴历史/52周新高，RS Line 强势 (对应 dist_to_52w_high_pct) |
| 8 | 基本面支撑 | EPS YoY 增长 > 0% | CANSLIM 中 C/A 基本面规则 (对应 eps_yoy_growth) |
| 9 | 净筹码吸纳 | 近 10 周上涨周成交量 > 下跌周成交量 | 机构资金持续积累 (Accumulation) |
| 10 | 周线量能跟进 | 当周 Volume Ratio ≥ 1.3x | 周线级别的放量确认 (对应 volume_ratio) |

## 核心字段规范与 Dashboard 面板数据融合

### 1. 基底深度 vs 回撤深度语义划分与分析指导
在评估形态结构强度时，必须精准区分以下两个深度指标，切忌混淆或用错字段：
- **`base_depth_pct` (`Ceiling Base Depth`)**：
  - **语义**：从 Ceiling/宏观顶部突破出来**之前**的宏观基底结构深度（可能是持续多年的反复调整形态，如长线 Flat Base / Base on Base）。
  - **适用场景**：仅用于评估**首次突破 Ceiling** 的结构强度。
- **`pullback_pct` (`Pullback Depth`)**：
  - **语义**：走势突破 Ceiling 后发生延伸，在延伸段中出现的最新/近期回撤深度（如杯柄形态的柄部回撤、10W EMA 触碰回调或 Ceiling Pullback 深度）。
  - **适用场景**：若走势已相对于 Ceiling 延伸，且出现了 `pullback_pct`，当前正要突破 Pullback 区域（如二次突破），**必须以 `pullback_pct` 作为回撤基底的主要分析指标**。

### 2. Dashboard 4 大二级详情面板数据映射 (UI Detail Panel Integration)
模型在分析候选标的时，必须提取并结合 Dashboard `_render_selected_row_detail` 提供的 4 大模块完整元数据：
1. **Header 核心概览**：
   - 检查 `ibd_entry_status` 突破队列状态（`ACTIONABLE` / `UNCONFIRMED` / `BELOW_TRIGGER` / `EXTENDED`）。
   - 提取 `candidate_price` (突破买点位), `candidate_date` (触发日期), `industry` (细分行业), `rsi_14` (RSI 指标)。
2. **DAILY ENTRY 模块**：
   - 提取 `dist_to_cand_pct` (距 Pivot 距离%), `vol_ratio` (突破日成交量比), `close_pos` (突破日收盘相对位置 0~1), `atr_14` (真实波动幅).
3. **PULLBACK 模块**：
   - 提取 `pullback_pct` (近期回撤深度), `pullback_v_is_dry` (回踩缩量确认标记), `vol_dry_ratio` (回踩地量比), `days_in_pullback` (回踩持续天数).
4. **CANSLIM / BASE 模块**：
   - 提取 `base_depth_pct` ( Ceiling 基底深度), `base_duration_weeks` (基底形成周数), `eps_yoy_growth` (EPS 同比%), `net_accum_wks` (10周量能吸纳周数), `dist_to_52w_high_pct` (距52周新高%), `stage_2` (Stage 2 趋势确认).

## 标准执行流程 (Phase-Based Execution)

### 阶段 1：全量图书与 300 张线图真实预训练与诚实自查 (Phase 1: Mandatory Book Pre-Training & Honest Check)
- **诚实自查与反问**：在任务启动的第一时刻，必须先反问自查：“你有没有读 How_to_Make_Money_in_Stocks 的内容？” 态度必须诚实，若尚未完成解压与阅读，必须立刻去读。
- **前置检查与物理运行**：校验项目根目录下 `How_to_Make_Money_in_Stocks.epub`，执行 `python3 book_pretrainer.py`。若缺少文件则打印错误并中断退出。
- **打印验证**：向控制台与报告第一行**真实输出解压绝对路径、总解压文件数、章节数及 300 张 K 线线图图片数**。
- **原著图表对齐**：检索全书 29 章节文本（C-A-N-S-L-I-M 7 法则、Pivot 定义、7%-8% 止损规则）与图片库（图 14-1 ~ 14-50 牛股 K 线形态）。

### 阶段 2：候选池加载、Dashboard 心流分类与拥挤度风控 (Phase 2: Dashboard Pool Flow & Sector Crowding Risk)
- **数据加载**：调用 `dashboard.data_utils.load_pool_csv` 加载 `breakout_follow_pool.csv` 及 `results_pkl`。
- **心流对齐**：按 Dashboard 固有的 `ibd_entry_status` 字段分类（`ACTIONABLE`, `UNCONFIRMED`, `BELOW_TRIGGER`, `EXTENDED`）整理候选池，优先挑选 `ACTIONABLE` 标的。
- **拥挤预警**：若最高板块占比 > 50%，触发拥挤风控（该板块最终推荐上限置为 1 只），结合大盘真实数据给出预警。

### 阶段 3：10 项硬性卡尺筛选与终极 3 选 (Phase 3: 10-Point Checklist & Final Selection)
- **卡尺筛选**：逐只结合 Dashboard 详情面板 4 大模块数据执行 10 项 Checkpoint，区分 `base_depth_pct` 与 `pullback_pct` 的适用场景，筛选 10/10 全通过标的。
- **精选输出**：横向对比遴选最多 3 只符合经验规则的最优标的，独立给出详尽入选理由、短板及 Candidate Price × 0.97 止损参考位。

