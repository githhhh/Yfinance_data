# Blind Rule Discovery R4 证据强度矩阵与数据索引

> 这是 `doc/blind_rule_discovery_r4_final_conclusion_for_gpt6.md` 的审计附录。  
> R4 实验实际执行 commit：`5f188d0262e518af736f0fa5e1f6ca3d3bf9197c`。  
> 研究属性：retrospective known-history characterization / robustness research；不是 unseen holdout，不是生产 Alpha 认证。

## 1. 证据强度矩阵

| 研究问题 | 当前数据事实 | 证据等级 | 当前允许的解释 | 当前不允许的解释 |
|---|---|---|---|---|
| Market regime 是否影响 BF 触发后的路径？ | R3 favorable：Fast Winner 5.55% vs All 5.55%；Stop First 20.18% vs 27.30%；Persistent Stop 13.25% vs 21.55%；3w MAE -3.80% vs -4.60% | **强历史证据** | favorable regime 主要改善 downside / stop risk | favorable regime 提高 3w +20% winner probability |
| 是否存在明确 loser / stop-risk 特征？ | `pullback_pct <= -14.4%`：Stop First 47.09%；`current_vs_ibd > 3.73%`：35.41%；`pct_above_ceiling > 22.9%`：38.28% | **强历史描述证据** | 深 pullback、延伸/追高与 stop-first 高度相关 | 已证明这些变量具有独立因果效应 |
| 最强风险组合是什么？ | `current_vs_ibd >= 3.73% AND pullback_pct <= -14.4%`：N=217，Stop First 60.83%，baseline 27.30%，W3 excess p50 -4.91%，3w MAE -11.6% | **很强 historical risk pocket** | 可作为 B0 risk-penalty / reject hypothesis 优先审计 | 可直接上线为 hard reject rule |
| 深 base / 深 pullback 是 winner feature 吗？ | `pullback_pct <= -14.4%` Fast Winner 11.03%，但 Stop First 47.09%，Path Edge -14.30pp | **明确否定单向 winner 解读** | 这是 high-volatility / high-dispersion setup | “深 pullback 因 Fast Winner 高所以应该加分” |
| same-snapshot 能否看到 winner / loser 分离？ | favorable：EPS 0.5715、dry 0.5737、pullback duration 0.5860、volume_ratio 0.4118；close_position pair-weighted 0.6322 但 equal-weight AUC p50=0.500 | **弱到中等描述证据** | 部分特征存在历史横截面倾向 | 已统计显著、已形成稳定 ranking alpha |
| favorable market 下是否有漂亮个股 family？ | `pullback_duration>=8 & volume_ratio<=0.88`：N=70，Fast 15.71%，Stop 12.86%，Path Edge +17.48pp，W3 excess +3.14%；低 extension + 长 pullback 也较好 | **强 in-sample hypothesis** | 整理充分、量不过热、entry 不追高值得保留为机制假设 | 该固定阈值已经泛化 |
| 两条件 stock rule 能否跨季度稳定？ | causal rolling：All 正 Path Edge 2/9，median -5.93pp；Favorable 0/9，median -6.47pp | **强反证** | 当前 threshold re-search 没有稳定迁移 | 已找到可替代 B0 的固定 1-2 条件 rule |
| selected trades 相对同期 baseline 如何？ | All：Fast 5.45% vs 6.04%，Stop 36.36% vs 28.61%，selected-fold pooled edge -8.34pp；Favorable：2.38% vs 5.63%，23.02% vs 21.55%，edge -4.71pp | **强反证** | 历史最优 rule 在下一季度整体恶化 | full-history Top20 可以作为生产规则 |
| 当前最可信策略价值在哪里？ | 正向 winner rule 不稳定；stop-risk family 更强、更一致 | **策略方向证据** | 优先研究风险降权 / reject，而非继续强行找正向 Top3 加分 | 已经完成 B0 生产规则修改 |

## 2. R4 核心总体数据

### 2.1 样本

- Replay candidate rows：10,686
- Executable trigger entries：8,983
- Censored：1,703
- Censor 原因：全部 `no_entry_within_buy_zone_window`
- Evaluable entries：8,962
- Ambiguous：21
- Entry quarters：2022Q4–2026Q2
- Stock/setup + causal execution features：22
- R3 favorable rows：1,822

### 2.2 All baseline

- Fast Winner 3w：497 / 8,962 = **5.55%**
- Stop First 3w：2,447 / 8,962 = **27.30%**
- Unresolved 3w：**67.15%**
- Stop-first then 12w winner：**21.09% of stops**
- Persistent Stop / evaluable：**21.55%**
- W1 return p25/p50/p75：-2.33% / +0.15% / +2.71%
- W2：-3.48% / +0.36% / +4.19%
- W3：-4.28% / +0.39% / +5.32%
- W4：-5.02% / +0.60% / +6.31%
- W3 excess p25/p50/p75：-4.66% / -0.30% / +4.27%
- 3w MAE / MFE p50：-4.60% / +5.00%
- 4w MAE / MFE p50：-5.39% / +5.94%

### 2.3 R3 favorable baseline

- Fast Winner 3w：101 / 1,819 = **5.55%**
- Stop First 3w：367 / 1,819 = **20.18%**
- Unresolved 3w：**74.27%**
- Stop-first then 12w winner：**34.33% of stops**
- Persistent Stop / evaluable：**13.25%**
- W1 return p25/p50/p75：-1.72% / +0.68% / +3.16%
- W2：-2.47% / +1.06% / +4.92%
- W3：-2.34% / +1.57% / +6.07%
- W4：-2.23% / +2.47% / +7.57%
- W3 excess p25/p50/p75：-3.86% / +0.11% / +4.51%
- 3w MAE / MFE p50：-3.80% / +5.18%
- 4w MAE / MFE p50：-4.15% / +6.31%

## 3. 最值得保留的 loser / risk 数据

### 3.1 单特征

| 特征条件 | Fast Winner | Stop First | 关键解释 |
|---|---:|---:|---|
| `pullback_pct <= -14.4%` | 11.03% | **47.09%** | 高方差，不是纯 winner feature |
| `current_vs_ibd_candidate_pct > 3.73%` | 8.49% | **35.41%** | 延伸/追高风险；stop lift +8.11pp |
| `pct_above_ceiling > 22.9%` | 8.83% | **38.28%** | stop lift +10.98pp |
| `entry_extension_pct > 3.46%` | 6.57% | **32.67%** | 实际成交追高也偏坏 |
| `pullback_v_is_dry=False` | 4.88% | 30.32% | 有一定风险差异，但不足以支持一票否决 |

### 3.2 最强 interaction

`current_vs_ibd_candidate_pct >= 3.73% AND pullback_pct <= -14.4%`

- N：217
- Fast Winner：10.60%
- Stop First：**60.83%**
- Stop lift：**+33.53pp**
- Persistent Stop：37.33%
- Stop 后 12w recovery：38.64%
- W3 return p50：-3.7%
- W3 excess p50：-4.91%
- 3w MAE / MFE：-11.6% / +6.8%
- higher-stop-risk quarter fraction：100%

**审计限定**：这是从已知历史 threshold search 中发现的高风险 pocket；100% quarter direction 本身仍受选择过程影响，不能等同于独立 prospective 验证或因果效应。

## 4. Same-snapshot 数据的准确读法

Same-snapshot 只比较同一 `snapshot_date` 的 Fast Winner 与 Stop First，因此能控制 broad-market state；但 unresolved 不参与该成对比较，所以不是“全体候选 winner probability”。

| Feature | All pair prob | Favorable pair prob | 审计解释 |
|---|---:|---:|---|
| `eps_yoy_growth` | 0.5311 | **0.5715** | favorable 中 winner 倾向较高 EPS |
| `pullback_v_is_dry` | 0.5400 | **0.5737** | favorable 有正向描述倾向 |
| `pullback_duration_weeks` | 0.5099 | **0.5860** | favorable winner 倾向较长整理 |
| `volume_ratio` | 0.4986 | **0.4118** | favorable winner 倾向较低/不过热量比 |
| `pullback_pct` | 0.4195 | **0.3980** | winner 倾向更深，但同时 stop risk 很高 |
| `dist_to_52w_high_pct` | 0.4007 | 0.4484 | winner 倾向离高点更远，但可能主要反映高波动 |
| `ibd_entry_close_position` | 0.5301 | 0.6322 | **不能单看**：favorable equal-weight snapshot AUC p50=0.500、median diff=-0.020、正差 fraction=43.8% |

这些值是 historical descriptive separation，不是 formal statistical significance；R4 没有 bootstrap CI / p-value / multiple-testing correction。

## 5. Favorable market + stock historical families

### 5.1 `pullback_duration >= 8 & volume_ratio <= 0.88`

- N：70
- Fast Winner：**15.71%** vs favorable baseline 5.55%
- Stop First：**12.86%** vs 20.18%
- Path Edge：**+17.48pp**
- W3 excess p50：**+3.14%**
- 3w MAE / MFE：-2.2% / +6.3%
- positive-quarter fraction：75%
- worst-quarter edge：-4.4%
- same-snapshot matched edge p50：+15.0%

### 5.2 `entry_extension <= 0.014 & pullback_duration >= 8`

- N：71
- Fast Winner：8.45%
- Stop First：**9.86%**
- Path Edge：**+13.21pp**
- W3 excess p50：+2.77%
- 3w MAE / MFE：-2.7% / +6.1%
- positive-quarter fraction：75%

### 5.3 `base_depth <= -32.2 & pullback_pct >= -11.8`

- N：148
- Fast Winner：8.78%
- Stop First：12.84%
- Path Edge：+10.57pp
- W3 excess p50：+1.60%
- positive-quarter fraction：100%
- worst-quarter edge：+3.1%

**审计限定**：这些 family 全部是 retrospective full-history hypothesis。R4 causal rolling 没有维持正向规则迁移，因此不允许把这些固定阈值称为 validated alpha。

## 6. Causal rolling 作为最终 anti-overfit 证据

### 6.1 因果门禁

- Expanding-window re-search
- 每 fold 阈值只来自过去训练季度
- 训练样本要求 `exit_date_w3 < test_quarter_start`
- 累计 purge：1,339 rows
- Post-purge overlap folds：0
- Test quarter in train：0

### 6.2 All scope

- Folds：9
- Positive Path Edge：2/9 all folds，2/8 evaluable
- Median evaluable Path Edge：**-5.93pp**
- Selected / evaluable：167 / 165
- Selected Fast Winner：**5.45%**
- Selected/traded-fold baseline Fast Winner：**6.04%**
- Selected Stop First：**36.36%**
- Selected/traded-fold baseline Stop First：**28.61%**
- Selected-fold pooled Path Edge：**-8.34pp**

### 6.3 R3 favorable scope

- Folds：9
- Zero-selection folds：5/9
- Positive Path Edge：0/9 all folds，0/4 evaluable
- Median evaluable Path Edge：**-6.47pp**
- Selected / evaluable：126 / 126
- Selected Fast Winner：**2.38%**
- Selected/traded-fold baseline Fast Winner：**5.63%**
- Selected Stop First：**23.02%**
- Selected/traded-fold baseline Stop First：**21.55%**
- Selected-fold pooled Path Edge：**-4.71pp**

### 6.4 字段命名纠正

代码字段 `pooled_matched_path_edge_selected_folds` 实际是 pooled selected rates 与同一 selected/traded folds baseline 的比较，**不是 snapshot-by-snapshot matched metric**。因此应解释为：

> selected-fold pooled Path Edge

真正 rolling same-snapshot evidence 应看每 fold 的 `test_matched_*` 字段及 `matched_positive_edge_fold_fraction`。

## 7. 对 B0 / Top3 的当前可执行研究含义

当前数据**不支持**直接写一个新的 Top3 正向加分公式。

证据更强的候选方向是：

1. 将明确的 stop-risk 结构作为 risk penalty / reject hypothesis；
2. 不把 high-volatility feature 因 Fast Winner enrichment 就当成正向加分；
3. `pullback_v_is_dry=False` 不应机械一票否决；dry=True 只有弱到中等 favorable separation；
4. 如果继续建模，优先使用连续 risk estimate / cross-sectional rank，而不是继续手调固定 threshold 组合；
5. favorable regime 只能作为已知历史条件化背景，不能把其 downside uplift 重复记入 stock ranking alpha。

## 8. 不能越界的统计与因果说明

1. R4 favorable scope 继承 R3 已知历史上选择出的 market regime，本身带有 retrospective selection；不是独立 OOS market filter。
2. R4 对 148 conditions、10,396 distinct-feature pairs 做搜索，没有 formal multiple-testing correction；full-history Top rule 仅用于 hypothesis generation。
3. same-snapshot 极端组比较排除了 unresolved，不能直接解释为全体候选上的概率模型。
4. 多个 stock features 高度相关；当前结果不能证明某个单字段具有独立因果贡献。
5. 3 周 +20% Fast Winner baseline 仅 5.55%，属于稀疏极端 outcome；这会提升 winner-rule 的方差并偏爱高波动 setup。
6. `2026Q2` 是边界 entry cohort，样本极少，不应与完整季度赋予同等解释权重。
7. 最终最强的泛化检验仍是 causal chronological rolling，而该检验对当前 1-2 条件 stock rules 给出负结果。

## 9. 原始数据产物索引

R4 输出目录：

```text
backtest/blind_rule_discovery/output/trigger_path_characterization_r4/
```

核心文件：

```text
trigger_path_metadata.json
trigger_path_samples.csv
feature_bin_characterization.csv
feature_extremes.csv
within_snapshot_feature_contrasts.csv
stock_interactions_all.csv
stock_interactions_r3_favorable.csv
winner_interactions_all.csv
winner_interactions_r3_favorable.csv
stop_risk_interactions_all.csv
stop_risk_interactions_r3_favorable.csv
rolling_stock_selection.csv
```

方法学与解释文档：

```text
backtest/blind_rule_discovery/TRIGGER_PATH_CHARACTERIZATION_PROTOCOL.md
doc/blind_rule_discovery_research_findings_and_next_direction.md
doc/blind_rule_discovery_r4_final_conclusion_for_gpt6.md
```

## 10. 最终审计结论

> **R1-R4 已经足以否定“随便搜几个历史阈值就能得到稳定 Alpha”的乐观假设。当前最可靠的数据发现是：市场 regime 对 downside risk 很重要；深 pullback 叠加延伸/追高是强 stop-first 风险 family；有利市场中存在若干历史上漂亮的“充分整理 + 不过热量能 + 低追高”正向 pocket，但它们没有通过 causal chronological rolling。当前 22 个 PIT stock/execution features 尚未产生稳定可迁移的 1-2 条件 stock-selection rule。下一步若继续，应优先研究风险估计、连续横截面排序和新的独立信息，而不是继续在同一批特征上追加 threshold search。**
