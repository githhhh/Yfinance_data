# Blind Rule Discovery R1-R5 最终审计结论

> 状态：本轮研究最终收口文档  
> 分支：`codex/clean-latest-quant-trade-replay-pools`  
> R4 正式执行代码基准：`5f188d0262e518af736f0fa5e1f6ca3d3bf9197c`  
> R5 正式执行代码基准：`8e33b63aecf34c0204e898f8b30ae6b220081670`  
> 研究属性：R2-R5 均为 **known-history retrospective research / chronological robustness research**。除 R1 已消费的真正 unseen holdout 外，本文不对任何后续结果作 untouched-OOS、独立 prospective 或生产 Alpha 认证。

---

## 1. 本文的唯一目的

本文只记录 R1-R5 的实验事实、数据结果、证据强弱和方法学边界。

不提供：

- B0 / Top3 修改建议；
- hard reject / penalty / 加分方向；
- 上线建议；
- “已发现 Alpha”判断；
- 对 GPT-6 的预设审计方向。

---

## 2. 实验链路总览

### R1 — 真正 unseen holdout

冻结规则：

```text
ibd_entry_volume_ratio >= 1.8
AND
eps_yoy_growth >= 15%
```

Holdout：

- selected resolved WR：30.13%；
- universe resolved WR：32.34%；
- selected 未超过 universe baseline；
- R1 FAIL；
- 该 holdout 已永久消费，之后均属于 known history。

### R2 — retrospective empirical-ceiling approximation

R2 在 known history 中出现过低覆盖、高历史 WR pocket，例如：

```text
base_depth_pct
+
ibd_entry_close_position
+
market/context condition
```

最佳历史 pocket resolved N 很小，且 evaluated-quarter support 弱；后续 R3 未稳定维持，因此 R2 的 shallow-base / high-close motif 不再视为稳定结论。

### R3 — exhaustive two-condition historical search

Full-history 最佳条件：

```text
M_8w_drawdown <= -4.72%
AND
M_dist_52w_high >= -5.69%
```

主要历史数据：

- selected N：1,822；
- resolved N：1,279；
- selected WR：43.78%；
- universe WR：28.34%；
- lift：+15.44pp；
- 12w excess p50：+2.05%；
- MAE p50：-6.68%；
- MFE p50：+16.19%。

但两项均为 broad-market `M_*` 特征，因此主要描述 market regime，而不是同一 snapshot 内的 stock ranking。

### R4 — executable-entry Winner / Stop / Path characterization

主要 outcome：

```text
Fast Winner 3w:
+20% before -8% within 15 sessions

Stop First 3w:
-8% before +20% within 15 sessions

Unresolved:
neither within 15 sessions
```

并同时报告：

- W1-W4 return / SPY excess；
- 3w/4w MAE/MFE；
- persistent stop；
- stop-first then 12w recovery；
- same-snapshot feature contrast；
- stock-only 1/2-condition interaction；
- causal chronological rolling。

### R5 — stop-risk-only chronological robustness

R5 只补 R4 未回答的问题：

> 如果训练折专门以 Stop First 风险为目标，过去历史能否在下一季度继续识别较高 Stop First / Persistent Stop 风险？

R5 同时验证 5 个 **R4 之后才冻结的 semantic families**。这些 family 方向来自已知 R4 历史，因此属于 post-hoc chronological robustness，不是 independent OOS。

---

## 3. R5 执行审计

正式执行事实：

- branch：`codex/clean-latest-quant-trade-replay-pools`；
- execution HEAD：`8e33b63aecf34c0204e898f8b30ae6b220081670`；
- pytest：105 total / 105 passed / 0 failed；
- warnings：3；
- pytest duration：20.54s；
- R5 正式执行：1 次；
- runtime：1m31.18s；
- DeepSeek / RD-Agent / LLM：0；
- source/config 未被 Gemini 修改；
- 开始和结束均仅保留既有 ` M market_analysis`。

### 3.1 W3 label purge

9 个 chronological folds：

- total purged rows：1,339；
- post-purge overlap folds：0；
- 每折 `test_quarter_in_train == false`；
- 每折 `w3_label_overlap_after_purge == false`；
- 每折 `train_max_exit_date_w3 < test_quarter_start`。

因此 R5 的 fold 内训练/测试标签边界按冻结协议通过。

---

## 4. R5 执行报告中的数据集摘要错误

Gemini 最终报告中的：

```text
total_replay_rows = 4,213
usable_entries    = 4,213
censored_rows     = 0
quarters          = 2022Q3 ... 2026Q2
```

不能接受为 R5 raw metadata 事实。

原因：同一报告的最后一个 `scope=all` rolling fold 已记录：

```text
train_rows_before_purge = 5,198
test_scope_n            = 3
```

因此仅从该 fold 就可推出该 R5 trigger-path frame 至少包含：

```text
5,198 + 3 = 5,201 rows
```

而 rolling 的首个 train window 明确从 `2022Q4` 开始，15 个 entry quarters 应对应：

```text
2022Q4 ... 2026Q2
```

不是 `2022Q3 ... 2026Q2`。

### 4.1 审计处理

本文：

- 不采用 Gemini 报告里的 `4,213 / 4,213 / 0` dataset summary；
- R5 的滚动和 family 指标继续按正式 CSV/metadata 汇总值记录；
- 不在没有原始 `stop_risk_metadata.json` 再读取的情况下重新猜测 `candidate_rows_before_maturity_filter`、`candidate_rows`、`censored_rows`；
- R5 rolling frame 可由最终 fold 计数确定为 5,201 个 usable trigger rows；
- entry-quarter 范围按 rolling 本身为 `2022Q4 ... 2026Q2`。

### 4.2 R4 / R5 population comparability 边界

R4 既有报告记录 8,983 usable trigger entries，而本次 R5 rolling frame 可反推为 5,201。

R4 与 R5 代码之间没有修改 trigger-path population 构造逻辑，因此该差异不能由 R5 代码本身解释。

在没有同时核对：

```text
R4 replay_dataset_sha256
vs
R5 replay_dataset_sha256
```

之前，**R4 与 R5 的绝对数值不能被描述成严格同一 replay artifact 上的 apples-to-apples 对照**。

因此本文只把 R4/R5 的对比写成“各自冻结实验下的结果差异”，不写成受控统计优劣试验。

---

## 5. R4 已确认的数据背景

R4 All baseline：

- evaluable N：8,962；
- Fast Winner 3w：5.55%；
- Stop First 3w：27.30%；
- Unresolved：67.15%；
- Persistent Stop：21.55%；
- W3 return p50：+0.39%；
- W3 excess p50：-0.30%；
- 3w MAE / MFE p50：-4.60% / +5.00%。

R3 favorable scope：

- Fast Winner 3w：5.55%；
- Stop First：20.18%；
- Persistent Stop：13.25%；
- W3 return p50：+1.57%；
- W3 excess p50：+0.11%；
- 3w MAE / MFE：-3.80% / +5.18%。

因此在 R4 数据中，favorable regime 最明确的关联是较低 downside / stop risk，而不是更高 3w +20% Fast Winner rate。

---

## 6. R4 full-history stock-risk characterization

### 6.1 单特征 historical associations

| 条件 | Fast Winner | Stop First | 备注 |
|---|---:|---:|---|
| `pullback_pct <= -14.4%` | 11.03% | 47.09% | Fast 与 Stop 同时提高，属于 high-dispersion pattern |
| `current_vs_ibd_candidate_pct > 3.73%` | 8.49% | 35.41% | Stop +8.11pp vs All baseline |
| `pct_above_ceiling > 22.9%` | 8.83% | 38.28% | Stop +10.98pp |
| `entry_extension_pct > 3.46%` | 6.57% | 32.67% | historical bin association |
| `pullback_v_is_dry=False` | 4.88% | 30.32% | 差异存在，但远弱于 deep-pullback / extension family |

### 6.2 strongest R4 historical stop pocket

```text
current_vs_ibd_candidate_pct >= 3.73%
AND
pullback_pct <= -14.4%
```

- N：217；
- Fast Winner：10.60%；
- Stop First：60.83%；
- baseline Stop First：27.30%；
- Stop lift：+33.53pp；
- Persistent Stop：37.33%；
- stop 后 12w recovery：38.64%；
- W3 return p50：-3.70%；
- W3 excess p50：-4.91%；
- 3w MAE / MFE：-11.60% / +6.80%。

这是 selected known-history pocket，不是独立 prospective threshold validation。

---

## 7. R4 quality-rule chronological rolling

### All

- folds：9；
- positive Path Edge：2/9；
- positive among evaluable：2/8；
- median evaluable Path Edge：-5.93pp；
- pooled selected Fast Winner：5.45%；
- selected-fold baseline Fast Winner：6.04%；
- pooled selected Stop First：36.36%；
- selected-fold baseline Stop First：28.61%；
- selected-fold pooled Path Edge：-8.34pp。

### R3 favorable

- folds：9；
- zero-selection：5/9；
- positive Path Edge：0/9；
- positive among evaluable：0/4；
- median evaluable Path Edge：-6.47pp；
- pooled selected Fast Winner：2.38%；
- selected-fold baseline Fast Winner：5.63%；
- pooled selected Stop First：23.02%；
- selected-fold baseline Stop First：21.55%；
- selected-fold pooled Path Edge：-4.71pp。

所以 R4 的 positive/quality selector 没有显示稳定的 chronological Path Edge。

---

## 8. R5 risk re-search — All scope

逐折 Stop First / Persistent Stop lift：

| Test | Selected/Eval | Stop Lift | Persistent Lift | Same-snapshot Stop Lift p50 |
|---|---:|---:|---:|---:|
| 2024Q2 | 26/26 | +5.69pp | -2.06pp | +16.46pp |
| 2024Q3 | 17/17 | -18.75pp | -16.93pp | -8.64pp |
| 2024Q4 | 31/30 | +13.01pp | +3.43pp | +15.93pp |
| 2025Q1 | 70/69 | +19.08pp | +13.51pp | +12.94pp |
| 2025Q2 | 7/5 | +2.72pp | -12.83pp | -16.13pp |
| 2025Q3 | 27/27 | +36.97pp | -0.53pp | +40.77pp |
| 2025Q4 | 10/10 | +1.76pp | +0.67pp | -19.92pp |
| 2026Q1 | 17/17 | +38.72pp | +13.06pp | +47.62pp |
| 2026Q2 | 0/0 | N/A | N/A | N/A |

Summary：

- total folds：9；
- zero-selection：1/9；
- evaluable Stop Lift folds：8；
- positive Stop Lift：7/8 = 87.5%；
- all-fold positive Stop Lift：7/9 = 77.78%；
- median Stop Lift：+9.35pp；
- pooled selected Stop First：47.26%；
- selected-fold pooled baseline Stop First：28.61%；
- pooled Stop Lift：+18.66pp；
- same-snapshot positive Stop Lift：5/8 = 62.5%。

Persistent Stop：

- positive folds：4/8；
- median Persistent Lift：+0.07pp；
- pooled selected Persistent Stop：29.85%；
- selected-fold baseline：22.01%；
- pooled lift：+7.84pp。

### 8.1 精确结论

R5 `all` risk re-search 对 **Stop First direction** 显示较高的 chronological consistency。

但不能把同一结果扩展为“同时稳定预测 Persistent Stop”：

- fold sign 仅 4/8 positive；
- median Persistent Lift 几乎为 0。

Pooled Persistent Lift 为正，但 fold-level consistency 明显弱于 Stop First。

---

## 9. R5 risk re-search — R3 favorable scope

可评估 fold 数据：

| Test | Selected/Eval | Stop Lift | Persistent Lift | Same-snapshot Stop Lift p50 |
|---|---:|---:|---:|---:|
| 2024Q2 | 60/59 | +1.65pp | +3.81pp | +0.83pp |
| 2024Q3 | 39/39 | +15.19pp | +8.57pp | +23.61pp |
| 2025Q2 | 3/3 | +84.21pp | +87.97pp | +93.55pp |
| 2025Q4 | 14/14 | +20.93pp | +16.07pp | +3.45pp |
| 2026Q1 | 1/1 | -28.00pp | -4.00pp | -31.82pp |

其它 4 folds 为 zero-selection / no favorable scope。

Summary：

- total folds：9；
- zero-selection：4/9；
- evaluable Stop Lift folds：5；
- positive Stop Lift：4/5 = 80%；
- median Stop Lift：+15.19pp；
- pooled selected Stop First：31.90%；
- selected-fold baseline：21.66%；
- pooled Stop Lift：+10.24pp；
- positive Persistent Lift：4/5；
- median Persistent Lift：+8.57pp；
- pooled Persistent Stop：24.14%；
- selected-fold baseline：14.01%；
- pooled Persistent Lift：+10.13pp；
- same-snapshot positive Stop Lift：4/5。

### 9.1 稀疏性限制

该 scope 只有 5 个可评估 folds，而且：

- 2025Q2 selected N = 3；
- 2026Q1 selected N = 1。

因此 +84pp / +88pp / +94pp 一类极端 lift 不能按稳定效应幅度理解。

方向性信息可记录，但幅度估计高度不稳定。

---

## 10. R5 post-R4 semantic families — All scope

这些 family 是在看过 R4 known history 后冻结的，因此本节只表示 **post-hoc chronological robustness**。

| Family | Eval folds | Positive Stop | Stop lift p50 | Positive Persistent | Persistent p50 | Matched Positive Stop | Matched Stop p50 | Evaluable N | Pooled Stop Lift | Pooled Persistent Lift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `deep_pullback` | 8 | 8/8 | +18.02pp | 7/8 | +8.00pp | 8/8 | +22.36pp | 659 | +17.25pp | +7.73pp |
| `deep_pullback_and_extended` | 8 | 7/8 | +32.13pp | 6/8 | +10.77pp | 7/8 | +24.71pp | 149 | +28.22pp | +11.54pp |
| `high_pct_above_ceiling` | 8 | 8/8 | +8.40pp | 8/8 | +5.29pp | 8/8 | +16.56pp | 665 | +9.19pp | +4.90pp |
| `extended_vs_candidate` | 9 | 8/9 | +5.60pp | 6/9 | +3.09pp | 62.5% | +5.89pp | 651 | +8.44pp | +1.33pp |
| `high_entry_extension` | 9 | 6/9 | +6.80pp | 5/9 | +0.85pp | 62.5% | +6.85pp | 657 | +5.68pp | -0.10pp |

### 10.1 最稳定的 historical directions

按 Stop First、Persistent Stop、same-snapshot 三个维度同时观察：

1. `deep_pullback`；
2. `high_pct_above_ceiling`；
3. `deep_pullback_and_extended`（lift 更大，但 N 更小）。

其中 `extended_vs_candidate` 的 Stop First direction 较稳定，但 Persistent Stop 较弱。

`high_entry_extension` 对 Persistent Stop 的证据最弱：pooled Persistent Lift 为 -0.10pp。

---

## 11. R5 semantic families — R3 favorable scope

| Family | Eval folds | Positive Stop | Stop p50 | Positive Persistent | Persistent p50 | Matched Positive Stop | Evaluable N | Pooled Stop Lift | Pooled Persistent Lift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `deep_pullback` | 5 | 5/5 | +13.26pp | 4/5 | +5.26pp | 4/5 | 192 | +16.36pp | +9.95pp |
| `deep_pullback_and_extended` | 4 | 3/4 | +16.70pp | 3/4 | +12.87pp | 3/4 | 42 | +25.42pp | +14.66pp |
| `high_pct_above_ceiling` | 5 | 4/5 | +10.35pp | 4/5 | +2.70pp | 4/5 | 398 | +6.23pp | +3.83pp |
| `extended_vs_candidate` | 5 | 4/5 | +7.00pp | 3/5 | +1.12pp | 3/5 | 291 | +9.27pp | +3.17pp |
| `high_entry_extension` | 5 | 5/5 | +5.92pp | 2/5 | -0.49pp | 4/5 | 321 | +6.07pp | +2.19pp |

Favorable fold 数较少，因此只记录方向，不把比例解释为精确稳定概率。

例如：

- 5/5 的 Wilson 95% interval 仍约为 56.6%-100%；
- 4/5 约为 37.6%-96.4%；
- 3/4 约为 30.1%-95.4%。

---

## 12. Same-snapshot 证据的准确边界

Same-snapshot selected-vs-unselected comparison 控制的是：

```text
相同 snapshot_date
=> 相同 broad-market state / 当日共同环境
```

它不能证明：

- 所有其他 stock-level confounders 已消除；
- feature/rule 与 Stop First 存在独立因果关系；
- 统计显著性；
- 对 unresolved 全体候选的 calibrated probability。

R5 没有 formal p-value、bootstrap CI 或 multiple-testing correction。

因此以下措辞不使用：

```text
“显著高于”
“纯粹个股风险能力”
“已证明 cross-sectional alpha”
```

准确措辞是：

> 在相同 snapshot 内，若干 post-R4 family 的 selected 组历史 Stop First rate 多数 folds 高于同 snapshot 未 selected 组。

---

## 13. Stop First 不等价于“最终坏股票”

R5 预测目标是：

```text
3 周内 -8% before +20%
```

这与“12 周后最终失败”不是同一件事。

例如部分 fold 中：

- 2025Q3 `all` risk rule：Stop First 59.26%，但 stop 后 12w recovery 为 75%；
- 2025Q2 `all`：仅一个 stop，12w recovery 100%；
- `deep_pullback` 各 folds 的 stop 后 12w recovery 大约在 27.8%-50.9%。

所以 R5 强证据首先描述的是 **路径 / 止损触发风险**，不是所有情况下的长期基本质量标签。

Persistent Stop 指标因此是必要的第二维度。

---

## 14. R4 positive-side 与 R5 risk-side 的比较边界

数据上：

```text
R4 quality-rule rolling:
All positive Path Edge       2/8 evaluable
Favorable positive Path Edge 0/4 evaluable

R5 risk re-search:
All positive Stop Lift       7/8 evaluable
Favorable positive Stop Lift 4/5 evaluable
```

在各自冻结 retrospective procedures 下，R5 Stop First direction 的时间一致性高于 R4 quality-rule Path Edge。

但不能进一步写成：

> “risk-side 已被统计证明显著、系统性地强于 winner-side”。

原因：

1. 研究目标不同：R4 优化 combined Path Edge，R5 专门优化 Stop First；
2. base rate 不同：R4 Fast Winner baseline 仅 5.55%，Stop First 约 27%；
3. R5 的研究问题是在看完 R4 known history 后才提出；
4. semantic family 也是 post-R4 才冻结；
5. R4 / R5 replay artifact SHA 尚未在执行报告中做直接一致性核对；
6. 两者不是预注册、同目标、同 population 的 head-to-head experiment。

因此只能记录“当前 retrospective procedures 下的方向一致性差异”，不能给出正式 superiority claim。

---

## 15. 最终证据分层

### 15.1 直接支持的历史事实

1. R1 unseen holdout rule 没有超过 universe WR baseline。
2. R3/R4 数据中 favorable market 的 Fast Winner rate 没升高，但 Stop First / Persistent Stop / MAE 较低。
3. R4 full-history 存在很强的 stop-risk associations，特别是 deep pullback 与 extension/above-ceiling 相关条件。
4. R4 quality-ranked stock-rule rolling 没有显示稳定 positive Path Edge。
5. R5 risk re-search 在 known-history chronological folds 中对 Stop First direction 显示较高一致性；All 为 7/8 evaluable positive。
6. R5 `all` risk re-search 对 Persistent Stop 的 fold consistency 明显弱：4/8 positive，median +0.07pp。
7. R5 post-R4 `deep_pullback`、`high_pct_above_ceiling`、`deep_pullback_and_extended` 在多个 chronological / same-snapshot 指标中保持相同 Stop First 风险方向。
8. `high_entry_extension` 对 Persistent Stop 没有同等级一致性。

### 15.2 只属于 retrospective robustness 的结论

- semantic family 的跨季度方向；
- past-only threshold regeneration 后的结果；
- R5 与 R4 的相对时间一致性描述。

这些均不是 untouched prospective validation。

### 15.3 当前数据不能支持的结论

- 已找到 production Alpha；
- 已找到可直接替代 B0 的 ranking formula；
- 某个 R5 threshold 可直接成为 hard reject；
- deep pullback / high-above-ceiling 对止损具有独立因果作用；
- same-snapshot metric 等于统计显著；
- risk-side 已经通过公平 head-to-head 实验证明优于 winner-side；
- Stop First detector 等同于长期 loser detector。

---

## 16. 对 Gemini 最终报告的审计修正

Gemini R5 执行本身通过，但最终文字报告有以下需要纠正的地方：

1. **Dataset summary 错误**：`4,213` 与 rolling raw counts 自相矛盾；本文不采用。
2. **Entry-quarter 起点错误**：报告写 `2022Q3`，rolling 实际从 `2022Q4` 开始。
3. **“样本外”措辞过强**：R5 是 known-history retrospective walk-forward，只能叫 chronological held-forward / robustness。
4. **“显著”措辞无统计依据**：未做 significance test / CI correction / multiple-testing correction。
5. **Same-snapshot 解释过强**：只能控制 snapshot-level common market context，不能证明“纯个股因果风险能力”。
6. **Q1 过度合并 Stop First 与 Persistent Stop**：All re-search 的 Stop First 很一致，但 Persistent Stop 只有 4/8 positive，median +0.07pp。
7. **Q3 过度比较 R4/R5**：不能把两种不同目标、不同 base rate、post-hoc research phase 直接写成 statistical superiority。
8. **Gemini Q3 的 R4 winner-side 数字与 R4 正式报告口径不一致**：本文继续使用冻结 R4 summary：All positive Path Edge 2/9、median -5.93pp；Favorable 0/9、median -6.47pp，而不是 Gemini Q3 中的另一组“13.9% vs 13.0%”数据。

---

## 17. 当前研究终点

R1-R5 已分别覆盖：

```text
unseen blind rule
historical empirical ceiling
market-regime separation
winner/stop path characterization
same-snapshot stock contrast
positive-rule chronological rolling
stop-risk chronological re-search
post-hoc semantic-family robustness
```

当前如果继续使用同一批已知历史做更多 threshold / family 搜索，只会进一步增加 research degrees of freedom，不能形成新的 untouched evidence。

下一次能够改变证据等级的数据必须来自新的、在规则/模型冻结之后产生的 prospective history；本文不预设该 future validation 应采用哪一种生产规则。
