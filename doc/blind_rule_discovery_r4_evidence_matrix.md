# Blind Rule Discovery R4 证据矩阵（纯数据版）

> 本文是 `doc/blind_rule_discovery_r4_final_conclusion_for_gpt6.md` 的数据索引。  
> R4 实际执行 commit：`5f188d0262e518af736f0fa5e1f6ca3d3bf9197c`。  
> 只记录数据事实、证据类型和方法学限制；不包含生产策略建议，也不指定 GPT-6 的判断方向。

## 1. 证据矩阵

| 研究问题 | 数据事实 | 证据类型 | 方法学限制 |
|---|---|---|---|
| R3 favorable 是否改变 3w Fast Winner？ | 5.55% vs All 5.55% | 直接总体统计 | retrospective known history |
| R3 favorable 是否改变 Stop First？ | 20.18% vs All 27.30% | 直接总体统计 | market-regime association，不等于独立因果效应 |
| R3 favorable 是否改变 Persistent Stop？ | 13.25% vs All 21.55% | 直接总体统计 | 同上 |
| R3 favorable 是否改变 MAE？ | 3w MAE p50 -3.80% vs -4.60% | 直接总体统计 | 同上 |
| 深 pullback 是否对应更高 stop risk？ | `pullback_pct <= -14.4%`：Fast 11.03%，Stop 47.09%，Path Edge -14.30pp | full-history bin statistic | threshold 来自 known-history characterization |
| extension 是否对应更高 stop risk？ | `current_vs_ibd > 3.73%`：Stop 35.41%；`pct_above_ceiling > 22.9%`：38.28% | full-history bin statistic | 未做 untouched prospective validation |
| strongest historical stop pocket | `current_vs_ibd >= 3.73% AND pullback_pct <= -14.4%`：N=217，Stop 60.83%，baseline 27.30% | selected full-history interaction | 从大量候选中筛选；无 formal multiple-testing correction |
| same-snapshot 是否存在 feature separation？ | favorable：EPS 0.5715、dry 0.5737、pullback duration 0.5860、volume_ratio 0.4118 | descriptive matched comparison | unresolved 不参与；无 bootstrap CI / p-value correction |
| close_position 是否稳定分离 winner/stop？ | favorable pair-weighted 0.6322，但 equal-weight snapshot AUC p50=0.500，median diff=-0.020 | mixed descriptive evidence | matched snapshots 16 |
| favorable full-history 是否存在正 Path Edge pockets？ | `pullback_duration>=8 & volume_ratio<=0.88`：N=70，Path Edge +17.48pp；另有低 extension + 长 pullback 等 pockets | selected full-history interaction | known-history search；rolling 未稳定复现 |
| R4 quality-ranked stock rules 是否 rolling 泛化？ | All：positive Path Edge 2/9，median -5.93pp；Favorable：0/9，median -6.47pp | chronological causal rolling | 只验证 quality-ranked selector，不等同于专门 stop-risk selector |
| selected trades 对同期 traded-fold baseline | All pooled Path Edge -8.34pp；Favorable -4.71pp | pooled chronological comparison | 不是严格 same-snapshot pooled metric |
| stop-risk rule 是否比 winner/quality rule 更稳定？ | R4 未专门验证 | 未回答 | R5 代码已冻结但尚未执行 |

---

## 2. R4 baseline 数据

### All

- Evaluable N：8,962
- Fast Winner：497 / 8,962 = **5.55%**
- Stop First：2,447 / 8,962 = **27.30%**
- Unresolved：**67.15%**
- Stop-first then 12w winner：**21.09% of stops**
- Persistent Stop：**21.55%**
- W3 return p25/p50/p75：-4.28% / +0.39% / +5.32%
- W3 excess p25/p50/p75：-4.66% / -0.30% / +4.27%
- 3w MAE / MFE p50：-4.60% / +5.00%
- 4w MAE / MFE p50：-5.39% / +5.94%

### R3 favorable

- Evaluable N：1,819
- Fast Winner：101 / 1,819 = **5.55%**
- Stop First：367 / 1,819 = **20.18%**
- Unresolved：**74.27%**
- Stop-first then 12w winner：**34.33% of stops**
- Persistent Stop：**13.25%**
- W3 return p25/p50/p75：-2.34% / +1.57% / +6.07%
- W3 excess p25/p50/p75：-3.86% / +0.11% / +4.51%
- 3w MAE / MFE p50：-3.80% / +5.18%
- 4w MAE / MFE p50：-4.15% / +6.31%

---

## 3. Full-history feature / interaction 数据

| 条件 | Fast Winner | Stop First | 备注 |
|---|---:|---:|---|
| `pullback_pct <= -14.4%` | 11.03% | 47.09% | Fast 与 Stop 同时升高 |
| `current_vs_ibd_candidate_pct > 3.73%` | 8.49% | 35.41% | Stop lift +8.11pp |
| `pct_above_ceiling > 22.9%` | 8.83% | 38.28% | Stop lift +10.98pp |
| `entry_extension_pct > 3.46%` | 6.57% | 32.67% | 单特征分箱 |
| `pullback_v_is_dry=False` | 4.88% | 30.32% | 与 overall baseline 有差异 |

Strongest stop-risk interaction：

```text
current_vs_ibd_candidate_pct >= 3.73%
AND
pullback_pct <= -14.4%
```

- N：217
- Fast Winner：10.60%
- Stop First：60.83%
- Stop lift：+33.53pp
- Persistent Stop：37.33%
- Stop 后 12w recovery：38.64%
- W3 excess p50：-4.91%
- 3w MAE / MFE：-11.60% / +6.80%
- higher-stop-risk quarter fraction：100%

---

## 4. Same-snapshot 数据

`P(Winner feature value > Stop feature value)`：

| Feature | All | Favorable |
|---|---:|---:|
| `eps_yoy_growth` | 0.5311 | 0.5715 |
| `pullback_v_is_dry` | 0.5400 | 0.5737 |
| `pullback_duration_weeks` | 0.5099 | 0.5860 |
| `volume_ratio` | 0.4986 | 0.4118 |
| `pullback_pct` | 0.4195 | 0.3980 |
| `dist_to_52w_high_pct` | 0.4007 | 0.4484 |
| `ibd_entry_close_position` | 0.5301 | 0.6322 |

`ibd_entry_close_position` favorable 补充：

- matched snapshots：16
- equal-weight snapshot AUC p50：0.500
- median difference：-0.020
- positive difference snapshot fraction：43.8%

本节只表示 Fast-Winner 与 Stop-First 两组的历史分布差异。

---

## 5. Favorable full-history positive Path Edge pockets

### `pullback_duration >= 8 AND volume_ratio <= 0.88`

- N：70
- Fast Winner：15.71% vs favorable baseline 5.55%
- Stop First：12.86% vs 20.18%
- Path Edge：+17.48pp
- W3 excess p50：+3.14%
- 3w MAE / MFE：-2.20% / +6.30%
- positive-quarter fraction：75%
- matched snapshot edge p50：+15.0pp

### `entry_extension_pct <= 1.4% AND pullback_duration >= 8`

- N：71
- Fast Winner：8.45%
- Stop First：9.86%
- Path Edge：+13.21pp
- W3 excess p50：+2.77%
- 3w MAE / MFE：-2.70% / +6.10%

### `base_depth_pct <= -32.2% AND pullback_pct >= -11.8%`

- N：148
- Fast Winner：8.78%
- Stop First：12.84%
- Path Edge：+10.57pp
- W3 excess p50：+1.60%
- positive-quarter fraction：100%

这些均为 known-history selected interactions。

---

## 6. R4 causal rolling

### All

- folds：9
- positive Path Edge：2/9
- positive among evaluable：2/8
- median evaluable Path Edge：-5.93pp
- selected N：167
- evaluable N：165
- pooled Fast Winner：5.45%
- traded-fold baseline Fast Winner：6.04%
- pooled Stop First：36.36%
- traded-fold baseline Stop First：28.61%
- selected-fold pooled Path Edge：-8.34pp

### Favorable

- folds：9
- zero-selection：5/9
- positive Path Edge：0/9
- positive among evaluable：0/4
- median evaluable Path Edge：-6.47pp
- selected N：126
- pooled Fast Winner：2.38%
- traded-fold baseline Fast Winner：5.63%
- pooled Stop First：23.02%
- traded-fold baseline Stop First：21.55%
- selected-fold pooled Path Edge：-4.71pp

Rolling W3 label purge：

- purged rows：1,339
- post-purge overlap folds：0

---

## 7. 研究状态边界

- R1：真正未见 holdout，结果未超过 universe winner-rate baseline。
- R2-R4：known-history retrospective research。
- R4：没有 formal multiple-testing correction。
- R4：quality-ranked rolling 为负或不稳定。
- R4：没有专门以 stop-risk 为唯一选择目标做 causal rolling。
- R5：代码、协议、测试已冻结；尚未正式执行，因此目前不存在 R5 数据结论。

在 R5 执行并完成审计前，本文不对“risk-side 是否比 winner-side 更稳定”作任何判断。
