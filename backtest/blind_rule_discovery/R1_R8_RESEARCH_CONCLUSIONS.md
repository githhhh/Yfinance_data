# R1–R8 Empirical Conclusions for B0 / Skill Ranking

Date: 2026-09-09  
Research branch: `codex/clean-latest-quant-trade-replay-pools`

## Purpose

This report consolidates the empirical conclusions from R1–R8. It is a decision-oriented synthesis, not another rule search.

The central question is not whether historical data can produce an attractive ranking formula. It is whether the available point-in-time evidence supports a **stable, repeatable rule for selecting the best BreakoutFollow candidates**, and which observed features instead describe downside/path risk without constituting ranking alpha.

The evidence base is the reconstructed executable population used by R4–R8. R8A binds to the same R4/R6 sample and contains 8,983 usable executable entries, 1,140 tickers and 182 snapshot weeks. Its outcome-independent `nonoverlap_w3` issuer panel contains 6,740 rows and is the preferred dependency-sensitive view.

This is known-history retrospective research. It is not an untouched holdout and does not authorize a production rule change by itself.

---

## Executive Conclusions

1. **R1–R8 did not identify a stable Winner-ranking alpha.** Historical pockets exist, but the evidence repeatedly fails rolling, outer-quarter, economic, or dependency-sensitive validation when used as a general selector.

2. **Stable path/risk structure does exist.** Several features distinguish Fast Winner from Stop First, and several historical pockets concentrate Stop First. However, higher Stop risk is not equivalent to lower expected value and is not sufficient evidence for a penalty or veto.

3. **The strongest economic risk hypothesis is the interaction `deep pullback AND highly extended vs candidate`.** In R7 it was the only one of six frozen risk policies to pass the predeclared W1/W2/W4 historical economic-direction gate. It is narrow and should be treated as a prospective risk-flag hypothesis, not a proven production exclusion rule.

4. **Generic penalties for deep pullback, extension, high `pct_above_ceiling`, or `pullback_v_is_dry=False` are not supported.** R7 either found mixed/negative economic value or R8A found insufficient stable Winner-vs-Stop discrimination.

5. **`pullback_v_is_dry=False` should not be treated as an intrinsic negative signal.** `dry=True` shows a mild Winner-favorable descriptive tendency, but it is not stable enough to justify a production bonus either.

6. **The evidence argues against replacing B0 with a more complicated weighted score.** A more defensible architecture is: strict validity/data-quality gates, a small number of separately labeled evidence-backed risk flags, and weak/tie-aware ordering among otherwise qualified candidates until prospective evidence demonstrates true ranking alpha.

---

## 1. What R1–R8 Say About Ranking Alpha

| Study | Main result | What it supports |
|---|---|---|
| R1 | `entry_volume >= 1.8` + `EPS >= 15%` failed holdout: 30.13% vs 32.34% universe winner rate | Simple quality thresholds are not demonstrated Winner alpha |
| R2 | Attractive ex-post rule reached 65.6% WR but had tiny support / only four quarters and was unstable | Ex-post high WR is insufficient evidence |
| R3 | Market-regime pair showed strong historical risk separation but weak rolling stability and many zero-selection folds | Market regime is better interpreted as exposure/path context than stock-ranking alpha |
| R4 | Historical feature pockets were easy to find, but rolling selector validation was poor | Historical associations do not survive as stable Winner selectors |
| Corrected R5 | Strong Stop-risk families persisted after leakage/composition corrections, but this was risk characterization rather than Winner ranking | Risk pockets are real enough to study economically |
| R6 | RD-Agent found Stop-enriched pockets, but only 2/9 total outer quarters satisfied both positive Stop lift and positive risk/cost tradeoff; simple comparator 0/9 | Adaptive search did not produce a stable general selector |
| R7 | Only one narrow interaction passed the historical economic-direction gate; generic risk rules were mixed or negative | Economic triage is necessary before turning risk association into a veto |
| R8A | 19 PIT features produced 0 `CONSISTENT_WINNER_HIGH` features in the primary `nonoverlap_w3` panel | No univariate feature supports a simple “higher is better” Winner ranking |

The repeated failure mode is consistent: a rule can look strong in pooled history yet lose stability after chronological freezing, dependency control, or economic accounting.

**Primary conclusion:** the current data do not support a claim that a weighted combination of the available PIT fields can reliably order qualified candidates from “best future Winner” to “worst future Winner.”

---

## 2. Market Regime Changes Risk, Not Fast-Winner Frequency

R4's executable baseline contained 8,983 entries (8,962 evaluable, 21 ambiguous):

- Fast Winner: **5.55%**
- Stop First: **27.30%**
- Unresolved at W3: **67.15%**
- W3 median return: **+0.39%**
- W3 median excess return: **-0.30%**
- median MAE: **-4.60%**
- median MFE: **+5.00%**

Under the R3-favorable market-regime filter:

- Fast Winner remained **5.55%**
- Stop First fell to **20.18%**
- Unresolved rose to **74.27%**
- W3 median return improved to **+1.57%**
- median MAE improved to **-3.80%**

This is a useful separation of concepts. Favorable regime information reduced downside/path stress without increasing the incidence of the defined three-week +20% Fast Winner.

**Implication:** market regime can reasonably influence exposure, aggressiveness, or risk context. The evidence does not justify using it as a stock-level Winner ranking factor.

---

## 3. Risk Association Is Not the Same as Economic Veto Value

R4/R5 found several strong historical Stop-risk associations. Examples include deep pullbacks, deep bases, distance below the 52-week high, extension relative to candidate price, and the interaction of deep pullback with extension.

The strongest historical interaction was approximately:

`current_vs_candidate high AND pullback_pct deep`

In the R4 historical characterization this pocket had very high Stop First incidence, but it also had elevated Fast Winner incidence. That fact alone prevents interpreting “high Stop rate” as “bad expected trade.”

R7 therefore evaluated six already-frozen risk policies economically rather than selecting a new champion. It used same-week, same-N random exclusion as a benchmark and held vetoed slots as cash rather than reallocating them.

### R7 decision matrix

| Frozen policy | R7 verdict |
|---|---|
| `deep_pullback` | `MIXED_ECONOMIC_DIRECTION` |
| `deep_pullback_and_extended` | **`HISTORICAL_ECONOMIC_DIRECTION`** |
| `extended_vs_candidate` | **`ECONOMIC_VALUE_NOT_DEMONSTRATED`** |
| `high_pct_above_ceiling` | `MIXED_ECONOMIC_DIRECTION` |
| `r6_rdagent` | `MIXED_ECONOMIC_DIRECTION` |
| `r6_simple` | `MIXED_ECONOMIC_DIRECTION` |

The only policy that crossed the full predeclared economic-direction gate was:

`pullback_pct <= past q20 AND current_vs_ibd_candidate_pct >= past q80`

The thresholds were fitted from the purged past in each fold; this is not a fixed `-14% / +3.7%` production threshold.

On the dependency-controlled `nonoverlap_w4` panel, its incremental return versus same-week random exclusion was approximately:

- W1: **+7.44 bp**
- W2: **+7.12 bp**
- W4: **+12.79 bp**

Its cash delta versus no-veto baseline was approximately:

- W1: **+8.15 bp**
- W2: **+7.52 bp**
- W4: **+9.89 bp**

At W4 the avoided-loss contribution was about +26.31 bp versus roughly -17.12 bp of foregone gains. All eight supported W4 quarters had positive incremental direction; leave-one-quarter, leave-one-ticker, and best-week removal remained positive. The W4 block-bootstrap interval was approximately +5.45 to +22.43 bp.

The policy is narrow: activation coverage in the relevant nonoverlap test panel was about **3.5%**, and the required pullback fields are not observed for every entry. Unknown observations were retained, never treated as safe.

**Implication:** this interaction deserves prospective monitoring as a separately labeled risk flag. R7 does not prove that it should be a hard rejection rule today.

---

## 4. R8A: Stable Winner / Stop Feature Atlas

R8A evaluated all 19 allowed snapshot-PIT features without a top-N prefilter. It compared Fast Winner and Stop First chronologically using purged-past normalization and a shared outcome-independent `nonoverlap_w3` ticker schedule.

The formal stability label required:

- at least 6 supported outer quarters;
- at least 75% same matched-week direction;
- absolute median Cliff's delta >= 0.10;
- aligned median matched-percentile gap and Cliff's delta;
- leave-one-quarter median sign stability.

### Stable numerical contrasts in `nonoverlap_w3`

| Feature | Label | Supported quarters | Direction consistency | Median matched percentile gap | Median Cliff delta |
|---|---|---:|---:|---:|---:|
| `dist_to_52w_high_pct` | `CONSISTENT_STOP_HIGH` | 8 | 75.0% | -0.1046 | **-0.2492** |
| `pullback_pct` | `CONSISTENT_STOP_HIGH` | 7 | **85.7%** | -0.0843 | **-0.2366** |
| `base_depth_pct` | `CONSISTENT_STOP_HIGH` | 8 | 75.0% | -0.0668 | -0.1717 |
| `base_mbox_count` | `CONSISTENT_STOP_HIGH` | 8 | **87.5%** | -0.0442 | -0.1408 |

There were **0 `CONSISTENT_WINNER_HIGH` features**.

### Critical interpretation of `STOP_HIGH`

`STOP_HIGH` is a **numeric direction label**: Stop observations tend to have numerically higher values than Winner observations. It does **not** mean “the more extreme/deeper value is more dangerous.”

For negatively signed percentage fields, lower means more negative/deeper. R8A therefore finds that, relative to Stop First, Fast Winners often had:

- **deeper pullbacks** (`pullback_pct` more negative);
- **deeper bases** (`base_depth_pct` more negative);
- **greater distance below the 52-week high** (`dist_to_52w_high_pct` more negative);
- lower `base_mbox_count`.

Examples from the nonoverlap panel:

- 2024Q2 `pullback_pct`: Winner median **-19.5%**, Stop median **-12.85%**, Cliff delta **-0.573**.
- 2025Q3 `pullback_pct`: Winner **-14.4%**, Stop **-13.3%**.
- 2026Q1 `pullback_pct`: Winner **-13.1%**, Stop **-10.6%**.
- 2026Q1 `dist_to_52w_high_pct`: Winner **-5.86%**, Stop **-2.48%**.

This does not contradict the R4/R5 observation that deep pullbacks can have a high Stop rate versus the full population. A region can simultaneously contain more Fast Winners **and** more Stop First outcomes while containing fewer unresolved trades. In other words, some features may describe **outcome intensity / path dispersion**, not a monotonic quality axis.

That distinction is central to B0 design: a feature associated with Stop risk cannot automatically be assigned a negative rank weight.

---

## 5. What R8A Says About Existing Checklist / Ranking Features

### `pullback_v_is_dry`

Primary `nonoverlap_w3` result:

- stability: `MIXED_OR_WEAK`
- supported quarters: 7
- Winner-high / Stop-high: 4 / 2 (with one zero-direction quarter)
- direction consistency: **57.1%**
- median matched percentile gap: **+0.0313**
- median Cliff delta: **+0.1147**
- high-tail Winner-vs-Stop log-odds: **+0.338**

This is consistent with `dry=True` being mildly constructive in some periods, but the cross-quarter direction is not stable.

**Supported conclusion:** `pullback_v_is_dry=False` should not be an automatic penalty or risk flag.  
**Not supported:** converting `dry=True` into a material production ranking bonus.

### Entry volume and weekly volume

- `ibd_entry_volume_ratio`: 4 Winner-high / 4 Stop-high quarters; median Cliff delta only about +0.076.
- `volume_ratio`: 5 / 3; median Cliff delta about +0.060.

These may remain useful quality/context facts, but they are not demonstrated stable Top3 ranking alpha.

### EPS growth

`eps_yoy_growth` was 5 Winner-high / 3 Stop-high with median Cliff delta about +0.088 and near-zero median matched-percentile gap. It did not pass the stability/effect gate.

EPS can remain a fundamental quality characteristic. The available experiment does not support assigning it a strong Winner-ranking weight.

### Breakout geometry / extension

`current_vs_ibd_candidate_pct`, `pct_above_ceiling`, `ibd_entry_close_position`, `ibd_entry_breakout_range_ratio`, and `ibd_entry_close_vs_trigger_pct` all failed the complete stability gate in R8A.

R7 further showed that **extension alone** was economically harmful as a veto despite its historical Stop association.

This is another example of why geometry can be useful for validity/context classification without necessarily being a monotonic ranking score.

---

## 6. Evidence Matrix for B0 / Skill Decisions

| Candidate interpretation / rule | Evidence status | Conclusion |
|---|---|---|
| Higher entry volume should materially rank candidates | Weak / unstable | **Do not use as strong ranking weight** |
| Higher EPS YoY should materially rank candidates | Weak discrimination in this sample | **Quality/context only unless future evidence improves** |
| `pullback_v_is_dry=False` deserves a penalty | R8A unstable; prior effect weak | **Not supported** |
| `pullback_v_is_dry=True` deserves a bonus | Mild favorable tendency, unstable | **Insufficient evidence** |
| Deep pullback alone deserves a penalty/veto | R7 mixed; R8A Winners can be deeper than Stops | **Not supported** |
| High extension alone deserves a penalty/veto | R7 economic value not demonstrated | **Contradicted as generic veto** |
| High `pct_above_ceiling` alone deserves a penalty/veto | R7 mixed | **Not supported** |
| Deep pullback + high extension is a risk pocket | R5 risk association + R7 economic triage | **Strongest historical risk-flag hypothesis** |
| Market regime should rank individual stocks | R3/R4 does not increase Fast Winner frequency | **Not supported** |
| Market regime can affect exposure/risk posture | Stop/MAE improve in favorable regime | **Supported as context** |
| A more complex weighted B0 score will solve Top3 selection | R1–R8 found no stable Winner alpha | **Not supported by current evidence** |
| Change production immediately | All work is known-history retrospective | **No — keep frozen** |

---

## 7. What This Means for B0 / Top3 Architecture

The strongest conclusion from the full research program is not a new formula. It is that **the current feature set has demonstrated more value for validity/risk description than for precise winner ordering**.

A future B0/Skill design should therefore avoid pretending that weak historical differences are precise ranking information.

A more evidence-consistent architecture would separate three functions:

### A. Hard validity / data-quality gates

Use only conditions that determine whether the setup or underlying data are valid enough to review. A hard gate should not be created merely because a feature historically correlated with Stop First.

### B. Explicit risk/context flags

Keep risk information separate from winner quality. The clearest current candidate is the narrow **deep-pullback + high-extension** interaction. Market regime belongs here as portfolio/exposure context rather than stock alpha.

Risk flags should be visible to the reviewer and can affect sizing/attention before they are allowed to become hard rejections.

### C. Weak or tie-aware ordering among valid candidates

Where no stable ranking alpha exists, avoid manufacturing precision from many small weights. Reasonable future designs include coarse tiers, ties, diversification constraints, or weak ordering based only on clearly interpretable evidence.

This architectural direction is an implication of the evidence, not itself a validated production strategy. It must still be evaluated prospectively before replacing B0.

---

## 8. Why the Search Should Stop Here

R8B compressed the first RD-Agent interaction prompt to roughly 1.2k characters, but the same proxy terminated the reasoning request at approximately the 60-second first-byte boundary. More importantly, R8A already delivered the missing full Winner/Stop atlas.

Continuing adaptive interaction search now has decreasing expected information value and increasing data-mining risk. The correct next step is not another retrospective search round designed to find something that passes.

If a production hypothesis is selected, it should be frozen in advance and evaluated on genuinely future observations without threshold retuning.

---

## 9. Research Boundaries

The conclusions above must retain the following limitations:

- R1–R8 use already-known historical periods; they are not a pristine untouched holdout.
- The R4/R8 population contains **usable executable entries only**, not all listings or all original signals.
- R7 audits endpoint-conditioned economic returns; it does not reproduce exact stop-order execution or full portfolio P&L.
- Some pullback-related fields have substantial missingness. Unknown values were retained rather than labeled safe.
- `nonoverlap_w3` reduces repeated-ticker/path dependence but does not make observations statistically independent.
- R6 adaptive Agent evidence has a model-alias/cache provenance caveat from earlier runs; R8A is deterministic and unaffected by that issue.
- No result here establishes causality.

---

## Final Research Position

The combined R1–R8 evidence supports the following research position:

> **The available PIT feature set is materially better at describing trade path and downside risk than at identifying the future top Winner. Stable Winner-ranking alpha has not been demonstrated. Risk associations should therefore not be silently converted into ranking penalties.**

The single most credible narrow historical risk hypothesis is **deep pullback combined with high extension versus candidate**, but it remains a prospective hypothesis rather than a production veto.

For the current B0/Skill system, the evidence specifically argues against automatic penalties for `pullback_v_is_dry=False`, deep pullback alone, extension alone, or high `pct_above_ceiling` alone, and against solving Top3 selection by adding more weakly supported score weights.

**KEEP PRODUCTION FROZEN until an explicit prospective rule is frozen and independently evaluated.**

---

## Evidence Map

- R4 executable sample: `backtest/blind_rule_discovery/output/trigger_path_characterization_r4/trigger_path_samples.csv`
- R6 adaptive risk discovery: `backtest/blind_rule_discovery/output/r6_risk_features_06/`
- R7 economic triage: `backtest/blind_rule_discovery/output/r7_economic_triage_01/`
- R8 deterministic Winner/Stop atlas: `backtest/blind_rule_discovery/output/r8_winner_stop_atlas_03/`
- Corrected R5 artifact/result commit: `a7933f6be74def607d8ebe06c1848204665be557`
