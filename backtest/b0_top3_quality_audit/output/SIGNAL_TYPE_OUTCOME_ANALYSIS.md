# Signal Type Outcome Analysis

## Executive Conclusion

This is a **research-only diagnostic** on the frozen EPS-recalibrated historical baseline. It does not modify production selection logic, rule thresholds, price outcomes, EPS PIT data, or the forward-shadow registration.

The analysis combines:

- the final EPS-recalibrated historical pools (`EPS_RECALIBRATED_V2`);
- all 97 actual B0 picks;
- per-pick W1 path outcomes for all 97 picks;
- W1/W2/W3/W4 per-pick common-support outcomes for full-Top3 weeks;
- stop/profit20/as-of path outcomes;
- fixed Train / contaminated-validation calendar splits.

### Final verdict

> **No signal type currently qualifies for a new explicit production weight.**

However, the data does materially improve our understanding of the existing selector:

1. **B0 is strongly signal-type non-neutral.** `ceiling` is 34.0% of ACTIONABLE candidates but 73.2% of B0 picks.
2. **ceiling is the only signal with both meaningful sample size and stable observed absolute returns across Train -> contaminated validation.**
3. **That does not prove independent ceiling alpha.** After controlling for week and rank position, ceiling residual return is approximately neutral/slightly negative across W1-W4.
4. **pivot is the strongest challenger hypothesis.** It shows large positive within-week/rank-adjusted residuals, but Train -> validation reverses sharply, so the apparent edge is not stable.
5. **ma10_touch_confirm has negative directional evidence**, especially W1/W2 and in the later period, but sample size is still too small for a production penalty.
6. **ceiling_pullback and three_weeks_tight remain underpowered.** No weight decision is justified.

The correct current interpretation is therefore:

> **The existing B0 behaves as a ceiling-heavy top-bucket selector, but this study does not establish that the ceiling overweight itself is the source of B0's historical advantage.**

---

## 1. Data and Method

### Frozen inputs

The analysis uses existing frozen/recalibrated research artifacts only:

- `backtest/ibd_skill_replay_pools/*/breakout_follow_pool.csv`
- `backtest/b0_top3_quality_audit/output/b0_selection_events.csv`
- `backtest/b0_top3_quality_audit/output/b0_path_quality_to_asof.csv`
- `backtest/b0_top3_quality_audit/output/b0_rank_position_weekly_detail.csv`

No price refresh, selector rerun, EPS refetch, or rule search is performed.

### Historical signal universe

The final EPS-recalibrated pools contain:

- signal candidates: **2,738**
- ACTIONABLE candidates: **733**
- actual B0 picks: **97**

### Horizon support

For all 97 B0 picks:

- W1 is available from the frozen path-quality output.

For W1-W4 signal-type comparisons, the analysis uses the existing common-support full-Top3 sample:

- W1: 25 weeks / 75 picks
- W2: 25 weeks / 75 picks
- W3: 24 weeks / 72 picks — diagnostic only
- W4: 23 weeks / 69 picks

This deliberately preserves the existing common-support maturity rule and does not impute missing horizons.

### Additional controls

Because signal type is strongly entangled with B0 rank/lane, raw return means are not treated as causal signal effects.

For the full-Top3 common-support panel, two additional diagnostics are computed:

1. **Within-week relative return**
   - individual return minus that week's B0 Top3 mean;
2. **Week + rank fixed-effect residual**
   - return minus week mean minus rank-position mean plus grand mean.

A deterministic cluster bootstrap over snapshot weeks is used to show the uncertainty around the fixed-effect residual mean.

These diagnostics reduce—but cannot fully remove—the selection-conditioning bias of observing only B0-selected names.

---

## 2. Signal Distribution Before and After B0

| Signal | Signal candidates | Signal share | ACTIONABLE | ACTIONABLE share | B0 picks | B0 share | Selection lift vs ACTIONABLE share |
|---|---:|---:|---:|---:|---:|---:|---:|
| ceiling | 781 | 28.5% | 249 | 34.0% | 71 | 73.2% | **2.15x** |
| pivot | 1,042 | 38.1% | 249 | 34.0% | 10 | 10.3% | **0.30x** |
| ma10_touch_confirm | 661 | 24.1% | 168 | 22.9% | 11 | 11.3% | **0.49x** |
| ceiling_pullback | 163 | 6.0% | 39 | 5.3% | 4 | 4.1% | 0.77x |
| three_weeks_tight | 91 | 3.3% | 28 | 3.8% | 1 | 1.0% | 0.27x |

### Key finding

`ceiling` and `pivot` have exactly the same number of ACTIONABLE candidates:

> **249 vs 249**

Yet B0 selected:

> **71 ceiling vs 10 pivot**

Therefore the current selector already contains a very strong **implicit signal-type preference**.

This is not a small side effect. It is one of the dominant structural properties of the current B0 output.

---

## 3. All 97 B0 Picks — W1 and Path Risk

| Signal | n | W1 median | W1 mean | W1 positive | Stop8 ever | Profit20 | As-of median | Executed median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ceiling | 71 | **+0.91%** | **+1.22%** | 57.7% | 45.1% | 42.3% | +3.30% | -2.50% |
| pivot | 10 | **+1.55%** | +0.06% | 60.0% | 50.0% | 40.0% | -0.44% | -4.61% |
| ma10_touch_confirm | 11 | **-1.24%** | **-1.33%** | 27.3% | **63.6%** | 54.5% | -0.13% | **-8.00%** |
| ceiling_pullback | 4 | +3.27% | +4.31% | 75.0% | 50.0% | 75.0% | +50.16% | +14.54% |
| three_weeks_tight | 1 | -0.92% | -0.92% | 0.0% | 0.0% | 0.0% | +2.08% | +2.08% |

### Important correction on ceiling_pullback

The full-Top3 common-support subset contains only two `ceiling_pullback` names and looks poor at later horizons.

That subset must **not** be used to label the signal type as weak.

Across all four actual B0 `ceiling_pullback` picks, W1 is instead:

- median: **+3.27%**
- mean: **+4.31%**
- positive rate: **75%**

But **n=4 remains far too small** for a production inference.

Correct classification:

> **INSUFFICIENT SAMPLE**, not negative evidence.

---

## 4. Train -> Contaminated Validation Stability — All-Pick W1

| Signal | Train n | Train W1 median | Val n | Val W1 median | Stability |
|---|---:|---:|---:|---:|---|
| ceiling | 45 | +0.56% | 26 | **+1.03%** | **stable / same sign** |
| pivot | 5 | **+3.66%** | 5 | **-2.09%** | **reversal** |
| ma10_touch_confirm | 9 | -1.24% | 2 | -6.31% | weak / worsening |
| ceiling_pullback | 4 | +3.27% | 0 | — | no validation support |
| three_weeks_tight | 1 | -0.92% | 0 | — | no validation support |

This is one of the strongest diagnostics in the study.

### ceiling

The observed W1 signal does **not** reverse:

- Train median: +0.56%
- contaminated validation median: +1.03%

### pivot

The observed W1 signal **does reverse**:

- Train median: +3.66%
- contaminated validation median: -2.09%

This is why pivot cannot be promoted from historical challenger to explicit production weight.

---

## 5. Common-Support W1-W4 Absolute Outcomes

### ceiling

| Horizon | n | Median | Mean | Positive | Train median | Val median |
|---|---:|---:|---:|---:|---:|---:|
| W1 | 59 | +0.43% | +1.19% | 55.9% | +0.10% | +1.03% |
| W2 | 59 | +0.60% | +1.41% | 55.9% | +0.60% | +1.10% |
| W3* | 56 | +1.41% | +3.09% | 64.3% | +1.09% | +1.54% |
| W4 | 54 | **+2.79%** | **+5.61%** | **64.8%** | +1.13% | **+4.19%** |

* W3 is diagnostic only.

This is the best stability profile among the signal types because it combines:

- large sample size;
- positive W1/W2/W4 medians;
- no Train -> validation sign reversal.

### pivot

| Horizon | n | Median | Mean | Positive | Train median | Val median |
|---|---:|---:|---:|---:|---:|---:|
| W1 | 8 | +1.55% | +0.15% | 62.5% | +3.66% | **-2.09%** |
| W2 | 8 | -0.94% | +0.15% | 37.5% | **+8.08%** | **-1.07%** |
| W3* | 8 | -0.63% | +0.88% | 37.5% | **+8.36%** | **-2.73%** |
| W4 | 7 | **+10.63%** | +1.18% | 57.1% | **+12.06%** | **-13.50%** |

The W4 median is eye-catching, but the temporal reversal is extreme.

Therefore:

> **pivot is high-convexity / high-instability evidence, not stable alpha evidence.**

### ma10_touch_confirm

| Horizon | n | Median | Mean | Positive | Train median | Val median |
|---|---:|---:|---:|---:|---:|---:|
| W1 | 6 | -1.66% | -3.92% | 16.7% | -1.66% | -6.31% |
| W2 | 6 | -2.87% | -2.90% | 33.3% | +0.63% | -8.90% |
| W3* | 6 | +0.72% | -0.44% | 66.7% | +2.52% | -5.97% |
| W4 | 6 | +0.44% | -1.73% | 50.0% | +3.45% | -8.01% |

This is directionally weak, particularly in W1/W2 and in the later period.

But with only six full-Top3 common-support observations:

> **CAUTION, not production penalty.**

### ceiling_pullback

Common-support n is only 2.

No W1-W4 production conclusion is statistically justified.

### three_weeks_tight

No meaningful W1-W4 common-support sample exists.

---

## 6. Does Signal Type Add Information Beyond Week and Rank?

Raw signal returns are confounded by:

- market week;
- B0 rank position;
- lane;
- geometry;
- volume;
- proximity;
- other selector dimensions.

The week + rank fixed-effect residual diagnostic asks a narrower question:

> after removing the average effect of that week and the average effect of Rank1/Rank2/Rank3, does a signal type still tend to sit above or below the expected return?

### Fixed-effect residual mean

| Signal | W1 | W2 | W3* | W4 |
|---|---:|---:|---:|---:|
| ceiling | +0.09% | -0.18% | -0.44% | -0.68% |
| pivot | **+1.84%** | **+2.78%** | **+3.61%** | **+7.26%** |
| ma10_touch_confirm | -3.00% | -1.96% | +0.10% | +0.85% |
| ceiling_pullback | -1.12% | +0.17% | -2.40% | -9.62% |

### Cluster-bootstrap 95% interval

ceiling:

- W1: [-0.21, +0.46]
- W2: [-0.79, +0.38]
- W3: [-1.39, +0.32]
- W4: [-1.95, +0.31]

Interpretation:

> **No independent positive ceiling signal effect is demonstrated after week/rank adjustment.**

This matters.

ceiling has the strongest observed stability, but the data does **not** show that “being ceiling” itself adds return once current selection/rank context is accounted for.

pivot:

- W1 residual bootstrap interval: [+0.28, +3.65]
- W2: [+0.22, +7.46]
- W3: [+0.04, +10.60]
- W4: [+0.01, +20.99]

Numerically this is the strongest challenger signal.

However, this must not be read as production-grade significance because:

1. pivot sample is only 7-8 observations per mature horizon;
2. only 5-6 weeks contain those observations;
3. the Train -> contaminated-validation medians reverse sharply;
4. several large winners materially affect the result;
5. the sample is post-selection conditioned by B0 itself.

Therefore the correct verdict remains:

> **PROMISING CHALLENGER / NOT STABLE ENOUGH TO WEIGHT**

ma10 and ceiling_pullback intervals are too wide to support an independent effect.

---

## 7. Signal Type Is Entangled with Existing Rank

B0 pick-order distribution:

| Signal | Rank1 | Rank2 | Rank3 |
|---|---:|---:|---:|
| ceiling | 31 | 23 | 17 |
| pivot | 1 | 3 | 6 |
| ma10_touch_confirm | 7 | 2 | 2 |
| ceiling_pullback | 1 | 3 | 0 |
| three_weeks_tight | 0 | 1 | 0 |

This confirms that signal type is already embedded in the current ranking structure.

Notably:

- ceiling is heavily represented at Rank1;
- pivot is mostly pushed to Rank3;
- ma10 is disproportionately Rank1 despite its weak W1 profile.

This is why raw signal averages cannot be converted directly into a new scoring table.

---

## 8. What This Study Does and Does Not Prove

### Supported

**A. The current B0 is ceiling-heavy by construction.**

This is quantitatively large, not cosmetic.

**B. ceiling is the most stable observed signal family in the current selected sample.**

Its absolute W1/W2/W4 results remain positive across the fixed Train -> contaminated-validation split.

**C. pivot deserves explicit challenger monitoring.**

It shows the strongest rank/week-adjusted upside signal.

**D. ma10_touch_confirm deserves caution.**

Its observed W1 and risk profile are weaker than ceiling and its later-period results deteriorate.

### Not supported

**A. We have not proved that ceiling's 2.15x selection overweight causes B0's historical outperformance.**

Its fixed-effect residual is neutral/slightly negative.

**B. We have not proved that pivot should be promoted.**

The historical signal reverses in the contaminated validation period.

**C. We have not proved that ma10 should be penalized.**

The sample is too small.

**D. We have not proved anything about ceiling_pullback or three_weeks_tight weights.**

Their selected sample is too sparse.

**E. This analysis does not establish that B0 is superior to simpler non-B0 heuristics.**

That is a separate baseline-challenge question.

---

## 9. Production / Research Decision Table

| Signal | Evidence quality | Observed behavior | Decision |
|---|---|---|---|
| ceiling | **highest** | stable absolute outcomes; no independent FE alpha | **KEEP CURRENT HANDLING** |
| pivot | low-moderate | strong upside residual; severe temporal reversal | **CHALLENGER ONLY** |
| ma10_touch_confirm | low | weak W1 / high stop rate / late deterioration | **CAUTION ONLY** |
| ceiling_pullback | very low | mixed; all-pick W1 strong but n=4 | **INSUFFICIENT SAMPLE** |
| three_weeks_tight | unusable | n=1 selected | **INSUFFICIENT SAMPLE** |

No new production weight qualifies.

---

## 10. Final Research Interpretation

The signal-type analysis changes the interpretation of B0 in an important way.

The selector is not simply “Top3 from a generic candidate pool.” It is already structurally concentrated in one signal family:

> **ceiling**

But the evidence does not justify saying:

> ceiling is the proven alpha source.

Instead:

> **ceiling currently behaves like the stable anchor of the selected bucket, while pivot behaves like an unstable high-upside challenger.**

That is a much stronger and more precise conclusion than treating Rank1/Rank2/Rank3 as a meaningful quality hierarchy.

At the same time, the study exposes a major remaining uncertainty:

> **If simple momentum / proximity / MA / other heuristic selectors can reproduce B0's advantage, then the ceiling-heavy B0 architecture may not be uniquely necessary.**

That question is intentionally left outside this report and should be tested as a separate Simple Baseline Challenge rather than by adding more weights to B0.

---

## Final Verdict

> **KEEP PRODUCTION FROZEN.**

> **Do not add explicit signal weights from this historical sample.**

Research priority:

1. preserve ceiling as the current reference/anchor;
2. track pivot as an independent challenger in forward data;
3. treat ma10 as a caution flag, not a penalty;
4. wait for more ceiling_pullback / three_weeks_tight observations;
5. separately test whether B0 can beat simple heuristic baselines under the same candidate universe and outcome protocol.
