# RD-Agent Research Hypotheses & Factor Proposals

## Core Hypothesis 1: Risk-Adjusted Momentum (Volatility Penalty)
* **Hypothesis**: High nominal momentum in breakout candidates often carries excessive tail volatility. Adjusting 10-day, 20-day, and 60-day price momentum by 20-day realized volatility (\(\text{mom} / (\text{rv\_20} + \epsilon)\)) penalizes chaotic, high-beta spikes and rewards steady institutional accumulation.
* **Proposed Factors**:
  - `risk_adj_mom_20`: \(\text{mom\_20} / (\text{rv\_20} + 10^{-5})\)
  - `risk_adj_mom_10`: \(\text{mom\_10} / (\text{rv\_20} + 10^{-5})\)
  - `risk_adj_mom_60`: \(\text{mom\_60} / (\text{rv\_20} + 10^{-5})\)

## Core Hypothesis 2: Volume-Confirmed Momentum Surge
* **Hypothesis**: Momentum signals accompanied by strong relative volume expansion (\(\text{vol\_ratio\_5\_20} > 1\)) exhibit higher breakout sustainability than low-volume drift.
* **Proposed Factor**:
  - `vol_confirmed_mom_20`: \(\text{mom\_20} \times \text{vol\_ratio\_5\_20}\)

## Core Hypothesis 3: 52-Week High Proximity Pressure
* **Hypothesis**: Candidates closest to 52-week highs have the least overhead supply resistance, leading to stronger follow-through post-breakout.
* **Proposed Factor**:
  - `prox_52w_high`: \(-\text{dist\_to\_52w\_high\_pct}\)
