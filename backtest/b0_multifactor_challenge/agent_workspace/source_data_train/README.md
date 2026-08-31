# B0 Candidate Factor Research Dataset

This is a PIT-safe weekly US-stock candidate panel. Each row is (snapshot_date, code).
The RD-Agent coding workspace contains TRAIN rows only. Use only columns in daily_pv.h5 / candidate_panel.h5 key=data.
Never use returns, stops, future/as-of outcomes, B0 membership/order, dates/tickers as fitted rules, or any hard-coded threshold learned from outcomes.
Implement an economically interpretable continuous factor. The factor program must write result.h5 with one numeric column and the same MultiIndex(datetime,instrument). Higher value should mean stronger expected future selection quality.
