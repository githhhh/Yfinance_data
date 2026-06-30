# Breakout Pool Local Dashboard

Local Streamlit dashboard for `breakout_follow_pool.csv`.

## Run

```bash
python dashboard/run_app.py --csv us/breakout_follow_pool.csv
```

## Verify

```bash
python dashboard/self_check.py --csv us/breakout_follow_pool.csv
python -m pytest dashboard/tests -q
```

## Logic Notes

- Custom Filter mode combines every enabled core and advanced condition with AND.
- Disabled advanced filters do not participate in filtering.
- C Rank Reference mode ignores Custom Filter mode conditions and sorts only by `rank_C_continuous asc` after `signal=True`.
- Default review starts from `Review: All Signals`, so signal-quality charts can show valid and invalid rows instead of collapsing to 100% valid.
- `Action: Clean Entry` is the tighter confirmed-entry preset for a buy-zone style working list.
- Charts are auxiliary only. They read the current filtered DataFrame and do not mutate global filters.
- Default chart dimensions are `Signal Quality Matrix`, `Structure Action Map`, and an expandable sector concentration view.
