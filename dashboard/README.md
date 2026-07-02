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
- Custom Filter mode no longer uses presets or a separate Sort Bar. Filters are shown as a trading decision funnel: route, entry confirmation and strength, weekly volume and price, structure, and grouping.
- Funnel tabs show active condition counts, and the current filters are summarized by funnel stage above the charts.
- Daily entry strength filters are only enabled when `ibd_entry_valid=True`.
- C Rank Reference mode ignores Custom Filter mode conditions and sorts only by `rank_C_continuous asc` after `signal=True`.
- C Rank Reference mode displays its fixed rules and formula reference above the table.
- Result Table defaults to `All Fields`, ordered by business groups, and includes the derived `base_duration_weeks` column.
- Charts are auxiliary only. They read the current filtered DataFrame and do not mutate global filters.
- Default chart dimensions are `Signal Quality Matrix`, `Structure Action Map`, and an expandable sector concentration view.
