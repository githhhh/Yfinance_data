# Documentation Index

## Current contracts

- `BREAKOUT_FOLLOW_POOL_SCHEMA.md` — BreakoutFollow Pool field/schema contract.
- `DESIGN_yfinance_data_midweek_review.md` — Midweek/complete-week data projection and comparison semantics.
- `STATIC_REVIEW_DASHBOARD_SPEC.md` — current static GitHub Pages UX, interaction, review-flow and public payload contract.

## Historical Dashboard design references

The following files document the Streamlit / AG Grid era. Their product reasoning remains useful, but their runtime/framework requirements are no longer authoritative:

- `BREAKOUT_POOL_LOCAL_DASHBOARD_DESIGN.md`
- `IBD_REVIEW_DASHBOARD_UX_UI_SPEC.md`
- `DESIGN_yfinance_data_midweek_review_ui_alignment.md`
- `2026-08-03-review-queue-flow-design.md`

When an old implementation detail conflicts with the current static Dashboard, follow `STATIC_REVIEW_DASHBOARD_SPEC.md`. Do not restore Streamlit or duplicate trading/business calculations in browser JavaScript merely to match historical implementation text.
