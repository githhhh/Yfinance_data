# Breakout Dashboard Filter Funnel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rework the Breakout Pool dashboard filters into a trading decision funnel, keep C Rank Reference as an independent mode with rules shown, and make the result table expose all fields in logical groups.

**Architecture:** Keep the existing Streamlit app and configuration-driven field model. Move filter eligibility and table ordering into `dashboard/field_config.py`, keep dataframe normalization and derived columns in `dashboard/data_utils.py`, and keep UI composition in `dashboard/app.py`.

**Tech Stack:** Python, Streamlit, pandas, plotly, streamlit-aggrid, pytest.

## Global Constraints

- Custom filter mode must not expose `C_continuous`, `rank_C_continuous`, or `is_priority` as ordinary filters.
- C Rank Reference mode must ignore custom filters and use `signal=True` plus `rank_C_continuous asc`.
- Daily entry strength filters only make sense when `ibd_entry_valid=True`.
- Table views must be logically grouped, and `All Fields` must include every source CSV field plus `base_duration_weeks`.
- Charts remain auxiliary and should continue to read from the filtered dataframe.

---

### Task 1: Field Configuration And Derived Columns

**Files:**
- Modify: `dashboard/field_config.py`
- Modify: `dashboard/data_utils.py`
- Test: `dashboard/tests/test_table_config.py`

**Interfaces:**
- Produces: `FILTER_FUNNEL_GROUPS`, `ALL_TABLE_COLUMNS`, `get_filter_funnel_groups()`, `get_all_table_columns()`.
- Produces: normalized dataframe column `base_duration_weeks`.

- [ ] **Step 1: Write failing tests** for filterable fields, grouped table columns, and `base_duration_weeks`.
- [ ] **Step 2: Run focused tests** with `python -m pytest dashboard/tests/test_table_config.py -q`.
- [ ] **Step 3: Implement configuration and derived column support.**
- [ ] **Step 4: Re-run focused tests.**

### Task 2: Custom Filter Funnel UI

**Files:**
- Modify: `dashboard/app.py`
- Test: `dashboard/tests/test_app_static.py`

**Interfaces:**
- Consumes: `get_filter_funnel_groups()`.
- Produces: five-stage Custom Filter UI: route, entry confirmation and strength, weekly strength, structure, grouping.

- [ ] **Step 1: Write failing static tests** for the funnel order and removal of preset UI.
- [ ] **Step 2: Run focused tests** with `python -m pytest dashboard/tests/test_app_static.py -q`.
- [ ] **Step 3: Implement the funnel UI and active-chip behavior.**
- [ ] **Step 4: Re-run focused tests.**

### Task 3: C Rank Rules And Full Table Views

**Files:**
- Modify: `dashboard/app.py`
- Modify: `dashboard/table_view.py`
- Test: `dashboard/tests/test_table_config.py`
- Test: `dashboard/tests/test_app_static.py`

**Interfaces:**
- Consumes: `get_all_table_columns()`.
- Produces: visible C Rank mode rules and `All Fields` default table view.

- [ ] **Step 1: Write failing tests** for C Rank rules text and all-field table coverage.
- [ ] **Step 2: Run focused tests.**
- [ ] **Step 3: Implement the C Rank explanation block and all-field table ordering.**
- [ ] **Step 4: Re-run focused tests.**

### Task 4: Verification And Delivery

**Files:**
- Modify tests only if behavior expectations need alignment with implemented requirements.

- [ ] **Step 1: Run self-check** with `PYTHONDONTWRITEBYTECODE=1 python dashboard/self_check.py --csv us/breakout_follow_pool.csv`.
- [ ] **Step 2: Run pytest** with `PYTHONDONTWRITEBYTECODE=1 python -m pytest dashboard/tests -q -p no:cacheprovider`.
- [ ] **Step 3: Review git diff.**
- [ ] **Step 4: Commit and push.**
