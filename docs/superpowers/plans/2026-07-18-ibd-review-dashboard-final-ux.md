# IBD Review Dashboard Final UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the two remaining IBD Review Dashboard UX issues with measured, real-browser evidence while preserving all business behavior.

**Architecture:** Keep Streamlit and AG Grid boundaries unchanged. Add stable keyed containers for scoped density control, extract Selected Row HTML generation into a pure function, and use semantic `details/summary` plus CSS Anchor Positioning for a hoverable, lockable, viewport-aware popup.

**Tech Stack:** Python 3.12, Streamlit 1.45.1, pandas 2.3.3, streamlit-aggrid 1.2.1.post2, pytest, Codex in-app Chromium browser.

## Global Constraints

- Do not change four-state definitions, filter logic, sort logic, calculated fields, or C Rank Reference rules.
- Preserve Tooltip, Near Trigger, Copy Codes, Manual Copy, Selected Row Detail, and AG Grid Code-copy behavior.
- Use `/Users/tbin/Documents/Yfinance_data/us/breakout_follow_pool.csv` for every runtime acceptance check.
- Do not use negative margins, transforms, hidden spacer elements, or broad page-wide widget overrides.
- Validate 1440x900 and 2048x1136 at 100% and 125% zoom with fresh reloads and current screenshots.

---

### Task 1: Reproducible dashboard runtime

**Files:**
- Use: `dashboard/requirements.txt`
- Create locally, untracked: `.venv-dashboard/`

**Interfaces:**
- Consumes: pinned packages in `dashboard/requirements.txt`
- Produces: `.venv-dashboard/bin/python` and `.venv-dashboard/bin/pytest`

- [ ] **Step 1: Create the isolated environment**

```bash
python -m venv .venv-dashboard
.venv-dashboard/bin/python -m pip install -r dashboard/requirements.txt
```

- [ ] **Step 2: Verify pinned runtime versions**

```bash
.venv-dashboard/bin/python -c "import streamlit, pandas; print(streamlit.__version__, pandas.__version__)"
```

Expected: `1.45.1 2.3.3`.

- [ ] **Step 3: Run the untouched baseline in the aligned environment**

```bash
PYTHONPATH=. .venv-dashboard/bin/pytest dashboard/tests -q
PYTHONPATH=. .venv-dashboard/bin/python dashboard/self_check.py --csv /Users/tbin/Documents/Yfinance_data/us/breakout_follow_pool.csv
```

Expected: 68 tests pass and every self-check line is `[PASS]`. If a test still fails, investigate it before production changes.

---

### Task 2: Selected Row semantic popup

**Files:**
- Modify: `dashboard/app.py`
- Test: `dashboard/tests/test_review_efficiency.py`
- Test: `dashboard/tests/test_behavioral_acceptance.py`

**Interfaces:**
- Consumes: one normalized `pandas.Series`
- Produces: `_build_selected_row_detail_html(row: pd.Series) -> str`
- Preserves: `_render_selected_row_detail(filtered_df, selected_code) -> None`

- [ ] **Step 1: Add failing markup behavior tests**

```python
def test_selected_row_popup_is_semantic_viewport_aware_and_ordered(sample_row):
    markup = _build_selected_row_detail_html(sample_row)
    assert "<details" in markup and "<summary" in markup
    assert "code-popup-toggle" not in markup
    assert "position-area:" in markup
    assert "position-try-fallbacks: flip-block" in markup
    assert "max-height: calc(100dvh - 24px)" in markup
    assert "overflow-y: auto" in markup
    assert markup.index("Daily Entry") < markup.index("Pullback") < markup.index("CANSLIM / Base")

def test_selected_row_popup_hides_empty_pullback_section(sample_row):
    sample_row["pullback_pct"] = None
    sample_row["pullback_pct_off_peak"] = None
    assert "data-popup-section=\"pullback\"" not in _build_selected_row_detail_html(sample_row)

def test_selected_row_popup_highlights_invalid_reject_reason(sample_row):
    sample_row["ibd_entry_valid"] = False
    sample_row["ibd_entry_reject_reason"] = "Low Volume"
    markup = _build_selected_row_detail_html(sample_row)
    assert "Low Volume" in markup
    assert "code-popup-reject" in markup
```

- [ ] **Step 2: Run the new tests and verify RED**

```bash
PYTHONPATH=. .venv-dashboard/bin/pytest dashboard/tests/test_review_efficiency.py dashboard/tests/test_behavioral_acceptance.py -q
```

Expected: FAIL because `_build_selected_row_detail_html` does not exist and the current checkbox popup violates the assertions.

- [ ] **Step 3: Extract HTML generation and implement the semantic trigger**

Implement `_build_selected_row_detail_html(row)` in `dashboard/app.py`. Use this structure:

```html
<details class="code-detail" data-selected-code="{code}">
  <summary class="code-hover-trigger" title="Hover, focus, or click to view details">{code} ▾</summary>
  <div class="code-hover-popup" role="region" aria-label="{code} secondary details">
    <!-- Daily Entry, optional Pullback, CANSLIM / Base -->
  </div>
</details>
```

Use `anchor-name: --selected-code`, `position-anchor: --selected-code`, `position: fixed`, `position-area: block-start span-inline-end`, and `position-try-fallbacks: flip-block`. Clamp width with `min(450px, calc(100vw - 24px))`; clamp height with `max-height: calc(100dvh - 24px); overflow-y: auto`. Show the popup for `.code-detail:hover`, `.code-detail:focus-within`, and `.code-detail[open]`. Build the Pullback section only when at least one source field formats to a value other than `n/a`.

- [ ] **Step 4: Run focused tests and verify GREEN**

```bash
PYTHONPATH=. .venv-dashboard/bin/pytest dashboard/tests/test_review_efficiency.py dashboard/tests/test_behavioral_acceptance.py -q
```

Expected: all focused tests pass.

- [ ] **Step 5: Commit the popup unit**

```bash
git add dashboard/app.py dashboard/tests/test_review_efficiency.py dashboard/tests/test_behavioral_acceptance.py
git commit -m "fix: close selected row popup interactions"
```

---

### Task 3: Keyed-container density system

**Files:**
- Modify: `dashboard/app.py`
- Test: `dashboard/tests/test_app_static.py`
- Test: `dashboard/tests/test_behavioral_acceptance.py`

**Interfaces:**
- Produces stable Streamlit keys: `dashboard_shell`, `dashboard_header`, `review_queue`, `status_cards`, `filters`, `filter_controls`, `results_toolbar`, `selected_row`, `results_grid`
- Preserves all existing widget keys and session-state names

- [ ] **Step 1: Add failing density-contract tests**

```python
def test_dashboard_uses_stable_keyed_density_containers():
    source = APP_SOURCE
    for key in ["dashboard_shell", "dashboard_header", "review_queue", "status_cards", "filters", "filter_controls", "results_toolbar", "selected_row", "results_grid"]:
        assert f'key="{key}"' in source

def test_density_css_is_scoped_and_has_no_visual_compensation_hacks():
    source = APP_SOURCE
    assert 'div[data-testid="stVerticalBlock"] > div' not in source
    assert "margin-top:28px" not in source
    assert "margin-bottom:4px" not in source
    assert "margin: -" not in source
    assert "transform:" not in source
    assert "height: 78px" in source
    assert ".st-key-status_cards" in source
```

- [ ] **Step 2: Run the new tests and verify RED**

```bash
PYTHONPATH=. .venv-dashboard/bin/pytest dashboard/tests/test_app_static.py dashboard/tests/test_behavioral_acceptance.py -q
```

Expected: FAIL on missing keys, 108px cards, broad selector, and spacer markup.

- [ ] **Step 3: Add keyed containers without changing business logic**

Wrap only existing render calls and sections. Keep widget keys and callbacks unchanged. Change status labels to exactly two visual lines:

```python
btn_label = f"{prefix}{dot} {display_name} · {count}\n{sub_map[status_name]}"
```

Place labels and controls in keyed sections instead of adding empty markdown spacers.

- [ ] **Step 4: Replace density CSS with key-scoped rules**

Use `.st-key-*` selectors and immediate stable descendants. Set status-card buttons to 78px and tool controls to 44px. Use positive `gap`, `padding`, and normal-flow alignment only. Scope the app-container top padding with `div[data-testid="stAppViewBlockContainer"]:has(.st-key-dashboard_shell)` so other Streamlit pages/widgets are unaffected.

- [ ] **Step 5: Run focused tests and verify GREEN**

```bash
PYTHONPATH=. .venv-dashboard/bin/pytest dashboard/tests/test_app_static.py dashboard/tests/test_behavioral_acceptance.py -q
```

Expected: all focused tests pass.

- [ ] **Step 6: Commit the density unit**

```bash
git add dashboard/app.py dashboard/tests/test_app_static.py dashboard/tests/test_behavioral_acceptance.py
git commit -m "fix: scope dashboard vertical density"
```

---

### Task 4: Real-browser iteration and nine-scenario acceptance

**Files:**
- Modify only if observed failures require it: `dashboard/app.py`, `dashboard/table_view.py`, relevant tests first
- Create artifacts: `dashboard/artifacts/ibd_review_ux_2026-07-18/*.png`
- Create measurements: `dashboard/artifacts/ibd_review_ux_2026-07-18/measurements.json`

**Interfaces:**
- Consumes: fresh service at `http://localhost:8501/`
- Produces: screenshots and DOM-derived geometry/interaction evidence

- [ ] **Step 1: Stop the old service and start the aligned current code**

```bash
PYTHONPATH=. .venv-dashboard/bin/python dashboard/run_app.py --csv /Users/tbin/Documents/Yfinance_data/us/breakout_follow_pool.csv --headless --server-port 8501
```

Use a fresh process after code changes and reload the browser tab after each restart.

- [ ] **Step 2: Measure the default 1440x900 and 2048x1136 layouts**

Collect `getBoundingClientRect()` for each keyed container, every status-card/control height, viewport/document widths, grid header top, and visible AG Grid row count. Assert no horizontal overflow, no wrapped controls, cards at 76–80px, at least five visible rows at 1440x900, and at least eight at 2048x1136.

- [ ] **Step 3: Exercise the requested data scenarios**

Use visible controls and AG Grid interactions for:

1. Default All Signals.
2. ACTIONABLE with Pullback data.
3. UNCONFIRMED plus Near Trigger.
4. UNCONFIRMED without Pullback data.
5. DELL or another real Reject Reason row.
6. C Rank Reference Top 20.

For each row-dependent scenario, confirm the popup code matches the selected summary code; verify Pullback presence/absence and Reject Reason emphasis from the live DOM.

- [ ] **Step 4: Exercise popup input modes**

Verify hover-in, trigger-to-popup traversal, mouse-out close, click lock/unlock, Tab focus, Enter, and Space. Confirm popup rectangle remains inside the viewport and internal overflow is used when constrained. Confirm Code cell click copies without selection change and another cell selects the row.

- [ ] **Step 5: Repeat at 100% and 125% browser zoom**

Repeat both target viewport sizes at each zoom. Save final screenshots and write the DOM measurements to `measurements.json`.

- [ ] **Step 6: For every observed failure, use a fresh RED-GREEN cycle**

Add one focused failing test, run it to confirm the expected failure, make one minimal correction, restart/reload, and repeat only the affected browser scenario. Stop and reassess the architecture after three failed fixes to the same root cause.

---

### Task 5: Full regression and final handoff

**Files:**
- Verify: all changed files and browser artifacts

**Interfaces:**
- Produces: final current-code evidence and limitation report

- [ ] **Step 1: Run full automated regression**

```bash
PATH="$PWD/.venv-dashboard/bin:$PATH" PYTHONPATH=. pytest dashboard/tests
```

Expected: all tests pass with zero failures.

- [ ] **Step 2: Run real-data self-check**

```bash
PATH="$PWD/.venv-dashboard/bin:$PATH" PYTHONPATH=. python dashboard/self_check.py --csv /Users/tbin/Documents/Yfinance_data/us/breakout_follow_pool.csv
```

Expected: every line is `[PASS]`.

- [ ] **Step 3: Verify the headless acceptance command**

```bash
PATH="$PWD/.venv-dashboard/bin:$PATH" python dashboard/run_app.py --csv /Users/tbin/Documents/Yfinance_data/us/breakout_follow_pool.csv --headless --server-port 8510
```

Expected: Streamlit reports a local URL and remains healthy until explicitly stopped.

- [ ] **Step 4: Inspect the final diff and artifacts**

```bash
git diff --check
git status --short
```

Confirm only scoped dashboard/test/design/plan/artifact files changed and no business-rule modules changed.

- [ ] **Step 5: Request focused code review and resolve findings**

Review the final diff against the design and nine acceptance scenarios. Fix every critical or important finding with a test-first cycle.

- [ ] **Step 6: Commit verified implementation and evidence**

```bash
git add dashboard/app.py dashboard/table_view.py dashboard/tests dashboard/artifacts docs/superpowers/plans/2026-07-18-ibd-review-dashboard-final-ux.md
git commit -m "fix: finish dashboard review ux acceptance"
```

- [ ] **Step 7: Report the evidence**

Link final screenshots, `measurements.json`, modified files, exact test counts, self-check output, and any remaining browser/platform limitation. Do not claim completion unless the fresh commands and all nine scenarios pass.
