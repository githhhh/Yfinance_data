# IBD Review Dashboard Final UX Design

## Scope

Close the remaining vertical-density and Selected Row secondary-detail issues in `dashboard/` without changing four-state semantics, filters, ordering, calculated fields, or C Rank Reference rules. Existing Tooltip, Near Trigger, Copy Codes, Manual Copy, Selected Row Detail, and AG Grid Code-copy behavior remain intact.

## Runtime baseline

The real acceptance dataset is `/Users/tbin/Documents/Yfinance_data/us/breakout_follow_pool.csv`. The repository pins Streamlit 1.45.1 and streamlit-aggrid 1.2.1.post2; acceptance runs in a workspace-local environment matching `dashboard/requirements.txt`, so the system Python installation is not mutated.

## Density design

Wrap the header, review queue, status cards, filters, filter controls, results toolbar, selected-row card, and grid in keyed Streamlit containers. Style only the generated `st-key-*` classes and their immediate stable descendants. Remove broad vertical-block overrides and visual spacer markup.

Status cards remain equal-height two-line review cards at 78px. All Signals, Copy Codes, Manual, Reset, and filter inputs use one 44px control rhythm. Section spacing is controlled by keyed-container padding/gaps, never negative margins, transforms, hidden placeholders, or page-wide widget overrides.

The grid keeps its current columns, 480px body height, selection behavior, and Code cell renderer. The first-screen targets are validated from actual DOM rectangles and visible grid rows at 1440x900 and 2048x1136, at browser zoom 100% and 125%.

## Selected Row detail design

The compact Selected Row summary remains above AG Grid. Its Code trigger becomes semantic `details/summary`, which provides click lock/unlock and native Tab plus Enter/Space behavior. Hover opens the same popup; moving from trigger into popup keeps it open, and leaving closes a non-locked popup.

The popup uses CSS Anchor Positioning with a block-axis fallback, allowing the browser to place it above or below according to available viewport space. Width is viewport-clamped, height is capped to the viewport, and only popup content scrolls. The popup DOM is regenerated from the current selected row on every Streamlit rerun, preventing stale stock content after row, filter, or mode changes.

Field order is:

1. Daily Entry: Trigger, Entry Date, Daily Entry Vol; invalid entries prominently show Reject Reason.
2. Pullback: Pullback Depth and Off Pullback Peak; the whole section is omitted when both are empty.
3. CANSLIM / Base: EPS YoY, To 52W High, 52W High, Ceiling/Base Depth, Base Duration.

The AG Grid Code cell renderer is unchanged: Code click copies only, while other cells retain single-row selection.

## Testing and acceptance

Tests are added before production changes for scoped keyed-container markup/CSS contracts, popup field ordering, conditional Pullback omission, Reject Reason emphasis, semantic trigger behavior, viewport flipping/scroll constraints, and existing Code-copy selection rules.

After each focused change, reload the current service and inspect fresh DOM state. Exercise all nine requested scenarios on the real CSV, record viewport/control/grid measurements, and save current screenshots. Finish with:

- `PYTHONPATH=. pytest dashboard/tests`
- `PYTHONPATH=. python dashboard/self_check.py --csv /Users/tbin/Documents/Yfinance_data/us/breakout_follow_pool.csv`
- `python dashboard/run_app.py --csv /Users/tbin/Documents/Yfinance_data/us/breakout_follow_pool.csv --headless`

No unrelated refactoring or business-rule changes are included.
