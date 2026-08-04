from __future__ import annotations


REVIEW_UI_CSS = r"""
:root {
    --bg: #0c1016;
    --panel: #151b23;
    --panel-soft: #111720;
    --input: #202b3a;
    --line: #35404d;
    --line-soft: #2b3440;
    --text: #f4f5f7;
    --muted: #9ca8b7;
    --subtle: #6f7a88;
    --green: #35df65;
    --cyan: #1fcdb4;
    --blue: #2791ff;
    --yellow: #ffd21f;
    --red: #f04444;
}

div[data-testid="stApp"]:has(.st-key-dashboard_shell) {
    background: var(--bg);
    color: var(--text);
    font-family: Arial Narrow, Roboto Condensed, Inter, ui-sans-serif,
        -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif;
    font-size: 14px;
    font-weight: 500;
}

div[data-testid="stApp"]:has(.st-key-dashboard_shell) [data-testid="stHeader"],
div[data-testid="stApp"]:has(.st-key-dashboard_shell) [data-testid="stSidebar"] {
    display: none !important;
}

div[data-testid="stMainBlockContainer"]:has(.st-key-dashboard_shell) {
    max-width: none !important;
    padding: 8px 24px 30px !important;
}

div[data-testid="stMainBlockContainer"]:has(.st-key-dashboard_shell)
    > div[data-testid="stVerticalBlockBorderWrapper"]
    > div[data-testid="stVerticalBlock"] {
    gap: 0 !important;
}

.st-key-dashboard_shell {
    gap: 0 !important;
}

.st-key-review_queue {
    gap: 0 !important;
}

.st-key-filters {
    gap: 0 !important;
}

.st-key-results_toolbar {
    gap: 0 !important;
}

.st-key-c_rank_results_toolbar {
    gap: 0 !important;
}

.st-key-selected_row {
    gap: 0 !important;
}

.st-key-ibd_selected_row {
    gap: 0 !important;
}

.st-key-results_grid {
    gap: 0 !important;
}

.st-key-dashboard_header > div[data-testid="stVerticalBlock"],
.st-key-review_queue > div[data-testid="stVerticalBlock"],
.st-key-filters > div[data-testid="stVerticalBlock"],
.st-key-results_toolbar > div[data-testid="stVerticalBlock"],
.st-key-c_rank_results_toolbar > div[data-testid="stVerticalBlock"],
.st-key-selected_row > div[data-testid="stVerticalBlock"],
.st-key-ibd_selected_row > div[data-testid="stVerticalBlock"],
.st-key-results_grid > div[data-testid="stVerticalBlock"] {
    gap: 0 !important;
}

.st-key-dashboard_header {
    min-height: 64px;
    border-bottom: 1px solid #aab1bb;
}

.st-key-dashboard_header > div[data-testid="stHorizontalBlock"] {
    align-items: flex-start;
    gap: 24px !important;
}

.dashboard-title {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 0 !important;
    margin: 0 !important;
    color: var(--text);
    font-family: Arial Narrow, Roboto Condensed, Inter, ui-sans-serif,
        -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif;
    font-size: 29px !important;
    font-weight: 800 !important;
    line-height: 1;
    letter-spacing: -1.1px;
}

.data-badge {
    display: inline-flex;
    align-items: center;
    height: 25px;
    border-radius: 4px;
    padding: 0 8px;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0;
}

.data-badge--fresh {
    background: #e8f7eb;
    color: #27783b;
}

.data-badge--aging {
    background: #fff4df;
    color: #a85b00;
}

.data-badge--stale {
    background: #ffebee;
    color: #c62828;
}

.data-badge--loaded {
    background: #e8eef6;
    color: #50647b;
}

.data-badge--error {
    background: #ffebee;
    color: #c62828;
}

.dashboard-snapshot {
    display: flex;
    align-items: center;
    gap: 5px;
    margin-top: 12px;
    color: #8d9bab;
    font-size: 12px;
    white-space: nowrap;
}

.dashboard-snapshot b {
    color: #aeb9c5;
}

.dashboard-snapshot .snapshot-mode-segment {
    color: #8d9bab;
}

.dashboard-snapshot .snapshot-mode-segment--midweek {
    color: #40d6ba;
    font-weight: 700;
}

.dashboard-snapshot .snapshot-freshness {
    font-weight: 700;
}

.dashboard-snapshot .snapshot-freshness--fresh {
    color: #35df65;
}

.dashboard-snapshot .snapshot-freshness--aging {
    color: #f5a623;
}

.dashboard-snapshot .snapshot-freshness--stale {
    color: #f04444;
}

.dashboard-snapshot .snapshot-freshness--unknown {
    color: #9ca8b7;
}

.st-key-dashboard_header div[class*="st-key-btn_info_rules"] button {
    width: 45px !important;
    min-width: 45px !important;
    max-width: 45px !important;
    height: 45px !important;
    min-height: 45px !important;
    max-height: 45px !important;
    border-radius: 11px !important;
    border: 1px solid #414b58 !important;
    background: transparent !important;
    color: var(--text) !important;
}

.st-key-dashboard_header div[data-testid="stColumn"]:has(.st-key-btn_info_rules)
    > div[data-testid="stVerticalBlockBorderWrapper"]
    > div[data-testid="stVerticalBlock"]
    > div[data-testid="stHorizontalBlock"] {
    display: grid !important;
    grid-template-columns: 45px 235px;
    justify-content: end;
    gap: 8px !important;
}

.st-key-dashboard_header div[class*="st-key-global_mode_selector"] div[role="radiogroup"],
.st-key-dashboard_header div[class*="st-key-global_mode_selector"] div[data-testid="stButtonGroup"],
.st-key-dashboard_header div[class*="st-key-global_mode_selector"] button,
.st-key-dashboard_header div[class*="st-key-global_mode_selector"] label {
    height: 45px !important;
    min-height: 45px !important;
    max-height: 45px !important;
}

.st-key-dashboard_header div[class*="st-key-global_mode_selector"] div[role="radiogroup"],
.st-key-dashboard_header div[class*="st-key-global_mode_selector"] div[data-testid="stButtonGroup"] {
    display: grid !important;
    grid-template-columns: 99px 136px;
    width: 235px !important;
    gap: 0 !important;
}

.st-key-dashboard_header div[class*="st-key-global_mode_selector"] button,
.st-key-dashboard_header div[class*="st-key-global_mode_selector"] label {
    min-width: 99px;
    border-color: #414b58 !important;
    border-radius: 0 !important;
    padding: 0 10px !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    white-space: nowrap !important;
}

.st-key-dashboard_header div[class*="st-key-global_mode_selector"] button:first-child,
.st-key-dashboard_header div[class*="st-key-global_mode_selector"] label:first-child {
    width: 99px !important;
    max-width: 99px !important;
    border-radius: 9px 0 0 9px !important;
}

.st-key-dashboard_header div[class*="st-key-global_mode_selector"] button:last-child,
.st-key-dashboard_header div[class*="st-key-global_mode_selector"] label:last-child {
    width: 136px !important;
    min-width: 136px;
    max-width: 136px !important;
    border-radius: 0 9px 9px 0 !important;
}

.st-key-dashboard_header div[class*="st-key-global_mode_selector"] button[aria-pressed="true"],
.st-key-dashboard_header div[class*="st-key-global_mode_selector"] label:has(input:checked) {
    border-color: #00d897 !important;
    background: rgb(0 216 151 / 12%) !important;
    color: #29f2b0 !important;
    box-shadow: inset 0 0 0 1px #00d897 !important;
}

.snapshot-segment {
    display: inline-block;
    min-width: 139px;
}

.snapshot-mode-segment {
    display: inline-block;
    min-width: 211px;
}

.st-key-review_queue {
    padding-top: 8px;
}

.st-key-review_queue_heading {
    min-height: 50px;
}

.st-key-review_queue_heading > div[data-testid="stHorizontalBlock"] {
    display: grid !important;
    grid-template-columns: minmax(0, 1fr) 276px 268px;
    gap: 9px !important;
    align-items: center;
}

.st-key-review_queue_heading > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
    width: 100% !important;
    min-width: 0 !important;
    flex: none !important;
}

.st-key-review_mode_controls,
.st-key-review_scope_controls {
    width: 100% !important;
    min-width: 0 !important;
    max-width: none !important;
    height: 31px;
    border: 1px solid #3d4855;
    border-radius: 8px;
    padding: 2px;
    overflow: hidden;
    background: var(--panel-soft);
    box-sizing: border-box;
}

.st-key-review_scope_controls {
    border-color: #3e72b7;
}

.st-key-review_mode_controls > div[data-testid="stVerticalBlock"],
.st-key-review_scope_controls > div[data-testid="stVerticalBlock"] {
    height: 100%;
    gap: 0 !important;
}

.st-key-review_mode_controls div[data-testid="stHorizontalBlock"],
.st-key-review_scope_controls div[data-testid="stHorizontalBlock"] {
    height: 100%;
    gap: 0 !important;
}

.st-key-review_mode_controls div[data-testid="stColumn"],
.st-key-review_scope_controls div[data-testid="stColumn"] {
    min-width: 0 !important;
    padding: 0 !important;
}

.st-key-review_mode_controls button,
.st-key-review_scope_controls button {
    width: 100% !important;
    height: 25px !important;
    min-height: 25px !important;
    border: 0 !important;
    border-radius: 5px !important;
    padding: 0 8px !important;
    white-space: nowrap !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    color: #b7c1cc !important;
    background: transparent !important;
    box-shadow: none !important;
    font-variant-numeric: tabular-nums;
}

.st-key-review_mode_controls button[kind="primary"] {
    color: #f7f8f9 !important;
    background: #243349 !important;
    box-shadow: inset 0 0 0 1px rgb(71 143 255 / 38%) !important;
}

.st-key-review_mode_controls button:disabled {
    cursor: not-allowed !important;
    color: #536170 !important;
    background: #10161e !important;
}

.st-key-review_scope_controls button[kind="primary"] {
    color: #f4f7fa !important;
    background: #1c2f48 !important;
    box-shadow: inset 0 -2px #3e91ff !important;
}

.st-key-review_scope_controls button:disabled {
    cursor: not-allowed !important;
    color: #536170 !important;
    background: #10161e !important;
}

.st-key-review_period_group p,
.st-key-review_scope_group p {
    margin: 0 0 3px !important;
    color: #929eac !important;
    font-size: 10px !important;
    font-weight: 800 !important;
    letter-spacing: 0.05em;
}

.weekend-scope-static {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 25px;
    color: #cbd5e1;
    font-size: 13px;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
}

.st-key-review_context_slot {
    height: auto;
    min-height: 54px !important;
    margin: 4px 0 5px;
    overflow: visible !important;
}

.st-key-review_context_slot:has(.st-key-quick_context_row) {
    border: 1px solid #303a46;
    border-radius: 7px;
    padding: 3px 10px;
    background: #11171e;
    box-sizing: border-box;
}

.st-key-review_context_slot > div[data-testid="stVerticalBlock"] {
    min-height: 48px !important;
    gap: 0 !important;
}

.st-key-quick_context_row > div[data-testid="stVerticalBlock"] {
    min-height: 48px;
    gap: 0 !important;
}

.st-key-quick_context_row > div[data-testid="stHorizontalBlock"] {
    display: grid !important;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) 74px;
    min-width: 0;
    min-height: 48px;
    align-items: center;
    gap: 10px !important;
}

.st-key-quick_context_row > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
    width: 100% !important;
    min-width: 0 !important;
    flex: none !important;
}

.st-key-quick_change_group > div[data-testid="stVerticalBlock"],
.st-key-quick_origin_group > div[data-testid="stVerticalBlock"] {
    display: grid !important;
    grid-template-columns: 82px minmax(0, 1fr);
    align-items: center;
    gap: 7px !important;
}

.st-key-quick_change_group p,
.st-key-quick_origin_group p {
    margin: 0 !important;
    color: #b5c0cc !important;
    font-size: 10px !important;
    font-weight: 900 !important;
    letter-spacing: 0.04em;
    white-space: nowrap;
}

.st-key-quick_change_group div[data-testid="stHorizontalBlock"],
.st-key-quick_origin_group div[data-testid="stHorizontalBlock"] {
    display: grid !important;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 6px !important;
}

.st-key-quick_change_group div[data-testid="stColumn"],
.st-key-quick_origin_group div[data-testid="stColumn"] {
    width: 100% !important;
    min-width: 0 !important;
    flex: none !important;
}

.st-key-review_context_slot div[class*="st-key-flow_card_"] {
    position: relative;
    min-width: 0;
}

.st-key-review_context_slot div[class*="st-key-flow_card_"]
    > div[data-testid="stElementContainer"]:not(:has(.flow-info-trigger))
    button[kind] {
    width: 100% !important;
    height: 34px !important;
    min-height: 34px !important;
    border: 1px solid #35414d !important;
    border-radius: 6px !important;
    padding: 0 28px 0 9px !important;
    background: #151c24 !important;
    color: #cbd4de !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    white-space: nowrap !important;
    font-variant-numeric: tabular-nums;
}

.st-key-review_context_slot div[class*="st-key-flow_card_"]
    > div[data-testid="stElementContainer"]:not(:has(.flow-info-trigger))
    button[kind="primary"] {
    border-color: #2d8f7f !important;
    background: #16302e !important;
    color: #e7f8f4 !important;
}

.st-key-review_context_slot div[class*="st-key-flow_card_"]
    > div[data-testid="stElementContainer"]:not(:has(.flow-info-trigger))
    button[kind]:hover {
    border-color: #536273 !important;
    background: #19222c !important;
    color: #f5f8fb !important;
}

.st-key-review_context_slot div[class*="st-key-flow_card_"]
    > div[data-testid="stElementContainer"]:not(:has(.flow-info-trigger))
    button[kind] p {
    display: grid !important;
    grid-template-columns: 16px minmax(0, 1fr) 30px;
    align-items: center;
    column-gap: 4px;
    width: 100%;
    min-width: 0;
    margin: 0 !important;
    overflow: visible;
    text-align: left;
}

.st-key-review_context_slot div[class*="st-key-flow_card_"]
    > div[data-testid="stElementContainer"]:not(:has(.flow-info-trigger))
    button[kind] p strong:first-child {
    grid-column: 1;
    font-size: 14px;
    line-height: 1;
    text-align: center;
}

.st-key-review_context_slot div[class*="st-key-flow_card_"]
    > div[data-testid="stElementContainer"]:not(:has(.flow-info-trigger))
    button[kind] p strong:last-child {
    grid-column: 3;
    width: 30px;
    color: #f3f6f9;
    font-weight: 900 !important;
    font-variant-numeric: tabular-nums;
    text-align: right;
}

.st-key-flow_card_became_actionable > div[data-testid="stElementContainer"]:not(:has(.flow-info-trigger)) button[kind] p strong:first-child {
    color: var(--green) !important;
}

.st-key-flow_card_left_actionable > div[data-testid="stElementContainer"]:not(:has(.flow-info-trigger)) button[kind] p strong:first-child {
    color: #ff6f61 !important;
}

.st-key-flow_card_other_changes > div[data-testid="stElementContainer"]:not(:has(.flow-info-trigger)) button[kind] p strong:first-child {
    color: #f59e0b !important;
}

.st-key-flow_card_new > div[data-testid="stElementContainer"]:not(:has(.flow-info-trigger)) button[kind] p strong:first-child {
    color: #22d3ee !important;
}

.st-key-flow_card_carry > div[data-testid="stElementContainer"]:not(:has(.flow-info-trigger)) button[kind] p strong:first-child {
    color: #94a3b8 !important;
}

.st-key-flow_card_reconfirmed > div[data-testid="stElementContainer"]:not(:has(.flow-info-trigger)) button[kind] p strong:first-child {
    color: #a78bfa !important;
}

.st-key-quick_clear_slot {
    min-width: 74px;
}

.st-key-quick_clear_slot button {
    height: 34px !important;
    min-height: 34px !important;
    border: 0 !important;
    background: transparent !important;
    color: var(--muted) !important;
    font-size: 10px !important;
}

.quick-clear-placeholder {
    display: block;
    width: 74px;
    height: 34px;
}

.weekend-context-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    height: 48px;
    border: 1px solid #303a46;
    border-radius: 7px;
    padding: 0 13px;
    background: #11171e;
    color: #74808d;
    font-size: 11px;
    box-sizing: border-box;
}

.weekend-context-bar strong {
    color: #c8d0d9;
    font-size: 12px;
    text-transform: uppercase;
}

.weekend-context-bar span:first-of-type {
    color: #8f9ba8;
}

.st-key-status_cards > div[data-testid="stHorizontalBlock"] {
    display: grid !important;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 16px !important;
}

.st-key-status_cards > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
    width: 100% !important;
    min-width: 0 !important;
    flex: none !important;
}

.st-key-status_cards div[class*="st-key-flow_card_"]
    > div[data-testid="stElementContainer"]:not(:has(.flow-info-trigger))
    button[kind] {
    width: 100% !important;
    height: 70px;
    min-height: 70px !important;
    max-height: 70px !important;
    justify-content: center !important;
    border: 1px solid #36414e !important;
    border-radius: 8px !important;
    padding: 0 32px 0 28px !important;
    background: var(--panel) !important;
    color: #d6dce3 !important;
    white-space: pre-line !important;
    box-shadow: none !important;
}

.st-key-status_cards div[class*="st-key-flow_card_"]
    > div[data-testid="stElementContainer"]:not(:has(.flow-info-trigger))
    button[kind="primary"] {
    border-color: #527060 !important;
    background: #19251f !important;
}

.st-key-status_cards div[class*="st-key-flow_card_"]
    > div[data-testid="stElementContainer"]:not(:has(.flow-info-trigger))
    button[kind]:hover {
    border-color: #536172 !important;
    background: #19212a !important;
}

.st-key-status_cards div[class*="st-key-flow_card_"]
    > div[data-testid="stElementContainer"]:not(:has(.flow-info-trigger))
    button[kind] p {
    display: grid !important;
    grid-template-columns: 19px minmax(0, 1fr);
    align-items: center;
    column-gap: 10px;
    margin: 0 !important;
    color: #d6dce3 !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    line-height: 1.35 !important;
    text-align: left;
    white-space: pre-line !important;
    font-variant-numeric: tabular-nums;
}

.st-key-status_cards div[class*="st-key-flow_card_"] {
    position: relative;
}

.st-key-status_cards div[class*="st-key-flow_card_"]
    > div[data-testid="stElementContainer"]:not(:has(.flow-info-trigger))
    button[kind] p::before {
    content: "";
    display: inline-block;
    width: 19px;
    height: 19px;
    grid-column: 1;
    grid-row: 1;
    margin-right: 0;
    border-radius: 50%;
    box-shadow: inset 0 2px 3px rgba(255, 255, 255, 0.45), 0 0 5px currentColor;
    vertical-align: middle;
}

.st-key-flow_card_actionable > div[data-testid="stElementContainer"]:not(:has(.flow-info-trigger)) button[kind] p::before {
    color: var(--green);
    background: #16c832;
}

.st-key-flow_card_unconfirmed > div[data-testid="stElementContainer"]:not(:has(.flow-info-trigger)) button[kind] p::before {
    color: var(--yellow);
    background: #ffc800;
}

.st-key-flow_card_below_trigger > div[data-testid="stElementContainer"]:not(:has(.flow-info-trigger)) button[kind] p::before {
    color: var(--red);
    background: #e71920;
}

.st-key-flow_card_extended > div[data-testid="stElementContainer"]:not(:has(.flow-info-trigger)) button[kind] p::before {
    color: var(--blue);
    background: #0968d5;
}

div[class*="st-key-flow_card_"] > div[data-testid="stElementContainer"]:has(.flow-info-trigger) {
    position: absolute;
    top: 7px;
    right: 7px;
    z-index: 4;
    width: 16px !important;
    height: 16px !important;
}

.flow-info-trigger {
    appearance: none;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 16px !important;
    min-width: 16px !important;
    max-width: 16px !important;
    height: 16px !important;
    min-height: 16px !important;
    max-height: 16px !important;
    border: 1px solid #4c5967 !important;
    border-radius: 50% !important;
    padding: 0 !important;
    background: #121921 !important;
    color: #82909f !important;
    cursor: pointer;
    font-family: inherit;
    font-size: 10px !important;
    line-height: 1;
}

.flow-info-trigger:hover,
.flow-info-trigger[aria-expanded="true"] {
    border-color: #6a7a8b !important;
    background: #1a242e !important;
    color: #d4dde6 !important;
}

.flow-tooltip-surface {
    position: fixed;
    z-index: 1000000;
    max-width: min(320px, calc(100vw - 24px));
    box-sizing: border-box;
    border: 1px solid #455261;
    border-radius: 7px;
    padding: 10px 12px;
    background: #202831;
    color: #f2f5f8;
    box-shadow: 0 8px 24px rgb(0 0 0 / 38%);
    font-family: Arial Narrow, Roboto Condensed, Inter, ui-sans-serif,
        -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif;
    font-size: 12px;
    line-height: 1.55;
    white-space: pre-line !important;
    pointer-events: none;
}

.flow-tooltip-surface[hidden] {
    display: none !important;
}

.st-key-filters_header {
    min-height: 45px;
    border: 1px solid #313b47;
    border-radius: 7px;
    background: #11161d;
    box-sizing: border-box;
}

.st-key-filters_header > div[data-testid="stHorizontalBlock"] {
    display: grid !important;
    grid-template-columns: max-content minmax(0, 1fr);
    min-height: 43px;
    align-items: center;
    gap: 12px !important;
    padding: 0 10px 0 4px;
}

.st-key-filters_header > div[data-testid="stHorizontalBlock"]:has(.st-key-btn_filters_reset) {
    grid-template-columns: max-content 80px minmax(0, 1fr);
}

.st-key-filters_header > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
    width: 100% !important;
    min-width: 0 !important;
    flex: none !important;
}

.st-key-filters_header div[class*="st-key-btn_filters_toggle"] button {
    position: relative;
    width: auto !important;
    height: 43px;
    min-height: 43px !important;
    max-height: 43px !important;
    justify-content: flex-start !important;
    border: 0 !important;
    border-radius: 5px !important;
    padding: 0 28px 0 10px !important;
    background: transparent !important;
    color: #dfe4e9 !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    text-align: left !important;
}

.st-key-filters_header div[class*="st-key-btn_filters_toggle"] button::after {
    content: "⌄";
    position: absolute;
    top: 0;
    right: 8px;
    bottom: 0;
    display: flex;
    align-items: center;
    color: #91a0af;
    font-size: 14px;
}

.st-key-filters_header:has(.filters-state-marker[data-expanded="true"])
    div[class*="st-key-btn_filters_toggle"] button::after {
    content: "⌃";
}

.filters-state-marker {
    display: none;
}

.st-key-filters_header > div[data-testid="stVerticalBlock"]
    > div[data-testid="stElementContainer"]:has(.filters-state-marker) {
    display: none;
}

.st-key-filters_header div[class*="st-key-btn_filters_toggle"] button:hover {
    background: #141b23 !important;
}

.st-key-filters_header div[class*="st-key-btn_filters_reset"] button {
    width: 80px !important;
    height: 31px !important;
    min-height: 31px !important;
    border: 1px solid #465365 !important;
    border-radius: 7px !important;
    padding: 0 12px !important;
    background: #151b23 !important;
    color: #c7d0db !important;
    font-size: 12px !important;
    font-weight: 650 !important;
    white-space: nowrap !important;
}

.st-key-filters_header div[class*="st-key-btn_filters_reset"] button:hover {
    border-color: #5b6a7d !important;
    background: #1a222c !important;
    color: #edf2f7 !important;
}

.st-key-filters {
    padding-top: 6px;
}

.st-key-filter_controls input,
.st-key-filter_controls div[data-baseweb="select"] > div {
    border-color: #3a4654 !important;
    background: var(--input) !important;
    color: var(--text) !important;
}

.st-key-filter_controls {
    padding: 12px 12px 16px;
    border: 1px solid #303a46;
    border-top: 0;
    border-radius: 0 0 7px 7px;
    background: #11171e;
}

.st-key-filter_controls > div[data-testid="stHorizontalBlock"] {
    display: grid !important;
    grid-template-columns: minmax(150px, 1.25fr) minmax(250px, 2fr) minmax(330px, 2.3fr);
    align-items: start;
    gap: 24px !important;
}

.st-key-filter_controls > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
    width: 100% !important;
    min-width: 0 !important;
    flex: none !important;
}

.st-key-filter_controls [data-testid="stCaptionContainer"] p {
    color: #93a1b0 !important;
    font-size: 10px !important;
    font-weight: 800 !important;
    letter-spacing: 0.05em;
}

.st-key-filter_controls div[data-testid="stPopover"] > button,
.st-key-filter_controls div[data-testid="stPopover"] button[kind] {
    min-height: 38px !important;
    border: 1px solid #3a4654 !important;
    border-radius: 6px !important;
    background: var(--input) !important;
    color: var(--text) !important;
    font-size: 13px !important;
    font-weight: 650 !important;
}

.st-key-filter_controls > div[data-testid="stHorizontalBlock"]
    > div[data-testid="stColumn"]:first-child div[data-testid="stPopover"] {
    margin-top: 14px;
}

.filter-slider-heading {
    height: 16px;
    color: #d4dce5;
    font-size: 11px;
    font-weight: 650;
    line-height: 16px;
    white-space: nowrap;
}

.filter-volume-inactive-marker {
    display: none;
}

.st-key-filter_controls div[data-testid="stSlider"] p {
    color: #c9d3de !important;
    font-size: 11px !important;
    white-space: nowrap;
}

.st-key-filter_controls div[data-testid="stSlider"] {
    min-width: 0;
}

.st-key-filter_controls div[data-testid="stSlider"] [data-testid="stSliderTickBar"] {
    display: none !important;
}

.st-key-filter_entry_volume:has(.filter-volume-inactive-marker)
    div[data-testid="stSlider"] [data-testid="stSliderThumbValue"],
.st-key-filter_weekly_volume:has(.filter-volume-inactive-marker)
    div[data-testid="stSlider"] [data-testid="stSliderThumbValue"] {
    display: none !important;
}

.st-key-filter_controls div[data-testid="stSlider"] [data-testid="stSliderThumbValue"] {
    color: #21d99a !important;
    font-weight: 750 !important;
}

.st-key-results_toolbar {
    min-height: 44px;
    padding-top: 4px;
    box-sizing: border-box;
}

.st-key-results_toolbar > div[data-testid="stHorizontalBlock"] {
    display: grid !important;
    grid-template-columns: max-content 154px minmax(0, 1fr);
    align-items: center;
    gap: 12px !important;
}

.st-key-results_toolbar > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
    width: 100% !important;
    min-width: 0 !important;
    flex: none !important;
}

.st-key-c_rank_results_toolbar {
    min-height: 80px;
    padding-top: 12px;
    padding-bottom: 12px;
    box-sizing: border-box;
}

.st-key-results_toolbar .results-summary,
.st-key-c_rank_results_toolbar .results-summary {
    color: #c5ceda;
    font-size: 14px;
    font-weight: 600;
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
}

.st-key-results_toolbar div[data-testid="stMarkdownContainer"]:has(.results-summary) {
    margin-bottom: 0 !important;
}

.st-key-c_rank_results_toolbar div[data-testid="stMarkdownContainer"]:has(.results-summary) {
    margin-bottom: 0 !important;
}

.st-key-results_actions {
    width: 100%;
    display: flex;
    justify-content: flex-start;
}

.st-key-results_toolbar iframe {
    width: 154px !important;
    min-width: 154px;
    max-width: 154px;
    height: 36px !important;
}

.st-key-c_rank_results_toolbar iframe {
    min-width: 144px;
    max-width: 160px;
    height: 44px !important;
}

.st-key-c_rank_results_toolbar div[data-baseweb="select"] > div {
    min-width: 170px;
    max-width: 190px;
    height: 44px !important;
}

.st-key-selected_row .selected-strip {
    display: grid;
    grid-template-columns: 194px repeat(4, minmax(0, 1fr));
    align-items: stretch;
    width: 100%;
    height: auto;
    min-height: 72px;
    box-sizing: border-box;
    overflow: visible;
    margin-bottom: 8px;
    border: 1px solid var(--line);
    border-radius: 7px;
    background: var(--panel);
    color: #f2f5f9;
}

.st-key-selected_row div[data-testid="stMarkdownContainer"]:has(.selected-strip) {
    margin-bottom: 0 !important;
}

.st-key-selected_row .selected-summary-cell {
    display: flex;
    min-width: 0;
    flex-direction: column;
    justify-content: center;
    padding: 9px 12px;
    border-right: 1px solid #36404b;
    text-align: center;
}

.st-key-selected_row .selected-summary-cell:last-child {
    border-right: 0;
}

.st-key-selected_row .selected-code-cell {
    text-align: left;
}

.st-key-selected_row .selected-strip--empty {
    display: flex;
    align-items: center;
    justify-content: center;
    color: #9ca8b7;
    font-size: 13px;
}

.st-key-ibd_selected_row .ibd-selected-strip {
    display: grid;
    grid-template-columns: 104px minmax(180px, 1.05fr) minmax(190px, 1.15fr) minmax(255px, 1.5fr) minmax(175px, 1.05fr);
    align-items: stretch;
    width: 100%;
    height: 64px;
    min-height: 64px;
    max-height: 64px;
    box-sizing: border-box;
    overflow: visible;
    margin: 0 0 4px;
    border: 1px solid var(--line);
    border-radius: 7px;
    background: var(--panel);
    color: #f2f5f9;
}

.st-key-ibd_selected_row div[data-testid="stMarkdownContainer"]:has(.ibd-selected-strip) {
    margin: 0 !important;
}

.st-key-ibd_selected_row .selected-summary-cell {
    display: flex;
    min-width: 0;
    flex-direction: column;
    justify-content: center;
    padding: 7px 10px;
    border-right: 1px solid #36404b;
    text-align: center;
    white-space: nowrap;
}

.st-key-ibd_selected_row .selected-summary-cell:last-child {
    border-right: 0;
}

.st-key-ibd_selected_row .selected-code-cell {
    text-align: left;
}

.st-key-ibd_selected_row .code-detail {
    display: inline-block;
    width: max-content;
}

.st-key-ibd_selected_row .code-hover-trigger {
    anchor-name: --ibd-selected-code;
    display: inline-block;
    color: #56a8ff;
    font-size: 18px;
    font-weight: 850;
    line-height: 1;
    white-space: nowrap;
    cursor: help;
    border-bottom: 1px dotted #56a8ff;
    list-style: none;
}

.st-key-ibd_selected_row .code-hover-trigger:hover,
.st-key-ibd_selected_row .code-hover-trigger:focus-visible {
    color: #80bdff;
    border-bottom-style: solid;
}

.st-key-ibd_selected_row .code-hover-trigger::-webkit-details-marker {
    display: none;
}

.st-key-ibd_selected_row .code-hover-trigger::marker {
    content: "";
}

.st-key-ibd_selected_row .code-hover-popup {
    display: none;
    position: fixed;
    position-anchor: --ibd-selected-code;
    position-area: block-start span-inline-end;
    position-try-fallbacks: flip-block;
    width: min(450px, calc(100vw - 24px));
    max-height: min(360px, calc(50dvh - 12px));
    overflow-y: auto;
    overscroll-behavior: contain;
    box-sizing: border-box;
    padding: 8px 0;
    z-index: 999999;
    text-align: left;
    white-space: normal;
}

.st-key-ibd_selected_row .code-detail:hover > .code-hover-popup,
.st-key-ibd_selected_row .code-hover-popup:hover,
.st-key-ibd_selected_row .code-detail:not([data-escape-dismissed="true"]):has(> .code-hover-trigger:focus-visible)
    > .code-hover-popup,
.st-key-ibd_selected_row .code-detail[open] > .code-hover-popup {
    display: block;
}

.st-key-ibd_selected_row .code-hover-surface {
    border: 1px solid #4a5a6a;
    border-radius: 8px;
    padding: 12px 14px;
    background: #1e2631;
    box-shadow: 0 8px 24px rgb(0 0 0 / 60%);
}

.st-key-ibd_selected_row .code-popup-section {
    margin-bottom: 10px;
    border-bottom: 1px solid #303947;
    padding-bottom: 8px;
}

.st-key-ibd_selected_row .code-popup-section:last-child {
    margin-bottom: 0;
    border-bottom: 0;
    padding-bottom: 0;
}

.st-key-ibd_selected_row .code-popup-title {
    margin-bottom: 6px;
    color: #aab7c5;
    font-size: 11px;
    font-weight: 750;
}

.st-key-ibd_selected_row .code-popup-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 6px 10px;
}

.st-key-ibd_selected_row .code-popup-grid-2 {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 6px 10px;
}

.st-key-ibd_selected_row .code-popup-item {
    color: #a0aec0;
    font-size: 11px;
}

.st-key-ibd_selected_row .code-popup-val {
    color: #f2f5f9;
    font-size: 13px;
    font-weight: 700;
}

.st-key-ibd_selected_row .code-popup-reject {
    margin-top: 8px;
    border: 1px solid #ffb300;
    border-radius: 6px;
    padding: 8px 10px;
    background: rgb(255 179 0 / 12%);
}

.st-key-ibd_selected_row .code-popup-reject .code-popup-val {
    color: #ffca54;
    font-size: 12px;
}

.st-key-ibd_selected_row .selected-label {
    margin-bottom: 4px;
    color: #94a2b1;
    font-size: 10px;
    font-weight: 750;
    line-height: 1;
    text-transform: uppercase;
    white-space: nowrap;
}

.st-key-ibd_selected_row .selected-value {
    min-width: 0;
    color: #f2f5f9;
    font-size: 14px;
    font-weight: 750;
    line-height: 1.1;
    white-space: nowrap;
}

.st-key-ibd_selected_row .selected-secondary {
    color: #a8b4c1;
    font-size: 11px;
    font-weight: 500;
}

.st-key-ibd_selected_row .ibd-selected-strip--empty {
    display: flex;
    align-items: center;
    justify-content: center;
    color: #9ca8b7;
    font-size: 13px;
}

button:focus-visible,
input:focus-visible,
select:focus-visible,
[tabindex]:focus-visible {
    outline: 2px solid #5aa2ff !important;
    outline-offset: 2px !important;
}

@media (width <= 1320px) {
    .st-key-ibd_selected_row .selected-secondary {
        display: none;
    }
}

@media (width < 1280px) {
    .st-key-review_context_slot:has(.st-key-quick_context_row) {
        padding-top: 5px;
        padding-bottom: 5px;
    }

    .st-key-quick_context_row > div[data-testid="stHorizontalBlock"] {
        grid-template-columns: minmax(0, 1fr) 74px;
        row-gap: 5px !important;
    }

    .st-key-quick_context_row > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(1),
    .st-key-quick_context_row > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2) {
        grid-column: 1;
    }

    .st-key-quick_context_row > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2) {
        grid-row: 2;
    }

    .st-key-quick_context_row > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(3) {
        grid-column: 2;
        grid-row: 1 / span 2;
        align-self: center;
    }
}

@media (width <= 1120px) {
    div[data-testid="stMainBlockContainer"]:has(.st-key-dashboard_shell) {
        padding: 22px 18px 28px !important;
    }

    .st-key-status_cards > div[data-testid="stHorizontalBlock"] {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .st-key-selected_row .selected-strip {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .st-key-selected_row .selected-code-cell {
        grid-column: 1 / -1;
    }

    .st-key-selected_row .selected-summary-cell {
        border-bottom: 1px solid #36404b;
    }
}

@media (width <= 760px) {
    .st-key-dashboard_header {
        min-height: 0;
        padding-bottom: 15px;
    }

    .st-key-dashboard_header > div[data-testid="stHorizontalBlock"] {
        display: grid !important;
        grid-template-columns: 1fr;
        width: 100%;
    }

    .st-key-dashboard_header > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
        width: 100% !important;
        min-width: 0 !important;
        flex: none !important;
    }

    .dashboard-snapshot {
        flex-wrap: wrap;
        margin-top: 14px;
        white-space: normal;
    }

    .snapshot-segment,
    .snapshot-mode-segment {
        min-width: 0;
    }

    .st-key-results_toolbar > div[data-testid="stHorizontalBlock"] {
        display: grid !important;
        grid-template-columns: 1fr;
        width: 100%;
    }

    .st-key-c_rank_results_toolbar > div[data-testid="stHorizontalBlock"] {
        display: grid !important;
        grid-template-columns: 1fr;
        width: 100%;
    }

    .st-key-results_toolbar > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
        width: 100% !important;
        min-width: 0 !important;
        flex: none !important;
    }

    .st-key-c_rank_results_toolbar > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
        width: 100% !important;
        min-width: 0 !important;
        flex: none !important;
    }

    .st-key-results_actions {
        width: 100%;
        max-width: none;
    }

    .st-key-results_toolbar iframe {
        width: 100% !important;
        min-width: 0;
    }

    .st-key-c_rank_results_toolbar iframe {
        width: 100% !important;
        min-width: 0;
    }

    .st-key-c_rank_results_toolbar div[data-baseweb="select"] > div {
        width: 100% !important;
        min-width: 170px;
    }

    .st-key-review_queue_heading > div[data-testid="stHorizontalBlock"] {
        grid-template-columns: 1fr 1fr;
        width: 100%;
    }

    .st-key-review_queue_heading > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child {
        grid-column: 1 / -1;
    }

    .st-key-status_cards > div[data-testid="stHorizontalBlock"] {
        grid-template-columns: 1fr;
    }

    .st-key-selected_row .selected-strip {
        grid-template-columns: 1fr;
    }

    .st-key-selected_row .selected-code-cell {
        grid-column: auto;
    }

    .st-key-results_toolbar > div[data-testid="stHorizontalBlock"] {
        grid-template-columns: minmax(0, 1fr) 145px;
    }

    .st-key-results_toolbar .results-summary {
        font-size: 12px;
    }

    .st-key-ibd_selected_row .ibd-selected-strip {
        grid-template-columns: 72px repeat(4, minmax(0, 1fr));
    }

    .st-key-ibd_selected_row .selected-summary-cell {
        padding: 6px 4px;
    }

    .st-key-ibd_selected_row .selected-label {
        font-size: 8px;
    }

    .st-key-ibd_selected_row .selected-value {
        font-size: 11px;
    }

}

@media (width <= 480px) {
    div[data-testid="stMainBlockContainer"]:has(.st-key-dashboard_shell) {
        padding: 18px 12px !important;
    }

    .st-key-review_queue_heading > div[data-testid="stHorizontalBlock"] {
        grid-template-columns: 1fr;
    }

    .st-key-review_queue_heading > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child {
        grid-column: auto;
    }

    .dashboard-title {
        font-size: 25px !important;
    }

    .st-key-dashboard_header div[data-testid="stColumn"]:has(.st-key-btn_info_rules) {
        width: 100% !important;
    }

    .st-key-dashboard_header div[data-testid="stColumn"]:has(.st-key-btn_info_rules) > div[data-testid="stVerticalBlockBorderWrapper"] > div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"] {
        display: grid !important;
        grid-template-columns: 45px minmax(0, 1fr);
        width: 100%;
    }

    .st-key-dashboard_header div[class*="st-key-global_mode_selector"] {
        width: 100%;
    }
}

@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        scroll-behavior: auto !important;
        transition-duration: 0.01ms !important;
        transition-delay: 0s !important;
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
    }
}
"""
