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
    padding: 29px 28px 34px !important;
}

.st-key-dashboard_shell > div[data-testid="stVerticalBlock"] {
    gap: 0 !important;
}

.st-key-dashboard_header {
    min-height: 78px;
    border-bottom: 1px solid #aab1bb;
}

.st-key-dashboard_header div[class*="st-key-btn_info_rules"] button {
    width: 45px !important;
    min-width: 45px !important;
    max-width: 45px !important;
    height: 45px !important;
    min-height: 45px !important;
    max-height: 45px !important;
    border-radius: 11px !important;
}

.st-key-dashboard_header div[class*="st-key-global_mode_selector"] div[role="radiogroup"],
.st-key-dashboard_header div[class*="st-key-global_mode_selector"] button,
.st-key-dashboard_header div[class*="st-key-global_mode_selector"] label {
    height: 45px !important;
    min-height: 45px !important;
    max-height: 45px !important;
}

.snapshot-segment {
    display: inline-block;
    min-width: 139px;
}

.snapshot-mode-segment {
    display: inline-block;
    min-width: 211px;
}

.st-key-review_queue_heading {
    min-height: 50px;
}

.st-key-review_queue_actions > div[data-testid="stHorizontalBlock"] {
    display: grid !important;
    grid-template-columns: 276px 268px;
    gap: 9px !important;
    align-items: center;
}

.st-key-review_mode_controls,
.st-key-review_scope_controls {
    width: 100% !important;
    min-width: 0 !important;
    max-width: none !important;
    height: 40px;
}

.st-key-review_mode_controls button,
.st-key-review_scope_controls button {
    width: 100% !important;
    height: 40px !important;
    min-height: 40px !important;
    padding: 0 10px !important;
    white-space: nowrap !important;
}

.st-key-review_context_slot {
    height: 48px;
    min-height: 48px !important;
    max-height: 48px !important;
    margin: 6px 0 8px;
}

.st-key-review_context_slot > div[data-testid="stVerticalBlock"] {
    height: 48px;
    min-height: 48px !important;
    max-height: 48px !important;
    gap: 0 !important;
}

.st-key-status_cards > div[data-testid="stHorizontalBlock"] {
    display: grid !important;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 16px !important;
}

.st-key-status_cards div[class*="st-key-flow_card_"] button[kind] {
    width: 100% !important;
    height: 70px;
    min-height: 70px !important;
    max-height: 70px !important;
}

.st-key-filters_header button {
    width: 100% !important;
    height: 45px;
    min-height: 45px !important;
    max-height: 45px !important;
}

.st-key-results_toolbar {
    min-height: 56px;
}

.st-key-results_toolbar iframe {
    min-width: 180px;
    height: 44px !important;
}

.st-key-results_toolbar div[data-baseweb="select"] > div {
    min-width: 120px;
    height: 44px !important;
}

.st-key-selected_row .selected-strip {
    display: grid;
    grid-template-columns: 194px repeat(4, minmax(0, 1fr));
    width: 100%;
    height: 60px;
    overflow: hidden;
}

button:focus-visible,
input:focus-visible,
select:focus-visible,
[tabindex]:focus-visible {
    outline: 2px solid #5aa2ff !important;
    outline-offset: 2px !important;
}

@media (width <= 1120px) {
    div[data-testid="stMainBlockContainer"]:has(.st-key-dashboard_shell) {
        padding: 22px 18px 28px !important;
    }

    .st-key-status_cards > div[data-testid="stHorizontalBlock"] {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .st-key-selected_row .selected-strip {
        grid-template-columns: 170px repeat(4, minmax(170px, 1fr));
        overflow-x: auto;
    }
}

@media (width <= 760px) {
    .st-key-review_queue_actions > div[data-testid="stHorizontalBlock"] {
        grid-template-columns: 1fr 1fr;
        width: 100%;
    }

    .st-key-review_context_slot {
        overflow: auto hidden !important;
    }

    .st-key-status_cards > div[data-testid="stHorizontalBlock"] {
        grid-template-columns: 1fr;
    }

    .st-key-selected_row .selected-strip {
        grid-template-columns: 150px repeat(4, 180px);
        overflow-x: auto;
    }
}

@media (width <= 480px) {
    div[data-testid="stMainBlockContainer"]:has(.st-key-dashboard_shell) {
        padding: 18px 12px !important;
    }

    .st-key-review_queue_actions > div[data-testid="stHorizontalBlock"] {
        grid-template-columns: 1fr;
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
