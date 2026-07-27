from __future__ import annotations

import json

import pandas as pd

from dashboard.field_config import (
    FIELD_CONFIG,
    QUALITY_ALIASES,
    QUALITY_META,
    QUALITY_ORDER,
    STATUS_META,
    get_field_label,
)

try:
    from st_aggrid import JsCode

    HAS_JS_CODE = True
except ImportError:
    HAS_JS_CODE = False


def _code_renderer_jscode():
    if not HAS_JS_CODE:
        return None
    return JsCode("""
    class CodeCellRenderer {
        init(params) {
            this.params = params;
            this.eGui = document.createElement('div');
            this.eGui.style.cssText = 'cursor:pointer; display:flex; align-items:center; justify-content:space-between; width:100%; font-weight:600; color:#1f77b4;';
            
            const codeText = params.value || '';
            this.eGui.innerHTML = '<span>' + codeText + '</span><span style="font-size:12px; margin-left:4px; opacity:0.75;">📋</span>';
            this.eGui.title = 'Click to copy ' + codeText;
            
            this.eGui.addEventListener('click', async (e) => {
                e.stopPropagation();
                const textToCopy = String(codeText);
                let success = false;
                if (navigator.clipboard && navigator.clipboard.writeText) {
                    try {
                        await navigator.clipboard.writeText(textToCopy);
                        success = true;
                    } catch (err) {
                        // fallback to execCommand
                    }
                }
                if (!success) {
                    try {
                        const ta = document.createElement('textarea');
                        ta.value = textToCopy;
                        ta.style.position = 'fixed';
                        ta.style.left = '-9999px';
                        document.body.appendChild(ta);
                        ta.focus();
                        ta.select();
                        success = document.execCommand('copy');
                        document.body.removeChild(ta);
                    } catch (err2) {
                        success = false;
                    }
                }
                if (success) {
                    this.eGui.innerHTML = '<span style="color:#2e7d32;">' + codeText + '</span><span style="font-size:11px; color:#2e7d32; margin-left:4px;">✓ Copied</span>';
                } else {
                    this.eGui.innerHTML = '<span style="color:#c62828;">' + codeText + '</span><span style="font-size:11px; color:#c62828; margin-left:4px;">Error</span>';
                }
                setTimeout(() => {
                    if (this.eGui) {
                        this.eGui.innerHTML = '<span>' + codeText + '</span><span style="font-size:12px; margin-left:4px; opacity:0.75;">📋</span>';
                    }
                }, 1500);
            });
        }
        getGui() {
            return this.eGui;
        }
    }
    """)


def _get_value_formatter(fmt: str | None):
    if not HAS_JS_CODE or not fmt:
        return None
    if fmt == "0.00%":
        return JsCode("""
        function(params) {
            if (params.value === null || params.value === undefined || params.value === '') return '';
            const val = Number(params.value);
            if (isNaN(val)) return params.value;
            if (val > 0) return '+' + val.toFixed(2) + '%';
            if (val < 0) return val.toFixed(2) + '%';
            return '0.00%';
        }
        """)
    if fmt == "0.0%":
        return JsCode("""
        function(params) {
            if (params.value === null || params.value === undefined || params.value === '') return '';
            const val = Number(params.value);
            if (isNaN(val)) return params.value;
            if (val > 0) return '+' + val.toFixed(1) + '%';
            if (val < 0) return val.toFixed(1) + '%';
            return '0.0%';
        }
        """)
    if fmt == "0.00x":
        return JsCode("""
        function(params) {
            if (params.value === null || params.value === undefined || params.value === '') return '';
            const val = Number(params.value);
            if (isNaN(val)) return params.value;
            return val.toFixed(2) + 'x';
        }
        """)
    if fmt == "0.00":
        return JsCode("""
        function(params) {
            if (params.value === null || params.value === undefined || params.value === '') return '';
            const val = Number(params.value);
            if (isNaN(val)) return params.value;
            return val.toFixed(2);
        }
        """)
    return None


def _breakout_quality_header_jscode():
    if not HAS_JS_CODE:
        return None
    return JsCode("""
    class BreakoutQualityHeader {
        init(params) {
            this.params = params;
            this.tooltip = null;
            this.eGui = document.createElement('div');
            this.eGui.style.cssText = 'display:flex; align-items:center; width:100%; height:100%; min-width:0; gap:6px; cursor:pointer;';

            this.label = document.createElement('span');
            this.label.textContent = params.displayName || 'Breakout Quality';
            this.label.style.cssText = 'min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;';

            this.sortIcon = document.createElement('span');
            this.sortIcon.style.cssText = 'width:10px; color:#94a3b8; font-size:10px; line-height:1;';

            this.menuButton = document.createElement('button');
            this.menuButton.type = 'button';
            this.menuButton.setAttribute('aria-label', 'Filter Breakout Quality');
            this.menuButton.style.cssText = 'width:18px; height:18px; margin-left:auto; padding:0; border:0; background:transparent; color:#9ca3af; cursor:pointer; display:flex; align-items:center; justify-content:center;';
            this.menuButton.innerHTML = '<span style="width:12px; height:12px; display:block; background:currentColor; clip-path:polygon(0 0,100% 0,62% 45%,62% 100%,38% 100%,38% 45%); opacity:0.85;"></span>';

            this.eGui.appendChild(this.label);
            this.eGui.appendChild(this.sortIcon);
            this.eGui.appendChild(this.menuButton);

            this.onSort = (event) => {
                if (event.target === this.menuButton || this.menuButton.contains(event.target)) return;
                params.progressSort(event.shiftKey);
            };
            this.onMenu = (event) => {
                event.stopPropagation();
                if (params.showColumnMenu) {
                    params.showColumnMenu(this.menuButton);
                } else if (params.showColumnMenuAfterButtonClick) {
                    params.showColumnMenuAfterButtonClick(this.menuButton);
                }
            };
            this.onMouseEnter = () => this.showTooltip();
            this.onMouseLeave = () => this.hideTooltip();
            this.onSortChanged = () => this.updateSortIcon();

            this.eGui.addEventListener('click', this.onSort);
            this.eGui.addEventListener('mouseenter', this.onMouseEnter);
            this.eGui.addEventListener('mouseleave', this.onMouseLeave);
            this.menuButton.addEventListener('click', this.onMenu);
            params.column.addEventListener('sortChanged', this.onSortChanged);
            this.updateSortIcon();
        }

        showTooltip() {
            this.hideTooltip();
            const tooltip = document.createElement('div');
            tooltip.className = 'breakout-tooltip';
            tooltip.style.cssText = 'position:fixed; width:292px; padding:12px 14px; border-radius:6px; background:#0b1329; color:#e2e8f0; box-shadow:0 8px 24px rgba(0,0,0,0.45); font-size:11px; line-height:1.35; border:1px solid rgba(148,163,184,0.22); z-index:999999; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; pointer-events:none;';
            tooltip.innerHTML = `
                <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:10px;">
                    <div style="font-weight:700; color:#ffffff; font-size:12px; letter-spacing:0;">Breakout Quality</div>
                    <div style="color:#94a3b8; font-size:10px;">strong to weak</div>
                </div>
                <div style="display:grid; grid-template-columns:56px 1fr; gap:14px; align-items:center;">
                    <div style="display:flex; flex-direction:column; align-items:center; justify-content:center;">
                        <div style="width:48px; height:7px; border-radius:2px; margin-bottom:3px; background:#2e7d32;"></div>
                        <div style="width:38px; height:7px; border-radius:2px; margin-bottom:3px; background:rgba(46,125,50,0.78);"></div>
                        <div style="width:28px; height:7px; border-radius:2px; margin-bottom:3px; background:rgba(46,125,50,0.56);"></div>
                        <div style="width:18px; height:7px; border-radius:2px; margin-bottom:3px; background:rgba(46,125,50,0.38);"></div>
                        <div style="width:8px; height:7px; border-radius:2px; background:rgba(46,125,50,0.22);"></div>
                    </div>
                    <div style="display:grid; grid-template-columns:1fr; gap:4px; font-size:11px; min-width:0;">
                        <div style="display:grid; grid-template-columns:112px 1fr; gap:8px; white-space:nowrap;">
                            <span style="color:#c8f7c5; font-weight:700;">Powerful</span><span style="color:#94a3b8;">Very wide breakout</span>
                        </div>
                        <div style="display:grid; grid-template-columns:112px 1fr; gap:8px; white-space:nowrap;">
                            <span style="color:#b7efb1; font-weight:700;">Strong</span><span style="color:#94a3b8;">High close + solid range</span>
                        </div>
                        <div style="display:grid; grid-template-columns:112px 1fr; gap:8px; white-space:nowrap;">
                            <span style="color:#d4f1d1; font-weight:600;">Tight</span><span style="color:#94a3b8;">High close + tight range</span>
                        </div>
                        <div style="display:grid; grid-template-columns:112px 1fr; gap:8px; white-space:nowrap;">
                            <span style="color:#e3f6e1; font-weight:500;">Constructive</span><span style="color:#94a3b8;">Decent close</span>
                        </div>
                        <div style="display:grid; grid-template-columns:112px 1fr; gap:8px; white-space:nowrap;">
                            <span style="color:#94a3b8; font-weight:400;">Weak</span><span style="color:#64748b;">Low close</span>
                        </div>
                    </div>
                </div>
            `;
            document.body.appendChild(tooltip);
            const anchor = this.eGui.getBoundingClientRect();
            const tip = tooltip.getBoundingClientRect();
            const left = Math.min(Math.max(8, anchor.left), Math.max(8, window.innerWidth - tip.width - 8));
            const below = anchor.bottom + 6;
            const top = below + tip.height < window.innerHeight ? below : Math.max(8, anchor.top - tip.height - 6);
            tooltip.style.left = left + 'px';
            tooltip.style.top = top + 'px';
            this.tooltip = tooltip;
        }

        hideTooltip() {
            if (this.tooltip && this.tooltip.parentNode) {
                this.tooltip.parentNode.removeChild(this.tooltip);
            }
            this.tooltip = null;
        }

        updateSortIcon() {
            const sort = this.params.column.getSort();
            this.sortIcon.textContent = sort === 'asc' ? '^' : sort === 'desc' ? 'v' : '';
        }

        getGui() {
            return this.eGui;
        }

        destroy() {
            this.hideTooltip();
            this.eGui.removeEventListener('click', this.onSort);
            this.eGui.removeEventListener('mouseenter', this.onMouseEnter);
            this.eGui.removeEventListener('mouseleave', this.onMouseLeave);
            this.menuButton.removeEventListener('click', this.onMenu);
            this.params.column.removeEventListener('sortChanged', this.onSortChanged);
        }
    }
    """)


def _breakout_quality_cell_renderer_jscode():
    if not HAS_JS_CODE:
        return None
    meta_json = json.dumps(QUALITY_META)
    return JsCode("""
    class BreakoutQualityCellRenderer {
        init(params) {
            this.eGui = document.createElement('span');
            this.tooltip = null;
            this.eGui.style.cssText = 'display:block; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;';
            this.onMouseEnter = () => this.showTooltip();
            this.onMouseLeave = () => this.hideTooltip();
            this.eGui.addEventListener('mouseenter', this.onMouseEnter);
            this.eGui.addEventListener('mouseleave', this.onMouseLeave);
            this.refresh(params);
        }

        refresh(params) {
            this.params = params;
            const data = params.data || {};
            const val = params.value == null ? '' : String(params.value);
            const meta = __QUALITY_META__;
            this.eGui.textContent = val;
            if (!val || val === 'nan' || val === 'None' || val === 'undefined') {
                this.tooltipLines = [];
                return true;
            }

            const pos = Number(data.ibd_entry_close_position);
            const rr = Number(data.ibd_entry_breakout_range_ratio);
            const posStr = isNaN(pos) ? 'n/a' : pos.toFixed(2);
            const rrStr = isNaN(rr) ? 'n/a' : rr.toFixed(2);
            const whyByGrade = {
                'Powerful Breakout': 'Very wide breakout',
                'Strong Close': 'High close + solid range',
                'Constructive Close (Tight)': 'High close + tight range',
                'Constructive Close': 'Decent close',
                'Weak Close': 'Low close'
            };
            const why = whyByGrade[val] || 'Quality inputs are incomplete';
            this.tooltipLines = [
                'Pos ' + posStr + ' · Range ' + rrStr,
                why
            ];
            return true;
        }

        showTooltip() {
            this.hideTooltip();
            if (!this.tooltipLines || !this.tooltipLines.length) return;
            const tooltip = document.createElement('div');
            tooltip.className = 'breakout-cell-tooltip';
            tooltip.style.cssText = 'position:fixed; width:190px; padding:8px 10px; border-radius:6px; background:#0b1329; color:#dbeafe; box-shadow:0 8px 22px rgba(0,0,0,0.42); font-size:11px; line-height:1.45; border:1px solid rgba(148,163,184,0.22); z-index:999999; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; pointer-events:none; white-space:pre-wrap;';
            tooltip.textContent = this.tooltipLines.join('\\n');
            document.body.appendChild(tooltip);
            const anchor = this.eGui.getBoundingClientRect();
            const tip = tooltip.getBoundingClientRect();
            const left = Math.min(Math.max(8, anchor.left), Math.max(8, window.innerWidth - tip.width - 8));
            const below = anchor.bottom + 6;
            const top = below + tip.height < window.innerHeight ? below : Math.max(8, anchor.top - tip.height - 6);
            tooltip.style.left = left + 'px';
            tooltip.style.top = top + 'px';
            this.tooltip = tooltip;
        }

        hideTooltip() {
            if (this.tooltip && this.tooltip.parentNode) {
                this.tooltip.parentNode.removeChild(this.tooltip);
            }
            this.tooltip = null;
        }

        getGui() {
            return this.eGui;
        }

        destroy() {
            this.hideTooltip();
            this.eGui.removeEventListener('mouseenter', this.onMouseEnter);
            this.eGui.removeEventListener('mouseleave', this.onMouseLeave);
        }
    }
    """.replace("__QUALITY_META__", meta_json))


def build_grid_options(columns: list[str]) -> dict:
    options = {
        "columnDefs": [_column_def(column) for column in columns],
        "defaultColDef": {
            "sortable": True,
            "filter": True,
            "resizable": True,
            "editable": False,
        },
        "enableBrowserTooltips": False,
        "tooltipShowDelay": 300,
        "tooltipHideDelay": 5000,
        "enableRangeSelection": False,
        "rowSelection": "single",
        "suppressRowClickSelection": False,
        "suppressDragLeaveHidesColumns": True,
        "animateRows": False,
    }
    if HAS_JS_CODE:
        options["getRowId"] = JsCode("""
        function(params) {
            return params.data && params.data.code ? String(params.data.code) : String(params.node ? params.node.rowIndex : Math.random());
        }
        """)
        options["components"] = {
            "breakoutQualityHeader": _breakout_quality_header_jscode(),
            "breakoutQualityCellRenderer": _breakout_quality_cell_renderer_jscode(),
        }
    return options


BREAKOUT_QUALITY_TOOLTIP_FIELDS = [
    "ibd_entry_close_vs_trigger_pct",
    "ibd_entry_close_position",
    "ibd_entry_breakout_range_ratio",
]


def _row_data_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    display_columns = [column for column in columns if column in df.columns]
    support_columns = (
        [column for column in BREAKOUT_QUALITY_TOOLTIP_FIELDS if column in df.columns]
        if "ibd_breakout_quality" in display_columns
        else []
    )
    return list(dict.fromkeys(display_columns + support_columns))


def _normalize_breakout_quality_display_values(df: pd.DataFrame) -> pd.DataFrame:
    if "ibd_breakout_quality" not in df.columns:
        return df
    result = df.copy()
    result["ibd_breakout_quality"] = result["ibd_breakout_quality"].replace(QUALITY_ALIASES)
    return result


def render_table(df: pd.DataFrame, columns: list[str], height: int = 620) -> str | None:
    display_columns = [column for column in columns if column in df.columns]
    row_columns = _row_data_columns(df, display_columns)
    grid_df = df[row_columns].copy() if row_columns else df.copy()
    grid_df = _normalize_breakout_quality_display_values(grid_df)
    grid_df.index = range(1, len(grid_df) + 1)

    for col in grid_df.columns:
        if col in display_columns and pd.api.types.is_float_dtype(grid_df[col]):
            grid_df[col] = grid_df[col].round(2)

    try:
        from st_aggrid import AgGrid
    except ImportError:
        import streamlit as st

        visible_df = grid_df[display_columns].copy() if display_columns else grid_df
        st.dataframe(visible_df, use_container_width=True, height=height)
        st.caption("Install streamlit-aggrid to enable pinning, drag columns, range selection, and copy support.")
        return None

    grid_response = AgGrid(
        grid_df,
        gridOptions=build_grid_options(display_columns),
        height=height,
        fit_columns_on_grid_load=False,
        allow_unsafe_jscode=HAS_JS_CODE,
        enable_enterprise_modules=False,
        update_on=["selectionChanged"],
        use_json_serialization=True,
    )
    selected = grid_response.get("selected_rows", None)
    if selected is None or len(selected) == 0:
        return None
    if isinstance(selected, pd.DataFrame):
        if "code" in selected.columns and not selected.empty:
            return str(selected.iloc[0]["code"])
    elif isinstance(selected, (list, tuple)):
        first = selected[0]
        if isinstance(first, dict) and "code" in first:
            return str(first["code"])
        elif isinstance(first, pd.Series) and "code" in first:
            return str(first["code"])
    return None


def _column_def(column: str) -> dict:
    definition = {
        "field": column,
        "headerName": get_field_label(column),
        "minWidth": 100,
        "sortable": True,
        "filter": True,
        "resizable": True,
        "pinned": None,
    }
    help_text = FIELD_CONFIG.get(column, {}).get("help")
    if help_text and not (HAS_JS_CODE and column == "ibd_breakout_quality"):
        definition["headerTooltip"] = help_text
    fmt = FIELD_CONFIG.get(column, {}).get("format")
    if fmt and HAS_JS_CODE:
        formatter = _get_value_formatter(fmt)
        if formatter:
            definition["valueFormatter"] = formatter
    if column == "code":
        definition["pinned"] = "left"
        definition["width"] = 85
        definition["minWidth"] = 85
        definition["lockPinned"] = False
        if HAS_JS_CODE:
            definition["cellRenderer"] = _code_renderer_jscode()
    elif column == "rank_C_continuous":
        definition["pinned"] = "right"
        definition["width"] = 85
        definition["minWidth"] = 85
    elif column == "ibd_candidate_rule":
        definition["width"] = 130
        definition["minWidth"] = 110
    elif column == "current_vs_ibd_candidate_pct":
        definition["width"] = 120
        definition["minWidth"] = 110
    elif column == "latest_close":
        definition["width"] = 100
        definition["minWidth"] = 90
    elif column == "volume_ratio":
        definition["width"] = 85
        definition["minWidth"] = 85
    elif column == "ibd_entry_status":
        definition["width"] = 115
        definition["minWidth"] = 105
    elif column == "ibd_entry_vol_or_reject":
        definition["minWidth"] = 180
        definition["flex"] = 1
    elif column == "ibd_breakout_quality":
        definition["minWidth"] = 210
        definition["flex"] = 1
        if HAS_JS_CODE:
            definition["headerComponent"] = "breakoutQualityHeader"
            definition["cellRenderer"] = "breakoutQualityCellRenderer"
            order_json = json.dumps(QUALITY_ORDER)
            definition["comparator"] = JsCode("""
            function(valueA, valueB, nodeA, nodeB, isDescending) {
                const order = __QUALITY_ORDER__;
                const keyA = valueA == null ? '' : String(valueA);
                const keyB = valueB == null ? '' : String(valueB);
                const rankA = Object.prototype.hasOwnProperty.call(order, keyA) ? order[keyA] : null;
                const rankB = Object.prototype.hasOwnProperty.call(order, keyB) ? order[keyB] : null;
                if (rankA === null && rankB === null) return 0;
                if (rankA === null) return isDescending ? -1 : 1;
                if (rankB === null) return isDescending ? 1 : -1;
                return rankA - rankB;
            }
            """.replace("__QUALITY_ORDER__", order_json))
    else:
        definition["width"] = 130

    if HAS_JS_CODE and column == "ibd_entry_status":
        meta_json = json.dumps(STATUS_META)
        definition["cellStyle"] = JsCode(f"""
        function(params) {{
            const val = String(params.value || '');
            const meta = {meta_json};
            if (meta[val] && meta[val].color) {{
                return {{'color': meta[val].color, 'fontWeight': '700'}};
            }}
            return {{'fontWeight': '600'}};
        }}
        """)
        definition["tooltipValueGetter"] = JsCode(f"""
        function(params) {{
            const val = String(params.value || '');
            const meta = {meta_json};
            if (meta[val] && meta[val].tooltip) {{
                return meta[val].tooltip;
            }}
            return val;
        }}
        """)
    elif HAS_JS_CODE and column == "ibd_breakout_quality":
        meta_json = json.dumps(QUALITY_META)
        definition["cellStyle"] = JsCode(f"""
        function(params) {{
            const val = String(params.value || '');
            const meta = {meta_json};
            if (meta[val] && meta[val].color) {{
                return {{
                    'color': meta[val].color,
                    'backgroundColor': meta[val].backgroundColor || 'transparent',
                    'backgroundImage': meta[val].backgroundImage || 'none',
                    'fontWeight': meta[val].fontWeight || '600',
                    'borderLeft': (meta[val].borderWidth || '3px') + ' solid ' + (meta[val].borderColor || meta[val].color)
                }};
            }}
            return {{'color': '#757575', 'fontWeight': '400'}};
        }}
        """)
    return definition
