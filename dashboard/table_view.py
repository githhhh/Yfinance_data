from __future__ import annotations

import pandas as pd

from dashboard.field_config import FIELD_CONFIG, get_field_label

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
            
            this.eGui.addEventListener('click', (e) => {
                e.stopPropagation();
                const textToCopy = String(codeText);
                
                const ta = document.createElement('textarea');
                ta.value = textToCopy;
                ta.style.position = 'fixed';
                ta.style.left = '-9999px';
                document.body.appendChild(ta);
                ta.focus();
                ta.select();
                try {
                    document.execCommand('copy');
                } catch (err) {
                    if (navigator.clipboard) {
                        navigator.clipboard.writeText(textToCopy);
                    }
                }
                document.body.removeChild(ta);
                
                this.eGui.innerHTML = '<span style="color:#2e7d32;">' + codeText + '</span><span style="font-size:11px; color:#2e7d32; margin-left:4px;">✓ Copied</span>';
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


def build_grid_options(columns: list[str]) -> dict:
    return {
        "columnDefs": [_column_def(column) for column in columns],
        "defaultColDef": {
            "sortable": True,
            "filter": True,
            "resizable": True,
            "editable": False,
        },
        "enableRangeSelection": True,
        "rowSelection": "multiple",
        "suppressDragLeaveHidesColumns": True,
        "animateRows": False,
    }


def render_table(df: pd.DataFrame, columns: list[str], height: int = 620) -> None:
    display_columns = [column for column in columns if column in df.columns]
    display_df = df[display_columns].copy() if display_columns else df.copy()
    display_df.index = range(1, len(display_df) + 1)

    for col in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[col]):
            display_df[col] = display_df[col].round(2)

    try:
        from st_aggrid import AgGrid
    except ImportError:
        import streamlit as st

        st.dataframe(display_df, use_container_width=True, height=height)
        st.caption("Install streamlit-aggrid to enable pinning, drag columns, range selection, and copy support.")
        return

    AgGrid(
        display_df,
        gridOptions=build_grid_options(display_columns),
        height=height,
        fit_columns_on_grid_load=False,
        allow_unsafe_jscode=HAS_JS_CODE,
        enable_enterprise_modules=False,
        update_mode="NO_UPDATE",
    )


def _column_def(column: str) -> dict:
    definition = {
        "field": column,
        "headerName": get_field_label(column),
        "minWidth": 120,
        "sortable": True,
        "filter": True,
        "resizable": True,
        "pinned": None,
    }
    fmt = FIELD_CONFIG.get(column, {}).get("format")
    if fmt and HAS_JS_CODE:
        formatter = _get_value_formatter(fmt)
        if formatter:
            definition["valueFormatter"] = formatter
    if column == "code":
        definition["pinned"] = "left"
        definition["minWidth"] = 110
        definition["lockPinned"] = False
        if HAS_JS_CODE:
            definition["cellRenderer"] = _code_renderer_jscode()
    return definition
