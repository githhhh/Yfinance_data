from __future__ import annotations

import pandas as pd


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
        allow_unsafe_jscode=False,
        enable_enterprise_modules=False,
        update_mode="NO_UPDATE",
    )


def _column_def(column: str) -> dict:
    definition = {
        "field": column,
        "headerName": column,
        "minWidth": 120,
        "sortable": True,
        "filter": True,
        "resizable": True,
        "pinned": None,
    }
    if column == "code":
        definition["pinned"] = "left"
        definition["minWidth"] = 96
        definition["lockPinned"] = False
    return definition
