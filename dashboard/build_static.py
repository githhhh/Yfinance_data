from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.data_utils import build_snapshot_freshness
from dashboard.field_config import (
    FLOW_CARD_META,
    STATUS_META,
    get_column_view_fields,
    get_default_table_columns,
    get_midweek_table_columns,
)
from dashboard.services.bf_midweek_review import (
    PoolMode,
    analyze_breakout_follow_pool,
    materialize_review_view,
)


DASHBOARD_DIR = Path(__file__).resolve().parent
STATIC_ASSETS = (
    "index.html",
    "app.js",
    "table_enhancements.js",
    "styles.css",
    "manifest.webmanifest",
)


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()[:10] if not isinstance(value, datetime) else value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return [
        {str(key): _json_value(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _complete_view(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["review_watch_active"] = result["signal"]
    result["review_effective_entry_status"] = result["ibd_entry_status"]
    result["review_priority"] = pd.to_numeric(
        result.get("rank_C_continuous"), errors="coerce"
    )
    return result


def _status_meta() -> dict[str, Any]:
    return {
        key: {
            "label": meta["label"],
            "subtitle": meta["subtitle"],
            "tone": meta["tone"],
            "color": meta["color"],
            "tooltip_title": meta["tooltip_title"],
            "tooltip": meta["tooltip"],
        }
        for key, meta in STATUS_META.items()
    }


def _flow_meta() -> dict[str, Any]:
    return {
        key: {
            "label": meta["label"],
            "symbol": meta["symbol"],
            "color": meta["color"],
            "tooltip_title": meta["tooltip_title"],
            "tooltip": meta["tooltip"],
        }
        for key, meta in FLOW_CARD_META.items()
    }


def build_dashboard_payload(
    *,
    complete_path: str | Path,
    midweek_path: str | Path,
    window_date: date,
) -> dict[str, Any]:
    analysis = analyze_breakout_follow_pool(
        complete_path,
        midweek_path,
        window_date=window_date,
    )

    complete = _complete_view(analysis.complete_pool)
    midweek = (
        materialize_review_view(analysis.midweek_review)
        if analysis.midweek_available
        else pd.DataFrame()
    )

    complete_snapshot = (
        analysis.complete_snapshot_date.isoformat()
        if analysis.complete_snapshot_date is not None
        else None
    )
    midweek_snapshot = (
        analysis.midweek_snapshot_date.isoformat()
        if analysis.midweek_snapshot_date is not None
        else None
    )
    freshness = build_snapshot_freshness(complete_snapshot, today=window_date)

    default_period = (
        "MIDWEEK"
        if analysis.mode in {PoolMode.MIDWEEK, PoolMode.MIDWEEK_WITHOUT_VALID_BASELINE}
        and analysis.midweek_available
        else "WEEKEND"
    )

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_date": window_date.isoformat(),
        "default_period": default_period,
        "meta": {
            "complete_snapshot_date": complete_snapshot,
            "midweek_snapshot_date": midweek_snapshot,
            "review_week_start": (
                analysis.review_week_start.isoformat()
                if analysis.review_week_start is not None
                else None
            ),
            "midweek_available": bool(analysis.midweek_available),
            "midweek_baseline_available": bool(analysis.midweek_baseline_available),
            "warnings": list(analysis.warnings),
            "summary": dict(analysis.summary),
            "complete_freshness": freshness,
        },
        "views": {
            "weekend": {
                "rows": _records(complete),
                "table_columns": get_default_table_columns(),
            },
            "midweek": {
                "rows": _records(midweek),
                "table_columns": (
                    get_midweek_table_columns()
                    if analysis.midweek_baseline_available
                    else get_default_table_columns()
                ),
            },
            "c_rank": {
                "rows": _records(complete),
                "table_columns": get_column_view_fields("C Rank Reference"),
            },
        },
        "ui": {
            "status_meta": _status_meta(),
            "flow_meta": _flow_meta(),
            "setup_options": [
                "All",
                "ceiling",
                "ceiling_pullback",
                "ma10_touch_confirm",
                "pivot",
                "three_weeks_tight",
            ],
        },
    }


def build_site(
    output_dir: str | Path,
    *,
    complete_path: str | Path,
    midweek_path: str | Path,
    window_date: date,
) -> Path:
    output = Path(output_dir).resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    for asset in STATIC_ASSETS:
        source = DASHBOARD_DIR / asset
        if not source.is_file():
            raise FileNotFoundError(f"Static dashboard asset missing: {source}")
        shutil.copy2(source, output / asset)

    payload = build_dashboard_payload(
        complete_path=complete_path,
        midweek_path=midweek_path,
        window_date=window_date,
    )
    data_dir = output / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "dashboard.json").write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    (output / ".nojekyll").write_text("", encoding="utf-8")
    return output


def _parse_date(value: str | None) -> date:
    if value:
        return date.fromisoformat(value)
    from zoneinfo import ZoneInfo

    return datetime.now(ZoneInfo("Asia/Shanghai")).date()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the static Breakout Pool review site.")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "_site"))
    parser.add_argument(
        "--complete",
        default=str(PROJECT_ROOT / "us" / "breakout_follow_pool.csv"),
    )
    parser.add_argument(
        "--midweek",
        default=str(PROJECT_ROOT / "us" / "breakout_follow_pool_midweek.csv"),
    )
    parser.add_argument("--window-date", default=None)
    args = parser.parse_args()

    output = build_site(
        args.output,
        complete_path=args.complete,
        midweek_path=args.midweek,
        window_date=_parse_date(args.window_date),
    )
    print(f"Static dashboard built: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
