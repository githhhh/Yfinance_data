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
from dashboard.field_config import FLOW_CARD_META, STATUS_META
from dashboard.rs_reference import (
    RSReferenceSnapshot,
    attach_rs_reference,
    fetch_latest_rs_reference,
    reference_meta,
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
    "rs_enhancements.js",
    "styles.css",
    "manifest.webmanifest",
)

PUBLIC_REVIEW_COLUMNS = (
    "code",
    "ibd_entry_status",
    "ibd_candidate_rule",
    "current_vs_ibd_candidate_pct",
    "ibd_breakout_quality",
    "latest_close",
    "ibd_entry_vol_or_reject",
    "volume_ratio",
    "rs_percentile",
)

# Public GitHub Pages contract. Pool/schema growth must never implicitly publish
# new columns. Add a field here only when the static UI intentionally consumes it.
PUBLIC_DASHBOARD_ROW_FIELDS = (
    "code",
    "signal",
    "snapshot_date",
    "review_watch_active",
    "review_effective_entry_status",
    "review_priority",
    "review_baseline_entry_status",
    "review_change_group",
    "review_change_label",
    "review_signal_origin",
    "ibd_entry_status",
    "ibd_candidate_rule",
    "ibd_candidate_price",
    "ibd_trigger_price",
    "current_vs_ibd_candidate_pct",
    "latest_close",
    "ibd_entry_valid",
    "ibd_entry_date",
    "ibd_entry_volume_ratio",
    "ibd_entry_vol_or_reject",
    "ibd_entry_reject_reason",
    "ibd_entry_close_position",
    "ibd_entry_breakout_range_ratio",
    "ibd_breakout_quality",
    "volume_ratio",
    "rs_percentile",
    "rs_1m_percentile",
    "rs_3m_percentile",
    "rs_6m_percentile",
    "eps_yoy_growth",
    "price_52_week_high",
    "dist_to_52w_high_pct",
    "pullback_pct",
    "pullback_pct_off_peak",
    "pullback_duration_weeks",
    "pullback_v_is_dry",
    "base_depth_pct",
    "base_duration_weeks",
    "industry",
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
    fields = [field for field in PUBLIC_DASHBOARD_ROW_FIELDS if field in frame.columns]
    return [
        {field: _json_value(row.get(field)) for field in fields}
        for row in frame.to_dict(orient="records")
    ]


def _complete_view(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["review_watch_active"] = result["signal"]
    result["review_effective_entry_status"] = result["ibd_entry_status"]
    # Weekend review has no transition priority. Keep the public field for a
    # stable row schema without falling back to the unvalidated C Rank.
    result["review_priority"] = None
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


def _review_columns(*, comparison: bool) -> list[str]:
    columns = list(PUBLIC_REVIEW_COLUMNS)
    if comparison:
        columns.insert(1, "review_change_label")
    return columns


def build_dashboard_payload(
    *,
    complete_path: str | Path,
    midweek_path: str | Path,
    window_date: date,
    rs_reference: RSReferenceSnapshot | None = None,
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

    complete = attach_rs_reference(
        complete,
        snapshot_date=analysis.complete_snapshot_date,
        reference=rs_reference,
    )
    midweek = attach_rs_reference(
        midweek,
        snapshot_date=analysis.midweek_snapshot_date,
        reference=rs_reference,
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
        "schema_version": 2,
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
            "rs_reference": reference_meta(
                rs_reference,
                complete_snapshot_date=analysis.complete_snapshot_date,
                midweek_snapshot_date=analysis.midweek_snapshot_date,
            ),
        },
        "views": {
            "weekend": {
                "rows": _records(complete),
                "table_columns": _review_columns(comparison=False),
            },
            "midweek": {
                "rows": _records(midweek),
                "table_columns": _review_columns(
                    comparison=bool(analysis.midweek_baseline_available)
                ),
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
    rs_reference: RSReferenceSnapshot | None = None,
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
        rs_reference=rs_reference,
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

    # RS is deliberately fail-soft. If the public source is unavailable or its
    # market date does not match a Pool snapshot, the site still builds and RS
    # is rendered as N/A.
    rs_reference = fetch_latest_rs_reference()
    output = build_site(
        args.output,
        complete_path=args.complete,
        midweek_path=args.midweek,
        window_date=_parse_date(args.window_date),
        rs_reference=rs_reference,
    )
    print(f"Static dashboard built: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
