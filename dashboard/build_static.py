from __future__ import annotations

import argparse
import json
import math
import os
import re
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
from dashboard.services.bf_observation import (
    build_observation_view,
    load_daily_snapshot,
    load_previous_midweek,
)
from dashboard.services.bf_review_presentation import decorate_observation_view
from dashboard.field_config import FLOW_CARD_META, STATUS_META
from dashboard.services.bf_midweek_review import (
    PoolMode,
    analyze_breakout_follow_pool,
    materialize_review_view,
)


DASHBOARD_DIR = Path(__file__).resolve().parent
STATIC_ASSETS = (
    "index.html",
    "pool-exit-watch.html",
    "app.js",
    "table_enhancements.js",
    "rs_runtime.js",
    "interaction_runtime.js",
    "styles.css",
    "manifest.webmanifest",
)

def _with_ga4_tag(index_html: str, measurement_id: str | None) -> str:
    """Opt-in GA4 for published static HTML; unconfigured builds stay unchanged."""
    if not measurement_id:
        return index_html
    if not re.fullmatch(r"G-[A-Z0-9]+", measurement_id):
        raise ValueError("Invalid GA4_MEASUREMENT_ID: expected G- followed by letters/digits")
    if index_html.count("<head>\n") != 1:
        raise ValueError("Expected exactly one HTML <head> for GA4 tag insertion")

    tag = (
        "  <!-- Google tag (gtag.js) -->\n"
        f'  <script async src="https://www.googletagmanager.com/gtag/js?id={measurement_id}"></script>\n'
        "  <script>\n"
        "    window.dataLayer = window.dataLayer || [];\n"
        "    function gtag(){dataLayer.push(arguments);}\n"
        "    gtag('js', new Date());\n"
        f"    gtag('config', '{measurement_id}');\n"
        "  </script>\n"
    )
    return index_html.replace("<head>\n", "<head>\n" + tag, 1)


PUBLIC_REVIEW_COLUMNS = (
    "code",
    "ibd_entry_status",
    "ibd_candidate_rule",
    "current_vs_ibd_candidate_pct",
    "ibd_breakout_quality",
    "latest_close",
    "ibd_entry_vol_or_reject",
    "volume_ratio",
)

# Public GitHub Pages contract. Pool/schema growth must never implicitly publish
# new columns. RS is intentionally absent: it is fetched independently by the
# browser at runtime and never becomes part of the authoritative Pool payload.
PUBLIC_DASHBOARD_ROW_FIELDS = (
    "code",
    "signal",
    "snapshot_date",
    "review_watch_active",
    "review_a_active",
    "review_a_price",
    "review_a_pct",
    "review_a_zone",
    "review_a_as_of",
    "review_a_event",
    "review_a_path",
    "review_a_week_event",
    "review_data_available",
    "review_membership",
    "review_membership_event",
    "review_a_has_value_change",
    "review_b_has_value_change",
    "review_b_details",
    "review_b_active",
    "review_b_price",
    "review_b_pct",
    "review_b_zone",
    "review_b_status",
    "review_b_entry_valid",
    "review_b_rule",
    "review_b_date",
    "review_b_close",
    "review_b_quality",
    "review_b_entry_volume",
    "review_b_entry_reason",
    "review_b_event",
    "review_b_display_event",
    "review_b_display_basis",
    "review_selected_price_event",
    "review_a_focus_basis",
    "review_effective_entry_status",
    "review_priority",
    "review_baseline_entry_status",
    "review_change_group",
    "review_change_label",
    "review_signal_origin",
    "review_weekend_buy_point_change",
    "review_new_buy_point",
    "review_new_buy_point_zone",
    "review_has_value_change",
    "review_buy_point_change_label",
    "review_buy_point_priority",
    "bf_watch_active",
    "bf_watch_type",
    "bf_watch_trigger_price",
    "bf_watch_distance_pct",
    "bf_watch_s_resistance",
    "bf_watch_s_distance_pct",
    "bf_watch_s_side",
    "bf_watch_m_resistance",
    "bf_watch_m_distance_pct",
    "bf_watch_m_side",
    "ibd_entry_status",
    "ibd_candidate_rule",
    "ibd_candidate_price",
    "overridden_signal_source",
    "buy_point_date",
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
    "eps_yoy_growth",
    "price_52_week_high",
    "dist_to_52w_high_pct",
    "pullback_pct",
    "pullback_pct_off_peak",
    "pullback_duration_weeks",
    "pullback_peak_date",
    "pullback_peak_price",
    "pullback_v_is_dry",
    "ceiling",
    "ceiling_date",
    "breakout_date",
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


def _text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return None
    return text


def _iso_date(value: Any) -> str | None:
    text = _text_or_none(value)
    if text is None:
        return None
    try:
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError:
        return None


def _candidate_extra(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = _text_or_none(value)
    if text is None:
        return {}
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _same_price(left: Any, right: Any) -> bool:
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return False
    return math.isfinite(left_value) and math.isfinite(right_value) and math.isclose(
        left_value,
        right_value,
        rel_tol=1e-9,
        abs_tol=1e-4,
    )


def _buy_point_date(row: dict[str, Any]) -> str | None:
    rule = (_text_or_none(row.get("ibd_candidate_rule")) or "").lower()
    candidate_price = row.get("ibd_candidate_price")
    extra = _candidate_extra(row.get("ibd_candidate_extra"))

    if rule in {"ceiling", "ceiling_breakout"}:
        # A carried candidate can outlive the current scan's ceiling. Only attach
        # the current ceiling date when it still describes this exact buy point.
        if _same_price(candidate_price, row.get("ceiling")):
            return _iso_date(row.get("ceiling_date"))
        return None

    if rule == "pivot":
        selected = extra.get("selected_pivot")
        if isinstance(selected, dict) and _same_price(candidate_price, selected.get("price")):
            selected_date = _iso_date(selected.get("resistance_date"))
            if selected_date:
                return selected_date

        matched_dates = {
            parsed_date
            for item in extra.get("pivot_candidates", [])
            if isinstance(item, dict) and _same_price(candidate_price, item.get("price"))
            for parsed_date in [_iso_date(item.get("resistance_date"))]
            if parsed_date is not None
        }
        return next(iter(matched_dates)) if len(matched_dates) == 1 else None

    if rule == "ceiling_pullback":
        pending_high = extra.get("pending_high")
        if pending_high is not None and not _same_price(candidate_price, pending_high):
            return None

        # pending_high can rise while the pullback remains inside the ceiling
        # zone, so confirm_date is not the date when the buy-point price formed.
        # Prefer an explicit upstream date when available. For legacy payloads,
        # touch_date is exact only when the original touch_high still equals the
        # final pending_high/candidate price.
        pending_high_date = _iso_date(extra.get("pending_high_date"))
        if pending_high_date:
            return pending_high_date
        touch_high = extra.get("touch_high")
        if (
            touch_high is not None
            and _same_price(candidate_price, touch_high)
            and _same_price(pending_high, touch_high)
        ):
            return _iso_date(extra.get("touch_date"))
        return None

    if rule == "ma10_touch_confirm":
        pending_high = extra.get("pending_high")
        if pending_high is not None and not _same_price(candidate_price, pending_high):
            return None
        # MA10 pending_high is max(zone High), which may be raised after the
        # initial touch. Neither touch_date nor confirm_date is authoritative.
        return _iso_date(extra.get("pending_high_date"))

    if rule == "three_weeks_tight":
        twk_high = extra.get("twk_high")
        if twk_high is not None and not _same_price(candidate_price, twk_high):
            return None
        # The buy point is the max High of the tight window; use only the date
        # that identifies the bar which actually supplied that High.
        return _iso_date(extra.get("twk_high_date"))

    return None


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        record: dict[str, Any] = {}
        for field in PUBLIC_DASHBOARD_ROW_FIELDS:
            if field == "overridden_signal_source":
                extra = _candidate_extra(row.get("ibd_candidate_extra"))
                record[field] = _text_or_none(extra.get("overridden_signal_source"))
            elif field == "buy_point_date":
                record[field] = _buy_point_date(row)
            elif field in {"ceiling_date", "breakout_date", "pullback_peak_date"} and field in row:
                record[field] = _iso_date(row.get(field))
            elif field in row:
                record[field] = _json_value(row.get(field))
        records.append(record)
    return records


def _complete_view(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["review_watch_active"] = result["signal"]
    result["review_effective_entry_status"] = result["ibd_entry_status"]
    result["review_priority"] = None
    return result


def _status_meta() -> dict[str, Any]:
    result = {
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
    result["NEAR_BREAKOUT"] = {
        "label": "NEAR BREAKOUT",
        "subtitle": "Approaching Trigger",
        "tone": "cyan",
        "color": "#1fcdb4",
        "tooltip_title": "NEAR BREAKOUT",
        "tooltip": (
            "含义：BreakoutFollow 形成中 setup 已进入突破前观察区，价格接近该结构当前 trigger；仍不是正式 signal 或入场。\n"
            "数量：当前范围内符合上游 BreakoutFollow Setup Watch 条件的标的数。\n"
            "点击：只看这类标的，并保留其他已选条件。"
        ),
    }
    return result


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
    daily: dict | None = None,
    previous_pool: pd.DataFrame | None = None,
) -> dict[str, Any]:
    analysis = analyze_breakout_follow_pool(
        complete_path,
        midweek_path,
        window_date=window_date,
    )

    complete = _complete_view(analysis.complete_pool)
    midweek = pd.DataFrame()
    review_baseline_kind = "UNAVAILABLE"
    review_baseline_date = None
    if analysis.midweek_available:
        # Futu uses the original trade projection; public Review uses an
        # independent A/B observation projection with immutable weekend anchor.
        base = materialize_review_view(analysis.midweek_review)
        snapshot_date = analysis.midweek_snapshot_date
        # Git provenance belongs to the exact committed Pool input pair.
        # External --complete/--midweek CSVs remain viewable, but cannot
        # borrow the repository's publication history for Review Now.
        published_input = (
            Path(midweek_path).resolve() ==
            (PROJECT_ROOT / "us" / "breakout_follow_pool_midweek.csv").resolve()
            and Path(complete_path).resolve() ==
            (PROJECT_ROOT / "us" / "breakout_follow_pool.csv").resolve()
        )
        effective_previous = previous_pool if published_input else None
        history_verified = published_input and effective_previous is not None
        if published_input and effective_previous is None and snapshot_date is not None:
            effective_previous, history_verified = load_previous_midweek(
                PROJECT_ROOT, snapshot_date, with_history_status=True,
            )
        first_midweek_snapshot = bool(
            analysis.midweek_baseline_available and history_verified
            and (effective_previous is None or effective_previous.empty)
        )
        if first_midweek_snapshot:
            review_baseline_kind = "WEEKEND"
            review_baseline_date = analysis.complete_snapshot_date.isoformat()
        elif effective_previous is not None and not effective_previous.empty and snapshot_date is not None:
            dates = pd.to_datetime(effective_previous["snapshot_date"], errors="coerce").dropna().dt.date.unique()
            if (len(dates) == 1 and dates[0] < snapshot_date
                    and dates[0].isocalendar()[:2] == snapshot_date.isocalendar()[:2]):
                review_baseline_kind = "MIDWEEK"
                review_baseline_date = dates[0].isoformat()
        # Keep all compared prices within the MIDWEEK Pool's market session.
        # The latest 1D PKL can be newer than the published Pool.
        observation_date = snapshot_date or window_date
        midweek = build_observation_view(
            analysis.complete_pool if analysis.midweek_baseline_available else pd.DataFrame(),
            analysis.midweek_pool,
            base,
            daily=daily if daily is not None else load_daily_snapshot(PROJECT_ROOT, window_date=observation_date),
            previous=effective_previous,
            as_of=observation_date,
        )
        midweek = decorate_observation_view(
            midweek,
            complete=analysis.complete_pool if analysis.midweek_baseline_available else pd.DataFrame(),
            current=analysis.midweek_pool,
            previous=effective_previous,
            first_midweek_snapshot=first_midweek_snapshot,
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
        "schema_version": 3,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_date": window_date.isoformat(),
        "default_period": default_period,
        "meta": {
            "complete_snapshot_date": complete_snapshot,
            "midweek_snapshot_date": midweek_snapshot,
            "midweek_current_signal_count": int(analysis.midweek_pool["signal"].map(lambda x: str(x).lower() in {"true", "1", "1.0"}).sum()) if not analysis.midweek_pool.empty else 0,
            "review_week_start": (
                analysis.review_week_start.isoformat()
                if analysis.review_week_start is not None
                else None
            ),
            "midweek_available": bool(analysis.midweek_available),
            "midweek_baseline_available": bool(analysis.midweek_baseline_available),
            "review_comparison_basis": review_baseline_kind,
            "review_comparison_date": review_baseline_date,
            "warnings": list(analysis.warnings),
            "summary": dict(analysis.summary),
            "complete_freshness": freshness,
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
                "ma10_pullback",
                "ma10_touch_confirm",
                "pivot",
                "three_weeks_tight",
            ],
        },
    }


def build_watch_payload(data_root: Path) -> dict:
    """Keep a broken or stale Midweek artifact from hiding a valid Complete Watch."""
    from pool_exit_watch_store import (
        parse_watch_csv, active_watch_codes, pool_identity,
        WATCH_COMPLETE, WATCH_MIDWEEK,
    )
    from bf_snapshot import complete_snapshot_week, is_valid_complete_baseline

    payload = {
        "available": False, "complete": [], "midweek": [],
        "dates": {}, "midweek_fallback": False,
    }
    errors = {}
    allowed = {
        "code", "snapshot_date", "original_ceiling", "current_ceiling",
        "last_exit_reason", "phase", "price_state", "vs_original_pct",
        "latest_close", "pending_status", "pending_high", "events", "exit_count",
    }
    have_midweek = False
    # A legal empty Watch has no row-level date. Its matching Pool owns the
    # snapshot identity, just as it does for nonempty Watch publications.
    watch_sources = (
        ("complete", WATCH_COMPLETE, "us/breakout_follow_pool.csv"),
        ("midweek", WATCH_MIDWEEK, "us/breakout_follow_pool_midweek.csv"),
    )
    for label, relative, pool_relative in watch_sources:
        source = data_root / relative
        if not source.is_file():
            continue
        try:
            # A syntactically valid CSV is not enough: a newer unpaired Pool
            # must not leave an old Watch displayed as current on Pages.
            # In fixture-only directories (no Git checkout) validate schema.
            if label == "complete" and (data_root / ".git").exists():
                active_watch_codes(data_root, required=True)
            pool = pd.read_csv(
                data_root / pool_relative, dtype={"code": str},
                keep_default_na=False, na_values=[""], encoding="utf-8-sig",
            )
            source_date, _ = pool_identity(pool)
            if label == "complete":
                complete_snapshot_week(date.fromisoformat(source_date))
            rows = parse_watch_csv(source)
            if any(row["snapshot_date"] != source_date for row in rows):
                raise ValueError("Watch snapshot date differs from corresponding Pool")
        except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
            errors[label] = type(exc).__name__
            continue
        payload[label] = [{k: v for k, v in row.items() if k in allowed}
                          for row in rows]
        payload["dates"][label] = source_date
        if label == "complete":
            payload["available"] = True
        else:
            have_midweek = True

    complete_date = payload["dates"].get("complete")
    midweek_date = payload["dates"].get("midweek")
    if not payload["available"]:
        # A provisional view cannot be trusted when its complete baseline is
        # unavailable. Never expose it as an independently authoritative view.
        payload["midweek"] = []
        payload["dates"].pop("midweek", None)
    elif (not have_midweek or
          (not payload["midweek"] and bool(payload["complete"])) or
          not is_valid_complete_baseline(
              date.fromisoformat(complete_date),
              date.fromisoformat(midweek_date))):
        payload["midweek_fallback"] = True
    if errors:
        payload["errors"] = errors
    return payload


def build_site(
    output_dir: str | Path,
    *,
    complete_path: str | Path,
    midweek_path: str | Path,
    window_date: date,
    ga4_measurement_id: str | None = None,
) -> Path:
    tagged_index = _with_ga4_tag(
        (DASHBOARD_DIR / "index.html").read_text(encoding="utf-8"),
        ga4_measurement_id,
    )
    output = Path(output_dir).resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    for asset in STATIC_ASSETS:
        source = DASHBOARD_DIR / asset
        if not source.is_file():
            raise FileNotFoundError(f"Static dashboard asset missing: {source}")
        shutil.copy2(source, output / asset)

    if ga4_measurement_id:
        (output / "index.html").write_text(tagged_index, encoding="utf-8")

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
    # Independent Watch payload; never disturb the original dashboard.json.
    watch_payload = build_watch_payload(PROJECT_ROOT)
    (data_dir / "pool_exit_watch.json").write_text(
        json.dumps(watch_payload, ensure_ascii=False, default=str, separators=(",", ":")),
        encoding="utf-8")

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
        ga4_measurement_id=os.environ.get("GA4_MEASUREMENT_ID"),
    )
    print(f"Static dashboard built: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
