"""Deterministic B0 baseline replay and production invariant auditing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from backtest.replay_eps import replay_signal_eps_lookup

from dashboard.skill_industry_eps_known import (
    SkillCandidate,
    rank_skill_industry_eps_known,
    select_skill_industry_eps_known,
)

logger = logging.getLogger(__name__)


def replay_b0_on_pool(
    pool_df: pd.DataFrame,
    snapshot_date: str,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Run deterministic B0 selection on a single pool DataFrame."""
    # Ensure snapshot_date is on rows
    df = pool_df.copy()
    df["snapshot_date"] = snapshot_date

    with replay_signal_eps_lookup(allow_network=False):
        selected: list[SkillCandidate] = select_skill_industry_eps_known(df, limit=limit)
    rows: list[dict[str, Any]] = []

    for pick_idx, item in enumerate(selected, 1):
        fvals = item.feature_values
        row_dict = {
            "snapshot_date": snapshot_date,
            "code": item.code,
            "pick_order": pick_idx,
            "entry_status": item.entry_status,
            "lane": item.lane,
            "industry": item.industry,
            "reason_codes": ";".join(item.reason_codes),
            "risk_codes": ";".join(item.risk_codes),
            "sort_key": str(item.sort_key),
            "raw_rank": item.raw_rank,
            "ibd_candidate_rule": str(fvals.get("ibd_candidate_rule", "") or ""),
            "ibd_candidate_price": fvals.get("ibd_candidate_price"),
            "latest_close": fvals.get("latest_close"),
            "current_vs_ibd_candidate_pct": fvals.get("current_vs_ibd_candidate_pct"),
            "ibd_entry_volume_ratio": fvals.get("ibd_entry_volume_ratio"),
            "ibd_entry_close_position": fvals.get("ibd_entry_close_position"),
            "ibd_entry_breakout_range_ratio": fvals.get("ibd_entry_breakout_range_ratio"),
            "dist_to_52w_high_pct": fvals.get("dist_to_52w_high_pct"),
            "volume_ratio": fvals.get("volume_ratio"),
            "eps_yoy_growth": fvals.get("eps_yoy_growth"),
            "effective_eps_yoy_growth": fvals.get("effective_eps_yoy_growth"),
            "price_data_status": "PENDING",
        }
        rows.append(row_dict)

    return rows


import hashlib
import json


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def run_b0_across_all_pools(
    pool_paths: Sequence[Path],
    output_events_csv: Path | str | None = "backtest/b0_top3_quality_audit/output/b0_selection_events.csv",
    output_invariant_csv: Path | str | None = "backtest/b0_top3_quality_audit/output/b0_production_invariant_audit.csv",
    golden_csv_path: Path | str = "backtest/b0_top3_quality_audit/golden/b0_top3_golden_reference.csv",
    golden_weekly_csv_path: Path | str = "backtest/b0_top3_quality_audit/golden/b0_top3_golden_weekly_reference.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run B0 replay on all historical pools and compare against frozen golden reference.
    
    Returns:
        (b0_events_df, invariant_audit_df)
    """
    all_b0_records: list[dict[str, Any]] = []
    invariant_records: list[dict[str, Any]] = []

    # Load frozen candidate-level golden reference if available
    golden_map: dict[str, list[dict[str, Any]]] = {}
    g_path = Path(golden_csv_path)
    if g_path.exists():
        golden_df = pd.read_csv(g_path, encoding="utf-8-sig")
        for snap_date, grp in golden_df.groupby("snapshot_date"):
            golden_map[str(snap_date)] = grp.sort_values("pick_order").to_dict(orient="records")

    # Load frozen week-level golden reference if available
    golden_weekly_map: dict[str, dict[str, Any]] = {}
    gw_path = Path(golden_weekly_csv_path)
    if gw_path.exists():
        golden_weekly_df = pd.read_csv(gw_path, encoding="utf-8-sig")
        for _, w_row in golden_weekly_df.iterrows():
            golden_weekly_map[str(w_row["snapshot_date"])] = w_row.to_dict()

    for path in pool_paths:
        snapshot_date = path.parent.name
        try:
            pool_sha256 = compute_file_sha256(path)
            pool_df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception as e:
            logger.warning(f"Failed to read pool at {path}: {e}")
            continue

        pool_df["snapshot_date"] = snapshot_date
        b0_records = replay_b0_on_pool(pool_df, snapshot_date=snapshot_date, limit=3)
        all_b0_records.extend(b0_records)

        replay_codes = [r["code"] for r in b0_records]
        replay_sort_keys = [str(r["sort_key"]) for r in b0_records]
        replay_reasons = [str(r["reason_codes"]) for r in b0_records]

        sig_payload = {
            "snapshot_date": snapshot_date,
            "codes": replay_codes,
            "sort_keys": replay_sort_keys,
            "reasons": replay_reasons,
        }
        replay_sig_sha = hashlib.sha256(json.dumps(sig_payload, sort_keys=True).encode("utf-8")).hexdigest()

        discrepancies: list[str] = []

        # 1. Week-Level Golden Audit (Verifies 0-pick weeks explicitly)
        gw_info = golden_weekly_map.get(str(snapshot_date))
        if gw_info is not None:
            exp_count = int(gw_info.get("expected_pick_count", 0))
            exp_codes_str = str(gw_info.get("expected_codes", "") or "")
            if pd.isna(gw_info.get("expected_codes")):
                exp_codes_str = ""
            exp_codes = [c.strip() for c in exp_codes_str.split(",") if c.strip()]
            exp_sig_sha = str(gw_info.get("selection_signature_sha256", ""))

            if len(b0_records) != exp_count:
                discrepancies.append(f"weekly_count_mismatch(replay={len(b0_records)},golden={exp_count})")
            if replay_codes != exp_codes:
                discrepancies.append(f"weekly_codes_mismatch(replay={replay_codes},golden={exp_codes})")
            if exp_sig_sha and replay_sig_sha != exp_sig_sha:
                discrepancies.append(f"weekly_signature_sha_mismatch")

        # 2. Candidate-Level Golden Audit
        golden_records = golden_map.get(str(snapshot_date), [])
        golden_codes = [r["code"] for r in golden_records]

        if len(b0_records) == 0 and len(golden_records) == 0:
            # Both have 0 recommendations -> verified exact 0 match
            pass
        elif len(b0_records) != len(golden_records):
            discrepancies.append(f"count_mismatch(replay={len(b0_records)},golden={len(golden_records)})")
        elif replay_codes != golden_codes:
            discrepancies.append(f"code_mismatch(replay={replay_codes},golden={golden_codes})")
        else:
            for idx, (rep, gol) in enumerate(zip(b0_records, golden_records), 1):
                if rep["code"] != gol["code"]:
                    discrepancies.append(f"pick_{idx}_code({rep['code']}!={gol['code']})")
                if str(rep["sort_key"]) != str(gol.get("sort_key")):
                    discrepancies.append(f"pick_{idx}_sort_key_drift")

        discrepancy_count = len(discrepancies)
        is_exact_match = bool(discrepancy_count == 0 and (len(b0_records) == len(golden_records) or gw_info is not None))

        invariant_records.append({
            "snapshot_date": snapshot_date,
            "pool_path": str(path),
            "pool_sha256": pool_sha256,
            "replay_top3_count": len(b0_records),
            "replay_codes": ",".join(replay_codes),
            "golden_codes": ",".join(golden_codes),
            "replay_sort_keys": " | ".join(replay_sort_keys),
            "replay_reasons": " | ".join(replay_reasons),
            "selection_signature_sha256": replay_sig_sha,
            "is_exact_match": is_exact_match,
            "discrepancy_count": discrepancy_count,
            "discrepancy_details": ";".join(discrepancies) if discrepancies else "NONE",
        })

    b0_events_df = pd.DataFrame(all_b0_records)
    invariant_df = pd.DataFrame(invariant_records)

    if output_events_csv is not None:
        out_p = Path(output_events_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        b0_events_df.to_csv(out_p, index=False, encoding="utf-8-sig")
        logger.info(f"Saved {len(b0_events_df)} B0 selection events to {out_p}")

    if output_invariant_csv is not None:
        inv_p = Path(output_invariant_csv)
        inv_p.parent.mkdir(parents=True, exist_ok=True)
        invariant_df.to_csv(inv_p, index=False, encoding="utf-8-sig")
        logger.info(f"Saved B0 production invariant audit to {inv_p}")

    return b0_events_df, invariant_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    from backtest.b0_top3_quality_audit.universe import scan_replay_pools
    pool_paths = scan_replay_pools()
    run_b0_across_all_pools(pool_paths)
