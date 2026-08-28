"""EPS-only historical replay recalibration.

This command intentionally does not rebuild technical replay pools or price facts.
It re-resolves signal EPS with explicit REPLAY semantics and writes only EPS
publication columns back into the existing historical pool CSVs.

Recommended:
    PYTHONPATH=. python backtest/ibd_skill_replay_pools/recalibrate_eps_pit.py --reset-store
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from eps_pit import EPSResolveMode, EPSStatus, enrich_pool_with_signal_eps
from eps_pit.lookup import SignalEPSLookup
from eps_pit.store import EPSPITStore

LOG = logging.getLogger("eps_pit_recalibration")
POOL_ROOT = Path("backtest/ibd_skill_replay_pools")


def _is_signal(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "1.0"}


def _is_eps_column(name: str) -> bool:
    return name.startswith("eps_") or name == "effective_eps_yoy_growth"


def _assert_non_eps_unchanged(before: pd.DataFrame, after: pd.DataFrame, path: Path) -> None:
    before_cols = [c for c in before.columns if not _is_eps_column(str(c))]
    after_cols = [c for c in after.columns if not _is_eps_column(str(c))]
    if before_cols != after_cols:
        raise RuntimeError(
            f"{path}: non-EPS column schema changed: before={before_cols}, after={after_cols}"
        )
    pd.testing.assert_frame_equal(
        before.loc[:, before_cols].reset_index(drop=True),
        after.loc[:, after_cols].reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def _value(row: pd.Series, key: str) -> Any:
    value = row.get(key)
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def recalibrate(
    *,
    pool_root: Path = POOL_ROOT,
    csv_path: str | None = None,
    reset_store: bool = False,
) -> dict[str, Any]:
    pool_paths = sorted(pool_root.glob("*/breakout_follow_pool.csv"))
    if not pool_paths:
        raise RuntimeError(f"No historical pools found under {pool_root}")

    resolved_store = Path(csv_path or SignalEPSLookup.DEFAULT_REPLAY_CSV_PATH)
    if reset_store and resolved_store.exists():
        LOG.warning("Resetting replay PIT store: %s", resolved_store)
        resolved_store.unlink()

    impact_rows: list[dict[str, Any]] = []
    provider_errors: list[str] = []
    future_violations: list[str] = []
    staged_pools: list[tuple[Path, pd.DataFrame, int]] = []
    store = EPSPITStore(str(resolved_store))

    for pool_path in pool_paths:
        snapshot = pool_path.parent.name
        before = pd.read_csv(pool_path, dtype={"code": str}, encoding="utf-8-sig")
        before["code"] = before["code"].astype(str).str.strip()

        after = enrich_pool_with_signal_eps(
            before.copy(),
            snapshot_date=snapshot,
            csv_path=str(resolved_store),
            refresh_missing=True,
            mode=EPSResolveMode.REPLAY,
        )
        _assert_non_eps_unchanged(before, after, pool_path)

        signal_mask = after.get("signal", pd.Series(False, index=after.index)).map(_is_signal)
        signal_rows = after.loc[signal_mask]
        status_series = signal_rows.get(
            "eps_yoy_growth_status",
            pd.Series("", index=signal_rows.index),
        ).astype(str)
        bad_provider = signal_rows[status_series.eq(EPSStatus.PROVIDER_ERROR.value)]
        if not bad_provider.empty:
            provider_errors.extend(
                f"{snapshot}:{code}" for code in bad_provider["code"].astype(str)
            )

        for idx in signal_rows.index:
            code = str(after.at[idx, "code"]).strip()
            result = store.get(snapshot, code)
            if result is not None and result.is_resolved and result.effective_date:
                if str(result.effective_date)[:10] > snapshot:
                    future_violations.append(
                        f"{snapshot}:{code}:{result.effective_date}"
                    )

            old_row = before.loc[idx]
            new_row = after.loc[idx]
            old_eps = pd.to_numeric(
                pd.Series([old_row.get("eps_yoy_growth")]), errors="coerce"
            ).iloc[0]
            new_eps = pd.to_numeric(
                pd.Series([new_row.get("eps_yoy_growth")]), errors="coerce"
            ).iloc[0]
            old_known = bool(pd.notna(old_eps))
            new_known = bool(pd.notna(new_eps))
            impact_rows.append(
                {
                    "snapshot_date": snapshot,
                    "code": code,
                    "old_eps": old_eps if old_known else pd.NA,
                    "new_eps": new_eps if new_known else pd.NA,
                    "old_source": _value(old_row, "eps_yoy_growth_source"),
                    "new_source": _value(new_row, "eps_yoy_growth_source"),
                    "new_status": _value(new_row, "eps_yoy_growth_status"),
                    "new_missing_reason": _value(
                        new_row, "eps_yoy_growth_missing_reason"
                    ),
                    "new_effective_date": (
                        result.effective_date if result is not None else None
                    ),
                    "old_known": old_known,
                    "new_known": new_known,
                    "old_eps25": bool(old_known and float(old_eps) >= 25.0),
                    "new_eps25": bool(new_known and float(new_eps) >= 25.0),
                }
            )

        staged_pools.append((pool_path, after, int(signal_mask.sum())))

    if provider_errors:
        raise RuntimeError(
            "Replay EPS provider_error rows block publication: "
            + ", ".join(provider_errors[:50])
        )
    if future_violations:
        raise RuntimeError(
            "Future EPS effective_date violations: "
            + ", ".join(future_violations[:50])
        )

    # Only publish corrected pool files after every week has passed provider,
    # PIT-date and non-EPS invariants. Durable resolved cache rows may already
    # exist, but the research pool set remains atomic from the caller's view.
    for pool_path, after, signal_count in staged_pools:
        after.to_csv(pool_path, index=False, encoding="utf-8-sig")
        LOG.info("Recalibrated %s (%s signal rows)", pool_path.parent.name, signal_count)

    impact_df = pd.DataFrame(impact_rows)
    impact_path = pool_root / "EPS_PIT_RECALIBRATION_IMPACT.csv"
    impact_df.to_csv(impact_path, index=False, encoding="utf-8-sig")

    changed_value = (
        impact_df["old_eps"].fillna(float("inf"))
        != impact_df["new_eps"].fillna(float("inf"))
    )
    source_changed = (
        impact_df["old_source"].fillna("").astype(str)
        != impact_df["new_source"].fillna("").astype(str)
    )
    summary = {
        "data_revision": "EPS_RECALIBRATED_V2",
        "mode": EPSResolveMode.REPLAY.value,
        "replay_store_path": str(resolved_store),
        "pool_count": len(pool_paths),
        "total_signal_rows": int(len(impact_df)),
        "old_resolved_rows": int(impact_df["old_known"].sum()),
        "new_resolved_rows": int(impact_df["new_known"].sum()),
        "eps_value_changed_count": int(changed_value.sum()),
        "unknown_to_known_count": int(
            ((~impact_df["old_known"]) & impact_df["new_known"]).sum()
        ),
        "known_to_unknown_count": int(
            (impact_df["old_known"] & (~impact_df["new_known"])).sum()
        ),
        "source_changed_count": int(source_changed.sum()),
        "eps25_state_changed_count": int(
            (impact_df["old_eps25"] != impact_df["new_eps25"]).sum()
        ),
        "new_provider_error_count": 0,
        "future_leakage_violation_count": 0,
    }
    (pool_root / "EPS_PIT_RECALIBRATION_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    report = [
        "# Historical EPS PIT Recalibration",
        "",
        "- Mode: REPLAY",
        f"- Replay store: {resolved_store}",
        f"- Historical pools: {summary['pool_count']}",
        f"- Signal rows: {summary['total_signal_rows']}",
        f"- Old resolved: {summary['old_resolved_rows']}",
        f"- New resolved: {summary['new_resolved_rows']}",
        f"- EPS value changed: {summary['eps_value_changed_count']}",
        f"- Unknown -> known: {summary['unknown_to_known_count']}",
        f"- Known -> unknown: {summary['known_to_unknown_count']}",
        f"- EPS>=25 state changed: {summary['eps25_state_changed_count']}",
        "- Future leakage violations: 0",
        "- Non-EPS fields: unchanged by construction and asserted per pool",
        "",
        "This is a data correction only. It does not rebuild price facts, technical "
        "replay features, selector rules, or research protocols.",
    ]
    (pool_root / "EPS_PIT_RECALIBRATION_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-root", default=str(POOL_ROOT))
    parser.add_argument("--csv-path", default=None)
    parser.add_argument(
        "--reset-store",
        action="store_true",
        help="Delete the REPLAY PIT store first so all observations are rebuilt.",
    )
    args = parser.parse_args()
    summary = recalibrate(
        pool_root=Path(args.pool_root),
        csv_path=args.csv_path,
        reset_store=args.reset_store,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
